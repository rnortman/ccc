//! Running a resolved agent: pick the account, spawn the harness, record the
//! session, render the result.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Context, Result};
use jiff::Timestamp;
use uuid::Uuid;

use crate::agents::AgentDef;
use crate::audit;
use crate::audit::{
    ChildPayload, Command as AuditCommand, EndPayload, PromptSource, Recorder, SpawnPayload,
    StartPayload,
};
use crate::budget::Accounts;
use crate::bus::{Bus, Live};
use crate::config::{Config, Effort, EnvSource, Harness, ProcessEnv};
use crate::harness::claude::{Account, ClaudeDriver};
use crate::harness::codex::{self, CodexDriver};
use crate::harness::goose::GooseDriver;
use crate::harness::{
    diag, report_warnings, Common, Driver, HarnessError, Invocation, Io, Mode, Origin, Outcome,
    RunContext,
};
use crate::output::RunOutput;
use crate::resolve::{self, Resolved};
use crate::session::{self, SessionRecord};
use crate::supervise::{self, PromptDelivery, PromptEnd, SuperviseOpts, Termination};
use crate::usage::{self, UsageSource};

/// Everything a turn reads from outside itself: the parsed config, the
/// environment, the clock, and the usage data account selection needs.
///
/// The usage read is a closure because it costs a broker round trip: a Codex
/// run, a run naming its own `--profile`, or a run under the ambient login
/// never makes one. The account read — key budgets and profile flags — is a
/// closure for the same reason: only a selection makes it, once each, the
/// limit continuation included. `bus` and `live` publish
/// this invocation into the agent tree; both are inert with the bus off.
pub struct Ctx<'a> {
    pub cfg: &'a Config,
    pub env: &'a dyn EnvSource,
    pub now: Timestamp,
    pub usage: &'a (dyn Fn() -> UsageSource + Sync),
    pub accounts: &'a (dyn Fn() -> Accounts + Sync),
    /// False at the terminal: selection never considers an API key there.
    api_keys_eligible: bool,
    pub bus: &'a dyn Bus,
    live: &'a Live,
    pub audit: &'a Recorder,
    session_locks: Arc<Mutex<Vec<(String, session::SessionLock)>>>,
}

impl<'a> Ctx<'a> {
    /// This context for a turn at the terminal, where no API key is selected.
    fn at_terminal(&self) -> Ctx<'a> {
        Ctx {
            cfg: self.cfg,
            env: self.env,
            now: self.now,
            usage: self.usage,
            accounts: self.accounts,
            api_keys_eligible: false,
            bus: self.bus,
            live: self.live,
            audit: self.audit,
            session_locks: Arc::clone(&self.session_locks),
        }
    }

    fn keys(&self) -> crate::select::Keys {
        crate::select::Keys {
            eligible: self.api_keys_eligible,
        }
    }

    fn owns_session(&self, id: &str) -> bool {
        self.session_locks
            .lock()
            .expect("session lock mutex poisoned")
            .iter()
            .any(|(held, _)| held == id)
    }

    fn retain_session_lock(&self, id: &str, lock: session::SessionLock) {
        let mut locks = self
            .session_locks
            .lock()
            .expect("session lock mutex poisoned");
        if !locks.iter().any(|(held, _)| held == id) {
            locks.push((id.to_string(), lock));
        }
    }
}

#[derive(Debug, Clone)]
struct Selected {
    account: Account,
    notes: Vec<String>,
}

impl std::ops::Deref for Selected {
    type Target = Account;
    fn deref(&self) -> &Self::Target {
        &self.account
    }
}

/// How a Claude turn's account is chosen.
enum AccountPick<'a> {
    /// The profile `--profile` named, or the usage-driven choice when `None`.
    Requested(Option<&'a str>),
    /// Already selected by the caller: the continuation of a usage-limited turn.
    Chosen(Selected),
}

#[derive(Debug, Clone)]
struct Prepared {
    r: Resolved,
    cwd: PathBuf,
    account: Option<Account>,
    notes: Vec<String>,
}

/// What `ccc launch` is asked to open: a fresh session of `agent`, a fresh
/// Claude session with no agent type, or a recorded session continued (or
/// forked) interactively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaunchTarget<'a> {
    Fresh {
        agent: &'a str,
        cwd: Option<&'a str>,
        cli_harness: Option<Harness>,
        /// Goose only: `--allow-read`.
        allow_read: &'a [PathBuf],
        /// Goose only: `--allow-write`.
        allow_write: &'a [PathBuf],
        /// Goose only: `--allow-exec`.
        allow_exec: &'a [PathBuf],
    },
    Roleless {
        cwd: Option<&'a str>,
        model: Option<&'a str>,
        effort: Option<Effort>,
    },
    Resume {
        id: &'a str,
        fork: bool,
    },
}

/// A launch with everything in hand except a running child: the built
/// invocation, what it was built from, and the record to save under whatever
/// session id it ends up with.
pub struct LaunchPrepared {
    inv: Invocation,
    r: Resolved,
    account: Option<Account>,
    notes: Vec<String>,
    common: Common,
    record: SessionRecord,
}

#[derive(Debug, Clone)]
struct Completion {
    code: i32,
    end: EndPayload,
}

struct DriveResult {
    result: Result<Outcome, HarnessError>,
    signal: Option<i32>,
}

#[derive(Debug)]
struct OperationError {
    error: anyhow::Error,
    end: Option<Box<EndPayload>>,
}

impl From<anyhow::Error> for OperationError {
    fn from(error: anyhow::Error) -> Self {
        Self { error, end: None }
    }
}

impl std::fmt::Display for OperationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}

impl std::error::Error for OperationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error.root_cause())
    }
}

impl PartialEq<i32> for Completion {
    fn eq(&self, other: &i32) -> bool {
        self.code == *other
    }
}

impl Completion {
    fn new(code: i32, end: EndPayload) -> Self {
        Self { code, end }
    }
}

/// The Goose confinement flags of `ccc run`: directories granted beyond the
/// run's cwd, and `--goose-nested`, which takes write and exec roots from the
/// flags only.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GooseFlags {
    pub allow_read: Vec<PathBuf>,
    pub allow_write: Vec<PathBuf>,
    pub allow_exec: Vec<PathBuf>,
    pub nested: bool,
}

impl GooseFlags {
    /// The first `--allow-*` flag given, spelled as on the command line.
    fn first_allow(&self) -> Option<String> {
        [
            ("--allow-read", &self.allow_read),
            ("--allow-write", &self.allow_write),
            ("--allow-exec", &self.allow_exec),
        ]
        .into_iter()
        .find_map(|(flag, dirs)| dirs.first().map(|d| format!("{flag} {}", d.display())))
    }
}

/// A one-shot turn for `agent`.
// TODO(ccc-detach): a parent whose shell tool times out before a long
// subagent finishes needs a `--detach` / `ccc wait` pair writing the
// output to a file; background execution covers it for now.
#[allow(clippy::too_many_arguments)]
pub fn run(
    agent: &str,
    cwd: Option<&str>,
    positional: Option<&str>,
    prompt_file: Option<&str>,
    profile: Option<&str>,
    cli_harness: Option<Harness>,
    goose: &GooseFlags,
    as_json: bool,
) -> Result<i32> {
    let mut input = std::io::stdin();
    let mut output = std::io::stdout();
    run_from_report(
        agent,
        cwd,
        positional,
        prompt_file,
        profile,
        cli_harness,
        goose,
        as_json,
        &ProcessEnv,
        Config::load,
        &mut input,
        &mut output,
        &|message| diag(message),
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn run_from<L>(
    agent: &str,
    cwd: Option<&str>,
    positional: Option<&str>,
    prompt_file: Option<&str>,
    profile: Option<&str>,
    cli_harness: Option<Harness>,
    as_json: bool,
    env: &dyn EnvSource,
    load_config: L,
    input: &mut dyn Read,
    output: &mut dyn Write,
) -> Result<i32>
where
    L: FnOnce() -> Result<Config>,
{
    run_from_report(
        agent,
        cwd,
        positional,
        prompt_file,
        profile,
        cli_harness,
        &GooseFlags::default(),
        as_json,
        env,
        load_config,
        input,
        output,
        &|message| diag(message),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_from_report<L>(
    agent: &str,
    cwd: Option<&str>,
    positional: Option<&str>,
    prompt_file: Option<&str>,
    profile: Option<&str>,
    cli_harness: Option<Harness>,
    goose: &GooseFlags,
    as_json: bool,
    env: &dyn EnvSource,
    load_config: L,
    input: &mut dyn Read,
    output: &mut dyn Write,
    report: &(dyn Fn(String) + Send + Sync),
) -> Result<i32>
where
    L: FnOnce() -> Result<Config>,
{
    let recorder = Recorder::new(env)?;
    let prompt_result = prompt_text_from(positional, prompt_file, input);
    let start = StartPayload {
        pid: std::process::id(),
        cmd: AuditCommand::Run,
        agent: Some(agent.to_string()),
        session: None,
        cwd_arg: cwd.map(str::to_string),
        profile_arg: profile.map(str::to_string),
        harness_arg: cli_harness,
        json: as_json,
        prompt_source: prompt_source(positional, prompt_file),
        prompt_file: prompt_file.map(str::to_string),
        prompt: prompt_result.as_ref().ok().cloned(),
        passthrough: vec![],
        fork: false,
    };
    let completion =
        audited_with_recorder_in(
            recorder,
            start,
            env,
            load_config,
            |ctx| match prompt_result {
                Ok(prompt) => run_with_report(
                    ctx,
                    agent,
                    cwd,
                    prompt,
                    profile,
                    cli_harness,
                    goose,
                    as_json,
                    output,
                    report,
                ),
                Err(e) => Err(e.into()),
            },
        )?;
    Ok(completion.code)
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn run_with(
    ctx: &Ctx,
    agent: &str,
    cwd: Option<&str>,
    prompt: String,
    profile: Option<&str>,
    cli_harness: Option<Harness>,
    as_json: bool,
    out: &mut dyn Write,
) -> std::result::Result<Completion, OperationError> {
    run_with_report(
        ctx,
        agent,
        cwd,
        prompt,
        profile,
        cli_harness,
        &GooseFlags::default(),
        as_json,
        out,
        &|message| diag(message),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_with_report(
    ctx: &Ctx,
    agent: &str,
    cwd: Option<&str>,
    prompt: String,
    profile: Option<&str>,
    cli_harness: Option<Harness>,
    goose: &GooseFlags,
    as_json: bool,
    out: &mut dyn Write,
    report: &(dyn Fn(String) + Send + Sync),
) -> std::result::Result<Completion, OperationError> {
    let prepared = prepare(ctx, agent, cwd, profile, cli_harness, goose)?;
    let Prepared {
        r,
        cwd,
        account,
        notes,
    } = prepared;

    let common = Common {
        cwd,
        extra_args: extra_args(ctx.cfg, r.harness),
        mode: Mode {
            io: Io::OneShot { prompt },
            origin: Origin::Fresh,
        },
        preset_session_id: presets_session(r.harness).then(|| Uuid::new_v4().to_string()),
    };
    let mut record = record_of(&r, &common.cwd, account.as_ref(), ctx.now);
    fill_lineage(ctx, &mut record);
    let driven = drive_with_meta_report(ctx, &r, &common, account.clone(), notes, report, &record)?;
    let DriveResult { result, signal } = driven;

    match result {
        Ok(mut outcome) => {
            if let Some(warning) = session_mismatch_warning(
                common.preset_session_id.as_deref(),
                Some(outcome.session_id.as_str()),
            ) {
                outcome.warnings.push(warning);
            }
            save_session_discovered(ctx, &outcome.session_id, &record);
            let session = outcome.session_id.clone();
            let result_text = outcome.result.clone();
            let native = outcome.native.clone();
            let warnings = outcome.warnings.clone();
            let end = EndPayload {
                exit: Some(0),
                signal,
                session: Some(session),
                result: Some(result_text),
                error: None,
                warnings,
                native: Some(native),
                turns: None,
            };
            let code = match finish(&r, &common.cwd, account.as_ref(), outcome, as_json, out) {
                Ok(code) => code,
                Err(e) => {
                    let mut failed = end.clone();
                    failed.error = Some(format!("{e:#}"));
                    return Err(OperationError {
                        error: e,
                        end: Some(Box::new(failed)),
                    });
                }
            };
            Ok(Completion::new(code, end))
        }
        Err(err) => {
            // A turn that got far enough to be given a session id is one the
            // caller can continue, so the id `fail` prints resolves.
            if let Some(id) = reported_session_id(&err) {
                save_session_discovered(ctx, &id, &record);
            }
            let session = reported_session_id(&err);
            let mut err = err;
            if let Some(warning) = session_mismatch_warning(
                common.preset_session_id.as_deref(),
                reported_session_id(&err).as_deref(),
            ) {
                if let Some(warnings) = err.warnings_mut() {
                    warnings.push(warning);
                }
            }
            let no_retry = profile.map(|_| PINNED);
            let not_continued =
                match retry_on_limit(ctx, &r, no_retry, account.as_ref(), &err, as_json, out) {
                    Some(LimitOutcome::Continued(done)) => return done,
                    Some(LimitOutcome::NotContinued(why)) => Some(why),
                    None => None,
                };
            let result_text = err.reported().map(|(message, _)| message.to_string());
            let mut warnings = err.warnings().to_vec();
            if let Some(why) = &not_continued {
                warnings.extend(why.lines().iter().cloned());
            }
            let native = err.reported().map(|(_, native)| native.clone());
            let (result_text, error_text) = match &err {
                HarnessError::Unusable { message, .. } => {
                    (Some(message.clone()), Some(message.clone()))
                }
                _ => (result_text, None),
            };
            let code = fail(
                &r,
                &common.cwd,
                account.as_ref(),
                err,
                not_continued,
                as_json,
                out,
            );
            let end = EndPayload {
                exit: Some(code),
                signal,
                session,
                result: result_text,
                error: error_text,
                warnings,
                native,
                turns: None,
            };
            Ok(Completion::new(code, end))
        }
    }
}

/// The prompt a usage-limited turn is continued with on another profile.
const LIMIT_RETRY_PROMPT: &str =
    "Your previous turn was cut off by a usage limit on another account. Continue from where you stopped.";

/// What became of a usage-limited turn.
enum LimitOutcome {
    /// A continuation ran on another profile; its outcome is the
    /// invocation's.
    Continued(std::result::Result<Completion, OperationError>),
    /// No continuation ran.
    NotContinued(NotContinued),
}

/// Why a usage-limited turn was not continued on another profile, in the
/// lines that say so on stderr and in the end record's warnings.
#[derive(Debug, Clone)]
enum NotContinued {
    /// No continuation was attempted; the line is not yet printed.
    Declined(String),
    /// The continuation was announced on stderr, after the first turn's
    /// warnings, and could not start; these lines are already printed.
    Failed(Vec<String>),
}

impl NotContinued {
    fn lines(&self) -> &[String] {
        match self {
            NotContinued::Declined(line) => std::slice::from_ref(line),
            NotContinued::Failed(lines) => lines,
        }
    }
}

const PINNED: &str = "--profile chose the account";
const ALREADY_CONTINUED: &str = "the turn was already the continuation of a limited one";
const NO_SESSION: &str = "the turn reported no session id to continue";
const NO_OTHER_PROFILE: &str = "no other usable profile";
/// What a Claude turn with no account runs as.
const AMBIENT: &str = "the ambient login";

/// Continues a usage-limited Claude turn once on another profile. `None`
/// when the turn was not limited. Otherwise the continuation's outcome —
/// success, or a failure after it ran, either way carrying the first turn's
/// warnings, the hand-off note and the first turn's native report in its end
/// record's warnings — or why it did not run: the caller's `no_retry`
/// reason, no session id, no other profile winning selection outright (none
/// credentialed, or only a fallback to configured order), or a continuation
/// that could not start.
fn retry_on_limit(
    ctx: &Ctx,
    r: &Resolved,
    no_retry: Option<&'static str>,
    account: Option<&Account>,
    err: &HarnessError,
    as_json: bool,
    out: &mut dyn Write,
) -> Option<LimitOutcome> {
    let HarnessError::Limited {
        session_id,
        message,
        native,
        ..
    } = err
    else {
        return None;
    };
    let limited = account.map_or(AMBIENT, |a| a.profile.as_str());
    let declined = |reason: &str| {
        Some(LimitOutcome::NotContinued(NotContinued::Declined(format!(
            "{limited} hit a usage limit: {message}; {reason}"
        ))))
    };
    if let Some(reason) = no_retry {
        return declined(reason);
    }
    let Some(session) = session_id else {
        return declined(NO_SESSION);
    };
    if account.is_none() {
        return declined(NO_OTHER_PROFILE);
    }
    // An error here is no candidate left once the limited profile is set
    // aside; the first turn already ran, so it is not a configuration error.
    let Ok(choice) = crate::select::choose(
        &ctx.cfg.claude,
        &ctx.cfg.tuning,
        ctx.env,
        &r.model,
        ctx.now,
        &(ctx.usage)(),
        &[limited.to_string()],
        ctx.keys(),
        ctx.accounts,
    ) else {
        return declined(NO_OTHER_PROFILE);
    };
    if choice.degraded.is_some() {
        return declined(NO_OTHER_PROFILE);
    }
    let next = choice.account.profile.clone();
    report_warnings(err.warnings());
    let note =
        format!("{limited} hit a usage limit: {message}; continuing session {session} on {next}");
    diag(&note);
    let handoff = |later: &mut Vec<String>| {
        let mut warnings = err.warnings().to_vec();
        warnings.push(note.clone());
        warnings.push(format!("the limited turn on {limited} reported {native}"));
        warnings.append(later);
        *later = warnings;
    };
    match resume_turn(
        ctx,
        session,
        false,
        AccountPick::Chosen(selected(choice)),
        LIMIT_RETRY_PROMPT.to_string(),
        as_json,
        out,
    ) {
        Ok(mut completion) => {
            handoff(&mut completion.end.warnings);
            Some(LimitOutcome::Continued(Ok(completion)))
        }
        // An end record means the continuation ran and failed afterwards
        // (writing its result); that failure is the invocation's.
        Err(OperationError {
            error,
            end: Some(mut end),
        }) => {
            handoff(&mut end.warnings);
            Some(LimitOutcome::Continued(Err(OperationError {
                error,
                end: Some(end),
            })))
        }
        Err(e) => {
            let failure = format!(
                "could not continue session {session} on {next}: {}",
                e.error
            );
            diag(&failure);
            Some(LimitOutcome::NotContinued(NotContinued::Failed(vec![
                note, failure,
            ])))
        }
    }
}

fn session_mismatch_warning(expected: Option<&str>, reported: Option<&str>) -> Option<String> {
    match (expected, reported) {
        (Some(expected), Some(actual)) if expected != actual => Some(format!(
            "the harness reported session id {actual}, expected {expected}"
        )),
        _ => None,
    }
}

/// A fork the harness answered under its parent's id. The fork record
/// must not be saved there — it would overwrite the parent — so the
/// save is skipped and this is the reason.
fn fork_collision_warning(mode: &Mode, reported: Option<&str>) -> Option<String> {
    match (mode.is_fork(), mode.starts_from(), reported) {
        (true, Some(parent), Some(actual)) if parent == actual => Some(format!(
            "the harness answered the fork under its parent's id {parent}; the fork was not recorded and the parent's record was left untouched"
        )),
        _ => None,
    }
}

/// Maps a resume argument to the harness session ID it names. An argument
/// with a directory entry at its session-record path is that session; with
/// no entry there, it is a trace selector naming its root's session. Nothing
/// is locked, written or launched.
fn resolve_resume_arg(env: &dyn EnvSource, arg: &str, err: &mut dyn Write) -> Result<String> {
    let path = session::path_in(&session::sessions_dir_in(env), arg)?;
    match std::fs::symlink_metadata(&path) {
        Ok(_) => return Ok(arg.to_string()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            return Err(e).with_context(|| format!("checking session record {}", path.display()))
        }
    }
    let trace = crate::audit_view::resolve_root_session(&Recorder::dir_in(env), arg, err)
        .with_context(|| {
            format!(
                "no session record at {}; looking {arg:?} up as an audit trace prefix",
                path.display()
            )
        })?;
    match trace {
        Some(id) => Ok(id),
        None => bail!(
            "unknown session id {arg:?} (no record at {}) and no audit trace matches {arg:?}",
            path.display()
        ),
    }
}

/// A follow-up turn on a recorded session, reproducing its invocation under a
/// freshly selected Claude account (`profile`, or the usage-driven choice);
/// with `fork`, the turn starts a new session from the recorded one.
pub fn resume(
    session_id: &str,
    fork: bool,
    profile: Option<&str>,
    positional: Option<&str>,
    prompt_file: Option<&str>,
    as_json: bool,
) -> Result<i32> {
    let mut input = std::io::stdin();
    let mut output = std::io::stdout();
    resume_from(
        session_id,
        fork,
        profile,
        positional,
        prompt_file,
        as_json,
        &ProcessEnv,
        Config::load,
        &mut input,
        &mut output,
        &mut std::io::stderr(),
    )
}

#[allow(clippy::too_many_arguments)]
fn resume_from<L>(
    session_id: &str,
    fork: bool,
    profile: Option<&str>,
    positional: Option<&str>,
    prompt_file: Option<&str>,
    as_json: bool,
    env: &dyn EnvSource,
    load_config: L,
    input: &mut dyn Read,
    output: &mut dyn Write,
    diagnostics: &mut dyn Write,
) -> Result<i32>
where
    L: FnOnce() -> Result<Config>,
{
    let resolved = resolve_resume_arg(env, session_id, diagnostics)?;
    let session_id = resolved.as_str();
    let recorder = Recorder::new(env)?;
    let prompt_result = prompt_text_from(positional, prompt_file, input);
    let start = StartPayload {
        pid: std::process::id(),
        cmd: AuditCommand::Resume,
        agent: None,
        session: Some(session_id.to_string()),
        cwd_arg: None,
        profile_arg: profile.map(str::to_string),
        harness_arg: None,
        json: as_json,
        prompt_source: prompt_source(positional, prompt_file),
        prompt_file: prompt_file.map(str::to_string),
        prompt: prompt_result.as_ref().ok().cloned(),
        passthrough: vec![],
        fork,
    };
    let completion =
        audited_with_recorder_in(
            recorder,
            start,
            env,
            load_config,
            |ctx| match prompt_result {
                Ok(prompt) => resume_with(ctx, session_id, fork, profile, prompt, as_json, output),
                Err(e) => Err(e.into()),
            },
        )?;
    Ok(completion.code)
}

/// Where the prompt comes from, given the positional and `--prompt-file`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptSpec<'a> {
    Inline(&'a str),
    File(&'a str),
    Stdin,
    Conflict,
}

/// A positional `-` means stdin, same as `--prompt-file -`; both together
/// is one instruction, not a conflict.
fn prompt_spec<'a>(positional: Option<&'a str>, prompt_file: Option<&'a str>) -> PromptSpec<'a> {
    match (positional, prompt_file) {
        (None | Some("-"), None | Some("-")) => PromptSpec::Stdin,
        (None, Some(path)) => PromptSpec::File(path),
        (Some(p), None) => PromptSpec::Inline(p),
        (Some(_), Some(_)) => PromptSpec::Conflict,
    }
}

fn prompt_source(positional: Option<&str>, file: Option<&str>) -> Option<PromptSource> {
    match prompt_spec(positional, file) {
        PromptSpec::Inline(_) => Some(PromptSource::Arg),
        PromptSpec::File(_) => Some(PromptSource::File),
        PromptSpec::Stdin => Some(PromptSource::Stdin),
        PromptSpec::Conflict => None,
    }
}

fn error_end(e: &anyhow::Error) -> EndPayload {
    EndPayload {
        exit: Some(2),
        signal: None,
        session: None,
        result: None,
        error: Some(format!("{e:#}")),
        warnings: vec![],
        native: None,
        turns: None,
    }
}

fn audited_with_recorder_in<F, L>(
    recorder: Recorder,
    start: StartPayload,
    env: &dyn EnvSource,
    load_config: L,
    run: F,
) -> Result<Completion>
where
    F: FnOnce(&Ctx<'_>) -> std::result::Result<Completion, OperationError>,
    L: FnOnce() -> Result<Config>,
{
    if let Err(e) = recorder.start(start.clone()) {
        let trace_path = recorder.trace_path();
        if trace_path.is_file() {
            recorder.end(error_end(&e));
        }
        return Err(e);
    }
    let cfg = match load_config() {
        Ok(c) => c,
        Err(e) => {
            recorder.end(error_end(&e));
            return Err(e);
        }
    };
    let now = Timestamp::now();
    let bus = crate::bus::open(
        &cfg,
        env,
        format!("ccc-{}", recorder.lineage().invocation_id),
    );
    let live = Live::new(
        bus.identity(),
        recorder.lineage(),
        &start,
        recorder.agents_config(),
        cfg.bus.prompt_chars,
        now,
    );
    let usage = || usage::fetch(&cfg.mqtt, &cfg.claude.selection, env);
    let accounts = || crate::budget::bus_accounts(&cfg, env, Timestamp::now(), &*bus);
    let ctx = Ctx {
        cfg: &cfg,
        env,
        now,
        usage: &usage,
        accounts: &accounts,
        api_keys_eligible: true,
        bus: &*bus,
        live: &live,
        audit: &recorder,
        session_locks: Arc::new(Mutex::new(Vec::new())),
    };
    let result = run(&ctx);
    match result {
        Ok(mut completion) => {
            close_bus(&*bus, &live, &mut completion.end, &stderr_note);
            recorder.end(completion.end.clone());
            Ok(completion)
        }
        Err(e) => {
            let mut end = e.end.map(|end| *end).unwrap_or_else(|| error_end(&e.error));
            close_bus(&*bus, &live, &mut end, &stderr_note);
            recorder.end(end);
            Err(e.error)
        }
    }
}

/// Ends this invocation on the bus from its end record, then closes the bus;
/// every note the bus raised is said through `say` and joins the end
/// record's warnings. This runs after any child has exited, so a note never
/// lands on a terminal a `launch` child owns.
fn close_bus(bus: &dyn Bus, live: &Live, end: &mut EndPayload, say: &dyn Fn(&str)) {
    live.ended(bus, end, Timestamp::now());
    for note in bus.close() {
        let line = format!("note: {note}");
        say(&line);
        end.warnings.push(line);
    }
}

fn stderr_note(line: &str) {
    diag(format_args!("{line}"));
}

/// Everything a resume needs before a harness is spawned: the record, the
/// resolution rebuilt from it, the account, and the record as it will be saved.
struct ResumePrepared {
    r: Resolved,
    account: Option<Account>,
    notes: Vec<String>,
    /// The record to persist for the session this invocation runs: the
    /// resumed record updated in place, or a fresh record derived from it
    /// when forking.
    record: SessionRecord,
}

/// Loads and admits the recorded session `id`, places the turn on the Claude
/// account `pick` names, and saves the updated record.
/// When forking, derives the fork's record instead and leaves the parent's
/// sidecar and lock alone.
fn resume_prologue(ctx: &Ctx, id: &str, fork: bool, pick: AccountPick) -> Result<ResumePrepared> {
    let parent = session::load_from(&session::sessions_dir_in(ctx.env), id)?;
    if parent.harness == Harness::Goose {
        if fork {
            bail!(
                "goose session {id} cannot be forked: Goose has no headless fork, and its interactive fork copies the parent's name, which is ccc's session id"
            );
        }
        if parent.goose_policy.is_none() {
            bail!("goose session {id} has no recorded confinement policy, so it cannot be resumed");
        }
    }
    if fork {
        // A fork never writes the parent's transcript, so a live owner does
        // not stop it. The check is read-only: taking the parent's lock, even
        // for an instant, would refuse a concurrent `ccc resume` of the parent.
        match recorded_owner(ctx, &parent) {
            Ok(Some((_, proc))) if matches!(proc.alive(), crate::procs::Liveness::Alive) => {
                eprintln!(
                    "note: session {id} is in use; the fork starts from its transcript as it is now"
                )
            }
            Ok(_) => {}
            Err(e) => diag(format_args!(
                "warning: could not tell whether session {id} is in use: {e:#}"
            )),
        }
    } else {
        admit_resume(ctx, id, &parent)?;
    }
    let def = match &parent.agent {
        Some(agent) => Some(AgentDef::load(&ctx.cfg.agents_dir_in(ctx.env), agent)?),
        None => None,
    };
    let r = resolved_from(&parent, def.as_ref());

    // The recorded profile and auth play no part: each turn is placed on the
    // account that is best now, under the configured auth, exactly as a fresh
    // run would be. Every profile reads one shared transcript store, so any
    // account can continue the session.
    let (account, notes) = turn_account(ctx, &r, pick)?;

    let mut record = if fork {
        SessionRecord {
            created_at: ctx.now,
            last_used_at: ctx.now,
            trace: None,
            inv: None,
            forked_from: Some(id.to_string()),
            ..parent.clone()
        }
    } else {
        let mut resumed = parent.clone();
        resumed.touch(ctx.now);
        resumed
    };
    record.profile = account.as_ref().map(|a| a.profile.clone());
    record.auth = account.as_ref().map(|a| a.auth);
    fill_lineage(ctx, &mut record);
    if !fork {
        save_session_required(ctx, id, &record)?;
    }
    Ok(ResumePrepared {
        r,
        account,
        notes,
        record,
    })
}

/// [`resume`] against an explicit context and output sink.
fn resume_with(
    ctx: &Ctx,
    session_id: &str,
    fork: bool,
    profile: Option<&str>,
    prompt: String,
    as_json: bool,
    out: &mut dyn Write,
) -> std::result::Result<Completion, OperationError> {
    resume_turn(
        ctx,
        session_id,
        fork,
        AccountPick::Requested(profile),
        prompt,
        as_json,
        out,
    )
}

/// Runs one resumed turn of `session_id` on the account `pick` names.
fn resume_turn(
    ctx: &Ctx,
    session_id: &str,
    fork: bool,
    pick: AccountPick,
    prompt: String,
    as_json: bool,
    out: &mut dyn Write,
) -> std::result::Result<Completion, OperationError> {
    let no_retry = match &pick {
        AccountPick::Requested(None) => None,
        AccountPick::Requested(Some(_)) => Some(PINNED),
        AccountPick::Chosen(_) => Some(ALREADY_CONTINUED),
    };
    let ResumePrepared {
        r,
        account,
        notes,
        record,
    } = resume_prologue(ctx, session_id, fork, pick)?;

    let common = Common {
        cwd: record.cwd.clone(),
        extra_args: extra_args(ctx.cfg, r.harness),
        mode: Mode {
            io: Io::OneShot { prompt },
            origin: Origin::Resume {
                id: session_id.to_string(),
                fork,
            },
        },
        preset_session_id: None,
    };
    let driven = drive_with_meta(ctx, &r, &common, account.clone(), notes, &record)?;
    let DriveResult { result, signal } = driven;

    match result {
        Ok(mut outcome) => {
            if let Some(warning) =
                fork_collision_warning(&common.mode, Some(outcome.session_id.as_str()))
            {
                outcome.warnings.push(warning);
            } else {
                save_session_discovered(ctx, &outcome.session_id, &record);
            }
            let session = outcome.session_id.clone();
            let result_text = outcome.result.clone();
            let native = outcome.native.clone();
            let warnings = outcome.warnings.clone();
            let end = EndPayload {
                exit: Some(0),
                signal,
                session: Some(session),
                result: Some(result_text),
                error: None,
                warnings,
                native: Some(native),
                turns: None,
            };
            let code = match finish(&r, &common.cwd, account.as_ref(), outcome, as_json, out) {
                Ok(code) => code,
                Err(e) => {
                    let mut failed = end.clone();
                    failed.error = Some(format!("{e:#}"));
                    return Err(OperationError {
                        error: e,
                        end: Some(Box::new(failed)),
                    });
                }
            };
            Ok(Completion::new(code, end))
        }
        Err(err) => {
            if !fork {
                // The record names the account this turn ran under, whether or
                // not the turn itself succeeded.
                save_session(ctx, session_id, &record);
            }
            let reported = reported_session_id(&err);
            let mut warnings = err.warnings().to_vec();
            let collision = fork_collision_warning(&common.mode, reported.as_deref());
            let collided = collision.is_some();
            match collision {
                Some(warning) => {
                    report_warnings(std::slice::from_ref(&warning));
                    warnings.push(warning);
                }
                None => {
                    if let Some(id) = reported.as_deref().filter(|id| *id != session_id) {
                        save_session_discovered(ctx, id, &record);
                    }
                }
            }
            let not_continued = if collided {
                None
            } else {
                match retry_on_limit(ctx, &r, no_retry, account.as_ref(), &err, as_json, out) {
                    Some(LimitOutcome::Continued(done)) => return done,
                    Some(LimitOutcome::NotContinued(why)) => Some(why),
                    None => None,
                }
            };
            if let Some(why) = &not_continued {
                warnings.extend(why.lines().iter().cloned());
            }
            let session = reported;
            let result_text = Some(err.message().to_string());
            let error_text = match &err {
                HarnessError::Unusable { message, .. } => Some(message.clone()),
                HarnessError::Reported { .. } | HarnessError::Limited { .. } => None,
            };
            let native = err.reported().map(|(_, native)| native.clone());
            let code = fail(
                &r,
                &common.cwd,
                account.as_ref(),
                err,
                not_continued,
                as_json,
                out,
            );
            let end = EndPayload {
                exit: Some(code),
                signal,
                session,
                result: result_text,
                error: error_text,
                warnings,
                native,
                turns: None,
            };
            Ok(Completion::new(code, end))
        }
    }
}

/// Supervises an interactive harness session and returns its exit status.
pub fn launch(
    target: LaunchTarget,
    profile: Option<&str>,
    passthrough: Vec<String>,
) -> Result<i32> {
    launch_in(
        target,
        profile,
        passthrough,
        &ProcessEnv,
        Config::load,
        &mut std::io::stderr(),
    )
}

fn launch_in<L>(
    target: LaunchTarget,
    profile: Option<&str>,
    passthrough: Vec<String>,
    env: &dyn EnvSource,
    load_config: L,
    diagnostics: &mut dyn Write,
) -> Result<i32>
where
    L: FnOnce() -> Result<Config>,
{
    let resolved;
    let target = match target {
        LaunchTarget::Resume { id, fork } => {
            resolved = resolve_resume_arg(env, id, diagnostics)?;
            LaunchTarget::Resume {
                id: resolved.as_str(),
                fork,
            }
        }
        other => other,
    };
    let recorder = Recorder::new(env)?;
    let start = launch_start(target, profile, &passthrough);
    let completion = audited_with_recorder_in(recorder, start, env, load_config, |ctx| {
        launch_with_ctx(ctx, target, profile, passthrough)
    })?;
    Ok(completion.code)
}

/// The start record `ccc launch` writes for `target`.
fn launch_start(
    target: LaunchTarget,
    profile: Option<&str>,
    passthrough: &[String],
) -> StartPayload {
    let (agent, session, cwd_arg, harness_arg, fork) = match target {
        LaunchTarget::Fresh {
            agent,
            cwd,
            cli_harness,
            ..
        } => (
            Some(agent.to_string()),
            None,
            cwd.map(str::to_string),
            cli_harness,
            false,
        ),
        LaunchTarget::Roleless { cwd, .. } => (None, None, cwd.map(str::to_string), None, false),
        LaunchTarget::Resume { id, fork } => (None, Some(id.to_string()), None, None, fork),
    };
    StartPayload {
        pid: std::process::id(),
        cmd: AuditCommand::Launch,
        agent,
        session,
        cwd_arg,
        profile_arg: profile.map(str::to_string),
        harness_arg,
        json: false,
        prompt_source: None,
        prompt_file: None,
        prompt: None,
        passthrough: passthrough.to_vec(),
        fork,
    }
}

fn launch_with_ctx(
    ctx: &Ctx,
    target: LaunchTarget,
    profile: Option<&str>,
    passthrough: Vec<String>,
) -> std::result::Result<Completion, OperationError> {
    if profile.is_some_and(|p| ctx.cfg.claude.api_keys.contains_key(p)) {
        // TODO(api-key-at-launch): an interactive session reports no cost, so
        // a key spent at the terminal would bypass the budget.
        return Err(anyhow::anyhow!(
            "an API key account cannot be used at the terminal: ccc cannot see what an interactive session spends"
        )
        .into());
    }
    let terminal = ctx.at_terminal();
    let ctx = &terminal;
    let LaunchPrepared {
        mut inv,
        r,
        account,
        notes,
        common,
        record,
    } = match target {
        LaunchTarget::Fresh {
            agent,
            cwd,
            cli_harness,
            allow_read,
            allow_write,
            allow_exec,
        } => launch_invocation(
            ctx,
            prepare(
                ctx,
                agent,
                cwd,
                profile,
                cli_harness,
                &GooseFlags {
                    allow_read: allow_read.to_vec(),
                    allow_write: allow_write.to_vec(),
                    allow_exec: allow_exec.to_vec(),
                    nested: false,
                },
            )?,
            passthrough,
        )?,
        LaunchTarget::Roleless { cwd, model, effort } => launch_invocation(
            ctx,
            prepare_resolved(
                ctx,
                resolve::resolve_roleless(ctx.cfg, model, effort)?,
                Vec::new(),
                cwd,
                profile,
            )?,
            passthrough,
        )?,
        LaunchTarget::Resume { id, fork } => {
            launch_resume_invocation(ctx, id, fork, profile, passthrough)?
        }
    };
    inv.env_set.extend(ctx.audit.lineage().child_env());
    let mut notes = notes;
    if let Some(note) = withhold_api_key(ctx.cfg, ctx.env, r.harness, &mut inv) {
        diag(format!("note: {note}"));
        notes.push(note);
    }
    spawn_prologue(
        ctx,
        &r,
        &common,
        account.as_ref(),
        notes,
        &record,
        &|message| diag(message),
    )?;
    if r.harness == Harness::Goose {
        // A resumed Goose session is marked as opened at a terminal only once
        // nothing but the spawn is left to fail; a fresh one was saved marked.
        if let Some(id) = common.mode.continues() {
            save_session_required(ctx, id, &record)?;
        }
        eprintln!(
            "note: tools added in this session with /extension or /builtin stay with it on resume"
        );
    }
    // The id this invocation is known to be before the harness says so: the
    // preset (fresh Claude or Goose) or the resumed id (non-fork resume).
    let expected = common
        .preset_session_id
        .clone()
        .or_else(|| common.mode.continues().map(str::to_string));
    let on_spawned = |proc, pgid| {
        let Ok(supervisor) = crate::procs::ProcId::current() else {
            diag("warning: could not identify supervisor for child audit");
            return;
        };
        if let Err(e) = ctx.audit.child(ChildPayload {
            proc,
            pgid,
            supervisor,
        }) {
            diag(format_args!("warning: child audit failed: {e:#}"));
        }
    };
    let on_signal = |payload| {
        if let Err(e) = ctx.audit.signal(payload) {
            diag(format_args!("warning: signal audit failed: {e:#}"));
        }
    };
    let supervised = supervise::run(
        &inv,
        &SuperviseOpts {
            detach_group: false,
            grace: std::time::Duration::from_secs_f64(ctx.cfg.process.kill_grace_s),
            capture: false,
            on_spawned: &on_spawned,
            on_signal: Some(&on_signal),
        },
        &mut |_: &str| {},
    )?;
    let status = supervised.status;
    let turns = ctx.audit.turns();
    let last = turns.last();
    #[cfg(unix)]
    let signal = std::os::unix::process::ExitStatusExt::signal(&status);
    #[cfg(not(unix))]
    let signal = None;
    let code = if supervised.termination.is_some() {
        crate::harness::EXIT_NO_RESULT
    } else {
        status
            .code()
            .or_else(|| signal.map(|_| crate::harness::EXIT_NO_RESULT))
            .unwrap_or(crate::harness::EXIT_NO_RESULT)
    };
    report_warnings(&supervised.warnings);
    let mut warnings = supervised.warnings;
    if r.harness == Harness::Goose {
        let (end, note) = goose_launch_end(expected, &turns);
        if let Some(note) = note {
            eprintln!("{note}");
        }
        return Ok(Completion::new(
            code,
            EndPayload {
                exit: Some(code),
                signal,
                error: supervised
                    .termination
                    .as_ref()
                    .map(|term| termination_message(term, &status)),
                warnings,
                ..end
            },
        ));
    }
    if let Some(warning) =
        session_mismatch_warning(expected.as_deref(), last.map(|t| t.session.as_str()))
    {
        report_warnings(std::slice::from_ref(&warning));
        warnings.push(warning);
    }
    let last_id = last.map(|t| t.session.as_str());
    if let Some(warning) = fork_collision_warning(&common.mode, last_id) {
        report_warnings(std::slice::from_ref(&warning));
        warnings.push(warning);
    } else {
        match last_id {
            Some(id) if expected.as_deref() != Some(id) => save_session_learned(ctx, id, &record),
            // The preset or resumed id, saved before the spawn.
            Some(_) => {}
            None if expected.is_none() => {
                eprintln!("note: no turn completed, so no session was recorded")
            }
            None => {}
        }
    }
    let end = EndPayload {
        exit: Some(code),
        signal,
        session: last.map(|t| t.session.clone()).or(expected),
        result: last.and_then(|t| t.result.clone()),
        error: supervised
            .termination
            .as_ref()
            .map(|term| termination_message(term, &status)),
        warnings,
        native: None,
        turns: Some(turns.len()),
    };
    Ok(Completion::new(code, end))
}

/// The session, result and turn count of a Goose launch's end record, and the
/// note for the human. Only the launch's own session is recorded: a last turn
/// in another session, one the human opened with `/new`, gets the note and
/// nothing saved, and the result is the last answer the launch's session gave.
fn goose_launch_end(
    expected: Option<String>,
    turns: &[audit::Turn],
) -> (EndPayload, Option<String>) {
    let note = match (expected.as_deref(), turns.last()) {
        (Some(own), Some(last)) if last.session != own => Some(format!(
            "note: the launch ended in a Goose session started with /new ({}); ccc does not record it; ccc launch --resume {own} reopens the session this launch started",
            last.session
        )),
        _ => None,
    };
    let result = turns
        .iter()
        .rev()
        .find(|t| Some(t.session.as_str()) == expected.as_deref())
        .and_then(|t| t.result.clone());
    let end = EndPayload {
        exit: None,
        signal: None,
        session: expected,
        result,
        error: None,
        warnings: Vec::new(),
        native: None,
        turns: Some(turns.len()),
    };
    (end, note)
}

/// Marks `cwd` trusted for `account`. Best-effort: a failure is a diag, not an
/// error, because the consequence is the status quo — the trust dialog, or
/// ignored workspace settings — not a failed run.
// TODO(trust-scope-gate): any cwd is trusted unconditionally and the marker
// outlives the run, so a `--cwd` naming an untrusted checkout makes that tree's
// hooks and MCP servers run under the account's credentials, and leaves the
// directory trusted for later interactive sessions too. Narrowing this — an
// allowlist of roots, propagating only a directory some profile already trusts,
// or auto-trusting from `launch` alone — is a policy choice, not a patch.
fn trust_cwd(account: Option<&Account>, env: &dyn EnvSource, cwd: &Path) {
    if let Err(err) = crate::harness::claude::trust_directory(account, env, cwd) {
        let whose = match account {
            Some(a) => format!("profile {}", a.profile),
            None => AMBIENT.to_string(),
        };
        diag(format_args!(
            "could not mark {} trusted for {whose}: {err:#}",
            cwd.display()
        ));
    }
}

/// The interactive invocation `launch` hands the terminal to, with the account
/// it was built for, so a caller acts on the account the child runs as.
fn launch_invocation(
    ctx: &Ctx,
    prepared: Prepared,
    passthrough: Vec<String>,
) -> Result<LaunchPrepared> {
    let Prepared {
        r,
        cwd,
        account,
        notes,
    } = prepared;

    // Configured arguments first, then what the caller typed, so the caller's
    // copy of a repeated flag is the one the harness sees last.
    let mut extra = extra_args(ctx.cfg, r.harness);
    extra.extend(passthrough);
    let common = Common {
        cwd,
        extra_args: extra,
        mode: Mode {
            io: Io::Interactive {
                self_exe: std::env::current_exe()?,
            },
            origin: Origin::Fresh,
        },
        preset_session_id: presets_session(r.harness).then(|| Uuid::new_v4().to_string()),
    };
    let inv = driver_for(ctx, &r, &common, account.clone(), None)?.build(&r, &common)?;
    let mut record = record_of(&r, &common.cwd, account.as_ref(), ctx.now);
    // Tools the human adds at the terminal are saved with a Goose session, so
    // its policy no longer bounds it for a confined agent's `ccc_resume`.
    record.opened_at_terminal = r.harness == Harness::Goose;
    fill_lineage(ctx, &mut record);
    Ok(LaunchPrepared {
        inv,
        r,
        account,
        notes,
        common,
        record,
    })
}

/// The interactive invocation that continues the recorded session `id`, or
/// forks it.
fn launch_resume_invocation(
    ctx: &Ctx,
    id: &str,
    fork: bool,
    profile: Option<&str>,
    passthrough: Vec<String>,
) -> Result<LaunchPrepared> {
    let ResumePrepared {
        r,
        account,
        notes,
        mut record,
    } = resume_prologue(ctx, id, fork, AccountPick::Requested(profile))?;
    // Saved with the mark only once the spawn prologue has passed.
    record.opened_at_terminal |= record.harness == Harness::Goose;
    let mut extra = extra_args(ctx.cfg, r.harness);
    extra.extend(passthrough);
    let common = Common {
        cwd: record.cwd.clone(),
        extra_args: extra,
        mode: Mode {
            io: Io::Interactive {
                self_exe: std::env::current_exe()?,
            },
            origin: Origin::Resume {
                id: id.to_string(),
                fork,
            },
        },
        preset_session_id: None,
    };
    let inv = driver_for(ctx, &r, &common, account.clone(), None)?.build(&r, &common)?;
    Ok(LaunchPrepared {
        inv,
        r,
        account,
        notes,
        common,
        record,
    })
}

/// The startup every entry point shares: resolve the agent, pick a working
/// directory, and mint a Claude account when applicable.
fn prepare(
    ctx: &Ctx,
    agent: &str,
    cwd: Option<&str>,
    profile: Option<&str>,
    cli_harness: Option<Harness>,
    goose: &GooseFlags,
) -> Result<Prepared> {
    let def = AgentDef::load(&ctx.cfg.agents_dir_in(ctx.env), agent)?;
    let (r, notes) = resolve::resolve(ctx.cfg, ctx.env, &def, cli_harness)?;
    for note in &notes {
        eprintln!("note: {note}");
    }
    if r.harness != Harness::Goose {
        if let Some(flag) = goose.first_allow() {
            bail!(
                "{flag} applies only to the goose harness; agent type {agent:?} resolved to {}, which would not honour it",
                r.harness
            );
        }
    }
    let mut prepared = prepare_resolved(ctx, r, notes, cwd, profile)?;
    if prepared.r.harness == Harness::Goose {
        // The policy sees `--cwd` as given: a nested run refuses a path that
        // is not already canonical rather than following a symlink on it.
        let given = match cwd {
            Some(c) => crate::config::expand_tilde_in(ctx.env, c),
            None => prepared.cwd.clone(),
        };
        prepared.r.goose = Some(goose_policy(ctx.cfg, ctx.env, &def, &given, goose)?);
    }
    Ok(prepared)
}

/// The confinement policy of a fresh Goose run of `def` in `cwd`.
pub fn goose_policy(
    cfg: &Config,
    env: &dyn EnvSource,
    def: &AgentDef,
    cwd: &Path,
    flags: &GooseFlags,
) -> Result<crate::goose_policy::GoosePolicy> {
    let self_exe = std::env::current_exe().context("locating the ccc binary")?;
    let protected =
        crate::goose_policy::ProtectedDirs::gather_in(env, cfg, &cfg.goose.root_in(env), &self_exe);
    crate::goose_policy::compute(
        &crate::goose_policy::PolicyRequest {
            tools: &def.goose_tools,
            agents: &def.goose_agents,
            cwd,
            allow_read: &flags.allow_read,
            allow_write: &flags.allow_write,
            allow_exec: &flags.allow_exec,
            nested: flags.nested,
        },
        &protected,
    )
}

/// Builds the Goose box for `policy` and starts it once with `/bin/true`, so a
/// missing bwrap, disabled user namespaces, or a root deleted or replaced by a
/// symlink fails here with exit 2 instead of at Goose's extension start.
fn box_preflight(ctx: &Ctx, policy: &crate::goose_policy::GoosePolicy) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        use crate::goose_tools::boxed;
        if !policy.needs_box() {
            return Ok(());
        }
        let settings = boxed::BoxSettings::from_config(ctx.cfg, ctx.env)?;
        let self_exe = std::env::current_exe().context("locating the ccc binary")?;
        boxed::preflight(
            policy,
            &settings,
            &self_exe,
            std::time::Duration::from_secs_f64(ctx.cfg.process.kill_grace_s),
        )
    }
    #[cfg(not(target_os = "linux"))]
    {
        if policy.needs_box() {
            bail!("a Goose run with read, write or shell tools needs the bubblewrap box, which needs Linux");
        }
        let _ = ctx;
        Ok(())
    }
}

/// Whether a fresh session of `harness` gets its id from ccc before the spawn.
fn presets_session(harness: Harness) -> bool {
    matches!(harness, Harness::Claude | Harness::Goose)
}

/// The startup after resolution: pick a working directory, mint a Claude
/// account when applicable, and append the account notes to `notes`.
fn prepare_resolved(
    ctx: &Ctx,
    r: Resolved,
    mut notes: Vec<String>,
    cwd: Option<&str>,
    profile: Option<&str>,
) -> Result<Prepared> {
    let cwd = resolve_cwd(ctx.env, cwd)?;
    let (account, account_notes) = turn_account(ctx, &r, AccountPick::Requested(profile))?;
    notes.extend(account_notes);
    Ok(Prepared {
        r,
        cwd,
        account,
        notes,
    })
}

/// Builds the driver, spawns the harness and parses what it produced.
#[allow(dead_code)]
fn drive(
    ctx: &Ctx,
    r: &Resolved,
    common: &Common,
    account: Option<Account>,
    selection_notes: Vec<String>,
    record: &SessionRecord,
) -> Result<Result<Outcome, HarnessError>> {
    Ok(drive_with_meta(ctx, r, common, account, selection_notes, record)?.result)
}

fn drive_with_meta(
    ctx: &Ctx,
    r: &Resolved,
    common: &Common,
    account: Option<Account>,
    selection_notes: Vec<String>,
    record: &SessionRecord,
) -> Result<DriveResult> {
    drive_with_meta_report(
        ctx,
        r,
        common,
        account,
        selection_notes,
        &|message| diag(message),
        record,
    )
}

/// Spawns the harness for `common` and parses its result. `record` arrives
/// complete, lineage included, and is saved as-is under any session id this
/// drive presets or learns.
fn drive_with_meta_report(
    ctx: &Ctx,
    r: &Resolved,
    common: &Common,
    account: Option<Account>,
    selection_notes: Vec<String>,
    report: &(dyn Fn(String) + Send + Sync),
    record: &SessionRecord,
) -> Result<DriveResult> {
    let profile = account.as_ref().map(|a| a.profile.clone());
    let output_file = match r.harness {
        Harness::Codex => Some(codex::new_output_file(ctx.env)?),
        Harness::Claude | Harness::Goose => None,
    };
    let driver = driver_for(ctx, r, common, account.clone(), output_file.clone())?;
    let mut inv = driver.build(r, common)?;
    inv.env_set.extend(ctx.audit.lineage().child_env());
    let mut selection_notes = selection_notes;
    if let Some(note) = withhold_api_key(ctx.cfg, ctx.env, r.harness, &mut inv) {
        report(format!("note: {note}"));
        selection_notes.push(note);
    }
    if let Err(e) = spawn_prologue(
        ctx,
        r,
        common,
        account.as_ref(),
        selection_notes,
        record,
        report,
    ) {
        if let Some(path) = &output_file {
            let _ = codex::take_output_file(path);
        }
        return Err(e);
    }
    let (mut driven, cost) = supervise_and_parse(
        ctx,
        r,
        common,
        &*driver,
        &inv,
        profile.as_deref(),
        output_file,
        report,
        record,
    );
    match account.as_ref() {
        Some(account) if account.key_file().is_some() => {
            settle_key_turn(ctx, account, cost, &mut driven.result);
        }
        Some(account)
            if r.harness == Harness::Claude
                && ctx.cfg.claude.profiles.contains(&account.profile) =>
        {
            settle_profile_turn(ctx, account, &r.model, &mut driven.result);
        }
        _ => {}
    }
    ctx.live.charged(ctx.bus, cost, Timestamp::now());
    Ok(driven)
}

/// Runs the built invocation under the supervisor and parses its result.
/// Returns the result together with the cost the harness reported for the
/// turn, `None` when it reported none: a child that never started or was
/// terminated, or a process none of whose results stated a cost in
/// `model_usage_since_last_result`. The cost is read before
/// a prompt shortfall discards a successful result, because that turn was
/// billed.
#[allow(clippy::too_many_arguments)]
fn supervise_and_parse(
    ctx: &Ctx,
    r: &Resolved,
    common: &Common,
    driver: &dyn Driver,
    inv: &Invocation,
    profile: Option<&str>,
    output_file: Option<PathBuf>,
    report: &(dyn Fn(String) + Send + Sync),
    record: &SessionRecord,
) -> (DriveResult, Option<f64>) {
    let run_ctx = RunContext::new(r, profile, inv);

    let on_spawned = |proc, pgid| {
        let Ok(supervisor) = crate::procs::ProcId::current() else {
            diag("warning: could not identify supervisor for child audit");
            return;
        };
        if let Err(e) = ctx.audit.child(ChildPayload {
            proc,
            pgid,
            supervisor,
        }) {
            diag(format_args!("warning: child audit failed: {e:#}"));
        }
    };
    let on_signal = |payload| {
        if let Err(e) = ctx.audit.signal(payload) {
            diag(format_args!("warning: signal audit failed: {e:#}"));
        }
    };
    let first_id = Arc::new(Mutex::new(None::<String>));
    let first_id_for_callback = Arc::clone(&first_id);
    let mut on_line = |line: &str| {
        let Some(id) = driver.early_session_id(line) else {
            return;
        };
        let mut seen = first_id_for_callback
            .lock()
            .expect("early session mutex poisoned");
        if seen.is_some() {
            return;
        }
        *seen = Some(id.clone());
        drop(seen);
        if common.mode.continues().is_some() {
            return;
        }
        if let Some(warning) = fork_collision_warning(&common.mode, Some(&id)) {
            report(format!("warning: {warning}"));
        } else {
            match acquire_session_lock(ctx, &id, LockPolicy::Discovery) {
                SessionLockResult::Acquired(Some(lock)) => {
                    ctx.retain_session_lock(&id, lock);
                    save_session(ctx, &id, record);
                }
                SessionLockResult::Acquired(None) => {}
                SessionLockResult::Contended { message } => report(format!(
                    "warning: session discovery failed for {id}: {message}"
                )),
                SessionLockResult::Io(e) => {
                    report(format!("warning: session discovery failed for {id}: {e:#}"))
                }
            }
        }
        if let Err(e) = ctx.audit.session(&id) {
            diag(format_args!("warning: session audit failed: {e:#}"));
        }
    };
    let supervised = match supervise::run(
        inv,
        &SuperviseOpts {
            detach_group: true,
            grace: std::time::Duration::from_secs_f64(ctx.cfg.process.kill_grace_s),
            capture: true,
            on_spawned: &on_spawned,
            on_signal: Some(&on_signal),
        },
        &mut on_line,
    ) {
        Ok(value) => value,
        Err(e) => {
            // Nothing else reaps the minted `-o` file; a child that never ran
            // leaves it empty and untouched.
            if let Some(path) = &output_file {
                let _ = codex::take_output_file(path);
            }
            return (
                DriveResult {
                    result: Err(HarnessError::unusable(format!("{e:#}"))
                        .with_session_id(first_id.lock().unwrap().clone().or_else(|| {
                            common
                                .mode
                                .continues()
                                .map(str::to_string)
                                .or_else(|| common.preset_session_id.clone())
                        }))
                        .annotate(&run_ctx)),
                    signal: None,
                },
                None,
            );
        }
    };
    report_warnings(&supervised.warnings);
    let prompt_shortfall = supervised.prompt.as_ref().and_then(prompt_shortfall);
    if let Some(shortfall) = &prompt_shortfall {
        report(format!("warning: {}", prompt_shortfall_text(shortfall)));
    }
    let signal = exit_signal(&supervised.status);
    let mut captured = supervised.captured;
    captured.output_file = output_file.as_deref().and_then(codex::take_output_file);
    if let Some(term) = supervised.termination {
        let message = termination_message_with_prompt(
            termination_message(&term, &supervised.status),
            prompt_shortfall.as_ref(),
        );
        return (
            DriveResult {
                result: Err(HarnessError::unusable(message)
                    .with_session_id(first_id.lock().unwrap().clone().or_else(|| {
                        common
                            .mode
                            .continues()
                            .map(str::to_string)
                            .or_else(|| common.preset_session_id.clone())
                    }))
                    .annotate(&run_ctx)),
                signal,
            },
            None,
        );
    }
    let parsed = driver.parse(&captured);
    let cost = reported_cost(&parsed);
    let result = match (prompt_shortfall, parsed) {
        (Some(delivery), Ok(outcome)) => Err(HarnessError::unusable(format!(
            "the prompt was not fully delivered: {} of {} bytes were written before {}; the harness answered an incomplete prompt and its result is discarded",
            delivery.written,
            delivery.total,
            prompt_end_reason(&delivery.ended),
        ))
        .with_session_id(Some(outcome.session_id).or_else(|| {
            first_id.lock().unwrap().clone().or_else(|| common.mode.continues().map(str::to_string).or_else(|| common.preset_session_id.clone()))
        }))
        .annotate(&run_ctx)),
        (Some(delivery), Err(mut error)) => {
            apply_prompt_shortfall(&mut error, &delivery);
            Err(error.annotate(&run_ctx))
        }
        (None, parsed) => parsed.map_err(|e| e.annotate(&run_ctx)),
    };
    (DriveResult { result, signal }, cost)
}

/// The cost a parsed turn reported, succeeded, refused or limited.
fn reported_cost(parsed: &Result<Outcome, HarnessError>) -> Option<f64> {
    let native = match parsed {
        Ok(outcome) => &outcome.native,
        Err(e) => e.reported()?.1,
    };
    crate::harness::claude::reported_cost(native)
}

/// Charges a finished API key turn to this host's ledger and publishes the
/// new level, then, when the turn was refused or limited with a message one
/// of the key's `exhausted_patterns` matches, raises the key's exhausted flag
/// and makes the failure a usage limit, so the hand-off machinery treats the
/// key as it treats a limited profile; the flagged level is published too. A
/// ledger that cannot be written costs a note and nothing else.
fn settle_key_turn(
    ctx: &Ctx,
    account: &Account,
    cost: Option<f64>,
    result: &mut std::result::Result<Outcome, HarnessError>,
) {
    let name = account.profile.as_str();
    let Some(key) = ctx.cfg.claude.api_keys.get(name) else {
        return;
    };
    let ledgers = crate::budget::Ledgers::in_env(ctx.env);
    let publish =
        |ledger: &crate::budget::Ledger| crate::budget::publish_level(ctx.bus, name, key, ledger);
    match ledgers.charge(name, key, Timestamp::now(), cost) {
        Ok(ledger) => publish(&ledger),
        Err(e) => add_note(
            result,
            format!("note: could not charge the turn to API key {name}'s ledger: {e:#}"),
        ),
    }
    let (session_id, message, native, warnings) = match result {
        Err(HarnessError::Reported {
            session_id,
            message,
            native,
            warnings,
        })
        | Err(HarnessError::Limited {
            session_id,
            message,
            native,
            warnings,
            ..
        }) => (session_id, message, native, warnings),
        _ => return,
    };
    if !crate::glob::any_match(&key.exhausted_patterns, message) {
        return;
    }
    let limited = HarnessError::Limited {
        session_id: session_id.take(),
        message: std::mem::take(message),
        native: native.take(),
        warnings: std::mem::take(warnings),
        limit: None,
    };
    let marked = ledgers.mark_exhausted(
        name,
        key,
        Timestamp::now(),
        Some(ctx.audit.lineage().invocation_id.clone()),
        limited.message(),
    );
    *result = Err(limited);
    match marked {
        Ok(ledger) => publish(&ledger),
        Err(e) => add_note(
            result,
            format!("note: could not record API key {name} as exhausted: {e:#}"),
        ),
    }
}

/// When a profile turn ended in a usage limit whose refusal carried a reset
/// time still ahead, raises the flag of the bucket the refusal exhausted in
/// the profile's limits file and publishes the file. A limit without one
/// costs a warning and raises nothing, as does a file that cannot be
/// written; the turn's outcome stands either way.
fn settle_profile_turn(
    ctx: &Ctx,
    account: &Account,
    model: &str,
    result: &mut std::result::Result<Outcome, HarnessError>,
) {
    let Err(HarnessError::Limited { message, limit, .. }) = &*result else {
        return;
    };
    let name = account.profile.as_str();
    let now = Timestamp::now();
    let Some((info, until)) = limit
        .as_deref()
        .and_then(|l| l.resets_at.filter(|t| *t > now).map(|t| (l, t)))
    else {
        add_note(
            result,
            format!("{name} was not flagged exhausted: the limit carried no reset time"),
        );
        return;
    };
    let fable = crate::glob::any_match(&ctx.cfg.claude.selection.fable_models, model);
    let bucket = crate::limits::bucket_for(info.kind.as_deref(), fable);
    let mut flag = claude_usage_mqtt_core::Exhausted::new(
        now,
        Some(ctx.audit.lineage().invocation_id.clone()),
        Some(message.clone()),
    );
    flag.until = Some(until);
    flag.kind = info.kind.clone();
    match crate::limits::Limits::in_env(ctx.env).raise(name, bucket, flag, now) {
        Ok(file) => crate::limits::publish(ctx.bus, name, &file),
        Err(e) => add_note(
            result,
            format!("could not record {name} as exhausted: {e:#}"),
        ),
    }
}

/// Puts `note` among the turn's warnings, which reach stderr and the end
/// record; a turn with no warnings to carry it gets it on stderr only.
fn add_note(result: &mut std::result::Result<Outcome, HarnessError>, note: String) {
    let warnings = match result {
        Ok(outcome) => Some(&mut outcome.warnings),
        Err(e) => e.warnings_mut(),
    };
    match warnings {
        Some(warnings) => warnings.push(note),
        None => diag(&note),
    }
}

/// With an API key account configured, removes `ANTHROPIC_API_KEY` from a
/// Codex or Goose child, so the only child that receives it is a Claude key
/// turn, which is charged. An `env_set` entry for the variable (Goose's
/// `[goose.env]`) is the operator's and is kept. Returns a note when the
/// removal was added and the variable is set and non-blank in `env`.
fn withhold_api_key(
    cfg: &Config,
    env: &dyn EnvSource,
    harness: Harness,
    inv: &mut Invocation,
) -> Option<String> {
    use crate::harness::claude::API_KEY_VAR;
    if harness == Harness::Claude || cfg.claude.api_keys.is_empty() {
        return None;
    }
    if inv.env_set.iter().any(|(k, _)| k == API_KEY_VAR) {
        return None;
    }
    if !inv.env_remove.iter().any(|k| k == API_KEY_VAR) {
        inv.env_remove.push(API_KEY_VAR.to_string());
    }
    env.get(API_KEY_VAR)
        .is_some_and(|v| !v.trim().is_empty())
        .then(|| {
            format!(
                "{API_KEY_VAR} removed from the child's environment; an API key account is configured and a {harness} turn is not charged to it"
            )
        })
}

/// Pre-spawn bookkeeping common to one-shot and interactive paths.
/// Contention on the preset lock is fatal; lock I/O and sidecar failures are
/// reported and the spawn proceeds. `record` arrives complete, lineage
/// included, and is saved as-is under the preset id.
fn spawn_prologue(
    ctx: &Ctx,
    r: &Resolved,
    common: &Common,
    account: Option<&Account>,
    notes: Vec<String>,
    record: &SessionRecord,
    report: &(dyn Fn(String) + Send + Sync),
) -> Result<()> {
    if r.harness == Harness::Goose {
        crate::harness::goose::prepare_root(&ctx.cfg.goose.root_in(ctx.env))?;
        if let Some(policy) = &r.goose {
            box_preflight(ctx, policy)?;
        }
    }
    let preset = (presets_session(r.harness) && common.mode.starts_from().is_none())
        .then_some(common.preset_session_id.as_deref())
        .flatten();
    if let Some(id) = preset {
        match acquire_session_lock(ctx, id, LockPolicy::Admission) {
            SessionLockResult::Acquired(Some(lock)) => ctx.retain_session_lock(id, lock),
            SessionLockResult::Acquired(None) => {}
            SessionLockResult::Contended { message } => return Err(anyhow::anyhow!(message)),
            SessionLockResult::Io(e) => report(format!(
                "warning: could not reserve fresh {} session {id}: {e:#}",
                r.harness
            )),
        }
    }
    let spawn = spawn_payload(
        r,
        &common.cwd,
        account,
        common
            .preset_session_id
            .clone()
            .or_else(|| common.mode.continues().map(str::to_string)),
        notes,
    );
    ctx.audit.spawn(spawn.clone())?;
    if r.harness == Harness::Claude {
        if account.is_none() {
            let removed = crate::harness::claude::conflicts_set(ctx.env);
            if !removed.is_empty() {
                report(format!(
                    "note: {} removed from the child's environment; {AMBIENT} is the Claude Code login, not an API key or cloud provider",
                    removed.join(", ")
                ));
            }
        }
        trust_cwd(account, ctx.env, &common.cwd);
    }
    if let Some(id) = preset {
        save_session(ctx, id, record);
    }
    ctx.live.spawned(ctx.bus, &spawn, Timestamp::now());
    Ok(())
}

fn prompt_shortfall(delivery: &PromptDelivery) -> Option<PromptDelivery> {
    (delivery.written < delivery.total).then(|| delivery.clone())
}

fn termination_message_with_prompt(
    mut message: String,
    delivery: Option<&PromptDelivery>,
) -> String {
    if let Some(delivery) = delivery.filter(|delivery| delivery.written < delivery.total) {
        message.push_str(&format!(
            "; prompt: {} of {} bytes delivered",
            delivery.written, delivery.total
        ));
    }
    message
}

fn apply_prompt_shortfall(error: &mut HarnessError, delivery: &PromptDelivery) {
    let shortfall = prompt_shortfall_text(delivery);
    match error {
        HarnessError::Reported { warnings, .. } | HarnessError::Limited { warnings, .. } => {
            warnings.push(shortfall)
        }
        HarnessError::Unusable { message, .. } => message.push_str(&format!("; {shortfall}")),
    }
}

fn prompt_end_reason(end: &PromptEnd) -> String {
    match end {
        PromptEnd::Complete => "completion".into(),
        PromptEnd::ChildClosed => "the harness closed its stdin".into(),
        PromptEnd::Stopped => "the harness exited".into(),
        PromptEnd::Failed(error) => error.clone(),
    }
}

fn prompt_shortfall_text(delivery: &PromptDelivery) -> String {
    format!(
        "the prompt was not fully delivered: {} of {} bytes were written before {}",
        delivery.written,
        delivery.total,
        prompt_end_reason(&delivery.ended)
    )
}

#[cfg(unix)]
fn exit_signal(status: &std::process::ExitStatus) -> Option<i32> {
    std::os::unix::process::ExitStatusExt::signal(status)
}

#[cfg(not(unix))]
fn exit_signal(_status: &std::process::ExitStatus) -> Option<i32> {
    None
}

fn termination_message(term: &Termination, status: &std::process::ExitStatus) -> String {
    let actual = status
        .code()
        .map(|code| format!("exit {code}"))
        .or_else(|| exit_signal(status).map(|signal| format!("signal {signal}")))
        .unwrap_or_else(|| "unknown outcome".into());
    let escalation = term
        .escalated
        .map(|reason| format!("{reason}; escalated to SIGKILL"))
        .unwrap_or_else(|| "termination observed before grace expiry".into());
    format!("harness terminated with {actual}; requested SIG{} (sender pid {}; reason {}; {}; swept {} descendants)", term.requested, term.by.map(|p| p.to_string()).unwrap_or_else(|| "unknown".into()), term.reason, escalation, term.swept)
}

fn spawn_payload(
    r: &Resolved,
    cwd: &Path,
    account: Option<&Account>,
    session: Option<String>,
    selection_notes: Vec<String>,
) -> SpawnPayload {
    SpawnPayload {
        harness: r.harness,
        model: r.model.clone(),
        effort: r.effort,
        profile: account.map(|a| a.profile.clone()),
        auth: account.map(|a| a.auth),
        sandbox: r.sandbox,
        tools: r.tools.clone(),
        agent: r.agent.clone(),
        cwd: cwd.display().to_string(),
        session,
        selection_notes,
        goose_policy: r.goose.clone(),
    }
}

/// The driver for one harness, ready to build an invocation.
fn driver_for<'a>(
    ctx: &'a Ctx,
    r: &'a Resolved,
    common: &Common,
    account: Option<Account>,
    output_file: Option<PathBuf>,
) -> Result<Box<dyn Driver + 'a>> {
    Ok(match r.harness {
        Harness::Claude => Box::new(ClaudeDriver {
            permission_mode: ctx.cfg.claude.permission_mode,
            account,
            env: ctx.env,
        }),
        Harness::Codex => Box::new(CodexDriver {
            approval: ctx.cfg.codex.approval,
            output_file,
            env: ctx.env,
        }),
        Harness::Goose => Box::new(GooseDriver {
            cfg: &ctx.cfg.goose,
            policy: r
                .goose
                .as_ref()
                .context("a goose run has no confinement policy")?,
            self_exe: std::env::current_exe().context("locating the ccc binary")?,
            session: crate::harness::goose::session_id(common)?,
            env: ctx.env,
        }),
    })
}

/// The session id a reported failure came with, if the harness got that far.
fn reported_session_id(err: &HarnessError) -> Option<String> {
    err.session_id().map(str::to_string)
}

fn extra_args(cfg: &Config, harness: Harness) -> Vec<String> {
    match harness {
        Harness::Claude => cfg.claude.extra_args.clone(),
        Harness::Codex => cfg.codex.extra_args.clone(),
        Harness::Goose => cfg.goose.extra_args.clone(),
    }
}

/// Prints a successful turn and returns its exit code.
fn finish(
    r: &Resolved,
    cwd: &std::path::Path,
    account: Option<&Account>,
    outcome: Outcome,
    as_json: bool,
    sink: &mut dyn Write,
) -> Result<i32> {
    report_warnings(&outcome.warnings);
    let out = RunOutput {
        session: outcome.session_id,
        harness: r.harness,
        model: r.model.clone(),
        effort: r.effort,
        profile: account.map(|a| a.profile.clone()),
        agent: r.agent.clone(),
        cwd: cwd.to_path_buf(),
        result: outcome.result,
        native: outcome.native,
    };
    writeln!(sink, "{}", out.render(as_json)).context("writing the result")?;
    Ok(0)
}

/// Prints a failed turn: the harness's message on stderr, and — in text mode —
/// as the body, so a caller reading stdout still sees why. `limit` says why a
/// usage-limited turn was not continued on another profile.
fn fail(
    r: &Resolved,
    cwd: &std::path::Path,
    account: Option<&Account>,
    err: HarnessError,
    limit: Option<NotContinued>,
    as_json: bool,
    sink: &mut dyn Write,
) -> i32 {
    match limit {
        // The failed continuation already printed the warnings and the limit.
        Some(NotContinued::Failed(_)) => {}
        Some(NotContinued::Declined(line)) => {
            report_warnings(err.warnings());
            diag(&line);
        }
        _ => {
            // The diagnostics the harness streamed before it failed usually
            // say more about why than the terminal message does.
            report_warnings(err.warnings());
            diag(&err);
        }
    }
    let code = err.exit_code();
    if let Some((message, native)) = err.reported() {
        let out = RunOutput {
            session: err.session_id().unwrap_or("-").to_string(),
            harness: r.harness,
            model: r.model.clone(),
            effort: r.effort,
            profile: account.map(|a| a.profile.clone()),
            agent: r.agent.clone(),
            cwd: cwd.to_path_buf(),
            result: message.to_string(),
            native: native.clone(),
        };
        let _ = writeln!(sink, "{}", out.render(as_json));
    } else if let HarnessError::Unusable {
        session_id: Some(session),
        message,
    } = err
    {
        let out = RunOutput {
            session,
            harness: r.harness,
            model: r.model.clone(),
            effort: r.effort,
            profile: account.map(|a| a.profile.clone()),
            agent: r.agent.clone(),
            cwd: cwd.to_path_buf(),
            result: message,
            native: serde_json::Value::Null,
        };
        let _ = writeln!(sink, "{}", out.render(as_json));
    }
    code
}

/// A sidecar failure costs a follow-up turn, not this one's result.
fn save_session(ctx: &Ctx, id: &str, record: &SessionRecord) {
    if !ctx.owns_session(id) {
        diag(format_args!(
            "warning: refusing to save unlocked session {id}"
        ));
        return;
    }
    if let Err(e) = session::save_in(&session::sessions_dir_in(ctx.env), id, record) {
        diag(format_args!(
            "warning: the session was not recorded, so `ccc resume {id}` will not find it: {e:#}"
        ));
    }
}

fn save_session_required(ctx: &Ctx, id: &str, record: &SessionRecord) -> Result<()> {
    if !ctx.owns_session(id) {
        bail!("session {id} is not locked by this ccc")
    }
    session::save_in(&session::sessions_dir_in(ctx.env), id, record)
}

fn save_session_discovered(ctx: &Ctx, id: &str, record: &SessionRecord) {
    if !ctx.owns_session(id) {
        match acquire_session_lock(ctx, id, LockPolicy::Discovery) {
            SessionLockResult::Acquired(Some(lock)) => ctx.retain_session_lock(id, lock),
            SessionLockResult::Acquired(None) => {}
            SessionLockResult::Contended { message } => {
                diag(format_args!(
                    "warning: session discovery failed for {id}: {message}"
                ));
                return;
            }
            SessionLockResult::Io(e) => {
                diag(format_args!(
                    "warning: session discovery failed for {id}: {e:#}"
                ));
                return;
            }
        }
    }
    save_session(ctx, id, record);
}

/// Saves `record` under an id this invocation learned only from the
/// harness's turns. An id that already has a sidecar is another session
/// the human switched to inside the TUI (`/resume`); its record belongs
/// to the invocation that made it and is left alone. A brand-new id
/// (`/clear`, `/new`, a fresh thread, a fork) has no sidecar and is saved.
fn save_session_learned(ctx: &Ctx, id: &str, record: &SessionRecord) {
    let dir = session::sessions_dir_in(ctx.env);
    if matches!(session::path_in(&dir, id), Ok(path) if path.is_file()) {
        eprintln!(
            "note: the harness ended in session {id}, which is already recorded; that record was left alone"
        );
        return;
    }
    save_session_discovered(ctx, id, record);
}

fn fill_lineage(ctx: &Ctx, record: &mut SessionRecord) {
    record.trace = Some(ctx.audit.lineage().trace_id.clone());
    record.inv = Some(ctx.audit.lineage().invocation_id.clone());
}

#[derive(Clone, Copy)]
enum LockPolicy {
    Admission,
    Discovery,
}

enum SessionLockResult {
    Acquired(Option<session::SessionLock>),
    Contended { message: String },
    Io(anyhow::Error),
}

fn acquire_session_lock(ctx: &Ctx, id: &str, _policy: LockPolicy) -> SessionLockResult {
    if ctx.owns_session(id) {
        return SessionLockResult::Acquired(None);
    }
    let dir = session::sessions_dir_in(ctx.env);
    let lock = match session::try_lock_in(&dir, id) {
        Ok(lock) => lock,
        Err(error) => return SessionLockResult::Io(error),
    };
    if lock.is_some() {
        return SessionLockResult::Acquired(lock);
    }
    let message = match session::load_from(&dir, id) {
        Ok(record) => contention_message(ctx, id, &record),
        Err(_) => format!("session {id} is in use by another ccc; wait for it to finish"),
    };
    SessionLockResult::Contended { message }
}

fn contention_message(ctx: &Ctx, id: &str, record: &SessionRecord) -> String {
    let (Some(trace), Some(inv)) = (record.trace.as_deref(), record.inv.as_deref()) else {
        return format!("session {id} is in use by another ccc; wait for it to finish");
    };
    let Ok(records) = audit::read_trace(&Recorder::dir_in(ctx.env), trace) else {
        return format!("session {id} is in use by another ccc; wait for it to finish");
    };
    let Some(start) = records.iter().find_map(|r| match r {
        audit::Record::Start(s) if s.inv == inv => Some(s.payload.pid),
        _ => None,
    }) else {
        return format!("session {id} is in use by another ccc; wait for it to finish");
    };
    format!(
        "session {id} is in use by another ccc (invocation {inv}, ccc pid {start}); ccc kill {inv} first, or wait"
    )
}

fn admit_resume(ctx: &Ctx, id: &str, record: &SessionRecord) -> Result<()> {
    if ctx.owns_session(id) {
        return Ok(());
    }
    let lock = match acquire_session_lock(ctx, id, LockPolicy::Admission) {
        SessionLockResult::Acquired(Some(lock)) => lock,
        SessionLockResult::Acquired(None) => anyhow::bail!("session {id} could not be locked"),
        SessionLockResult::Contended { message } => anyhow::bail!("{message}"),
        SessionLockResult::Io(error) => return Err(error),
    };
    ctx.retain_session_lock(id, lock);
    if let Some((inv, proc)) = recorded_owner(ctx, record)? {
        check_orphan_liveness(id, &inv, &proc, |identity| identity.alive(), diag)?;
    }
    Ok(())
}

/// The harness process last recorded under `record`'s owning invocation:
/// `None` when the record has no lineage, its trace file is absent, or
/// that invocation never recorded a child.
fn recorded_owner(
    ctx: &Ctx,
    record: &SessionRecord,
) -> Result<Option<(String, crate::procs::ProcId)>> {
    let (Some(trace), Some(inv)) = (record.trace.as_deref(), record.inv.as_deref()) else {
        return Ok(None);
    };
    Uuid::parse_str(trace).context("invalid session trace backlink")?;
    Uuid::parse_str(inv).context("invalid session invocation backlink")?;
    let dir = Recorder::dir_in(ctx.env);
    let path = dir.join(format!("{trace}.jsonl"));
    if !path.is_file() {
        return Ok(None);
    }
    let records = audit::read_trace(&dir, trace)?;
    Ok(records.iter().rev().find_map(|r| match r {
        audit::Record::Child(c) if c.inv == inv => Some((inv.to_string(), c.payload.proc.clone())),
        _ => None,
    }))
}

fn check_orphan_liveness(
    id: &str,
    inv: &str,
    proc: &crate::procs::ProcId,
    probe: impl Fn(&crate::procs::ProcId) -> crate::procs::Liveness,
    mut report: impl FnMut(String),
) -> Result<()> {
    match probe(proc) {
        crate::procs::Liveness::Alive => bail!(
            "session {id} is still running under invocation {inv} (harness pid {}, its ccc is gone); ccc kill {inv} first",
            proc.pid
        ),
        crate::procs::Liveness::Unknown(reason) => report(format_args!(
            "warning: could not verify orphan liveness for session {id}, invocation {inv}: {reason}"
        ).to_string()),
        crate::procs::Liveness::Dead => {}
    }
    Ok(())
}

/// What a resume needs to reproduce this invocation.
pub fn record_of(
    r: &Resolved,
    cwd: &std::path::Path,
    account: Option<&Account>,
    now: Timestamp,
) -> SessionRecord {
    SessionRecord {
        harness: r.harness,
        agent: r.agent.clone(),
        model: r.model.clone(),
        effort: r.effort,
        profile: account.map(|a| a.profile.clone()),
        auth: account.map(|a| a.auth),
        cwd: cwd.to_path_buf(),
        sandbox: r.sandbox,
        tools: r.tools.clone(),
        created_at: now,
        last_used_at: now,
        trace: None,
        inv: None,
        forked_from: None,
        goose_policy: r.goose.clone(),
        opened_at_terminal: false,
    }
}

/// The recorded run, with the agent's description and body re-read from its
/// definition, or left empty for a roleless session. Everything the harness is
/// pinned by comes from the record.
pub fn resolved_from(record: &SessionRecord, def: Option<&AgentDef>) -> Resolved {
    Resolved {
        agent: record.agent.clone(),
        description: def.map(|d| d.description.clone()).unwrap_or_default(),
        harness: record.harness,
        model: record.model.clone(),
        effort: record.effort,
        tools: record.tools.clone(),
        sandbox: record.sandbox,
        body: def
            .map(|d| d.body_for(record.harness).to_string())
            .unwrap_or_default(),
        goose: record.goose_policy.clone(),
    }
}

/// The account and selection notes for a turn of any harness: Claude turns
/// get the account `pick` names, or none — the ambient login — when no
/// profile is requested and none is configured; Codex and Goose turns get
/// none and ignore `pick`.
fn turn_account(
    ctx: &Ctx,
    r: &Resolved,
    pick: AccountPick,
) -> Result<(Option<Account>, Vec<String>)> {
    match r.harness {
        Harness::Claude => {
            let s = match pick {
                AccountPick::Requested(None) if ctx.cfg.claude.ambient() => {
                    return Ok((None, Vec::new()))
                }
                AccountPick::Requested(profile) => fresh_account(ctx, r, profile)?,
                AccountPick::Chosen(s) => s,
            };
            Ok((Some(s.account), s.notes))
        }
        Harness::Codex | Harness::Goose => Ok((None, Vec::new())),
    }
}

/// The account for a Claude turn, new or resumed: the API key or profile
/// named by `--profile`, or the usage-driven choice under the configured auth.
fn fresh_account(ctx: &Ctx, r: &Resolved, profile: Option<&str>) -> Result<Selected> {
    if let Some(profile) = profile {
        let account = Account::for_key(&ctx.cfg.claude, profile, ctx.env)
            .unwrap_or_else(|| Account::from_config(&ctx.cfg.claude, profile, ctx.env));
        if let Err(missing) = account.require_credential() {
            bail!(missing);
        }
        return Ok(Selected {
            account,
            notes: Vec::new(),
        });
    }
    select_account(ctx, r, &(ctx.usage)())
}

fn select_account(ctx: &Ctx, r: &Resolved, source: &UsageSource) -> Result<Selected> {
    let choice = crate::select::choose(
        &ctx.cfg.claude,
        &ctx.cfg.tuning,
        ctx.env,
        &r.model,
        ctx.now,
        source,
        &[],
        ctx.keys(),
        ctx.accounts,
    )?;
    Ok(selected(choice))
}

/// The account `choice` picked, with the choice reported on stderr and noted
/// for the spawn record.
fn selected(choice: crate::select::Choice) -> Selected {
    choice.report();
    let mut notes = choice.warnings.clone();
    if let Some(reason) = &choice.degraded {
        notes.push(format!("falling back to configured order: {reason}"));
    }
    notes.extend(choice.tier_note());
    notes.extend(choice.key_notes.iter().cloned());
    Selected {
        account: choice.account,
        notes,
    }
}

/// The prompt: the positional argument, a file, or stdin. A positional `-`,
/// `--prompt-file -` and an otherwise absent prompt all mean stdin.
#[cfg(test)]
pub fn prompt_text(positional: Option<&str>, prompt_file: Option<&str>) -> Result<String> {
    prompt_text_from(positional, prompt_file, &mut std::io::stdin())
}

/// [`prompt_text`] reading its stdin arm from an explicit stream.
pub fn prompt_text_from(
    positional: Option<&str>,
    prompt_file: Option<&str>,
    stdin: &mut dyn Read,
) -> Result<String> {
    let prompt = match prompt_spec(positional, prompt_file) {
        PromptSpec::Conflict => bail!(
            "the prompt was given both positionally and with --prompt-file; use one: a quoted \
             PROMPT argument, --prompt-file PATH, or \"-\" (either place) to read stdin"
        ),
        PromptSpec::Inline(p) => p.to_string(),
        PromptSpec::Stdin => {
            let mut buf = String::new();
            stdin
                .read_to_string(&mut buf)
                .context("reading the prompt from stdin")?;
            buf
        }
        PromptSpec::File(path) => {
            std::fs::read_to_string(path).with_context(|| format!("reading prompt file {path}"))?
        }
    };
    if prompt.trim().is_empty() {
        bail!("the prompt is empty");
    }
    Ok(prompt)
}

/// The directory the subagent runs in, canonicalized so a recorded session
/// resumes where it started and symlinks resolve before use.
pub fn resolve_cwd(env: &dyn EnvSource, cwd: Option<&str>) -> Result<PathBuf> {
    let path = match cwd {
        Some(c) => crate::config::expand_tilde_in(env, c),
        None => std::env::current_dir().context("reading the current directory")?,
    };
    let canonical = std::fs::canonicalize(&path)
        .with_context(|| format!("working directory {}", path.display()))?;
    if !canonical.is_dir() {
        bail!("working directory {} is not a directory", path.display());
    }
    Ok(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::Sandbox;
    use crate::audit;
    use crate::config::{ClaudeAuth, Effort, PermissionMode};
    use crate::harness::modes::run_mode;
    use crate::procs::ProcId;
    #[cfg(target_os = "linux")]
    use crate::testutil::BoundedChild;
    use crate::testutil::TempDir;
    use crate::usage::{ProfileUsage, UsageReport, UsageWindow};
    use claude_usage_mqtt_core::{AuthState, FIVE_HOUR};
    use jiff::SignedDuration;
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    static PROCESS_SIGNAL_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn resolved() -> Resolved {
        Resolved {
            agent: Some("explorer".into()),
            description: "Investigates".into(),
            harness: Harness::Claude,
            model: "claude-haiku-4-5[1m]".into(),
            effort: Some(Effort::Low),
            tools: Some(vec!["Read".into()]),
            sandbox: Sandbox::ReadOnly,
            body: "the system prompt".into(),
            goose: None,
        }
    }

    fn account() -> Account {
        let cfg = crate::config::ClaudeConfig {
            auth: ClaudeAuth::Token,
            token_dir: "/t".into(),
            profiles_dir: "/p".into(),
            ..crate::config::ClaudeConfig::default()
        };
        Account::from_config(&cfg, "team03", &std::collections::BTreeMap::new())
    }

    fn at(now: Timestamp, hours: i64) -> Option<Timestamp> {
        now.checked_add(SignedDuration::from_hours(hours)).ok()
    }

    /// One profile's snapshot: a 5-hour window and a general weekly window.
    fn snapshot(now: Timestamp, profile: &str, util5: f64, util_g: f64) -> ProfileUsage {
        ProfileUsage {
            profile: profile.to_string(),
            auth: AuthState::Ok,
            stale_reason: None,
            updated_at: now,
            windows: vec![
                UsageWindow {
                    key: FIVE_HOUR.to_string(),
                    utilization: util5,
                    resets_at: at(now, 3),
                },
                UsageWindow {
                    key: "seven_day".to_string(),
                    utilization: util_g,
                    resets_at: at(now, 100),
                },
            ],
        }
    }

    fn report(profiles: Vec<ProfileUsage>) -> UsageSource {
        UsageSource::Report(UsageReport {
            profiles,
            daemon_status: Some("online".to_string()),
        })
    }

    /// A scratch installation: an agents directory holding one definition, a
    /// state home for sidecars, a token directory and a working directory.
    struct Fixture {
        dir: TempDir,
        cfg: Config,
        env: BTreeMap<String, String>,
        now: Timestamp,
        audit: Recorder,
    }

    impl Fixture {
        fn new(label: &str) -> Fixture {
            let dir = TempDir::new(label);
            for sub in ["agents", "work", "tokens", "state"] {
                std::fs::create_dir_all(dir.path().join(sub)).unwrap();
            }
            std::fs::write(
                dir.path().join("agents/explorer.md"),
                "---\nname: explorer\ndescription: Investigates\nmodel: claude-haiku-4-5\neffort: low\ntools: Read, Grep\n---\nthe system prompt\n",
            )
            .unwrap();

            let mut cfg = Config {
                agents_dir: Some(dir.path().join("agents").display().to_string()),
                ..Config::default()
            };
            cfg.claude.auth = ClaudeAuth::Token;
            cfg.claude.token_dir = dir.path().join("tokens").display().to_string();
            cfg.claude.profiles_dir = dir.path().join("profiles").display().to_string();
            cfg.claude.profiles = vec!["team03".into(), "team04".into()];

            let env: BTreeMap<String, String> = [
                ("HOME".to_string(), dir.path().display().to_string()),
                (
                    "XDG_STATE_HOME".to_string(),
                    dir.path().join("state").display().to_string(),
                ),
            ]
            .into_iter()
            .collect();

            Fixture {
                dir,
                cfg,
                env: env.clone(),
                now: Timestamp::now(),
                audit: Recorder::new(&env).unwrap(),
            }
        }

        fn path(&self, rel: &str) -> PathBuf {
            self.dir.path().join(rel)
        }

        fn work(&self) -> String {
            self.path("work").display().to_string()
        }

        /// A token file for `profile`, mode 0600 as the driver requires.
        fn token(&self, profile: &str) {
            let path = self.path("tokens").join(format!("{profile}.token"));
            std::fs::write(&path, "sk-ant-oat-x\n").unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
            }
        }

        /// A `.credentials.json` for `profile`, as `auth = config-dir` wants.
        fn config_dir_credential(&self, profile: &str) {
            let dir = self.path("profiles").join(profile);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(".credentials.json"), "{}").unwrap();
        }

        /// Installs a stand-in for the `claude` binary that records how it was
        /// called and prints `body` as its `--output-format json` object.
        fn claude_script(&mut self, body: &str) {
            let path = self.path("fake-claude");
            let out = self.path("calls");
            std::fs::create_dir_all(&out).unwrap();
            let script = format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > {out}/argv\npwd > {out}/cwd\nenv > {out}/env\ncat > {out}/prompt\nprintf '%s' '{body}'\n",
                out = out.display()
            );
            self.write_executable(&path, &script);
            self.env
                .insert("CCC_CLAUDE_BIN".to_string(), path.display().to_string());
        }

        fn codex_script(&mut self, body: &str) {
            let path = self.path("fake-codex");
            self.write_executable(&path, &format!("#!/bin/sh\ncat >/dev/null\n{body}\n"));
            self.env
                .insert("CCC_CODEX_BIN".to_string(), path.display().to_string());
        }

        fn claude_custom_script(&mut self, body: &str) {
            let path = self.path("fake-claude");
            std::fs::create_dir_all(self.path("calls")).unwrap();
            self.write_executable(&path, &format!("#!/bin/sh\n{body}\n"));
            self.env
                .insert("CCC_CLAUDE_BIN".to_string(), path.display().to_string());
        }

        fn write_executable(&self, path: &Path, contents: &str) {
            let temporary = self.path(&format!(".fixture-{}", Uuid::new_v4()));
            std::fs::write(&temporary, contents).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&temporary, std::fs::Permissions::from_mode(0o755))
                    .unwrap();
            }
            std::fs::rename(temporary, path).unwrap();
        }

        fn launch_script(&mut self, body: &str, codex: bool) {
            let path = self.path(if codex { "fake-codex" } else { "fake-claude" });
            let script = format!("#!/bin/sh\n{body}\n");
            self.write_executable(&path, &script);
            self.env.insert(
                if codex {
                    "CCC_CODEX_BIN"
                } else {
                    "CCC_CLAUDE_BIN"
                }
                .to_string(),
                path.display().to_string(),
            );
        }

        fn call(&self, name: &str) -> String {
            std::fs::read_to_string(self.path("calls").join(name)).unwrap()
        }

        fn install_cli_config(&mut self) {
            let config = format!(
                "[process]\nkill_grace_s = 0.2\n[claude]\nauth = \"token\"\nprofiles = [\"team03\"]\ntoken_dir = \"{}\"\nprofiles_dir = \"{}\"\n",
                self.path("tokens").display(),
                self.path("profiles").display(),
            );
            let path = self.path("config.toml");
            std::fs::write(&path, config).unwrap();
            self.env
                .insert("CCC_CONFIG".into(), path.display().to_string());
            let agents = format!("agents_dir = \"{}\"\n", self.path("agents").display());
            let path = self.path("agents.toml");
            std::fs::write(&path, agents).unwrap();
            self.env
                .insert("CCC_AGENTS_CONFIG".into(), path.display().to_string());
        }

        /// `CCC_CONFIG=<path> CCC_AGENTS_CONFIG=<path>`, shell-quoted, for a
        /// hand-built child environment.
        #[cfg(target_os = "linux")]
        fn config_env_literals(&self) -> String {
            format!(
                "CCC_CONFIG={} CCC_AGENTS_CONFIG={}",
                shell_literal(&self.env["CCC_CONFIG"]),
                shell_literal(&self.env["CCC_AGENTS_CONFIG"]),
            )
        }

        #[cfg(target_os = "linux")]
        fn spawn_cli_with_timeout(&self, label: &str, args: &[&str], seconds: u64) -> SpawnedCli {
            let stdout = self.path(&format!(".cli-{label}-{}.stdout", Uuid::new_v4()));
            let stderr = self.path(&format!(".cli-{label}-{}.stderr", Uuid::new_v4()));
            let stdout_file = std::fs::File::create(&stdout).unwrap();
            let stderr_file = std::fs::File::create(&stderr).unwrap();
            let path = std::env::var_os("PATH").unwrap_or_default();
            let mut command = std::process::Command::new("timeout");
            command
                .args(["-k", "1"])
                .arg(seconds.to_string())
                .arg(crate::testutil::ccc_binary())
                .args(args)
                .env_clear()
                .env("PATH", path)
                .envs(&self.env)
                .stdout(std::process::Stdio::from(stdout_file))
                .stderr(std::process::Stdio::from(stderr_file));
            let child = BoundedChild::spawn(&mut command).unwrap();
            SpawnedCli {
                child,
                stdout,
                stderr,
            }
        }

        #[cfg(target_os = "linux")]
        fn spawn_cli(&self, label: &str, args: &[&str]) -> SpawnedCli {
            self.spawn_cli_with_timeout(label, args, 9)
        }

        #[cfg(target_os = "linux")]
        fn spawn_cli_by(&self, label: &str, args: &[&str], deadline: u64) -> SpawnedCli {
            let remaining = deadline.saturating_sub(boot_seconds());
            assert!(
                remaining >= 1,
                "deadline for {label} has less than one second remaining"
            );
            self.spawn_cli_with_timeout(label, args, remaining)
        }

        #[cfg(target_os = "linux")]
        fn cli(&self, args: &[&str]) -> std::process::Output {
            self.spawn_cli("sync", args).finish(Duration::from_secs(10))
        }

        fn sessions_dir(&self) -> PathBuf {
            session::sessions_dir_in(&self.env)
        }

        fn ctx<'a>(&'a self, usage: &'a (dyn Fn() -> UsageSource + Sync)) -> Ctx<'a> {
            self.ctx_with_accounts(usage, &no_accounts)
        }

        fn ctx_with_accounts<'a>(
            &'a self,
            usage: &'a (dyn Fn() -> UsageSource + Sync),
            accounts: &'a (dyn Fn() -> Accounts + Sync),
        ) -> Ctx<'a> {
            Ctx {
                cfg: &self.cfg,
                env: &self.env,
                now: self.now,
                usage,
                accounts,
                api_keys_eligible: true,
                bus: &crate::bus::NoBus,
                live: &OFF,
                audit: &self.audit,
                session_locks: Arc::new(Mutex::new(Vec::new())),
            }
        }

        /// A context that publishes to `bus` as `live`.
        fn ctx_with_bus<'a>(
            &'a self,
            usage: &'a (dyn Fn() -> UsageSource + Sync),
            accounts: &'a (dyn Fn() -> Accounts + Sync),
            bus: &'a dyn Bus,
            live: &'a Live,
        ) -> Ctx<'a> {
            Ctx {
                bus,
                live,
                ..self.ctx_with_accounts(usage, accounts)
            }
        }
    }

    static OFF: Live = Live::off();

    #[cfg(target_os = "linux")]
    struct SpawnedCli {
        child: BoundedChild,
        stdout: PathBuf,
        stderr: PathBuf,
    }

    #[cfg(target_os = "linux")]
    impl SpawnedCli {
        fn finish_at(mut self, deadline: Instant) -> std::process::Output {
            let status = self.child.finish_by(deadline).unwrap_or_else(|error| {
                let stdout = std::fs::read_to_string(&self.stdout).unwrap_or_default();
                let stderr = std::fs::read_to_string(&self.stderr).unwrap_or_default();
                panic!("ccc fixture did not finish: {error}; stdout={stdout}; stderr={stderr}");
            });
            let stdout = std::fs::read(&self.stdout).unwrap();
            let stderr = std::fs::read(&self.stderr).unwrap();
            std::process::Output {
                status,
                stdout,
                stderr,
            }
        }

        fn finish(self, limit: Duration) -> std::process::Output {
            self.finish_at(Instant::now() + limit)
        }

        fn finish_by_boot(mut self, deadline: u64) -> std::process::Output {
            let status = self
                .child
                .finish_by_boot(deadline as f64, boot_seconds_unfloored)
                .unwrap_or_else(|error| {
                    panic!("ccc fixture did not finish before boot deadline {deadline}: {error}")
                });
            let stdout = std::fs::read(&self.stdout).unwrap();
            let stderr = std::fs::read(&self.stderr).unwrap();
            std::process::Output {
                status,
                stdout,
                stderr,
            }
        }
    }

    const OK_RESULT: &str = r#"{"type":"result","session_id":"9f22dd1b-aaaa","result":"the answer","is_error":false,"num_turns":1,"total_cost_usd":0.01,"model_usage_since_last_result":{"claude-sonnet-5-5":{"costUSD":0.01}}}"#;

    const FORK_RESULT: &str = r#"{"type":"result","session_id":"fork-1111","result":"forked answer","is_error":false,"num_turns":1,"total_cost_usd":0.01,"model_usage_since_last_result":{"claude-sonnet-5-5":{"costUSD":0.01}}}"#;

    /// Writes a sidecar for `id` recorded an hour ago and returns its text, so
    /// a test can check the parent of a fork is left byte-identical.
    fn seed_parent(f: &Fixture, id: &str, harness: Harness) -> String {
        let mut r = resolved();
        r.harness = harness;
        let account = match harness {
            Harness::Claude => Some(account()),
            Harness::Codex => {
                r.model = "gpt-5.6-luna".into();
                None
            }
            Harness::Goose => None,
        };
        let record = record_of(
            &r,
            Path::new(&f.work()),
            account.as_ref(),
            at(f.now, -1).unwrap(),
        );
        session::save_in(&f.sessions_dir(), id, &record).unwrap();
        sidecar_text(f, id)
    }

    fn sidecar_text(f: &Fixture, id: &str) -> String {
        std::fs::read_to_string(session::path_in(&f.sessions_dir(), id).unwrap()).unwrap()
    }

    /// The `.json` sidecars in the sessions directory, by file name.
    fn sidecar_names(f: &Fixture) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(f.sessions_dir())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".json"))
            .collect();
        names.sort();
        names
    }

    /// Whether this process can take `id`'s lock now.
    fn lock_is_free(f: &Fixture, id: &str) -> bool {
        session::try_lock_in(&f.sessions_dir(), id)
            .unwrap()
            .is_some()
    }

    #[cfg(target_os = "linux")]
    fn boot_seconds() -> u64 {
        boot_seconds_unfloored().floor() as u64
    }

    #[cfg(target_os = "linux")]
    fn boot_seconds_unfloored() -> f64 {
        std::fs::read_to_string("/proc/uptime")
            .expect("reading /proc/uptime")
            .split_whitespace()
            .next()
            .expect("/proc/uptime has no boot time")
            .parse()
            .expect("parsing /proc/uptime boot time")
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn boot_seconds_tracks_boottime() {
        // `/proc/uptime` may have its own origin (lxcfs reports the
        // container's uptime), so this checks the rate, not the origin.
        fn offset() -> f64 {
            let mut timestamp = libc::timespec {
                tv_sec: 0,
                tv_nsec: 0,
            };
            let result = unsafe { libc::clock_gettime(libc::CLOCK_BOOTTIME, &mut timestamp) };
            assert_eq!(result, 0);
            let kernel = timestamp.tv_sec as f64 + timestamp.tv_nsec as f64 / 1e9;
            kernel - boot_seconds_unfloored()
        }
        let offset_before = offset();
        let first = boot_seconds();
        std::thread::sleep(Duration::from_millis(1100));
        let second = boot_seconds();
        let offset_after = offset();
        assert!((first + 1..=first + 2).contains(&second));
        assert!(
            (offset_after - offset_before).abs() < 0.5,
            "/proc/uptime drifted from CLOCK_BOOTTIME: {offset_before} then {offset_after}"
        );
    }

    #[cfg(target_os = "linux")]
    fn shell_literal(value: &str) -> String {
        format!("'{}'", value.replace('\'', "'\\''"))
    }

    fn run_captured_test(
        inv: &Invocation,
        output_file: Option<&Path>,
    ) -> Result<(crate::harness::Captured, Vec<String>)> {
        let supervised = run_supervised_test(inv)?;
        let mut captured = supervised.captured;
        captured.output_file = output_file.and_then(codex::take_output_file);
        Ok((captured, supervised.warnings))
    }

    fn run_supervised_test(inv: &Invocation) -> Result<supervise::Supervised> {
        let _lock = supervise::TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let on_spawned = |_: crate::procs::ProcId, _: u32| {};
        let mut on_line = |_line: &str| {};
        supervise::run(
            inv,
            &SuperviseOpts {
                detach_group: true,
                grace: std::time::Duration::from_millis(200),
                capture: true,
                on_spawned: &on_spawned,
                on_signal: None,
            },
            &mut on_line,
        )
    }

    fn run_launch_test(inv: &Invocation) -> Result<std::process::ExitStatus> {
        let _lock = supervise::TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let on_spawned = |_: crate::procs::ProcId, _: u32| {};
        let mut on_line = |_line: &str| {};
        Ok(supervise::run(
            inv,
            &SuperviseOpts {
                detach_group: false,
                grace: std::time::Duration::from_millis(200),
                capture: false,
                on_spawned: &on_spawned,
                on_signal: None,
            },
            &mut on_line,
        )?
        .status)
    }

    #[test]
    fn the_prompt_comes_from_the_positional_or_a_file() {
        let dir = TempDir::new("prompt");
        let path = dir.path().join("p.txt");
        std::fs::write(&path, "from the file\n").unwrap();

        assert_eq!(prompt_text(Some("inline"), None).unwrap(), "inline");
        assert_eq!(
            prompt_text(None, Some(path.to_str().unwrap())).unwrap(),
            "from the file\n"
        );

        assert_eq!(
            prompt_spec(Some("inline"), Some("p.txt")),
            PromptSpec::Conflict
        );
        let err = prompt_text(Some("inline"), Some("p.txt"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--prompt-file"), "{err}");

        let err = prompt_text(Some("   "), None).unwrap_err().to_string();
        assert!(err.contains("empty"), "{err}");

        let err = format!(
            "{:#}",
            prompt_text(None, Some("/nonexistent/prompt")).unwrap_err()
        );
        assert!(err.contains("reading prompt file"), "{err}");
    }

    #[test]
    fn prompt_sources_and_failures_are_classified_without_consuming_harness_input() {
        assert_eq!(prompt_source(Some("arg"), Some("file")), None);
        assert_eq!(
            prompt_source(Some("inline"), None),
            Some(crate::audit::PromptSource::Arg)
        );
        assert_eq!(
            prompt_source(None, Some("prompt.txt")),
            Some(crate::audit::PromptSource::File)
        );
        assert_eq!(
            prompt_source(None, Some("-")),
            Some(crate::audit::PromptSource::Stdin)
        );
        assert_eq!(
            prompt_source(None, None),
            Some(crate::audit::PromptSource::Stdin)
        );
        assert!(prompt_text_from(Some("a"), Some("file"), &mut &b""[..]).is_err());
        assert!(prompt_text_from(Some("  "), None, &mut &b""[..]).is_err());
        assert!(prompt_text_from(None, Some("-"), &mut &b" \n"[..]).is_err());
    }

    /// Panics if the prompt reader touches stdin.
    struct UnreadableStdin;

    impl Read for UnreadableStdin {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            panic!("stdin must not be read");
        }
    }

    #[test]
    fn prompt_source_and_prompt_text_agree_on_every_argument_shape() {
        let dir = TempDir::new("prompt-shapes");
        let path = dir.path().join("p.txt");
        std::fs::write(&path, "from the file\n").unwrap();
        let path = path.to_str().unwrap();
        use crate::audit::PromptSource::{Arg, File, Stdin};
        for (positional, file, source, expected) in [
            (None, None, Some(Stdin), Some("piped\n")),
            (None, Some("-"), Some(Stdin), Some("piped\n")),
            (None, Some(path), Some(File), Some("from the file\n")),
            (Some("text"), None, Some(Arg), Some("text")),
            (Some("-"), None, Some(Stdin), Some("piped\n")),
            (Some("-"), Some("-"), Some(Stdin), Some("piped\n")),
            (Some("-"), Some(path), None, None),
            (Some("text"), Some("-"), None, None),
            (Some("text"), Some(path), None, None),
        ] {
            let case = format!("{positional:?} {file:?}");
            assert_eq!(prompt_source(positional, file), source, "{case}");
            match expected {
                Some(text) => {
                    let mut stdin = "piped\n".as_bytes();
                    assert_eq!(
                        prompt_text_from(positional, file, &mut stdin).unwrap(),
                        text,
                        "{case}"
                    );
                    if source == Some(Stdin) {
                        assert!(stdin.is_empty(), "{case}: stdin read once, fully");
                    }
                }
                None => {
                    assert!(
                        prompt_text_from(positional, file, &mut UnreadableStdin).is_err(),
                        "{case}"
                    );
                }
            }
        }
    }

    #[test]
    fn degraded_selection_notes_match_the_reported_fallback() {
        let f = Fixture::new("selection-notes");
        f.token("team03");
        let usage = || UsageSource::Unavailable("offline".into());
        let ctx = f.ctx(&usage);
        let selected = select_account(&ctx, &resolved(), &(ctx.usage)()).unwrap();
        assert_eq!(
            selected.notes,
            vec![
                "profile team04 has no credential at ".to_string()
                    + &f.path("tokens/team04.token").display().to_string()
                    + "; not a candidate",
                "falling back to configured order: usage data unavailable: offline".to_string()
            ]
        );
        let explicit = fresh_account(&ctx, &resolved(), Some("team03")).unwrap();
        assert!(explicit.notes.is_empty());
    }

    #[test]
    fn a_tier_two_selection_is_noted_once_after_the_warnings() {
        let f = Fixture::new("selection-tier-note");
        f.token("team03");
        let usage = || report(vec![snapshot(f.now, "team03", 92.0, 10.0)]);
        let ctx = f.ctx(&usage);
        let selected = select_account(&ctx, &resolved(), &(ctx.usage)()).unwrap();
        assert_eq!(selected.account.profile, "team03");
        assert_eq!(
            selected.notes,
            vec![
                "profile team04 has no credential at ".to_string()
                    + &f.path("tokens/team04.token").display().to_string()
                    + "; not a candidate",
                "team03 chosen from tier 2: in the 5-hour ramp".to_string()
            ]
        );
    }

    #[test]
    fn drive_records_spawn_and_injects_both_lineage_variables() {
        let mut f = Fixture::new("lineage-spawn");
        f.token("team03");
        f.claude_script(OK_RESULT);
        let usage = || UsageSource::Unavailable("not consulted".into());
        let ctx = f.ctx(&usage);
        let common = Common {
            cwd: f.path("work"),
            extra_args: vec![],
            mode: run_mode(),
            preset_session_id: Some("preset".into()),
        };
        let account = Account::from_config(&f.cfg.claude, "team03", &f.env);
        let record = record_of(&resolved(), &common.cwd, Some(&account), ctx.now);
        let result = drive(&ctx, &resolved(), &common, Some(account), vec![], &record).unwrap();
        assert!(result.is_ok());
        let records = audit::read_trace(
            &Recorder::dir_in(&f.env),
            ctx.audit.lineage().trace_id.as_str(),
        )
        .unwrap();
        assert!(records
            .iter()
            .any(|record| matches!(record, audit::Record::Spawn(_))));
        let call = f.call("env");
        let trace = call
            .lines()
            .find_map(|line| line.strip_prefix("CCC_TRACE_ID="))
            .unwrap();
        let parent = call
            .lines()
            .find_map(|line| line.strip_prefix("CCC_PARENT_INVOCATION_ID="))
            .unwrap();
        assert_eq!(trace, ctx.audit.lineage().trace_id);
        assert_eq!(parent, ctx.audit.lineage().invocation_id);
    }

    #[test]
    fn preset_claude_session_contention_fails_before_spawn() {
        let _lock = supervise::TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut f = Fixture::new("session-admission-contention");
        f.token("team03");
        f.claude_custom_script("touch \"$HOME/ran\"");
        let held = session::try_lock_in(&f.sessions_dir(), "p")
            .unwrap()
            .unwrap();
        let usage = || UsageSource::Unavailable("not consulted".into());
        let ctx = f.ctx(&usage);
        std::fs::write(
            Recorder::dir_in(&f.env).join(format!("{}.jsonl", ctx.audit.lineage().trace_id)),
            "",
        )
        .unwrap();
        let common = Common {
            cwd: f.path("work"),
            extra_args: vec![],
            mode: run_mode(),
            preset_session_id: Some("p".into()),
        };
        let diagnostics = Mutex::new(Vec::new());
        let claude_account = Account::from_config(&f.cfg.claude, "team03", &f.env);
        let record = record_of(&resolved(), &common.cwd, Some(&claude_account), ctx.now);
        let result = match drive_with_meta_report(
            &ctx,
            &resolved(),
            &common,
            Some(claude_account),
            vec![],
            &|message| diagnostics.lock().unwrap().push(message),
            &record,
        ) {
            Ok(_) => panic!("contention spawned unexpectedly"),
            Err(error) => error,
        };
        drop(held);
        assert_eq!(
            result.to_string(),
            "session p is in use by another ccc; wait for it to finish"
        );
        assert!(diagnostics.lock().unwrap().is_empty());
        assert!(!f.path("ran").exists());
        let records = audit::read_trace(
            &Recorder::dir_in(&f.env),
            ctx.audit.lineage().trace_id.as_str(),
        )
        .unwrap();
        assert!(!records
            .iter()
            .any(|record| matches!(record, audit::Record::Spawn(_))));
    }

    #[test]
    fn claude_session_reservation_io_warns_and_still_spawns() {
        let _lock = supervise::TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut f = Fixture::new("session-admission-io");
        f.token("team03");
        f.claude_script(OK_RESULT);
        std::fs::create_dir_all(f.path("state/ccc")).unwrap();
        std::fs::write(f.sessions_dir(), "blocked").unwrap();
        let usage = || UsageSource::Unavailable("not consulted".into());
        let ctx = f.ctx(&usage);
        let common = Common {
            cwd: f.path("work"),
            extra_args: vec![],
            mode: run_mode(),
            preset_session_id: Some("p".into()),
        };
        let diagnostics = Mutex::new(Vec::new());
        let claude_account = Account::from_config(&f.cfg.claude, "team03", &f.env);
        let record = record_of(&resolved(), &common.cwd, Some(&claude_account), ctx.now);
        let result = drive_with_meta_report(
            &ctx,
            &resolved(),
            &common,
            Some(claude_account),
            vec![],
            &|message| diagnostics.lock().unwrap().push(message),
            &record,
        )
        .unwrap();
        assert!(result.result.is_ok());
        assert!(f.path("calls/argv").exists());
        let diagnostics = diagnostics.lock().unwrap();
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].starts_with("warning: could not reserve fresh claude session p: "));
        assert!(diagnostics[0].len() > "warning: could not reserve fresh Claude session p: ".len());
        let records = audit::read_trace(
            &Recorder::dir_in(&f.env),
            ctx.audit.lineage().trace_id.as_str(),
        )
        .unwrap();
        assert_eq!(
            records
                .iter()
                .filter(|record| matches!(record, audit::Record::Spawn(_)))
                .count(),
            1
        );
    }

    #[test]
    fn codex_session_discovery_contention_warns_without_saving_sidecar() {
        let mut f = Fixture::new("session-discovery-contention");
        f.codex_script("printf '%s\\n' '{\"type\":\"thread.started\",\"thread_id\":\"t\"}' '{\"type\":\"item.completed\",\"item\":{\"type\":\"agent_message\",\"text\":\"answer\"}}' '{\"type\":\"turn.completed\",\"usage\":{}}'");
        let held = session::try_lock_in(&f.sessions_dir(), "t")
            .unwrap()
            .unwrap();
        let usage = || UsageSource::Unavailable("not consulted".into());
        let ctx = f.ctx(&usage);
        let mut resolved = resolved();
        resolved.harness = Harness::Codex;
        resolved.model = "gpt-5.6-luna".into();
        let common = Common {
            cwd: f.path("work"),
            extra_args: vec![],
            mode: run_mode(),
            preset_session_id: None,
        };
        let diagnostics = Mutex::new(Vec::new());
        let record = record_of(&resolved, &common.cwd, None, ctx.now);
        let result = drive_with_meta_report(
            &ctx,
            &resolved,
            &common,
            None,
            vec![],
            &|message| diagnostics.lock().unwrap().push(message),
            &record,
        )
        .unwrap();
        drop(held);
        let outcome = result.result.unwrap();
        assert_eq!(&outcome.session_id, "t");
        let diagnostics = diagnostics.lock().unwrap();
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].starts_with("warning: session discovery failed for t:"));
        assert!(diagnostics[0].contains("in use by another ccc"));
        assert!(!session::path_in(&f.sessions_dir(), "t").unwrap().exists());
    }

    #[test]
    fn an_audit_spawn_failure_prevents_the_harness_and_reaps_codex_output() {
        let mut f = Fixture::new("audit-spawn-failure");
        f.env.insert(
            "CCC_CODEX_BIN".into(),
            f.path("should-not-run").display().to_string(),
        );
        let usage = || UsageSource::Unavailable("not consulted".into());
        let ctx = f.ctx(&usage);
        let mut r = resolved();
        r.harness = Harness::Codex;
        let common = Common {
            cwd: f.path("work"),
            extra_args: vec![],
            mode: run_mode(),
            preset_session_id: None,
        };

        let audit_dir = Recorder::dir_in(&f.env);
        std::fs::remove_dir_all(&audit_dir).unwrap();
        std::fs::write(&audit_dir, "blocked").unwrap();
        let record = record_of(&r, &common.cwd, None, ctx.now);
        let err = drive(&ctx, &r, &common, None, vec![], &record).unwrap_err();
        assert!(err.to_string().contains("audit"), "{err:#}");
        assert!(!f.path("should-not-run").exists());
        let tmp = crate::config::state_home_in(&f.env).join("ccc/tmp");
        assert!(std::fs::read_dir(tmp).unwrap().next().is_none());
    }

    #[test]
    fn launch_spawn_payload_keeps_the_claude_preset_session() {
        let payload = spawn_payload(
            &resolved(),
            Path::new("/srv/work"),
            Some(&account()),
            Some("preset-session".into()),
            vec![],
        );
        assert_eq!(payload.session.as_deref(), Some("preset-session"));
        assert_eq!(payload.profile.as_deref(), Some("team03"));
    }

    /// Configures `[claude.api_keys.<name>]` with a 0600 key file holding
    /// `sk-ant-api-<name>`.
    fn api_key(f: &mut Fixture, name: &str) {
        let path = f.path("tokens").join(format!("{name}.key"));
        std::fs::write(&path, format!("sk-ant-api-{name}\n")).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        f.cfg.claude.api_keys.insert(
            name.into(),
            crate::config::ApiKeyConfig::new(&path.display().to_string(), 420.0, &["*"]),
        );
    }

    #[cfg(unix)]
    #[test]
    fn profile_naming_a_key_runs_on_the_key_without_selection() {
        for ambient in [false, true] {
            let mut f = Fixture::new("run-api-key");
            if ambient {
                f.cfg.claude.profiles.clear();
            }
            api_key(&mut f, "team");
            f.env
                .insert("ANTHROPIC_API_KEY".into(), "sk-ant-inherited".into());
            f.env.insert(
                "CLAUDE_CONFIG_DIR".into(),
                f.path("elsewhere").display().to_string(),
            );
            f.claude_script(OK_RESULT);
            start_run(&f, Some(PromptSource::Arg), None, Some("hi"));
            let usage = || -> UsageSource { panic!("selection consulted for --profile <key>") };
            let mut out = Vec::new();
            let completion = run_with(
                &f.ctx(&usage),
                "explorer",
                Some(&f.work()),
                "hi".into(),
                Some("team"),
                None,
                false,
                &mut out,
            )
            .unwrap();
            assert_eq!(completion, 0, "ambient={ambient}");
            record_completion(&f, completion);

            let env = f.call("env");
            assert!(
                env.lines()
                    .any(|l| l == "ANTHROPIC_API_KEY=sk-ant-api-team"),
                "{env}"
            );
            assert!(!env.contains("sk-ant-inherited"), "{env}");
            assert!(
                !env.lines().any(|l| l.starts_with("CLAUDE_CONFIG_DIR=")),
                "{env}"
            );
            assert!(
                !env.lines()
                    .any(|l| l.starts_with("CLAUDE_CODE_OAUTH_TOKEN=")),
                "{env}"
            );

            let spawn = audit_records(&f)
                .into_iter()
                .find_map(|r| match r {
                    audit::Record::Spawn(spawn) => Some(spawn),
                    _ => None,
                })
                .unwrap();
            assert_eq!(spawn.payload.profile.as_deref(), Some("team"));
            assert_eq!(spawn.payload.auth, Some(ClaudeAuth::ApiKey));
            let saved = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
            assert_eq!(saved.profile.as_deref(), Some("team"));
            assert_eq!(saved.auth, Some(ClaudeAuth::ApiKey));

            let trusted: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(f.path(".claude.json")).unwrap())
                    .unwrap();
            let cwd = std::fs::canonicalize(f.work()).unwrap();
            assert_eq!(
                trusted["projects"][cwd.to_str().unwrap()]["hasTrustDialogAccepted"],
                serde_json::json!(true)
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn profile_naming_a_key_without_its_key_file_is_refused() {
        let mut f = Fixture::new("run-api-key-missing");
        api_key(&mut f, "team");
        std::fs::remove_file(f.path("tokens/team.key")).unwrap();
        f.claude_script(OK_RESULT);
        let usage = || UsageSource::Unavailable("not consulted".into());
        let mut out = Vec::new();
        let err = run_with(
            &f.ctx(&usage),
            "explorer",
            Some(&f.work()),
            "hi".into(),
            Some("team"),
            None,
            false,
            &mut out,
        )
        .unwrap_err();
        let message = format!("{:#}", err.error);
        assert!(
            message.contains("API key team has no credential at"),
            "{message}"
        );
        assert!(!f.path("calls/argv").exists());
    }

    /// This host's ledger for the key `name`, if one was written.
    fn key_ledger(f: &Fixture, name: &str) -> Option<crate::budget::Ledger> {
        let text = std::fs::read_to_string(f.path("state/ccc/budget").join(format!("{name}.json")))
            .ok()?;
        Some(serde_json::from_str(&text).unwrap())
    }

    #[cfg(unix)]
    fn pinned_run(f: &Fixture, profile: &str) -> Completion {
        start_run(f, Some(PromptSource::Arg), None, Some("hi"));
        let usage = || UsageSource::Unavailable("not consulted".into());
        let mut out = Vec::new();
        run_with(
            &f.ctx(&usage),
            "explorer",
            Some(&f.work()),
            "hi".into(),
            Some(profile),
            None,
            false,
            &mut out,
        )
        .unwrap()
    }

    #[cfg(unix)]
    #[test]
    fn a_key_turn_is_charged_its_reported_cost_and_a_profile_turn_is_not() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-api-key-charge");
        api_key(&mut f, "team");
        f.token("team03");
        f.claude_script(OK_RESULT);
        assert_eq!(pinned_run(&f, "team03"), 0);
        assert!(key_ledger(&f, "team").is_none());
        assert!(!f.path("state/ccc/budget/team03.json").exists());
        assert_eq!(pinned_run(&f, "team"), 0);
        assert_eq!(pinned_run(&f, "team"), 0);
        let ledger = key_ledger(&f, "team").unwrap();
        assert_eq!(
            (ledger.usd, ledger.turns, ledger.unmeasured_turns),
            (0.02, 2, 0)
        );
        assert!(ledger.exhausted.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn a_key_turn_without_a_cost_is_counted_unmeasured() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-api-key-no-cost");
        api_key(&mut f, "team");
        f.claude_script(r#"{"type":"result","session_id":"9f22dd1b-aaaa","result":"the answer","is_error":false,"num_turns":1}"#);
        assert_eq!(pinned_run(&f, "team"), 0);
        let ledger = key_ledger(&f, "team").unwrap();
        assert_eq!(
            (ledger.usd, ledger.turns, ledger.unmeasured_turns),
            (0.0, 1, 1)
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_key_turn_with_a_running_total_but_no_stated_cost_is_counted_unmeasured() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-api-key-total-only");
        api_key(&mut f, "team");
        f.claude_script(r#"{"type":"result","session_id":"9f22dd1b-aaaa","result":"the answer","is_error":false,"num_turns":1,"total_cost_usd":0.01}"#);
        assert_eq!(pinned_run(&f, "team"), 0);
        let ledger = key_ledger(&f, "team").unwrap();
        assert_eq!(
            (ledger.usd, ledger.turns, ledger.unmeasured_turns),
            (0.0, 1, 1)
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_terminated_key_turn_is_counted_unmeasured() {
        let _signal_lock = PROCESS_SIGNAL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut f = Fixture::new("run-api-key-signal");
        api_key(&mut f, "team");
        f.cfg.process.kill_grace_s = 0.2;
        let binary = f.path("signalled-claude");
        f.write_executable(&binary, "#!/bin/sh\ntrap 'exit 1' TERM\nsleep 5 & wait\n");
        f.env
            .insert("CCC_CLAUDE_BIN".into(), binary.display().to_string());
        supervise::set_test_signal(true);
        let mut input = "prompt".as_bytes();
        let mut output = Vec::new();
        let code = run_from(
            "explorer",
            Some(&f.work()),
            Some("prompt"),
            None,
            Some("team"),
            None,
            false,
            &f.env,
            || Ok(f.cfg.clone()),
            &mut input,
            &mut output,
        )
        .unwrap();
        assert_eq!(code, crate::harness::EXIT_NO_RESULT);
        let ledger = key_ledger(&f, "team").unwrap();
        assert_eq!(
            (ledger.usd, ledger.turns, ledger.unmeasured_turns),
            (0.0, 1, 1)
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_key_turn_whose_prompt_fell_short_is_charged_what_it_reported() {
        let prompt = "p".repeat(1024 * 1024);
        let mut f = Fixture::new("run-api-key-shortfall");
        api_key(&mut f, "team");
        f.claude_custom_script(&format!(
            "head -c 1 >/dev/null; printf '%s' '{}'",
            OK_RESULT
        ));
        let mut input = "ignored".as_bytes();
        let mut output = Vec::new();
        let code = run_from_report(
            "explorer",
            Some(&f.work()),
            Some(&prompt),
            None,
            Some("team"),
            None,
            &GooseFlags::default(),
            false,
            &f.env,
            || Ok(f.cfg.clone()),
            &mut input,
            &mut output,
            &|_| {},
        )
        .unwrap();
        assert_eq!(code, crate::harness::EXIT_NO_RESULT);
        let ledger = key_ledger(&f, "team").unwrap();
        assert_eq!(
            (ledger.usd, ledger.turns, ledger.unmeasured_turns),
            (0.01, 1, 0)
        );
    }

    fn refused_turn(message: &str) -> String {
        serde_json::json!({
            "type": "result",
            "session_id": "9f22dd1b-aaaa",
            "result": message,
            "is_error": true,
            "num_turns": 1,
            "total_cost_usd": 0.0,
            "model_usage_since_last_result": {},
        })
        .to_string()
    }

    #[cfg(unix)]
    #[test]
    fn a_key_turn_refused_for_spend_raises_the_flag_and_is_limited() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-api-key-exhausted");
        api_key(&mut f, "team");
        f.claude_script(&refused_turn(
            "Your credit balance is too low to access the Anthropic API.",
        ));
        let completion = pinned_run(&f, "team");
        assert_eq!(completion, crate::harness::EXIT_LIMITED);
        assert!(
            warned(&completion, &["team hit a usage limit", PINNED]),
            "{:?}",
            completion.end.warnings
        );
        let ledger = key_ledger(&f, "team").unwrap();
        assert_eq!((ledger.turns, ledger.unmeasured_turns), (1, 0));
        let flag = ledger.exhausted.unwrap();
        assert_eq!(
            flag.session.as_deref(),
            Some(f.audit.lineage().invocation_id.as_str())
        );
        assert!(flag.message.unwrap().contains("credit balance is too low"));
    }

    fn refused_429(message: &str) -> String {
        serde_json::json!({
            "type": "result",
            "session_id": "9f22dd1b-aaaa",
            "result": message,
            "is_error": true,
            "api_error_status": 429,
            "num_turns": 1,
            "total_cost_usd": 0.0,
            "model_usage_since_last_result": {},
        })
        .to_string()
    }

    #[cfg(unix)]
    #[test]
    fn a_key_turn_limited_at_parse_raises_the_flag_only_when_a_pattern_matches() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-api-key-429-spend");
        api_key(&mut f, "team");
        f.claude_script(&refused_429(
            "Your credit balance is too low to access the Anthropic API.",
        ));
        assert_eq!(pinned_run(&f, "team"), crate::harness::EXIT_LIMITED);
        let flag = key_ledger(&f, "team").unwrap().exhausted.unwrap();
        assert!(flag.message.unwrap().contains("credit balance is too low"));

        let mut f = Fixture::new("run-api-key-429-other");
        api_key(&mut f, "team");
        f.claude_script(&refused_429("API Error: Rate limited."));
        assert_eq!(pinned_run(&f, "team"), crate::harness::EXIT_LIMITED);
        let ledger = key_ledger(&f, "team").unwrap();
        assert_eq!(ledger.turns, 1);
        assert!(ledger.exhausted.is_none());
    }

    /// A refused profile turn as stream-json: a rejected rate-limit event
    /// resetting at `resets_at` (Unix seconds), when given, then the 429
    /// result.
    fn refused_stream(resets_at: Option<i64>) -> String {
        let mut lines = Vec::new();
        if let Some(resets_at) = resets_at {
            lines.push(
                serde_json::json!({
                    "type": "rate_limit_event",
                    "rate_limit_info": {
                        "status": "rejected",
                        "resetsAt": resets_at,
                        "rateLimitType": "seven_day_overage_included",
                    },
                })
                .to_string(),
            );
        }
        lines.push(refused_429(
            "Fable limit reached. Switch to another model to continue.",
        ));
        lines.join("\n")
    }

    fn profile_limits(f: &Fixture, profile: &str) -> Option<crate::limits::LimitsFile> {
        let text =
            std::fs::read_to_string(f.path("state/ccc/limits").join(format!("{profile}.json")))
                .ok()?;
        Some(serde_json::from_str(&text).unwrap())
    }

    #[cfg(unix)]
    #[test]
    fn a_limited_profile_turn_with_a_reset_raises_writes_and_publishes_the_bucket_flag() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-profile-limited");
        f.token("team03");
        f.cfg.claude.selection.fable_models = vec!["*haiku*".into()];
        let resets_at = Timestamp::now().as_second() + 86_400;
        f.claude_script(&refused_stream(Some(resets_at)));
        let usage = || UsageSource::Unavailable("not consulted".into());
        let (completion, bus) = bus_run(&f, Some("team03"), &usage, vec![]);
        assert_eq!(completion, crate::harness::EXIT_LIMITED);

        let file = profile_limits(&f, "team03").expect("limits file written");
        assert_eq!(file.buckets.keys().collect::<Vec<_>>(), vec!["fable"]);
        let flag = &file.buckets["fable"];
        assert_eq!(flag.until.map(|t| t.as_second()), Some(resets_at));
        assert_eq!(flag.kind.as_deref(), Some("seven_day_overage_included"));
        assert_eq!(
            flag.session.as_deref(),
            Some(f.audit.lineage().invocation_id.as_str())
        );
        assert!(flag.message.as_deref().unwrap().contains("Fable limit"));

        let put = bus
            .calls()
            .into_iter()
            .filter_map(|c| match c {
                crate::bus::Recorded::PutAccount(s) => Some(*s),
                _ => None,
            })
            .find(|s| s.account == "team03")
            .expect("the profile record was published");
        assert_eq!(
            put.account_kind,
            claude_usage_mqtt_core::AccountKind::Subscription
        );
        assert_eq!((put.period.as_str(), put.usd), ("-", 0.0));
        assert_eq!(put.exhausted_buckets, file.buckets);

        // Another model draws on the general quota for the same refusal.
        let mut f = Fixture::new("run-profile-limited-general");
        f.token("team03");
        f.claude_script(&refused_stream(Some(resets_at)));
        assert_eq!(pinned_run(&f, "team03"), crate::harness::EXIT_LIMITED);
        let file = profile_limits(&f, "team03").unwrap();
        assert_eq!(file.buckets.keys().collect::<Vec<_>>(), vec!["general"]);
    }

    #[cfg(unix)]
    #[test]
    fn a_limited_profile_turn_without_a_future_reset_warns_and_raises_nothing() {
        let _lock = test_io_lock();
        for (label, resets_at) in [("none", None), ("past", Some(1_000_000_000))] {
            let mut f = Fixture::new(&format!("run-profile-limited-{label}"));
            f.token("team03");
            f.claude_script(&refused_stream(resets_at));
            let completion = pinned_run(&f, "team03");
            assert_eq!(completion, crate::harness::EXIT_LIMITED);
            assert!(
                warned(
                    &completion,
                    &["team03 was not flagged exhausted: the limit carried no reset time"]
                ),
                "{label}: {:?}",
                completion.end.warnings
            );
            assert!(profile_limits(&f, "team03").is_none(), "{label}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_limited_turn_under_an_unconfigured_profile_raises_and_publishes_nothing() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-profile-limited-unconfigured");
        assert!(!f.cfg.claude.profiles.contains(&"team09".to_string()));
        f.token("team09");
        f.claude_script(&refused_stream(Some(Timestamp::now().as_second() + 86_400)));
        let usage = || UsageSource::Unavailable("not consulted".into());
        let (completion, bus) = bus_run(&f, Some("team09"), &usage, vec![]);
        assert_eq!(completion, crate::harness::EXIT_LIMITED);
        assert!(profile_limits(&f, "team09").is_none());
        assert!(!bus
            .calls()
            .into_iter()
            .any(|c| matches!(c, crate::bus::Recorded::PutAccount(s) if s.account == "team09")));
    }

    #[cfg(unix)]
    #[test]
    fn an_unwritable_limits_file_costs_a_warning_and_not_the_limit() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-profile-limits-unwritable");
        f.token("team03");
        // A directory where the limits directory belongs.
        std::fs::create_dir_all(f.path("state/ccc")).unwrap();
        std::fs::write(f.path("state/ccc/limits"), "not a directory").unwrap();
        f.claude_script(&refused_stream(Some(Timestamp::now().as_second() + 3_600)));
        let completion = pinned_run(&f, "team03");
        assert_eq!(completion, crate::harness::EXIT_LIMITED);
        assert!(
            warned(&completion, &["could not record team03 as exhausted"]),
            "{:?}",
            completion.end.warnings
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_successful_profile_turn_writes_no_limits_file() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-profile-ok-no-limits");
        f.token("team03");
        f.claude_script(OK_RESULT);
        assert_eq!(pinned_run(&f, "team03"), 0);
        assert!(profile_limits(&f, "team03").is_none());
    }

    #[cfg(unix)]
    #[test]
    fn a_key_turn_refused_otherwise_stays_a_reported_failure() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-api-key-reported");
        api_key(&mut f, "team");
        f.claude_script(&refused_turn("Something else went wrong."));
        let completion = pinned_run(&f, "team");
        assert_eq!(completion, crate::harness::EXIT_HARNESS_FAILED);
        let ledger = key_ledger(&f, "team").unwrap();
        assert_eq!(ledger.turns, 1);
        assert!(ledger.exhausted.is_none());
    }

    #[cfg(unix)]
    #[test]
    fn an_unwritable_ledger_costs_a_note_and_not_the_turn() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-api-key-damaged");
        api_key(&mut f, "team");
        std::fs::create_dir_all(f.path("state/ccc/budget")).unwrap();
        std::fs::write(f.path("state/ccc/budget/team.json"), "{damaged").unwrap();
        f.claude_script(OK_RESULT);
        let completion = pinned_run(&f, "team");
        assert_eq!(completion, 0);
        assert!(
            warned(&completion, &["could not charge", "team", "damaged"]),
            "{:?}",
            completion.end.warnings
        );
    }

    #[cfg(unix)]
    #[test]
    fn an_unwritable_ledger_still_limits_a_turn_refused_for_spend() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-api-key-damaged-exhausted");
        api_key(&mut f, "team");
        std::fs::create_dir_all(f.path("state/ccc/budget")).unwrap();
        std::fs::write(f.path("state/ccc/budget/team.json"), "{damaged").unwrap();
        f.claude_script(&refused_turn(
            "Your credit balance is too low to access the Anthropic API.",
        ));
        let completion = pinned_run(&f, "team");
        assert_eq!(completion, crate::harness::EXIT_LIMITED);
        for needle in [
            "could not charge",
            "could not record API key team as exhausted",
            "team hit a usage limit",
        ] {
            assert!(
                warned(&completion, &[needle]),
                "{needle}: {:?}",
                completion.end.warnings
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn every_launch_form_refuses_a_key() {
        let mut f = Fixture::new("launch-api-key");
        api_key(&mut f, "team");
        f.launch_script("exit 0", false);
        start_launch(&f);
        let usage = || UsageSource::Unavailable("not consulted".into());
        let targets = [
            LaunchTarget::Fresh {
                agent: "explorer",
                cwd: None,
                cli_harness: None,
                allow_read: &[],
                allow_write: &[],
                allow_exec: &[],
            },
            LaunchTarget::Roleless {
                cwd: None,
                model: None,
                effort: None,
            },
            LaunchTarget::Resume {
                id: "9f22dd1b-aaaa",
                fork: false,
            },
            LaunchTarget::Resume {
                id: "9f22dd1b-aaaa",
                fork: true,
            },
        ];
        for target in targets {
            let err = launch_with_ctx(&f.ctx(&usage), target, Some("team"), vec![]).unwrap_err();
            let message = format!("{:#}", err.error);
            assert!(
                message.contains("an API key account cannot be used at the terminal"),
                "{message}"
            );
        }
        assert!(!audit_records(&f)
            .iter()
            .any(|r| matches!(r, audit::Record::Spawn(_))));
    }

    #[cfg(unix)]
    fn start_launch(f: &Fixture) {
        f.audit
            .start(StartPayload {
                pid: std::process::id(),
                cmd: AuditCommand::Launch,
                agent: Some("explorer".into()),
                session: None,
                cwd_arg: Some(f.work()),
                profile_arg: None,
                harness_arg: None,
                json: false,
                prompt_source: None,
                prompt_file: None,
                prompt: None,
                passthrough: vec![],
                fork: false,
            })
            .unwrap();
    }

    #[cfg(unix)]
    fn launch_records(f: &Fixture) -> Vec<audit::Record> {
        audit::read_trace(
            &Recorder::dir_in(&f.env),
            f.audit.lineage().trace_id.as_str(),
        )
        .unwrap()
    }

    #[cfg(unix)]
    fn start_run(
        f: &Fixture,
        source: Option<PromptSource>,
        prompt_file: Option<&str>,
        prompt: Option<&str>,
    ) {
        f.audit
            .start(StartPayload {
                pid: std::process::id(),
                cmd: AuditCommand::Run,
                agent: Some("explorer".into()),
                session: None,
                cwd_arg: Some(f.work()),
                profile_arg: None,
                harness_arg: None,
                json: false,
                prompt_source: source,
                prompt_file: prompt_file.map(String::from),
                prompt: prompt.map(String::from),
                passthrough: vec![],
                fork: false,
            })
            .unwrap();
    }

    #[cfg(unix)]
    fn start_resume(f: &Fixture, session: &str) {
        f.audit
            .start(StartPayload {
                pid: std::process::id(),
                cmd: AuditCommand::Resume,
                agent: None,
                session: Some(session.into()),
                cwd_arg: None,
                profile_arg: None,
                harness_arg: None,
                json: false,
                prompt_source: Some(PromptSource::Arg),
                prompt_file: None,
                prompt: Some("again".into()),
                passthrough: vec![],
                fork: false,
            })
            .unwrap();
    }

    #[cfg(unix)]
    fn audit_records(f: &Fixture) -> Vec<audit::Record> {
        launch_records(f)
    }

    #[cfg(unix)]
    fn audit_end(f: &Fixture) -> audit::EndRecord {
        audit_records(f)
            .into_iter()
            .find_map(|r| match r {
                audit::Record::End(end) => Some(end),
                _ => None,
            })
            .expect("audit end")
    }

    fn record_completion(f: &Fixture, completion: Completion) {
        f.audit.end(completion.end);
    }

    #[cfg(unix)]
    fn latest_records(env: &BTreeMap<String, String>) -> Vec<audit::Record> {
        let dir = Recorder::dir_in(env);
        let index = audit::read_index(&dir).unwrap().pop().unwrap();
        audit::read_trace(&dir, &index.trace).unwrap()
    }

    #[cfg(unix)]
    fn assert_child_fixture_identity(
        env: &BTreeMap<String, String>,
        identity_file: &Path,
        expected_pgid: Option<u32>,
    ) {
        let fields = std::fs::read_to_string(identity_file)
            .unwrap()
            .split_whitespace()
            .map(str::to_string)
            .collect::<Vec<_>>();
        assert_eq!(fields.len(), 4);
        let proc = ProcId {
            pid: fields[0].parse().unwrap(),
            start: fields[2].parse().unwrap(),
            boot_id: fields[3].clone(),
        };
        let pgid: u32 = fields[1].parse().unwrap();
        let children = latest_records(env)
            .into_iter()
            .filter_map(|record| match record {
                audit::Record::Child(child) => Some(child.payload),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(children.len(), 1);
        let child = &children[0];
        assert_eq!(child.proc, proc);
        assert_eq!(child.pgid, expected_pgid.unwrap_or(pgid));
        assert_eq!(child.supervisor, ProcId::current().unwrap());
        if expected_pgid.is_none() {
            assert_eq!(child.pgid, child.proc.pid);
        }
    }

    #[cfg(unix)]
    #[test]
    fn signalled_run_records_child_signals_and_raw_death_signal() {
        let _signal_lock = PROCESS_SIGNAL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut f = Fixture::new("run-supervisor-signal");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.cfg.process.kill_grace_s = 0.2;
        let binary = f.path("signal-claude");
        let identity = f.path("run-identity");
        f.write_executable(
            &binary,
            &format!(
                "#!/bin/sh\nprintf '%s %s %s %s' $$ \"$(ps -o pgid= -p $$ | tr -d ' ')\" \"$(awk '{{print $22}}' /proc/$$/stat)\" \"$(cat /proc/sys/kernel/random/boot_id)\" > {}\ntrap '' TERM\nsleep 10\n",
                identity.display()
            ),
        );
        f.env
            .insert("CCC_CLAUDE_BIN".into(), binary.display().to_string());
        supervise::set_test_signal(true);
        let mut input = "prompt".as_bytes();
        let mut output = Vec::new();
        let code = run_from(
            "explorer",
            Some(&f.work()),
            Some("prompt"),
            None,
            None,
            None,
            false,
            &f.env,
            || Ok(f.cfg.clone()),
            &mut input,
            &mut output,
        )
        .unwrap();
        assert_eq!(code, crate::harness::EXIT_NO_RESULT);
        let records = latest_records(&f.env);
        assert_child_fixture_identity(&f.env, &identity, None);
        assert!(records
            .iter()
            .any(|r| matches!(r, audit::Record::Signal(_))));
        let end = records
            .into_iter()
            .find_map(|r| match r {
                audit::Record::End(end) => Some(end),
                _ => None,
            })
            .unwrap();
        assert_eq!(end.payload.exit, Some(crate::harness::EXIT_NO_RESULT));
        assert_eq!(end.payload.signal, Some(libc::SIGKILL));
        assert!(end.payload.session.is_some());
    }

    #[cfg(unix)]
    #[test]
    fn caught_term_run_has_exit_three_and_names_the_caught_exit() {
        let _signal_lock = PROCESS_SIGNAL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut f = Fixture::new("run-supervisor-caught");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.cfg.process.kill_grace_s = 0.2;
        let binary = f.path("caught-claude");
        f.write_executable(&binary, "#!/bin/sh\ntrap 'exit 1' TERM\nsleep 10 & wait\n");
        f.env
            .insert("CCC_CLAUDE_BIN".into(), binary.display().to_string());
        supervise::set_test_signal(true);
        let mut input = "prompt".as_bytes();
        let mut output = Vec::new();
        let code = run_from(
            "explorer",
            Some(&f.work()),
            Some("prompt"),
            None,
            None,
            None,
            false,
            &f.env,
            || Ok(f.cfg.clone()),
            &mut input,
            &mut output,
        )
        .unwrap();
        assert_eq!(code, crate::harness::EXIT_NO_RESULT);
        let end = latest_records(&f.env)
            .into_iter()
            .find_map(|r| match r {
                audit::Record::End(end) => Some(end),
                _ => None,
            })
            .unwrap();
        assert_eq!(end.payload.signal, None);
        assert!(end.payload.error.unwrap().contains("exit 1"));
    }

    #[cfg(unix)]
    #[test]
    fn signalled_resume_records_the_child_and_raw_death_signal() {
        let _signal_lock = PROCESS_SIGNAL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut f = Fixture::new("resume-supervisor-signal");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.cfg.process.kill_grace_s = 0.2;
        let record = record_of(&resolved(), Path::new(&f.work()), Some(&account()), f.now);
        session::save_in(&f.sessions_dir(), "resume-session", &record).unwrap();
        let binary = f.path("resume-claude");
        let identity = f.path("resume-identity");
        f.write_executable(
            &binary,
            &format!(
                "#!/bin/sh\nprintf '%s %s %s %s' $$ \"$(ps -o pgid= -p $$ | tr -d ' ')\" \"$(awk '{{print $22}}' /proc/$$/stat)\" \"$(cat /proc/sys/kernel/random/boot_id)\" > {}\ntrap '' TERM\nsleep 10\n",
                identity.display()
            ),
        );
        f.env
            .insert("CCC_CLAUDE_BIN".into(), binary.display().to_string());
        supervise::set_test_signal(true);
        let mut input = "prompt".as_bytes();
        let mut output = Vec::new();
        let code = resume_from(
            "resume-session",
            false,
            None,
            Some("prompt"),
            None,
            false,
            &f.env,
            || Ok(f.cfg.clone()),
            &mut input,
            &mut output,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(code, crate::harness::EXIT_NO_RESULT);
        let records = latest_records(&f.env);
        assert_child_fixture_identity(&f.env, &identity, None);
        let end = records
            .into_iter()
            .find_map(|r| match r {
                audit::Record::End(end) => Some(end),
                _ => None,
            })
            .unwrap();
        assert_eq!(end.payload.signal, Some(libc::SIGKILL));
        assert_eq!(end.payload.session.as_deref(), Some("resume-session"));
    }

    #[cfg(unix)]
    #[test]
    fn launch_supervisor_records_child_identity() {
        let _lock = supervise::TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut f = Fixture::new("launch-supervisor-child");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        let identity = f.path("launch-identity");
        f.launch_script(
            &format!(
                "printf '%s %s %s %s' $$ \"$(ps -o pgid= -p $$ | tr -d ' ')\" \"$(awk '{{print $22}}' /proc/$$/stat)\" \"$(cat /proc/sys/kernel/random/boot_id)\" > {}\nexit 0",
                identity.display()
            ),
            false,
        );
        start_launch(&f);
        let usage = || UsageSource::Unavailable("not consulted".into());
        let completion = launch_with_ctx(
            &f.ctx(&usage),
            LaunchTarget::Fresh {
                agent: "explorer",
                cwd: Some(&f.work()),
                cli_harness: None,
                allow_read: &[],
                allow_write: &[],
                allow_exec: &[],
            },
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(completion.code, 0);
        record_completion(&f, completion);
        assert_child_fixture_identity(
            &f.env,
            &identity,
            Some(
                std::fs::read_to_string(&identity)
                    .unwrap()
                    .split_whitespace()
                    .nth(1)
                    .unwrap()
                    .parse()
                    .unwrap(),
            ),
        );
        assert_eq!(audit_end(&f).payload.signal, None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn launch_child_starts_with_managed_masks_clear() {
        let inv = Invocation {
            program: "/bin/sh".into(),
            args: vec![
                "-c".into(),
                "blk=$(awk '$1 == \"SigBlk:\" {print $2}' /proc/self/status); ign=$(awk '$1 == \"SigIgn:\" {print $2}' /proc/self/status); test \"$blk\" = 0000000000000000; case \"$ign\" in *[2367]) exit 1;; esac".into(),
            ],
            env_set: vec![],
            env_remove: vec![],
            cwd: PathBuf::from("/tmp"),
            stdin: None,
        };
        assert_eq!(run_launch_test(&inv).unwrap().code(), Some(0));
    }

    #[cfg(unix)]
    #[test]
    fn a_claude_run_passes_a_configured_permission_mode_to_the_child() {
        let mut f = Fixture::new("run-permission-mode");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.cfg.claude.permission_mode = PermissionMode::DontAsk;
        f.claude_script(OK_RESULT);
        start_run(&f, Some(PromptSource::Arg), None, Some("go"));
        let usage = || UsageSource::Unavailable("not consulted".into());
        let mut out = Vec::new();
        let completion = run_with(
            &f.ctx(&usage),
            "explorer",
            Some(&f.work()),
            "go".into(),
            None,
            None,
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(completion, 0);
        record_completion(&f, completion);

        let argv = argv_of(&f);
        let i = argv
            .iter()
            .position(|a| a == "--permission-mode")
            .unwrap_or_else(|| panic!("--permission-mode: {argv:?}"));
        assert_eq!(argv[i + 1], "dontAsk", "{argv:?}");
    }

    #[cfg(unix)]
    #[test]
    fn a_fresh_run_records_routing_lineage_result_and_index() {
        let mut f = Fixture::new("run-audit-integration");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        std::fs::write(
            f.path("agents/explorer.md"),
            "---\nname: explorer\ndescription: Investigates\nharness: claude\nmodel: claude-haiku-4-5\neffort: low\ntools: Read, Grep\n---\nthe system prompt\n",
        )
        .unwrap();
        f.env.insert("CCC_HARNESS_ALL".into(), "codex".into());
        f.claude_script(OK_RESULT);
        start_run(&f, Some(PromptSource::Arg), None, Some("go and look"));
        let usage = || UsageSource::Unavailable("not consulted".into());
        let mut out = Vec::new();
        let completion = run_with(
            &f.ctx(&usage),
            "explorer",
            Some(&f.work()),
            "go and look".into(),
            None,
            None,
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(completion, 0);
        record_completion(&f, completion);

        let records = audit_records(&f);
        assert_eq!(
            records
                .iter()
                .filter(|r| matches!(r, audit::Record::Start(_)))
                .count(),
            1
        );
        assert_eq!(
            records
                .iter()
                .filter(|r| matches!(r, audit::Record::Spawn(_)))
                .count(),
            1
        );
        let spawn = records
            .iter()
            .find_map(|r| match r {
                audit::Record::Spawn(spawn) => Some(spawn),
                _ => None,
            })
            .unwrap();
        assert_eq!(spawn.payload.harness, Harness::Claude);
        assert_eq!(spawn.payload.model, "claude-haiku-4-5");
        assert_eq!(spawn.payload.effort, Some(Effort::Low));
        assert_eq!(spawn.payload.profile.as_deref(), Some("team03"));
        assert_eq!(spawn.payload.auth, Some(ClaudeAuth::Token));
        assert_eq!(spawn.payload.agent.as_deref(), Some("explorer"));
        assert_eq!(
            spawn.payload.cwd,
            f.path("work").canonicalize().unwrap().display().to_string()
        );
        assert!(spawn.payload.session.is_some());
        assert_eq!(
            spawn.payload.tools,
            Some(vec!["Read".into(), "Grep".into()])
        );
        assert_eq!(spawn.payload.sandbox, Sandbox::WorkspaceWrite);
        assert!(spawn.payload.selection_notes[0].contains("CCC_HARNESS_ALL"));
        assert!(spawn.payload.selection_notes[0].contains("skipped"));
        assert!(spawn.payload.selection_notes[1..]
            .iter()
            .any(|note| note.contains("falling back")));

        let index = audit::read_index(&Recorder::dir_in(&f.env)).unwrap();
        let index_line = &index[0];
        let start = records
            .iter()
            .find_map(|r| match r {
                audit::Record::Start(s) => Some(s),
                _ => None,
            })
            .unwrap();
        assert_eq!(start.inv, f.audit.lineage().invocation_id);
        assert_eq!(spawn.inv, f.audit.lineage().invocation_id);
        assert_eq!(index_line.trace, f.audit.lineage().trace_id);
        assert_eq!(index_line.inv, f.audit.lineage().invocation_id);
        assert_eq!(index_line.pid, start.payload.pid);
        let end = audit_end(&f);
        assert_eq!(end.payload.exit, Some(0));
        assert_eq!(end.payload.session.as_deref(), Some("9f22dd1b-aaaa"));
        assert_eq!(end.payload.result.as_deref(), Some("the answer"));
        assert_eq!(end.payload.native.as_ref().unwrap()["num_turns"], 1);
        let env_text = f.call("env");
        assert_eq!(
            env_text
                .lines()
                .find_map(|line| line.strip_prefix("CCC_TRACE_ID=")),
            Some(f.audit.lineage().trace_id.as_str())
        );
        assert_eq!(
            env_text
                .lines()
                .find_map(|line| line.strip_prefix("CCC_PARENT_INVOCATION_ID=")),
            Some(f.audit.lineage().invocation_id.as_str())
        );
        let preset = spawn.payload.session.as_deref().unwrap();
        assert!(f.sessions_dir().join("9f22dd1b-aaaa.json").exists());
        assert!(f.sessions_dir().join(format!("{preset}.json")).exists());
        assert_eq!(end.payload.warnings.len(), 1);
        assert!(
            end.payload.warnings[0].contains(preset)
                && end.payload.warnings[0].contains("9f22dd1b-aaaa")
        );
        assert_eq!(
            audit::read_index(&Recorder::dir_in(&f.env)).unwrap().len(),
            1
        );
    }

    #[cfg(unix)]
    #[test]
    fn fresh_prompt_sources_are_recorded_and_prompt_failures_end_without_spawn() {
        for (label, source, prompt_file, prompt) in [
            ("run-prompt-arg", PromptSource::Arg, None, "from arg"),
            (
                "run-prompt-file",
                PromptSource::File,
                Some("prompt.txt"),
                "from file",
            ),
            ("run-prompt-stdin", PromptSource::Stdin, None, "from stdin"),
        ] {
            let mut f = Fixture::new(label);
            f.token("team03");
            f.cfg.claude.profiles = vec!["team03".into()];
            f.claude_script(OK_RESULT);
            if let Some(path) = prompt_file {
                std::fs::write(f.path(path), prompt).unwrap();
            }
            start_run(&f, Some(source), prompt_file, Some(prompt));
            let usage = || UsageSource::Unavailable("not consulted".into());
            let mut out = Vec::new();
            run_with(
                &f.ctx(&usage),
                "explorer",
                Some(&f.work()),
                prompt.into(),
                None,
                None,
                false,
                &mut out,
            )
            .unwrap();
            let start = audit_records(&f)
                .into_iter()
                .find_map(|r| match r {
                    audit::Record::Start(start) => Some(start),
                    _ => None,
                })
                .unwrap();
            assert_eq!(start.payload.prompt_source, Some(source));
            assert_eq!(start.payload.prompt.as_deref(), Some(prompt));
        }

        for (label, positional, file, input) in [
            ("run-prompt-both", Some("arg"), Some("prompt.txt"), "file"),
            ("run-prompt-unreadable", None, Some("missing.txt"), ""),
            ("run-prompt-empty", None, None, "   \n"),
        ] {
            let f = Fixture::new(label);
            if let Some(path) = file {
                if path == "prompt.txt" {
                    std::fs::write(f.path(path), "file").unwrap();
                }
            }
            start_run(&f, prompt_source(positional, file), file, None);
            let mut stdin = input.as_bytes();
            let err = prompt_text_from(positional, file, &mut stdin).unwrap_err();
            let unchanged = format!("{err:#}");
            f.audit.end(EndPayload {
                exit: Some(2),
                signal: None,
                session: None,
                result: None,
                error: Some(unchanged.clone()),
                warnings: vec![],
                native: None,
                turns: None,
            });
            let end = audit_end(&f);
            assert_eq!(end.payload.exit, Some(2));
            assert_eq!(end.payload.error.as_deref(), Some(unchanged.as_str()));
            let start = audit_records(&f)
                .into_iter()
                .find_map(|r| match r {
                    audit::Record::Start(start) => Some(start),
                    _ => None,
                })
                .unwrap();
            assert_eq!(start.payload.prompt, None);
            assert!(!audit_records(&f)
                .iter()
                .any(|r| matches!(r, audit::Record::Spawn(_))));
            assert!(!f.path("calls/argv").exists());
        }
    }

    #[cfg(unix)]
    #[test]
    fn resume_output_and_audit_describe_the_newly_selected_account() {
        // The session recorded team03 under `token`; the config now says
        // `config-dir`, where only team04 has a credential, and team03 is
        // still usable. The turn runs on team04 with the ordinary selector's
        // warning about team03.
        for as_json in [false, true] {
            let mut f = Fixture::new("resume-audit-selected");
            f.token("team03");
            f.config_dir_credential("team04");
            f.cfg.claude.auth = ClaudeAuth::ConfigDir;
            f.claude_script(OK_RESULT);
            let record = claude_record(&f, Some("team03"), ClaudeAuth::Token);
            session::save_in(&f.sessions_dir(), "9f22dd1b-aaaa", &record).unwrap();
            start_resume(&f, "9f22dd1b-aaaa");
            let usage = || {
                report(vec![
                    snapshot(Timestamp::now(), "team03", 0.0, 10.0),
                    snapshot(Timestamp::now(), "team04", 0.0, 50.0),
                ])
            };
            let mut out = Vec::new();
            resume_with(
                &f.ctx(&usage),
                "9f22dd1b-aaaa",
                false,
                None,
                "again".into(),
                as_json,
                &mut out,
            )
            .unwrap();
            let text = String::from_utf8(out).unwrap();
            if as_json {
                let v: serde_json::Value = serde_json::from_str(&text).unwrap();
                assert_eq!(v["profile"], "team04", "{text}");
            } else {
                assert!(text.contains("the answer"), "{text}");
                assert!(text.ends_with("profile=team04\n"), "{text:?}");
            }
            let records = audit_records(&f);
            let spawn = spawn_of(&records);
            assert_eq!(spawn.payload.profile.as_deref(), Some("team04"));
            assert_eq!(spawn.payload.auth, Some(ClaudeAuth::ConfigDir));
            let notes = &spawn.payload.selection_notes;
            assert!(
                notes.iter().any(|n| n.contains("team03")),
                "the selector's credential warning survives: {notes:?}"
            );
            for gone in ["keeping", "re-select", "recorded auth", "no longer usable"] {
                assert!(!notes.iter().any(|n| n.contains(gone)), "{gone}: {notes:?}");
            }
            let saved = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
            assert_eq!(saved.profile.as_deref(), Some("team04"));
            assert_eq!(saved.auth, Some(ClaudeAuth::ConfigDir));
        }
    }

    #[cfg(unix)]
    #[test]
    fn launch_exit_and_turn_records_have_the_complete_lifecycle() {
        let _lock = supervise::TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut exit = Fixture::new("launch-exit");
        exit.token("team03");
        exit.cfg.claude.profiles = vec!["team03".into()];
        exit.launch_script("exit 7", false);
        start_launch(&exit);
        let usage = || UsageSource::Unavailable("not consulted".into());
        let completion = launch_with_ctx(
            &exit.ctx(&usage),
            LaunchTarget::Fresh {
                agent: "explorer",
                cwd: Some(&exit.work()),
                cli_harness: None,
                allow_read: &[],
                allow_write: &[],
                allow_exec: &[],
            },
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(completion, 7);
        record_completion(&exit, completion);
        let records = launch_records(&exit);
        assert!(matches!(records.first(), Some(audit::Record::Start(_))));
        assert!(matches!(records.get(1), Some(audit::Record::Spawn(_))));
        let end = records
            .iter()
            .find_map(|r| match r {
                audit::Record::End(end) => Some(end),
                _ => None,
            })
            .expect("launch end");
        assert_eq!(end.payload.exit, Some(7));
        assert!(end.duration_ms < u64::MAX);

        let mut turn = Fixture::new("launch-turn");
        turn.token("team03");
        turn.cfg.claude.profiles = vec!["team03".into()];
        let turn_trace = turn.path("state/ccc/audit").display().to_string();
        turn.launch_script(
            &r#"printf '{"t":"turn","ts":"2026-01-01T00:00:00Z","inv":"%s","harness":"claude","session":"turn-session","turn":null,"result":"turn result","transcript":null}\n' "$CCC_PARENT_INVOCATION_ID" >> "$AUDIT/$CCC_TRACE_ID.jsonl""#.replace("$AUDIT", &turn_trace),
            false,
        );
        start_launch(&turn);
        let completion = launch_with_ctx(
            &turn.ctx(&usage),
            LaunchTarget::Fresh {
                agent: "explorer",
                cwd: Some(&turn.work()),
                cli_harness: None,
                allow_read: &[],
                allow_write: &[],
                allow_exec: &[],
            },
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(completion, 0);
        record_completion(&turn, completion);
        let end = launch_records(&turn)
            .into_iter()
            .find_map(|r| match r {
                audit::Record::End(end) => Some(end),
                _ => None,
            })
            .expect("launch end");
        assert_eq!(end.payload.session.as_deref(), Some("turn-session"));
        assert_eq!(end.payload.result.as_deref(), Some("turn result"));
        assert_eq!(end.payload.turns, Some(1));
    }

    #[cfg(unix)]
    #[test]
    fn launch_no_turn_uses_claude_and_codex_fallbacks() {
        let _lock = supervise::TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut claude = Fixture::new("launch-no-turn-claude");
        claude.token("team03");
        claude.cfg.claude.profiles = vec!["team03".into()];
        claude.launch_script("exit 0", false);
        start_launch(&claude);
        let usage = || UsageSource::Unavailable("not consulted".into());
        let completion = launch_with_ctx(
            &claude.ctx(&usage),
            LaunchTarget::Fresh {
                agent: "explorer",
                cwd: Some(&claude.work()),
                cli_harness: None,
                allow_read: &[],
                allow_write: &[],
                allow_exec: &[],
            },
            None,
            vec![],
        )
        .unwrap();
        record_completion(&claude, completion);
        let end = launch_records(&claude)
            .into_iter()
            .find_map(|r| match r {
                audit::Record::End(end) => Some(end),
                _ => None,
            })
            .unwrap();
        assert!(end.payload.session.is_some());
        assert_eq!(end.payload.turns, Some(0));

        let mut codex = Fixture::new("launch-no-turn-codex");
        codex.cfg.agents.insert(
            "explorer".into(),
            crate::config::AgentOverrides {
                codex_model: Some("gpt-5.6-terra".into()),
                ..Default::default()
            },
        );
        codex.launch_script("exit 0", true);
        start_launch(&codex);
        let completion = launch_with_ctx(
            &codex.ctx(&usage),
            LaunchTarget::Fresh {
                agent: "explorer",
                cwd: Some(&codex.work()),
                cli_harness: Some(Harness::Codex),
                allow_read: &[],
                allow_write: &[],
                allow_exec: &[],
            },
            None,
            vec![],
        )
        .unwrap();
        record_completion(&codex, completion);
        let end = launch_records(&codex)
            .into_iter()
            .find_map(|r| match r {
                audit::Record::End(end) => Some(end),
                _ => None,
            })
            .unwrap();
        assert_eq!(end.payload.session, None);
        assert_eq!(end.payload.turns, Some(0));
    }

    #[cfg(unix)]
    #[test]
    fn launch_signal_returns_no_result_and_parent_sigint_is_ignored() {
        let _lock = supervise::TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let usage = || UsageSource::Unavailable("not consulted".into());
        let mut signal = Fixture::new("launch-signal");
        signal.token("team03");
        signal.cfg.claude.profiles = vec!["team03".into()];
        signal.launch_script("kill -TERM $$", false);
        start_launch(&signal);
        let completion = launch_with_ctx(
            &signal.ctx(&usage),
            LaunchTarget::Fresh {
                agent: "explorer",
                cwd: Some(&signal.work()),
                cli_harness: None,
                allow_read: &[],
                allow_write: &[],
                allow_exec: &[],
            },
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(completion, crate::harness::EXIT_NO_RESULT);
        record_completion(&signal, completion);
        let end = launch_records(&signal)
            .into_iter()
            .find_map(|r| match r {
                audit::Record::End(end) => Some(end),
                _ => None,
            })
            .unwrap();
        assert_eq!(end.payload.signal, Some(15));

        let mut parent = Fixture::new("launch-parent-sigint");
        parent.token("team03");
        parent.cfg.claude.profiles = vec!["team03".into()];
        parent.launch_script("n=0; ign=0; while [ $n -lt 500 ]; do mask=$(awk '/^SigIgn:/ {print $2}' /proc/$PPID/status); ign=$((0x$mask & 2)); [ $ign -ne 0 ] && break; sleep .01; n=$((n+1)); done; [ $ign -ne 0 ] || exit 9; kill -INT $PPID; exit 7", false);
        start_launch(&parent);
        let completion = launch_with_ctx(
            &parent.ctx(&usage),
            LaunchTarget::Fresh {
                agent: "explorer",
                cwd: Some(&parent.work()),
                cli_harness: None,
                allow_read: &[],
                allow_write: &[],
                allow_exec: &[],
            },
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(completion, 7);
        record_completion(&parent, completion);
        assert!(launch_records(&parent)
            .iter()
            .any(|r| matches!(r, audit::Record::End(_))));
    }

    #[cfg(unix)]
    #[test]
    fn supervise_waits_for_exit_and_returns_signal_status() {
        let exit = Invocation {
            program: "/bin/sh".into(),
            args: vec!["-c".into(), "exit 7".into()],
            env_set: vec![],
            env_remove: vec![],
            cwd: "/tmp".into(),
            stdin: None,
        };
        assert_eq!(run_launch_test(&exit).unwrap().code(), Some(7));
        let signal = Invocation {
            program: "/bin/sh".into(),
            args: vec!["-c".into(), "kill -TERM $$".into()],
            env_set: vec![],
            env_remove: vec![],
            cwd: "/tmp".into(),
            stdin: None,
        };
        assert_eq!(
            std::os::unix::process::ExitStatusExt::signal(&run_launch_test(&signal).unwrap()),
            Some(15)
        );

        let parent_signal = Invocation {
            program: "/bin/sh".into(),
            args: vec!["-c".into(), "n=0; ign=0; while [ $n -lt 500 ]; do mask=$(awk '/^SigIgn:/ {print $2}' /proc/$PPID/status); ign=$((0x$mask & 2)); [ $ign -ne 0 ] && break; sleep .01; n=$((n+1)); done; [ $ign -ne 0 ] || exit 9; kill -INT $PPID; exit 7".into()],
            env_set: vec![],
            env_remove: vec![],
            cwd: "/tmp".into(),
            stdin: None,
        };
        assert_eq!(run_launch_test(&parent_signal).unwrap().code(), Some(7));
    }

    #[test]
    fn a_fresh_claude_session_mismatch_is_named_without_changing_the_authority() {
        let warning =
            session_mismatch_warning(Some("preset"), Some("reported")).expect("mismatch warning");
        assert!(warning.contains("preset"), "{warning}");
        assert!(warning.contains("reported"), "{warning}");
        assert_eq!(session_mismatch_warning(Some("same"), Some("same")), None);
        assert_eq!(session_mismatch_warning(None, Some("reported")), None);
    }

    #[test]
    fn a_fork_collision_is_named_only_for_a_fork_answered_under_its_parent() {
        let fork = crate::harness::modes::one_shot(crate::harness::modes::resumed("parent", true));
        let warning = fork_collision_warning(&fork, Some("parent")).expect("collision warning");
        assert!(warning.contains("parent's id parent"), "{warning}");
        assert_eq!(fork_collision_warning(&fork, Some("child")), None);
        assert_eq!(fork_collision_warning(&fork, None), None);
        let resume =
            crate::harness::modes::one_shot(crate::harness::modes::resumed("parent", false));
        assert_eq!(fork_collision_warning(&resume, Some("parent")), None);
        assert_eq!(fork_collision_warning(&run_mode(), Some("parent")), None);
    }

    #[test]
    fn an_absent_prompt_and_a_dash_both_read_stdin() {
        let mut stdin = "piped in\n".as_bytes();
        assert_eq!(
            prompt_text_from(None, None, &mut stdin).unwrap(),
            "piped in\n"
        );
        let mut stdin = "piped in\n".as_bytes();
        assert_eq!(
            prompt_text_from(None, Some("-"), &mut stdin).unwrap(),
            "piped in\n"
        );

        let mut empty = "".as_bytes();
        let err = prompt_text_from(None, None, &mut empty)
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty"), "{err}");
        let mut blank = "  \n".as_bytes();
        let err = prompt_text_from(None, Some("-"), &mut blank)
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty"), "{err}");
        let mut blank = "  \n".as_bytes();
        let err = prompt_text_from(Some("-"), None, &mut blank)
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty"), "{err}");
    }

    #[test]
    fn the_working_directory_is_absolute_and_must_exist() {
        let dir = TempDir::new("cwd");
        let env: BTreeMap<String, String> = [("HOME".to_string(), "/home/tester".to_string())]
            .into_iter()
            .collect();

        let got = resolve_cwd(&env, Some(dir.path().to_str().unwrap())).unwrap();
        assert!(got.is_absolute());
        assert_eq!(got, std::fs::canonicalize(dir.path()).unwrap());

        let err = format!(
            "{:#}",
            resolve_cwd(&env, Some("/nonexistent/place")).unwrap_err()
        );
        assert!(err.contains("/nonexistent/place"), "{err}");

        // A tilde is expanded against the supplied environment, not the shell.
        let err = format!("{:#}", resolve_cwd(&env, Some("~/nope")).unwrap_err());
        assert!(err.contains("/home/tester/nope"), "{err}");
    }

    #[test]
    fn a_working_directory_that_is_a_file_is_refused() {
        let dir = TempDir::new("cwd-file");
        let file = dir.write_file("notadir", "x", 0o600);
        let env: BTreeMap<String, String> = BTreeMap::new();
        let err = format!(
            "{:#}",
            resolve_cwd(&env, Some(file.to_str().unwrap())).unwrap_err()
        );
        assert!(err.contains("is not a directory"), "{err}");
    }

    #[test]
    fn a_record_captures_everything_a_resume_pins() {
        let now = Timestamp::now();
        let r = record_of(
            &resolved(),
            std::path::Path::new("/srv"),
            Some(&account()),
            now,
        );
        assert_eq!(r.harness, Harness::Claude);
        assert_eq!(r.agent.as_deref(), Some("explorer"));
        assert_eq!(r.model, "claude-haiku-4-5[1m]");
        assert_eq!(r.effort, Some(Effort::Low));
        assert_eq!(r.profile.as_deref(), Some("team03"));
        assert_eq!(r.auth, Some(ClaudeAuth::Token));
        assert_eq!(r.cwd, PathBuf::from("/srv"));
        assert_eq!(r.sandbox, Sandbox::ReadOnly);
        assert_eq!(r.tools, Some(vec!["Read".to_string()]));
        assert_eq!(r.created_at, r.last_used_at);

        let mut codex_r = resolved();
        codex_r.harness = Harness::Codex;
        let r = record_of(&codex_r, std::path::Path::new("/srv"), None, now);
        assert_eq!(r.profile, None);
        assert_eq!(r.auth, None);
    }

    #[test]
    fn a_resume_takes_its_pinning_from_the_record_and_its_body_from_the_definition() {
        let now = Timestamp::now();
        let mut record = record_of(
            &resolved(),
            std::path::Path::new("/srv"),
            Some(&account()),
            now,
        );
        record.model = "claude-opus-5[1M]".into();
        record.sandbox = Sandbox::WorkspaceWrite;

        let def = AgentDef::parse(
            "---\nname: explorer\ndescription: Newer text\nharness: codex\ncodex_model: gpt-5\nmodel: claude-sonnet-4-5\n---\nthe newer body\n",
            "explorer",
        )
        .unwrap();

        let mut def_with_recorded_overlay = def.clone();
        def_with_recorded_overlay
            .overlays
            .insert(Harness::Claude, "the edited Claude overlay".into());

        let r = resolved_from(&record, Some(&def_with_recorded_overlay));
        assert_eq!(r.model, "claude-opus-5[1M]");
        assert_eq!(r.effort, Some(Effort::Low));
        assert_eq!(r.sandbox, Sandbox::WorkspaceWrite);
        assert_eq!(r.tools, Some(vec!["Read".to_string()]));
        assert_eq!(r.body, "the edited Claude overlay");
        assert_eq!(r.description, "Newer text");

        let mut def_with_other_overlay = def;
        def_with_other_overlay
            .overlays
            .insert(Harness::Codex, "the Codex overlay".into());
        let r = resolved_from(&record, Some(&def_with_other_overlay));
        assert_eq!(r.body, "the newer body");
        assert!(!def_with_other_overlay.allows(Harness::Claude));
    }

    #[test]
    fn a_roleless_record_resolves_with_an_empty_body_and_no_agent() {
        let mut record = record_of(
            &resolved(),
            std::path::Path::new("/srv"),
            Some(&account()),
            Timestamp::now(),
        );
        record.agent = None;
        let r = resolved_from(&record, None);
        assert_eq!(r.agent, None);
        assert_eq!(r.label(), "-");
        assert_eq!(r.body, "");
        assert_eq!(r.description, "");
        assert_eq!(r.model, "claude-haiku-4-5[1m]");
        assert_eq!(r.effort, Some(Effort::Low));
    }

    /// A roleless Claude sidecar under `id`, with the agents directory emptied
    /// so that any definition load fails.
    fn seed_roleless(f: &Fixture, id: &str) {
        let mut r = resolved();
        r.agent = None;
        r.tools = None;
        let record = record_of(
            &r,
            Path::new(&f.work()),
            Some(&account()),
            at(f.now, -1).unwrap(),
        );
        session::save_in(&f.sessions_dir(), id, &record).unwrap();
        std::fs::remove_file(f.path("agents/explorer.md")).unwrap();
    }

    #[test]
    fn a_roleless_session_resumes_without_a_definition() {
        let mut f = Fixture::new("resume-roleless");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        seed_roleless(&f, "9f22dd1b-aaaa");
        f.claude_script(OK_RESULT);
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);
        let ctx = f.ctx(&usage);

        let mut out: Vec<u8> = Vec::new();
        let completion = resume_with(
            &ctx,
            "9f22dd1b-aaaa",
            false,
            None,
            "again".into(),
            true,
            &mut out,
        )
        .unwrap();
        assert_eq!(completion, 0);

        let argv: Vec<String> = f.call("argv").lines().map(String::from).collect();
        let i = argv.iter().position(|a| a == "--resume").expect("--resume");
        assert_eq!(argv[i + 1], "9f22dd1b-aaaa");
        assert!(!argv.iter().any(|a| a == "--agent"), "{argv:?}");
        assert!(!argv.iter().any(|a| a == "--agents"), "{argv:?}");
        let json: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert!(json["agent"].is_null(), "{json}");
        let saved = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
        assert_eq!(saved.agent, None);
        let spawn = audit_records(&f)
            .into_iter()
            .find_map(|r| match r {
                audit::Record::Spawn(spawn) => Some(spawn),
                _ => None,
            })
            .expect("a spawn record");
        assert_eq!(spawn.payload.agent, None);
    }

    #[test]
    fn a_roleless_session_builds_an_interactive_resume_without_a_definition() {
        let mut f = Fixture::new("launch-resume-roleless");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        seed_roleless(&f, "9f22dd1b-aaaa");
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);
        let ctx = f.ctx(&usage);
        let prepared = launch_resume_invocation(&ctx, "9f22dd1b-aaaa", true, None, vec![]).unwrap();
        assert_eq!(prepared.r.agent, None);
        assert_eq!(prepared.record.agent, None);
        assert_eq!(prepared.inv.value_of("--resume"), Some("9f22dd1b-aaaa"));
        assert!(prepared.inv.has_arg("--fork-session"));
        assert!(!prepared.inv.has_arg("--agent"), "{:?}", prepared.inv.args);
        assert!(!prepared.inv.has_arg("--agents"), "{:?}", prepared.inv.args);
    }

    #[test]
    fn fresh_prepare_keeps_population_routing_notes_before_account_notes() {
        let mut f = Fixture::new("prepare-routing-notes");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        std::fs::write(
            f.path("agents/explorer.md"),
            "---\nname: explorer\ndescription: Investigates\nharness: claude\nmodel: claude-haiku-4-5\neffort: low\ntools: Read, Grep\n---\nthe system prompt\n",
        )
        .unwrap();
        f.env.insert("CCC_HARNESS_ALL".into(), "codex".into());
        let usage = || UsageSource::Unavailable("offline".into());
        let ctx = f.ctx(&usage);
        let prepared = prepare(
            &ctx,
            "explorer",
            Some(&f.work()),
            None,
            None,
            &GooseFlags::default(),
        )
        .unwrap();
        let account_notes = select_account(&ctx, &prepared.r, &(ctx.usage)())
            .unwrap()
            .notes;
        assert_eq!(prepared.r.harness, Harness::Claude);
        assert_eq!(
            prepared.notes.first().map(String::as_str),
            Some("CCC_HARNESS_ALL selected codex for agent \"explorer\", but allowed harnesses are: claude; skipped")
        );
        assert_eq!(&prepared.notes[1..], account_notes.as_slice());
        assert!(prepared.notes[1..]
            .iter()
            .all(|note| !note.contains("CCC_HARNESS_") && !note.contains("[agents.")));
    }

    // TODO(test-cli-acceptance-silently-absent-off-linux): restore safe CLI-level coverage on supported non-Linux Unix without relying on Linux procfs/process-group identity — the routing-note tests below and cli_run_then_resume_admits_the_completed_session_from_a_fresh_process.
    #[cfg(target_os = "linux")]
    #[test]
    fn cli_which_json_keeps_routing_notes_on_stderr() {
        let mut f = Fixture::new("cli-which-routing-note");
        f.token("team03");
        f.install_cli_config();
        std::fs::write(
            f.path("agents/explorer.md"),
            "---\nname: explorer\ndescription: Investigates\nharness: claude\nmodel: claude-haiku-4-5\n---\nthe system prompt\n",
        )
        .unwrap();
        f.env.insert("CCC_HARNESS_ALL".into(), "codex".into());

        let output = f.cli(&["which", "explorer", "--json"]);
        assert!(output.status.success(), "{output:?}");
        let stdout = String::from_utf8(output.stdout).unwrap();
        let json: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        assert_eq!(json["config"], f.env["CCC_CONFIG"]);
        assert_eq!(json["agents_config"], f.env["CCC_AGENTS_CONFIG"]);
        assert!(!stdout.contains("CCC_HARNESS_ALL") && !stdout.contains("note:"));
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(
            stderr.starts_with(&format!(
                "config: {} (CCC_CONFIG)\nagents config: {} (CCC_AGENTS_CONFIG)\nnote: ",
                f.env["CCC_CONFIG"], f.env["CCC_AGENTS_CONFIG"]
            )),
            "{stderr}"
        );
        let notes: Vec<_> = stderr
            .lines()
            .filter(|line| line.starts_with("note: "))
            .collect();
        assert_eq!(notes.len(), 1, "{stderr}");
        assert!(notes[0].contains("CCC_HARNESS_ALL"), "{stderr}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cli_refuses_an_old_layout_combined_config_before_spawning() {
        let mut f = Fixture::new("cli-old-layout-config");
        f.token("team03");
        f.install_cli_config();
        std::fs::write(
            f.path("agents/explorer.md"),
            "---\nname: explorer\ndescription: Investigates\nharness: claude\nmodel: claude-haiku-4-5\n---\nthe system prompt\n",
        )
        .unwrap();
        let combined = f.path("combined.toml");
        std::fs::write(
            &combined,
            format!(
                "agents_dir = \"{}\"\n[claude]\nprofiles = [\"team03\"]\n",
                f.path("agents").display()
            ),
        )
        .unwrap();
        f.env
            .insert("CCC_CONFIG".into(), combined.display().to_string());
        f.env.remove("CCC_AGENTS_CONFIG");
        f.claude_script(OK_RESULT);

        let output = f.cli(&["run", "explorer", "--cwd", &f.work(), "go"]);
        assert_eq!(output.status.code(), Some(2), "{output:?}");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(
            stderr.contains(&format!(
                "parsing config {} (CCC_CONFIG)",
                combined.display()
            )),
            "{stderr}"
        );
        assert!(
            stderr.contains("`agents_dir` belongs in the agents config, not the site config."),
            "{stderr}"
        );
        assert!(stderr.contains("agents.toml (default)"), "{stderr}");
        assert!(!f.path("calls").join("argv").exists());
        let index = audit::read_index(&Recorder::dir_in(&f.env))
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(index.config, combined.to_string_lossy());
        assert!(index.config_exists);
        assert!(
            index.agents_config.ends_with("ccc/agents.toml"),
            "{}",
            index.agents_config
        );
        assert!(!index.agents_config_exists);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cli_run_keeps_routing_notes_on_stderr() {
        let mut f = Fixture::new("cli-run-routing-note");
        f.token("team03");
        f.install_cli_config();
        std::fs::write(
            f.path("agents/explorer.md"),
            "---\nname: explorer\ndescription: Investigates\nharness: claude\nmodel: claude-haiku-4-5\n---\nthe system prompt\n",
        )
        .unwrap();
        f.env.insert("CCC_HARNESS_ALL".into(), "codex".into());
        f.claude_script(OK_RESULT);

        let output = f.cli(&["run", "explorer", "--cwd", &f.work(), "--json", "go"]);
        assert!(output.status.success(), "{output:?}");
        let stdout = String::from_utf8(output.stdout).unwrap();
        let _: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        assert!(!stdout.contains("CCC_HARNESS_ALL") && !stdout.contains("note:"));
        let stderr = String::from_utf8(output.stderr).unwrap();
        let notes: Vec<_> = stderr
            .lines()
            .filter(|line| line.starts_with("note: "))
            .collect();
        assert_eq!(notes.len(), 1, "{stderr}");
        assert!(notes[0].contains("CCC_HARNESS_ALL"), "{stderr}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cli_run_then_resume_admits_the_completed_session_from_a_fresh_process() {
        let mut f = Fixture::new("cli-run-resume-fresh-process");
        f.token("team03");
        f.install_cli_config();
        std::fs::write(
            f.path("agents/explorer.md"),
            "---\nname: explorer\ndescription: Investigates\nharness: claude\nmodel: claude-haiku-4-5\n---\nthe system prompt\n",
        )
        .unwrap();
        f.claude_script(OK_RESULT);

        let first = f.cli(&["run", "explorer", "--cwd", &f.work(), "go"]);
        assert!(first.status.success(), "{first:?}");
        let session = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
        let trace = session.trace.clone().expect("run trace backlink");
        let invocation = session.inv.clone().expect("run invocation backlink");
        let records = audit::read_trace(&Recorder::dir_in(&f.env), &trace).unwrap();
        assert!(
            records.iter().any(|record| matches!(
                record,
                audit::Record::Child(child) if child.inv == invocation
            )),
            "run trace {trace} has no Child record for invocation {invocation}: {records:?}"
        );

        let second = f.cli(&["resume", "9f22dd1b-aaaa", "again"]);
        assert!(second.status.success(), "{second:?}");
        let argv: Vec<String> = f.call("argv").lines().map(String::from).collect();
        let at = argv
            .iter()
            .position(|arg| arg == "--resume")
            .unwrap_or_else(|| panic!("resume did not pass --resume to the harness: {argv:?}"));
        assert_eq!(
            argv.get(at + 1).map(String::as_str),
            Some("9f22dd1b-aaaa"),
            "resume passed the wrong session to the harness: {argv:?}"
        );
        assert_eq!(
            f.call("prompt"),
            "again",
            "resume forwarded the wrong prompt"
        );
        let stdout = String::from_utf8_lossy(&second.stdout);
        assert!(
            stdout.contains("the answer"),
            "resume printed no result: stdout={stdout} stderr={}",
            String::from_utf8_lossy(&second.stderr)
        );
        let resumed = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
        let resumed_trace = resumed.trace.as_deref().expect("resume trace backlink");
        let resumed_inv = resumed.inv.as_deref().expect("resume invocation backlink");
        assert_ne!(
            resumed_inv,
            invocation.as_str(),
            "resume did not mint a new invocation (trace {resumed_trace}, run trace {trace})"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cli_clears_inherited_lineage_and_returns_file_backed_streams() {
        const INNER: &str = "CCC_CLI_FILE_BACKED_INNER";
        const DEADLINE: &str = "CCC_CLI_FILE_BACKED_DEADLINE";
        const VERDICT: &str = "CCC_CLI_FILE_BACKED_VERDICT";
        const TRACE: &str = "11111111-1111-4111-8111-111111111111";
        const PARENT: &str = "22222222-2222-4222-8222-222222222222";

        if std::env::var_os(INNER).is_some() {
            let deadline: u64 = std::env::var(DEADLINE).unwrap().parse().unwrap();
            let verdict = PathBuf::from(std::env::var(VERDICT).unwrap());
            let mut f = Fixture::new("cli-file-backed-inner");
            f.token("team03");
            f.install_cli_config();
            std::fs::write(
                f.path("agents/explorer.md"),
                "---\nname: explorer\ndescription: Investigates\nharness: claude\nmodel: claude-haiku-4-5\n---\nthe system prompt\n",
            )
            .unwrap();
            let env_file = f.path("calls/env");
            f.claude_custom_script(&format!(
                "cat >/dev/null; env > '{}'; printf '%s' '{OK_RESULT}'; printf '%s' 'diagnostic' >&2",
                env_file.display()
            ));
            let output = f
                .spawn_cli_by(
                    "inner",
                    &["run", "explorer", "--cwd", &f.work(), "--json", "go"],
                    deadline,
                )
                .finish_by_boot(deadline + 2);
            assert!(output.status.success(), "{output:?}");
            assert!(String::from_utf8(output.stdout)
                .unwrap()
                .contains("the answer"));
            let env = std::fs::read_to_string(env_file).unwrap();
            assert!(env.contains("HOME="));
            assert!(env.contains("CCC_CONFIG="));
            assert!(env.contains("CCC_AGENTS_CONFIG="));
            assert!(!env.contains(TRACE));
            assert!(!env.contains(PARENT));
            f.env.insert("CCC_HARNESS_ALL".into(), "codex".into());
            let streams = f.cli(&["which", "explorer", "--json"]);
            assert!(streams.status.success());
            assert!(!streams.stdout.is_empty());
            assert!(String::from_utf8(streams.stderr)
                .unwrap()
                .contains("CCC_HARNESS_ALL"));
            std::fs::write(verdict, "ok").unwrap();
            return;
        }

        let outer_dir = TempDir::new("cli-file-backed-outer");
        let deadline = boot_seconds() + 8;
        let verdict = outer_dir.path().join("verdict");
        let stdout = outer_dir.path().join("stdout");
        let stderr = outer_dir.path().join("stderr");
        let stdout_file = std::fs::File::create(&stdout).unwrap();
        let stderr_file = std::fs::File::create(&stderr).unwrap();
        let remaining = deadline.saturating_sub(boot_seconds());
        assert!(
            remaining >= 1,
            "shared boot deadline expired before outer timeout wrapper: deadline={deadline}"
        );
        let mut command = std::process::Command::new("timeout");
        command
            .args(["-k", "1"])
            .arg(remaining.to_string())
            .arg(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "exec::tests::cli_clears_inherited_lineage_and_returns_file_backed_streams",
                "--nocapture",
            ])
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env(INNER, "1")
            .env(DEADLINE, deadline.to_string())
            .env(VERDICT, &verdict)
            .env("CCC_TRACE_ID", TRACE)
            .env("CCC_PARENT_INVOCATION_ID", PARENT)
            .stdout(std::process::Stdio::from(stdout_file))
            .stderr(std::process::Stdio::from(stderr_file));
        let mut child = BoundedChild::spawn(&mut command).unwrap();
        let status = child
            .finish_by_boot((deadline + 2) as f64, boot_seconds_unfloored)
            .unwrap_or_else(|error| {
                let stdout = std::fs::read_to_string(&stdout).unwrap_or_default();
                let stderr = std::fs::read_to_string(&stderr).unwrap_or_default();
                panic!("outer test did not finish: {error}; stdout={stdout}; stderr={stderr}");
            });
        let stdout_text = std::fs::read_to_string(&stdout).unwrap_or_default();
        let stderr_text = std::fs::read_to_string(&stderr).unwrap_or_default();
        assert!(
            status.success(),
            "{status}; stdout={stdout_text}; stderr={stderr_text}"
        );
        assert!(
            verdict.exists(),
            "missing verdict; stdout={stdout_text}; stderr={stderr_text}"
        );
    }

    #[cfg(target_os = "linux")]
    fn trace_snapshot(f: &Fixture, trace: &str) -> Vec<audit::Record> {
        let mut warnings = Vec::new();
        let records =
            audit::read_trace_tolerant(&audit::Recorder::dir_in(&f.env), trace, &mut warnings)
                .unwrap_or_else(|error| {
                    panic!(
                        "reading audit trace {trace} at {}: {error}",
                        audit::Recorder::dir_in(&f.env).display()
                    )
                });
        assert!(
            warnings.is_empty(),
            "audit trace {trace} at {} emitted warnings: {}",
            audit::Recorder::dir_in(&f.env).display(),
            String::from_utf8_lossy(&warnings)
        );
        records
    }

    #[cfg(target_os = "linux")]
    fn run_trace(
        f: &Fixture,
        run: &SpawnedCli,
        deadline: Duration,
    ) -> (String, String, audit::ChildPayload) {
        let limit = Instant::now() + deadline;
        let mut last_index_error: Option<String> = None;
        loop {
            let index_result = audit::read_index(&audit::Recorder::dir_in(&f.env));
            match index_result {
                Ok(index) => {
                    if let Some(entry) = index
                        .iter()
                        .rev()
                        .find(|entry| entry.cmd == audit::Command::Run)
                    {
                        let records = trace_snapshot(f, &entry.trace);
                        if let Some(audit::Record::Child(child)) = records
                            .iter()
                            .find(|record| matches!(record, audit::Record::Child(_)))
                        {
                            return (
                                entry.trace.clone(),
                                entry.inv.clone(),
                                child.payload.clone(),
                            );
                        }
                    }
                }
                Err(error) => last_index_error = Some(error.to_string()),
            }
            assert!(
                Instant::now() < limit,
                "run did not record a child before deadline (wrapper {}): {}",
                run.child.pid(),
                last_index_error
                    .as_deref()
                    .unwrap_or("no index entry was available")
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[cfg(target_os = "linux")]
    fn index_traces(f: &Fixture) -> std::collections::HashSet<String> {
        audit::read_index(&audit::Recorder::dir_in(&f.env))
            .unwrap_or_default()
            .into_iter()
            .map(|entry| entry.trace)
            .collect()
    }

    #[cfg(target_os = "linux")]
    fn run_trace_by(
        f: &Fixture,
        run: &SpawnedCli,
        aggregate_deadline: u64,
        before: &std::collections::HashSet<String>,
    ) -> (String, String, audit::ChildPayload) {
        let started = Instant::now();
        let mut last_index_error: Option<String> = None;
        loop {
            let index_result = audit::read_index(&audit::Recorder::dir_in(&f.env));
            match index_result {
                Ok(index) => {
                    if let Some(entry) = index.iter().find(|entry| {
                        entry.cmd == audit::Command::Run && !before.contains(&entry.trace)
                    }) {
                        let records = trace_snapshot(f, &entry.trace);
                        if let Some(audit::Record::Child(child)) = records
                            .iter()
                            .find(|record| matches!(record, audit::Record::Child(_)))
                        {
                            return (
                                entry.trace.clone(),
                                entry.inv.clone(),
                                child.payload.clone(),
                            );
                        }
                    }
                }
                Err(error) => last_index_error = Some(error.to_string()),
            }
            assert!(
                boot_seconds() < aggregate_deadline,
                "run did not record a child before aggregate deadline (wrapper {}): {}",
                run.child.pid(),
                last_index_error
                    .as_deref()
                    .unwrap_or("no index entry was available")
            );
            assert!(
                started.elapsed() < Duration::from_secs(2),
                "run did not record a child before discovery deadline (wrapper {}): {}",
                run.child.pid(),
                last_index_error
                    .as_deref()
                    .unwrap_or("no index entry was available")
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[cfg(target_os = "linux")]
    struct BodyResult {
        code: i32,
        stdout: String,
        stderr: String,
    }

    #[cfg(target_os = "linux")]
    fn body_result(f: &Fixture, name: &str) -> BodyResult {
        let read = |suffix: &str| {
            std::fs::read_to_string(f.path(&format!("results/{name}.{suffix}")))
                .unwrap_or_else(|error| panic!("reading {name}.{suffix}: {error}"))
        };
        let code_text = read("code");
        let code = code_text
            .trim()
            .parse()
            .unwrap_or_else(|error| panic!("malformed exit code for {name}: {error}"));
        BodyResult {
            code,
            stdout: read("stdout"),
            stderr: read("stderr"),
        }
    }

    #[cfg(target_os = "linux")]
    fn supervised_fixture(
        f: &mut Fixture,
        label: &str,
    ) -> (SpawnedCli, String, String, audit::ChildPayload, ProcId) {
        f.token("team03");
        f.install_cli_config();
        let pid_file = f.path(&format!("{label}-grandchild"));
        f.claude_custom_script(&format!(
            "trap 'exit 0' TERM\nsetsid sleep 10 & echo $! > '{}'\nwait",
            pid_file.display()
        ));
        let run = f.spawn_cli(
            label,
            &["run", "explorer", "--cwd", &f.work(), "--json", "go"],
        );
        let (trace, inv, child) = run_trace(f, &run, Duration::from_secs(2));
        let grandchild_limit = Instant::now() + Duration::from_secs(2);
        let grandchild = loop {
            if let Ok(text) = std::fs::read_to_string(&pid_file) {
                if let Ok(pid) = text.trim().parse() {
                    if let Some(identity) = ProcId::of(pid) {
                        break identity;
                    }
                }
            }
            assert!(
                Instant::now() < grandchild_limit,
                "grandchild marker was not written"
            );
            std::thread::sleep(Duration::from_millis(10));
        };
        (run, trace, inv, child, grandchild)
    }

    #[cfg(target_os = "linux")]
    fn assert_dead(identity: &ProcId) {
        let limit = Instant::now() + Duration::from_secs(2);
        while identity.alive() == crate::procs::Liveness::Alive {
            assert!(
                Instant::now() < limit,
                "process {} remained alive",
                identity.pid
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[cfg(target_os = "linux")]
    fn assert_dead_exact(identity: &ProcId) {
        assert_eq!(
            identity.alive(),
            crate::procs::Liveness::Dead,
            "process {} was not dead",
            identity.pid
        );
    }

    #[cfg(target_os = "linux")]
    fn end_record(f: &Fixture, trace: &str) -> audit::EndRecord {
        trace_snapshot(f, trace)
            .into_iter()
            .find_map(|record| match record {
                audit::Record::End(end) => Some(end),
                _ => None,
            })
            .expect("unique end record")
    }

    #[cfg(target_os = "linux")]
    fn assert_dead_by(identity: &ProcId, aggregate_deadline: u64) {
        while identity.alive() == crate::procs::Liveness::Alive {
            assert!(
                boot_seconds() < aggregate_deadline,
                "process {} remained alive before aggregate deadline",
                identity.pid
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[cfg(target_os = "linux")]
    fn records_until(f: &Fixture, trace: &str, aggregate_deadline: u64) -> Vec<audit::Record> {
        loop {
            let records = trace_snapshot(f, trace);
            if records
                .iter()
                .any(|record| matches!(record, audit::Record::End(_)))
            {
                return records;
            }
            assert!(
                boot_seconds() < aggregate_deadline,
                "trace {trace} did not record an end before aggregate deadline"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn ancestry_authority_b1_real_process_refuses_outside_trace() {
        let _lock = PROCESS_SIGNAL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut f = Fixture::new("authority-b1");
        f.token("team03");
        f.install_cli_config();
        let aggregate_deadline = boot_seconds() + 8;

        f.claude_custom_script(&format!(
            "r=$(({aggregate_deadline} - $(cut -d. -f1 /proc/uptime))); [ \"$r\" -ge 1 ] || exit 0\nexec sleep \"$r\""
        ));
        let before = index_traces(&f);
        let outside = f.spawn_cli_by(
            "outside",
            &["run", "explorer", "--cwd", &f.work(), "--json", "go"],
            aggregate_deadline,
        );
        let (outside_trace, outside_inv, outside_child) =
            run_trace_by(&f, &outside, aggregate_deadline, &before);
        assert_eq!(outside_child.pgid, outside_child.proc.pid);
        assert_eq!(
            outside_child.supervisor.alive(),
            crate::procs::Liveness::Alive
        );
        assert_eq!(outside_child.proc.alive(), crate::procs::Liveness::Alive);

        let body_path = f.path("authority-body");
        std::fs::create_dir_all(f.path("results")).unwrap();
        let path_value = |path: &Path| shell_literal(&path.display().to_string());
        let common_env = format!(
            "env -i PATH={} HOME={} XDG_STATE_HOME={} {} CCC_CLAUDE_BIN={}",
            shell_literal(&std::env::var("PATH").unwrap()),
            path_value(Path::new(&f.env["HOME"])),
            path_value(Path::new(&f.env["XDG_STATE_HOME"])),
            f.config_env_literals(),
            path_value(Path::new(&f.env["CCC_CLAUDE_BIN"])),
        );
        let command = |name: &str, args: &[&str], extra: Option<String>| {
            let prefix = extra.map_or_else(
                || common_env.clone(),
                |extra| format!("{common_env} {extra}"),
            );
            let args = args
                .iter()
                .map(|arg| shell_literal(arg))
                .collect::<Vec<_>>()
                .join(" ");
            let stdout = path_value(&f.path(&format!("results/{name}.stdout")));
            let stderr = path_value(&f.path(&format!("results/{name}.stderr")));
            let code = path_value(&f.path(&format!("results/{name}.code")));
            format!(
                "{prefix} {} {args} >{stdout} 2>{stderr}; code=$?; printf '%s' \"$code\" >{code};",
                path_value(&crate::testutil::ccc_binary()),
            )
        };
        let spoof = format!(
            "CCC_TRACE_ID={} CCC_PARENT_INVOCATION_ID={}",
            shell_literal(&outside_trace),
            shell_literal(&outside_inv)
        );
        let body = format!(
            "{}\n{}\n{}\n{}\n{}\nprintf '%s' '{OK_RESULT}'",
            command("status-absent", &["status", "--json"], None),
            command("kill-absent", &["kill", &outside_inv], None),
            command("status-trace", &["status", "--trace", &outside_trace], None,),
            command("kill-spoof", &["kill", &outside_inv], Some(spoof)),
            command("kill-all", &["kill", "--all"], None),
        );
        f.write_executable(&body_path, &body);
        f.claude_custom_script(&format!(
            "r=$((({aggregate_deadline} - 2) - $(cut -d. -f1 /proc/uptime))); [ \"$r\" -ge 1 ] || exit 99\nexec timeout -k 1 \"$r\" sh {}",
            path_value(&body_path)
        ));

        let before_outer = index_traces(&f);
        let outer = f.spawn_cli_by(
            "outer",
            &["run", "explorer", "--cwd", &f.work(), "--json", "go"],
            aggregate_deadline,
        );
        let (outer_trace, outer_inv, outer_child) =
            run_trace_by(&f, &outer, aggregate_deadline, &before_outer);
        assert_eq!(outer_child.pgid, outer_child.proc.pid);
        let outer_output = outer.finish_by_boot(aggregate_deadline + 2);
        assert_eq!(outer_output.status.code(), Some(0), "{outer_output:?}");
        let outer_records = records_until(&f, &outer_trace, aggregate_deadline + 2);
        let outer_ends = outer_records
            .iter()
            .filter_map(|record| match record {
                audit::Record::End(end) => Some(end),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(outer_ends.len(), 1);
        assert_eq!(
            outer_ends[0].payload.exit,
            Some(0),
            "end={:?}",
            outer_ends[0],
        );
        assert!(outer_records
            .iter()
            .all(|record| !matches!(record, audit::Record::Signal(_))));

        let status_absent = body_result(&f, "status-absent");
        assert_eq!(status_absent.code, 0);
        let status_json: serde_json::Value = serde_json::from_str(&status_absent.stdout).unwrap();
        assert_eq!(status_json["scope"]["kind"], "subtree");
        assert_eq!(status_json["scope"]["source"], "ancestry");
        assert_eq!(status_json["scope"]["trace"], outer_trace);
        assert_eq!(status_json["scope"]["root"], outer_inv);
        assert_eq!(status_json["invocations"], serde_json::json!([]));
        assert!(status_absent.stderr.contains(&format!(
            "ccc: lineage recovered from process ancestry (invocation {outer_inv}); inherited CCC_* variables were absent"
        )));

        let kill_absent = body_result(&f, "kill-absent");
        assert_eq!(kill_absent.code, 2);
        assert!(kill_absent.stderr.contains(&format!(
            "ccc: {outside_inv} is not a subordinate of this agent"
        )));
        assert!(!kill_absent.stdout.contains("signalled"));

        let status_trace = body_result(&f, "status-trace");
        assert_eq!(status_trace.code, 2);
        assert!(status_trace.stdout.is_empty());
        assert!(status_trace
            .stderr
            .contains("--trace is not available to a spawned agent"));
        assert!(status_trace
            .stderr
            .contains("ccc status shows your own subagents"));

        let kill_spoof = body_result(&f, "kill-spoof");
        assert_eq!(kill_spoof.code, 2);
        assert!(kill_spoof.stderr.contains(&format!(
            "ccc: lineage recovered from process ancestry (invocation {outer_inv}); inherited CCC_* variables were inconsistent"
        )));
        assert!(kill_spoof.stderr.contains(&format!(
            "ccc: {outside_inv} is not a subordinate of this agent"
        )));

        let kill_all = body_result(&f, "kill-all");
        assert_eq!(kill_all.code, 0);
        assert_eq!(kill_all.stdout, "");

        let outside_records = trace_snapshot(&f, &outside_trace);
        assert!(outside_records.iter().any(|record| matches!(record,
            audit::Record::Start(start) if start.inv == outside_inv
        )));
        assert!(outside_records.iter().any(|record| matches!(record,
            audit::Record::Child(child) if child.inv == outside_inv
        )));
        assert!(outside_records
            .iter()
            .all(|record| !matches!(record, audit::Record::Signal(_))));
        assert_eq!(
            outside_child.supervisor.alive(),
            crate::procs::Liveness::Alive
        );
        assert_eq!(outside_child.proc.alive(), crate::procs::Liveness::Alive);

        assert_eq!(outside_child.pgid, outside_child.proc.pid);
        crate::procs::signal_group(outside_child.pgid, libc::SIGKILL).unwrap();
        assert_dead_by(&outside_child.proc, aggregate_deadline + 2);
        assert_dead_by(&outside_child.supervisor, aggregate_deadline + 2);
        let outside_output = outside.finish_by_boot(aggregate_deadline + 2);
        assert_ne!(outside_output.status.code(), Some(0));
    }

    #[cfg(target_os = "linux")]
    // TODO(test-no-acceptance-for-an-authorised-agent-scoped-kill): design a bounded real-process case proving an ancestry-scoped agent can kill its own genuine subordinate
    #[test]
    fn ancestry_authority_b2_nested_separate_traces_anchor_at_nearest_harness() {
        let _lock = PROCESS_SIGNAL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut f = Fixture::new("authority-b2");
        f.token("team03");
        f.install_cli_config();
        let aggregate_deadline = boot_seconds() + 8;
        let path_value = |path: &Path| shell_literal(&path.display().to_string());
        let binary = path_value(&crate::testutil::ccc_binary());
        let common_env = format!(
            "PATH={} HOME={} XDG_STATE_HOME={} {}",
            shell_literal(&std::env::var("PATH").unwrap()),
            path_value(Path::new(&f.env["HOME"])),
            path_value(Path::new(&f.env["XDG_STATE_HOME"])),
            f.config_env_literals(),
        );
        std::fs::create_dir_all(f.path("results")).unwrap();

        let outside_fake = f.path("b2-outside-fake");
        f.write_executable(
            &outside_fake,
            &format!(
                "r=$(({aggregate_deadline} - $(cut -d. -f1 /proc/uptime))); [ \"$r\" -ge 1 ] || exit 0\nsleep \"$r\"\nprintf '%s' '{OK_RESULT}'"
            ),
        );
        f.env
            .insert("CCC_CLAUDE_BIN".into(), outside_fake.display().to_string());
        let before = index_traces(&f);
        let outside = f.spawn_cli_by(
            "b2-outside",
            &["run", "explorer", "--cwd", &f.work(), "--json", "go"],
            aggregate_deadline,
        );
        let (trace_a, outside_inv, outside_child) =
            run_trace_by(&f, &outside, aggregate_deadline, &before);
        assert_eq!(outside_child.pgid, outside_child.proc.pid);

        let background_fake = f.path("b2-background-fake");
        f.write_executable(
            &background_fake,
            &format!(
                "r=$(({aggregate_deadline} - $(cut -d. -f1 /proc/uptime))); [ \"$r\" -ge 1 ] || exit 0\nsleep \"$r\"\nprintf '%s' '{OK_RESULT}'"
            ),
        );
        let inner_fake = f.path("b2-inner-fake");
        let inner_body = f.path("b2-inner-body");
        let outer_body = f.path("b2-outer-body");

        let result_path =
            |name: &str, suffix: &str| path_value(&f.path(&format!("results/{name}.{suffix}")));
        let inner_root_result = path_value(&f.path("b2-inner-root"));
        let record_command_raw = |name: &str, env: String, args: String| {
            format!(
                "r=$(({aggregate_deadline} - $(cut -d. -f1 /proc/uptime))); [ \"$r\" -ge 1 ] || exit 98; env -i {common_env} {env} timeout -k 1 \"$r\" {binary} {args} >{} 2>{}; code=$?; printf '%s' \"$code\" >{};",
                result_path(name, "stdout"),
                result_path(name, "stderr"),
                result_path(name, "code"),
            )
        };
        let record_command = |name: &str, env: String, args: &[&str]| {
            record_command_raw(
                name,
                env,
                args.iter()
                    .map(|arg| shell_literal(arg))
                    .collect::<Vec<_>>()
                    .join(" "),
            )
        };
        let inner_common = format!("CCC_CLAUDE_BIN={}", path_value(&inner_fake));
        let spoof_env = "CCC_TRACE_ID=\"$trace_a\" CCC_PARENT_INVOCATION_ID=\"$outer_inv\"";
        let background_target = "\"$background_inv\"";

        let inner_status_spoof = record_command(
            "b2-inner-status-spoof",
            format!("{} {}", inner_common, spoof_env),
            &["status", "--json"],
        );
        let inner_kill_spoof = record_command_raw(
            "b2-inner-kill-spoof",
            format!("{} {}", inner_common, spoof_env),
            format!("kill {background_target}"),
        );
        let inner_all_spoof = record_command(
            "b2-inner-all-spoof",
            format!("{} {}", inner_common, spoof_env),
            &["kill", "--all"],
        );
        let inner_status_agree = record_command(
            "b2-inner-status-agree",
            format!(
                "{} CCC_TRACE_ID=\"$inner_trace\" CCC_PARENT_INVOCATION_ID=\"$inner_root\"",
                inner_common
            ),
            &["status", "--json"],
        );
        let inner_kill_agree = record_command_raw(
            "b2-inner-kill-agree",
            format!(
                "{} CCC_TRACE_ID=\"$inner_trace\" CCC_PARENT_INVOCATION_ID=\"$inner_root\"",
                inner_common
            ),
            format!("kill {background_target}"),
        );
        let inner_all_agree = record_command(
            "b2-inner-all-agree",
            format!(
                "{} CCC_TRACE_ID=\"$inner_trace\" CCC_PARENT_INVOCATION_ID=\"$inner_root\"",
                inner_common
            ),
            &["kill", "--all"],
        );
        let inner_body_text = format!(
            "trace_a=$(cat {})\nouter_inv=$(cat {})\nbackground_inv=$(cat {})\ninner_trace=\"$INNER_TRACE_ID\"\ninner_root=\"$INNER_ROOT_ID\"\n[ -n \"$trace_a\" ] && [ -n \"$outer_inv\" ] && [ -n \"$background_inv\" ] && [ -n \"$inner_trace\" ] && [ -n \"$inner_root\" ] || exit 97\nprintf '%s' \"$inner_root\" >{}\n{}\n{}\n{}\n{}\n{}\n{}\nprintf '%s' '{OK_RESULT}'",
            path_value(&f.path("b2-trace-a")),
            path_value(&f.path("b2-outer-inv")),
            path_value(&f.path("b2-background-inv")),
            inner_root_result,
            inner_status_spoof,
            inner_kill_spoof,
            inner_all_spoof,
            inner_status_agree,
            inner_kill_agree,
            inner_all_agree,
        );
        f.write_executable(&inner_body, &inner_body_text);
        f.write_executable(
            &inner_fake,
            &format!(
                "r=$((({aggregate_deadline} - 2) - $(cut -d. -f1 /proc/uptime))); [ \"$r\" -ge 1 ] || exit 99\nexec timeout -k 1 \"$r\" env INNER_TRACE_ID=\"$CCC_TRACE_ID\" INNER_ROOT_ID=\"$CCC_PARENT_INVOCATION_ID\" sh {}",
                path_value(&inner_body)
            ),
        );

        let background_command = format!(
            "(r=$(({aggregate_deadline} - $(cut -d. -f1 /proc/uptime))); [ \"$r\" -ge 1 ] || exit 98; CCC_CLAUDE_BIN={} timeout -k 1 \"$r\" {binary} run explorer --cwd {} --json go >{} 2>{}; code=$?; printf '%s' \"$code\" >{} ) &",
            path_value(&background_fake),
            path_value(Path::new(&f.work())),
            result_path("b2-background", "stdout"),
            result_path("b2-background", "stderr"),
            result_path("b2-background", "code"),
        );
        let outer_status_probe = format!(
            "r=$(({aggregate_deadline} - $(cut -d. -f1 /proc/uptime))); [ \"$r\" -ge 1 ] || exit 98; CCC_CLAUDE_BIN={} timeout -k 1 \"$r\" {binary} status --json >{} 2>{}; code=$?; printf '%s' \"$code\" >{};",
            path_value(&background_fake),
            result_path("b2-outer-probe", "stdout"),
            result_path("b2-outer-probe", "stderr"),
            result_path("b2-outer-probe", "code"),
        );
        let foreground = format!(
            "r=$(({aggregate_deadline} - $(cut -d. -f1 /proc/uptime))); [ \"$r\" -ge 1 ] || exit 98; env -i {common_env} CCC_CLAUDE_BIN={} timeout -k 1 \"$r\" {binary} run explorer --cwd {} --json go >{} 2>{}; code=$?; printf '%s' \"$code\" >{};",
            path_value(&inner_fake),
            path_value(Path::new(&f.work())),
            result_path("b2-foreground", "stdout"),
            result_path("b2-foreground", "stderr"),
            result_path("b2-foreground", "code"),
        );
        let final_status = format!(
            "r=$(({aggregate_deadline} - $(cut -d. -f1 /proc/uptime))); [ \"$r\" -ge 1 ] || exit 98; CCC_CLAUDE_BIN={} timeout -k 1 \"$r\" {binary} status --json >{} 2>{}; code=$?; printf '%s' \"$code\" >{};",
            path_value(&background_fake),
            result_path("b2-outer-final", "stdout"),
            result_path("b2-outer-final", "stderr"),
            result_path("b2-outer-final", "code"),
        );
        let write_ids = format!(
            "printf '%s' \"$trace_a\" >{}\nprintf '%s' \"$outer_inv\" >{}\nprintf '%s' \"$background_inv\" >{}\nprintf '%s' \"$inner_root\" >{}",
            path_value(&f.path("b2-trace-a")),
            path_value(&f.path("b2-outer-inv")),
            path_value(&f.path("b2-background-inv")),
            inner_root_result,
        );
        let outer_body_text = format!(
            "{}\nfor n in $(seq 1 200); do {}\n  if [ -s {} ] && grep -q '\\\"inv\\\":' {}; then break; fi\n  sleep 0.01\ndone\ntrace_a=$(sed -n 's/.*\\\"trace\\\":\\\"\\([^\\\"]*\\)\\\".*/\\1/p' {})\nouter_inv=$(sed -n 's/.*\\\"root\\\":\\\"\\([^\\\"]*\\)\\\".*/\\1/p' {})\nbackground_inv=$(sed -n 's/.*\\\"inv\\\":\\\"\\([^\\\"]*\\)\\\".*/\\1/p' {})\n[ -n \"$trace_a\" ] && [ -n \"$outer_inv\" ] && [ -n \"$background_inv\" ] || exit 96\n{}\n{}\n{}\n{}\n{}\nprintf '%s' '{OK_RESULT}'",
            background_command,
            outer_status_probe,
            result_path("b2-outer-probe", "stdout"),
            result_path("b2-outer-probe", "stdout"),
            result_path("b2-outer-probe", "stdout"),
            result_path("b2-outer-probe", "stdout"),
            result_path("b2-outer-probe", "stdout"),
            write_ids,
            foreground,
            final_status,
            result_path("b2-outer-final", "stdout"),
            result_path("b2-outer-final", "stdout"),
        );
        f.write_executable(&outer_body, &outer_body_text);
        let outer_fake = f.path("b2-outer-fake");
        f.write_executable(
            &outer_fake,
            &format!(
            "r=$((({aggregate_deadline} - 2) - $(cut -d. -f1 /proc/uptime))); [ \"$r\" -ge 1 ] || exit 99\nexec timeout -k 1 \"$r\" sh {}",
                path_value(&outer_body)
            ),
        );
        f.env
            .insert("CCC_CLAUDE_BIN".into(), outer_fake.display().to_string());
        let before_outer = index_traces(&f);
        let outer = f.spawn_cli_by(
            "b2-outer",
            &["run", "explorer", "--cwd", &f.work(), "--json", "go"],
            aggregate_deadline,
        );
        let (trace_outer, outer_inv, outer_child) =
            run_trace_by(&f, &outer, aggregate_deadline, &before_outer);
        assert_eq!(outer_child.pgid, outer_child.proc.pid);
        let outer_output = outer.finish_by_boot(aggregate_deadline + 2);
        assert_eq!(
            outer_output.status.code(),
            Some(0),
            "{outer_output:?}; body={}; probe={}; probe-err={}; probe-code={}; bg={}; bg-err={}; bg-code={}",
            std::fs::read_to_string(&outer_body).unwrap_or_default(),
            std::fs::read_to_string(f.path("results/b2-outer-probe.stdout")).unwrap_or_default(),
            std::fs::read_to_string(f.path("results/b2-outer-probe.stderr")).unwrap_or_default(),
            std::fs::read_to_string(f.path("results/b2-outer-probe.code")).unwrap_or_default(),
            std::fs::read_to_string(f.path("results/b2-background.stdout")).unwrap_or_default(),
            std::fs::read_to_string(f.path("results/b2-background.stderr")).unwrap_or_default(),
            std::fs::read_to_string(f.path("results/b2-background.code")).unwrap_or_default(),
        );
        let outer_records = records_until(&f, &trace_outer, aggregate_deadline + 2);
        assert_eq!(
            outer_records
                .iter()
                .filter(|r| matches!(r, audit::Record::End(_)))
                .count(),
            1
        );
        assert!(outer_records
            .iter()
            .all(|r| !matches!(r, audit::Record::Signal(_))));
        let outer_end = outer_records
            .iter()
            .find_map(|record| match record {
                audit::Record::End(end) => Some(end),
                _ => None,
            })
            .unwrap();
        assert_eq!(outer_end.payload.exit, Some(0));
        let background_inv = std::fs::read_to_string(f.path("b2-background-inv"))
            .unwrap()
            .trim()
            .to_string();
        assert_eq!(body_result(&f, "b2-foreground").code, 0);
        let final_status = body_result(&f, "b2-outer-final");
        assert_eq!(final_status.code, 0);
        let final_json: serde_json::Value = serde_json::from_str(&final_status.stdout).unwrap();
        assert_eq!(final_json["scope"]["trace"], trace_outer);
        assert!(final_json["invocations"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["inv"] == background_inv && row["state"] == "running"));

        let inner_traces = audit::read_index(&audit::Recorder::dir_in(&f.env)).unwrap();
        let trace_b = inner_traces
            .iter()
            .filter(|entry| {
                entry.cmd == audit::Command::Run
                    && entry.trace != trace_a
                    && entry.trace != trace_outer
            })
            .map(|entry| entry.trace.clone())
            .next()
            .expect("inner trace catalog entry");
        let expected_inner_root = std::fs::read_to_string(f.path("b2-inner-root"))
            .unwrap()
            .trim()
            .to_string();
        let mut observed_inner_roots = Vec::new();

        for (name, diagnostic) in [
            ("b2-inner-status-spoof", true),
            ("b2-inner-status-agree", false),
        ] {
            let result = body_result(&f, name);
            assert_eq!(result.code, 0);
            let json: serde_json::Value = serde_json::from_str(&result.stdout).unwrap();
            assert_eq!(json["scope"]["kind"], "subtree");
            assert_eq!(json["scope"]["source"], "ancestry");
            assert_eq!(json["scope"]["trace"], trace_b);
            let observed_root = json["scope"]["root"].as_str().expect("scope root");
            assert_eq!(observed_root, expected_inner_root);
            observed_inner_roots.push(observed_root.to_owned());
            assert!(json["invocations"].as_array().unwrap().iter().all(|row| {
                row["inv"] != outer_inv && row["inv"] != background_inv && row["inv"] != outside_inv
            }));
            assert_eq!(
                result
                    .stderr
                    .contains("inherited CCC_* variables were inconsistent"),
                diagnostic
            );
        }
        for name in ["b2-inner-kill-spoof", "b2-inner-kill-agree"] {
            let result = body_result(&f, name);
            assert_eq!(result.code, 2);
            assert!(result.stderr.contains("not a subordinate"));
            assert!(!result.stdout.contains("signalled"));
        }
        for name in ["b2-inner-all-spoof", "b2-inner-all-agree"] {
            let result = body_result(&f, name);
            assert_eq!(result.code, 0);
            assert!(result.stdout.is_empty());
        }
        assert_eq!(observed_inner_roots[0], observed_inner_roots[1]);
        let inner_root = expected_inner_root;
        let inner_records = records_until(&f, &trace_b, aggregate_deadline + 2);
        let inner_start = inner_records.iter().find_map(|record| match record {
            audit::Record::Start(start) if start.inv == inner_root => Some(start),
            _ => None,
        });
        assert!(inner_start.is_some_and(|start| start.parent.is_none()));
        assert_eq!(
            inner_records
                .iter()
                .filter(|r| matches!(r, audit::Record::End(_)))
                .count(),
            1
        );
        let inner_end = inner_records
            .iter()
            .find_map(|record| match record {
                audit::Record::End(end) => Some(end),
                _ => None,
            })
            .unwrap();
        assert_eq!(inner_end.payload.exit, Some(0));
        let trace_a_records = trace_snapshot(&f, &trace_a);
        assert!(trace_a_records.iter().any(
            |record| matches!(record, audit::Record::Start(start) if start.inv == outside_inv)
        ));
        assert!(trace_a_records.iter().any(
            |record| matches!(record, audit::Record::Child(child) if child.inv == outside_inv)
        ));
        assert!(trace_a_records
            .iter()
            .all(|r| !matches!(r, audit::Record::Signal(_))));
        let trace_b_records = trace_snapshot(&f, &trace_b);
        assert!(trace_b_records.iter().any(
            |record| matches!(record, audit::Record::Start(start) if start.inv == inner_root)
        ));
        assert!(trace_b_records
            .iter()
            .any(|record| matches!(record, audit::Record::End(end) if end.inv == inner_root)));
        assert!(trace_b_records
            .iter()
            .all(|r| !matches!(r, audit::Record::Signal(_))));

        let background_child = outer_records
            .iter()
            .find_map(|record| match record {
                audit::Record::Child(child) if child.inv == background_inv => {
                    Some(child.payload.clone())
                }
                _ => None,
            })
            .expect("background child record");
        assert_eq!(background_child.pgid, background_child.proc.pid);
        assert_eq!(background_child.proc.alive(), crate::procs::Liveness::Alive);
        assert_eq!(
            background_child.supervisor.alive(),
            crate::procs::Liveness::Alive
        );
        crate::procs::signal_group(background_child.pgid, libc::SIGKILL).unwrap();
        assert_dead_by(&background_child.proc, aggregate_deadline + 2);
        assert_dead_by(&background_child.supervisor, aggregate_deadline + 2);

        assert_eq!(outside_child.proc.alive(), crate::procs::Liveness::Alive);
        crate::procs::signal_group(outside_child.pgid, libc::SIGKILL).unwrap();
        assert_dead_by(&outside_child.proc, aggregate_deadline + 2);
        assert_dead_by(&outside_child.supervisor, aggregate_deadline + 2);
        let outside_output = outside.finish_by_boot(aggregate_deadline + 2);
        assert_ne!(outside_output.status.code(), Some(0));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn kill_acceptance_waited_and_no_wait_use_fresh_supervised_runs() {
        let _lock = PROCESS_SIGNAL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for no_wait in [false, true] {
            let mut f = Fixture::new(if no_wait { "kill-no-wait" } else { "kill-wait" });
            let (mut run, trace, inv, child, grandchild) =
                supervised_fixture(&mut f, if no_wait { "no-wait" } else { "wait" });
            let started = Instant::now();
            let args = if no_wait {
                vec!["kill", "--no-wait", inv.as_str()]
            } else {
                vec!["kill", inv.as_str()]
            };
            let output = f.cli(&args);
            let elapsed = started.elapsed();
            assert!(output.status.success(), "{output:?}");
            assert!(
                elapsed
                    < if no_wait {
                        Duration::from_secs(1)
                    } else {
                        Duration::from_secs(3)
                    }
            );
            if no_wait {
                assert!(String::from_utf8_lossy(&output.stdout).contains("signalled"));
                let signals: Vec<_> = trace_snapshot(&f, &trace)
                    .into_iter()
                    .filter_map(|record| match record {
                        audit::Record::Signal(signal)
                            if signal.payload.reason == "ccc kill"
                                && matches!(
                                    signal.payload.to,
                                    audit::SignalDestination::Supervisor
                                ) =>
                        {
                            Some(signal)
                        }
                        _ => None,
                    })
                    .collect();
                assert_eq!(signals.len(), 1);
                assert_eq!(signals[0].payload.signal, libc::SIGTERM);
                assert!(signals[0].payload.by_inv.is_none());
                assert_ne!(signals[0].payload.by.pid, child.supervisor.pid);
                assert_ne!(signals[0].payload.by.pid, child.proc.pid);
                assert_ne!(signals[0].payload.by.pid, grandchild.pid);
            } else {
                assert_dead_exact(&child.supervisor);
                assert_dead_exact(&child.proc);
                assert_dead_exact(&grandchild);
            }
            let run_status = run
                .child
                .finish_by(Instant::now() + Duration::from_secs(2))
                .unwrap();
            assert_eq!(run_status.code(), Some(3));
            let end = end_record(&f, &trace);
            assert_eq!(end.payload.exit, Some(3));
            assert_eq!(end.payload.signal, None);
            let records = trace_snapshot(&f, &trace);
            assert_eq!(
                records
                    .iter()
                    .filter(|r| matches!(r, audit::Record::End(_)))
                    .count(),
                1
            );
            if !no_wait {
                let signals: Vec<_> = records
                    .iter()
                    .filter_map(|record| match record {
                        audit::Record::Signal(signal)
                            if signal.payload.reason == "ccc kill"
                                && matches!(
                                    signal.payload.to,
                                    audit::SignalDestination::Supervisor
                                ) =>
                        {
                            Some(signal)
                        }
                        _ => None,
                    })
                    .collect();
                assert_eq!(signals.len(), 1);
                assert_eq!(signals[0].payload.signal, libc::SIGTERM);
                assert!(signals[0].payload.by_inv.is_none());
                assert_ne!(signals[0].payload.by.pid, child.supervisor.pid);
                assert_ne!(signals[0].payload.by.pid, child.proc.pid);
                assert_ne!(signals[0].payload.by.pid, grandchild.pid);
                let expected_session = records.iter().find_map(|record| match record {
                    audit::Record::Spawn(spawn) => spawn.payload.session.as_deref(),
                    _ => None,
                });
                assert_eq!(end.payload.session.as_deref(), expected_session);
                let error = end.payload.error.as_deref().unwrap_or_default();
                assert!(error.contains(&format!("sender pid {}", signals[0].payload.by.pid)));
                assert!(error.contains("swept 1 descendants"));
            }
            if no_wait {
                assert_dead(&child.supervisor);
                assert_dead(&child.proc);
                assert_dead(&grandchild);
            }
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn kill_acceptance_now_kills_supervisor_and_descendant() {
        let _lock = PROCESS_SIGNAL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut f = Fixture::new("kill-now");
        let (mut run, trace, inv, child, grandchild) = supervised_fixture(&mut f, "now");
        let started = Instant::now();
        let output = f.cli(&["kill", "--now", &inv]);
        assert!(output.status.success(), "{output:?}");
        assert!(started.elapsed() < Duration::from_millis(5200));
        assert_dead_exact(&child.supervisor);
        assert_dead_exact(&child.proc);
        assert_dead_exact(&grandchild);
        let status = run
            .child
            .finish_by(Instant::now() + Duration::from_secs(2))
            .unwrap();
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(status.signal(), Some(libc::SIGKILL));
        let end = end_record(&f, &trace);
        assert_eq!(end.payload.exit, Some(3));
        assert_eq!(end.payload.signal, None);
        assert!(end
            .payload
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("ccc kill --now"));
        assert!(end
            .payload
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("1 descendants swept"));
        let expected_session =
            trace_snapshot(&f, &trace)
                .into_iter()
                .find_map(|record| match record {
                    audit::Record::Spawn(spawn) => spawn.payload.session,
                    _ => None,
                });
        assert_eq!(end.payload.session.as_deref(), expected_session.as_deref());
        let signals = trace_snapshot(&f, &trace)
            .into_iter()
            .filter_map(|record| match record {
                audit::Record::Signal(signal) if signal.payload.reason == "ccc kill --now" => {
                    Some(signal)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(signals.len(), 2);
        assert!(signals.iter().any(|signal| matches!(
            signal.payload.to,
            audit::SignalDestination::Supervisor
        ) && signal.payload.signal == libc::SIGKILL));
        assert!(signals.iter().any(|signal| matches!(
            signal.payload.to,
            audit::SignalDestination::Child
        ) && signal.payload.signal == libc::SIGKILL
            && signal.payload.by.pid != child.supervisor.pid
            && signal.payload.by.pid != child.proc.pid
            && signal.payload.by.pid != grandchild.pid));
        assert!(signals.iter().all(|signal| {
            signal.payload.by.pid != child.supervisor.pid
                && signal.payload.by.pid != child.proc.pid
                && signal.payload.by.pid != grandchild.pid
        }));
        assert_dead(&child.supervisor);
        assert_dead(&child.proc);
        assert_dead(&grandchild);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn kill_acceptance_grace_cleans_real_orphan_group() {
        let _lock = PROCESS_SIGNAL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut f = Fixture::new("kill-orphan");
        f.token("team03");
        f.install_cli_config();
        let pid_file = f.path("orphan-grandchild");
        let script = f.path("orphan-harness");
        f.write_executable(
            &script,
            &format!(
                "#!/bin/sh\ntrap '' TERM\nsetsid sleep 10 & echo $! > '{}'\nwait\n",
                pid_file.display()
            ),
        );
        let stdout = std::fs::File::create(f.path("orphan.stdout")).unwrap();
        let stderr = std::fs::File::create(f.path("orphan.stderr")).unwrap();
        let mut command = std::process::Command::new(&script);
        command
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap())
            .env("HOME", &f.env["HOME"])
            .stdout(std::process::Stdio::from(stdout))
            .stderr(std::process::Stdio::from(stderr));
        let mut harness = BoundedChild::spawn(&mut command).unwrap();
        let harness_identity = ProcId::of(harness.pid()).unwrap();
        let process_group = unsafe { libc::getpgid(harness.pid() as libc::pid_t) };
        assert_eq!(process_group, harness.pid() as libc::pid_t);
        let marker_limit = Instant::now() + Duration::from_secs(2);
        let grandchild = loop {
            if let Ok(text) = std::fs::read_to_string(&pid_file) {
                if let Ok(pid) = text.trim().parse() {
                    if let Some(identity) = ProcId::of(pid) {
                        break identity;
                    }
                }
            }
            assert!(
                Instant::now() < marker_limit,
                "orphan grandchild marker was not written"
            );
            std::thread::sleep(Duration::from_millis(10));
        };
        let mut supervisor = std::process::Command::new("true").spawn().unwrap();
        let supervisor_pid = supervisor.id();
        let supervisor_identity = ProcId::of(supervisor_pid).unwrap();
        supervisor.wait().unwrap();
        let recorder = audit::Recorder::new(&f.env).unwrap();
        let trace = recorder.lineage().trace_id.clone();
        let inv = recorder.lineage().invocation_id.clone();
        recorder
            .start(StartPayload {
                pid: supervisor_identity.pid,
                cmd: audit::Command::Run,
                agent: Some("explorer".into()),
                session: None,
                cwd_arg: Some(f.work()),
                profile_arg: None,
                harness_arg: Some(Harness::Claude),
                json: true,
                prompt_source: Some(PromptSource::Arg),
                prompt_file: None,
                prompt: Some("go".into()),
                passthrough: vec![],
                fork: false,
            })
            .unwrap();
        recorder
            .child(audit::ChildPayload {
                proc: harness_identity.clone(),
                pgid: harness_identity.pid,
                supervisor: supervisor_identity.clone(),
            })
            .unwrap();
        recorder.session("orphan-session").unwrap();
        let started = Instant::now();
        let output = f.cli(&["kill", "--grace", "0.2", &inv]);
        assert!(output.status.success(), "{output:?}");
        assert!(started.elapsed() < Duration::from_millis(5200));
        assert_dead_exact(&harness_identity);
        assert_dead_exact(&grandchild);
        harness
            .finish_by(Instant::now() + Duration::from_secs(2))
            .unwrap();
        assert_dead(&harness_identity);
        assert_dead(&grandchild);
        let end = end_record(&f, &trace);
        assert_eq!(end.payload.session.as_deref(), Some("orphan-session"));
        assert_eq!(end.payload.exit, Some(3));
        assert_eq!(end.payload.signal, None);
        assert!(end
            .payload
            .error
            .as_deref()
            .unwrap_or_default()
            .contains("1 descendants swept"));
        assert!(end
            .payload
            .error
            .as_deref()
            .unwrap_or_default()
            .contains(&format!("supervisor pid {}", supervisor_identity.pid)));
        let signals = trace_snapshot(&f, &trace)
            .into_iter()
            .filter_map(|record| match record {
                audit::Record::Signal(signal) => Some(signal),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(signals.len(), 2);
        assert_eq!(signals[0].payload.signal, libc::SIGTERM);
        assert_eq!(signals[0].payload.reason, "ccc kill");
        assert_eq!(signals[1].payload.signal, libc::SIGKILL);
        assert_eq!(signals[1].payload.reason, "grace expired");
        assert!(signals.iter().all(|signal| {
            signal.payload.by.pid != harness_identity.pid
                && signal.payload.by.pid != supervisor_identity.pid
                && signal.payload.by.pid != grandchild.pid
        }));
    }

    #[test]
    fn a_reported_failure_prints_the_message_and_a_resolvable_session_id() {
        let err = HarnessError::Reported {
            session_id: Some("9f22dd1b".into()),
            message: "5-hour limit reached".into(),
            native: serde_json::json!({"num_turns": 1}),
            warnings: vec![],
        };
        let mut out: Vec<u8> = Vec::new();
        let code = fail(
            &resolved(),
            std::path::Path::new("/srv"),
            Some(&account()),
            err,
            None,
            false,
            &mut out,
        );
        assert_eq!(code, crate::harness::EXIT_HARNESS_FAILED);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("5-hour limit reached"), "{text}");
        assert!(
            text.contains("[ccc] session=9f22dd1b harness=claude"),
            "{text}"
        );
        assert!(text.contains("profile=team03"), "{text}");
    }

    #[test]
    fn a_limited_failure_is_exit_4_and_prints_a_resumable_session_id() {
        let err = HarnessError::Limited {
            session_id: Some("9f22dd1b".into()),
            message: LIMITED_MESSAGE.into(),
            native: serde_json::json!({"num_turns": 1}),
            warnings: vec![],
            limit: None,
        };
        let mut out: Vec<u8> = Vec::new();
        let code = fail(
            &resolved(),
            std::path::Path::new("/srv"),
            Some(&account()),
            err,
            Some(NotContinued::Declined(format!(
                "team03 hit a usage limit: {LIMITED_MESSAGE}; {NO_OTHER_PROFILE}"
            ))),
            false,
            &mut out,
        );
        assert_eq!(code, crate::harness::EXIT_LIMITED);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains(LIMITED_MESSAGE), "{text}");
        assert!(text.contains("[ccc] session=9f22dd1b"), "{text}");
        assert!(text.contains("profile=team03"), "{text}");
    }

    const LIMITED_MESSAGE: &str = "You've hit your session limit · resets 3pm (America/New_York)";

    /// A fake `claude` whose n-th call prints `turns[n-1]` (the last one
    /// thereafter), saving each call's prompt and argv.
    #[cfg(unix)]
    fn claude_turns(f: &mut Fixture, turns: &[String]) {
        claude_turns_after(f, turns, "");
    }

    /// [`claude_turns`], running the shell `prelude` at the start of every call.
    #[cfg(unix)]
    fn claude_turns_after(f: &mut Fixture, turns: &[String], prelude: &str) {
        let calls = f.path("calls");
        std::fs::create_dir_all(&calls).unwrap();
        for (i, body) in turns.iter().enumerate() {
            std::fs::write(calls.join(format!("turn-{}.json", i + 1)), body).unwrap();
        }
        std::fs::write(calls.join("turn-last.json"), turns.last().unwrap()).unwrap();
        f.claude_custom_script(&format!(
            "{prelude}\nc='{}'\nn=$(($(cat \"$c/n\" 2>/dev/null || echo 0) + 1))\necho \"$n\" > \"$c/n\"\ncat > \"$c/prompt-$n\"\nprintf '%s\\n' \"$@\" > \"$c/argv-$n\"\nif [ -e \"$c/turn-$n.json\" ]; then cat \"$c/turn-$n.json\"; else cat \"$c/turn-last.json\"; fi",
            calls.display()
        ));
    }

    fn limited_turn(session: &str) -> String {
        serde_json::json!({
            "type": "result",
            "session_id": session,
            "result": LIMITED_MESSAGE,
            "is_error": true,
            "num_turns": 1,
        })
        .to_string()
    }

    /// Whether some end-record warning contains every one of `parts`.
    fn warned(completion: &Completion, parts: &[&str]) -> bool {
        completion
            .end
            .warnings
            .iter()
            .any(|w| parts.iter().all(|p| w.contains(p)))
    }

    /// Both profiles have headroom; team03 is preferred.
    fn two_usable_profiles() -> UsageSource {
        report(vec![
            snapshot(Timestamp::now(), "team03", 0.0, 10.0),
            snapshot(Timestamp::now(), "team04", 0.0, 50.0),
        ])
    }

    #[cfg(unix)]
    fn spawned_profiles(f: &Fixture) -> Vec<(String, Option<String>)> {
        launch_records(f)
            .into_iter()
            .filter_map(|r| match r {
                audit::Record::Spawn(s) => Some((s.inv, s.payload.profile)),
                _ => None,
            })
            .collect()
    }

    #[cfg(unix)]
    fn limited_run(
        f: &Fixture,
        usage: &(dyn Fn() -> UsageSource + Sync),
        profile: Option<&str>,
    ) -> (Completion, String) {
        start_run(f, Some(PromptSource::Arg), None, Some("go and look"));
        let ctx = f.ctx(usage);
        let mut out: Vec<u8> = Vec::new();
        let completion = run_with(
            &ctx,
            "explorer",
            Some(&f.work()),
            "go and look".to_string(),
            profile,
            None,
            false,
            &mut out,
        )
        .unwrap();
        (completion, String::from_utf8(out).unwrap())
    }

    #[cfg(unix)]
    #[test]
    fn a_limited_run_continues_once_on_another_usable_profile() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-limited-retry");
        f.token("team03");
        f.token("team04");
        claude_turns(
            &mut f,
            &[limited_turn("9f22dd1b-aaaa"), OK_RESULT.to_string()],
        );

        let (completion, text) = limited_run(&f, &two_usable_profiles, None);
        assert_eq!(completion, 0);
        assert!(text.starts_with("the answer"), "{text}");
        assert!(text.contains("profile=team04"), "{text}");
        assert_eq!(f.call("prompt-2"), LIMIT_RETRY_PROMPT);
        let argv: Vec<String> = f.call("argv-2").lines().map(String::from).collect();
        let i = argv.iter().position(|a| a == "--resume").expect("--resume");
        assert_eq!(argv[i + 1], "9f22dd1b-aaaa");

        // Both turns belong to this one invocation.
        let spawns = spawned_profiles(&f);
        let inv = f.audit.lineage().invocation_id.clone();
        assert_eq!(
            spawns,
            vec![
                (inv.clone(), Some("team03".to_string())),
                (inv, Some("team04".to_string())),
            ]
        );
        assert_eq!(completion.end.exit, Some(0));
        assert!(
            completion
                .end
                .warnings
                .iter()
                .any(|w| w.contains("team03 hit a usage limit") && w.contains("on team04")),
            "{:?}",
            completion.end.warnings
        );
        assert!(
            warned(&completion, &["limited turn on team03", "\"num_turns\":1"]),
            "{:?}",
            completion.end.warnings
        );
        let record = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
        assert_eq!(record.profile.as_deref(), Some("team04"));
    }

    #[cfg(unix)]
    #[test]
    fn a_fresh_selection_and_its_limit_continuation_follow_the_configured_weights() {
        use crate::tuning::{AccountLayer, Priority, Tuning, TuningLayer};
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-weighted");
        f.cfg.claude.profiles = vec!["team03".into(), "team04".into(), "team05".into()];
        for p in ["team03", "team04", "team05"] {
            f.token(p);
        }
        let weight = |w: f64| AccountLayer {
            priority: Priority {
                weight: Some(w),
                weekly_reset: None,
            },
            ..AccountLayer::default()
        };
        // Unweighted, team03 scores highest and team04 beats team05 on order.
        f.cfg.tuning = Tuning::default().overlay(&TuningLayer {
            accounts: BTreeMap::from([
                ("team04".to_string(), weight(3.0)),
                ("team05".to_string(), weight(10.0)),
            ]),
            ..TuningLayer::default()
        });
        claude_turns(
            &mut f,
            &[limited_turn("9f22dd1b-aaaa"), OK_RESULT.to_string()],
        );
        let usage = || {
            report(vec![
                snapshot(Timestamp::now(), "team03", 0.0, 10.0),
                snapshot(Timestamp::now(), "team05", 0.0, 50.0),
                snapshot(Timestamp::now(), "team04", 0.0, 50.0),
            ])
        };

        let (completion, _) = limited_run(&f, &usage, None);
        assert_eq!(completion, 0);
        let inv = f.audit.lineage().invocation_id.clone();
        assert_eq!(
            spawned_profiles(&f),
            vec![
                (inv.clone(), Some("team05".to_string())),
                (inv, Some("team04".to_string())),
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_limit_continuation_records_its_selection_notes() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-limited-notes");
        f.cfg.claude.profiles = vec!["team03".into(), "team04".into(), "team05".into()];
        f.token("team03");
        f.token("team04");
        claude_turns(
            &mut f,
            &[limited_turn("9f22dd1b-aaaa"), OK_RESULT.to_string()],
        );

        let (completion, _) = limited_run(&f, &two_usable_profiles, None);
        assert_eq!(completion, 0);
        let notes: Vec<Vec<String>> = launch_records(&f)
            .into_iter()
            .filter_map(|r| match r {
                audit::Record::Spawn(s) => Some(s.payload.selection_notes),
                _ => None,
            })
            .collect();
        assert_eq!(notes.len(), 2);
        let retry = &notes[1];
        assert!(
            retry
                .iter()
                .any(|n| n.contains("team05") && n.contains("not a candidate")),
            "{retry:?}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_limited_run_with_no_other_usable_profile_exits_4_with_its_session() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-limited-no-other");
        f.token("team03");
        f.token("team04");
        claude_turns(
            &mut f,
            &[limited_turn("9f22dd1b-aaaa"), OK_RESULT.to_string()],
        );
        // team04 publishes no usage, so only a fallback could pick it.
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);

        let (completion, text) = limited_run(&f, &usage, None);
        assert_eq!(completion, crate::harness::EXIT_LIMITED);
        assert_eq!(completion.end.exit, Some(crate::harness::EXIT_LIMITED));
        assert!(text.contains(LIMITED_MESSAGE), "{text}");
        assert!(text.contains("[ccc] session=9f22dd1b-aaaa"), "{text}");
        assert!(text.contains("profile=team03"), "{text}");
        assert_eq!(spawned_profiles(&f).len(), 1);
        assert!(
            warned(&completion, &["team03 hit a usage limit", NO_OTHER_PROFILE]),
            "{:?}",
            completion.end.warnings
        );
    }

    /// A usage read that fails the test: the ambient login must never make one.
    fn usage_never_read() -> UsageSource {
        panic!("the ambient login must not read usage")
    }

    #[test]
    fn an_ambient_invocation_exports_no_account_and_keeps_the_endpoint_scrub() {
        let mut f = Fixture::new("ambient-invocation");
        f.cfg.claude.profiles = Vec::new();
        let ctx = f.ctx(&usage_never_read);

        let prepared = prepare(
            &ctx,
            "explorer",
            Some(&f.work()),
            None,
            None,
            &GooseFlags::default(),
        )
        .unwrap();
        assert!(prepared.notes.is_empty(), "{:?}", prepared.notes);
        let LaunchPrepared { inv, account, .. } =
            launch_invocation(&ctx, prepared, vec![]).unwrap();
        assert_eq!(account, None);
        assert!(inv.env_set.is_empty(), "{inv:?}");
        for var in ["CLAUDE_CONFIG_DIR", "CLAUDE_CODE_OAUTH_TOKEN"] {
            assert!(!inv.env_remove.iter().any(|v| v == var), "{inv:?}");
        }
        for var in ["ANTHROPIC_API_KEY", "CLAUDE_CODE_USE_BEDROCK"] {
            assert!(inv.env_remove.iter().any(|v| v == var), "{inv:?}");
        }
    }

    #[test]
    fn an_ambient_spawn_names_the_endpoint_variables_it_removes() {
        /// The diagnostics of one spawn, profiled or ambient, with `vars` set.
        fn spawn_diagnostics(label: &str, vars: &[(&str, &str)], profiled: bool) -> Vec<String> {
            let mut f = Fixture::new(label);
            if profiled {
                f.cfg.claude.auth = ClaudeAuth::ConfigDir;
                f.config_dir_credential("team03");
            } else {
                f.cfg.claude.profiles = Vec::new();
            }
            f.claude_script(OK_RESULT);
            for (var, value) in vars {
                f.env.insert(var.to_string(), value.to_string());
            }
            let account = profiled.then(|| Account::from_config(&f.cfg.claude, "team03", &f.env));
            let ctx = f.ctx(&usage_never_read);
            let common = Common {
                cwd: f.path("work"),
                extra_args: vec![],
                mode: run_mode(),
                preset_session_id: Some("p".into()),
            };
            let diagnostics = Mutex::new(Vec::new());
            let record = record_of(&resolved(), &common.cwd, account.as_ref(), ctx.now);
            let result = drive_with_meta_report(
                &ctx,
                &resolved(),
                &common,
                account,
                vec![],
                &|message| diagnostics.lock().unwrap().push(message),
                &record,
            )
            .unwrap();
            assert!(result.result.is_ok());
            diagnostics.into_inner().unwrap()
        }

        let conflicts = [
            ("ANTHROPIC_API_KEY", "sk-test"),
            ("CLAUDE_CODE_USE_BEDROCK", "1"),
        ];
        assert_eq!(
            spawn_diagnostics("ambient-scrub-note", &conflicts, false),
            vec![format!(
                "note: ANTHROPIC_API_KEY, CLAUDE_CODE_USE_BEDROCK removed from the child's environment; {AMBIENT} is the Claude Code login, not an API key or cloud provider"
            )]
        );
        // Nothing removed, nothing said; and a profiled spawn is not the
        // ambient login, whatever it removes.
        assert!(spawn_diagnostics("ambient-scrub-quiet", &[], false).is_empty());
        assert!(spawn_diagnostics("profiled-scrub-quiet", &conflicts, true).is_empty());
    }

    #[test]
    fn profile_still_names_an_account_when_profiles_is_empty() {
        let mut f = Fixture::new("ambient-profile-flag");
        f.cfg.claude.profiles = Vec::new();
        f.token("team03");
        let ctx = f.ctx(&usage_never_read);

        let prepared = prepare(
            &ctx,
            "explorer",
            Some(&f.work()),
            Some("team03"),
            None,
            &GooseFlags::default(),
        )
        .unwrap();
        assert_eq!(prepared.account.unwrap().profile, "team03");
        let err = prepare(
            &ctx,
            "explorer",
            Some(&f.work()),
            Some("team04"),
            None,
            &GooseFlags::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("profile team04 has no credential"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn a_run_under_the_ambient_login_records_no_profile_and_resumes_ambiently() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("ambient-run");
        f.cfg.claude.profiles = Vec::new();
        f.claude_script(OK_RESULT);

        start_run(&f, Some(PromptSource::Arg), None, Some("go and look"));
        let mut out: Vec<u8> = Vec::new();
        let completion = run_with(
            &f.ctx(&usage_never_read),
            "explorer",
            Some(&f.work()),
            "go and look".to_string(),
            None,
            None,
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(completion, 0);
        let text = String::from_utf8(out).unwrap();
        assert!(text.ends_with(" profile=-\n"), "{text:?}");
        assert!(!f
            .call("env")
            .contains("CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat-x"));
        let inv = f.audit.lineage().invocation_id.clone();
        assert_eq!(spawned_profiles(&f), vec![(inv, None)]);
        let records = audit_records(&f);
        let spawn = spawn_of(&records);
        assert_eq!(spawn.payload.auth, None);
        assert!(spawn.payload.selection_notes.is_empty());

        let record = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
        assert_eq!(record.profile, None);
        assert_eq!(record.auth, None);

        // The trust marker goes where an ambient `claude` reads it.
        let work = std::fs::canonicalize(f.work()).unwrap();
        let home_config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(f.path(".claude.json")).unwrap())
                .unwrap();
        assert_eq!(
            home_config["projects"][work.display().to_string()]["hasTrustDialogAccepted"],
            serde_json::json!(true)
        );

        start_resume(&f, "9f22dd1b-aaaa");
        let mut out: Vec<u8> = Vec::new();
        let completion = resume_with(
            &f.ctx(&usage_never_read),
            "9f22dd1b-aaaa",
            false,
            None,
            "again".into(),
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(completion, 0);
        let text = String::from_utf8(out).unwrap();
        assert!(text.ends_with(" profile=-\n"), "{text:?}");
        let record = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
        assert_eq!(record.profile, None);
        assert_eq!(record.auth, None);
    }

    #[cfg(unix)]
    #[test]
    fn an_ambient_run_marks_trust_in_the_inherited_config_dir_it_leaves_in_place() {
        let _lock = test_io_lock();
        // Absolute, and relative to the agent's cwd (not ccc's).
        for relative in [false, true] {
            let mut f = Fixture::new("ambient-inherited-config-dir");
            f.cfg.claude.profiles = Vec::new();
            f.claude_script(OK_RESULT);
            let work = std::fs::canonicalize(f.work()).unwrap();
            let (value, inherited) = if relative {
                (".claude-local".to_string(), work.join(".claude-local"))
            } else {
                let dir = f.path("own-login");
                (dir.display().to_string(), dir)
            };
            std::fs::create_dir_all(&inherited).unwrap();
            f.env.insert("CLAUDE_CONFIG_DIR".into(), value);

            start_run(&f, Some(PromptSource::Arg), None, Some("go and look"));
            let mut out: Vec<u8> = Vec::new();
            let completion = run_with(
                &f.ctx(&usage_never_read),
                "explorer",
                Some(&f.work()),
                "go and look".to_string(),
                None,
                None,
                false,
                &mut out,
            )
            .unwrap();
            assert_eq!(completion, 0);

            // The child keeps the variable, so it reads this file; that the
            // ambient invocation removes no `CLAUDE_CONFIG_DIR` is asserted by
            // `an_ambient_invocation_exports_no_account_and_keeps_the_endpoint_scrub`.
            let config: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(inherited.join(".claude.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(
                config["projects"][work.display().to_string()]["hasTrustDialogAccepted"],
                serde_json::json!(true)
            );
            assert!(!f.path(".claude.json").exists(), "relative={relative}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn an_ambient_run_that_hits_a_usage_limit_exits_4_without_continuing() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("ambient-limited");
        f.cfg.claude.profiles = Vec::new();
        claude_turns(
            &mut f,
            &[limited_turn("9f22dd1b-aaaa"), OK_RESULT.to_string()],
        );

        let (completion, text) = limited_run(&f, &usage_never_read, None);
        assert_eq!(completion, crate::harness::EXIT_LIMITED);
        assert!(text.contains("[ccc] session=9f22dd1b-aaaa"), "{text}");
        assert!(text.contains("profile=-"), "{text}");
        assert_eq!(spawned_profiles(&f).len(), 1);
        assert!(
            warned(
                &completion,
                &["the ambient login hit a usage limit", NO_OTHER_PROFILE]
            ),
            "{:?}",
            completion.end.warnings
        );
    }

    /// [`limited_run`] with this host's ledgers behind the budget closure.
    #[cfg(unix)]
    fn keyed_limited_run(
        f: &Fixture,
        usage: &(dyn Fn() -> UsageSource + Sync),
    ) -> (Completion, String) {
        start_run(f, Some(PromptSource::Arg), None, Some("go and look"));
        let accounts = || crate::budget::bus_accounts(&f.cfg, &f.env, f.now, &crate::bus::NoBus);
        let ctx = f.ctx_with_accounts(usage, &accounts);
        let mut out: Vec<u8> = Vec::new();
        let completion = run_with(
            &ctx,
            "explorer",
            Some(&f.work()),
            "go and look".to_string(),
            None,
            None,
            false,
            &mut out,
        )
        .unwrap();
        (completion, String::from_utf8(out).unwrap())
    }

    #[cfg(unix)]
    #[test]
    fn the_limit_continuation_makes_its_own_account_read_and_sees_the_new_flag() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-limited-reads-flag");
        f.token("team03");
        f.token("team04");
        let resets_at = Timestamp::now().as_second() + 86_400;
        claude_turns(
            &mut f,
            &[refused_stream(Some(resets_at)), OK_RESULT.to_string()],
        );
        start_run(&f, Some(PromptSource::Arg), None, Some("go and look"));
        let reads = Mutex::new(Vec::new());
        let accounts = || {
            let read = crate::budget::bus_accounts(&f.cfg, &f.env, f.now, &crate::bus::NoBus);
            reads.lock().unwrap().push(read.clone());
            read
        };
        let ctx = f.ctx_with_accounts(&two_usable_profiles, &accounts);
        let mut out: Vec<u8> = Vec::new();
        let completion = run_with(
            &ctx,
            "explorer",
            Some(&f.work()),
            "go and look".to_string(),
            None,
            None,
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(completion, 0);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("profile=team04"), "{text}");
        let reads = reads.into_inner().unwrap();
        assert_eq!(reads.len(), 2, "{reads:?}");
        assert!(reads[0].flags["team03"].is_empty(), "{reads:?}");
        assert_eq!(
            reads[1].flags["team03"].keys().collect::<Vec<_>>(),
            vec!["general"]
        );
    }

    /// A usage closure answering `reads[n]` on its n-th call, the last
    /// thereafter.
    fn usage_reads(reads: Vec<UsageSource>) -> impl Fn() -> UsageSource + Sync {
        let n = Mutex::new(0usize);
        move || {
            let mut n = n.lock().unwrap();
            let i = (*n).min(reads.len() - 1);
            *n += 1;
            reads[i].clone()
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_subscription_limit_hands_off_to_a_key_when_no_profile_has_quota() {
        for profiles in [&["team03", "team04"][..], &["team03"][..]] {
            let _lock = test_io_lock();
            let mut f = Fixture::new("run-limited-to-key");
            f.cfg.claude.profiles = profiles.iter().map(|p| p.to_string()).collect();
            for p in profiles {
                f.token(p);
            }
            api_key(&mut f, "team");
            claude_turns(
                &mut f,
                &[limited_turn("9f22dd1b-aaaa"), OK_RESULT.to_string()],
            );
            let usage = || {
                report(vec![
                    snapshot(Timestamp::now(), "team03", 0.0, 10.0),
                    snapshot(Timestamp::now(), "team04", 0.0, 100.0),
                ])
            };

            let (completion, text) = keyed_limited_run(&f, &usage);
            assert_eq!(completion, 0, "{profiles:?}: {text}");
            assert!(text.contains("profile=team"), "{text}");
            let inv = f.audit.lineage().invocation_id.clone();
            assert_eq!(
                spawned_profiles(&f),
                vec![
                    (inv.clone(), Some("team03".to_string())),
                    (inv, Some("team".to_string())),
                ]
            );
            assert!(
                warned(&completion, &["team03 hit a usage limit", "on team"]),
                "{:?}",
                completion.end.warnings
            );
            let ledger = key_ledger(&f, "team").unwrap();
            assert_eq!(ledger.turns, 1);
            assert!((ledger.usd - 0.01).abs() < 1e-9);
            let record = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
            assert_eq!(record.profile.as_deref(), Some("team"));
            assert_eq!(record.auth, Some(ClaudeAuth::ApiKey));
        }
    }

    /// Starts the fixture's audit as `ccc run explorer <prompt>` and returns
    /// the start payload.
    fn bus_start(f: &Fixture, prompt: &str) -> StartPayload {
        let start = StartPayload {
            pid: std::process::id(),
            cmd: AuditCommand::Run,
            agent: Some("explorer".into()),
            session: None,
            cwd_arg: Some(f.work()),
            profile_arg: None,
            harness_arg: None,
            json: false,
            prompt_source: Some(PromptSource::Arg),
            prompt_file: None,
            prompt: Some(prompt.into()),
            passthrough: vec![],
            fork: false,
        };
        f.audit.start(start.clone()).unwrap();
        start
    }

    /// Runs the explorer on `profile` (selection when `None`) with a
    /// recording bus, closes the bus as the audited path does, and returns
    /// the completion and the bus.
    fn bus_run(
        f: &Fixture,
        profile: Option<&str>,
        usage: &(dyn Fn() -> UsageSource + Sync),
        close_notes: Vec<String>,
    ) -> (Completion, crate::bus::RecordingBus) {
        let start = bus_start(f, "go and look");
        let mut bus =
            crate::bus::RecordingBus::new("ccc", "atlas", crate::bus::Remote::Records(vec![]));
        bus.close_notes = close_notes;
        let live = Live::new(
            bus.identity(),
            f.audit.lineage(),
            &start,
            "/cfg/agents.toml",
            0,
            f.now,
        );
        let accounts = || crate::budget::bus_accounts(&f.cfg, &f.env, f.now, &bus);
        let ctx = f.ctx_with_bus(usage, &accounts, &bus, &live);
        let mut out = Vec::new();
        let mut completion = run_with(
            &ctx,
            "explorer",
            Some(&f.work()),
            "go and look".into(),
            profile,
            None,
            false,
            &mut out,
        )
        .unwrap();
        let said = Mutex::new(Vec::new());
        close_bus(&bus, &live, &mut completion.end, &|line| {
            said.lock().unwrap().push(line.to_string())
        });
        let said = said.into_inner().unwrap();
        let closed: Vec<_> = bus
            .close_notes
            .iter()
            .map(|n| format!("note: {n}"))
            .collect();
        assert_eq!(said, closed);
        assert!(completion.end.warnings.ends_with(&closed));
        (completion, bus)
    }

    /// The top-level keys a session record may carry: the generic schema's.
    const GENERIC_FIELDS: &[&str] = &[
        "v",
        "id",
        "publisher",
        "host",
        "state",
        "started_at",
        "updated_at",
        "parent",
        "root",
        "session",
        "harness",
        "model",
        "agent",
        "account",
        "account_kind",
        "cwd",
        "pid",
        "title",
        "cost_usd",
        "ended_at",
        "duration_ms",
        "ok",
        "meta",
    ];

    #[cfg(unix)]
    #[test]
    fn a_key_turn_is_published_at_spawn_charged_and_ended() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-bus-key");
        api_key(&mut f, "team");
        f.claude_script(OK_RESULT);
        let usage = || UsageSource::Unavailable("not consulted".into());
        let refused = "the broker refused a publish (NotAuthorized); check the ACL for claude/usage/agents/ccc/#";
        let (completion, bus) = bus_run(&f, Some("team"), &usage, vec![refused.into()]);
        assert_eq!(completion, 0);
        assert!(
            completion
                .end
                .warnings
                .contains(&format!("note: {refused}")),
            "{:?}",
            completion.end.warnings
        );
        let inv = f.audit.lineage().invocation_id.clone();
        let calls = bus.calls();
        use crate::bus::Recorded;
        let [Recorded::PutSession(running), Recorded::PutAccount(level), Recorded::PutSession(costed), Recorded::EndSession(ended), Recorded::Close] =
            calls.as_slice()
        else {
            panic!("{calls:#?}");
        };

        assert_eq!(running.id, inv);
        assert_eq!(running.root.as_deref(), Some(inv.as_str()));
        assert_eq!(running.parent, None);
        assert_eq!(
            (running.publisher.as_str(), running.host.as_str()),
            ("ccc", "atlas")
        );
        assert_eq!(running.state, claude_usage_mqtt_core::SessionState::Running);
        assert_eq!(running.harness.as_deref(), Some("claude"));
        assert_eq!(running.model.as_deref(), Some("claude-haiku-4-5"));
        assert_eq!(running.agent.as_deref(), Some("explorer"));
        assert_eq!(running.account.as_deref(), Some("team"));
        assert_eq!(
            running.account_kind,
            Some(claude_usage_mqtt_core::AccountKind::ApiKey)
        );
        assert_eq!(running.pid, Some(std::process::id()));
        assert!(running.cwd.as_deref().unwrap().ends_with("work"));
        assert!(running.session.is_some());
        assert_eq!(running.cost_usd, None);
        assert_eq!(running.title, None);
        assert_eq!(running.meta["cmd"], "run");
        assert_eq!(running.meta["auth"], "api-key");
        assert_eq!(running.meta["effort"], "low");
        assert_eq!(running.meta["agents_config"], "/cfg/agents.toml");

        assert_eq!(level.account, "team");
        assert_eq!(
            (level.publisher.as_str(), level.host.as_str()),
            ("ccc", "atlas")
        );
        assert!((level.usd - 0.01).abs() < 1e-9);
        assert_eq!(level.turns, Some(1));

        assert_eq!(costed.cost_usd, Some(0.01));
        assert_eq!(costed.state, claude_usage_mqtt_core::SessionState::Running);

        assert_eq!(ended.state, claude_usage_mqtt_core::SessionState::Ended);
        assert_eq!(ended.ok, Some(true));
        assert_eq!(ended.cost_usd, Some(0.01));
        assert_eq!(ended.session.as_deref(), Some("9f22dd1b-aaaa"));
        assert_eq!(ended.meta["exit"], 0);
        assert!(ended.ended_at.is_some() && ended.duration_ms.is_some());
        let wire = serde_json::to_value(ended).unwrap();
        for key in wire.as_object().unwrap().keys() {
            assert!(
                GENERIC_FIELDS.contains(&key.as_str()),
                "{key} at the top level"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_key_turn_refused_for_spend_publishes_the_flagged_level_and_ends_not_ok() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-bus-exhausted");
        api_key(&mut f, "team");
        f.claude_script(&refused_turn(
            "Your credit balance is too low to access the Anthropic API.",
        ));
        let usage = || UsageSource::Unavailable("not consulted".into());
        let (completion, bus) = bus_run(&f, Some("team"), &usage, vec![]);
        assert_eq!(completion, crate::harness::EXIT_LIMITED);
        let levels: Vec<_> = bus
            .calls()
            .into_iter()
            .filter_map(|c| match c {
                crate::bus::Recorded::PutAccount(level) => Some(level),
                _ => None,
            })
            .collect();
        assert_eq!(levels.len(), 2, "{levels:?}");
        assert!(levels[0].exhausted.is_none());
        let flag = levels[1].exhausted.as_ref().unwrap();
        assert_eq!(
            flag.session.as_deref(),
            Some(f.audit.lineage().invocation_id.as_str())
        );
        let (is_end, ended) = bus.sessions().pop().unwrap();
        assert!(is_end);
        assert_eq!(ended.ok, Some(false));
        assert_eq!(ended.meta["exit"], crate::harness::EXIT_LIMITED);
        // The refusal reported a cost of zero.
        assert_eq!(ended.cost_usd, Some(0.0));
    }

    #[cfg(unix)]
    #[test]
    fn a_hand_off_puts_the_record_again_with_the_new_account_and_sums_the_cost() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-bus-handoff");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.token("team03");
        api_key(&mut f, "team");
        let limited_with_cost = serde_json::json!({
            "type": "result",
            "session_id": "9f22dd1b-aaaa",
            "result": LIMITED_MESSAGE,
            "is_error": true,
            "num_turns": 1,
            "total_cost_usd": 0.25,
            "model_usage_since_last_result": {"claude-sonnet-5-5": {"costUSD": 0.25}},
        })
        .to_string();
        claude_turns(&mut f, &[limited_with_cost, OK_RESULT.to_string()]);
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);
        let (completion, bus) = bus_run(&f, None, &usage, vec![]);
        assert_eq!(completion, 0);
        let accounts: Vec<_> = bus
            .sessions()
            .into_iter()
            .filter(|(is_end, _)| !is_end)
            .map(|(_, r)| (r.account, r.account_kind, r.cost_usd))
            .collect();
        use claude_usage_mqtt_core::AccountKind;
        assert_eq!(
            accounts,
            vec![
                (Some("team03".into()), Some(AccountKind::Subscription), None),
                (
                    Some("team03".into()),
                    Some(AccountKind::Subscription),
                    Some(0.25)
                ),
                (Some("team".into()), Some(AccountKind::ApiKey), Some(0.25)),
                (Some("team".into()), Some(AccountKind::ApiKey), Some(0.26)),
            ]
        );
        let (_, ended) = bus.sessions().pop().unwrap();
        assert_eq!((ended.ok, ended.cost_usd), (Some(true), Some(0.26)));
        // The hand-off's account read went through the bus, levels first: the
        // key's and the profile's.
        assert!(bus
            .calls()
            .iter()
            .any(|c| matches!(c, crate::bus::Recorded::Fetch(levels) if levels.len() == 2)));
    }

    /// A result of a resumed process: Claude Code's running total carries
    /// the session's earlier 5.00, and the process states only its own 0.01.
    const RESUMED_RESULT: &str = r#"{"type":"result","session_id":"9f22dd1b-aaaa","result":"the answer","is_error":false,"num_turns":1,"total_cost_usd":5.01,"model_usage_since_last_result":{"claude-sonnet-5-5":{"costUSD":0.01}}}"#;

    #[cfg(unix)]
    #[test]
    fn a_resumed_key_turn_is_charged_only_what_its_process_states() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("resume-key-stated-cost");
        api_key(&mut f, "team");
        claude_turns(&mut f, &[OK_RESULT.to_string(), RESUMED_RESULT.to_string()]);
        assert_eq!(pinned_run(&f, "team"), 0);

        let start = StartPayload {
            pid: std::process::id(),
            cmd: AuditCommand::Run,
            agent: Some("explorer".into()),
            session: None,
            cwd_arg: Some(f.work()),
            profile_arg: None,
            harness_arg: None,
            json: false,
            prompt_source: Some(PromptSource::Arg),
            prompt_file: None,
            prompt: Some("again".into()),
            passthrough: vec![],
            fork: false,
        };
        let bus =
            crate::bus::RecordingBus::new("ccc", "atlas", crate::bus::Remote::Records(vec![]));
        let live = Live::new(
            bus.identity(),
            f.audit.lineage(),
            &start,
            "/cfg/agents.toml",
            0,
            f.now,
        );
        let usage = || UsageSource::Unavailable("not consulted".into());
        let accounts = || crate::budget::bus_accounts(&f.cfg, &f.env, f.now, &bus);
        let ctx = f.ctx_with_bus(&usage, &accounts, &bus, &live);
        let mut out: Vec<u8> = Vec::new();
        let completion = resume_with(
            &ctx,
            "9f22dd1b-aaaa",
            false,
            Some("team"),
            "again".to_string(),
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(completion, 0);

        let ledger = key_ledger(&f, "team").unwrap();
        assert!((ledger.usd - 0.02).abs() < 1e-9, "{}", ledger.usd);
        assert_eq!((ledger.turns, ledger.unmeasured_turns), (2, 0));
        let (_, last) = bus.sessions().pop().unwrap();
        assert_eq!(last.cost_usd, Some(0.01));
    }

    #[cfg(unix)]
    #[test]
    fn a_hand_off_charges_the_key_only_what_the_continuation_states() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-bus-handoff-stated");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.token("team03");
        api_key(&mut f, "team");
        let limited = serde_json::json!({
            "type": "result",
            "session_id": "9f22dd1b-aaaa",
            "result": LIMITED_MESSAGE,
            "is_error": true,
            "num_turns": 1,
            "total_cost_usd": 0.25,
            "model_usage_since_last_result": {"claude-sonnet-5-5": {"costUSD": 0.25}},
        })
        .to_string();
        let continued = serde_json::json!({
            "type": "result",
            "session_id": "9f22dd1b-aaaa",
            "result": "the answer",
            "is_error": false,
            "num_turns": 1,
            "total_cost_usd": 7.26,
            "model_usage_since_last_result": {"claude-sonnet-5-5": {"costUSD": 0.01}},
        })
        .to_string();
        claude_turns(&mut f, &[limited, continued]);
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);
        let (completion, bus) = bus_run(&f, None, &usage, vec![]);
        assert_eq!(completion, 0);
        let ledger = key_ledger(&f, "team").unwrap();
        assert!((ledger.usd - 0.01).abs() < 1e-9, "{}", ledger.usd);
        assert_eq!((ledger.turns, ledger.unmeasured_turns), (1, 0));
        let (_, ended) = bus.sessions().pop().unwrap();
        let cost = ended.cost_usd.unwrap();
        assert!((cost - 0.26).abs() < 1e-9, "{cost}");
    }

    #[cfg(unix)]
    #[test]
    fn with_the_bus_off_a_run_keeps_no_record() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-bus-off");
        f.claude_script(OK_RESULT);
        f.token("team03");
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);
        let start = bus_start(&f, "go and look");
        let live = Live::new(None, f.audit.lineage(), &start, "/a", 4, f.now);
        let ctx = f.ctx_with_bus(&usage, &no_accounts, &crate::bus::NoBus, &live);
        let mut out = Vec::new();
        let completion = run_with(
            &ctx,
            "explorer",
            Some(&f.work()),
            "go and look".into(),
            None,
            None,
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(completion, 0);
        assert!(live.record().is_none());
    }

    #[cfg(unix)]
    #[test]
    fn a_subscription_limit_hands_off_to_a_key_past_a_profile_with_doubtful_data() {
        let stale = |now: Timestamp| ProfileUsage {
            stale_reason: Some("upstream poll failed".into()),
            ..snapshot(now, "team04", 0.0, 10.0)
        };
        let with_fable = |now: Timestamp, name: &str| {
            let mut s = snapshot(now, name, 0.0, 10.0);
            s.windows.push(UsageWindow {
                key: "7-day_fable".into(),
                utilization: 10.0,
                resets_at: at(now, 100),
            });
            s
        };
        for fable in [false, true] {
            let _lock = test_io_lock();
            let mut f = Fixture::new("run-limited-key-doubtful");
            if fable {
                // The explorer's model counts as Fable work here.
                f.cfg.claude.selection.fable_models = vec!["*haiku*".into()];
            }
            f.token("team03");
            f.token("team04");
            api_key(&mut f, "team");
            claude_turns(
                &mut f,
                &[limited_turn("9f22dd1b-aaaa"), OK_RESULT.to_string()],
            );
            let usage = || {
                let now = Timestamp::now();
                report(vec![
                    with_fable(now, "team03"),
                    if fable {
                        snapshot(now, "team04", 0.0, 10.0)
                    } else {
                        stale(now)
                    },
                ])
            };

            let (completion, text) = keyed_limited_run(&f, &usage);
            // team04 cannot take the work (stale, or no Fable window), so the
            // key does.
            assert_eq!(completion, 0, "{fable}: {text}");
            let inv = f.audit.lineage().invocation_id.clone();
            assert_eq!(
                spawned_profiles(&f),
                vec![
                    (inv.clone(), Some("team03".to_string())),
                    (inv, Some("team".to_string())),
                ],
                "{fable}"
            );
            assert_eq!(key_ledger(&f, "team").unwrap().turns, 1);
        }
    }

    #[cfg(unix)]
    #[test]
    fn an_exhausted_key_turn_continues_on_a_profile_or_exits_4() {
        for profile_able in [true, false] {
            let _lock = test_io_lock();
            let mut f = Fixture::new("run-key-exhausted-handoff");
            f.token("team03");
            f.token("team04");
            api_key(&mut f, "team");
            claude_turns(
                &mut f,
                &[
                    refused_turn("Your credit balance is too low to access the Anthropic API."),
                    OK_RESULT.to_string(),
                ],
            );
            let now = Timestamp::now();
            let spent = report(vec![
                snapshot(now, "team03", 0.0, 100.0),
                snapshot(now, "team04", 0.0, 100.0),
            ]);
            let later = if profile_able {
                report(vec![
                    snapshot(now, "team03", 0.0, 100.0),
                    snapshot(now, "team04", 0.0, 10.0),
                ])
            } else {
                // No snapshot for either: nothing usable, only a fallback.
                report(vec![])
            };
            let usage = usage_reads(vec![spent, later]);

            let (completion, text) = keyed_limited_run(&f, &usage);
            let inv = f.audit.lineage().invocation_id.clone();
            let ledger = key_ledger(&f, "team").unwrap();
            assert!(ledger.exhausted.is_some());
            if profile_able {
                assert_eq!(completion, 0, "{text}");
                assert_eq!(
                    spawned_profiles(&f),
                    vec![
                        (inv.clone(), Some("team".to_string())),
                        (inv, Some("team04".to_string())),
                    ]
                );
                assert!(
                    warned(&completion, &["team hit a usage limit", "on team04"]),
                    "{:?}",
                    completion.end.warnings
                );
            } else {
                assert_eq!(completion, crate::harness::EXIT_LIMITED, "{text}");
                assert!(text.contains("[ccc] session=9f22dd1b-aaaa"), "{text}");
                assert_eq!(spawned_profiles(&f), vec![(inv, Some("team".to_string()))]);
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_launch_never_selects_a_key() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("launch-no-key-selection");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        api_key(&mut f, "team");
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 100.0)]);
        let accounts = || crate::budget::bus_accounts(&f.cfg, &f.env, f.now, &crate::bus::NoBus);
        let ctx = f.ctx_with_accounts(&usage, &accounts);
        let terminal = ctx.at_terminal();
        let picked = select_account(&terminal, &resolved(), &usage()).unwrap();
        assert_eq!(picked.account.profile, "team03");
        // The same selection outside a launch takes the key.
        let picked = select_account(&ctx, &resolved(), &usage()).unwrap();
        assert_eq!(picked.account.profile, "team");
        assert!(
            picked
                .notes
                .iter()
                .any(|n| n.contains("using API key team")),
            "{:?}",
            picked.notes
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_limited_run_on_a_single_profile_exits_4_not_2() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-limited-single");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        claude_turns(&mut f, &[limited_turn("9f22dd1b-aaaa")]);
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);

        let (completion, text) = limited_run(&f, &usage, None);
        assert_eq!(completion, crate::harness::EXIT_LIMITED);
        assert!(text.contains("[ccc] session=9f22dd1b-aaaa"), "{text}");
        assert_eq!(spawned_profiles(&f).len(), 1);
        assert!(
            warned(&completion, &[NO_OTHER_PROFILE]),
            "{:?}",
            completion.end.warnings
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_limited_run_under_an_explicit_profile_is_not_retried() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-limited-pinned");
        f.token("team03");
        f.token("team04");
        claude_turns(
            &mut f,
            &[limited_turn("9f22dd1b-aaaa"), OK_RESULT.to_string()],
        );

        let (completion, text) = limited_run(&f, &two_usable_profiles, Some("team03"));
        assert_eq!(completion, crate::harness::EXIT_LIMITED);
        assert!(text.contains("profile=team03"), "{text}");
        assert_eq!(spawned_profiles(&f).len(), 1);
        assert!(
            warned(&completion, &["team03 hit a usage limit", PINNED]),
            "{:?}",
            completion.end.warnings
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_limited_retry_exits_4_with_the_retry_session_and_profile() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-limited-twice");
        f.token("team03");
        f.token("team04");
        claude_turns(&mut f, &[limited_turn("9f22dd1b-aaaa")]);

        let (completion, text) = limited_run(&f, &two_usable_profiles, None);
        assert_eq!(completion, crate::harness::EXIT_LIMITED);
        assert!(text.contains("[ccc] session=9f22dd1b-aaaa"), "{text}");
        assert!(text.contains("profile=team04"), "{text}");
        assert!(
            warned(
                &completion,
                &["team04 hit a usage limit", ALREADY_CONTINUED]
            ),
            "{:?}",
            completion.end.warnings
        );
        assert_eq!(
            spawned_profiles(&f)
                .into_iter()
                .map(|(_, p)| p)
                .collect::<Vec<_>>(),
            vec![Some("team03".to_string()), Some("team04".to_string())]
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_limited_resume_continues_once_on_another_usable_profile() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("resume-limited-retry");
        f.token("team03");
        f.token("team04");
        claude_turns(
            &mut f,
            &[
                OK_RESULT.to_string(),
                limited_turn("9f22dd1b-aaaa"),
                OK_RESULT.to_string(),
            ],
        );
        let (completion, _) = limited_run(&f, &two_usable_profiles, None);
        assert_eq!(completion, 0);

        let ctx = f.ctx(&two_usable_profiles);
        let mut out: Vec<u8> = Vec::new();
        let completion = resume_with(
            &ctx,
            "9f22dd1b-aaaa",
            false,
            None,
            "again".to_string(),
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(completion, 0);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("profile=team04"), "{text}");
        assert_eq!(f.call("prompt-2"), "again");
        assert_eq!(f.call("prompt-3"), LIMIT_RETRY_PROMPT);
    }

    #[cfg(unix)]
    #[test]
    fn a_limited_turn_without_a_session_id_is_not_continued() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-limited-no-session");
        f.token("team03");
        f.token("team04");
        let no_session = serde_json::json!({
            "type": "result",
            "result": LIMITED_MESSAGE,
            "is_error": true,
            "num_turns": 1,
        })
        .to_string();
        claude_turns(&mut f, &[no_session, OK_RESULT.to_string()]);

        let (completion, text) = limited_run(&f, &two_usable_profiles, None);
        assert_eq!(completion, crate::harness::EXIT_LIMITED);
        assert!(text.contains("[ccc] session=- "), "{text}");
        assert_eq!(spawned_profiles(&f).len(), 1);
        assert!(
            warned(&completion, &["team03 hit a usage limit", NO_SESSION]),
            "{:?}",
            completion.end.warnings
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_continuation_that_cannot_start_reports_the_limited_turn() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-limited-no-start");
        f.token("team03");
        f.token("team04");
        // The agent definition vanishes during the first turn, so the
        // continuation's prologue cannot load it.
        let agent = f.path("agents/explorer.md");
        claude_turns_after(
            &mut f,
            &[limited_turn("9f22dd1b-aaaa"), OK_RESULT.to_string()],
            &format!("rm -f '{}'", agent.display()),
        );

        let (completion, text) = limited_run(&f, &two_usable_profiles, None);
        assert_eq!(completion, crate::harness::EXIT_LIMITED);
        assert_eq!(completion.end.exit, Some(crate::harness::EXIT_LIMITED));
        assert!(text.contains(LIMITED_MESSAGE), "{text}");
        assert!(text.contains("[ccc] session=9f22dd1b-aaaa"), "{text}");
        assert!(text.contains("profile=team03"), "{text}");
        assert_eq!(spawned_profiles(&f).len(), 1);
        assert!(
            warned(&completion, &["continuing session 9f22dd1b-aaaa on team04"]),
            "{:?}",
            completion.end.warnings
        );
        assert!(
            warned(
                &completion,
                &["could not continue session 9f22dd1b-aaaa on team04"]
            ),
            "{:?}",
            completion.end.warnings
        );
    }

    /// A sink that accepts nothing, as stdout does once its reader is gone.
    struct ClosedSink;

    impl Write for ClosedSink {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::BrokenPipe.into())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_continuation_that_ran_but_could_not_print_is_that_failure_not_a_limit() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("run-limited-closed-out");
        f.token("team03");
        f.token("team04");
        claude_turns(
            &mut f,
            &[limited_turn("9f22dd1b-aaaa"), OK_RESULT.to_string()],
        );
        start_run(&f, Some(PromptSource::Arg), None, Some("go and look"));
        let ctx = f.ctx(&two_usable_profiles);

        let err = match run_with(
            &ctx,
            "explorer",
            Some(&f.work()),
            "go and look".to_string(),
            None,
            None,
            false,
            &mut ClosedSink,
        ) {
            Ok(completion) => panic!("expected a write failure, got exit {}", completion.code),
            Err(err) => err,
        };
        assert!(
            format!("{:#}", err.error).contains("writing the result"),
            "{err}"
        );
        let end = err.end.expect("the continuation's end record");
        assert_eq!(end.result.as_deref(), Some("the answer"));
        assert!(
            end.warnings
                .iter()
                .any(|w| w.contains("continuing session 9f22dd1b-aaaa on team04")),
            "{:?}",
            end.warnings
        );
        assert_eq!(spawned_profiles(&f).len(), 2);
    }

    #[cfg(unix)]
    #[test]
    fn a_limited_fork_answered_under_its_parent_is_not_continued() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("fork-limited-collision");
        f.token("team03");
        f.token("team04");
        claude_turns(
            &mut f,
            &[OK_RESULT.to_string(), limited_turn("9f22dd1b-aaaa")],
        );
        let (completion, _) = limited_run(&f, &two_usable_profiles, None);
        assert_eq!(completion, 0);

        let ctx = f.ctx(&two_usable_profiles);
        let mut out: Vec<u8> = Vec::new();
        let completion = resume_with(
            &ctx,
            "9f22dd1b-aaaa",
            true,
            None,
            "branch off".to_string(),
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(completion, crate::harness::EXIT_LIMITED);
        assert_eq!(f.call("n").trim(), "2", "no continuation spawned");
    }

    #[cfg(unix)]
    #[test]
    fn a_limited_fork_continues_its_own_session() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("fork-limited-retry");
        f.token("team03");
        f.token("team04");
        claude_turns(
            &mut f,
            &[
                OK_RESULT.to_string(),
                limited_turn("fork-1111"),
                OK_RESULT.to_string(),
            ],
        );
        let (completion, _) = limited_run(&f, &two_usable_profiles, None);
        assert_eq!(completion, 0);

        let ctx = f.ctx(&two_usable_profiles);
        let mut out: Vec<u8> = Vec::new();
        let completion = resume_with(
            &ctx,
            "9f22dd1b-aaaa",
            true,
            None,
            "branch off".to_string(),
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(completion, 0);
        assert_eq!(f.call("prompt-3"), LIMIT_RETRY_PROMPT);
        let argv: Vec<String> = f.call("argv-3").lines().map(String::from).collect();
        let i = argv.iter().position(|a| a == "--resume").expect("--resume");
        assert_eq!(argv[i + 1], "fork-1111");
        assert!(!argv.iter().any(|a| a == "--fork-session"), "{argv:?}");
    }

    #[test]
    fn a_reported_failure_without_a_session_id_says_so_in_the_trailer() {
        let err = HarnessError::Reported {
            session_id: None,
            message: "not logged in".into(),
            native: serde_json::json!({"num_turns": 0}),
            warnings: vec![],
        };
        let mut out: Vec<u8> = Vec::new();
        fail(
            &resolved(),
            std::path::Path::new("/srv"),
            None,
            err,
            None,
            false,
            &mut out,
        );
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("session=-"), "{text}");
        assert!(text.contains("profile=-"), "{text}");
    }

    #[test]
    fn a_reported_failure_in_json_mode_carries_the_message_and_the_native_fields() {
        let err = HarnessError::Reported {
            session_id: Some("9f22dd1b".into()),
            message: "5-hour limit reached".into(),
            native: serde_json::json!({"num_turns": 1}),
            warnings: vec![],
        };
        let mut out: Vec<u8> = Vec::new();
        fail(
            &resolved(),
            std::path::Path::new("/srv"),
            Some(&account()),
            err,
            None,
            true,
            &mut out,
        );
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["session"], "9f22dd1b");
        assert_eq!(v["result"], "5-hour limit reached");
        assert_eq!(v["native"]["num_turns"], 1);
        assert_eq!(v["profile"], "team03");
    }

    #[test]
    fn an_unusable_result_is_exit_3_and_prints_no_body() {
        let err = HarnessError::unusable("claude produced no JSON result");
        let mut out: Vec<u8> = Vec::new();
        let code = fail(
            &resolved(),
            std::path::Path::new("/srv"),
            None,
            err,
            None,
            false,
            &mut out,
        );
        assert_eq!(code, crate::harness::EXIT_NO_RESULT);
        // Nothing goes to stdout: the caller's only channel is the diagnostic.
        assert!(out.is_empty(), "{}", String::from_utf8_lossy(&out));
    }

    #[test]
    fn extra_arguments_follow_the_resolved_harness() {
        let mut cfg = Config::default();
        cfg.claude.extra_args = vec!["--add-dir".into(), "/srv".into()];
        cfg.codex.extra_args = vec!["--all".into()];
        cfg.goose.extra_args = vec!["--debug".into()];
        assert_eq!(extra_args(&cfg, Harness::Claude), vec!["--add-dir", "/srv"]);
        assert_eq!(extra_args(&cfg, Harness::Codex), vec!["--all"]);
        assert_eq!(extra_args(&cfg, Harness::Goose), vec!["--debug"]);
    }

    #[test]
    fn an_explicit_profile_must_have_a_credential() {
        let f = Fixture::new("explicit-profile");
        let none = || UsageSource::Unavailable("not consulted".into());
        let ctx = f.ctx(&none);

        let err = fresh_account(&ctx, &resolved(), Some("team03"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no credential"), "{err}");

        f.token("team03");
        let account = fresh_account(&ctx, &resolved(), Some("team03")).unwrap();
        assert_eq!(account.account.profile, "team03");
        assert_eq!(account.account.auth, ClaudeAuth::Token);
    }

    /// A Claude record for `profile` under `auth`, as a previous turn left it.
    fn claude_record(f: &Fixture, profile: Option<&str>, auth: ClaudeAuth) -> SessionRecord {
        let mut record = record_of(&resolved(), std::path::Path::new("/srv"), None, f.now);
        record.profile = profile.map(String::from);
        record.auth = Some(auth);
        record
    }

    /// Saves `record` as session `id` and returns the sidecar's text.
    fn save_record(f: &Fixture, id: &str, record: &SessionRecord) -> String {
        session::save_in(&f.sessions_dir(), id, record).unwrap();
        sidecar_text(f, id)
    }

    /// `resume_prologue` for `id` against `usage`, in a context of its own so
    /// its session lock is released on return.
    fn prologue_account(
        f: &Fixture,
        usage: &(dyn Fn() -> UsageSource + Sync),
        id: &str,
        fork: bool,
        profile: Option<&str>,
    ) -> Result<ResumePrepared> {
        resume_prologue(&f.ctx(usage), id, fork, AccountPick::Requested(profile))
    }

    fn no_accounts() -> Accounts {
        Accounts::default()
    }

    fn usage_not_consulted() -> UsageSource {
        panic!("usage consulted despite an explicit profile or a Codex session")
    }

    #[test]
    fn a_resume_moves_to_the_best_account_even_from_a_usable_one() {
        for auth in [ClaudeAuth::Token, ClaudeAuth::ConfigDir] {
            for roleless in [false, true] {
                let label = format!("resume-best-{auth}-{roleless}");
                let mut f = Fixture::new(&label);
                for profile in ["team03", "team04"] {
                    f.token(profile);
                    f.config_dir_credential(profile);
                }
                f.cfg.claude.auth = auth;
                let mut record = claude_record(&f, Some("team03"), auth);
                if roleless {
                    record.agent = None;
                }
                save_record(&f, "9f22dd1b-aaaa", &record);

                // team03 is usable, but team04 has more left.
                let team04_best = || {
                    report(vec![
                        snapshot(Timestamp::now(), "team03", 30.0, 60.0),
                        snapshot(Timestamp::now(), "team04", 0.0, 10.0),
                    ])
                };
                let prepared =
                    prologue_account(&f, &team04_best, "9f22dd1b-aaaa", false, None).unwrap();
                let account = prepared.account.unwrap();
                assert_eq!(account.profile, "team04", "{label}");
                assert_eq!(account.auth, auth, "{label}");
                let saved = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
                assert_eq!(saved.profile.as_deref(), Some("team04"), "{label}");
                assert_eq!(saved.agent, record.agent, "{label}");

                // The profile just persisted gets no preference either.
                let team03_best = || {
                    report(vec![
                        snapshot(Timestamp::now(), "team03", 0.0, 10.0),
                        snapshot(Timestamp::now(), "team04", 30.0, 60.0),
                    ])
                };
                let prepared =
                    prologue_account(&f, &team03_best, "9f22dd1b-aaaa", false, None).unwrap();
                assert_eq!(prepared.account.unwrap().profile, "team03", "{label}");
                let saved = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
                assert_eq!(saved.profile.as_deref(), Some("team03"), "{label}");
            }
        }
    }

    #[test]
    fn a_resume_scores_candidates_for_the_recorded_model() {
        let f = Fixture::new("resume-select-fable");
        f.token("team03");
        f.token("team04");
        // Only team04 publishes a Fable window, so only it can run a Fable
        // model, however much of its other buckets team03 has left.
        let usage = || {
            let now = Timestamp::now();
            let mut team04 = snapshot(now, "team04", 50.0, 50.0);
            team04.windows.push(UsageWindow {
                key: "7-day_fable".to_string(),
                utilization: 50.0,
                resets_at: at(now, 100),
            });
            report(vec![snapshot(now, "team03", 0.0, 0.0), team04])
        };
        let mut record = claude_record(&f, Some("team03"), ClaudeAuth::Token);
        save_record(&f, "opus-session", &record);
        record.model = "claude-fable-5-1[1m]".into();
        save_record(&f, "fable-session", &record);

        let fable = prologue_account(&f, &usage, "fable-session", false, None).unwrap();
        assert_eq!(fable.account.unwrap().profile, "team04");
        let opus = prologue_account(&f, &usage, "opus-session", false, None).unwrap();
        assert_eq!(opus.account.unwrap().profile, "team03");
    }

    #[test]
    fn a_resume_without_usable_data_falls_back_like_a_fresh_run() {
        let f = Fixture::new("resume-fallback");
        f.token("team03");
        f.token("team04");
        let stale = |profile: &str| ProfileUsage {
            stale_reason: Some("rate limited by the API".to_string()),
            ..snapshot(Timestamp::now(), profile, 0.0, 0.0)
        };
        let sources = [
            UsageSource::Unavailable("connection refused".to_string()),
            UsageSource::Report(UsageReport {
                profiles: vec![],
                daemon_status: Some("offline".to_string()),
            }),
            report(vec![snapshot(Timestamp::now(), "nobody", 0.0, 0.0)]),
            report(vec![stale("team03"), stale("team04")]),
            report(vec![
                snapshot(Timestamp::now(), "team03", 99.9, 0.0),
                snapshot(Timestamp::now(), "team04", 99.9, 0.0),
            ]),
        ];
        for source in sources {
            // The session last ran on team04, not the first configured profile.
            save_record(
                &f,
                "9f22dd1b-aaaa",
                &claude_record(&f, Some("team04"), ClaudeAuth::Token),
            );
            let usage = || source.clone();
            let fresh = fresh_account(&f.ctx(&usage), &resolved(), None).unwrap();
            let prepared = prologue_account(&f, &usage, "9f22dd1b-aaaa", false, None).unwrap();
            assert_eq!(fresh.account.profile, "team03", "{source:?}");
            assert_eq!(prepared.account.unwrap().profile, "team03", "{source:?}");
            assert_eq!(prepared.notes, fresh.notes, "{source:?}");
            assert!(
                prepared.notes.iter().any(|n| n.contains("falling back")),
                "{source:?}: {:?}",
                prepared.notes
            );
        }
    }

    #[test]
    fn a_resume_never_falls_back_to_a_profile_no_longer_configured() {
        let mut f = Fixture::new("resume-removed-profile");
        f.token("team09");
        let spent = || {
            report(vec![
                snapshot(Timestamp::now(), "team09", 0.0, 0.0),
                snapshot(Timestamp::now(), "team03", 10.0, 10.0),
            ])
        };
        let before = save_record(
            &f,
            "9f22dd1b-aaaa",
            &claude_record(&f, Some("team09"), ClaudeAuth::Token),
        );
        let err = prologue_account(&f, &spent, "9f22dd1b-aaaa", false, None)
            .err()
            .expect("no credentialed configured profile");
        assert!(
            format!("{err:#}").contains("no configured Claude profile"),
            "{err:#}"
        );
        assert_eq!(sidecar_text(&f, "9f22dd1b-aaaa"), before);

        // With a configured candidate, the historical profile is not among
        // those scored, however good its numbers.
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        let prepared = prologue_account(&f, &spent, "9f22dd1b-aaaa", false, None).unwrap();
        assert_eq!(prepared.account.unwrap().profile, "team03");
    }

    #[test]
    fn a_config_dir_selection_needs_config_dir_credentials() {
        // The session ran under `token`; the config now says `config-dir`, and
        // no profile directory has been populated, so selection has nothing to
        // choose from and says which file it looked for.
        let mut f = Fixture::new("resume-select-no-config-dir");
        f.token("team03");
        f.token("team04");
        f.cfg.claude.auth = ClaudeAuth::ConfigDir;
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 0.0)]);
        let before = save_record(
            &f,
            "9f22dd1b-aaaa",
            &claude_record(&f, Some("team03"), ClaudeAuth::Token),
        );
        let err = prologue_account(&f, &usage, "9f22dd1b-aaaa", false, None)
            .err()
            .expect("no config-dir credential");
        let err = format!("{err:#}");
        assert!(err.contains("no configured Claude profile"), "{err}");
        assert!(err.contains(".credentials.json"), "{err}");
        assert_eq!(sidecar_text(&f, "9f22dd1b-aaaa"), before);
    }

    #[test]
    fn a_resume_of_a_record_without_an_account_selects_normally() {
        let f = Fixture::new("resume-no-account");
        f.token("team03");
        f.token("team04");
        let usage = || {
            report(vec![
                snapshot(Timestamp::now(), "team03", 0.0, 90.0),
                snapshot(Timestamp::now(), "team04", 0.0, 10.0),
            ])
        };
        let mut record = claude_record(&f, None, ClaudeAuth::Token);
        record.auth = None;
        save_record(&f, "9f22dd1b-aaaa", &record);
        let prepared = prologue_account(&f, &usage, "9f22dd1b-aaaa", false, None).unwrap();
        assert_eq!(prepared.account.unwrap().profile, "team04");
        assert_eq!(prepared.record.profile.as_deref(), Some("team04"));
        assert_eq!(prepared.record.auth, Some(ClaudeAuth::Token));
    }

    #[test]
    fn an_explicit_resume_profile_bypasses_usage_under_the_configured_auth() {
        let f = Fixture::new("resume-explicit");
        f.token("team03");
        // Unlisted, and recorded under another auth mode.
        f.token("team09");
        save_record(
            &f,
            "9f22dd1b-aaaa",
            &claude_record(&f, Some("team03"), ClaudeAuth::ConfigDir),
        );
        let usage = usage_not_consulted;
        for fork in [false, true] {
            let prepared =
                prologue_account(&f, &usage, "9f22dd1b-aaaa", fork, Some("team09")).unwrap();
            let account = prepared.account.unwrap();
            assert_eq!(account.profile, "team09", "{fork}");
            assert_eq!(account.auth, ClaudeAuth::Token, "{fork}");
            assert!(prepared.notes.is_empty(), "{:?}", prepared.notes);
            assert_eq!(prepared.record.profile.as_deref(), Some("team09"));
            assert_eq!(prepared.record.auth, Some(ClaudeAuth::Token));
        }

        // A missing credential fails without falling back or saving.
        let before = sidecar_text(&f, "9f22dd1b-aaaa");
        for fork in [false, true] {
            let err = prologue_account(&f, &usage, "9f22dd1b-aaaa", fork, Some("team05"))
                .err()
                .expect("no credential");
            assert!(format!("{err:#}").contains("no credential"), "{err:#}");
            assert_eq!(sidecar_text(&f, "9f22dd1b-aaaa"), before);
        }
    }

    #[test]
    fn a_fork_selects_its_own_account_and_leaves_the_parent_alone() {
        let f = Fixture::new("fork-select");
        f.token("team03");
        f.token("team04");
        let parent = seed_parent(&f, "9f22dd1b-aaaa", Harness::Claude);
        let usage = || {
            report(vec![
                snapshot(Timestamp::now(), "team03", 30.0, 60.0),
                snapshot(Timestamp::now(), "team04", 0.0, 10.0),
            ])
        };
        let auto = prologue_account(&f, &usage, "9f22dd1b-aaaa", true, None).unwrap();
        assert_eq!(auto.record.profile.as_deref(), Some("team04"));
        assert_eq!(auto.record.forked_from.as_deref(), Some("9f22dd1b-aaaa"));
        let explicit = prologue_account(
            &f,
            &usage_not_consulted,
            "9f22dd1b-aaaa",
            true,
            Some("team03"),
        )
        .unwrap();
        assert_eq!(explicit.record.profile.as_deref(), Some("team03"));
        assert_eq!(explicit.record.auth, Some(ClaudeAuth::Token));
        assert_eq!(sidecar_text(&f, "9f22dd1b-aaaa"), parent);
        assert_eq!(sidecar_names(&f), vec!["9f22dd1b-aaaa.json".to_string()]);
    }

    #[test]
    fn a_codex_resume_ignores_the_profile_and_consults_no_usage() {
        let f = Fixture::new("resume-codex-profile");
        f.token("team03");
        seed_parent(&f, "thread-1", Harness::Codex);
        for fork in [false, true] {
            let prepared =
                prologue_account(&f, &usage_not_consulted, "thread-1", fork, Some("team03"))
                    .unwrap();
            assert!(prepared.account.is_none());
            assert!(prepared.notes.is_empty());
            assert_eq!(prepared.record.profile, None);
            assert_eq!(prepared.record.auth, None);
        }
        let saved = session::load_from(&f.sessions_dir(), "thread-1").unwrap();
        assert_eq!(saved.profile, None);
    }

    #[test]
    fn a_run_spawns_the_harness_records_the_session_and_prints_the_trailer() {
        let mut f = Fixture::new("run-end-to-end");
        f.token("team03");
        f.token("team04");
        f.claude_script(OK_RESULT);
        let usage = || {
            report(vec![
                snapshot(Timestamp::now(), "team03", 0.0, 10.0),
                snapshot(Timestamp::now(), "team04", 0.0, 90.0),
            ])
        };
        let ctx = f.ctx(&usage);

        let mut out: Vec<u8> = Vec::new();
        let completion = run_with(
            &ctx,
            "explorer",
            Some(&f.work()),
            "go and look".to_string(),
            None,
            None,
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(completion, 0);

        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("the answer"), "{text}");
        assert!(
            text.contains("[ccc] session=9f22dd1b-aaaa harness=claude model=claude-haiku-4-5 effort=low profile=team03"),
            "{text}"
        );

        // The child was called as a one-shot run, in the requested directory,
        // with the prompt on stdin.
        let argv: Vec<String> = f.call("argv").lines().map(String::from).collect();
        assert_eq!(argv[0], "-p");
        assert!(argv.contains(&"--agent".to_string()), "{argv:?}");
        assert!(!argv.contains(&"--resume".to_string()), "{argv:?}");
        let i = argv
            .iter()
            .position(|a| a == "--permission-mode")
            .unwrap_or_else(|| panic!("--permission-mode: {argv:?}"));
        assert_eq!(argv[i + 1], "auto", "{argv:?}");
        assert_eq!(f.call("prompt"), "go and look");
        assert_eq!(
            std::fs::canonicalize(f.call("cwd").trim()).unwrap(),
            std::fs::canonicalize(f.path("work")).unwrap()
        );

        // The sidecar records what a resume has to reproduce.
        let record = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
        assert_eq!(record.agent.as_deref(), Some("explorer"));
        assert_eq!(record.model, "claude-haiku-4-5");
        assert_eq!(record.profile.as_deref(), Some("team03"));
        assert_eq!(record.auth, Some(ClaudeAuth::Token));
        assert_eq!(record.tools, Some(vec!["Read".into(), "Grep".into()]));
        assert_eq!(record.cwd, std::fs::canonicalize(f.path("work")).unwrap());
    }

    #[test]
    fn a_resume_replays_the_recorded_invocation() {
        let mut f = Fixture::new("resume-end-to-end");
        f.token("team03");
        f.claude_script(OK_RESULT);
        f.cfg.claude.profiles = vec!["team03".into()];
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);
        let ctx = f.ctx(&usage);

        let mut out: Vec<u8> = Vec::new();
        run_with(
            &ctx,
            "explorer",
            Some(&f.work()),
            "first".to_string(),
            None,
            None,
            false,
            &mut out,
        )
        .unwrap();
        let first = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();

        // A second turn, started from somewhere else entirely: the recorded
        // working directory is the one the child sees.
        let mut out: Vec<u8> = Vec::new();
        let code = resume_with(
            &ctx,
            "9f22dd1b-aaaa",
            false,
            None,
            "again".to_string(),
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(code, 0);

        let argv: Vec<String> = f.call("argv").lines().map(String::from).collect();
        let i = argv.iter().position(|a| a == "--resume").expect("--resume");
        assert_eq!(argv[i + 1], "9f22dd1b-aaaa");
        assert!(!argv.iter().any(|a| a == "--fork-session"), "{argv:?}");
        assert_eq!(f.call("prompt"), "again");
        assert_eq!(
            std::fs::canonicalize(f.call("cwd").trim()).unwrap(),
            first.cwd
        );

        let second = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
        assert_eq!(second.created_at, first.created_at);
        assert_eq!(second.profile.as_deref(), Some("team03"));
        assert_eq!(second.auth, Some(ClaudeAuth::Token));
    }

    #[cfg(unix)]
    #[test]
    fn fresh_claude_sidecar_precedes_harness_and_lock_is_held_during_turn() {
        let mut f = Fixture::new("fresh-claude-early-sidecar");
        f.token("team03");
        let marker = f.path("calls/pre-sidecar");
        let marker_text = marker.display().to_string();
        let sessions = f.sessions_dir().display().to_string();
        f.claude_custom_script(&format!(
            "cat >/dev/null; id=; previous=; for arg in \"$@\"; do [ \"$previous\" = --session-id ] && id=\"$arg\"; previous=\"$arg\"; done; test -s \"{sessions}/$id.json\"; flock -n \"{sessions}/$id.lock\" -c : && exit 41; printf '%s' ok > {marker_text}; printf '%s' '{OK_RESULT}'",
        ));
        let mut input = "go".as_bytes();
        let mut output = Vec::new();
        let code = run_from(
            "explorer",
            Some(&f.work()),
            Some("go"),
            None,
            None,
            None,
            false,
            &f.env,
            || Ok(f.cfg.clone()),
            &mut input,
            &mut output,
        )
        .unwrap();
        assert_eq!(code, 0);
        assert_eq!(std::fs::read_to_string(marker).unwrap(), "ok");
        assert!(std::fs::read_dir(f.sessions_dir()).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".lock")
        }));
    }

    #[cfg(unix)]
    #[test]
    fn terminated_codex_persists_one_early_session_event_and_trailer() {
        let _signal_lock = PROCESS_SIGNAL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut f = Fixture::new("codex-early-terminated");
        f.env.insert("CCC_HARNESS_ALL".into(), "codex".into());
        f.env
            .insert("CCC_CODEX_MODEL_EXPLORER".into(), "gpt-5.6-luna".into());
        f.codex_script(
            "printf '%s\\n' '{\"type\":\"thread.started\",\"thread_id\":\"thread-early\"}'; sleep 10",
        );
        supervise::set_test_signal(true);
        let mut input = "go".as_bytes();
        let mut output = Vec::new();
        let started = Instant::now();
        let code = run_from(
            "explorer",
            Some(&f.work()),
            Some("go"),
            None,
            None,
            None,
            false,
            &f.env,
            || Ok(f.cfg.clone()),
            &mut input,
            &mut output,
        )
        .unwrap();
        assert!(started.elapsed() < Duration::from_secs(3));
        assert_eq!(code, crate::harness::EXIT_NO_RESULT);
        let record = session::load_from(&f.sessions_dir(), "thread-early").unwrap();
        assert_eq!(record.harness, Harness::Codex);
        let records = latest_records(&f.env);
        assert_eq!(
            records
                .iter()
                .filter(|r| matches!(r, audit::Record::Session(_)))
                .count(),
            1
        );
        let end = records
            .into_iter()
            .find_map(|r| match r {
                audit::Record::End(e) => Some(e),
                _ => None,
            })
            .unwrap();
        assert_eq!(end.payload.session.as_deref(), Some("thread-early"));
        assert!(String::from_utf8(output)
            .unwrap()
            .contains("session=thread-early"));
    }

    #[cfg(unix)]
    #[test]
    fn codex_completion_and_best_effort_sidecar_and_audit_failures_are_independent() {
        let mut normal = Fixture::new("codex-normal-sidecar");
        normal.env.insert("CCC_HARNESS_ALL".into(), "codex".into());
        normal
            .env
            .insert("CCC_CODEX_MODEL_EXPLORER".into(), "gpt-5.6-luna".into());
        normal.codex_script("printf '%s\\n' '{\"type\":\"thread.started\",\"thread_id\":\"thread-normal\"}' '{\"type\":\"item.completed\",\"item\":{\"type\":\"agent_message\",\"text\":\"answer\"}}' '{\"type\":\"turn.completed\",\"usage\":{}}'");
        let mut input = "go".as_bytes();
        let mut output = Vec::new();
        assert_eq!(
            run_from(
                "explorer",
                Some(&normal.work()),
                Some("go"),
                None,
                None,
                None,
                false,
                &normal.env,
                || Ok(normal.cfg.clone()),
                &mut input,
                &mut output
            )
            .unwrap(),
            0
        );
        assert!(session::load_from(&normal.sessions_dir(), "thread-normal").is_ok());
        assert_eq!(
            latest_records(&normal.env)
                .iter()
                .filter(|r| matches!(r, audit::Record::Session(_)))
                .count(),
            1
        );

        let mut sidecar = Fixture::new("codex-sidecar-warning");
        sidecar.env.insert("CCC_HARNESS_ALL".into(), "codex".into());
        sidecar
            .env
            .insert("CCC_CODEX_MODEL_EXPLORER".into(), "gpt-5.6-luna".into());
        let blocked_sessions = sidecar.sessions_dir().display().to_string();
        sidecar.codex_script(&format!("rm -rf '{blocked_sessions}'; printf blocked > '{blocked_sessions}'; printf '%s\\n' '{{\"type\":\"thread.started\",\"thread_id\":\"thread-sidecar-warning\"}}'"));
        let mut input = "go".as_bytes();
        let mut output = Vec::new();
        run_from(
            "explorer",
            Some(&sidecar.work()),
            Some("go"),
            None,
            None,
            None,
            false,
            &sidecar.env,
            || Ok(sidecar.cfg.clone()),
            &mut input,
            &mut output,
        )
        .unwrap();
        assert_eq!(
            latest_records(&sidecar.env)
                .iter()
                .filter(|r| matches!(r, audit::Record::Session(_)))
                .count(),
            1
        );

        let mut audit_failure = Fixture::new("codex-audit-warning");
        audit_failure
            .env
            .insert("CCC_HARNESS_ALL".into(), "codex".into());
        audit_failure
            .env
            .insert("CCC_CODEX_MODEL_EXPLORER".into(), "gpt-5.6-luna".into());
        let trace_path = Recorder::dir_in(&audit_failure.env)
            .join(format!("{}.jsonl", audit_failure.audit.lineage().trace_id));
        audit_failure.codex_script(&format!("rm -f '{}'; mkdir '{}'; printf '%s\\n' '{{\"type\":\"thread.started\",\"thread_id\":\"thread-audit-warning\"}}'", trace_path.display(), trace_path.display()));
        let mut input = "go".as_bytes();
        let mut output = Vec::new();
        run_from(
            "explorer",
            Some(&audit_failure.work()),
            Some("go"),
            None,
            None,
            None,
            false,
            &audit_failure.env,
            || Ok(audit_failure.cfg.clone()),
            &mut input,
            &mut output,
        )
        .unwrap();
        assert!(session::load_from(&audit_failure.sessions_dir(), "thread-audit-warning").is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn resume_rewrites_ownership_before_spawn_and_retains_it_on_failure() {
        let mut f = Fixture::new("resume-prewrite");
        f.token("team03");
        let old = "2020-01-01T00:00:00Z".parse().unwrap();
        let mut record = record_of(&resolved(), Path::new(&f.work()), Some(&account()), old);
        record.last_used_at = old;
        record.trace = Some("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa".into());
        record.inv = Some("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb".into());
        session::save_in(&f.sessions_dir(), "resume-prewrite", &record).unwrap();
        let marker = f.path("calls/resume-record");
        let marker_text = marker.display().to_string();
        let sessions = f.sessions_dir().display().to_string();
        let current = f.audit.lineage();
        let current_trace = current.trace_id.clone();
        let current_invocation = current.invocation_id.clone();
        let second_drive = f.path("calls/second-drive");
        f.claude_custom_script(&format!(
            "if [ ! -e '{}' ]; then touch '{}'; grep -q '{}' '{sessions}/resume-prewrite.json' && grep -q '{}' '{sessions}/resume-prewrite.json' && ! grep -q 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa' '{sessions}/resume-prewrite.json' && ! grep -q 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb' '{sessions}/resume-prewrite.json' && printf ok > '{marker_text}'; printf '%s' '{OK_RESULT}'; else printf '%s' '{{\"not_a_completion\":true}}'; fi",
            second_drive.display(),
            second_drive.display(),
            current_trace,
            current_invocation
        ));
        start_resume(&f, "resume-prewrite");
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 0.0)]);
        let ctx = f.ctx(&usage);
        let mut output = Vec::new();
        let completion = resume_with(
            &ctx,
            "resume-prewrite",
            false,
            None,
            "again".into(),
            false,
            &mut output,
        )
        .unwrap();
        let updated = session::load_from(&f.sessions_dir(), "resume-prewrite").unwrap();
        assert!(updated.last_used_at > old);
        assert_eq!(updated.profile.as_deref(), Some("team03"));
        assert_eq!(updated.auth, Some(ClaudeAuth::Token));
        assert!(updated.trace.is_some() && updated.inv.is_some());
        assert_eq!(std::fs::read_to_string(marker).unwrap(), "ok");
        f.audit.end(completion.end);

        let failure = resume_with(
            &ctx,
            "resume-prewrite",
            false,
            None,
            "again".into(),
            false,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(failure.code, 3);
        let retained = session::load_from(&f.sessions_dir(), "resume-prewrite").unwrap();
        assert_eq!(retained.trace.as_deref(), Some(current_trace.as_str()));
        assert_eq!(retained.inv.as_deref(), Some(current_invocation.as_str()));
    }

    #[cfg(unix)]
    #[test]
    fn a_locked_resume_is_exit_two_with_detailed_or_legacy_admission_diagnostics() {
        let mut f = Fixture::new("resume-lock-admission");
        f.token("team03");
        f.install_cli_config();
        f.claude_script(OK_RESULT);
        start_resume(&f, "locked-session");
        let mut record = record_of(&resolved(), Path::new(&f.work()), Some(&account()), f.now);
        record.trace = Some(f.audit.lineage().trace_id.clone());
        record.inv = Some(f.audit.lineage().invocation_id.clone());
        session::save_in(&f.sessions_dir(), "locked-session", &record).unwrap();
        let lock = session::try_lock_in(&f.sessions_dir(), "locked-session")
            .unwrap()
            .unwrap();
        {
            let usage = || UsageSource::Unavailable("unused".into());
            let ctx = f.ctx(&usage);
            let err = resume_with(
                &ctx,
                "locked-session",
                false,
                None,
                "again".into(),
                false,
                &mut Vec::new(),
            )
            .unwrap_err();
            let text = format!("{err:#}");
            assert!(
                text.contains("invocation") && text.contains("ccc pid"),
                "{text}"
            );
            assert_eq!(error_end(&err.error).exit, Some(2));
        }

        let mut legacy = record.clone();
        legacy.trace = None;
        legacy.inv = None;
        session::save_in(&f.sessions_dir(), "locked-session", &legacy).unwrap();
        {
            let usage = || UsageSource::Unavailable("unused".into());
            let ctx = f.ctx(&usage);
            let err = resume_with(
                &ctx,
                "locked-session",
                false,
                None,
                "again".into(),
                false,
                &mut Vec::new(),
            )
            .unwrap_err();
            assert!(format!("{err:#}").contains("wait for it to finish"));
            assert_eq!(error_end(&err.error).exit, Some(2));
        }
        drop(lock);
        let _released = session::try_lock_in(&f.sessions_dir(), "locked-session");
    }

    #[test]
    fn a_one_shot_claude_fork_saves_only_the_forks_sidecar() {
        let mut f = Fixture::new("fork-claude");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        let parent = seed_parent(&f, "9f22dd1b-aaaa", Harness::Claude);
        f.claude_script(FORK_RESULT);
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);
        let ctx = f.ctx(&usage);

        let mut out: Vec<u8> = Vec::new();
        let completion = resume_with(
            &ctx,
            "9f22dd1b-aaaa",
            true,
            None,
            "again".into(),
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(completion, 0);
        assert_eq!(completion.end.session.as_deref(), Some("fork-1111"));

        let argv: Vec<String> = f.call("argv").lines().map(String::from).collect();
        let i = argv.iter().position(|a| a == "--resume").expect("--resume");
        assert_eq!(argv[i + 1], "9f22dd1b-aaaa");
        assert_eq!(argv[i + 2], "--fork-session");
        assert!(!argv.iter().any(|a| a == "--session-id"), "{argv:?}");

        assert_eq!(sidecar_text(&f, "9f22dd1b-aaaa"), parent);
        let fork = session::load_from(&f.sessions_dir(), "fork-1111").unwrap();
        assert_eq!(fork.forked_from.as_deref(), Some("9f22dd1b-aaaa"));
        assert_eq!(fork.created_at, f.now);
        assert_eq!(fork.last_used_at, f.now);
        assert_eq!(fork.profile.as_deref(), Some("team03"));
        let lineage = f.audit.lineage();
        assert_eq!(fork.trace.as_deref(), Some(lineage.trace_id.as_str()));
        assert_eq!(fork.inv.as_deref(), Some(lineage.invocation_id.as_str()));
        assert!(String::from_utf8(out)
            .unwrap()
            .contains("session=fork-1111"));
        assert!(lock_is_free(&f, "9f22dd1b-aaaa"));
    }

    #[test]
    fn a_failed_claude_fork_turn_saves_under_the_reported_id() {
        let mut f = Fixture::new("fork-claude-failed");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        let parent = seed_parent(&f, "9f22dd1b-aaaa", Harness::Claude);
        f.claude_script(
            r#"{"type":"result","session_id":"fork-2222","result":"nope","is_error":true}"#,
        );
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);
        let ctx = f.ctx(&usage);

        let completion = resume_with(
            &ctx,
            "9f22dd1b-aaaa",
            true,
            None,
            "again".into(),
            false,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(completion, crate::harness::EXIT_HARNESS_FAILED);
        let fork = session::load_from(&f.sessions_dir(), "fork-2222").unwrap();
        assert_eq!(fork.forked_from.as_deref(), Some("9f22dd1b-aaaa"));
        assert_eq!(sidecar_text(&f, "9f22dd1b-aaaa"), parent);
    }

    #[test]
    fn a_failed_claude_fork_turn_reported_under_the_parent_is_not_recorded() {
        let mut f = Fixture::new("fork-claude-failed-same-id");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        let parent = seed_parent(&f, "9f22dd1b-aaaa", Harness::Claude);
        f.claude_script(
            r#"{"type":"result","session_id":"9f22dd1b-aaaa","result":"nope","is_error":true}"#,
        );
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);
        let ctx = f.ctx(&usage);

        let completion = resume_with(
            &ctx,
            "9f22dd1b-aaaa",
            true,
            None,
            "again".into(),
            false,
            &mut Vec::new(),
        )
        .unwrap();
        assert_ne!(completion.code, 0);
        assert_eq!(sidecar_text(&f, "9f22dd1b-aaaa"), parent);
        assert_eq!(sidecar_names(&f), vec!["9f22dd1b-aaaa.json".to_string()]);
        assert!(
            completion
                .end
                .warnings
                .iter()
                .any(|w| w.contains("parent's id 9f22dd1b-aaaa")),
            "{:?}",
            completion.end.warnings
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_dead_claude_fork_names_no_session() {
        let _signal_lock = PROCESS_SIGNAL_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut f = Fixture::new("fork-claude-dead");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        let parent = seed_parent(&f, "9f22dd1b-aaaa", Harness::Claude);
        f.claude_custom_script("cat >/dev/null; sleep 5");
        supervise::set_test_signal(true);
        let mut input = "again".as_bytes();
        let mut output = Vec::new();
        let started = Instant::now();
        let code = resume_from(
            "9f22dd1b-aaaa",
            true,
            None,
            Some("again"),
            None,
            false,
            &f.env,
            || Ok(f.cfg.clone()),
            &mut input,
            &mut output,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(code, crate::harness::EXIT_NO_RESULT);
        assert!(started.elapsed() < Duration::from_secs(3));
        let text = String::from_utf8(output).unwrap();
        assert!(!text.contains("session="), "{text}");

        let records = latest_records(&f.env);
        let start = records
            .iter()
            .find_map(|r| match r {
                audit::Record::Start(s) => Some(&s.payload),
                _ => None,
            })
            .unwrap();
        assert!(start.fork);
        assert_eq!(start.session.as_deref(), Some("9f22dd1b-aaaa"));
        let spawn = records
            .iter()
            .find_map(|r| match r {
                audit::Record::Spawn(s) => Some(&s.payload),
                _ => None,
            })
            .unwrap();
        assert_eq!(spawn.session, None);
        let end = records
            .iter()
            .find_map(|r| match r {
                audit::Record::End(e) => Some(&e.payload),
                _ => None,
            })
            .unwrap();
        assert_eq!(end.session, None);
        assert_eq!(sidecar_names(&f), vec!["9f22dd1b-aaaa.json".to_string()]);
        assert_eq!(sidecar_text(&f, "9f22dd1b-aaaa"), parent);

        f.env.insert(
            "CCC_CLAUDE_BIN".to_string(),
            f.path("nonexistent").display().to_string(),
        );
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);
        let ctx = f.ctx(&usage);
        let completion = resume_with(
            &ctx,
            "9f22dd1b-aaaa",
            true,
            None,
            "again".into(),
            false,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(completion, crate::harness::EXIT_NO_RESULT);
        assert_eq!(completion.end.session, None);
        assert_eq!(sidecar_names(&f), vec!["9f22dd1b-aaaa.json".to_string()]);
        assert_eq!(sidecar_text(&f, "9f22dd1b-aaaa"), parent);
    }

    #[test]
    fn a_one_shot_codex_fork_discovers_the_thread_with_the_fork_record() {
        let mut f = Fixture::new("fork-codex");
        let parent = seed_parent(&f, "thread-parent", Harness::Codex);
        let calls = f.path("calls");
        std::fs::create_dir_all(&calls).unwrap();
        f.codex_script(&format!(
            "printf '%s\\n' \"$@\" > '{}/argv'; printf '%s\\n' '{{\"type\":\"thread.started\",\"thread_id\":\"fork-thread\"}}' '{{\"type\":\"item.completed\",\"item\":{{\"type\":\"agent_message\",\"text\":\"forked\"}}}}' '{{\"type\":\"turn.completed\",\"usage\":{{}}}}'",
            calls.display()
        ));
        let usage = || UsageSource::Unavailable("not consulted".into());
        let ctx = f.ctx(&usage);

        let completion = resume_with(
            &ctx,
            "thread-parent",
            true,
            None,
            "again".into(),
            false,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(completion, 0);

        let argv: Vec<String> = f.call("argv").lines().map(String::from).collect();
        assert_eq!(&argv[..4], &["exec", "fork", "thread-parent", "--json"]);
        let config = |key: &str| {
            argv.windows(2)
                .find(|w| w[0] == "-c" && w[1].starts_with(&format!("{key}=")))
                .map(|w| w[1].clone())
        };
        assert_eq!(config("model").as_deref(), Some("model=\"gpt-5.6-luna\""));
        assert!(config("sandbox_mode").is_some(), "{argv:?}");
        assert!(!argv.iter().any(|a| a == "-C" || a == "-s"), "{argv:?}");

        let fork = session::load_from(&f.sessions_dir(), "fork-thread").unwrap();
        assert_eq!(fork.forked_from.as_deref(), Some("thread-parent"));
        assert_eq!(fork.created_at, f.now);
        assert_eq!(
            fork.inv.as_deref(),
            Some(ctx.audit.lineage().invocation_id.as_str())
        );
        assert_eq!(
            fork.trace.as_deref(),
            Some(ctx.audit.lineage().trace_id.as_str())
        );
        let records = audit_records(&f);
        let sessions: Vec<&str> = records
            .iter()
            .filter_map(|r| match r {
                audit::Record::Session(s) => Some(s.session.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(sessions, vec!["fork-thread"]);
        let spawn = records
            .iter()
            .find_map(|r| match r {
                audit::Record::Spawn(s) => Some(&s.payload),
                _ => None,
            })
            .unwrap();
        assert_eq!(spawn.session, None);
        assert_eq!(sidecar_text(&f, "thread-parent"), parent);
    }

    #[test]
    fn a_codex_fork_whose_thread_started_echoes_the_parent_is_not_recorded() {
        let mut f = Fixture::new("fork-codex-same-id");
        let parent = seed_parent(&f, "thread-parent", Harness::Codex);
        f.codex_script(
            "printf '%s\\n' '{\"type\":\"thread.started\",\"thread_id\":\"thread-parent\"}' '{\"type\":\"item.completed\",\"item\":{\"type\":\"agent_message\",\"text\":\"forked\"}}' '{\"type\":\"turn.completed\",\"usage\":{}}'",
        );
        let usage = || UsageSource::Unavailable("not consulted".into());
        let ctx = f.ctx(&usage);

        let completion = resume_with(
            &ctx,
            "thread-parent",
            true,
            None,
            "again".into(),
            false,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(completion.code, 0);
        assert_eq!(sidecar_text(&f, "thread-parent"), parent);
        assert_eq!(sidecar_names(&f), vec!["thread-parent.json".to_string()]);
        assert!(lock_is_free(&f, "thread-parent"));
        assert!(
            completion
                .end
                .warnings
                .iter()
                .any(|w| w.contains("parent's id thread-parent")),
            "{:?}",
            completion.end.warnings
        );
        let sessions: Vec<String> = audit_records(&f)
            .iter()
            .filter_map(|r| match r {
                audit::Record::Session(s) => Some(s.session.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(sessions, vec!["thread-parent".to_string()]);
    }

    #[test]
    fn a_fork_of_a_locked_parent_proceeds() {
        let mut f = Fixture::new("fork-locked-parent");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        seed_parent(&f, "9f22dd1b-aaaa", Harness::Claude);
        let held = session::try_lock_in(&f.sessions_dir(), "9f22dd1b-aaaa")
            .unwrap()
            .unwrap();
        f.claude_script(FORK_RESULT);
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);
        let ctx = f.ctx(&usage);

        let completion = resume_with(
            &ctx,
            "9f22dd1b-aaaa",
            true,
            None,
            "again".into(),
            false,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(completion, 0);
        assert!(session::load_from(&f.sessions_dir(), "fork-1111").is_ok());

        let err = resume_with(
            &ctx,
            "9f22dd1b-aaaa",
            false,
            None,
            "again".into(),
            false,
            &mut Vec::new(),
        )
        .unwrap_err();
        assert_eq!(error_end(&err.error).exit, Some(2));
        drop(held);
    }

    #[test]
    fn a_fork_never_takes_the_parents_lock() {
        let mut f = Fixture::new("fork-no-parent-lock");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        seed_parent(&f, "9f22dd1b-aaaa", Harness::Claude);
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);
        let ctx = f.ctx(&usage);
        let held = session::try_lock_in(&f.sessions_dir(), "9f22dd1b-aaaa")
            .unwrap()
            .unwrap();

        let prepared =
            resume_prologue(&ctx, "9f22dd1b-aaaa", true, AccountPick::Requested(None)).unwrap();
        assert_eq!(
            prepared.record.forked_from.as_deref(),
            Some("9f22dd1b-aaaa")
        );
        drop(held);
        assert!(!ctx.owns_session("9f22dd1b-aaaa"));
        assert!(lock_is_free(&f, "9f22dd1b-aaaa"));
    }

    #[cfg(unix)]
    #[test]
    fn recorded_owner_finds_the_last_child_of_the_owning_invocation() {
        let f = Fixture::new("recorded-owner");
        let usage = || UsageSource::Unavailable("not consulted".into());
        let ctx = f.ctx(&usage);
        let mut record = record_of(&resolved(), Path::new(&f.work()), Some(&account()), f.now);
        assert!(recorded_owner(&ctx, &record).unwrap().is_none());

        start_launch(&f);
        let me = crate::procs::ProcId::current().unwrap();
        f.audit
            .child(ChildPayload {
                proc: me.clone(),
                pgid: std::process::id(),
                supervisor: me,
            })
            .unwrap();
        let lineage = f.audit.lineage();
        record.trace = Some(lineage.trace_id.clone());
        record.inv = Some(lineage.invocation_id.clone());
        let (inv, proc) = recorded_owner(&ctx, &record).unwrap().expect("owner");
        assert_eq!(inv, lineage.invocation_id);
        assert_eq!(proc.pid, std::process::id());

        record.trace = Some(Uuid::new_v4().to_string());
        assert!(recorded_owner(&ctx, &record).unwrap().is_none());
    }

    #[test]
    fn a_fork_that_comes_back_as_its_parent_is_not_recorded() {
        let mut f = Fixture::new("fork-same-id");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        let parent = seed_parent(&f, "9f22dd1b-aaaa", Harness::Claude);
        f.claude_script(OK_RESULT);
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);
        let ctx = f.ctx(&usage);

        let completion = resume_with(
            &ctx,
            "9f22dd1b-aaaa",
            true,
            None,
            "again".into(),
            false,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(completion, 0);
        assert_eq!(sidecar_text(&f, "9f22dd1b-aaaa"), parent);
        assert_eq!(sidecar_names(&f), vec!["9f22dd1b-aaaa.json".to_string()]);
        assert!(
            completion
                .end
                .warnings
                .iter()
                .any(|w| w.contains("parent's id 9f22dd1b-aaaa")),
            "{:?}",
            completion.end.warnings
        );
    }

    #[cfg(unix)]
    fn start_launch_resume(f: &Fixture, session: &str, fork: bool) {
        f.audit
            .start(StartPayload {
                pid: std::process::id(),
                cmd: AuditCommand::Launch,
                agent: None,
                session: Some(session.into()),
                cwd_arg: None,
                profile_arg: None,
                harness_arg: None,
                json: false,
                prompt_source: None,
                prompt_file: None,
                prompt: None,
                passthrough: vec![],
                fork,
            })
            .unwrap();
    }

    /// A shell line appending one `turn` record for this invocation, as the
    /// harness's Stop hook or notify callback would.
    #[cfg(unix)]
    fn turn_line(f: &Fixture, harness: &str, session: &str) -> String {
        format!(
            r#"printf '{{"t":"turn","ts":"2026-01-01T00:00:00Z","inv":"%s","harness":"{harness}","session":"{session}","turn":null,"result":"turn result","transcript":null}}\n' "$CCC_PARENT_INVOCATION_ID" >> "{}/$CCC_TRACE_ID.jsonl""#,
            f.path("state/ccc/audit").display()
        )
    }

    /// A shell line recording the harness's argv, cwd and environment.
    #[cfg(unix)]
    fn record_call_line(f: &Fixture) -> String {
        let calls = f.path("calls");
        std::fs::create_dir_all(&calls).unwrap();
        format!(
            "printf '%s\\n' \"$@\" > '{c}/argv'; pwd > '{c}/cwd'; env > '{c}/env'",
            c = calls.display()
        )
    }

    #[cfg(unix)]
    fn spawn_of(records: &[audit::Record]) -> &audit::SpawnRecord {
        records
            .iter()
            .find_map(|r| match r {
                audit::Record::Spawn(s) => Some(s),
                _ => None,
            })
            .expect("audit spawn")
    }

    #[cfg(unix)]
    fn start_of(records: &[audit::Record]) -> &audit::StartRecord {
        records
            .iter()
            .find_map(|r| match r {
                audit::Record::Start(s) => Some(s),
                _ => None,
            })
            .expect("audit start")
    }

    #[cfg(unix)]
    fn test_io_lock() -> std::sync::MutexGuard<'static, ()> {
        supervise::TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[cfg(unix)]
    fn argv_of(f: &Fixture) -> Vec<String> {
        f.call("argv").lines().map(String::from).collect()
    }

    #[cfg(unix)]
    fn claude_usage() -> UsageSource {
        report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)])
    }

    #[cfg(unix)]
    #[test]
    fn a_fresh_claude_launch_saves_its_sidecar_and_trusts_cwd_before_the_child_runs() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("launch-fresh-claude-sidecar");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        let sessions = f.sessions_dir().display().to_string();
        let work = std::fs::canonicalize(f.path("work")).unwrap();
        let claude_json = f.path(".claude.json").display().to_string();
        f.launch_script(
            &format!(
                "id=; previous=; for arg in \"$@\"; do [ \"$previous\" = --session-id ] && id=\"$arg\"; previous=\"$arg\"; done\n\
                 test -s \"{sessions}/$id.json\" || exit 41\n\
                 flock -n \"{sessions}/$id.lock\" -c : && exit 42\n\
                 grep -qF '{work}' '{claude_json}' || exit 43\n\
                 grep -q hasTrustDialogAccepted '{claude_json}' || exit 44\n\
                 exit 0",
                work = work.display()
            ),
            false,
        );
        start_launch(&f);
        let usage = || UsageSource::Unavailable("not consulted".into());
        let completion = launch_with_ctx(
            &f.ctx(&usage),
            LaunchTarget::Fresh {
                agent: "explorer",
                cwd: Some(&f.work()),
                cli_harness: None,
                allow_read: &[],
                allow_write: &[],
                allow_exec: &[],
            },
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(completion.code, 0);
        let names = sidecar_names(&f);
        assert_eq!(names.len(), 1, "{names:?}");
        let preset = names[0].trim_end_matches(".json").to_string();
        let records = launch_records(&f);
        assert_eq!(
            spawn_of(&records).payload.session.as_deref(),
            Some(preset.as_str())
        );
        let saved = session::load_from(&f.sessions_dir(), &preset).unwrap();
        assert_eq!(saved.agent.as_deref(), Some("explorer"));
        assert_eq!(saved.profile.as_deref(), Some("team03"));
        assert_eq!(saved.harness, Harness::Claude);
        let lineage = f.audit.lineage();
        assert_eq!(saved.trace.as_deref(), Some(lineage.trace_id.as_str()));
        assert_eq!(saved.inv.as_deref(), Some(lineage.invocation_id.as_str()));
        assert_eq!(completion.end.session.as_deref(), Some(preset.as_str()));
        assert!(
            completion.end.warnings.is_empty(),
            "{:?}",
            completion.end.warnings
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_roleless_launch_runs_claude_with_the_expanded_model_and_no_agent() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("launch-roleless");
        f.token("team03");
        f.token("team04");
        f.cfg.claude.profiles = vec!["team03".into(), "team04".into()];
        // The definition directory is empty: a roleless launch loads none.
        std::fs::remove_dir_all(f.path("agents")).unwrap();
        std::fs::create_dir_all(f.path("agents")).unwrap();
        let argv = f.path("argv");
        f.launch_script(
            &format!("printf '%s\\n' \"$@\" > '{}'\nexit 0", argv.display()),
            false,
        );
        let work = f.work();
        let target = LaunchTarget::Roleless {
            cwd: Some(&work),
            model: Some("fable"),
            effort: Some(Effort::High),
        };
        let passthrough = vec!["--verbose".to_string()];
        f.audit
            .start(launch_start(target, None, &passthrough))
            .unwrap();
        // Only team04 publishes a Fable window, so only it can run Fable,
        // though team03 has more of every other bucket left.
        let usage = || {
            let now = Timestamp::now();
            let mut team04 = snapshot(now, "team04", 50.0, 50.0);
            team04.windows.push(UsageWindow {
                key: "7-day_fable".to_string(),
                utilization: 50.0,
                resets_at: at(now, 100),
            });
            report(vec![snapshot(now, "team03", 0.0, 0.0), team04])
        };
        let completion = launch_with_ctx(&f.ctx(&usage), target, None, passthrough).unwrap();
        assert_eq!(completion.code, 0);

        let args: Vec<String> = std::fs::read_to_string(&argv)
            .unwrap()
            .lines()
            .map(str::to_string)
            .collect();
        let after = |flag: &str| {
            let i = args.iter().position(|a| a == flag).expect(flag);
            args[i + 1].clone()
        };
        assert_eq!(after("--model"), "fable[1m]");
        assert_eq!(after("--effort"), "high");
        assert!(!args.iter().any(|a| a == "--agent"), "{args:?}");
        assert!(!args.iter().any(|a| a == "--agents"), "{args:?}");
        assert_eq!(args.last().map(String::as_str), Some("--verbose"));
        let preset = after("--session-id");

        let records = launch_records(&f);
        let spawn = spawn_of(&records);
        assert_eq!(spawn.payload.agent, None);
        assert_eq!(spawn.payload.model, "fable[1m]");
        assert_eq!(spawn.payload.effort, Some(Effort::High));
        assert_eq!(spawn.payload.profile.as_deref(), Some("team04"));
        assert_eq!(spawn.payload.session.as_deref(), Some(preset.as_str()));

        let raw = std::fs::read_to_string(f.sessions_dir().join(format!("{preset}.json"))).unwrap();
        let json: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert!(json["agent"].is_null(), "{raw}");
        let saved = session::load_from(&f.sessions_dir(), &preset).unwrap();
        assert_eq!(saved.agent, None);
        assert_eq!(saved.model, "fable[1m]");
        assert_eq!(saved.profile.as_deref(), Some("team04"));
        assert_eq!(saved.harness, Harness::Claude);
    }

    #[cfg(unix)]
    #[test]
    fn a_roleless_launch_with_a_bad_launch_effort_fails_before_spawning() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("launch-roleless-bad-effort");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.cfg.launch.effort = Some("minimal".into());
        let ran = f.path("ran");
        f.launch_script(&format!("touch '{}'\nexit 0", ran.display()), false);
        let consulted = std::sync::atomic::AtomicBool::new(false);
        let usage = || {
            consulted.store(true, std::sync::atomic::Ordering::SeqCst);
            UsageSource::Unavailable("consulted".into())
        };
        let err = launch_with_ctx(
            &f.ctx(&usage),
            LaunchTarget::Roleless {
                cwd: Some(&f.work()),
                model: None,
                effort: None,
            },
            None,
            vec![],
        )
        .expect_err("a bad [launch] effort is an error");
        assert!(format!("{err:#}").contains("[launch] effort"), "{err:#}");
        assert!(!consulted.load(std::sync::atomic::Ordering::SeqCst));
        assert!(!ran.exists());
    }

    #[test]
    fn launch_start_records_each_target() {
        let passthrough = vec!["--verbose".to_string()];
        let fields = |target| {
            let s = launch_start(target, Some("p"), &passthrough);
            assert_eq!(s.cmd, AuditCommand::Launch);
            assert_eq!(s.profile_arg.as_deref(), Some("p"));
            assert_eq!(s.passthrough, passthrough);
            (s.agent, s.session, s.cwd_arg, s.harness_arg, s.fork)
        };
        assert_eq!(
            fields(LaunchTarget::Roleless {
                cwd: Some("/srv"),
                model: Some("fable"),
                effort: Some(Effort::High),
            }),
            (None, None, Some("/srv".to_string()), None, false)
        );
        assert_eq!(
            fields(LaunchTarget::Fresh {
                agent: "explorer",
                cwd: Some("/srv"),
                cli_harness: Some(Harness::Codex),
                allow_read: &[],
                allow_write: &[],
                allow_exec: &[],
            }),
            (
                Some("explorer".to_string()),
                None,
                Some("/srv".to_string()),
                Some(Harness::Codex),
                false
            )
        );
        assert_eq!(
            fields(LaunchTarget::Resume {
                id: "ID",
                fork: true
            }),
            (None, Some("ID".to_string()), None, None, true)
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_fresh_codex_launch_saves_the_sidecar_from_its_last_turn() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("launch-fresh-codex-sidecar");
        f.cfg.agents.insert(
            "explorer".into(),
            crate::config::AgentOverrides {
                codex_model: Some("gpt-5.6-terra".into()),
                ..Default::default()
            },
        );
        let script = format!("{}\nexit 0", turn_line(&f, "codex", "thread-launch"));
        f.launch_script(&script, true);
        start_launch(&f);
        let usage = || UsageSource::Unavailable("not consulted".into());
        let completion = launch_with_ctx(
            &f.ctx(&usage),
            LaunchTarget::Fresh {
                agent: "explorer",
                cwd: Some(&f.work()),
                cli_harness: Some(Harness::Codex),
                allow_read: &[],
                allow_write: &[],
                allow_exec: &[],
            },
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(completion.code, 0);
        let saved = session::load_from(&f.sessions_dir(), "thread-launch").unwrap();
        assert_eq!(saved.harness, Harness::Codex);
        assert_eq!(saved.model, "gpt-5.6-terra");
        assert_eq!(saved.agent.as_deref(), Some("explorer"));
        assert_eq!(saved.profile, None);
        let lineage = f.audit.lineage();
        assert_eq!(saved.trace.as_deref(), Some(lineage.trace_id.as_str()));
        assert_eq!(saved.inv.as_deref(), Some(lineage.invocation_id.as_str()));
        assert_eq!(spawn_of(&launch_records(&f)).payload.session, None);
        assert_eq!(completion.end.session.as_deref(), Some("thread-launch"));
        assert!(
            completion.end.warnings.is_empty(),
            "{:?}",
            completion.end.warnings
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_launch_that_ends_in_a_foreign_session_leaves_its_record_alone() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("launch-foreign-session");
        let foreign = seed_parent(&f, "thread-foreign", Harness::Codex);
        f.cfg.agents.insert(
            "explorer".into(),
            crate::config::AgentOverrides {
                codex_model: Some("gpt-5.6-terra".into()),
                ..Default::default()
            },
        );
        let script = format!("{}\nexit 0", turn_line(&f, "codex", "thread-foreign"));
        f.launch_script(&script, true);
        start_launch(&f);
        let usage = || UsageSource::Unavailable("not consulted".into());
        let completion = launch_with_ctx(
            &f.ctx(&usage),
            LaunchTarget::Fresh {
                agent: "explorer",
                cwd: Some(&f.work()),
                cli_harness: Some(Harness::Codex),
                allow_read: &[],
                allow_write: &[],
                allow_exec: &[],
            },
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(completion.code, 0);
        assert_eq!(sidecar_text(&f, "thread-foreign"), foreign);
        assert_eq!(sidecar_names(&f), vec!["thread-foreign.json".to_string()]);
        assert_eq!(completion.end.session.as_deref(), Some("thread-foreign"));
        assert!(
            completion.end.warnings.is_empty(),
            "{:?}",
            completion.end.warnings
        );
    }

    #[cfg(unix)]
    #[test]
    fn an_interactive_resume_that_ends_in_a_new_session_records_it_and_warns() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("launch-resume-new-session");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        seed_parent(&f, "9f22dd1b-aaaa", Harness::Claude);
        let seeded = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
        let script = format!("{}\nexit 0", turn_line(&f, "claude", "fresh-3333"));
        f.launch_script(&script, false);
        start_launch_resume(&f, "9f22dd1b-aaaa", false);
        let usage = claude_usage;
        let ctx = f.ctx(&usage);
        let completion = launch_with_ctx(
            &ctx,
            LaunchTarget::Resume {
                id: "9f22dd1b-aaaa",
                fork: false,
            },
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(completion.code, 0);
        let saved = session::load_from(&f.sessions_dir(), "fresh-3333").unwrap();
        assert_eq!(saved.agent, seeded.agent);
        assert_eq!(saved.harness, seeded.harness);
        assert_eq!(saved.model, seeded.model);
        assert_eq!(saved.cwd, seeded.cwd);
        assert_eq!(saved.forked_from, None);
        assert_eq!(
            sidecar_names(&f),
            vec![
                "9f22dd1b-aaaa.json".to_string(),
                "fresh-3333.json".to_string()
            ]
        );
        assert!(
            completion
                .end
                .warnings
                .iter()
                .any(|w| w.contains("reported session id fresh-3333, expected 9f22dd1b-aaaa")),
            "{:?}",
            completion.end.warnings
        );
        assert_eq!(completion.end.session.as_deref(), Some("fresh-3333"));
        drop(ctx);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_zero_turn_codex_launch_records_nothing_and_says_so() {
        let mut f = Fixture::new("launch-zero-turn-codex");
        f.install_cli_config();
        f.launch_script("exit 0", true);
        f.env
            .insert("CCC_CODEX_MODEL_EXPLORER".into(), "gpt-5.6-terra".into());
        let work = f.work();
        let output = f.cli(&["launch", "explorer", "--harness", "codex", "--cwd", &work]);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(0), "{stderr}");
        assert!(
            stderr.contains("note: no turn completed, so no session was recorded"),
            "{stderr}"
        );
        assert!(!f.sessions_dir().exists() || sidecar_names(&f).is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn an_interactive_resume_replays_the_record_for_claude() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("launch-resume-claude");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        seed_parent(&f, "9f22dd1b-aaaa", Harness::Claude);
        let seeded = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
        let script = format!(
            "{}\ngrep -q \"$CCC_PARENT_INVOCATION_ID\" '{}/9f22dd1b-aaaa.json' || exit 45\n{}\nexit 0",
            record_call_line(&f),
            f.sessions_dir().display(),
            turn_line(&f, "claude", "9f22dd1b-aaaa"),
        );
        f.launch_script(&script, false);
        start_launch_resume(&f, "9f22dd1b-aaaa", false);
        let usage = claude_usage;
        let ctx = f.ctx(&usage);
        let completion = launch_with_ctx(
            &ctx,
            LaunchTarget::Resume {
                id: "9f22dd1b-aaaa",
                fork: false,
            },
            None,
            vec!["--verbose".into()],
        )
        .unwrap();
        assert_eq!(completion.code, 0);

        let argv = argv_of(&f);
        let i = argv.iter().position(|a| a == "--resume").expect("--resume");
        assert_eq!(argv[i + 1], "9f22dd1b-aaaa");
        for absent in ["--fork-session", "-p", "--session-id"] {
            assert!(!argv.iter().any(|a| a == absent), "{absent}: {argv:?}");
        }
        assert!(argv.iter().any(|a| a == "--settings"), "{argv:?}");
        assert_eq!(argv.last().map(String::as_str), Some("--verbose"));
        assert_eq!(
            std::fs::canonicalize(f.call("cwd").trim()).unwrap(),
            std::fs::canonicalize(&seeded.cwd).unwrap()
        );
        assert_eq!(
            spawn_of(&launch_records(&f)).payload.session.as_deref(),
            Some("9f22dd1b-aaaa")
        );
        assert_eq!(completion.end.session.as_deref(), Some("9f22dd1b-aaaa"));
        assert!(
            completion.end.warnings.is_empty(),
            "{:?}",
            completion.end.warnings
        );
        let saved = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
        assert_eq!(saved.last_used_at, f.now);
        assert_eq!(saved.created_at, seeded.created_at);
        assert_eq!(saved.profile.as_deref(), Some("team03"));
        assert!(!lock_is_free(&f, "9f22dd1b-aaaa"));
        drop(ctx);
        assert!(lock_is_free(&f, "9f22dd1b-aaaa"));
    }

    #[cfg(unix)]
    #[test]
    fn an_interactive_resume_replays_the_record_for_codex() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("launch-resume-codex");
        seed_parent(&f, "thread-parent", Harness::Codex);
        let script = format!(
            "{}\n{}\nexit 0",
            record_call_line(&f),
            turn_line(&f, "codex", "thread-parent")
        );
        f.launch_script(&script, true);
        start_launch_resume(&f, "thread-parent", false);
        let usage = || UsageSource::Unavailable("not consulted".into());
        let completion = launch_with_ctx(
            &f.ctx(&usage),
            LaunchTarget::Resume {
                id: "thread-parent",
                fork: false,
            },
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(completion.code, 0);
        let argv = argv_of(&f);
        assert_eq!(
            &argv[..3],
            &["resume", "thread-parent", "--include-non-interactive"]
        );
        assert!(argv.iter().any(|a| a == "-C"), "{argv:?}");
        assert!(argv.iter().any(|a| a == "-s"), "{argv:?}");
        for absent in ["exec", "--json", "-o"] {
            assert!(!argv.iter().any(|a| a == absent), "{absent}: {argv:?}");
        }
        assert!(
            argv.windows(2)
                .any(|w| w[0] == "-m" && w[1] == "gpt-5.6-luna"),
            "{argv:?}"
        );
        assert_eq!(
            spawn_of(&launch_records(&f)).payload.session.as_deref(),
            Some("thread-parent")
        );
        assert_eq!(completion.end.session.as_deref(), Some("thread-parent"));
    }

    #[cfg(unix)]
    #[test]
    fn an_interactive_resume_refuses_what_ccc_resume_refuses() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("launch-resume-refusals");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        let marker = f.path("ran");
        f.launch_script(&format!("touch '{}'", marker.display()), false);
        start_launch_resume(&f, "9f22dd1b-aaaa", false);
        let usage = claude_usage;
        let refuse = |id: &str| {
            let ctx = f.ctx(&usage);
            launch_with_ctx(&ctx, LaunchTarget::Resume { id, fork: false }, None, vec![])
                .unwrap_err()
        };

        let err = refuse("never-recorded");
        assert!(format!("{err:#}").contains("unknown session id"), "{err:#}");
        assert_eq!(error_end(&err.error).exit, Some(2));

        seed_parent(&f, "9f22dd1b-aaaa", Harness::Claude);
        let held = session::try_lock_in(&f.sessions_dir(), "9f22dd1b-aaaa")
            .unwrap()
            .unwrap();
        let err = refuse("9f22dd1b-aaaa");
        assert!(format!("{err:#}").contains("in use"), "{err:#}");
        assert_eq!(error_end(&err.error).exit, Some(2));
        drop(held);

        let mut child = std::process::Command::new("sleep")
            .arg("2")
            .spawn()
            .unwrap();
        f.audit
            .child(ChildPayload {
                proc: crate::procs::ProcId::of(child.id()).unwrap(),
                pgid: child.id(),
                supervisor: crate::procs::ProcId::current().unwrap(),
            })
            .unwrap();
        let mut record = record_of(&resolved(), Path::new(&f.work()), Some(&account()), f.now);
        record.trace = Some(f.audit.lineage().trace_id.clone());
        record.inv = Some(f.audit.lineage().invocation_id.clone());
        session::save_in(&f.sessions_dir(), "orphan-live", &record).unwrap();
        let err = refuse("orphan-live");
        unsafe {
            libc::kill(child.id() as libc::pid_t, libc::SIGTERM);
        }
        child.wait().unwrap();
        assert!(format!("{err:#}").contains("still running"), "{err:#}");
        assert_eq!(error_end(&err.error).exit, Some(2));

        let mut ghost = record_of(&resolved(), Path::new(&f.work()), Some(&account()), f.now);
        ghost.agent = Some("ghost".into());
        session::save_in(&f.sessions_dir(), "ghost-session", &ghost).unwrap();
        let err = refuse("ghost-session");
        assert_eq!(error_end(&err.error).exit, Some(2));

        assert!(!marker.exists());
    }

    #[cfg(unix)]
    #[test]
    fn an_interactive_resume_with_profile_uses_it_after_the_credential_check() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("launch-resume-profile");
        f.token("team04");
        seed_parent(&f, "9f22dd1b-aaaa", Harness::Claude);
        let script = format!("{}\nexit 0", record_call_line(&f));
        f.launch_script(&script, false);
        start_launch_resume(&f, "9f22dd1b-aaaa", false);
        let usage = || UsageSource::Unavailable("not consulted".into());
        let completion = launch_with_ctx(
            &f.ctx(&usage),
            LaunchTarget::Resume {
                id: "9f22dd1b-aaaa",
                fork: false,
            },
            Some("team04"),
            vec![],
        )
        .unwrap();
        assert_eq!(completion.code, 0);
        assert!(f.call("env").contains("CLAUDE_CODE_OAUTH_TOKEN="));
        let saved = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
        assert_eq!(saved.profile.as_deref(), Some("team04"));
        assert_eq!(saved.auth, Some(ClaudeAuth::Token));

        std::fs::remove_file(f.path("calls/argv")).unwrap();
        let err = launch_with_ctx(
            &f.ctx(&usage),
            LaunchTarget::Resume {
                id: "9f22dd1b-aaaa",
                fork: false,
            },
            Some("team05"),
            vec![],
        )
        .unwrap_err();
        assert_eq!(error_end(&err.error).exit, Some(2));
        assert!(!f.path("calls/argv").exists());
    }

    #[cfg(unix)]
    #[test]
    fn an_interactive_claude_fork_saves_the_fork_from_its_last_turn() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("launch-fork-claude");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        let parent = seed_parent(&f, "9f22dd1b-aaaa", Harness::Claude);
        let script = format!(
            "{}\n{}\nexit 0",
            record_call_line(&f),
            turn_line(&f, "claude", "fork-1111")
        );
        f.launch_script(&script, false);
        start_launch_resume(&f, "9f22dd1b-aaaa", true);
        let usage = claude_usage;
        let ctx = f.ctx(&usage);
        let completion = launch_with_ctx(
            &ctx,
            LaunchTarget::Resume {
                id: "9f22dd1b-aaaa",
                fork: true,
            },
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(completion.code, 0);
        let argv = argv_of(&f);
        let i = argv.iter().position(|a| a == "--resume").expect("--resume");
        assert_eq!(argv[i + 1], "9f22dd1b-aaaa");
        assert_eq!(argv[i + 2], "--fork-session");
        assert!(!argv.iter().any(|a| a == "--session-id"), "{argv:?}");
        assert_eq!(sidecar_text(&f, "9f22dd1b-aaaa"), parent);
        let fork = session::load_from(&f.sessions_dir(), "fork-1111").unwrap();
        assert_eq!(fork.forked_from.as_deref(), Some("9f22dd1b-aaaa"));
        assert_eq!(fork.created_at, f.now);
        assert_eq!(fork.last_used_at, f.now);
        let lineage = f.audit.lineage();
        assert_eq!(fork.trace.as_deref(), Some(lineage.trace_id.as_str()));
        assert_eq!(fork.inv.as_deref(), Some(lineage.invocation_id.as_str()));
        let records = launch_records(&f);
        assert!(start_of(&records).payload.fork);
        assert_eq!(
            start_of(&records).payload.session.as_deref(),
            Some("9f22dd1b-aaaa")
        );
        assert_eq!(spawn_of(&records).payload.session, None);
        assert_eq!(completion.end.session.as_deref(), Some("fork-1111"));
        assert!(
            completion.end.warnings.is_empty(),
            "{:?}",
            completion.end.warnings
        );
        assert!(lock_is_free(&f, "9f22dd1b-aaaa"));
        drop(ctx);

        let mut quit = Fixture::new("launch-fork-claude-quit");
        quit.token("team03");
        quit.cfg.claude.profiles = vec!["team03".into()];
        let parent = seed_parent(&quit, "9f22dd1b-aaaa", Harness::Claude);
        quit.launch_script("exit 0", false);
        start_launch_resume(&quit, "9f22dd1b-aaaa", true);
        let completion = launch_with_ctx(
            &quit.ctx(&usage),
            LaunchTarget::Resume {
                id: "9f22dd1b-aaaa",
                fork: true,
            },
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(completion.code, 0);
        assert_eq!(sidecar_names(&quit), vec!["9f22dd1b-aaaa.json".to_string()]);
        assert_eq!(sidecar_text(&quit, "9f22dd1b-aaaa"), parent);
        assert_eq!(completion.end.session, None);
        assert!(
            completion.end.warnings.is_empty(),
            "{:?}",
            completion.end.warnings
        );
    }

    #[cfg(unix)]
    #[test]
    fn an_interactive_codex_fork_saves_the_fork_from_its_last_turn() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("launch-fork-codex");
        let parent = seed_parent(&f, "thread-parent", Harness::Codex);
        let script = format!(
            "{}\n{}\nexit 0",
            record_call_line(&f),
            turn_line(&f, "codex", "fork-thread")
        );
        f.launch_script(&script, true);
        start_launch_resume(&f, "thread-parent", true);
        let usage = || UsageSource::Unavailable("not consulted".into());
        let completion = launch_with_ctx(
            &f.ctx(&usage),
            LaunchTarget::Resume {
                id: "thread-parent",
                fork: true,
            },
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(completion.code, 0);
        let argv = argv_of(&f);
        assert_eq!(&argv[..2], &["fork", "thread-parent"]);
        assert!(
            !argv.iter().any(|a| a == "--include-non-interactive"),
            "{argv:?}"
        );
        let fork = session::load_from(&f.sessions_dir(), "fork-thread").unwrap();
        assert_eq!(fork.forked_from.as_deref(), Some("thread-parent"));
        assert_eq!(sidecar_text(&f, "thread-parent"), parent);
        assert_eq!(spawn_of(&launch_records(&f)).payload.session, None);
        assert_eq!(completion.end.session.as_deref(), Some("fork-thread"));
    }

    #[cfg(unix)]
    #[test]
    fn an_interactive_fork_of_a_locked_parent_proceeds() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("launch-fork-locked-parent");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        seed_parent(&f, "9f22dd1b-aaaa", Harness::Claude);
        let held = session::try_lock_in(&f.sessions_dir(), "9f22dd1b-aaaa")
            .unwrap()
            .unwrap();
        let script = format!("{}\nexit 0", turn_line(&f, "claude", "fork-1111"));
        f.launch_script(&script, false);
        start_launch_resume(&f, "9f22dd1b-aaaa", true);
        let usage = claude_usage;
        let completion = launch_with_ctx(
            &f.ctx(&usage),
            LaunchTarget::Resume {
                id: "9f22dd1b-aaaa",
                fork: true,
            },
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(completion.code, 0);
        assert!(session::load_from(&f.sessions_dir(), "fork-1111").is_ok());

        let err = launch_with_ctx(
            &f.ctx(&usage),
            LaunchTarget::Resume {
                id: "9f22dd1b-aaaa",
                fork: false,
            },
            None,
            vec![],
        )
        .unwrap_err();
        assert_eq!(error_end(&err.error).exit, Some(2));
        drop(held);
    }

    #[cfg(unix)]
    #[test]
    fn an_interactive_fork_that_comes_back_as_its_parent_is_not_recorded() {
        let _lock = test_io_lock();
        let mut f = Fixture::new("launch-fork-same-id");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        let parent = seed_parent(&f, "9f22dd1b-aaaa", Harness::Claude);
        let script = format!("{}\nexit 0", turn_line(&f, "claude", "9f22dd1b-aaaa"));
        f.launch_script(&script, false);
        start_launch_resume(&f, "9f22dd1b-aaaa", true);
        let usage = claude_usage;
        let completion = launch_with_ctx(
            &f.ctx(&usage),
            LaunchTarget::Resume {
                id: "9f22dd1b-aaaa",
                fork: true,
            },
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(completion.code, 0);
        assert_eq!(sidecar_text(&f, "9f22dd1b-aaaa"), parent);
        assert_eq!(sidecar_names(&f), vec!["9f22dd1b-aaaa.json".to_string()]);
        assert!(
            completion
                .end
                .warnings
                .iter()
                .any(|w| w.contains("parent's id 9f22dd1b-aaaa")),
            "{:?}",
            completion.end.warnings
        );
    }

    #[cfg(unix)]
    #[test]
    fn resume_orphan_admission_checks_alive_dead_missing_and_unknown_identity() {
        let f = Fixture::new("resume-orphan-admission");
        f.token("team03");
        start_resume(&f, "orphan-live");
        let mut child = std::process::Command::new("sleep")
            .arg("2")
            .spawn()
            .unwrap();
        let proc = crate::procs::ProcId::of(child.id()).unwrap();
        f.audit
            .child(ChildPayload {
                proc: proc.clone(),
                pgid: child.id(),
                supervisor: crate::procs::ProcId::current().unwrap(),
            })
            .unwrap();
        let mut record = record_of(&resolved(), Path::new(&f.work()), Some(&account()), f.now);
        record.trace = Some(f.audit.lineage().trace_id.clone());
        record.inv = Some(f.audit.lineage().invocation_id.clone());
        session::save_in(&f.sessions_dir(), "orphan-live", &record).unwrap();
        {
            let usage = || UsageSource::Unavailable("unused".into());
            let ctx = f.ctx(&usage);
            let err = admit_resume(&ctx, "orphan-live", &record).unwrap_err();
            assert!(err.to_string().contains("still running"), "{err:#}");
        }
        #[cfg(unix)]
        unsafe {
            libc::kill(child.id() as libc::pid_t, libc::SIGTERM);
        }
        child.wait().unwrap();
        {
            let usage = || UsageSource::Unavailable("unused".into());
            let ctx = f.ctx(&usage);
            admit_resume(&ctx, "orphan-dead", &record).unwrap();
        }

        let mut missing = record.clone();
        missing.trace = Some("00000000-0000-0000-0000-000000000000".into());
        missing.inv = record.inv.clone();
        {
            let usage = || UsageSource::Unavailable("unused".into());
            let ctx = f.ctx(&usage);
            admit_resume(&ctx, "orphan-missing", &missing).unwrap();
        }

        let mut unknown = record;
        unknown.trace = Some(f.audit.lineage().trace_id.clone());
        unknown.inv = Some(f.audit.lineage().invocation_id.clone());
        let mut unknown_child = std::process::Command::new("sleep")
            .arg("1")
            .spawn()
            .unwrap();
        let fake = crate::procs::ProcId {
            pid: unknown_child.id(),
            start: 0,
            boot_id: "wrong-boot".into(),
        };
        f.audit
            .child(ChildPayload {
                proc: fake.clone(),
                pgid: unknown_child.id(),
                supervisor: crate::procs::ProcId::current().unwrap(),
            })
            .unwrap();
        let mut warnings = Vec::new();
        check_orphan_liveness(
            "orphan-unknown",
            unknown.inv.as_deref().unwrap(),
            &fake,
            |_| crate::procs::Liveness::Unknown("proc hidden".into()),
            |warning| warnings.push(warning),
        )
        .unwrap();
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("orphan-unknown"));
        assert!(warnings[0].contains("proc hidden"));
        let usage = || UsageSource::Unavailable("unused".into());
        {
            let ctx = f.ctx(&usage);
            admit_resume(&ctx, "orphan-unknown", &unknown).unwrap();
        }
        #[cfg(unix)]
        unsafe {
            libc::kill(unknown_child.id() as libc::pid_t, libc::SIGTERM);
        }
        unknown_child.wait().unwrap();
    }

    #[test]
    fn unusable_ids_survive_annotation_and_fail_renders_matching_trailers() {
        let err = HarnessError::unusable("terminated").with_session_id(Some("known".into()));
        let annotated = err.annotate(&RunContext {
            agent: Some("explorer".into()),
            harness: Harness::Claude,
            model: "m".into(),
            effort: None,
            profile: None,
            cwd: PathBuf::from("/tmp"),
            argv: vec!["claude".into()],
        });
        assert_eq!(annotated.session_id(), Some("known"));
        let r = resolved();
        let mut text = Vec::new();
        assert_eq!(
            fail(
                &r,
                Path::new("/tmp"),
                None,
                annotated.clone(),
                None,
                false,
                &mut text
            ),
            3
        );
        assert!(String::from_utf8(text).unwrap().contains("session=known"));
        let mut json = Vec::new();
        fail(
            &r,
            Path::new("/tmp"),
            None,
            annotated,
            None,
            true,
            &mut json,
        );
        let value: serde_json::Value =
            serde_json::from_str(&String::from_utf8(json).unwrap()).unwrap();
        assert_eq!(value["session"], "known");
        let mut none = Vec::new();
        fail(
            &r,
            Path::new("/tmp"),
            None,
            HarnessError::unusable("none"),
            None,
            false,
            &mut none,
        );
        assert!(none.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn termination_message_renders_all_outcomes_and_escalation_reasons() {
        let exited = std::process::Command::new("sh")
            .arg("-c")
            .arg("exit 7")
            .output()
            .unwrap()
            .status;
        let signalled = std::process::Command::new("sh")
            .arg("-c")
            .arg("kill -TERM $$")
            .output()
            .unwrap()
            .status;
        for (status, sender, swept, escalation) in [
            (&exited, Some(9), 0, None),
            (&exited, None, 2, Some("grace expired")),
            (&signalled, Some(9), 0, Some("second signal")),
        ] {
            let message = termination_message(
                &Termination {
                    requested: libc::SIGTERM,
                    by: sender,
                    reason: "signal to ccc",
                    escalated: escalation,
                    swept,
                },
                status,
            );
            assert!(message.contains(if status.code().is_some() {
                "exit"
            } else {
                "signal"
            }));
            assert!(message.contains("requested SIG15"));
            assert!(message.contains(&format!(
                "sender pid {}",
                sender.map_or("unknown".into(), |pid| pid.to_string())
            )));
            assert!(message.contains("reason signal to ccc"));
            assert!(message.contains(&format!("swept {swept} descendants")));
            if let Some(reason) = escalation {
                assert!(message.contains(reason));
            } else {
                assert!(message.contains("termination observed before grace expiry"));
            }
        }
    }

    #[test]
    fn a_moved_resume_records_the_account_the_turn_ran_under() {
        // The recorded profile is still usable but team04 ranks higher, so the
        // turn runs on team04 under the configured auth; the sidecar names
        // both. A later failed turn keeps the account it ran under too.
        let mut f = Fixture::new("resume-moved");
        f.claude_script(OK_RESULT);
        f.config_dir_credential("team03");
        f.token("team03");
        f.token("team04");
        f.cfg.claude.auth = ClaudeAuth::Token;
        let usage = || {
            report(vec![
                snapshot(Timestamp::now(), "team03", 30.0, 60.0),
                snapshot(Timestamp::now(), "team04", 0.0, 0.0),
            ])
        };

        let cwd = std::fs::canonicalize(f.path("work")).unwrap();
        let mut record = record_of(&resolved(), &cwd, None, f.now);
        record.profile = Some("team03".into());
        record.auth = Some(ClaudeAuth::ConfigDir);
        session::save_in(&f.sessions_dir(), "9f22dd1b-aaaa", &record).unwrap();

        let mut out: Vec<u8> = Vec::new();
        let code = resume_with(
            &f.ctx(&usage),
            "9f22dd1b-aaaa",
            false,
            None,
            "again".to_string(),
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(code, 0);
        assert!(
            String::from_utf8(out).unwrap().contains("profile=team04"),
            "the trailer names the account that ran"
        );

        let after = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
        assert_eq!(after.profile.as_deref(), Some("team04"));
        assert_eq!(after.auth, Some(ClaudeAuth::Token));

        f.claude_script(
            r#"{"type":"result","session_id":"9f22dd1b-aaaa","result":"5-hour limit reached","is_error":true,"num_turns":1}"#,
        );
        let usage = || {
            report(vec![
                snapshot(Timestamp::now(), "team03", 0.0, 0.0),
                snapshot(Timestamp::now(), "team04", 30.0, 60.0),
            ])
        };
        let mut out: Vec<u8> = Vec::new();
        let code = resume_with(
            &f.ctx(&usage),
            "9f22dd1b-aaaa",
            false,
            None,
            "again".to_string(),
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(code, crate::harness::EXIT_HARNESS_FAILED);
        assert!(String::from_utf8(out).unwrap().contains("profile=team03"));
        let failed = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
        assert_eq!(failed.profile.as_deref(), Some("team03"));
        assert_eq!(failed.auth, Some(ClaudeAuth::Token));
    }

    #[test]
    fn a_failed_turn_still_records_the_session_it_printed() {
        let mut f = Fixture::new("run-failed");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.claude_script(
            r#"{"type":"result","session_id":"9f22dd1b-bbbb","result":"5-hour limit reached","is_error":true,"num_turns":1}"#,
        );
        start_run(&f, Some(PromptSource::Arg), None, Some("go"));
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);
        let ctx = f.ctx(&usage);

        let mut out: Vec<u8> = Vec::new();
        let completion = run_with(
            &ctx,
            "explorer",
            Some(&f.work()),
            "go".to_string(),
            None,
            None,
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(completion, crate::harness::EXIT_HARNESS_FAILED);
        record_completion(&f, completion);

        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("5-hour limit reached"), "{text}");
        assert!(text.contains("session=9f22dd1b-bbbb"), "{text}");
        // The id the caller was handed resolves, so the turn can be retried.
        let record = session::load_from(&f.sessions_dir(), "9f22dd1b-bbbb").unwrap();
        assert_eq!(record.agent.as_deref(), Some("explorer"));
        let end = audit_end(&f);
        assert_eq!(end.payload.exit, Some(crate::harness::EXIT_HARNESS_FAILED));
        assert_eq!(end.payload.session.as_deref(), Some("9f22dd1b-bbbb"));
        assert_eq!(end.payload.result.as_deref(), Some("5-hour limit reached"));
        assert_eq!(end.payload.native.as_ref().unwrap()["num_turns"], 1);
    }

    #[cfg(unix)]
    #[test]
    fn unusable_process_and_unknown_agent_are_audited_with_exit_three_and_two() {
        let mut process = Fixture::new("run-audit-process-error");
        process.token("team03");
        process.cfg.claude.profiles = vec!["team03".into()];
        process.env.insert(
            "CCC_CLAUDE_BIN".into(),
            process.path("does-not-exist").display().to_string(),
        );
        start_run(&process, Some(PromptSource::Arg), None, Some("go"));
        let usage = || UsageSource::Unavailable("not consulted".into());
        let mut out = Vec::new();
        let completion = run_with(
            &process.ctx(&usage),
            "explorer",
            Some(&process.work()),
            "go".into(),
            None,
            None,
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(completion, crate::harness::EXIT_NO_RESULT);
        record_completion(&process, completion);
        let end = audit_end(&process);
        assert_eq!(end.payload.exit, Some(crate::harness::EXIT_NO_RESULT));
        assert!(end.payload.session.is_some());
        assert!(end.payload.result.is_some());
        assert!(end.payload.error.is_some());
        assert!(audit_records(&process)
            .iter()
            .any(|r| matches!(r, audit::Record::Spawn(_))));

        let unknown = Fixture::new("run-audit-unknown-agent");
        start_run(&unknown, Some(PromptSource::Arg), None, Some("go"));
        let mut out = Vec::new();
        let err = run_with(
            &unknown.ctx(&usage),
            "does-not-exist",
            Some(&unknown.work()),
            "go".into(),
            None,
            None,
            false,
            &mut out,
        )
        .unwrap_err();
        assert!(err.to_string().contains("does-not-exist"));
        unknown.audit.end(error_end(&err.error));
        let end = audit_end(&unknown);
        assert_eq!(end.payload.exit, Some(2));
        assert!(end.payload.error.unwrap().contains("does-not-exist"));
        assert!(!audit_records(&unknown)
            .iter()
            .any(|r| matches!(r, audit::Record::Spawn(_))));
    }

    #[cfg(unix)]
    #[test]
    fn resume_audit_start_has_no_agent_and_spawn_uses_the_sidecar_agent() {
        let mut f = Fixture::new("resume-audit-identity");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.claude_script(OK_RESULT);
        let record = claude_record(&f, Some("team03"), ClaudeAuth::Token);
        session::save_in(&f.sessions_dir(), "9f22dd1b-aaaa", &record).unwrap();
        start_resume(&f, "9f22dd1b-aaaa");
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);
        let mut out = Vec::new();
        resume_with(
            &f.ctx(&usage),
            "9f22dd1b-aaaa",
            false,
            None,
            "again".into(),
            false,
            &mut out,
        )
        .unwrap();
        let records = audit_records(&f);
        let start = records
            .iter()
            .find_map(|r| match r {
                audit::Record::Start(start) => Some(start),
                _ => None,
            })
            .unwrap();
        assert_eq!(start.payload.agent, None);
        assert_eq!(start.payload.session.as_deref(), Some("9f22dd1b-aaaa"));
        let spawn = records
            .iter()
            .find_map(|r| match r {
                audit::Record::Spawn(spawn) => Some(spawn),
                _ => None,
            })
            .unwrap();
        assert_eq!(spawn.payload.agent.as_deref(), Some("explorer"));
    }

    #[test]
    fn a_sidecar_that_cannot_be_written_costs_the_next_turn_not_this_one() {
        let mut f = Fixture::new("run-unwritable-state");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.claude_script(OK_RESULT);
        // A state home that is a regular file: every sidecar write fails.
        let blocked = f.path("blocked");
        std::fs::write(&blocked, "").unwrap();
        f.env
            .insert("XDG_STATE_HOME".to_string(), blocked.display().to_string());
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);
        let ctx = f.ctx(&usage);

        let mut out: Vec<u8> = Vec::new();
        let code = run_with(
            &ctx,
            "explorer",
            Some(&f.work()),
            "go".to_string(),
            None,
            None,
            false,
            &mut out,
        )
        .unwrap();
        assert_eq!(code, 0);
        assert!(
            String::from_utf8(out).unwrap().contains("the answer"),
            "the turn's result is still delivered"
        );
    }

    #[test]
    fn launch_drops_the_one_shot_flags_and_appends_the_passthrough_last() {
        let mut f = Fixture::new("launch");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.cfg.claude.extra_args = vec!["--add-dir".into(), "/srv".into()];
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);
        let ctx = f.ctx(&usage);

        let LaunchPrepared { inv, account, .. } = launch_invocation(
            &ctx,
            prepare(
                &ctx,
                "explorer",
                Some(&f.work()),
                None,
                None,
                &GooseFlags::default(),
            )
            .unwrap(),
            vec!["--foo".into()],
        )
        .unwrap();

        // The account handed back is the one the child runs as, so a caller
        // acting on it cannot drift from the invocation's environment.
        let account = account.expect("a Claude agent has an account");
        assert_eq!(account.profile, "team03");
        assert!(
            inv.env_set
                .iter()
                .any(|(k, _)| k == "CLAUDE_CODE_OAUTH_TOKEN"),
            "{inv:?}"
        );

        assert!(!inv.args.contains(&"-p".to_string()), "{:?}", inv.args);
        assert!(
            !inv.args.contains(&"--output-format".to_string()),
            "{:?}",
            inv.args
        );
        assert_eq!(
            &inv.args[inv.args.len() - 3..],
            ["--add-dir", "/srv", "--foo"]
        );
        assert_eq!(inv.stdin, None);
        assert_eq!(inv.cwd, std::fs::canonicalize(f.path("work")).unwrap());
    }

    #[test]
    fn a_claude_spawn_marks_its_working_directory_trusted_first() {
        let mut f = Fixture::new("drive-trust");
        f.token("team03");
        f.env.insert(
            "CCC_CLAUDE_BIN".to_string(),
            "/nonexistent/claude".to_string(),
        );
        let usage = || UsageSource::Unavailable("not consulted".into());
        let ctx = f.ctx(&usage);

        let common = Common {
            cwd: f.path("work"),
            extra_args: vec![],
            mode: run_mode(),
            preset_session_id: None,
        };
        let account = Account::from_config(&f.cfg.claude, "team03", &f.env);

        // The spawn itself fails, and the marker is written all the same.
        let record = record_of(&resolved(), &common.cwd, Some(&account), ctx.now);
        assert!(
            drive(&ctx, &resolved(), &common, Some(account), vec![], &record)
                .unwrap()
                .is_err()
        );
        let config = f.path(".claude.json");
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&config).unwrap()).unwrap();
        assert_eq!(
            v["projects"][f.path("work").display().to_string()]["hasTrustDialogAccepted"],
            serde_json::json!(true)
        );
    }

    #[test]
    fn a_codex_spawn_writes_no_trust_marker() {
        let mut f = Fixture::new("drive-trust-codex");
        f.env.insert(
            "CCC_CODEX_BIN".to_string(),
            "/nonexistent/codex".to_string(),
        );
        let usage = || UsageSource::Unavailable("not consulted".into());
        let ctx = f.ctx(&usage);

        let mut r = resolved();
        r.harness = Harness::Codex;
        let common = Common {
            cwd: f.path("work"),
            extra_args: vec![],
            mode: run_mode(),
            preset_session_id: None,
        };
        // The account exists, so only the harness keeps the marker unwritten.
        f.token("team03");
        let account = Account::from_config(&f.cfg.claude, "team03", &f.env);
        let record = record_of(&r, &common.cwd, Some(&account), ctx.now);
        assert!(drive(&ctx, &r, &common, Some(account), vec![], &record)
            .unwrap()
            .is_err());
        assert!(!f.path(".claude.json").exists());

        let record = record_of(&r, &common.cwd, None, ctx.now);
        assert!(drive(&ctx, &r, &common, None, vec![], &record)
            .unwrap()
            .is_err());
        assert!(!f.path(".claude.json").exists());
    }

    #[test]
    fn a_trust_write_that_fails_does_not_fail_the_spawn() {
        let mut f = Fixture::new("drive-trust-broken");
        f.token("team03");
        f.dir.write_file(".claude.json", "[1, 2]", 0o600);
        f.env.insert(
            "CCC_CLAUDE_BIN".to_string(),
            "/nonexistent/claude".to_string(),
        );
        let usage = || UsageSource::Unavailable("not consulted".into());
        let ctx = f.ctx(&usage);

        let common = Common {
            cwd: f.path("work"),
            extra_args: vec![],
            mode: run_mode(),
            preset_session_id: None,
        };
        let account = Account::from_config(&f.cfg.claude, "team03", &f.env);

        // The failure that comes back is the spawn's, not the trust write's,
        // and the config the operator broke is left as it is.
        let record = record_of(&resolved(), &common.cwd, Some(&account), ctx.now);
        let err = drive(&ctx, &resolved(), &common, Some(account), vec![], &record)
            .unwrap()
            .unwrap_err();
        assert_eq!(err.exit_code(), 3, "{}", err.message());
        assert_eq!(
            std::fs::read_to_string(f.path(".claude.json")).unwrap(),
            "[1, 2]"
        );
    }

    #[test]
    fn a_codex_spawn_that_never_starts_is_annotated_and_leaves_no_output_file() {
        let mut f = Fixture::new("drive-spawn-failure");
        f.env.insert(
            "CCC_CODEX_BIN".to_string(),
            "/nonexistent/codex".to_string(),
        );
        let usage = || UsageSource::Unavailable("not consulted".into());
        let ctx = f.ctx(&usage);

        let mut r = resolved();
        r.harness = Harness::Codex;
        r.model = "gpt-5.6-luna".into();
        let common = Common {
            cwd: f.path("work"),
            extra_args: vec![],
            mode: run_mode(),
            preset_session_id: None,
        };

        let record = record_of(&r, &common.cwd, None, ctx.now);
        let err = drive(&ctx, &r, &common, None, vec![], &record)
            .unwrap()
            .unwrap_err();
        assert_eq!(err.exit_code(), crate::harness::EXIT_NO_RESULT);
        let msg = err.message();
        assert!(msg.contains("/nonexistent/codex"), "{msg}");
        // The run context is what makes an unusable result debuggable.
        for part in ["agent=explorer", "harness=codex", "gpt-5.6-luna"] {
            assert!(msg.contains(part), "{msg} lacks {part}");
        }

        // The `-o` file minted for the child that never ran is reaped.
        let tmp = crate::config::state_home_in(&f.env).join("ccc/tmp");
        let left: Vec<PathBuf> = std::fs::read_dir(&tmp)
            .map(|d| d.map(|e| e.unwrap().path()).collect())
            .unwrap_or_default();
        assert!(left.is_empty(), "{left:?}");
    }

    #[test]
    fn the_spawned_child_gets_the_prompt_and_its_streams_are_captured() {
        let inv = Invocation {
            program: "/bin/sh".into(),
            args: vec!["-c".into(), "cat; echo to-stderr >&2; exit 7".into()],
            env_set: vec![],
            env_remove: vec![],
            cwd: PathBuf::from("/tmp"),
            stdin: Some("the prompt".into()),
        };
        let (captured, warnings) = run_captured_test(&inv, None).unwrap();
        assert_eq!(captured.stdout, "the prompt");
        assert_eq!(captured.stderr.trim(), "to-stderr");
        assert_eq!(captured.exit_code, Some(7));
        assert_eq!(captured.output_file, None);
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    #[test]
    fn a_prompt_and_a_reply_larger_than_a_pipe_buffer_cannot_deadlock() {
        // Both directions exceed the 64 KiB pipe buffer, so a sequential
        // write-then-read would block forever instead of failing.
        let prompt = "p".repeat(1024 * 1024);
        let inv = Invocation {
            program: "/bin/sh".into(),
            args: vec![
                "-c".into(),
                "cat; yes long-line | head -c 1000000; yes err | head -c 200000 >&2".into(),
            ],
            env_set: vec![],
            env_remove: vec![],
            cwd: PathBuf::from("/tmp"),
            stdin: Some(prompt.clone()),
        };
        let supervised = run_supervised_test(&inv).unwrap();
        let captured = supervised.captured;
        let delivery = supervised.prompt.expect("prompt delivery");
        assert_eq!(delivery.written, delivery.total);
        assert_eq!(delivery.total, 1024 * 1024);
        assert_eq!(delivery.ended, supervise::PromptEnd::Complete);
        assert!(
            captured.stdout.starts_with(&prompt),
            "prompt not echoed: stdout_len={} stderr_len={} prefix={} warnings={:?} status={:?} termination={:?}",
            captured.stdout.len(),
            captured.stderr.len(),
            captured.stdout.get(..prompt.len()).is_some_and(|prefix| prefix == prompt),
            supervised.warnings,
            supervised.status,
            supervised.termination,
        );
        assert_eq!(captured.stdout.len(), prompt.len() + 1_000_000);
        assert_eq!(captured.stderr.len(), 200_000);
        assert!(supervised.warnings.is_empty(), "{:?}", supervised.warnings);
    }

    #[cfg(unix)]
    #[test]
    fn exec_spawn_panic_is_exit_three_with_session_and_failure_audit() {
        let _lock = supervise::TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut f = Fixture::new("exec-spawn-panic");
        f.token("team03");
        f.claude_custom_script("trap '' TERM; sleep 10");
        supervise::set_test_io_fault(
            supervise::TestIoStream::Prompt,
            supervise::TestIoFault::SpawnPanic,
        );
        let mut input = "go".as_bytes();
        let mut output = Vec::new();
        let code = run_from(
            "explorer",
            Some(&f.work()),
            None,
            None,
            None,
            None,
            false,
            &f.env,
            || Ok(f.cfg.clone()),
            &mut input,
            &mut output,
        )
        .unwrap();
        assert_eq!(code, crate::harness::EXIT_NO_RESULT);
        let text = String::from_utf8(output).unwrap();
        let records = latest_records(&f.env);
        let spawn = records
            .iter()
            .find_map(|record| match record {
                audit::Record::Spawn(spawn) => Some(spawn.payload.session.clone()),
                _ => None,
            })
            .flatten()
            .expect("preset session");
        assert!(!spawn.is_empty());
        assert!(text.contains(&format!("session={spawn}")), "{text}");
        let end = records
            .iter()
            .find_map(|record| match record {
                audit::Record::End(end) => Some(end.clone()),
                _ => None,
            })
            .expect("audit end");
        assert_eq!(end.payload.exit, Some(3));
        assert_eq!(end.payload.session.as_deref(), Some(spawn.as_str()));
        assert!(end
            .payload
            .error
            .unwrap_or_default()
            .contains("supervisor failed"));
        assert_eq!(end.payload.signal, None);
        let child = records
            .iter()
            .find_map(|record| match record {
                audit::Record::Child(child) => Some(child.payload.proc.clone()),
                _ => None,
            })
            .expect("child record");
        assert_eq!(child.alive(), crate::procs::Liveness::Dead);
        let signals = records
            .iter()
            .filter_map(|record| match record {
                audit::Record::Signal(signal) => Some(&signal.payload),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].signal, libc::SIGKILL);
        assert!(matches!(signals[0].to, audit::SignalDestination::Child));
        assert_eq!(signals[0].reason, "supervisor failure");
    }

    #[cfg(unix)]
    #[test]
    fn exec_waiter_panic_discards_a_valid_result_and_keeps_session() {
        let _lock = supervise::TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut f = Fixture::new("exec-waiter-panic");
        f.token("team03");
        f.claude_script(OK_RESULT);
        supervise::set_test_io_fault(
            supervise::TestIoStream::Waiter,
            supervise::TestIoFault::Panic,
        );
        let mut input = "go".as_bytes();
        let mut output = Vec::new();
        let code = run_from(
            "explorer",
            Some(&f.work()),
            None,
            None,
            None,
            None,
            false,
            &f.env,
            || Ok(f.cfg.clone()),
            &mut input,
            &mut output,
        )
        .unwrap();
        assert_eq!(code, crate::harness::EXIT_NO_RESULT);
        let text = String::from_utf8(output).unwrap();
        let records = latest_records(&f.env);
        let preset = records
            .iter()
            .find_map(|record| match record {
                audit::Record::Spawn(spawn) => Some(spawn.payload.session.clone()),
                _ => None,
            })
            .flatten()
            .expect("preset session");
        assert!(!preset.is_empty());
        assert!(text.contains(&format!("session={preset}")), "{text}");
        let end = records
            .iter()
            .find_map(|record| match record {
                audit::Record::End(end) => Some(end.clone()),
                _ => None,
            })
            .expect("audit end");
        assert!(end
            .payload
            .error
            .unwrap_or_default()
            .contains("signal waiter panicked"));
        assert_eq!(end.payload.session.as_deref(), Some(preset.as_str()));
        assert_eq!(end.payload.signal, None);
        assert!(records
            .iter()
            .any(|record| matches!(record, audit::Record::Child(_))));
        assert!(!records
            .iter()
            .any(|record| matches!(record, audit::Record::Signal(_))));
    }

    #[cfg(unix)]
    #[test]
    fn exec_prompt_shortfalls_route_success_failure_and_full_delivery() {
        let prompt = "p".repeat(1024 * 1024);
        let end_for = |f: &Fixture| {
            latest_records(&f.env)
                .into_iter()
                .find_map(|record| match record {
                    audit::Record::End(end) => Some(end),
                    _ => None,
                })
                .expect("audit end")
        };
        let mut short = Fixture::new("prompt-short");
        short.token("team03");
        short.claude_custom_script(&format!(
            "head -c 1 >/dev/null; printf '%s' '{}'",
            OK_RESULT
        ));
        let mut short_input = "ignored".as_bytes();
        let mut short_output = Vec::new();
        let short_warnings = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let short_warnings_for_report = std::sync::Arc::clone(&short_warnings);
        assert_eq!(
            run_from_report(
                "explorer",
                Some(&short.work()),
                Some(&prompt),
                None,
                None,
                None,
                &GooseFlags::default(),
                false,
                &short.env,
                || Ok(short.cfg.clone()),
                &mut short_input,
                &mut short_output,
                &move |message| short_warnings_for_report.lock().unwrap().push(message),
            )
            .unwrap(),
            3
        );
        let short_end = end_for(&short);
        let short_error = short_end.payload.error.clone().unwrap();
        assert_eq!(short_end.payload.session.as_deref(), Some("9f22dd1b-aaaa"));
        assert!(short_output
            .windows("session=9f22dd1b-aaaa".len())
            .any(|window| window == b"session=9f22dd1b-aaaa"));
        assert_eq!(short_end.payload.result, Some(short_error.clone()));
        assert!(short_error.contains("not fully delivered"));
        assert!(
            short_error.contains("the harness closed its stdin")
                || short_error.contains("the harness exited")
        );
        let short_number = short_error
            .split("not fully delivered: ")
            .nth(1)
            .and_then(|text| text.split(" of ").next())
            .unwrap()
            .parse::<usize>()
            .unwrap();
        assert!(short_number < 1_048_576);
        let warnings = short_warnings.lock().unwrap();
        let warning = warnings
            .iter()
            .find(|warning| warning.starts_with("warning: the prompt was not fully delivered"))
            .expect("prompt shortfall warning");
        assert!(warning.contains(&format!("{short_number} of 1048576")));
        assert!(short_error.contains(&format!("{short_number} of 1048576")));
        assert!(!latest_records(&short.env)
            .iter()
            .any(|record| matches!(record, audit::Record::Signal(_))));

        let mut complete = Fixture::new("prompt-complete");
        complete.token("team03");
        complete.claude_custom_script(&format!("cat >/dev/null; printf '%s' '{}'", OK_RESULT));
        let mut complete_input = "ignored".as_bytes();
        let mut complete_output = Vec::new();
        assert_eq!(
            run_from(
                "explorer",
                Some(&complete.work()),
                Some(&prompt),
                None,
                None,
                None,
                true,
                &complete.env,
                || Ok(complete.cfg.clone()),
                &mut complete_input,
                &mut complete_output,
            )
            .unwrap(),
            0
        );

        let mut failed = Fixture::new("prompt-failed");
        failed.token("team03");
        failed.claude_custom_script(
            "head -c 1 >/dev/null; printf '%s' '{\"type\":\"result\",\"session_id\":\"9f22dd1b-bbbb\",\"result\":\"failed\",\"is_error\":true,\"num_turns\":1}'; exit 1",
        );
        let mut failed_input = "ignored".as_bytes();
        let mut failed_output = Vec::new();
        assert_eq!(
            run_from(
                "explorer",
                Some(&failed.work()),
                Some(&prompt),
                None,
                None,
                None,
                true,
                &failed.env,
                || Ok(failed.cfg.clone()),
                &mut failed_input,
                &mut failed_output,
            )
            .unwrap(),
            1
        );
        assert!(end_for(&failed)
            .payload
            .warnings
            .iter()
            .any(|warning| warning.contains("not fully delivered")));
    }

    #[test]
    fn a_child_that_exits_before_reading_the_prompt_is_not_a_warning() {
        // The child's own output explains itself; a broken pipe adds nothing.
        let inv = Invocation {
            program: "/bin/sh".into(),
            args: vec!["-c".into(), "exit 0".into()],
            env_set: vec![],
            env_remove: vec![],
            cwd: PathBuf::from("/tmp"),
            stdin: Some("p".repeat(1024 * 1024)),
        };
        let supervised = run_supervised_test(&inv).unwrap();
        assert_eq!(supervised.captured.exit_code, Some(0));
        let delivery = supervised.prompt.expect("prompt delivery");
        assert!(delivery.written < delivery.total);
        assert!(
            matches!(
                delivery.ended,
                supervise::PromptEnd::ChildClosed | supervise::PromptEnd::Stopped
            ),
            "unexpected prompt delivery: {delivery:?}"
        );
        assert!(supervised.warnings.is_empty(), "{:?}", supervised.warnings);
    }

    #[test]
    fn prompt_end_reason_renders_each_shortfall_end_state() {
        assert_eq!(
            prompt_end_reason(&PromptEnd::ChildClosed),
            "the harness closed its stdin"
        );
        assert_eq!(prompt_end_reason(&PromptEnd::Stopped), "the harness exited");
        assert_eq!(
            prompt_end_reason(&PromptEnd::Failed(
                "prompt writer poll failed: injected poll failure".into()
            )),
            "prompt writer poll failed: injected poll failure"
        );
        assert_eq!(
            prompt_shortfall(&PromptDelivery {
                written: 4,
                total: 4,
                ended: PromptEnd::Complete,
            }),
            None
        );
    }

    #[test]
    fn prompt_shortfall_helpers_preserve_message_and_delivery_facts() {
        let delivery = PromptDelivery {
            written: 4,
            total: 9,
            ended: PromptEnd::Stopped,
        };
        assert_eq!(
            termination_message_with_prompt("terminated".into(), Some(&delivery)),
            "terminated; prompt: 4 of 9 bytes delivered"
        );
        assert_eq!(
            termination_message_with_prompt("terminated".into(), None),
            "terminated"
        );
        assert_eq!(
            termination_message_with_prompt(
                "terminated".into(),
                Some(&PromptDelivery {
                    written: 9,
                    total: 9,
                    ended: PromptEnd::Complete,
                }),
            ),
            "terminated"
        );

        let mut reported = HarnessError::Reported {
            session_id: Some("reported-session".into()),
            message: "parse failure".into(),
            native: serde_json::json!({}),
            warnings: vec![],
        };
        apply_prompt_shortfall(&mut reported, &delivery);
        assert!(reported
            .warnings()
            .iter()
            .any(|warning| warning == &prompt_shortfall_text(&delivery)));
        let mut unusable =
            HarnessError::unusable("parse failure").with_session_id(Some("session".into()));
        apply_prompt_shortfall(&mut unusable, &delivery);
        match unusable {
            HarnessError::Unusable {
                message,
                session_id,
                ..
            } => {
                assert_eq!(message, "parse failure; the prompt was not fully delivered: 4 of 9 bytes were written before the harness exited");
                assert_eq!(session_id.as_deref(), Some("session"));
            }
            _ => panic!("expected unusable error"),
        }
    }

    #[test]
    fn the_output_file_a_child_wrote_is_captured_and_removed() {
        let dir = TempDir::new("spawn-output");
        let path = dir.path().join("out.txt");
        std::fs::write(&path, "").unwrap();
        let inv = Invocation {
            program: "/bin/sh".into(),
            args: vec![
                "-c".into(),
                format!(
                    "cat > /dev/null; printf 'the final message' > {}",
                    path.display()
                ),
            ],
            env_set: vec![],
            env_remove: vec![],
            cwd: PathBuf::from("/tmp"),
            stdin: Some("go".into()),
        };
        let (captured, _) = run_captured_test(&inv, Some(&path)).unwrap();
        assert_eq!(captured.output_file.as_deref(), Some("the final message"));
        assert!(!path.exists(), "the message was left on disk");
    }

    #[test]
    fn spawning_a_missing_binary_says_what_it_tried_to_run() {
        let inv = Invocation {
            program: "/nonexistent/harness".into(),
            args: vec![],
            env_set: vec![],
            env_remove: vec![],
            cwd: PathBuf::from("/tmp"),
            stdin: None,
        };
        let err = format!("{:#}", run_captured_test(&inv, None).unwrap_err());
        assert!(err.contains("spawning /nonexistent/harness"), "{err}");
    }

    #[test]
    fn test_untested_audited_wrapper_covers_start_config_operation_and_success_paths() {
        let f = Fixture::new("audited-wrapper");
        let mut called = false;
        let recorder = Recorder::new(&f.env).unwrap();
        let trace = recorder.lineage().trace_id.clone();
        let completion = audited_with_recorder_in(
            recorder,
            start_payload_for_test(AuditCommand::Run),
            &ProcessEnv,
            || Ok(Config::default()),
            |_| {
                called = true;
                Ok(Completion::new(
                    0,
                    EndPayload {
                        exit: Some(0),
                        signal: None,
                        session: None,
                        result: None,
                        error: None,
                        warnings: vec![],
                        native: None,
                        turns: None,
                    },
                ))
            },
        )
        .unwrap();
        assert!(called && completion.code == 0);
        let records = audit::read_trace(&Recorder::dir_in(&f.env), &trace).unwrap();
        assert_eq!(
            records
                .iter()
                .filter(|r| matches!(r, audit::Record::Start(_)))
                .count(),
            1
        );
        assert_eq!(
            records
                .iter()
                .filter(|r| matches!(r, audit::Record::End(_)))
                .count(),
            1
        );

        let mut config_called = false;
        let config_recorder = Recorder::new(&f.env).unwrap();
        let config_trace = config_recorder.lineage().trace_id.clone();
        let config_err = audited_with_recorder_in(
            config_recorder,
            start_payload_for_test(AuditCommand::Run),
            &ProcessEnv,
            || Err(anyhow::anyhow!("config test")),
            |_| {
                config_called = true;
                panic!("operation invoked")
            },
        )
        .unwrap_err();
        assert!(config_err.to_string().contains("config test"));
        assert!(!config_called);
        let config_records = audit::read_trace(&Recorder::dir_in(&f.env), &config_trace).unwrap();
        assert_eq!(config_records.len(), 2);
        assert!(matches!(config_records[0], audit::Record::Start(_)));
        let end = match &config_records[1] {
            audit::Record::End(end) => end,
            other => panic!("expected end, got {other:?}"),
        };
        assert_eq!(end.payload.exit, Some(2));
        assert!(end.payload.session.is_none());
        assert!(end.payload.result.is_none());
        assert!(end.payload.native.is_none());
        assert!(end.payload.warnings.is_empty());
        assert!(end
            .payload
            .error
            .as_deref()
            .unwrap()
            .contains("config test"));
    }

    fn start_payload_for_test(cmd: AuditCommand) -> StartPayload {
        StartPayload {
            pid: std::process::id(),
            cmd,
            agent: Some("explorer".into()),
            session: None,
            cwd_arg: None,
            profile_arg: None,
            harness_arg: None,
            json: false,
            prompt_source: Some(PromptSource::Arg),
            prompt_file: None,
            prompt: Some("prompt".into()),
            passthrough: vec![],
            fork: false,
        }
    }

    #[test]
    fn test_missing_start_write_order_failure_and_best_effort_end_are_audited() {
        let f = Fixture::new("audited-failures");
        let recorder = Recorder::new(&f.env).unwrap();
        std::fs::create_dir(recorder.trace_path()).unwrap();
        let mut loaded = false;
        assert!(audited_with_recorder_in(
            recorder,
            start_payload_for_test(AuditCommand::Run),
            &ProcessEnv,
            || {
                loaded = true;
                Ok(Config::default())
            },
            |_| panic!("operation invoked")
        )
        .is_err());
        assert!(!loaded);

        let f = Fixture::new("audited-index-failure");
        let recorder = Recorder::new(&f.env).unwrap();
        std::fs::create_dir(Recorder::dir_in(&f.env).join("traces.jsonl")).unwrap();
        assert!(audited_with_recorder_in(
            recorder,
            start_payload_for_test(AuditCommand::Run),
            &ProcessEnv,
            Config::load,
            |_| panic!("operation invoked")
        )
        .is_err());
        let trace = std::fs::read_dir(Recorder::dir_in(&f.env))
            .unwrap()
            .find_map(|e| {
                e.ok().and_then(|e| {
                    let p = e.path();
                    (p.is_file() && p.extension().is_some_and(|x| x == "jsonl")).then_some(p)
                })
            })
            .unwrap();
        let id = trace.file_stem().unwrap().to_str().unwrap();
        assert_eq!(
            audit::read_trace(Recorder::dir_in(&f.env).as_path(), id)
                .unwrap()
                .iter()
                .filter(|r| matches!(r, audit::Record::End(_)))
                .count(),
            1
        );
    }

    #[test]
    fn test_presence_only_lineage_assertions_compare_exact_values_and_start_fields() {
        let mut f = Fixture::new("lineage-exact");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.claude_script(OK_RESULT);
        let usage = || UsageSource::Unavailable("not consulted".into());
        let mut out = Vec::new();
        let mut start_payload = start_payload_for_test(AuditCommand::Run);
        start_payload.prompt = Some("go".into());
        f.audit.start(start_payload).unwrap();
        let completion = run_with(
            &f.ctx(&usage),
            "explorer",
            Some(&f.work()),
            "go".into(),
            None,
            None,
            false,
            &mut out,
        )
        .unwrap();
        record_completion(&f, completion);
        let environment_text = f.call("env");
        let trace_value = environment_text
            .lines()
            .find_map(|line| line.strip_prefix("CCC_TRACE_ID="))
            .unwrap();
        let parent_value = environment_text
            .lines()
            .find_map(|line| line.strip_prefix("CCC_PARENT_INVOCATION_ID="))
            .unwrap();
        assert_eq!(trace_value, f.audit.lineage().trace_id.as_str());
        assert_eq!(parent_value, f.audit.lineage().invocation_id.as_str());
        let start = audit_records(&f)
            .into_iter()
            .find_map(|r| match r {
                audit::Record::Start(s) => Some(s),
                _ => None,
            })
            .unwrap();
        assert_eq!(start.payload.prompt_source, Some(PromptSource::Arg));
        assert_eq!(start.payload.prompt.as_deref(), Some("go"));
    }

    struct FailingWrite;
    impl Write for FailingWrite {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("output sink failed"))
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn trace_records_for(env: &BTreeMap<String, String>) -> Vec<audit::Record> {
        let index = audit::read_index(&Recorder::dir_in(env)).unwrap();
        audit::read_trace(&Recorder::dir_in(env), &index.last().unwrap().trace).unwrap()
    }

    #[cfg(unix)]
    #[test]
    fn test_self_asserting_start_payload_fields_uses_the_production_prompt_path() {
        for (label, positional, file, input, expected_source) in [
            ("arg", Some("from arg"), None, "", PromptSource::Arg),
            ("file", None, Some("prompt.txt"), "", PromptSource::File),
            ("stdin", None, None, "from stdin", PromptSource::Stdin),
            ("dash", Some("-"), None, "from stdin", PromptSource::Stdin),
        ] {
            let mut f = Fixture::new(&format!("run-from-{label}"));
            f.token("team03");
            f.cfg.claude.profiles = vec!["team03".into()];
            f.claude_script(OK_RESULT);
            if let Some(file) = file {
                std::fs::write(f.path(file), "from file").unwrap();
            }
            let prompt_path = file.map(|file| f.path(file).display().to_string());
            let mut input = input.as_bytes();
            let mut output = Vec::new();
            let cfg = f.cfg.clone();
            run_from(
                "explorer",
                Some(&f.work()),
                positional,
                prompt_path.as_deref(),
                None,
                None,
                false,
                &f.env,
                || Ok(cfg),
                &mut input,
                &mut output,
            )
            .unwrap();
            let records = trace_records_for(&f.env);
            let spawn_session = records
                .iter()
                .find_map(|r| match r {
                    audit::Record::Spawn(s) => s.payload.session.clone(),
                    _ => None,
                })
                .unwrap();
            assert!(f.sessions_dir().join("9f22dd1b-aaaa.json").exists());
            assert!(f
                .sessions_dir()
                .join(format!("{spawn_session}.json"))
                .exists());
            let start = records
                .into_iter()
                .find_map(|r| match r {
                    audit::Record::Start(s) => Some(s),
                    _ => None,
                })
                .unwrap();
            assert_eq!(start.payload.prompt_source, Some(expected_source));
            assert_eq!(
                start.payload.prompt.as_deref(),
                Some(if expected_source == PromptSource::Arg {
                    "from arg"
                } else if expected_source == PromptSource::File {
                    "from file"
                } else {
                    "from stdin"
                })
            );
            assert_eq!(start.payload.prompt_file.as_deref(), prompt_path.as_deref());
        }
        for (label, positional, file, input, source, message) in [
            (
                "conflict",
                Some("arg"),
                Some("prompt.txt"),
                "file",
                None,
                "--prompt-file",
            ),
            (
                "dash-conflict",
                Some("-"),
                Some("prompt.txt"),
                "file",
                None,
                "--prompt-file",
            ),
            (
                "unreadable",
                None,
                Some("missing.txt"),
                "",
                Some(PromptSource::File),
                "reading prompt file",
            ),
            (
                "empty",
                None,
                None,
                " \n",
                Some(PromptSource::Stdin),
                "empty",
            ),
        ] {
            let f = Fixture::new(&format!("run-from-fail-{label}"));
            // The conflict rows' file exists, so only the conflict can fail them.
            std::fs::write(f.path("prompt.txt"), "from file").unwrap();
            let file_path = file.map(|file| f.path(file).display().to_string());
            let mut input = input.as_bytes();
            let mut output = Vec::new();
            let cfg = f.cfg.clone();
            assert!(run_from(
                "explorer",
                Some(&f.work()),
                positional,
                file_path.as_deref(),
                None,
                None,
                false,
                &f.env,
                || Ok(cfg),
                &mut input,
                &mut output
            )
            .is_err());
            let records = trace_records_for(&f.env);
            let start = records
                .iter()
                .find_map(|r| match r {
                    audit::Record::Start(s) => Some(s),
                    _ => None,
                })
                .unwrap();
            assert_eq!(start.payload.prompt_source, source, "{label}");
            let end = records
                .iter()
                .find_map(|r| match r {
                    audit::Record::End(e) => Some(e),
                    _ => None,
                })
                .unwrap();
            assert_eq!(end.payload.exit, Some(2), "{label}");
            let error = end.payload.error.as_deref().unwrap();
            assert!(error.contains(message), "{label}: {error}");
            assert!(!records.iter().any(|r| matches!(r, audit::Record::Spawn(_))));
        }
    }

    #[test]
    fn test_resume_from_records_prompt_fields_and_rejects_prompt_failures() {
        for (label, positional, file, input, expected_source, expected_prompt) in [
            (
                "arg",
                Some("resume arg"),
                None,
                "",
                PromptSource::Arg,
                "resume arg",
            ),
            (
                "file",
                None,
                Some("resume.txt"),
                "",
                PromptSource::File,
                "resume file",
            ),
            (
                "stdin",
                None,
                None,
                "resume stdin",
                PromptSource::Stdin,
                "resume stdin",
            ),
            (
                "dash",
                Some("-"),
                None,
                "resume stdin",
                PromptSource::Stdin,
                "resume stdin",
            ),
        ] {
            let mut f = Fixture::new(&format!("resume-from-{label}"));
            f.token("team03");
            f.cfg.claude.profiles = vec!["team03".into()];
            f.claude_script(OK_RESULT);
            if let Some(file) = file {
                std::fs::write(f.path(file), "resume file").unwrap();
            }
            let requested = "9f22dd1b-aaaa";
            let record = record_of(&resolved(), Path::new(&f.work()), Some(&account()), f.now);
            session::save_in(&f.sessions_dir(), requested, &record).unwrap();
            let cfg = f.cfg.clone();
            let mut input = input.as_bytes();
            let mut output = Vec::new();
            let file_path = file.map(|file| f.path(file).display().to_string());
            resume_from(
                requested,
                false,
                None,
                positional,
                file_path.as_deref(),
                false,
                &f.env,
                || Ok(cfg),
                &mut input,
                &mut output,
                &mut Vec::new(),
            )
            .unwrap();
            let records = trace_records_for(&f.env);
            let start = records
                .iter()
                .find_map(|r| match r {
                    audit::Record::Start(s) => Some(s),
                    _ => None,
                })
                .unwrap();
            assert_eq!(start.payload.cmd, AuditCommand::Resume);
            assert!(start.payload.agent.is_none());
            assert_eq!(start.payload.session.as_deref(), Some(requested));
            assert_eq!(start.payload.prompt_source, Some(expected_source));
            assert_eq!(start.payload.prompt_file.as_deref(), file_path.as_deref());
            assert_eq!(start.payload.prompt.as_deref(), Some(expected_prompt));
        }

        for (label, positional, file, input, message) in [
            (
                "conflict",
                Some("arg"),
                Some("missing.txt"),
                "",
                "--prompt-file",
            ),
            (
                "dash-conflict",
                Some("-"),
                Some("missing.txt"),
                "",
                "--prompt-file",
            ),
            (
                "missing",
                None,
                Some("missing.txt"),
                "",
                "reading prompt file",
            ),
            ("empty", None, None, " \n", "prompt"),
        ] {
            let mut f = Fixture::new(&format!("resume-from-fail-{label}"));
            f.token("team03");
            f.cfg.claude.profiles = vec!["team03".into()];
            f.claude_script(OK_RESULT);
            let record = record_of(&resolved(), Path::new(&f.work()), Some(&account()), f.now);
            session::save_in(&f.sessions_dir(), "9f22dd1b-aaaa", &record).unwrap();
            let cfg = f.cfg.clone();
            let mut input = input.as_bytes();
            let mut output = Vec::new();
            assert!(resume_from(
                "9f22dd1b-aaaa",
                false,
                None,
                positional,
                file,
                false,
                &f.env,
                || Ok(cfg),
                &mut input,
                &mut output,
                &mut Vec::new(),
            )
            .is_err());
            let records = trace_records_for(&f.env);
            let start = records
                .iter()
                .find_map(|r| match r {
                    audit::Record::Start(s) => Some(s),
                    _ => None,
                })
                .unwrap();
            assert!(start.payload.agent.is_none());
            assert_eq!(start.payload.session.as_deref(), Some("9f22dd1b-aaaa"));
            assert!(start.payload.prompt.is_none());
            if label.ends_with("conflict") {
                assert!(start.payload.prompt_source.is_none(), "{label}");
            }
            let ends = records
                .iter()
                .filter_map(|r| match r {
                    audit::Record::End(e) => Some(e),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(ends.len(), 1);
            assert_eq!(ends[0].payload.exit, Some(2));
            assert!(ends[0].payload.error.as_deref().unwrap().contains(message));
            assert!(!records.iter().any(|r| matches!(r, audit::Record::Spawn(_))));
        }
    }

    #[test]
    fn errhandling_end_record_preserves_run_outcome_when_stdout_write_fails() {
        let mut f = Fixture::new("run-output-failure");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.claude_script(OK_RESULT);
        let cfg = f.cfg.clone();
        let mut input = b"go".as_slice();
        let mut sink = FailingWrite;
        let err = run_from(
            "explorer",
            Some(&f.work()),
            Some("go"),
            None,
            None,
            None,
            false,
            &f.env,
            || Ok(cfg),
            &mut input,
            &mut sink,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("writing the result"),
            "resume/run returned an unexpected error: {err:#}"
        );
        let end = trace_records_for(&f.env)
            .into_iter()
            .find_map(|r| match r {
                audit::Record::End(e) => Some(e),
                _ => None,
            })
            .unwrap();
        assert_eq!(end.payload.exit, Some(0));
        assert_eq!(end.payload.session.as_deref(), Some("9f22dd1b-aaaa"));
        assert_eq!(end.payload.result.as_deref(), Some("the answer"));
        assert_eq!(end.payload.native.as_ref().unwrap()["num_turns"], 1);
        assert!(end.payload.error.unwrap().contains("writing the result"));
    }

    #[cfg(unix)]
    #[test]
    fn test_missing_end_warnings_fresh_run_with_matching_preset_is_empty() {
        let mut f = Fixture::new("run-matching-preset");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        let script = f.path("fake-claude");
        f.write_executable(&script, "#!/bin/sh\npreset=\nwhile [ \"$#\" -gt 0 ]; do\n  if [ \"$1\" = \"--session-id\" ]; then preset=\"$2\"; fi\n  shift\ndone\nprintf '{\"type\":\"result\",\"session_id\":\"%s\",\"result\":\"the answer\",\"is_error\":false,\"num_turns\":1,\"total_cost_usd\":0.01,\"model_usage_since_last_result\":{\"m\":{\"costUSD\":0.01}}}' \"$preset\"\n");
        f.env
            .insert("CCC_CLAUDE_BIN".into(), script.display().to_string());
        let cfg = f.cfg.clone();
        let mut input = b"prompt".as_slice();
        let mut output = Vec::new();
        run_from(
            "explorer",
            Some(&f.work()),
            Some("prompt"),
            None,
            None,
            None,
            false,
            &f.env,
            || Ok(cfg),
            &mut input,
            &mut output,
        )
        .unwrap();
        let records = trace_records_for(&f.env);
        let spawn = records
            .iter()
            .find_map(|r| match r {
                audit::Record::Spawn(s) => Some(s.payload.session.clone()),
                _ => None,
            })
            .flatten();
        let end = records
            .into_iter()
            .find_map(|r| match r {
                audit::Record::End(e) => Some(e),
                _ => None,
            })
            .unwrap();
        assert!(end.payload.warnings.is_empty());
        assert_eq!(end.payload.session, spawn);
    }

    #[test]
    fn errhandling_end_record_preserves_resume_outcome_when_stdout_write_fails() {
        let mut f = Fixture::new("resume-output-failure");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.claude_script(OK_RESULT);
        let requested = "9f22dd1b-aaaa";
        let record = record_of(&resolved(), Path::new(&f.work()), Some(&account()), f.now);
        session::save_in(&f.sessions_dir(), requested, &record).unwrap();
        let cfg = f.cfg.clone();
        let mut input = b"again".as_slice();
        let mut sink = FailingWrite;
        let err = resume_from(
            "9f22dd1b-aaaa",
            false,
            None,
            Some("again"),
            None,
            false,
            &f.env,
            || Ok(cfg),
            &mut input,
            &mut sink,
            &mut Vec::new(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("writing the result"),
            "resume/run returned an unexpected error: {err:#}"
        );
        let ends = trace_records_for(&f.env)
            .into_iter()
            .filter_map(|r| match r {
                audit::Record::End(e) => Some(e),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(ends.len(), 1);
        let end = &ends[0];
        assert_eq!(end.payload.exit, Some(0));
        assert_eq!(end.payload.session.as_deref(), Some("9f22dd1b-aaaa"));
        assert_eq!(end.payload.result.as_deref(), Some("the answer"));
        assert!(end.payload.native.is_some());
        assert!(end
            .payload
            .error
            .as_deref()
            .unwrap()
            .contains("writing the result"));
    }

    struct TimedPromptReader {
        read_at: Option<Timestamp>,
        bytes: Vec<u8>,
    }
    impl Read for TimedPromptReader {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            if self.bytes.is_empty() {
                return Ok(0);
            }
            self.read_at = Some(Timestamp::now());
            std::thread::sleep(std::time::Duration::from_millis(20));
            let n = self.bytes.len().min(output.len());
            output[..n].copy_from_slice(&self.bytes[..n]);
            self.bytes.drain(..n);
            Ok(n)
        }
    }

    #[test]
    fn correctness_start_timestamp_includes_prompt_read_time_and_resolved_prompt() {
        let mut f = Fixture::new("prompt-timing");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.claude_script(OK_RESULT);
        let cfg = f.cfg.clone();
        let mut input = TimedPromptReader {
            read_at: None,
            bytes: b"delayed prompt".to_vec(),
        };
        let mut output = Vec::new();
        run_from(
            "explorer",
            Some(&f.work()),
            None,
            None,
            None,
            None,
            false,
            &f.env,
            || Ok(cfg),
            &mut input,
            &mut output,
        )
        .unwrap();
        let records = trace_records_for(&f.env);
        let start = records
            .iter()
            .find_map(|r| match r {
                audit::Record::Start(s) => Some(s),
                _ => None,
            })
            .unwrap();
        let end = records
            .iter()
            .find_map(|r| match r {
                audit::Record::End(e) => Some(e),
                _ => None,
            })
            .unwrap();
        assert!(start.ts <= input.read_at.unwrap());
        assert_eq!(start.payload.prompt.as_deref(), Some("delayed prompt"));
        assert!(end.duration_ms >= 15);
    }

    #[cfg(unix)]
    #[test]
    fn test_vacuous_launch_duration_and_end_warnings_are_integration_assertions() {
        let _lock = supervise::TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut f = Fixture::new("launch-duration");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.launch_script("sleep 0.02; exit 0", false);
        start_launch(&f);
        let usage = || UsageSource::Unavailable("not consulted".into());
        let completion = launch_with_ctx(
            &f.ctx(&usage),
            LaunchTarget::Fresh {
                agent: "explorer",
                cwd: Some(&f.work()),
                cli_harness: None,
                allow_read: &[],
                allow_write: &[],
                allow_exec: &[],
            },
            None,
            vec![],
        )
        .unwrap();
        record_completion(&f, completion);
        let end = audit_end(&f);
        assert!(end.duration_ms > 0 && end.duration_ms < 10_000);
        assert!(end.payload.warnings.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn correctness_launch_does_not_warn_on_a_preset_session_mismatch() {
        let _lock = supervise::TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut f = Fixture::new("launch-mismatch");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.launch_script("exit 0", false);
        start_launch(&f);
        let trace = f.audit.lineage().trace_id.clone();
        audit::append_turn(
            &Recorder::dir_in(&f.env),
            &trace,
            audit::Turn {
                ts: Timestamp::now(),
                inv: f.audit.lineage().invocation_id.clone(),
                harness: Harness::Claude,
                session: "reported-session".into(),
                turn: None,
                result: Some("result".into()),
                transcript: None,
            },
        )
        .unwrap();
        let usage = || UsageSource::Unavailable("not consulted".into());
        let completion = launch_with_ctx(
            &f.ctx(&usage),
            LaunchTarget::Fresh {
                agent: "explorer",
                cwd: Some(&f.work()),
                cli_harness: None,
                allow_read: &[],
                allow_write: &[],
                allow_exec: &[],
            },
            None,
            vec![],
        )
        .unwrap();
        record_completion(&f, completion);
        let records = audit_records(&f);
        let spawn = records
            .iter()
            .find_map(|r| match r {
                audit::Record::Spawn(s) => Some(s),
                _ => None,
            })
            .unwrap();
        let end = audit_end(&f);
        assert_eq!(end.payload.session.as_deref(), Some("reported-session"));
        assert_eq!(end.payload.warnings.len(), 1);
        assert!(end.payload.warnings[0].contains(spawn.payload.session.as_deref().unwrap()));
        assert!(end.payload.warnings[0].contains("reported-session"));
    }

    #[cfg(unix)]
    #[test]
    fn test_weak_launch_preset_session_assert() {
        let _lock = supervise::TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut f = Fixture::new("launch-preset-fallback");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.launch_script("exit 0", false);
        start_launch(&f);
        let usage = || UsageSource::Unavailable("not consulted".into());
        let completion = launch_with_ctx(
            &f.ctx(&usage),
            LaunchTarget::Fresh {
                agent: "explorer",
                cwd: Some(&f.work()),
                cli_harness: None,
                allow_read: &[],
                allow_write: &[],
                allow_exec: &[],
            },
            None,
            vec![],
        )
        .unwrap();
        record_completion(&f, completion);
        let records = audit_records(&f);
        let preset = records
            .iter()
            .find_map(|r| match r {
                audit::Record::Spawn(s) => Some(s.payload.session.clone()),
                _ => None,
            })
            .flatten();
        let end = audit_end(&f);
        assert_eq!(end.payload.session, preset);
        assert!(end.payload.warnings.is_empty());
    }

    #[test]
    fn test_missing_end_best_effort_failure() {
        let dir = Fixture::new("end-best-effort");
        let recorder = Recorder::new(&dir.env).unwrap();
        recorder
            .start(start_payload_for_test(AuditCommand::Run))
            .unwrap();
        let path = recorder.trace_path();
        let backup = path.with_extension("jsonl.save");
        std::fs::rename(&path, &backup).unwrap();
        std::fs::create_dir(&path).unwrap();
        recorder.end(EndPayload {
            exit: Some(0),
            signal: None,
            session: None,
            result: None,
            error: None,
            warnings: vec![],
            native: None,
            turns: None,
        });
        std::fs::remove_dir(&path).unwrap();
        std::fs::rename(&backup, &path).unwrap();
        recorder.end(EndPayload {
            exit: Some(0),
            signal: None,
            session: None,
            result: None,
            error: None,
            warnings: vec![],
            native: None,
            turns: None,
        });
        assert_eq!(
            audit::read_trace(&Recorder::dir_in(&dir.env), &recorder.lineage().trace_id)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn test_missing_index_line_content() {
        let dir = TempDir::new("index-content");
        let mut environment = BTreeMap::new();
        environment.insert("XDG_STATE_HOME".into(), dir.path().display().to_string());
        environment.insert("CCC_CONFIG".into(), "/etc/ccc.toml".into());
        environment.insert(
            "CCC_AGENTS_CONFIG".into(),
            "/nonexistent/agents.toml".into(),
        );
        environment.insert("CCC_MODEL_ALL".into(), "model".into());
        environment.insert("CCC_UNKNOWN".into(), "secret".into());
        let cwd_before = std::env::current_dir().unwrap();
        let recorder = Recorder::new(&environment).unwrap();
        recorder
            .start(StartPayload {
                cwd_arg: None,
                ..start_payload_for_test(AuditCommand::Run)
            })
            .unwrap();
        let index = audit::read_index(&Recorder::dir_in(&environment))
            .unwrap()
            .pop()
            .unwrap();
        assert_eq!(index.trace, recorder.lineage().trace_id);
        assert_eq!(index.inv, recorder.lineage().invocation_id);
        assert_eq!(index.cwd, cwd_before.to_string_lossy());
        assert_eq!(index.cmd, AuditCommand::Run);
        assert_eq!(
            index.env.get("CCC_CONFIG").map(String::as_str),
            Some("/etc/ccc.toml")
        );
        assert_eq!(index.agents_config, "/nonexistent/agents.toml");
        assert!(!index.agents_config_exists);
        assert_eq!(
            index.env.get("CCC_AGENTS_CONFIG").map(String::as_str),
            Some("/nonexistent/agents.toml")
        );
        assert!(index.env_other.iter().any(|x| x == "CCC_UNKNOWN"));
        let raw =
            std::fs::read_to_string(Recorder::dir_in(&environment).join("traces.jsonl")).unwrap();
        assert!(!raw.contains("secret"));

        let mut f = Fixture::new("index-production-content");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        f.claude_script(OK_RESULT);
        let config_path = f.path("configured.toml");
        std::fs::write(&config_path, "").unwrap();
        f.env
            .insert("CCC_CONFIG".into(), config_path.display().to_string());
        let agents_config_path = f.path("configured-agents.toml");
        std::fs::write(&agents_config_path, "").unwrap();
        f.env.insert(
            "CCC_AGENTS_CONFIG".into(),
            agents_config_path.display().to_string(),
        );
        f.env.insert("CCC_MODEL_ALL".into(), "model".into());
        f.env.insert("CCC_UNKNOWN".into(), "secret".into());
        let requested_cwd = f.work();
        let expected_cwd = std::fs::canonicalize(&requested_cwd).unwrap();
        let cfg = f.cfg.clone();
        let mut input = b"prompt".as_slice();
        let mut output = Vec::new();
        run_from(
            "explorer",
            Some(&requested_cwd),
            Some("prompt"),
            None,
            None,
            None,
            false,
            &f.env,
            || Ok(cfg),
            &mut input,
            &mut output,
        )
        .unwrap();
        let index = audit::read_index(&Recorder::dir_in(&f.env))
            .unwrap()
            .pop()
            .unwrap();
        let records = audit::read_trace(&Recorder::dir_in(&f.env), &index.trace).unwrap();
        let start = records
            .iter()
            .find_map(|r| match r {
                audit::Record::Start(s) => Some(s),
                _ => None,
            })
            .unwrap();
        assert_eq!(index.inv, start.inv);
        assert_eq!(index.pid, start.payload.pid);
        assert_eq!(index.cmd, AuditCommand::Run);
        assert_eq!(index.agent.as_deref(), Some("explorer"));
        assert_eq!(index.cwd, expected_cwd.to_string_lossy());
        assert_eq!(index.config, config_path.to_string_lossy());
        assert!(index.config_exists);
        assert_eq!(index.agents_config, agents_config_path.to_string_lossy());
        assert!(index.agents_config_exists);
        assert_eq!(
            index.env.get("CCC_CONFIG"),
            Some(&config_path.to_string_lossy().into_owned())
        );
        assert_eq!(
            index.env.get("CCC_MODEL_ALL").map(String::as_str),
            Some("model")
        );
        assert!(index.env_other.iter().any(|x| x == "CCC_UNKNOWN"));
        assert!(!serde_json::to_string(&index).unwrap().contains("secret"));
        assert!(
            !std::fs::read_to_string(Recorder::dir_in(&f.env).join("traces.jsonl"))
                .unwrap()
                .contains("secret")
        );
    }

    const TRACE_A: &str = "aaaaaaaa-1111-4111-8111-111111111111";
    const TRACE_B: &str = "bbbbbbbb-2222-4222-8222-222222222222";

    fn test_ts(ts: &str) -> Timestamp {
        ts.parse().unwrap()
    }

    /// A start record for `inv` under `parent`, from `start_payload_for_test`.
    fn start_record(ts: &str, inv: &str, parent: Option<&str>) -> audit::Record {
        audit::Record::Start(audit::StartRecord {
            ts: test_ts(ts),
            inv: inv.into(),
            parent: parent.map(str::to_string),
            payload: start_payload_for_test(AuditCommand::Run),
        })
    }

    fn session_record(ts: &str, inv: &str, session: &str) -> audit::Record {
        audit::Record::Session(audit::SessionRecord {
            ts: test_ts(ts),
            inv: inv.into(),
            session: session.into(),
            native: None,
        })
    }

    /// Writes `records` as trace `trace` in `audit_dir`.
    fn write_audit_trace(audit_dir: &Path, trace: &str, records: &[audit::Record]) {
        let text = records
            .iter()
            .map(|r| serde_json::to_string(r).unwrap() + "\n")
            .collect::<String>();
        std::fs::write(audit_dir.join(format!("{trace}.jsonl")), text).unwrap();
    }

    /// Appends an index line naming `inv` as `trace`'s root to `audit_dir`.
    fn append_index_line(audit_dir: &Path, trace: &str, inv: &str) {
        let line = audit::IndexLine {
            trace: trace.into(),
            ts: test_ts("2026-01-01T00:00:00Z"),
            inv: inv.into(),
            pid: 1,
            cmd: AuditCommand::Run,
            agent: Some("explorer".into()),
            cwd: "/w".into(),
            config: "/c".into(),
            config_exists: true,
            agents_config: "/a".into(),
            agents_config_exists: true,
            env: BTreeMap::new(),
            env_other: vec![],
        };
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(audit_dir.join("traces.jsonl"))
            .unwrap();
        writeln!(file, "{}", serde_json::to_string(&line).unwrap()).unwrap();
    }

    /// A state home holding only an audit directory with `TRACE_A`, whose
    /// indexed root `root` ran under session `TRACE_B`.
    fn resolver_env(label: &str) -> (TempDir, BTreeMap<String, String>) {
        let dir = TempDir::new(label);
        let env = BTreeMap::from([(
            "XDG_STATE_HOME".to_string(),
            dir.path().join("state").display().to_string(),
        )]);
        let audit_dir = Recorder::dir_in(&env);
        std::fs::create_dir_all(&audit_dir).unwrap();
        write_audit_trace(
            &audit_dir,
            TRACE_A,
            &[
                start_record("2026-01-01T00:00:00Z", "root", None),
                session_record("2026-01-01T00:00:01Z", "root", TRACE_B),
            ],
        );
        append_index_line(&audit_dir, TRACE_A, "root");
        (dir, env)
    }

    fn resolve_arg(env: &BTreeMap<String, String>, arg: &str) -> Result<String> {
        resolve_resume_arg(env, arg, &mut Vec::new())
    }

    #[test]
    fn resume_arg_trace_prefix_resolves_once_to_the_root_session() {
        let (_dir, env) = resolver_env("resume-arg-trace");
        assert_eq!(resolve_arg(&env, &TRACE_A[..8]).unwrap(), TRACE_B);
        assert_eq!(resolve_arg(&env, TRACE_A).unwrap(), TRACE_B);
        assert!(!session::sessions_dir_in(&env).exists());
    }

    #[test]
    fn resume_arg_existing_sidecar_entries_win_over_matching_traces() {
        let (_dir, env) = resolver_env("resume-arg-exact");
        let sessions = session::sessions_dir_in(&env);
        std::fs::create_dir_all(&sessions).unwrap();
        let prefix = &TRACE_A[..8];
        std::fs::write(sessions.join(format!("{prefix}.json")), "not a record").unwrap();
        assert_eq!(resolve_arg(&env, prefix).unwrap(), prefix);
        std::fs::create_dir(sessions.join(format!("{TRACE_A}.json"))).unwrap();
        assert_eq!(resolve_arg(&env, TRACE_A).unwrap(), TRACE_A);
        #[cfg(unix)]
        {
            let short = &TRACE_A[..6];
            std::os::unix::fs::symlink(
                sessions.join("nowhere"),
                sessions.join(format!("{short}.json")),
            )
            .unwrap();
            assert_eq!(resolve_arg(&env, short).unwrap(), short);
        }
    }

    #[test]
    fn resume_arg_exact_ids_need_no_audit_history() {
        let dir = TempDir::new("resume-arg-no-audit");
        let env = BTreeMap::from([(
            "XDG_STATE_HOME".to_string(),
            dir.path().display().to_string(),
        )]);
        let sessions = session::sessions_dir_in(&env);
        std::fs::create_dir_all(&sessions).unwrap();
        std::fs::write(sessions.join("child-session.json"), "{}").unwrap();
        assert_eq!(resolve_arg(&env, "child-session").unwrap(), "child-session");
        let e = format!("{:#}", resolve_arg(&env, "missing").unwrap_err());
        assert!(e.contains("unknown session id \"missing\""), "{e}");
        assert!(e.contains("no audit trace matches \"missing\""), "{e}");
    }

    #[test]
    fn resume_arg_invalid_ids_and_probe_failures_never_fall_back() {
        let (_dir, env) = resolver_env("resume-arg-invalid");
        for bad in ["", "../x", "a/b"] {
            let e = format!("{:#}", resolve_arg(&env, bad).unwrap_err());
            assert!(e.contains("invalid session id"), "{bad}: {e}");
        }
        let sessions = session::sessions_dir_in(&env);
        std::fs::create_dir_all(sessions.parent().unwrap()).unwrap();
        std::fs::write(&sessions, "a file where the directory belongs").unwrap();
        let e = format!("{:#}", resolve_arg(&env, &TRACE_A[..8]).unwrap_err());
        assert!(e.contains("checking session record"), "{e}");
    }

    #[test]
    fn resume_arg_catalog_failures_name_the_missing_session_record() {
        let dir = TempDir::new("resume-arg-catalog-failure");
        let env = BTreeMap::from([(
            "XDG_STATE_HOME".to_string(),
            dir.path().display().to_string(),
        )]);
        let audit_dir = Recorder::dir_in(&env);
        std::fs::create_dir_all(audit_dir.parent().unwrap()).unwrap();
        std::fs::write(&audit_dir, "a file where the directory belongs").unwrap();
        let e = format!("{:#}", resolve_arg(&env, "missing").unwrap_err());
        let record = session::sessions_dir_in(&env).join("missing.json");
        assert!(
            e.contains(&format!("no session record at {}", record.display())),
            "{e}"
        );
        assert!(
            e.contains("looking \"missing\" up as an audit trace prefix"),
            "{e}"
        );
        assert!(
            e.contains(&format!("reading audit directory {}", audit_dir.display())),
            "{e}"
        );
    }

    #[test]
    fn resume_arg_does_not_reinterpret_the_derived_session_as_a_trace() {
        let (_dir, env) = resolver_env("resume-arg-once");
        let audit_dir = Recorder::dir_in(&env);
        std::fs::write(
            audit_dir.join(format!("{TRACE_B}.jsonl")),
            "{\"t\":\"session\",\"ts\":\"2026-01-01T00:00:00Z\",\"inv\":\"x\",\"session\":\"other\"}\n",
        )
        .unwrap();
        assert_eq!(resolve_arg(&env, &TRACE_A[..8]).unwrap(), TRACE_B);
    }

    #[test]
    fn resume_from_resolution_failure_has_no_side_effects() {
        struct Sentinel;
        impl Read for Sentinel {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                panic!("the prompt was read before resolution")
            }
        }
        let (_dir, env) = resolver_env("resume-arg-boundary");
        let audit_dir = Recorder::dir_in(&env);
        let before = std::fs::read(audit_dir.join("traces.jsonl")).unwrap();
        let mut out = Vec::new();
        for json in [false, true] {
            let e = resume_from(
                "cccccccc",
                false,
                None,
                Some("-"),
                None,
                json,
                &env,
                || panic!("configuration loaded before resolution"),
                &mut Sentinel,
                &mut out,
                &mut Vec::new(),
            )
            .unwrap_err();
            assert!(format!("{e:#}").contains("no audit trace matches"), "{e:#}");
        }
        assert!(out.is_empty());
        assert_eq!(
            std::fs::read(audit_dir.join("traces.jsonl")).unwrap(),
            before
        );
        assert_eq!(std::fs::read_dir(&audit_dir).unwrap().count(), 2);
        assert!(!session::sessions_dir_in(&env).exists());
    }

    #[test]
    fn launch_resolution_failure_has_no_side_effects() {
        let (_dir, env) = resolver_env("launch-arg-boundary");
        let audit_dir = Recorder::dir_in(&env);
        let twin = "aaaaaaaa-9999-4999-8999-999999999999";
        std::fs::copy(
            audit_dir.join(format!("{TRACE_A}.jsonl")),
            audit_dir.join(format!("{twin}.jsonl")),
        )
        .unwrap();
        let rootless = "dddddddd-4444-4444-8444-444444444444";
        write_audit_trace(
            &audit_dir,
            rootless,
            &[start_record("2026-01-01T00:00:00Z", "child", Some("gone"))],
        );
        let snapshot = || {
            std::fs::read_dir(&audit_dir)
                .unwrap()
                .map(|e| {
                    let e = e.unwrap();
                    (e.file_name(), std::fs::read(e.path()).unwrap())
                })
                .collect::<BTreeMap<_, _>>()
        };
        let before = snapshot();
        let cases = [
            ("cccccccc", vec!["no audit trace matches"]),
            ("aaaaaaaa", vec!["ambiguous trace selector", TRACE_A, twin]),
            ("dddddddd", vec!["cannot resume trace", rootless]),
        ];
        for (arg, expected) in cases {
            for fork in [false, true] {
                let e = launch_in(
                    LaunchTarget::Resume { id: arg, fork },
                    None,
                    vec![],
                    &env,
                    || panic!("configuration loaded before resolution"),
                    &mut Vec::new(),
                )
                .unwrap_err();
                let e = format!("{e:#}");
                for text in &expected {
                    assert!(e.contains(text), "{arg}: {e}");
                }
            }
        }
        assert_eq!(snapshot(), before);
        assert!(!session::sessions_dir_in(&env).exists());
    }

    /// Records `trace` in `f`'s audit directory: its indexed root `root_inv`
    /// ran under `session`, while a child written first and a second
    /// parentless invocation carry other sessions.
    fn seed_listed_trace(f: &Fixture, trace: &str, root_inv: &str, session: &str) {
        let audit_dir = Recorder::dir_in(&f.env);
        write_audit_trace(
            &audit_dir,
            trace,
            &[
                start_record("2026-01-01T00:00:02Z", "child-inv", Some(root_inv)),
                session_record("2026-01-01T00:00:03Z", "child-inv", "child-session"),
                start_record("2026-01-01T00:00:00Z", root_inv, None),
                session_record("2026-01-01T00:00:01Z", root_inv, session),
                start_record("2026-01-01T00:00:04Z", "other-root", None),
                session_record("2026-01-01T00:00:05Z", "other-root", "other-session"),
            ],
        );
        append_index_line(&audit_dir, trace, root_inv);
    }

    /// The first data token of the plain `ccc log` listing, which must hold
    /// exactly one row.
    fn listed_token(f: &Fixture) -> String {
        let options = crate::audit_view::LogOptions {
            trace: None,
            latest: false,
            flat: false,
            verbose: false,
            json: false,
        };
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = crate::audit_view::log_with(
            &Recorder::dir_in(&f.env),
            &options,
            Timestamp::now(),
            120,
            &mut out,
            &mut err,
        )
        .unwrap();
        assert_eq!(code, 0);
        assert!(err.is_empty(), "{}", String::from_utf8_lossy(&err));
        let text = String::from_utf8(out).unwrap();
        let mut lines = text.lines();
        assert!(lines.next().unwrap().starts_with("TRACE "), "{text}");
        let rows: Vec<&str> = lines.collect();
        assert_eq!(rows.len(), 1, "{text}");
        rows[0].split_whitespace().next().unwrap().to_string()
    }

    /// `ccc log <token>` selects the trace and its tree shows `session`.
    fn assert_log_selects(f: &Fixture, token: &str, trace: &str, session: &str) {
        let options = crate::audit_view::LogOptions {
            trace: Some(token.to_string()),
            latest: false,
            flat: false,
            verbose: true,
            json: false,
        };
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = crate::audit_view::log_with(
            &Recorder::dir_in(&f.env),
            &options,
            Timestamp::now(),
            200,
            &mut out,
            &mut err,
        )
        .unwrap();
        assert_eq!(code, 0, "{}", String::from_utf8_lossy(&err));
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains(trace), "{text}");
        assert!(text.contains(session), "{text}");
    }

    /// The trace the latest invocation opened: fresh, never `seeded`.
    #[cfg(unix)]
    fn latest_trace(f: &Fixture, seeded: &str) -> String {
        let index = audit::read_index(&Recorder::dir_in(&f.env))
            .unwrap()
            .pop()
            .unwrap();
        assert_ne!(index.trace, seeded);
        index.trace
    }

    #[cfg(unix)]
    #[test]
    fn a_trace_token_resume_with_a_profile_runs_and_audits_that_account() {
        let mut f = Fixture::new("resume-by-trace-profile");
        f.token("team03");
        f.token("team09");
        f.cfg.claude.profiles = vec!["team03".into()];
        seed_parent(&f, "9f22dd1b-aaaa", Harness::Claude);
        seed_listed_trace(&f, TRACE_A, "root-inv", "9f22dd1b-aaaa");
        let token = listed_token(&f);
        f.claude_script(OK_RESULT);

        let mut input = "unused".as_bytes();
        let mut output = Vec::new();
        let code = resume_from(
            &token,
            false,
            Some("team09"),
            Some("again"),
            None,
            true,
            &f.env,
            || Ok(f.cfg.clone()),
            &mut input,
            &mut output,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(code, 0, "{}", String::from_utf8_lossy(&output));
        let v: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(v["profile"], "team09");
        assert!(f.call("env").contains("CLAUDE_CODE_OAUTH_TOKEN="));

        let records = latest_records(&f.env);
        let start = start_of(&records);
        assert_eq!(start.payload.session.as_deref(), Some("9f22dd1b-aaaa"));
        assert_eq!(start.payload.profile_arg.as_deref(), Some("team09"));
        let spawn = spawn_of(&records);
        assert_eq!(spawn.payload.profile.as_deref(), Some("team09"));
        assert_eq!(spawn.payload.auth, Some(ClaudeAuth::Token));
        assert!(spawn.payload.selection_notes.is_empty());
        let saved = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
        assert_eq!(saved.profile.as_deref(), Some("team09"));
    }

    #[cfg(unix)]
    #[test]
    fn a_listed_trace_token_resumes_its_root_claude_session() {
        let mut f = Fixture::new("resume-by-trace-claude");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        seed_parent(&f, "9f22dd1b-aaaa", Harness::Claude);
        let seeded = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
        seed_listed_trace(&f, TRACE_A, "root-inv", "9f22dd1b-aaaa");
        let token = listed_token(&f);
        assert_eq!(token, &TRACE_A[..8]);
        assert_log_selects(&f, &token, TRACE_A, "9f22dd1b-aaaa");
        f.claude_script(OK_RESULT);

        let mut input = "unused".as_bytes();
        let mut output = Vec::new();
        let mut diagnostics = Vec::new();
        let code = resume_from(
            &token,
            false,
            None,
            Some("again"),
            None,
            false,
            &f.env,
            || Ok(f.cfg.clone()),
            &mut input,
            &mut output,
            &mut diagnostics,
        )
        .unwrap();
        assert_eq!(code, 0, "{}", String::from_utf8_lossy(&output));
        assert_eq!(
            String::from_utf8(diagnostics).unwrap(),
            format!("note: trace {TRACE_A} resolves to root session 9f22dd1b-aaaa\n")
        );
        assert!(!String::from_utf8_lossy(&output).contains("note:"));

        let argv = argv_of(&f);
        let i = argv.iter().position(|a| a == "--resume").expect("--resume");
        assert_eq!(argv[i + 1], "9f22dd1b-aaaa");
        assert!(!argv.iter().any(|a| a == "--fork-session"), "{argv:?}");
        assert!(!argv.iter().any(|a| a.contains(&token)), "{argv:?}");
        assert_eq!(f.call("prompt"), "again");
        assert_eq!(
            std::fs::canonicalize(f.call("cwd").trim()).unwrap(),
            seeded.cwd
        );

        let trace = latest_trace(&f, TRACE_A);
        let records = latest_records(&f.env);
        let start = start_of(&records);
        assert_eq!(start.parent, None);
        assert_eq!(start.payload.cmd, AuditCommand::Resume);
        assert_eq!(start.payload.session.as_deref(), Some("9f22dd1b-aaaa"));
        assert!(!start.payload.fork);
        assert_eq!(start.payload.profile_arg, None);
        assert_eq!(
            spawn_of(&records).payload.session.as_deref(),
            Some("9f22dd1b-aaaa")
        );
        let end = records
            .iter()
            .find_map(|r| match r {
                audit::Record::End(e) => Some(&e.payload),
                _ => None,
            })
            .unwrap();
        assert_eq!(end.session.as_deref(), Some("9f22dd1b-aaaa"));

        assert_eq!(sidecar_names(&f), vec!["9f22dd1b-aaaa.json".to_string()]);
        let saved = session::load_from(&f.sessions_dir(), "9f22dd1b-aaaa").unwrap();
        assert_eq!(saved.created_at, seeded.created_at);
        assert_eq!(saved.profile.as_deref(), Some("team03"));
        assert_eq!(saved.cwd, seeded.cwd);
        assert_eq!(saved.trace.as_deref(), Some(trace.as_str()));
        assert_eq!(saved.forked_from, None);
        let locks: Vec<String> = std::fs::read_dir(f.sessions_dir())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".lock"))
            .collect();
        assert_eq!(locks, vec!["9f22dd1b-aaaa.lock".to_string()]);
        assert!(lock_is_free(&f, "9f22dd1b-aaaa"));
    }

    #[cfg(unix)]
    #[test]
    fn a_listed_trace_token_forks_its_root_codex_session() {
        let mut f = Fixture::new("fork-by-trace-codex");
        let parent = seed_parent(&f, "thread-parent", Harness::Codex);
        seed_listed_trace(&f, TRACE_A, "root-inv", "thread-parent");
        let token = listed_token(&f);
        assert_log_selects(&f, &token, TRACE_A, "thread-parent");
        let calls = f.path("calls");
        std::fs::create_dir_all(&calls).unwrap();
        f.codex_script(&format!(
            "printf '%s\\n' \"$@\" > '{}/argv'; printf '%s\\n' '{{\"type\":\"thread.started\",\"thread_id\":\"fork-thread\"}}' '{{\"type\":\"item.completed\",\"item\":{{\"type\":\"agent_message\",\"text\":\"forked\"}}}}' '{{\"type\":\"turn.completed\",\"usage\":{{}}}}'",
            calls.display()
        ));

        let mut input = "unused".as_bytes();
        let mut output = Vec::new();
        let mut diagnostics = Vec::new();
        let code = resume_from(
            &token,
            true,
            None,
            Some("again"),
            None,
            true,
            &f.env,
            || Ok(f.cfg.clone()),
            &mut input,
            &mut output,
            &mut diagnostics,
        )
        .unwrap();
        assert_eq!(code, 0, "{}", String::from_utf8_lossy(&output));
        assert_eq!(
            String::from_utf8(diagnostics).unwrap(),
            format!("note: trace {TRACE_A} resolves to root session thread-parent\n")
        );
        assert!(!String::from_utf8_lossy(&output).contains("note:"));
        let json: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(json["session"], "fork-thread", "{json}");

        let argv = argv_of(&f);
        assert_eq!(&argv[..4], &["exec", "fork", "thread-parent", "--json"]);

        let trace = latest_trace(&f, TRACE_A);
        let records = latest_records(&f.env);
        let start = start_of(&records);
        assert_eq!(start.payload.session.as_deref(), Some("thread-parent"));
        assert!(start.payload.fork);
        assert_eq!(spawn_of(&records).payload.session, None);

        let fork = session::load_from(&f.sessions_dir(), "fork-thread").unwrap();
        assert_eq!(fork.forked_from.as_deref(), Some("thread-parent"));
        assert_eq!(fork.trace.as_deref(), Some(trace.as_str()));
        assert_eq!(sidecar_text(&f, "thread-parent"), parent);
        assert_eq!(
            sidecar_names(&f),
            vec![
                "fork-thread.json".to_string(),
                "thread-parent.json".to_string()
            ]
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_trace_token_fork_answered_under_the_root_session_is_a_collision() {
        let mut f = Fixture::new("fork-by-trace-collision");
        f.token("team03");
        f.cfg.claude.profiles = vec!["team03".into()];
        let parent = seed_parent(&f, "9f22dd1b-aaaa", Harness::Claude);
        seed_listed_trace(&f, TRACE_A, "root-inv", "9f22dd1b-aaaa");
        let token = listed_token(&f);
        f.claude_script(OK_RESULT);

        let mut input = "unused".as_bytes();
        let code = resume_from(
            &token,
            true,
            None,
            Some("again"),
            None,
            false,
            &f.env,
            || Ok(f.cfg.clone()),
            &mut input,
            &mut Vec::new(),
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(code, 0);
        let argv = argv_of(&f);
        let i = argv.iter().position(|a| a == "--resume").expect("--resume");
        assert_eq!(argv[i + 1], "9f22dd1b-aaaa");
        assert!(argv.iter().any(|a| a == "--fork-session"), "{argv:?}");
        assert_eq!(sidecar_text(&f, "9f22dd1b-aaaa"), parent);
        assert_eq!(sidecar_names(&f), vec!["9f22dd1b-aaaa.json".to_string()]);
        let end = latest_records(&f.env)
            .into_iter()
            .find_map(|r| match r {
                audit::Record::End(e) => Some(e.payload),
                _ => None,
            })
            .unwrap();
        assert!(
            end.warnings
                .iter()
                .any(|w| w.contains("parent's id 9f22dd1b-aaaa")),
            "{:?}",
            end.warnings
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_listed_trace_token_relaunches_its_root_session_interactively() {
        let _lock = test_io_lock();
        for (harness, fork) in [
            (Harness::Codex, false),
            (Harness::Claude, false),
            (Harness::Codex, true),
            (Harness::Claude, true),
        ] {
            let codex = harness == Harness::Codex;
            let session = if codex {
                "thread-parent"
            } else {
                "9f22dd1b-aaaa"
            };
            let reported = if fork { "fork-session" } else { session };
            let mut f = Fixture::new("launch-by-trace");
            f.token("team03");
            f.cfg.claude.profiles = vec!["team03".into()];
            let parent = seed_parent(&f, session, harness);
            let seeded = session::load_from(&f.sessions_dir(), session).unwrap();
            seed_listed_trace(&f, TRACE_A, "root-inv", session);
            let token = listed_token(&f);
            let script = format!(
                "{}\n{}\nexit 0",
                record_call_line(&f),
                turn_line(&f, if codex { "codex" } else { "claude" }, reported)
            );
            f.launch_script(&script, codex);

            let mut diagnostics = Vec::new();
            let code = launch_in(
                LaunchTarget::Resume { id: &token, fork },
                None,
                vec![],
                &f.env,
                || Ok(f.cfg.clone()),
                &mut diagnostics,
            )
            .unwrap();
            assert_eq!(code, 0, "{harness:?} fork={fork}");
            assert_eq!(
                String::from_utf8(diagnostics).unwrap(),
                format!("note: trace {TRACE_A} resolves to root session {session}\n")
            );

            let argv = argv_of(&f);
            match (codex, fork) {
                (true, false) => assert_eq!(
                    &argv[..3],
                    &["resume", session, "--include-non-interactive"]
                ),
                (true, true) => assert_eq!(&argv[..2], &["fork", session]),
                (false, _) => {
                    let i = argv.iter().position(|a| a == "--resume").expect("--resume");
                    assert_eq!(argv[i + 1], session);
                    assert_eq!(argv.iter().any(|a| a == "--fork-session"), fork, "{argv:?}");
                }
            }
            assert!(!argv.iter().any(|a| a.contains(&token)), "{argv:?}");
            assert_eq!(
                std::fs::canonicalize(f.call("cwd").trim()).unwrap(),
                seeded.cwd
            );

            let trace = latest_trace(&f, TRACE_A);
            let records = latest_records(&f.env);
            let start = start_of(&records);
            assert_eq!(start.parent, None);
            assert_eq!(start.payload.cmd, AuditCommand::Launch);
            assert_eq!(start.payload.session.as_deref(), Some(session));
            assert_eq!(start.payload.fork, fork);
            if fork {
                assert_eq!(spawn_of(&records).payload.session, None);
                assert_eq!(sidecar_text(&f, session), parent);
                let forked = session::load_from(&f.sessions_dir(), reported).unwrap();
                assert_eq!(forked.forked_from.as_deref(), Some(session));
                assert_eq!(forked.trace.as_deref(), Some(trace.as_str()));
                let mut names = vec![format!("{reported}.json"), format!("{session}.json")];
                names.sort();
                assert_eq!(sidecar_names(&f), names);
            } else {
                assert_eq!(spawn_of(&records).payload.session.as_deref(), Some(session));
                let saved = session::load_from(&f.sessions_dir(), session).unwrap();
                assert_eq!(saved.created_at, seeded.created_at);
                assert_eq!(saved.profile, seeded.profile);
                assert_eq!(saved.trace.as_deref(), Some(trace.as_str()));
                assert_eq!(sidecar_names(&f), vec![format!("{session}.json")]);
            }
            assert!(lock_is_free(&f, session));
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_trace_token_whose_root_session_has_no_record_fails_as_an_unknown_session() {
        let _lock = test_io_lock();
        let f = Fixture::new("resume-by-trace-unrecorded");
        seed_listed_trace(&f, TRACE_A, "root-inv", "unrecorded-session");
        let token = listed_token(&f);
        let mut output = Vec::new();
        let mut diagnostics = Vec::new();
        let e = resume_from(
            &token,
            false,
            None,
            Some("again"),
            None,
            false,
            &f.env,
            || Ok(f.cfg.clone()),
            &mut "unused".as_bytes(),
            &mut output,
            &mut diagnostics,
        )
        .unwrap_err();
        assert!(
            format!("{e:#}").contains("unknown session id \"unrecorded-session\""),
            "{e:#}"
        );
        assert_eq!(
            String::from_utf8(diagnostics).unwrap(),
            format!("note: trace {TRACE_A} resolves to root session unrecorded-session\n")
        );
        let trace = latest_trace(&f, TRACE_A);
        let start = start_of(&latest_records(&f.env)).payload.clone();
        assert_eq!(start.session.as_deref(), Some("unrecorded-session"));
        assert_ne!(trace, TRACE_A);
        assert!(!f.sessions_dir().exists());

        let e = launch_in(
            LaunchTarget::Resume {
                id: &token,
                fork: false,
            },
            None,
            vec![],
            &f.env,
            || Ok(f.cfg.clone()),
            &mut Vec::new(),
        )
        .unwrap_err();
        assert!(
            format!("{e:#}").contains("unknown session id \"unrecorded-session\""),
            "{e:#}"
        );
        assert!(!f.sessions_dir().exists());
    }

    const GOOSE_EXPLORER: &str =
        "---\nname: gexplorer\ndescription: Reads\nharness: goose\ngoose_model: qwen3\n---\nexplore the tree\n";
    const GOOSE_WRITER: &str = "---\nname: gwriter\ndescription: Writes\nharness: goose\ngoose_model: qwen3\ngoose_tools: write\n---\nwrite things\n";

    /// A Goose JSON result: the last line that is exactly `{` opens it.
    const GOOSE_RESULT: &str = "{\n  \"messages\": [{\"role\": \"assistant\", \"content\": [{\"type\": \"text\", \"text\": \"goose answer\"}]}],\n  \"metadata\": {\"total_tokens\": 7, \"status\": \"completed\"}\n}";

    impl Fixture {
        /// Adds the Goose agent definitions and a stand-in `goose` that records
        /// its argv, cwd, environment and stdin, then prints [`GOOSE_RESULT`].
        fn goose(&mut self) {
            std::fs::write(self.path("agents/gexplorer.md"), GOOSE_EXPLORER).unwrap();
            std::fs::write(self.path("agents/gwriter.md"), GOOSE_WRITER).unwrap();
            let out = self.path("calls");
            std::fs::create_dir_all(&out).unwrap();
            let script = format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > {out}/argv\npwd > {out}/cwd\nenv > {out}/env\ncat > {out}/prompt\ncat <<'EOF'\n{GOOSE_RESULT}\nEOF\n",
                out = out.display()
            );
            // In its own directory: a write root may not overlap the
            // directory of CCC_GOOSE_BIN.
            std::fs::create_dir_all(self.path("bin")).unwrap();
            let path = self.path("bin/fake-goose");
            self.write_executable(&path, &script);
            self.env
                .insert("CCC_GOOSE_BIN".to_string(), path.display().to_string());
            // A stand-in `bwrap` the launch preflight finds on `PATH`, so these
            // tests do not depend on the host being able to start a box. Its
            // own directory, apart from `bin/`, which may not overlap a write
            // root.
            std::fs::create_dir_all(self.path("fake-bwrap")).unwrap();
            self.write_executable(&self.path("fake-bwrap/bwrap"), "#!/bin/sh\nexit 0\n");
            self.env.insert(
                "PATH".to_string(),
                format!("{}:/usr/bin:/bin", self.path("fake-bwrap").display()),
            );
        }

        fn canonical(&self, rel: &str) -> PathBuf {
            std::fs::canonicalize(self.path(rel)).unwrap()
        }

        fn run_goose(&self, agent: &str, cwd: &str, flags: &GooseFlags) -> (Result<i32>, String) {
            let mut input = "go".as_bytes();
            let mut output = Vec::new();
            let code = run_from_report(
                agent,
                Some(cwd),
                Some("hello goose"),
                None,
                None,
                None,
                flags,
                false,
                &self.env,
                || Ok(self.cfg.clone()),
                &mut input,
                &mut output,
                &|message| diag(message),
            );
            (code, String::from_utf8(output).unwrap())
        }

        fn resume_goose(&self, id: &str, fork: bool) -> (Result<i32>, String) {
            let mut input = "again".as_bytes();
            let mut output = Vec::new();
            let code = resume_from(
                id,
                fork,
                None,
                Some("again"),
                None,
                false,
                &self.env,
                || Ok(self.cfg.clone()),
                &mut input,
                &mut output,
                &mut Vec::new(),
            );
            (code, String::from_utf8(output).unwrap())
        }

        fn goose_argv(&self) -> Vec<String> {
            self.call("argv").lines().map(str::to_string).collect()
        }
    }

    fn spawn_record(env: &BTreeMap<String, String>) -> SpawnPayload {
        latest_records(env)
            .into_iter()
            .find_map(|r| match r {
                audit::Record::Spawn(s) => Some(s.payload),
                _ => None,
            })
            .expect("a spawn record")
    }

    fn flag_value<'a>(argv: &'a [String], flag: &str) -> Option<&'a str> {
        let i = argv.iter().position(|a| a == flag)?;
        argv.get(i + 1).map(String::as_str)
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_goose_run_passes_the_real_box_preflight() {
        if crate::testutil::skip_without_box_host() {
            return;
        }
        let mut f = Fixture::new("goose-real-box");
        f.goose();
        f.env
            .insert("PATH".to_string(), "/usr/bin:/bin".to_string());
        let (code, out) = f.run_goose("gexplorer", &f.work(), &GooseFlags::default());
        assert_eq!(code.unwrap(), 0, "{out}");
    }

    #[cfg(unix)]
    #[test]
    fn a_goose_run_presets_its_session_and_records_its_policy() {
        let mut f = Fixture::new("goose-run");
        f.goose();
        let (code, out) = f.run_goose("gexplorer", &f.work(), &GooseFlags::default());
        assert_eq!(code.unwrap(), 0, "{out}");
        assert!(out.starts_with("goose answer"), "{out}");
        assert!(out.contains("harness=goose"), "{out}");
        assert!(out.contains("effort=-"), "{out}");

        let argv = f.goose_argv();
        assert_eq!(&argv[..4], ["run", "-q", "--output-format", "json"]);
        assert!(argv.iter().any(|a| a == "--no-profile"), "{argv:?}");
        let id = flag_value(&argv, "-n").unwrap().to_string();
        Uuid::parse_str(&id).expect("the preset id is a UUID");
        assert!(out.contains(&format!("session={id}")), "{out}");
        assert_eq!(flag_value(&argv, "--model"), Some("qwen3"));
        let ext = flag_value(&argv, "--with-extension").unwrap();
        assert!(ext.starts_with("box:"), "{ext}");
        assert!(ext.contains(" goose-tools --role box --policy "), "{ext}");
        assert!(argv.iter().all(|a| !a.starts_with("ccc:")), "{argv:?}");
        assert_eq!(f.call("prompt"), "hello goose");

        let root = f.cfg.goose.root_in(&f.env);
        let work = std::fs::canonicalize(crate::harness::goose::work_dir(&root)).unwrap();
        assert_eq!(f.call("cwd").trim(), work.display().to_string());
        assert!(crate::harness::goose::plugin_file(&root).is_file());
        let system = flag_value(&argv, "--system").unwrap();
        assert!(
            system.contains(&format!(
                "Your working directory is {}.",
                f.canonical("work").display()
            )),
            "{system}"
        );

        let record = session::load_from(&f.sessions_dir(), &id).unwrap();
        assert_eq!(record.harness, Harness::Goose);
        let policy = record.goose_policy.expect("the sidecar carries the policy");
        assert_eq!(policy.read_roots, vec![f.canonical("work")]);
        assert!(policy.write_roots.is_empty() && policy.exec_roots.is_empty());
        let hex = ext.rsplit(' ').next().unwrap();
        assert_eq!(
            crate::goose_policy::GoosePolicy::from_hex(hex).unwrap(),
            policy
        );

        let spawn = spawn_record(&f.env);
        assert_eq!(spawn.session.as_deref(), Some(id.as_str()));
        assert_eq!(spawn.goose_policy, Some(policy));
    }

    #[cfg(unix)]
    #[test]
    fn a_goose_resume_reuses_the_recorded_policy_without_extensions() {
        let mut f = Fixture::new("goose-resume");
        f.goose();
        std::fs::create_dir_all(f.path("extra")).unwrap();
        let flags = GooseFlags {
            allow_read: vec![f.path("extra")],
            ..GooseFlags::default()
        };
        let (code, out) = f.run_goose("gexplorer", &f.work(), &flags);
        assert_eq!(code.unwrap(), 0, "{out}");
        let id = flag_value(&f.goose_argv(), "-n").unwrap().to_string();
        let recorded = session::load_from(&f.sessions_dir(), &id)
            .unwrap()
            .goose_policy
            .unwrap();
        assert_eq!(
            recorded.read_roots,
            vec![f.canonical("work"), f.canonical("extra")]
        );

        // A changed definition does not change a recorded session's policy.
        std::fs::write(
            f.path("agents/gexplorer.md"),
            GOOSE_WRITER.replace("gwriter", "gexplorer"),
        )
        .unwrap();
        let (code, out) = f.resume_goose(&id, false);
        assert_eq!(code.unwrap(), 0, "{out}");
        assert!(out.contains(&format!("session={id}")), "{out}");
        let argv = f.goose_argv();
        assert!(argv.iter().any(|a| a == "--resume"), "{argv:?}");
        assert_eq!(flag_value(&argv, "-n"), Some(id.as_str()));
        assert!(!argv
            .iter()
            .any(|a| a == "--with-extension" || a == "--no-profile"));
        assert_eq!(f.call("prompt"), "again");
        let record = session::load_from(&f.sessions_dir(), &id).unwrap();
        assert_eq!(record.goose_policy, Some(recorded.clone()));
        assert_eq!(spawn_record(&f.env).goose_policy, Some(recorded));
    }

    #[cfg(unix)]
    #[test]
    fn a_goose_fork_is_refused_and_the_parent_left_alone() {
        let mut f = Fixture::new("goose-fork");
        f.goose();
        let (code, _) = f.run_goose("gexplorer", &f.work(), &GooseFlags::default());
        assert_eq!(code.unwrap(), 0);
        let id = flag_value(&f.goose_argv(), "-n").unwrap().to_string();
        let path = session::path_in(&f.sessions_dir(), &id).unwrap();
        let before = std::fs::read(&path).unwrap();
        std::fs::remove_file(f.path("calls/argv")).unwrap();

        let err = f.resume_goose(&id, true).0.unwrap_err();
        assert!(
            format!("{err:#}").contains("Goose has no headless fork"),
            "{err:#}"
        );
        assert!(!f.path("calls/argv").exists(), "goose was spawned");
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    #[cfg(unix)]
    #[test]
    fn a_goose_record_without_a_policy_is_not_resumed() {
        let mut f = Fixture::new("goose-no-policy");
        f.goose();
        let (code, _) = f.run_goose("gexplorer", &f.work(), &GooseFlags::default());
        assert_eq!(code.unwrap(), 0);
        let id = flag_value(&f.goose_argv(), "-n").unwrap().to_string();
        let dir = f.sessions_dir();
        let mut record = session::load_from(&dir, &id).unwrap();
        record.goose_policy = None;
        session::save_in(&dir, &id, &record).unwrap();
        std::fs::remove_file(f.path("calls/argv")).unwrap();

        let err = f.resume_goose(&id, false).0.unwrap_err();
        assert!(
            format!("{err:#}").contains("no recorded confinement policy"),
            "{err:#}"
        );
        assert!(!f.path("calls/argv").exists());
    }

    #[test]
    fn allow_flags_are_refused_on_a_non_goose_resolution() {
        let mut f = Fixture::new("goose-allow-claude");
        f.goose();
        f.claude_script(OK_RESULT);
        let flags = GooseFlags {
            allow_read: vec![f.path("work")],
            ..GooseFlags::default()
        };
        let err = f.run_goose("explorer", &f.work(), &flags).0.unwrap_err();
        let text = format!("{err:#}");
        assert!(text.contains("--allow-read"), "{text}");
        assert!(text.contains("applies only to the goose harness"), "{text}");
        assert!(!f.path("calls/argv").exists(), "the harness was spawned");
    }

    /// A stand-in `claude` that answers after `sleep` seconds under the session
    /// id ccc passed it, fresh or resumed.
    #[cfg(target_os = "linux")]
    fn claude_echoing_its_session(f: &mut Fixture, sleep: u32) {
        f.claude_custom_script(&format!(
            "cat >/dev/null\nid=; prev=\nfor a in \"$@\"; do case \"$prev\" in --session-id|--resume) id=$a;; esac; prev=$a; done\nsleep {sleep}\nprintf '{{\"type\":\"result\",\"session_id\":\"%s\",\"result\":\"the answer\",\"is_error\":false,\"num_turns\":1,\"total_cost_usd\":0.01,\"model_usage_since_last_result\":{{\"m\":{{\"costUSD\":0.01}}}}}}' \"$id\""
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn goose_host_tools_run_a_real_nested_ccc_in_slices_and_stop_it_resumably() {
        use crate::goose_policy::{GoosePolicy, GooseTool};
        use crate::goose_tools::host::HostTools;
        use crate::goose_tools::Toolset;
        let mut f = Fixture::new("goose-host-nested");
        f.token("team03");
        f.install_cli_config();
        // Its own git top, so a host /tmp/.git does not widen the surface.
        std::fs::create_dir_all(f.path("work/.git")).unwrap();
        claude_echoing_its_session(&mut f, 3);
        let work = std::fs::canonicalize(f.path("work")).unwrap();
        let policy = GoosePolicy {
            tools: [GooseTool::Ccc].into(),
            agents: vec!["explorer".into()],
            read_roots: vec![work.clone()],
            write_roots: vec![],
            exec_roots: vec![work.clone()],
        };
        let mut env = f.env.clone();
        env.insert("PATH".into(), "/usr/bin:/bin".into());
        let tools = HostTools::new(
            policy,
            crate::testutil::ccc_binary(),
            f.sessions_dir(),
            Duration::from_secs(1),
            Duration::from_secs(5),
        )
        .with_child_env(env);

        let first = tools.call(
            "ccc_run",
            &serde_json::json!({"agent": "explorer", "prompt": "go"}),
        );
        assert!(!first.is_error, "{first:?}");
        let v: serde_json::Value = serde_json::from_str(&first.text).unwrap();
        assert_eq!(v["status"], "running", "{v}");
        let run = v["run"].as_str().unwrap().to_string();
        let mut done = serde_json::Value::Null;
        for _ in 0..8 {
            let r = tools.call("ccc_wait", &serde_json::json!({"run": run}));
            assert!(!r.is_error, "{r:?}");
            done = serde_json::from_str(&r.text).unwrap();
            if done["status"] == "done" {
                break;
            }
        }
        assert_eq!(done["result"], "the answer", "{done}");
        assert_eq!(done["harness"], "claude");
        let first_id = done["session"].as_str().unwrap().to_string();
        let record = session::load_from(&f.sessions_dir(), &first_id).unwrap();
        assert_eq!(record.cwd, work);

        // A turn still running when the server shuts down is terminated, and
        // its session stays recorded and resumable. Left alone, the child would
        // run about 4 s more, so a prompt SIGTERM end shows it was terminated.
        claude_echoing_its_session(&mut f, 5);
        let second = tools.call(
            "ccc_run",
            &serde_json::json!({"agent": "explorer", "prompt": "go"}),
        );
        let v: serde_json::Value = serde_json::from_str(&second.text).unwrap();
        assert_eq!(v["status"], "running", "{v}");
        let stopping = Instant::now();
        tools.shutdown();
        let took = stopping.elapsed();
        assert!(took < Duration::from_millis(2500), "shutdown took {took:?}");
        let end = trace_records_for(&f.env)
            .into_iter()
            .find_map(|r| match r {
                audit::Record::End(end) => Some(end),
                _ => None,
            })
            .expect("the nested run's end record");
        assert_eq!(end.payload.signal, Some(libc::SIGTERM), "{:?}", end.payload);
        assert_ne!(end.payload.exit, Some(0), "{:?}", end.payload);
        let ids: Vec<String> = std::fs::read_dir(f.sessions_dir())
            .unwrap()
            .filter_map(|e| {
                let name = e.unwrap().file_name().into_string().unwrap();
                name.strip_suffix(".json").map(str::to_string)
            })
            .filter(|id| *id != first_id)
            .collect();
        assert_eq!(ids.len(), 1, "{ids:?}");
        assert_eq!(end.payload.session.as_deref(), Some(ids[0].as_str()));
        claude_echoing_its_session(&mut f, 0);
        let resumed = f.cli(&["resume", &ids[0], "--json", "again"]);
        assert!(resumed.status.success(), "{resumed:?}");
        let out: serde_json::Value = serde_json::from_slice(&resumed.stdout).unwrap();
        assert_eq!(out["session"], ids[0].as_str());
    }

    /// A host server for `policy` whose children are the real `ccc` under the
    /// fixture's environment.
    #[cfg(target_os = "linux")]
    fn host_tools_for(
        f: &Fixture,
        policy: crate::goose_policy::GoosePolicy,
    ) -> crate::goose_tools::host::HostTools {
        crate::goose_tools::host::HostTools::new(
            policy,
            crate::testutil::ccc_binary(),
            f.sessions_dir(),
            Duration::from_secs(1),
            Duration::from_secs(5),
        )
        .with_child_env(f.env.clone())
    }

    /// Follows a `running` reply with `ccc_wait` until the run is done, and
    /// returns the final reply.
    #[cfg(target_os = "linux")]
    fn host_call_until_done(
        tools: &crate::goose_tools::host::HostTools,
        name: &str,
        args: serde_json::Value,
    ) -> crate::goose_tools::ToolReply {
        use crate::goose_tools::Toolset;
        let mut reply = tools.call(name, &args);
        for _ in 0..10 {
            if reply.is_error {
                return reply;
            }
            let v: serde_json::Value = serde_json::from_str(&reply.text).unwrap();
            if v["status"] != "running" {
                return reply;
            }
            reply = tools.call("ccc_wait", &serde_json::json!({"run": v["run"]}));
        }
        panic!("the run did not finish: {reply:?}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_goose_writer_child_through_ccc_run_writes_only_where_allow_write_says() {
        use crate::goose_policy::{GoosePolicy, GooseTool};
        let mut f = Fixture::new("goose-host-writer-child");
        f.goose();
        f.install_cli_config();
        // The config directory is protected: keep it off the write root's path.
        std::fs::create_dir_all(f.path("conf")).unwrap();
        for (var, file) in [
            ("CCC_CONFIG", "config.toml"),
            ("CCC_AGENTS_CONFIG", "agents.toml"),
        ] {
            let moved = f.path("conf").join(file);
            std::fs::rename(f.path(file), &moved).unwrap();
            f.env.insert(var.into(), moved.display().to_string());
        }
        // Its own git top, so a host /tmp/.git does not widen the surface.
        std::fs::create_dir_all(f.path("work/.git")).unwrap();
        std::fs::create_dir_all(f.path("out/sub")).unwrap();
        let work = f.canonical("work");
        let out = f.canonical("out");
        let sub = f.canonical("out/sub");
        let tools = host_tools_for(
            &f,
            GoosePolicy {
                tools: [GooseTool::Write, GooseTool::Ccc].into(),
                agents: vec!["gwriter".into()],
                read_roots: vec![out.clone()],
                write_roots: vec![out.clone()],
                exec_roots: vec![work.clone()],
            },
        );

        let refused = host_call_until_done(
            &tools,
            "ccc_run",
            serde_json::json!({"agent": "gwriter", "prompt": "write"}),
        );
        assert!(refused.is_error, "{refused:?}");
        assert!(refused.text.contains("allow_write"), "{}", refused.text);
        assert!(!f.path("calls/argv").exists(), "goose was spawned");

        let done = host_call_until_done(
            &tools,
            "ccc_run",
            serde_json::json!({"agent": "gwriter", "prompt": "write", "allow_write": [sub]}),
        );
        assert!(!done.is_error, "{done:?}");
        let v: serde_json::Value = serde_json::from_str(&done.text).unwrap();
        assert_eq!(v["result"], "goose answer", "{v}");
        let id = v["session"].as_str().unwrap();
        let record = session::load_from(&f.sessions_dir(), id).unwrap();
        assert_eq!(record.cwd, work);
        let policy = record.goose_policy.unwrap();
        assert_eq!(policy.write_roots, vec![sub.clone()]);
        assert_eq!(policy.read_roots, vec![work, sub]);
        assert!(policy.exec_roots.is_empty());
        let argv = f.goose_argv();
        let ext = flag_value(&argv, "--with-extension").unwrap();
        assert_eq!(
            GoosePolicy::from_hex(ext.rsplit(' ').next().unwrap()).unwrap(),
            policy
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_host_role_started_by_goose_checks_write_roots_against_the_host_home() {
        use crate::goose_policy::{GoosePolicy, GooseTool};
        use std::io::{BufRead, Write};
        let mut f = Fixture::new("goose-host-restored-home");
        f.goose();
        f.install_cli_config();
        std::fs::create_dir_all(f.path("conf")).unwrap();
        for (var, file) in [
            ("CCC_CONFIG", "config.toml"),
            ("CCC_AGENTS_CONFIG", "agents.toml"),
        ] {
            let moved = f.path("conf").join(file);
            std::fs::rename(f.path(file), &moved).unwrap();
            f.env.insert(var.into(), moved.display().to_string());
        }
        let mut config = std::fs::OpenOptions::new()
            .append(true)
            .open(f.path("conf/config.toml"))
            .unwrap();
        config.write_all(b"[goose]\ntool_wait_s = 30\n").unwrap();
        std::fs::create_dir_all(f.path("work/.git")).unwrap();
        std::fs::create_dir_all(f.path("out/realhome/proj")).unwrap();
        std::fs::create_dir_all(f.path("emptyhome")).unwrap();
        let work = f.canonical("work");
        let out = f.canonical("out");
        let real_home = f.canonical("out/realhome");
        let proj = f.canonical("out/realhome/proj");
        let policy = GoosePolicy {
            tools: [GooseTool::Write, GooseTool::Ccc].into(),
            agents: vec!["gwriter".into()],
            read_roots: vec![out.clone()],
            write_roots: vec![out],
            exec_roots: vec![work],
        };
        let carried = crate::goose_policy::hex_encode(
            serde_json::json!({"HOME": real_home})
                .to_string()
                .as_bytes(),
        );

        let mut server = std::process::Command::new(crate::testutil::ccc_binary())
            .args([
                "goose-tools",
                "--role",
                "host",
                "--policy",
                &policy.to_hex(),
            ])
            .env_clear()
            .envs(&f.env)
            .env("HOME", f.path("emptyhome"))
            .env(crate::harness::goose::HOST_ENV_VAR, &carried)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let mut stdin = server.stdin.take().unwrap();
        let mut replies = std::io::BufReader::new(server.stdout.take().unwrap()).lines();
        let mut call = |id: u32, write_root: &Path| -> serde_json::Value {
            let request = serde_json::json!({"jsonrpc": "2.0", "id": id, "method": "tools/call",
                "params": {"name": "ccc_run", "arguments":
                    {"agent": "gwriter", "prompt": "write", "allow_write": [write_root]}}});
            writeln!(stdin, "{request}").unwrap();
            let reply: serde_json::Value =
                serde_json::from_str(&replies.next().unwrap().unwrap()).unwrap();
            reply["result"].clone()
        };

        let refused = call(1, &real_home);
        assert_eq!(refused["isError"], true, "{refused}");
        let text = refused["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("is $HOME"), "{text}");
        assert!(!f.path("calls/argv").exists(), "goose was spawned");

        let done = call(2, &proj);
        assert_eq!(done["isError"], false, "{done}");
        let v: serde_json::Value =
            serde_json::from_str(done["content"][0]["text"].as_str().unwrap()).unwrap();
        assert_eq!(v["status"], "done", "{v}");
        assert_eq!(v["result"], "goose answer", "{v}");

        drop(stdin);
        assert!(server.wait().unwrap().success());
        assert_eq!(std::fs::read_dir(f.path("emptyhome")).unwrap().count(), 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_host_role_stops_on_sigterm_or_sighup_and_terminates_its_running_children() {
        use crate::goose_policy::{GoosePolicy, GooseTool};
        use std::io::{BufRead, Write};
        for (sig, code) in [(libc::SIGTERM, 143), (libc::SIGHUP, 129)] {
            let mut f = Fixture::new("goose-host-signal");
            f.token("team03");
            f.install_cli_config();
            let mut config = std::fs::OpenOptions::new()
                .append(true)
                .open(f.path("config.toml"))
                .unwrap();
            config.write_all(b"[goose]\ntool_wait_s = 1\n").unwrap();
            // Its own git top, so a host /tmp/.git does not widen the surface.
            std::fs::create_dir_all(f.path("work/.git")).unwrap();
            claude_echoing_its_session(&mut f, 5);
            let work = f.canonical("work");
            let policy = GoosePolicy {
                tools: [GooseTool::Ccc].into(),
                agents: vec!["explorer".into()],
                read_roots: vec![work.clone()],
                write_roots: vec![],
                exec_roots: vec![work],
            };
            let mut server = std::process::Command::new(crate::testutil::ccc_binary())
                .args([
                    "goose-tools",
                    "--role",
                    "host",
                    "--policy",
                    &policy.to_hex(),
                ])
                .env_clear()
                .envs(&f.env)
                .env("PATH", "/usr/bin:/bin")
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .spawn()
                .unwrap();
            // Kept open until the server has exited: the signal, not EOF, stops it.
            let mut stdin = server.stdin.take().unwrap();
            let mut replies = std::io::BufReader::new(server.stdout.take().unwrap()).lines();
            let request = serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "ccc_run", "arguments": {"agent": "explorer", "prompt": "go"}}});
            writeln!(stdin, "{request}").unwrap();
            let reply: serde_json::Value =
                serde_json::from_str(&replies.next().unwrap().unwrap()).unwrap();
            let text = reply["result"]["content"][0]["text"].as_str().unwrap();
            let v: serde_json::Value = serde_json::from_str(text).unwrap();
            assert_eq!(v["status"], "running", "{v}");

            let stopping = Instant::now();
            crate::procs::signal(server.id(), sig).unwrap();
            let status = server.wait().unwrap();
            let took = stopping.elapsed();
            drop(stdin);
            assert_eq!(status.code(), Some(code), "signal {sig}: {status:?}");
            assert!(
                took < Duration::from_millis(2500),
                "signal {sig}: took {took:?}"
            );
            let end = trace_records_for(&f.env)
                .into_iter()
                .find_map(|r| match r {
                    audit::Record::End(end) => Some(end),
                    _ => None,
                })
                .expect("the nested run's end record");
            assert_eq!(end.payload.signal, Some(libc::SIGTERM), "{:?}", end.payload);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_depth_two_goose_chain_cannot_start_a_child_in_a_worktree_its_parent_can_write() {
        use crate::goose_policy::{check_separation, GoosePolicy, GooseTool};
        use crate::goose_tools::Toolset;
        let mut f = Fixture::new("goose-depth-two");
        f.goose();
        f.install_cli_config();
        std::fs::write(
            f.path("agents/gcoord.md"),
            "---\nname: gcoord\ndescription: Delegates\nharness: goose\ngoose_model: qwen3\ngoose_tools: ccc\ngoose_agents: explorer\n---\ndelegate\n",
        )
        .unwrap();
        // W holds a main checkout; E is a plain tree (its own top) holding a
        // worktree of that checkout, whose git directory lies under W.
        let gitdir = f.path("w/main/.git/worktrees/wt");
        std::fs::create_dir_all(&gitdir).unwrap();
        std::fs::write(gitdir.join("commondir"), "../..\n").unwrap();
        std::fs::create_dir_all(f.path("e/.git")).unwrap();
        std::fs::create_dir_all(f.path("e/wt")).unwrap();
        std::fs::write(
            f.path("e/wt/.git"),
            format!(
                "gitdir: {}\n",
                std::fs::canonicalize(&gitdir).unwrap().display()
            ),
        )
        .unwrap();
        let w = f.canonical("w");
        let e = f.canonical("e");
        let wt = f.canonical("e/wt");
        check_separation(std::slice::from_ref(&w), std::slice::from_ref(&e))
            .expect("the top-level separation check accepts W and E");

        let parent = host_tools_for(
            &f,
            GoosePolicy {
                tools: [GooseTool::Write, GooseTool::Ccc].into(),
                agents: vec!["gcoord".into()],
                read_roots: vec![w.clone()],
                write_roots: vec![w.clone()],
                exec_roots: vec![e.clone()],
            },
        );
        let done = host_call_until_done(
            &parent,
            "ccc_run",
            serde_json::json!({"agent": "gcoord", "prompt": "go", "allow_exec": [e]}),
        );
        assert!(!done.is_error, "{done:?}");

        let argv = f.goose_argv();
        let ext = flag_value(&argv, "--with-extension").unwrap();
        assert!(ext.contains(" goose-tools --role host --policy "), "{ext}");
        let child = GoosePolicy::from_hex(ext.rsplit(' ').next().unwrap()).unwrap();
        assert!(child.write_roots.is_empty(), "{child:?}");
        assert_eq!(child.exec_roots, vec![e.clone()]);
        let sessions_before = std::fs::read_dir(f.sessions_dir()).unwrap().count();
        let server = host_tools_for(&f, child);
        let refused = server.call(
            "ccc_run",
            &serde_json::json!({"agent": "explorer", "prompt": "review", "cwd": wt}),
        );
        assert!(refused.is_error, "{refused:?}");
        assert!(refused.text.contains("--allow-exec"), "{}", refused.text);
        assert_eq!(
            std::fs::read_dir(f.sessions_dir()).unwrap().count(),
            sessions_before,
            "a child was started"
        );
        server.shutdown();
        parent.shutdown();
    }

    #[test]
    fn goose_nested_is_inert_on_a_non_goose_resolution() {
        let mut f = Fixture::new("goose-nested-claude");
        f.claude_script(OK_RESULT);
        f.token("team03");
        f.token("team04");
        let usage = || report(vec![snapshot(f.now, "team03", 10.0, 10.0)]);
        let ctx = f.ctx(&usage);
        let flags = GooseFlags {
            nested: true,
            ..GooseFlags::default()
        };
        let prepared = prepare(&ctx, "explorer", Some(&f.work()), None, None, &flags).unwrap();
        assert_eq!(prepared.r.harness, Harness::Claude);
        assert_eq!(prepared.r.goose, None);
    }

    #[test]
    fn an_allow_flag_the_definition_cannot_use_is_refused_before_spawning() {
        let mut f = Fixture::new("goose-allow-unusable");
        f.goose();
        std::fs::create_dir_all(f.path("out")).unwrap();
        let flags = GooseFlags {
            allow_write: vec![f.path("out")],
            ..GooseFlags::default()
        };
        let err = f.run_goose("gexplorer", &f.work(), &flags).0.unwrap_err();
        assert!(format!("{err:#}").contains("lacks \"write\""), "{err:#}");
        assert!(!f.path("calls/argv").exists());
        assert!(
            !f.sessions_dir().exists()
                || std::fs::read_dir(f.sessions_dir())
                    .unwrap()
                    .next()
                    .is_none()
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_nested_goose_writer_takes_its_write_roots_from_the_flags_only() {
        let mut f = Fixture::new("goose-nested-writer");
        f.goose();
        std::fs::create_dir_all(f.path("out")).unwrap();
        let work = f.canonical("work").display().to_string();

        let bare = GooseFlags {
            nested: true,
            ..GooseFlags::default()
        };
        let err = f.run_goose("gwriter", &work, &bare).0.unwrap_err();
        assert!(format!("{err:#}").contains("allow_write"), "{err:#}");
        assert!(!f.path("calls/argv").exists());

        let flags = GooseFlags {
            allow_write: vec![f.canonical("out")],
            nested: true,
            ..GooseFlags::default()
        };
        let (code, out) = f.run_goose("gwriter", &work, &flags);
        assert_eq!(code.unwrap(), 0, "{out}");
        let id = flag_value(&f.goose_argv(), "-n").unwrap().to_string();
        let policy = session::load_from(&f.sessions_dir(), &id)
            .unwrap()
            .goose_policy
            .unwrap();
        assert_eq!(policy.write_roots, vec![f.canonical("out")]);
        assert_eq!(
            policy.read_roots,
            vec![f.canonical("work"), f.canonical("out")]
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_top_level_goose_writer_gets_its_cwd_as_a_write_root() {
        let mut f = Fixture::new("goose-writer");
        f.goose();
        let (code, out) = f.run_goose("gwriter", &f.work(), &GooseFlags::default());
        assert_eq!(code.unwrap(), 0, "{out}");
        let id = flag_value(&f.goose_argv(), "-n").unwrap().to_string();
        let policy = session::load_from(&f.sessions_dir(), &id)
            .unwrap()
            .goose_policy
            .unwrap();
        assert_eq!(policy.write_roots, vec![f.canonical("work")]);
    }

    #[cfg(unix)]
    #[test]
    fn a_nested_goose_run_refuses_a_cwd_that_is_not_canonical() {
        let mut f = Fixture::new("goose-nested-cwd");
        f.goose();
        std::fs::create_dir_all(f.path("real")).unwrap();
        std::os::unix::fs::symlink(f.canonical("real"), f.path("link")).unwrap();
        let link = f.canonical("work").parent().unwrap().join("link");
        let nested = GooseFlags {
            nested: true,
            ..GooseFlags::default()
        };
        let err = f
            .run_goose("gexplorer", &link.display().to_string(), &nested)
            .0
            .unwrap_err();
        assert!(format!("{err:#}").contains("not canonical"), "{err:#}");
        assert!(!f.path("calls/argv").exists());

        // Without --goose-nested the operator's symlink is resolved and the
        // target recorded.
        let (code, out) = f.run_goose(
            "gexplorer",
            &link.display().to_string(),
            &GooseFlags::default(),
        );
        assert_eq!(code.unwrap(), 0, "{out}");
        let id = flag_value(&f.goose_argv(), "-n").unwrap().to_string();
        let policy = session::load_from(&f.sessions_dir(), &id)
            .unwrap()
            .goose_policy
            .unwrap();
        assert_eq!(policy.read_roots, vec![f.canonical("real")]);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_failing_box_preflight_refuses_a_fresh_goose_run_before_any_spawn() {
        let mut f = Fixture::new("goose-preflight");
        f.goose();
        let missing = f.path("missing");
        type Edit = fn(&mut Config, &Path);
        let edits: [(&str, Edit); 2] = [
            ("bwrap", |c: &mut Config, m: &Path| {
                c.goose.bwrap = m.join("bwrap").display().to_string()
            }),
            ("box_ro", |c: &mut Config, m: &Path| {
                c.goose.box_ro = vec![m.display().to_string()]
            }),
        ];
        for (label, edit) in edits {
            let mut cfg = f.cfg.clone();
            edit(&mut cfg, &missing);
            let saved = std::mem::replace(&mut f.cfg, cfg);
            let err = f
                .run_goose("gexplorer", &f.work(), &GooseFlags::default())
                .0
                .unwrap_err();
            f.cfg = saved;
            let text = format!("{err:#}");
            assert!(
                text.contains(&missing.display().to_string()),
                "{label}: {text}"
            );
            if label == "bwrap" {
                assert!(text.contains("bubblewrap"), "{text}");
            }
            assert!(!f.path("calls/argv").exists(), "{label}: goose was spawned");
            let sessions = f.sessions_dir();
            assert!(
                !sessions.exists() || std::fs::read_dir(&sessions).unwrap().next().is_none(),
                "{label}: a session was saved"
            );
        }

        // An agent with only the ccc tools runs no box, so needs no bwrap.
        std::fs::write(
            f.path("agents/gcoord.md"),
            "---\nname: gcoord\ndescription: Delegates\nharness: goose\ngoose_model: qwen3\ngoose_tools: ccc\ngoose_agents: explorer\n---\ndelegate\n",
        )
        .unwrap();
        f.cfg.goose.bwrap = missing.join("bwrap").display().to_string();
        let (code, out) = f.run_goose("gcoord", &f.work(), &GooseFlags::default());
        assert_eq!(code.unwrap(), 0, "{out}");
        let argv = f.goose_argv();
        let extensions: Vec<&String> = argv
            .iter()
            .zip(argv.iter().skip(1))
            .filter(|(flag, _)| *flag == "--with-extension")
            .map(|(_, value)| value)
            .collect();
        assert_eq!(extensions.len(), 1, "{argv:?}");
        assert!(extensions[0].starts_with("ccc:"), "{argv:?}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_resume_whose_recorded_root_is_gone_or_a_symlink_fails_the_preflight() {
        let mut f = Fixture::new("goose-preflight-resume");
        f.goose();
        std::fs::create_dir_all(f.path("extra")).unwrap();
        std::fs::create_dir_all(f.path("elsewhere")).unwrap();
        let flags = GooseFlags {
            allow_read: vec![f.path("extra")],
            ..GooseFlags::default()
        };
        let (code, out) = f.run_goose("gexplorer", &f.work(), &flags);
        assert_eq!(code.unwrap(), 0, "{out}");
        let id = flag_value(&f.goose_argv(), "-n").unwrap().to_string();
        std::fs::remove_file(f.path("calls/argv")).unwrap();
        let extra = f.canonical("extra");

        std::fs::remove_dir(&extra).unwrap();
        let err = f.resume_goose(&id, false).0.unwrap_err();
        assert!(
            format!("{err:#}").contains(&extra.display().to_string()),
            "{err:#}"
        );
        assert!(!f.path("calls/argv").exists(), "goose was spawned");

        std::os::unix::fs::symlink(f.path("elsewhere"), &extra).unwrap();
        let err = f.resume_goose(&id, false).0.unwrap_err();
        let text = format!("{err:#}");
        assert!(
            text.contains(&extra.display().to_string()) && text.contains("is a symlink"),
            "{text}"
        );
        assert!(!f.path("calls/argv").exists(), "goose was spawned");
    }

    #[cfg(unix)]
    #[test]
    fn an_uncreatable_goose_root_fails_before_any_session_or_spawn() {
        let mut f = Fixture::new("goose-bad-root");
        f.goose();
        std::fs::write(f.path("file"), "").unwrap();
        let root = f.path("file/goose");
        f.cfg.goose.root = Some(root.display().to_string());
        let err = f
            .run_goose("gexplorer", &f.work(), &GooseFlags::default())
            .0
            .unwrap_err();
        assert!(
            format!("{err:#}").contains(&root.display().to_string()),
            "{err:#}"
        );
        assert!(!f.path("calls/argv").exists(), "goose was spawned");
        let sessions = f.sessions_dir();
        assert!(
            !sessions.exists() || std::fs::read_dir(&sessions).unwrap().next().is_none(),
            "a session was saved"
        );
        assert!(
            !latest_records(&f.env)
                .iter()
                .any(|r| matches!(r, audit::Record::Spawn(_))),
            "a spawn record was written"
        );
    }

    const GOOSE_DELEGATOR: &str = "---\nname: gdelegator\ndescription: Writes and delegates\nharness: goose\ngoose_model: qwen3\ngoose_tools: write, ccc\ngoose_agents: explorer\n---\ndelegate\n";

    /// A Goose hook payload the fake hands one plugin command.
    enum GooseHook<'a> {
        Start(&'a str),
        Stop(&'a str, Option<&'a str>),
    }

    impl Fixture {
        /// A stand-in `goose session`: records its argv, cwd, environment and
        /// the session files present when it starts, then runs the plugin's
        /// command for each hook under `sh -c`, with the payload on stdin and
        /// the real `ccc` as the hook binary.
        #[cfg(target_os = "linux")]
        fn goose_session(&mut self, hooks: &[GooseHook]) {
            self.goose();
            std::fs::write(self.path("agents/gdelegator.md"), GOOSE_DELEGATOR).unwrap();
            let out = self.path("calls");
            let mut script = format!(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > {out}/argv\npwd > {out}/cwd\nenv > {out}/env\nls {sessions} > {out}/sessions 2>/dev/null\nexport XDG_STATE_HOME={state}\nexport {hook}={ccc}\n",
                out = out.display(),
                sessions = shell_literal(&self.sessions_dir().display().to_string()),
                state = shell_literal(&self.env["XDG_STATE_HOME"]),
                hook = crate::harness::goose::HOOK_VAR,
                ccc = shell_literal(&crate::testutil::ccc_binary().display().to_string()),
            );
            for hook in hooks {
                let (kind, payload) = match hook {
                    GooseHook::Start(native) => (
                        "goose-session-start",
                        serde_json::json!({ "session_id": native }),
                    ),
                    GooseHook::Stop(native, message) => (
                        "goose-stop",
                        serde_json::json!({ "session_id": native, "last_assistant_message": message }),
                    ),
                };
                script.push_str(&format!(
                    "printf '%s' {payload} | sh -c {command}\n",
                    payload = shell_literal(&payload.to_string()),
                    command = shell_literal(&crate::harness::goose::hook_command(kind)),
                ));
            }
            self.write_executable(&self.path("bin/fake-goose"), &script);
        }

        #[cfg(target_os = "linux")]
        fn launch_goose(&self, target: LaunchTarget) -> Result<EndPayload> {
            let _lock = supervise::TEST_IO_TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            start_launch(self);
            let usage = || UsageSource::Unavailable("not consulted".into());
            let completion =
                launch_with_ctx(&self.ctx(&usage), target, None, vec![]).map_err(|e| e.error)?;
            let end = completion.end.clone();
            record_completion(self, completion);
            Ok(end)
        }
    }

    fn fresh_goose<'a>(
        agent: &'a str,
        cwd: &'a str,
        allow: [&'a [PathBuf]; 3],
    ) -> LaunchTarget<'a> {
        LaunchTarget::Fresh {
            agent,
            cwd: Some(cwd),
            cli_harness: None,
            allow_read: allow[0],
            allow_write: allow[1],
            allow_exec: allow[2],
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_goose_launch_records_its_session_policy_and_mapped_turn() {
        let mut f = Fixture::new("goose-launch");
        f.goose_session(&[
            GooseHook::Start("20261101_1"),
            GooseHook::Stop("20261101_1", Some("launch answer")),
        ]);
        let work = f.work();
        let end = f
            .launch_goose(fresh_goose("gexplorer", &work, [&[], &[], &[]]))
            .unwrap();

        let argv = f.goose_argv();
        assert_eq!(argv[0], "session");
        assert!(argv.iter().any(|a| a == "--no-profile"), "{argv:?}");
        assert!(!argv
            .iter()
            .any(|a| a == "-q" || a == "-i" || a == "--resume"));
        let id = flag_value(&argv, "-n").unwrap().to_string();
        Uuid::parse_str(&id).expect("the preset id is a UUID");
        assert!(flag_value(&argv, "--with-extension")
            .unwrap()
            .starts_with("box:"));
        let root = f.cfg.goose.root_in(&f.env);
        let work_dir = std::fs::canonicalize(crate::harness::goose::work_dir(&root)).unwrap();
        assert_eq!(f.call("cwd").trim(), work_dir.display().to_string());
        let env = f.call("env");
        assert!(
            env.lines().any(|l| l == format!("CCC_GOOSE_SESSION={id}")),
            "{env}"
        );
        assert!(
            env.lines().any(|l| l.starts_with("CCC_GOOSE_HOOK=/")),
            "{env}"
        );
        assert!(env.lines().any(|l| l == "GOOSE_MODE=auto"), "{env}");
        // The sidecar was saved before Goose started.
        assert!(
            f.call("sessions")
                .lines()
                .any(|l| l == format!("{id}.json")),
            "{}",
            f.call("sessions")
        );

        let record = session::load_from(&f.sessions_dir(), &id).unwrap();
        assert!(record.opened_at_terminal);
        assert_eq!(
            record.goose_policy.as_ref().unwrap().read_roots,
            vec![f.canonical("work")]
        );
        assert_eq!(spawn_record(&f.env).goose_policy, record.goose_policy);

        assert_eq!(end.session.as_deref(), Some(id.as_str()));
        assert_eq!(end.result.as_deref(), Some("launch answer"));
        assert_eq!(end.turns, Some(1));
        let records = launch_records(&f);
        let mapped = records
            .iter()
            .find_map(|r| match r {
                audit::Record::Session(s) => Some(s),
                _ => None,
            })
            .expect("the SessionStart hook recorded the mapping");
        assert_eq!(mapped.session, id);
        assert_eq!(mapped.native.as_deref(), Some("20261101_1"));
        let turn = records
            .iter()
            .find_map(|r| match r {
                audit::Record::Turn(t) => Some(t),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            (turn.harness, turn.session.as_str()),
            (Harness::Goose, id.as_str())
        );
        let ended = records
            .iter()
            .find_map(|r| match r {
                audit::Record::End(e) => Some(&e.payload),
                _ => None,
            })
            .unwrap();
        assert_eq!(ended.session.as_deref(), Some(id.as_str()));
        assert_eq!(ended.result.as_deref(), Some("launch answer"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_goose_launch_records_only_its_own_session() {
        type Case<'a> = (&'a str, Vec<GooseHook<'a>>, Option<&'a str>, usize);
        let cases: [Case; 3] = [
            (
                "new-after-mapped",
                vec![
                    GooseHook::Start("20261101_1"),
                    GooseHook::Stop("20261101_1", Some("mine")),
                    GooseHook::Start("20261101_2"),
                    GooseHook::Stop("20261101_2", Some("from /new")),
                ],
                Some("mine"),
                2,
            ),
            (
                "unmapped-only",
                vec![GooseHook::Stop("20261101_2", Some("from /new"))],
                None,
                1,
            ),
            ("no-hooks", vec![], None, 0),
        ];
        for (label, hooks, result, turns) in cases {
            let mut f = Fixture::new(&format!("goose-launch-{label}"));
            f.goose_session(&hooks);
            let work = f.work();
            let end = f
                .launch_goose(fresh_goose("gexplorer", &work, [&[], &[], &[]]))
                .unwrap();
            let id = flag_value(&f.goose_argv(), "-n").unwrap().to_string();
            assert_eq!(end.session.as_deref(), Some(id.as_str()), "{label}");
            assert_eq!(end.result.as_deref(), result, "{label}");
            assert_eq!(end.turns, Some(turns), "{label}");
            let saved = std::fs::read_dir(f.sessions_dir())
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .filter(|n| n.ends_with(".json"))
                .collect::<Vec<_>>();
            assert_eq!(saved, [format!("{id}.json")], "{label}");
            let mapped = launch_records(&f)
                .iter()
                .filter(|r| matches!(r, audit::Record::Session(_)))
                .count();
            assert_eq!(mapped, usize::from(label == "new-after-mapped"), "{label}");
        }
    }

    #[test]
    fn a_goose_launch_ending_in_a_new_session_notes_its_own_id() {
        let turn = |session: &str, result: &str| audit::Turn {
            ts: jiff::Timestamp::now(),
            inv: "inv".into(),
            harness: Harness::Goose,
            session: session.into(),
            turn: None,
            result: Some(result.into()),
            transcript: None,
        };
        let own = "own-id".to_string();
        let (end, note) = goose_launch_end(
            Some(own.clone()),
            &[turn("own-id", "mine"), turn("20261101_2", "from /new")],
        );
        assert_eq!(end.result.as_deref(), Some("mine"));
        let note = note.expect("a note after a /new turn");
        assert!(note.contains("(20261101_2)"), "{note}");
        assert!(
            note.contains("ccc launch --resume own-id reopens"),
            "{note}"
        );
        for turns in [vec![turn("own-id", "mine")], vec![]] {
            let (_, note) = goose_launch_end(Some(own.clone()), &turns);
            assert_eq!(note, None, "{turns:?}");
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_goose_launch_resume_reopens_a_headless_session_marked_as_at_a_terminal() {
        let mut f = Fixture::new("goose-launch-resume");
        f.goose();
        let (code, out) = f.run_goose("gexplorer", &f.work(), &GooseFlags::default());
        assert_eq!(code.unwrap(), 0, "{out}");
        let id = flag_value(&f.goose_argv(), "-n").unwrap().to_string();
        let before = session::load_from(&f.sessions_dir(), &id).unwrap();
        assert!(!before.opened_at_terminal);

        // A launch that fails its box preflight never reached a terminal.
        std::fs::remove_file(f.path("calls/argv")).unwrap();
        let saved = f.cfg.clone();
        f.cfg.goose.bwrap = f.path("missing/bwrap").display().to_string();
        let err = f
            .launch_goose(LaunchTarget::Resume {
                id: &id,
                fork: false,
            })
            .unwrap_err();
        f.cfg = saved;
        assert!(format!("{err:#}").contains("bubblewrap"), "{err:#}");
        assert!(!f.path("calls/argv").exists(), "goose was spawned");
        assert!(
            !session::load_from(&f.sessions_dir(), &id)
                .unwrap()
                .opened_at_terminal
        );

        f.goose_session(&[
            GooseHook::Start("20261101_4"),
            GooseHook::Stop("20261101_4", Some("resumed answer")),
        ]);
        let end = f
            .launch_goose(LaunchTarget::Resume {
                id: &id,
                fork: false,
            })
            .unwrap();
        let argv = f.goose_argv();
        assert_eq!(&argv[..2], ["session", "--resume"]);
        assert_eq!(flag_value(&argv, "-n"), Some(id.as_str()));
        assert!(!argv
            .iter()
            .any(|a| a == "--with-extension" || a == "--no-profile"));
        assert!(f
            .call("env")
            .lines()
            .any(|l| l == format!("CCC_GOOSE_SESSION={id}")));
        assert_eq!(end.session.as_deref(), Some(id.as_str()));
        assert_eq!(end.result.as_deref(), Some("resumed answer"));
        let after = session::load_from(&f.sessions_dir(), &id).unwrap();
        assert!(after.opened_at_terminal);
        assert_eq!(after.goose_policy, before.goose_policy);

        // A later headless resume keeps the mark.
        f.goose();
        let (code, out) = f.resume_goose(&id, false);
        assert_eq!(code.unwrap(), 0, "{out}");
        assert!(
            session::load_from(&f.sessions_dir(), &id)
                .unwrap()
                .opened_at_terminal
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn goose_launch_grants_and_refusals() {
        let mut f = Fixture::new("goose-launch-grants");
        f.goose_session(&[]);
        // Own git tops, so a host /tmp/.git does not join the surfaces.
        for dir in ["exec/.git", "write", "work/.git"] {
            std::fs::create_dir_all(f.path(dir)).unwrap();
        }
        let work = f.work();
        let write = [f.path("write")];
        let exec = [f.path("exec")];
        let end = f
            .launch_goose(fresh_goose("gdelegator", &work, [&[], &write, &exec]))
            .unwrap();
        let id = end.session.unwrap();
        let policy = session::load_from(&f.sessions_dir(), &id)
            .unwrap()
            .goose_policy
            .unwrap();
        assert_eq!(
            policy.write_roots,
            vec![f.canonical("work"), f.canonical("write")]
        );
        assert_eq!(policy.exec_roots, vec![f.canonical("exec")]);
        assert!(
            f.call("argv").contains("ccc:"),
            "the ccc extension is named"
        );
        std::fs::remove_file(f.path("calls/argv")).unwrap();

        // An overlapping write and exec grant.
        std::fs::create_dir_all(f.path("exec/sub")).unwrap();
        let inside = [f.path("exec/sub")];
        let err = f
            .launch_goose(fresh_goose("gdelegator", &work, [&[], &inside, &exec]))
            .unwrap_err();
        assert!(
            format!("{err:#}").contains(&f.canonical("exec/sub").display().to_string()),
            "{err:#}"
        );
        assert!(!f.path("calls/argv").exists());

        // --allow-* on a Claude launch.
        let read = [f.path("write")];
        let err = f
            .launch_goose(fresh_goose("explorer", &work, [&read, &[], &[]]))
            .unwrap_err();
        assert!(
            format!("{err:#}").contains("applies only to the goose harness"),
            "{err:#}"
        );

        // A failing preflight.
        let saved = f.cfg.clone();
        f.cfg.goose.bwrap = f.path("missing/bwrap").display().to_string();
        let err = f
            .launch_goose(fresh_goose("gexplorer", &work, [&[], &[], &[]]))
            .unwrap_err();
        f.cfg = saved;
        assert!(format!("{err:#}").contains("bubblewrap"), "{err:#}");
        assert!(!f.path("calls/argv").exists(), "goose was spawned");
        let saved = std::fs::read_dir(f.sessions_dir())
            .unwrap()
            .filter(|e| e.as_ref().unwrap().path().extension() == Some("json".as_ref()))
            .count();
        assert_eq!(saved, 1);
    }

    #[cfg(unix)]
    #[test]
    fn launch_resume_fork_on_goose_is_refused_and_the_record_left_alone() {
        let mut f = Fixture::new("goose-launch-fork");
        f.goose();
        let (code, _) = f.run_goose("gexplorer", &f.work(), &GooseFlags::default());
        assert_eq!(code.unwrap(), 0);
        let id = flag_value(&f.goose_argv(), "-n").unwrap().to_string();
        let path = session::path_in(&f.sessions_dir(), &id).unwrap();
        let before = std::fs::read(&path).unwrap();
        std::fs::remove_file(f.path("calls/argv")).unwrap();

        let err = launch_in(
            LaunchTarget::Resume {
                id: &id,
                fork: true,
            },
            None,
            vec![],
            &f.env,
            || Ok(f.cfg.clone()),
            &mut Vec::new(),
        )
        .unwrap_err();
        assert!(
            format!("{err:#}").contains("Goose has no headless fork"),
            "{err:#}"
        );
        assert!(!f.path("calls/argv").exists(), "goose was spawned");
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }

    const WITHHELD_NOTE: &str = "ANTHROPIC_API_KEY removed from the child's environment; an API key account is configured and a {} turn is not charged to it";

    /// The interactive invocation `launch_invocation` builds for `harness`:
    /// the Codex or Goose explorer, or the Claude one on a token profile.
    fn launch_built(f: &mut Fixture, harness: Harness) -> Invocation {
        let (agent, cli_harness) = match harness {
            Harness::Codex => {
                f.cfg.agents.insert(
                    "explorer".into(),
                    crate::config::AgentOverrides {
                        codex_model: Some("gpt-5.6-terra".into()),
                        ..Default::default()
                    },
                );
                ("explorer", Some(Harness::Codex))
            }
            Harness::Goose => {
                f.goose();
                ("gexplorer", None)
            }
            Harness::Claude => {
                f.token("team03");
                f.cfg.claude.profiles = vec!["team03".into()];
                ("explorer", None)
            }
        };
        let usage = || report(vec![snapshot(Timestamp::now(), "team03", 0.0, 10.0)]);
        let ctx = f.ctx(&usage);
        let prepared = prepare(
            &ctx,
            agent,
            Some(&f.work()),
            None,
            cli_harness,
            &GooseFlags::default(),
        )
        .unwrap();
        let LaunchPrepared { inv, r, .. } = launch_invocation(&ctx, prepared, vec![]).unwrap();
        assert_eq!(r.harness, harness);
        inv
    }

    fn removes_api_key(inv: &Invocation) -> bool {
        inv.env_remove.iter().any(|v| v == "ANTHROPIC_API_KEY")
    }

    #[test]
    fn codex_and_goose_children_lose_the_api_key_only_when_a_key_is_configured() {
        for harness in [Harness::Codex, Harness::Goose] {
            for (configured, inherited, noted) in [
                (true, Some("sk-ant-shell"), true),
                (true, None, false),
                (true, Some("  "), false),
                (false, Some("sk-ant-shell"), false),
            ] {
                let label = format!("withhold-{harness}-{configured}-{inherited:?}");
                let mut f = Fixture::new("withhold-key");
                if configured {
                    api_key(&mut f, "team");
                }
                if let Some(value) = inherited {
                    f.env.insert("ANTHROPIC_API_KEY".into(), value.into());
                }
                let built = launch_built(&mut f, harness);
                assert!(!removes_api_key(&built), "{label}: {built:?}");
                let mut inv = built.clone();
                let note = withhold_api_key(&f.cfg, &f.env, harness, &mut inv);
                if configured {
                    assert!(removes_api_key(&inv), "{label}: {inv:?}");
                    assert_eq!(inv.env_set, built.env_set, "{label}");
                } else {
                    assert_eq!(inv, built, "{label}: nothing changes without a key");
                }
                let expected = noted.then(|| WITHHELD_NOTE.replace("{}", &harness.to_string()));
                assert_eq!(note, expected, "{label}");
            }
        }
    }

    #[test]
    fn a_goose_env_api_key_is_kept_with_a_key_configured() {
        let mut f = Fixture::new("withhold-goose-env");
        api_key(&mut f, "team");
        f.env
            .insert("ANTHROPIC_API_KEY".into(), "sk-ant-shell".into());
        f.cfg
            .goose
            .env
            .insert("ANTHROPIC_API_KEY".into(), "sk-ant-goose".into());
        let mut inv = launch_built(&mut f, Harness::Goose);
        let built = inv.clone();
        assert_eq!(
            withhold_api_key(&f.cfg, &f.env, Harness::Goose, &mut inv),
            None
        );
        assert_eq!(inv, built);
        assert!(inv
            .env_set
            .iter()
            .any(|(k, v)| k == "ANTHROPIC_API_KEY" && v == "sk-ant-goose"));
        assert!(!removes_api_key(&inv), "{inv:?}");
    }

    #[test]
    fn a_claude_invocation_is_left_to_its_account() {
        let mut f = Fixture::new("withhold-claude");
        api_key(&mut f, "team");
        f.env
            .insert("ANTHROPIC_API_KEY".into(), "sk-ant-shell".into());
        let mut inv = launch_built(&mut f, Harness::Claude);
        let built = inv.clone();
        assert_eq!(
            withhold_api_key(&f.cfg, &f.env, Harness::Claude, &mut inv),
            None
        );
        assert_eq!(inv, built);
    }

    #[cfg(unix)]
    #[test]
    fn a_codex_turn_with_a_key_configured_records_the_withheld_key_once() {
        let mut f = Fixture::new("withhold-codex-drive");
        api_key(&mut f, "team");
        f.env
            .insert("ANTHROPIC_API_KEY".into(), "sk-ant-shell".into());
        std::fs::create_dir_all(f.path("calls")).unwrap();
        f.codex_script(&format!(
            "env > '{}'\nprintf '%s\\n' '{{\"type\":\"thread.started\",\"thread_id\":\"t-withheld\"}}' '{{\"type\":\"item.completed\",\"item\":{{\"type\":\"agent_message\",\"text\":\"answer\"}}}}' '{{\"type\":\"turn.completed\",\"usage\":{{}}}}'",
            f.path("calls/env").display()
        ));
        let usage = || UsageSource::Unavailable("not consulted".into());
        let ctx = f.ctx(&usage);
        let mut resolved = resolved();
        resolved.harness = Harness::Codex;
        resolved.model = "gpt-5.6-luna".into();
        let common = Common {
            cwd: f.path("work"),
            extra_args: vec![],
            mode: run_mode(),
            preset_session_id: None,
        };
        let diagnostics = Mutex::new(Vec::new());
        let record = record_of(&resolved, &common.cwd, None, ctx.now);
        let result = drive_with_meta_report(
            &ctx,
            &resolved,
            &common,
            None,
            vec![],
            &|message| diagnostics.lock().unwrap().push(message),
            &record,
        )
        .unwrap();
        assert!(result.result.is_ok());
        let note = WITHHELD_NOTE.replace("{}", "codex");
        assert_eq!(*diagnostics.lock().unwrap(), vec![format!("note: {note}")]);
        let records = audit::read_trace(
            &Recorder::dir_in(&f.env),
            ctx.audit.lineage().trace_id.as_str(),
        )
        .unwrap();
        let notes: Vec<Vec<String>> = records
            .into_iter()
            .filter_map(|r| match r {
                audit::Record::Spawn(s) => Some(s.payload.selection_notes),
                _ => None,
            })
            .collect();
        assert_eq!(notes, vec![vec![note]]);
        assert!(
            !f.call("env")
                .lines()
                .any(|l| l.starts_with("ANTHROPIC_API_KEY=")),
            "the Codex child inherited the key"
        );
    }
}
