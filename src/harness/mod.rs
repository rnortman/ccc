//! Argv/env construction and output parsing for the two harnesses.
//!
//! Nothing here spawns a process: `build` returns an [`Invocation`] describing
//! one, and `parse` turns captured output into an [`Outcome`].

pub mod claude;
pub mod codex;

use std::path::PathBuf;
use std::process::{Command, Stdio};

use anyhow::{bail, Result};

use crate::config::{Effort, EnvSource, Harness};
use crate::resolve::Resolved;

/// Exit code for a harness that ran and reported a failed turn.
pub const EXIT_HARNESS_FAILED: i32 = 1;
/// Exit code for a harness that produced no usable structured result.
pub const EXIT_NO_RESULT: i32 = 3;
/// Exit code for a turn cut off by a usage limit that no other profile could
/// take over.
pub const EXIT_LIMITED: i32 = 4;

/// Ceiling for a single rendered command-line argument; Linux
/// `MAX_ARG_STRLEN` is 128 KiB.
pub const MAX_ARG_BYTES: usize = 100 * 1024;

/// Rejects a rendered argument that would not survive `execve`.
pub fn check_arg_len(agent: &str, what: &str, rendered: &str) -> Result<()> {
    if rendered.len() > MAX_ARG_BYTES {
        bail!(
            "agent {agent:?} renders to {} bytes of {what}, over the {MAX_ARG_BYTES}-byte command-line limit",
            rendered.len()
        );
    }
    Ok(())
}

/// Which kind of invocation is being built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mode {
    pub io: Io,
    pub origin: Origin,
}

/// Who holds the conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Io {
    /// A single turn; the prompt is written to stdin and stdout is parsed.
    OneShot { prompt: String },
    /// A human at the terminal; `self_exe` is what the harness's callback runs.
    Interactive { self_exe: PathBuf },
}

/// Where the conversation starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Origin {
    Fresh,
    /// Continue `id`; with `fork`, as a new session that leaves `id` untouched.
    Resume {
        id: String,
        fork: bool,
    },
}

impl Mode {
    /// `true` when a human holds the conversation.
    pub fn is_launch(&self) -> bool {
        matches!(self.io, Io::Interactive { .. })
    }

    /// What is written to the child's stdin and then closed.
    pub fn stdin(&self) -> Option<String> {
        match &self.io {
            Io::OneShot { prompt } => Some(prompt.clone()),
            Io::Interactive { .. } => None,
        }
    }

    /// The session whose transcript the harness is handed: the resumed id,
    /// fork or not. Drivers build `--resume <id>` / `resume <id>` /
    /// `fork <id>` from it; audit `start`/`spawn` use it as the "resumed" session.
    pub fn starts_from(&self) -> Option<&str> {
        match &self.origin {
            Origin::Resume { id, .. } => Some(id.as_str()),
            Origin::Fresh => None,
        }
    }

    /// The session this invocation *is*, when known before the harness says
    /// so: the resumed id on a non-fork resume; `None` on a fresh start and on
    /// a fork (a fork's id does not exist yet).
    pub fn continues(&self) -> Option<&str> {
        match &self.origin {
            Origin::Resume { id, fork: false } => Some(id.as_str()),
            _ => None,
        }
    }

    pub fn is_fork(&self) -> bool {
        matches!(self.origin, Origin::Resume { fork: true, .. })
    }
}

/// `Mode` values tests build most often; shared so the driver and exec
/// test modules do not each carry a copy.
#[cfg(test)]
pub(crate) mod modes {
    use super::{Io, Mode, Origin};

    pub(crate) fn one_shot(origin: Origin) -> Mode {
        Mode {
            io: Io::OneShot {
                prompt: "go".into(),
            },
            origin,
        }
    }

    pub(crate) fn interactive(self_exe: &str, origin: Origin) -> Mode {
        Mode {
            io: Io::Interactive {
                self_exe: self_exe.into(),
            },
            origin,
        }
    }

    pub(crate) fn resumed(id: &str, fork: bool) -> Origin {
        Origin::Resume {
            id: id.into(),
            fork,
        }
    }

    pub(crate) fn run_mode() -> Mode {
        one_shot(Origin::Fresh)
    }
}

/// The parts of an invocation that are the same on every harness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Common {
    pub cwd: PathBuf,
    pub extra_args: Vec<String>,
    pub mode: Mode,
    pub preset_session_id: Option<String>,
}

/// Everything a finished child process produced.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Captured {
    pub stdout: String,
    pub stderr: String,
    /// Contents of the file the caller asked the child to write its final
    /// message to, if it asked for one.
    pub output_file: Option<String>,
    pub exit_code: Option<i32>,
}

/// Abstraction over a single harness (Claude or Codex).
pub trait Driver: Sync {
    fn build(&self, r: &Resolved, c: &Common) -> Result<Invocation>;

    fn parse(&self, captured: &Captured) -> Result<Outcome, HarnessError>;

    /// Extracts a session id from a single line of harness stdout, if present.
    fn early_session_id(&self, _line: &str) -> Option<String> {
        None
    }
}

/// What was being attempted, for the diagnostic on an unusable result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunContext {
    pub agent: Option<String>,
    pub harness: Harness,
    pub model: String,
    pub effort: Option<Effort>,
    pub profile: Option<String>,
    pub cwd: PathBuf,
    pub argv: Vec<String>,
}

impl RunContext {
    pub fn new(r: &Resolved, profile: Option<&str>, inv: &Invocation) -> RunContext {
        let mut argv = vec![inv.program.clone()];
        argv.extend(inv.args.iter().map(|a| elide(a)));
        RunContext {
            agent: r.agent.clone(),
            harness: r.harness,
            model: r.model.clone(),
            effort: r.effort,
            profile: profile.map(String::from),
            cwd: inv.cwd.clone(),
            argv,
        }
    }
}

impl std::fmt::Display for RunContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "agent={} harness={} model={} effort={} profile={} cwd={} argv={:?}",
            self.agent.as_deref().unwrap_or("-"),
            self.harness,
            self.model,
            self.effort.map(|e| e.as_str()).unwrap_or("-"),
            self.profile.as_deref().unwrap_or("-"),
            self.cwd.display(),
            self.argv,
        )
    }
}

fn elide(arg: &str) -> String {
    const KEEP: usize = 120;
    if arg.len() <= KEEP {
        return arg.to_string();
    }
    let head: String = arg.chars().take(KEEP).collect();
    format!("{head}… <{} bytes>", arg.len())
}

/// Writes one diagnostic line to stderr, so stdout stays the subagent's message.
pub fn diag(message: impl std::fmt::Display) {
    eprintln!("ccc: {message}");
}

/// Emits the non-fatal diagnostics an [`Outcome`] collected.
pub fn report_warnings(warnings: &[String]) {
    for w in warnings {
        diag(format_args!("warning: {w}"));
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Invocation {
    pub program: String,
    pub args: Vec<String>,
    pub env_set: Vec<(String, String)>,
    pub env_remove: Vec<String>,
    pub cwd: PathBuf,
    /// The prompt, written to the child's stdin and then closed; `None` for
    /// `launch`, which inherits the terminal.
    pub stdin: Option<String>,
}

/// Environment values carry account credentials, so they are never printed.
impl std::fmt::Debug for Invocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let env_set: Vec<(&str, &str)> = self
            .env_set
            .iter()
            .map(|(k, _)| (k.as_str(), "***"))
            .collect();
        f.debug_struct("Invocation")
            .field("program", &self.program)
            .field("args", &self.args)
            .field("env_set", &env_set)
            .field("env_remove", &self.env_remove)
            .field("cwd", &self.cwd)
            .field("stdin", &self.stdin)
            .finish()
    }
}

impl Invocation {
    /// A `Command` ready to spawn. Stdin is piped when a prompt is present.
    pub fn command(&self) -> Command {
        let mut cmd = Command::new(&self.program);
        cmd.args(&self.args).current_dir(&self.cwd);
        for (k, v) in &self.env_set {
            cmd.env(k, v);
        }
        for k in &self.env_remove {
            cmd.env_remove(k);
        }
        if self.stdin.is_some() {
            cmd.stdin(Stdio::piped());
        }
        cmd
    }

    /// The value following `flag`, if `flag` is present with one.
    #[cfg(test)]
    pub fn value_of(&self, flag: &str) -> Option<&str> {
        let i = self.args.iter().position(|a| a == flag)?;
        self.args.get(i + 1).map(String::as_str)
    }

    /// The value of a `-c <key>=<value>` config override.
    #[cfg(test)]
    pub fn config_override(&self, key: &str) -> Option<&str> {
        let prefix = format!("{key}=");
        self.args
            .iter()
            .enumerate()
            .find(|(i, a)| a.starts_with(&prefix) && *i > 0 && self.args[i - 1] == "-c")
            .map(|(_, a)| &a[prefix.len()..])
    }

    /// Whether `arg` appears anywhere in the argv.
    #[cfg(test)]
    pub fn has_arg(&self, arg: &str) -> bool {
        self.args.iter().any(|a| a == arg)
    }
}

/// A successful turn.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub session_id: String,
    pub result: String,
    /// Harness-native usage and cost fields, passed through to `--json`.
    pub native: serde_json::Value,
    /// Non-fatal diagnostics for stderr.
    pub warnings: Vec<String>,
}

/// Why a turn did not produce an [`Outcome`].
#[derive(Debug, Clone, PartialEq)]
pub enum HarnessError {
    /// The harness ran and reported failure.
    Reported {
        session_id: Option<String>,
        message: String,
        native: serde_json::Value,
        /// Non-fatal diagnostics the harness streamed before it failed; they
        /// usually carry the explanation the terminal message lacks.
        warnings: Vec<String>,
    },
    /// The harness ran and reported that the account hit a usage limit.
    Limited {
        session_id: Option<String>,
        message: String,
        native: serde_json::Value,
        warnings: Vec<String>,
    },
    /// The harness crashed, or its output held no final message.
    Unusable {
        session_id: Option<String>,
        message: String,
    },
}

impl HarnessError {
    pub fn exit_code(&self) -> i32 {
        match self {
            HarnessError::Reported { .. } => EXIT_HARNESS_FAILED,
            HarnessError::Limited { .. } => EXIT_LIMITED,
            HarnessError::Unusable { .. } => EXIT_NO_RESULT,
        }
    }

    pub fn message(&self) -> &str {
        match self {
            HarnessError::Reported { message, .. }
            | HarnessError::Limited { message, .. }
            | HarnessError::Unusable { message, .. } => message,
        }
    }

    /// The diagnostics collected before the failure, if any.
    pub fn warnings(&self) -> &[String] {
        match self {
            HarnessError::Reported { warnings, .. } | HarnessError::Limited { warnings, .. } => {
                warnings
            }
            HarnessError::Unusable { .. } => &[],
        }
    }

    /// The harness-assigned session id, if known.
    pub fn session_id(&self) -> Option<&str> {
        match self {
            HarnessError::Reported { session_id, .. }
            | HarnessError::Limited { session_id, .. }
            | HarnessError::Unusable { session_id, .. } => session_id.as_deref(),
        }
    }

    pub fn unusable(message: impl Into<String>) -> HarnessError {
        HarnessError::Unusable {
            session_id: None,
            message: message.into(),
        }
    }

    /// Attaches a session id to an `Unusable` error. No-op if one is already set.
    pub fn with_session_id(mut self, session_id: Option<String>) -> HarnessError {
        if let HarnessError::Unusable {
            session_id: current,
            ..
        } = &mut self
        {
            if current.is_none() {
                *current = session_id;
            }
        }
        self
    }

    /// The message and native fields of a turn the harness reported failed,
    /// limited or not; `None` for an unusable one.
    pub fn reported(&self) -> Option<(&str, &serde_json::Value)> {
        match self {
            HarnessError::Reported {
                message, native, ..
            }
            | HarnessError::Limited {
                message, native, ..
            } => Some((message, native)),
            HarnessError::Unusable { .. } => None,
        }
    }

    /// Mutable access to the warnings of a reported or limited failure.
    pub fn warnings_mut(&mut self) -> Option<&mut Vec<String>> {
        match self {
            HarnessError::Reported { warnings, .. } | HarnessError::Limited { warnings, .. } => {
                Some(warnings)
            }
            HarnessError::Unusable { .. } => None,
        }
    }

    /// Adds the run context to an [`Unusable`] result. Leaves [`Reported`]
    /// and [`Limited`] failures unchanged.
    pub fn annotate(self, ctx: &RunContext) -> HarnessError {
        match self {
            HarnessError::Unusable {
                session_id,
                message,
            } => HarnessError::Unusable {
                session_id,
                message: format!("{message}; {ctx}"),
            },
            other => other,
        }
    }
}

impl std::fmt::Display for HarnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

/// The harness binary: `$var` if set, else `default`.
fn binary_in(env: &dyn EnvSource, var: &str, default: &str) -> String {
    match env.get(var) {
        Some(v) if !v.trim().is_empty() => v,
        _ => default.to_string(),
    }
}

/// The last few lines of `text`, for error messages.
fn tail(text: &str, lines: usize) -> String {
    let all: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let start = all.len().saturating_sub(lines);
    all[start..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inv() -> Invocation {
        Invocation {
            program: "codex".into(),
            args: [
                "exec",
                "-m",
                "gpt-5.6-luna",
                "-c",
                "model_reasoning_effort=\"low\"",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            env_set: vec![],
            env_remove: vec![],
            cwd: PathBuf::from("/tmp"),
            stdin: Some("hi".into()),
        }
    }

    #[test]
    fn debug_never_prints_an_environment_value() {
        let mut i = inv();
        i.env_set = vec![("CLAUDE_CODE_OAUTH_TOKEN".into(), "sk-ant-oat-secret".into())];
        let text = format!("{i:?}");
        assert!(!text.contains("sk-ant-oat-secret"), "{text}");
        assert!(text.contains("CLAUDE_CODE_OAUTH_TOKEN"));
        assert!(text.contains("***"));
    }

    #[test]
    fn unusable_errors_carry_what_was_being_run() {
        use crate::agents::Sandbox;
        let r = crate::resolve::Resolved {
            agent: Some("explorer".into()),
            description: "d".into(),
            harness: Harness::Codex,
            model: "gpt-5.6-luna".into(),
            effort: Some(Effort::Low),
            tools: None,
            sandbox: Sandbox::ReadOnly,
            body: "b".into(),
        };
        let ctx = RunContext::new(&r, Some("team03"), &inv());
        let annotated = HarnessError::unusable("boom").annotate(&ctx);
        let msg = annotated.message();
        for part in [
            "boom",
            "explorer",
            "codex",
            "gpt-5.6-luna",
            "team03",
            "/tmp",
        ] {
            assert!(msg.contains(part), "{msg} lacks {part}");
        }
        // A reported failure keeps the harness's own wording.
        let reported = HarnessError::Reported {
            session_id: None,
            message: "rate limit".into(),
            native: serde_json::Value::Null,
            warnings: vec![],
        };
        assert_eq!(reported.annotate(&ctx).message(), "rate limit");
    }

    #[test]
    fn long_arguments_are_elided_not_dumped() {
        let mut i = inv();
        i.args.push("x".repeat(5000));
        let r = RunContext::new(
            &crate::resolve::Resolved {
                agent: Some("a".into()),
                description: "d".into(),
                harness: Harness::Codex,
                model: "m".into(),
                effort: None,
                tools: None,
                sandbox: crate::agents::Sandbox::ReadOnly,
                body: "b".into(),
            },
            None,
            &i,
        );
        let text = r.to_string();
        assert!(text.contains("<5000 bytes>"), "{text}");
        assert!(text.len() < 1000, "{text}");
    }

    #[test]
    fn oversized_arguments_are_refused_with_one_message() {
        assert!(check_arg_len("a", "--agents JSON", "short").is_ok());
        let err = check_arg_len("a", "--agents JSON", &"x".repeat(MAX_ARG_BYTES + 1))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--agents JSON"), "{err}");
        assert!(err.contains("command-line limit"), "{err}");
    }

    #[test]
    fn mode_accessors_follow_both_axes() {
        let origins = [
            (Origin::Fresh, None, None, false),
            (
                Origin::Resume {
                    id: "t1".into(),
                    fork: false,
                },
                Some("t1"),
                Some("t1"),
                false,
            ),
            (
                Origin::Resume {
                    id: "t1".into(),
                    fork: true,
                },
                Some("t1"),
                None,
                true,
            ),
        ];
        let ios = [
            (
                Io::OneShot {
                    prompt: "go".into(),
                },
                Some("go"),
                false,
            ),
            (
                Io::Interactive {
                    self_exe: "/ccc".into(),
                },
                None,
                true,
            ),
        ];
        for (origin, starts_from, continues, is_fork) in &origins {
            for (io, stdin, is_launch) in &ios {
                let mode = Mode {
                    io: io.clone(),
                    origin: origin.clone(),
                };
                assert_eq!(mode.stdin().as_deref(), *stdin, "{mode:?}");
                assert_eq!(mode.is_launch(), *is_launch, "{mode:?}");
                assert_eq!(mode.starts_from(), *starts_from, "{mode:?}");
                assert_eq!(mode.continues(), *continues, "{mode:?}");
                assert_eq!(mode.is_fork(), *is_fork, "{mode:?}");
            }
        }
    }

    #[test]
    fn argv_accessors() {
        let i = inv();
        assert_eq!(i.value_of("-m"), Some("gpt-5.6-luna"));
        assert_eq!(i.config_override("model_reasoning_effort"), Some("\"low\""));
        assert_eq!(i.config_override("sandbox_mode"), None);
        assert!(i.has_arg("exec"));
        assert!(!i.has_arg("resume"));
    }

    #[test]
    fn error_exit_codes() {
        let reported = HarnessError::Reported {
            session_id: Some("abc".into()),
            message: "rate limit".into(),
            native: serde_json::Value::Null,
            warnings: vec!["model mismatch; using the default".into()],
        };
        assert_eq!(reported.exit_code(), EXIT_HARNESS_FAILED);
        assert_eq!(reported.session_id(), Some("abc"));
        assert_eq!(reported.warnings(), ["model mismatch; using the default"]);
        assert!(HarnessError::unusable("boom").warnings().is_empty());
        assert_eq!(HarnessError::unusable("boom").exit_code(), EXIT_NO_RESULT);
        let limited = HarnessError::Limited {
            session_id: Some("abc".into()),
            message: "You've hit your session limit".into(),
            native: serde_json::Value::Null,
            warnings: vec![],
        };
        assert_eq!(limited.exit_code(), EXIT_LIMITED);
        assert_eq!(limited.session_id(), Some("abc"));
        assert_eq!(
            limited.reported().map(|(m, _)| m),
            Some("You've hit your session limit")
        );
    }

    #[test]
    fn with_session_id_preserves_precedence_and_reported_errors() {
        let fresh = HarnessError::unusable("fresh").with_session_id(Some("early".into()));
        assert_eq!(fresh.session_id(), Some("early"));
        let parsed = HarnessError::unusable("parsed").with_session_id(Some("parsed".into()));
        assert_eq!(parsed.clone().with_session_id(Some("early".into())), parsed);
        assert_eq!(parsed.clone().with_session_id(None), parsed);
        assert_eq!(
            HarnessError::unusable("none")
                .with_session_id(None)
                .session_id(),
            None
        );
        let reported = HarnessError::Reported {
            session_id: Some("reported-session".into()),
            message: "reported message".into(),
            native: serde_json::json!({"native": true}),
            warnings: vec!["warning".into()],
        };
        assert_eq!(
            reported.clone().with_session_id(Some("early".into())),
            reported
        );
        assert_eq!(reported.clone().with_session_id(None), reported);
    }

    #[test]
    fn the_binary_override_is_used_only_when_it_names_something() {
        use std::collections::BTreeMap;
        let unset: BTreeMap<String, String> = BTreeMap::new();
        assert_eq!(binary_in(&unset, "CCC_CLAUDE_BIN", "claude"), "claude");

        let set: BTreeMap<String, String> =
            [("CCC_CLAUDE_BIN".to_string(), "/opt/claude".to_string())]
                .into_iter()
                .collect();
        assert_eq!(binary_in(&set, "CCC_CLAUDE_BIN", "claude"), "/opt/claude");

        let blank: BTreeMap<String, String> = [("CCC_CODEX_BIN".to_string(), "  ".to_string())]
            .into_iter()
            .collect();
        assert_eq!(binary_in(&blank, "CCC_CODEX_BIN", "codex"), "codex");
    }

    #[test]
    fn command_carries_the_program_args_cwd_and_environment_edits() {
        let mut i = inv();
        i.env_set = vec![("CLAUDE_CODE_OAUTH_TOKEN".into(), "sk-secret".into())];
        i.env_remove = vec!["CLAUDE_CONFIG_DIR".into()];
        let cmd = i.command();

        assert_eq!(cmd.get_program(), "codex");
        let args: Vec<&str> = cmd.get_args().map(|a| a.to_str().unwrap()).collect();
        assert_eq!(args, i.args);
        assert_eq!(cmd.get_current_dir(), Some(PathBuf::from("/tmp").as_path()));

        let envs: Vec<(String, Option<String>)> = cmd
            .get_envs()
            .map(|(k, v)| {
                (
                    k.to_str().unwrap().to_string(),
                    v.map(|v| v.to_str().unwrap().to_string()),
                )
            })
            .collect();
        assert!(
            envs.contains(&(
                "CLAUDE_CODE_OAUTH_TOKEN".to_string(),
                Some("sk-secret".to_string())
            )),
            "{envs:?}"
        );
        // A removal is recorded as an explicit `None`, not as an absent entry.
        assert!(
            envs.contains(&("CLAUDE_CONFIG_DIR".to_string(), None)),
            "{envs:?}"
        );
    }

    #[test]
    fn a_piped_prompt_reaches_the_child_and_the_stream_is_closed() {
        use std::io::{Read, Write};
        let i = Invocation {
            program: "/bin/cat".into(),
            args: vec![],
            env_set: vec![],
            env_remove: vec![],
            cwd: PathBuf::from("/tmp"),
            stdin: Some("the prompt".into()),
        };
        let mut cmd = i.command();
        cmd.stdout(Stdio::piped());
        let mut child = cmd.spawn().expect("spawning /bin/cat");
        {
            let mut sink = child.stdin.take().expect("stdin was not piped");
            sink.write_all(i.stdin.as_ref().unwrap().as_bytes())
                .unwrap();
        }
        let mut out = String::new();
        child
            .stdout
            .take()
            .unwrap()
            .read_to_string(&mut out)
            .unwrap();
        assert_eq!(out, "the prompt");
        assert!(child.wait().unwrap().success());
    }

    #[test]
    fn tail_keeps_the_last_nonempty_lines() {
        assert_eq!(tail("a\n\nb\nc\n", 2), "b\nc");
        assert_eq!(tail("only\n", 3), "only");
        assert_eq!(tail("", 3), "");
    }
}
