//! `ccc` — resolves an agent type to a harness, model, effort and account, and
//! runs it as a one-shot subprocess.

mod agents;
mod audit;
mod audit_view;
mod config;
mod exec;
mod fsutil;
mod glob;
mod harness;
mod output;
mod procs;
mod resolve;
mod select;
mod session;
mod status;
mod supervise;
#[cfg(test)]
mod testutil;
mod usage;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use jiff::Timestamp;
use std::io::{Read, Write};
use std::path::Path;

use crate::agents::AgentDef;
use crate::config::{Config, Effort, Harness, ProcessEnv};
use crate::select::Choice;

/// Exit code for usage, config and agent-definition errors, raised before any
/// harness is spawned.
const EXIT_USAGE: i32 = 2;

#[derive(Debug, Parser)]
#[command(
    name = "ccc",
    about = "Run an agent type on Claude Code or Codex",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Show invocation audit logs.
    Log(LogArgs),
    /// Show live and completed invocation status.
    Status(StatusArgs),
    Kill(KillArgs),
    /// Run an agent type as a one-shot subagent.
    Run(RunArgs),
    /// Continue a previous session by its id or by a trace prefix from `ccc log`.
    Resume(ResumeArgs),
    /// Print the resolution for an agent type without running anything.
    Which(WhichArgs),
    /// Start an interactive session as an agent type.
    Launch(LaunchArgs),
    #[command(hide = true)]
    Hook(HookArgs),
}

#[derive(Debug, Args)]
struct StatusArgs {
    /// A unique trace-id prefix.
    #[arg(long)]
    trace: Option<String>,
    /// Emit the machine-readable status projection.
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct KillArgs {
    #[arg(required_unless_present = "all", conflicts_with = "all")]
    targets: Vec<String>,
    #[arg(long, conflicts_with = "targets", required_unless_present = "targets")]
    all: bool,
    #[arg(long)]
    trace: Option<String>,
    #[arg(long)]
    now: bool,
    #[arg(long, value_parser = parse_grace)]
    grace: Option<f64>,
    #[arg(long)]
    no_wait: bool,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct LogArgs {
    /// A unique trace-id prefix.
    #[arg(group = "selector")]
    trace: Option<String>,
    /// Select the newest trace.
    #[arg(long, conflicts_with = "trace", group = "selector")]
    latest: bool,
    #[arg(long, requires = "selector", conflicts_with = "json")]
    flat: bool,
    #[arg(long)]
    verbose: bool,
    #[arg(long, requires = "selector", conflicts_with_all = ["flat", "verbose"])]
    json: bool,
}

#[derive(Debug, Args)]
struct RunArgs {
    /// Agent type, i.e. the stem of a `<type>.md` definition.
    agent: String,
    /// Working directory for the subagent.
    #[arg(long)]
    cwd: Option<String>,
    /// Read the prompt from this file, or from stdin when it is `-`.
    #[arg(long)]
    prompt_file: Option<String>,
    /// Claude account to use, bypassing usage-driven selection.
    #[arg(long)]
    profile: Option<String>,
    /// Force a harness.
    #[arg(long, value_parser = parse_harness)]
    harness: Option<Harness>,
    /// Emit a JSON object instead of text plus a trailer line.
    #[arg(long)]
    json: bool,
    /// The prompt; `-` reads it from stdin, as does omitting it.
    prompt: Option<String>,
}

#[derive(Debug, Args)]
struct ResumeArgs {
    /// Session id returned by a previous run, or a unique trace prefix from
    /// `ccc log`, which continues that trace's root session.
    #[arg(value_name = "SESSION-ID|TRACE-PREFIX")]
    session_id: String,
    /// Read the prompt from this file, or from stdin when it is `-`.
    #[arg(long)]
    prompt_file: Option<String>,
    /// Start a new session from this one, leaving the original untouched.
    #[arg(long)]
    fork: bool,
    /// Claude account for this turn, bypassing usage-driven selection; without
    /// it the account is selected anew from current usage.
    #[arg(long)]
    profile: Option<String>,
    #[arg(long)]
    json: bool,
    /// The prompt; `-` reads it from stdin, as does omitting it.
    prompt: Option<String>,
}

#[derive(Debug, Args)]
struct WhichArgs {
    agent: String,
    #[arg(long, value_parser = parse_harness)]
    harness: Option<Harness>,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Args)]
struct LaunchArgs {
    /// Agent type; omitted when resuming, the record names it. With neither an
    /// agent type nor --resume, a plain Claude session with no agent type.
    #[arg(conflicts_with = "resume")]
    agent: Option<String>,
    /// Session id, or unique trace prefix from `ccc log` naming its root
    /// session, to continue interactively; harness, agent, model and cwd come
    /// from the record.
    #[arg(long, value_name = "SESSION-ID|TRACE-PREFIX")]
    resume: Option<String>,
    /// With --resume: start a new session from the recorded one, leaving it untouched.
    #[arg(long, requires = "resume")]
    fork: bool,
    #[arg(long, conflicts_with = "resume")]
    cwd: Option<String>,
    #[arg(long)]
    profile: Option<String>,
    #[arg(long, value_parser = parse_harness, conflicts_with = "resume", requires = "agent")]
    harness: Option<Harness>,
    /// Without an agent type: model, as an alias from `[launch.aliases]` or a full model string.
    #[arg(long, value_parser = parse_model, conflicts_with_all = ["agent", "resume"])]
    model: Option<String>,
    /// Without an agent type: Claude effort.
    #[arg(long, value_parser = parse_claude_effort, conflicts_with_all = ["agent", "resume"])]
    effort: Option<Effort>,
    /// Arguments passed through to the harness verbatim; everything after `--`.
    #[arg(last = true, allow_hyphen_values = true)]
    passthrough: Vec<String>,
}

#[derive(Debug, Args)]
struct HookArgs {
    kind: audit::HookKind,
    payload: Option<String>,
}

fn parse_harness(s: &str) -> Result<Harness, String> {
    Harness::parse(s).map_err(|e| e.to_string())
}

fn parse_model(s: &str) -> Result<String, String> {
    if s.trim().is_empty() {
        Err("model must not be blank".into())
    } else {
        Ok(s.to_string())
    }
}

fn parse_claude_effort(s: &str) -> Result<Effort, String> {
    Effort::parse_for(s, Harness::Claude).map_err(|e| e.to_string())
}

fn parse_grace(s: &str) -> Result<f64, String> {
    let value = s
        .parse::<f64>()
        .map_err(|_| "grace must be a finite number greater than 0".to_string())?;
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err("grace must be a finite number greater than 0".into())
    }
}

fn main() {
    let cli = Cli::parse();
    match dispatch(cli) {
        Ok(code) => std::process::exit(code),
        Err(err) => {
            harness::diag(format_args!("{err:#}"));
            std::process::exit(EXIT_USAGE);
        }
    }
}

/// What `ccc launch` opens for the parsed flags: `--resume` wins, then an
/// agent type, then a roleless session.
fn launch_target(args: &LaunchArgs) -> exec::LaunchTarget<'_> {
    match args.resume.as_deref() {
        Some(id) => exec::LaunchTarget::Resume {
            id,
            fork: args.fork,
        },
        None => match args.agent.as_deref() {
            Some(agent) => exec::LaunchTarget::Fresh {
                agent,
                cwd: args.cwd.as_deref(),
                cli_harness: args.harness,
            },
            None => exec::LaunchTarget::Roleless {
                cwd: args.cwd.as_deref(),
                model: args.model.as_deref(),
                effort: args.effort,
            },
        },
    }
}

fn dispatch(cli: Cli) -> Result<i32> {
    match cli.command {
        Command::Log(args) => log(args),
        Command::Status(args) => status(args),
        Command::Kill(args) => kill(args),
        Command::Which(args) => which(args),
        Command::Run(args) => exec::run(
            &args.agent,
            args.cwd.as_deref(),
            args.prompt.as_deref(),
            args.prompt_file.as_deref(),
            args.profile.as_deref(),
            args.harness,
            args.json,
        ),
        Command::Resume(args) => exec::resume(
            &args.session_id,
            args.fork,
            args.profile.as_deref(),
            args.prompt.as_deref(),
            args.prompt_file.as_deref(),
            args.json,
        ),
        Command::Launch(mut args) => {
            let passthrough = std::mem::take(&mut args.passthrough);
            exec::launch(launch_target(&args), args.profile.as_deref(), passthrough)
        }
        Command::Hook(args) => {
            let mut stdin = std::io::stdin();
            let mut stdout = std::io::stdout();
            let mut stderr = std::io::stderr();
            Ok(dispatch_hook(
                args.kind,
                args.payload,
                &mut stdin,
                &audit::Recorder::dir_in(&ProcessEnv),
                &ProcessEnv,
                &mut stderr,
                &mut stdout,
            ))
        }
    }
}

fn status(args: StatusArgs) -> Result<i32> {
    let cfg = Config::load()?;
    status::status_with(
        status::StatusContext {
            dir: &audit::Recorder::dir_in(&ProcessEnv),
            cfg: &cfg,
            env: &ProcessEnv,
            now: Timestamp::now(),
            width: audit_view::terminal_width(),
            trace: args.trace,
            json: args.json,
        },
        &mut std::io::stdout(),
        &mut std::io::stderr(),
    )
}

fn kill(args: KillArgs) -> Result<i32> {
    let cfg = Config::load()?;
    status::kill_with(
        status::KillContext {
            dir: &audit::Recorder::dir_in(&ProcessEnv),
            cfg: &cfg,
            env: &ProcessEnv,
            now: Timestamp::now(),
            trace: args.trace,
            targets: args.targets,
            all: args.all,
            immediate: args.now,
            grace: args.grace,
            no_wait: args.no_wait,
            json: args.json,
            ops: &status::SystemProcessOps,
            scope_root: None,
        },
        &mut std::io::stdout(),
        &mut std::io::stderr(),
    )
}

fn dispatch_hook(
    kind: audit::HookKind,
    argv_payload: Option<String>,
    input: &mut dyn Read,
    dir: &Path,
    env: &dyn config::EnvSource,
    diagnostic: &mut dyn Write,
    _stdout: &mut dyn Write,
) -> i32 {
    let payload = match argv_payload {
        Some(payload) => payload,
        None => {
            let mut bytes = Vec::new();
            if let Err(e) = input.read_to_end(&mut bytes) {
                let _ = writeln!(diagnostic, "ccc: reading hook payload: {e}");
                return 0;
            }
            match String::from_utf8(bytes) {
                Ok(payload) => payload,
                Err(_) => {
                    let _ = writeln!(diagnostic, "ccc: hook payload was not UTF-8");
                    return 0;
                }
            }
        }
    };
    if let Err(e) = audit::ingest_hook(dir, kind, &payload, env) {
        let _ = writeln!(diagnostic, "ccc: {e}");
    }
    0
}

fn log(args: LogArgs) -> Result<i32> {
    audit_view::log_with(
        &audit::Recorder::dir_in(&ProcessEnv),
        &audit_view::LogOptions {
            trace: args.trace,
            latest: args.latest,
            flat: args.flat,
            verbose: args.verbose,
            json: args.json,
        },
        Timestamp::now(),
        audit_view::terminal_width(),
        &mut std::io::stdout(),
        &mut std::io::stderr(),
    )
}

fn which(args: WhichArgs) -> Result<i32> {
    let (cfg, sources) = Config::load_with_sources_in(&ProcessEnv)?;
    let usage = || usage::fetch(&cfg.mqtt, &cfg.claude.selection, &ProcessEnv);
    which_with(
        &WhichCtx {
            cfg: &cfg,
            sources: &sources,
            env: &ProcessEnv,
            now: Timestamp::now(),
            usage: &usage,
        },
        &args.agent,
        args.harness,
        args.json,
        &mut std::io::stdout(),
        &mut std::io::stderr(),
    )
}

struct WhichCtx<'a> {
    cfg: &'a Config,
    /// The files `cfg` was loaded from.
    sources: &'a config::ConfigSources,
    env: &'a dyn config::EnvSource,
    now: Timestamp,
    usage: &'a dyn Fn() -> usage::UsageSource,
}

/// [`which`] against an explicit context and output sinks. `err` receives the
/// two config paths and the routing notes.
fn which_with(
    ctx: &WhichCtx<'_>,
    agent: &str,
    cli_harness: Option<Harness>,
    as_json: bool,
    out: &mut dyn std::io::Write,
    err: &mut dyn std::io::Write,
) -> Result<i32> {
    writeln!(err, "config: {}", config_path_text(&ctx.sources.site))?;
    writeln!(
        err,
        "agents config: {}",
        config_path_text(&ctx.sources.agents)
    )?;
    let def = AgentDef::load(&ctx.cfg.agents_dir_in(ctx.env), agent)?;
    let (r, notes) = resolve::resolve(ctx.cfg, ctx.env, &def, cli_harness)?;
    for note in &notes {
        writeln!(err, "note: {note}")?;
    }
    let prompt = if def.overlays.contains_key(&r.harness) {
        format!("{}.{}.md", agent, r.harness)
    } else {
        format!("{agent}.md")
    };

    // Only a Claude resolution with configured profiles has an account to
    // choose, so only that one reads usage.
    let choice = match r.harness {
        Harness::Claude if ctx.cfg.claude.ambient() => None,
        Harness::Claude => {
            let source = (ctx.usage)();
            Some(select::choose(
                &ctx.cfg.claude,
                ctx.env,
                &r.model,
                ctx.now,
                &source,
                &[],
            )?)
        }
        Harness::Codex => None,
    };
    if let Some(c) = &choice {
        c.report();
    }

    writeln!(
        out,
        "{}",
        which_output(&r, choice.as_ref(), &prompt, ctx.sources, as_json)?
    )?;
    Ok(0)
}

/// `<path> (<VAR>|default)`, plus ` (missing, defaults)` when the file was not
/// read.
fn config_path_text(s: &config::ConfigSource) -> String {
    let missing = if s.read { "" } else { " (missing, defaults)" };
    format!("{}{missing}", s.path)
}

/// What `ccc which` prints in place of the candidate table when Claude runs
/// under the caller's own login.
const AMBIENT_LINE: &str = "ambient login ([claude] profiles is empty)";

/// The rendered resolution: a line plus the candidate table (or, for a Claude
/// run under the ambient login, [`AMBIENT_LINE`]), or a JSON object that also
/// names the two config paths. A Claude resolution without a `choice` is the
/// ambient login: every other Claude resolution chooses a profile or fails.
fn which_output(
    r: &resolve::Resolved,
    choice: Option<&Choice>,
    prompt: &str,
    sources: &config::ConfigSources,
    as_json: bool,
) -> Result<String> {
    if as_json {
        let candidates: Vec<serde_json::Value> = choice
            .map(|c| c.scored.iter().map(scored_json).collect())
            .unwrap_or_default();
        let out = serde_json::json!({
            "agent": r.agent,
            "harness": r.harness.as_str(),
            "model": r.model,
            "effort": r.effort.map(config::Effort::as_str),
            "sandbox": r.sandbox.as_str(),
            "prompt": prompt,
            "tools": r.tools,
            "profile": choice.map(|c| c.account.profile.clone()),
            "degraded": choice.and_then(|c| c.degraded.clone()),
            "candidates": candidates,
            "config": sources.site.path.path.to_string_lossy(),
            "agents_config": sources.agents.path.path.to_string_lossy(),
        });
        Ok(serde_json::to_string_pretty(&out)?)
    } else {
        let mut out = format!(
            "agent={} harness={} model={} effort={} sandbox={} prompt={} profile={}",
            r.label(),
            r.harness,
            r.model,
            r.effort.map(config::Effort::as_str).unwrap_or("-"),
            r.sandbox.as_str(),
            prompt,
            choice.map(|c| c.account.profile.as_str()).unwrap_or("-"),
        );
        if r.harness == Harness::Claude && choice.is_none() {
            out.push('\n');
            out.push_str(AMBIENT_LINE);
        }
        if let Some(c) = choice {
            out.push_str(&format!(
                "\n{:<10} {:>6} {:>6} {:>6} {:>7} {:>6} {:>4} {:>8} {:>6}  {}",
                "profile",
                "util5",
                "util_g",
                "util_f",
                "hours_g",
                "gate5",
                "tier",
                "score",
                "usable",
                "reason/note"
            ));
            for s in &c.scored {
                out.push_str(&format!(
                    "\n{:<10} {:>6.1} {:>6} {:>6} {:>7} {:>6.2} {:>4} {:>8.3} {:>6}  {}",
                    s.profile,
                    s.util5,
                    cell(s.general.map(|b| b.util), 1),
                    cell(s.fable.map(|b| b.util), 1),
                    cell(s.general.map(|b| b.hours), 1),
                    s.gate5,
                    s.tier().map_or("-".to_string(), |t| t.to_string()),
                    s.score,
                    if s.usable { "yes" } else { "no" },
                    s.reason
                        .clone()
                        .or_else(|| s.note())
                        .as_deref()
                        .unwrap_or("-")
                ));
            }
        }
        Ok(out)
    }
}

/// One table cell for a number a profile may not have: a bucket the account
/// does not have prints `-` rather than a zero it would be read as.
fn cell(value: Option<f64>, precision: usize) -> String {
    match value {
        Some(v) => format!("{v:.precision$}"),
        None => "-".to_string(),
    }
}

fn scored_json(s: &select::Scored) -> serde_json::Value {
    serde_json::json!({
        "profile": s.profile,
        "util5": s.util5,
        "util_g": s.general.map(|b| b.util),
        "util_f": s.fable.map(|b| b.util),
        "hours_g": s.general.map(|b| b.hours),
        "hours_f": s.fable.map(|b| b.hours),
        "five_hour_floor_pct": s.five_hour_floor_pct,
        "five_hour_knee_pct": s.five_hour_knee_pct,
        "gate5": s.gate5,
        "tier": s.tier(),
        "score": s.score,
        "usable": s.usable,
        "reason": s.reason,
        "note": s.note(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::Sandbox;
    use crate::config::Effort;
    use crate::testutil::TempDir;
    use clap::CommandFactory;
    use std::collections::BTreeMap;

    fn resolved() -> resolve::Resolved {
        resolve::Resolved {
            agent: Some("explorer".into()),
            description: "Investigates".into(),
            harness: Harness::Claude,
            model: "claude-opus-5[1M]".into(),
            effort: Some(Effort::Low),
            tools: Some(vec!["Read".into(), "Grep".into()]),
            sandbox: Sandbox::ReadOnly,
            body: "body".into(),
        }
    }

    #[test]
    fn the_clap_definition_is_well_formed() {
        Cli::command().debug_assert();
    }

    #[test]
    fn launch_passes_trailing_arguments_through_verbatim() {
        let cli =
            Cli::try_parse_from(["ccc", "launch", "orchestrator", "--", "--foo", "-x"]).unwrap();
        match cli.command {
            Command::Launch(a) => {
                assert_eq!(a.agent.as_deref(), Some("orchestrator"));
                assert_eq!(a.passthrough, vec!["--foo", "-x"]);
            }
            other => panic!("parsed as {other:?}"),
        }
    }

    #[test]
    fn launch_resume_form_takes_no_agent_and_routes_passthrough() {
        let launch = |argv: &[&str]| match Cli::try_parse_from(argv).unwrap().command {
            Command::Launch(a) => a,
            other => panic!("parsed as {other:?}"),
        };

        let a = launch(&["ccc", "launch", "--resume", "ID"]);
        assert_eq!(a.agent, None);
        assert_eq!(a.resume.as_deref(), Some("ID"));
        assert!(!a.fork);
        assert!(a.passthrough.is_empty());

        let a = launch(&[
            "ccc",
            "launch",
            "--resume",
            "ID",
            "--fork",
            "--",
            "--verbose",
        ]);
        assert!(a.fork);
        assert_eq!(a.passthrough, vec!["--verbose"]);

        let a = launch(&[
            "ccc",
            "launch",
            "--resume",
            "ID",
            "--profile",
            "p",
            "--",
            "--settings",
            "x",
        ]);
        assert_eq!(a.profile.as_deref(), Some("p"));
        assert_eq!(a.passthrough, vec!["--settings", "x"]);

        let a = launch(&["ccc", "launch", "explorer"]);
        assert_eq!(a.agent.as_deref(), Some("explorer"));
        assert!(a.passthrough.is_empty());

        for argv in [
            ["ccc", "launch", "--fork"].as_slice(),
            ["ccc", "launch", "--resume", "ID", "--cwd", "/srv"].as_slice(),
            ["ccc", "launch", "--resume", "ID", "--harness", "codex"].as_slice(),
            ["ccc", "launch", "--resume", "ID", "explorer"].as_slice(),
            ["ccc", "launch", "explorer", "--verbose"].as_slice(),
        ] {
            assert!(
                Cli::try_parse_from(argv).is_err(),
                "rejected form: {argv:?}"
            );
        }
    }

    #[test]
    fn launch_without_an_agent_is_roleless_and_takes_model_and_effort() {
        let launch = |argv: &[&str]| match Cli::try_parse_from(argv).unwrap().command {
            Command::Launch(a) => a,
            other => panic!("parsed as {other:?}"),
        };

        let a = launch(&["ccc", "launch"]);
        assert_eq!(a.agent, None);
        assert_eq!(a.resume, None);
        assert_eq!(a.model, None);
        assert_eq!(a.effort, None);

        let a = launch(&[
            "ccc",
            "launch",
            "--cwd",
            "/srv",
            "--profile",
            "p",
            "--model",
            "fable",
            "--effort",
            "high",
            "--",
            "--verbose",
        ]);
        assert_eq!(a.agent, None);
        assert_eq!(a.cwd.as_deref(), Some("/srv"));
        assert_eq!(a.profile.as_deref(), Some("p"));
        assert_eq!(a.model.as_deref(), Some("fable"));
        assert_eq!(a.effort, Some(Effort::High));
        assert_eq!(a.passthrough, vec!["--verbose"]);

        for argv in [
            ["ccc", "launch", "explorer", "--model", "opus"].as_slice(),
            ["ccc", "launch", "explorer", "--effort", "high"].as_slice(),
            ["ccc", "launch", "--resume", "ID", "--model", "opus"].as_slice(),
            ["ccc", "launch", "--resume", "ID", "--effort", "high"].as_slice(),
            ["ccc", "launch", "--harness", "codex"].as_slice(),
            ["ccc", "launch", "--effort", "minimal"].as_slice(),
            ["ccc", "launch", "--effort", "bogus"].as_slice(),
            ["ccc", "launch", "--model", ""].as_slice(),
            ["ccc", "launch", "--model", "  "].as_slice(),
        ] {
            assert!(
                Cli::try_parse_from(argv).is_err(),
                "rejected form: {argv:?}"
            );
        }
    }

    #[test]
    fn launch_flags_map_to_the_launch_target() {
        let target = |argv: &[&str], want: exec::LaunchTarget| {
            let Command::Launch(a) = Cli::try_parse_from(argv).unwrap().command else {
                panic!("{argv:?} is not a launch");
            };
            assert_eq!(launch_target(&a), want, "{argv:?}");
        };
        target(
            &["ccc", "launch"],
            exec::LaunchTarget::Roleless {
                cwd: None,
                model: None,
                effort: None,
            },
        );
        target(
            &[
                "ccc", "launch", "--cwd", "/srv", "--model", "fable", "--effort", "high",
            ],
            exec::LaunchTarget::Roleless {
                cwd: Some("/srv"),
                model: Some("fable"),
                effort: Some(Effort::High),
            },
        );
        target(
            &[
                "ccc",
                "launch",
                "explorer",
                "--cwd",
                "/srv",
                "--harness",
                "codex",
            ],
            exec::LaunchTarget::Fresh {
                agent: "explorer",
                cwd: Some("/srv"),
                cli_harness: Some(Harness::Codex),
            },
        );
        target(
            &["ccc", "launch", "--resume", "ID", "--fork"],
            exec::LaunchTarget::Resume {
                id: "ID",
                fork: true,
            },
        );
    }

    #[test]
    fn status_is_public_and_accepts_only_trace_and_json() {
        let parsed = Cli::try_parse_from(["ccc", "status", "--trace", "abcd", "--json"]).unwrap();
        match parsed.command {
            Command::Status(args) => {
                assert_eq!(args.trace.as_deref(), Some("abcd"));
                assert!(args.json);
            }
            other => panic!("parsed as {other:?}"),
        }
        assert!(Cli::try_parse_from(["ccc", "status", "--latest"]).is_err());
        assert!(Cli::command().render_help().to_string().contains("status"));
    }

    #[test]
    fn kill_is_public() {
        let parsed = Cli::try_parse_from([
            "ccc",
            "kill",
            "abc",
            "def",
            "--trace",
            "1234",
            "--now",
            "--grace",
            "0.5",
            "--no-wait",
            "--json",
        ])
        .unwrap();
        match parsed.command {
            Command::Kill(args) => {
                assert!(args.now && args.no_wait && args.json);
                assert_eq!(args.targets.len(), 2);
            }
            other => panic!("parsed as {other:?}"),
        }
        assert!(Cli::try_parse_from(["ccc", "kill"]).is_err());
        assert!(Cli::try_parse_from(["ccc", "kill", "abc", "--all"]).is_err());
        assert!(Cli::try_parse_from(["ccc", "kill", "abc", "--grace", "0"]).is_err());
        assert!(Cli::try_parse_from(["ccc", "kill", "abc", "--grace", "-1"]).is_err());
        assert!(Cli::try_parse_from(["ccc", "kill", "abc", "--grace", "NaN"]).is_err());
        assert!(Cli::try_parse_from(["ccc", "kill", "--all", "--trace", "1234"]).is_ok());
        assert!(Cli::command().render_help().to_string().contains("kill"));
    }

    #[test]
    fn hook_is_hidden_but_accepts_codex_payload_as_one_argument() {
        let parsed = Cli::try_parse_from([
            "ccc",
            "hook",
            "codex-notify",
            r#"{"type":"agent-turn-complete"}"#,
        ])
        .unwrap();
        match parsed.command {
            Command::Hook(args) => assert_eq!(
                args.payload.as_deref(),
                Some(r#"{"type":"agent-turn-complete"}"#)
            ),
            other => panic!("parsed as {other:?}"),
        }
        let rendered = Cli::command().render_help().to_string();
        assert!(!rendered.contains("hook"));
    }

    #[test]
    fn both_resume_forms_advertise_a_session_id_or_trace_prefix() {
        let mut command = Cli::command();
        for name in ["resume", "launch"] {
            let help = command
                .find_subcommand_mut(name)
                .unwrap()
                .render_help()
                .to_string();
            assert!(help.contains("SESSION-ID|TRACE-PREFIX"), "{name}: {help}");
            assert!(help.contains("trace prefix"), "{name}: {help}");
        }
    }

    #[test]
    fn resume_takes_a_profile_alongside_every_other_flag() {
        let cases: [&[&str]; 5] = [
            &["ccc", "resume", "9f22dd1b", "--profile", "team03"],
            &[
                "ccc",
                "resume",
                "9f22dd1b",
                "--profile",
                "team03",
                "--fork",
                "go",
            ],
            &[
                "ccc",
                "resume",
                "9f22dd1b",
                "--json",
                "--profile",
                "team03",
                "go",
            ],
            &[
                "ccc",
                "resume",
                "9f22dd1b",
                "--profile",
                "team03",
                "--prompt-file",
                "p.md",
            ],
            &["ccc", "resume", "20260924", "--profile", "team03", "-"],
        ];
        for argv in cases {
            match Cli::try_parse_from(argv).unwrap().command {
                Command::Resume(a) => {
                    assert_eq!(a.profile.as_deref(), Some("team03"), "{argv:?}");
                    assert_eq!(a.session_id, argv[2], "{argv:?}");
                    assert_eq!(a.fork, argv.contains(&"--fork"), "{argv:?}");
                    assert_eq!(a.json, argv.contains(&"--json"), "{argv:?}");
                    let file = argv.contains(&"--prompt-file").then_some("p.md");
                    assert_eq!(a.prompt_file.as_deref(), file, "{argv:?}");
                }
                other => panic!("parsed as {other:?}"),
            }
        }
        match Cli::try_parse_from(["ccc", "resume", "9f22dd1b", "go"])
            .unwrap()
            .command
        {
            Command::Resume(a) => assert_eq!(a.profile, None),
            other => panic!("parsed as {other:?}"),
        }
        let help = Cli::command()
            .find_subcommand_mut("resume")
            .unwrap()
            .render_help()
            .to_string();
        assert!(help.contains("--profile"), "{help}");
        assert!(help.contains("bypassing usage-driven selection"), "{help}");
    }

    #[test]
    fn an_unknown_hook_kind_is_rejected_at_parse_time() {
        let err = Cli::try_parse_from(["ccc", "hook", "unknown"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown hook kind"), "{err}");
    }

    struct FailingReader;
    impl Read for FailingReader {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("test read"))
        }
    }

    #[test]
    fn test_missing_hook_dispatch_exit_zero() {
        let dir = TempDir::new("hook-dispatch");
        let trace = uuid::Uuid::new_v4().to_string();
        let inv = uuid::Uuid::new_v4().to_string();
        let env: BTreeMap<String, String> = [
            ("CCC_TRACE_ID".into(), trace.clone()),
            ("CCC_PARENT_INVOCATION_ID".into(), inv),
        ]
        .into_iter()
        .collect();
        let payload = r#"{"type":"agent-turn-complete","thread-id":"thread","turn-id":"turn","last-assistant-message":null}"#;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let mut input = payload.as_bytes();
        assert_eq!(
            dispatch_hook(
                audit::HookKind::CodexNotify,
                None,
                &mut input,
                dir.path(),
                &env,
                &mut err,
                &mut out
            ),
            0
        );
        assert!(out.is_empty() && err.is_empty());
        assert_eq!(audit::read_trace(dir.path(), &trace).unwrap().len(), 1);

        let before = audit::read_trace(dir.path(), &trace).unwrap().len();
        out.clear();
        err.clear();
        let mut non_utf8 = &b"\xff"[..];
        assert_eq!(
            dispatch_hook(
                audit::HookKind::ClaudeStop,
                None,
                &mut non_utf8,
                dir.path(),
                &env,
                &mut err,
                &mut out,
            ),
            0
        );
        assert!(out.is_empty());
        assert_eq!(err.iter().filter(|b| **b == b'\n').count(), 1);
        assert_eq!(audit::read_trace(dir.path(), &trace).unwrap().len(), before);

        for (kind, body) in [
            (audit::HookKind::ClaudeStop, Some("{")),
            (audit::HookKind::ClaudeStop, None),
        ] {
            out.clear();
            err.clear();
            let mut input: Box<dyn Read> = if body.is_some() {
                Box::new(&b"\xff"[..])
            } else {
                Box::new(&b"{}"[..])
            };
            assert_eq!(
                dispatch_hook(
                    kind,
                    body.map(str::to_owned),
                    &mut input,
                    dir.path(),
                    &BTreeMap::new(),
                    &mut err,
                    &mut out
                ),
                0
            );
            assert!(out.is_empty());
            assert_eq!(err.iter().filter(|b| **b == b'\n').count(), 1);
        }
        out.clear();
        err.clear();
        let mut bad = FailingReader;
        assert_eq!(
            dispatch_hook(
                audit::HookKind::ClaudeStop,
                None,
                &mut bad,
                dir.path(),
                &env,
                &mut err,
                &mut out
            ),
            0
        );
        assert!(out.is_empty() && String::from_utf8_lossy(&err).starts_with("ccc:"));
    }

    #[test]
    fn an_unknown_harness_is_rejected_at_parse_time() {
        let err = Cli::try_parse_from(["ccc", "which", "explorer", "--harness", "gemini"])
            .unwrap_err()
            .to_string();
        assert!(err.contains("gemini"), "{err}");
        let cli = Cli::try_parse_from(["ccc", "which", "explorer", "--harness", "codex"]).unwrap();
        match cli.command {
            Command::Which(a) => assert_eq!(a.harness, Some(Harness::Codex)),
            other => panic!("parsed as {other:?}"),
        }
    }

    #[test]
    fn run_takes_its_prompt_from_the_flags_or_the_positional() {
        let cli = Cli::try_parse_from([
            "ccc",
            "run",
            "explorer",
            "--cwd",
            "/srv",
            "--profile",
            "team03",
            "--json",
            "go",
        ])
        .unwrap();
        match cli.command {
            Command::Run(a) => {
                assert_eq!(a.cwd.as_deref(), Some("/srv"));
                assert_eq!(a.profile.as_deref(), Some("team03"));
                assert!(a.json);
                assert_eq!(a.prompt.as_deref(), Some("go"));
            }
            other => panic!("parsed as {other:?}"),
        }
        let cli = Cli::try_parse_from(["ccc", "resume", "9f22dd1b", "--prompt-file", "-"]).unwrap();
        match cli.command {
            Command::Resume(a) => {
                assert_eq!(a.session_id, "9f22dd1b");
                assert_eq!(a.prompt_file.as_deref(), Some("-"));
                assert!(!a.fork);
            }
            other => panic!("parsed as {other:?}"),
        }
        let cli = Cli::try_parse_from(["ccc", "resume", "9f22dd1b", "--fork", "go"]).unwrap();
        match cli.command {
            Command::Resume(a) => {
                assert_eq!(a.session_id, "9f22dd1b");
                assert!(a.fork);
                assert_eq!(a.prompt.as_deref(), Some("go"));
            }
            other => panic!("parsed as {other:?}"),
        }
    }

    #[test]
    fn a_bare_dash_parses_as_the_positional_prompt() {
        let cli = Cli::try_parse_from(["ccc", "run", "explorer", "--cwd", "/srv", "-"]).unwrap();
        match cli.command {
            Command::Run(a) => {
                assert_eq!(a.prompt.as_deref(), Some("-"));
                assert_eq!(a.prompt_file, None);
            }
            other => panic!("parsed as {other:?}"),
        }
        let cli = Cli::try_parse_from(["ccc", "resume", "9f22dd1b", "-"]).unwrap();
        match cli.command {
            Command::Resume(a) => {
                assert_eq!(a.prompt.as_deref(), Some("-"));
                assert_eq!(a.prompt_file, None);
            }
            other => panic!("parsed as {other:?}"),
        }
    }

    #[test]
    fn launch_takes_the_same_account_and_harness_overrides_as_run() {
        let cli = Cli::try_parse_from([
            "ccc",
            "launch",
            "orchestrator",
            "--cwd",
            "/srv",
            "--profile",
            "team01",
            "--harness",
            "codex",
        ])
        .unwrap();
        match cli.command {
            Command::Launch(a) => {
                assert_eq!(a.cwd.as_deref(), Some("/srv"));
                assert_eq!(a.profile.as_deref(), Some("team01"));
                assert_eq!(a.harness, Some(Harness::Codex));
                assert!(a.passthrough.is_empty());
                assert!(a.resume.is_none());
                assert!(!a.fork);
            }
            other => panic!("parsed as {other:?}"),
        }
    }

    #[test]
    fn log_selector_and_presentation_matrix_is_enforced_by_clap() {
        for argv in [
            ["ccc", "log"].as_slice(),
            ["ccc", "log", "--verbose"].as_slice(),
            ["ccc", "log", "abc"].as_slice(),
            ["ccc", "log", "--latest"].as_slice(),
            ["ccc", "log", "abc", "--flat"].as_slice(),
            ["ccc", "log", "--latest", "--json"].as_slice(),
        ] {
            assert!(Cli::try_parse_from(argv).is_ok(), "accepted form: {argv:?}");
        }
        for argv in [
            ["ccc", "log", "abc", "--latest"].as_slice(),
            ["ccc", "log", "--flat"].as_slice(),
            ["ccc", "log", "--json"].as_slice(),
            ["ccc", "log", "abc", "--json", "--flat"].as_slice(),
            ["ccc", "log", "abc", "--json", "--verbose"].as_slice(),
        ] {
            assert!(
                Cli::try_parse_from(argv).is_err(),
                "rejected form: {argv:?}"
            );
        }
    }

    /// Both files read from the paths `env` names.
    fn sources_in(env: &std::collections::BTreeMap<String, String>) -> config::ConfigSources {
        let paths = config::ConfigPaths::in_env(env);
        config::ConfigSources {
            site: config::ConfigSource {
                path: paths.site,
                read: true,
            },
            agents: config::ConfigSource {
                path: paths.agents,
                read: true,
            },
        }
    }

    fn test_sources() -> config::ConfigSources {
        sources_in(&std::collections::BTreeMap::from([
            ("CCC_CONFIG".to_string(), "/site.toml".to_string()),
            ("CCC_AGENTS_CONFIG".to_string(), "/agents.toml".to_string()),
        ]))
    }

    #[test]
    fn which_renders_both_ways() {
        let text = which_output(&resolved(), None, "explorer.md", &test_sources(), false).unwrap();
        assert_eq!(
            text,
            "agent=explorer harness=claude model=claude-opus-5[1M] effort=low sandbox=read-only prompt=explorer.md profile=-\nambient login ([claude] profiles is empty)"
        );

        let v: serde_json::Value = serde_json::from_str(
            &which_output(&resolved(), None, "explorer.md", &test_sources(), true).unwrap(),
        )
        .unwrap();
        assert_eq!(v["agent"], "explorer");
        assert_eq!(v["harness"], "claude");
        assert_eq!(v["model"], "claude-opus-5[1M]");
        assert_eq!(v["effort"], "low");
        assert_eq!(v["prompt"], "explorer.md");
        assert_eq!(v["sandbox"], "read-only");
        assert_eq!(v["tools"], serde_json::json!(["Read", "Grep"]));
        assert!(v["profile"].is_null());
        assert_eq!(v["candidates"], serde_json::json!([]));
        assert_eq!(v["config"], "/site.toml");
        assert_eq!(v["agents_config"], "/agents.toml");

        let mut r = resolved();
        r.effort = None;
        r.tools = None;
        assert!(
            which_output(&r, None, "explorer.md", &test_sources(), false)
                .unwrap()
                .contains("effort=-")
        );
        let v: serde_json::Value = serde_json::from_str(
            &which_output(&r, None, "explorer.md", &test_sources(), true).unwrap(),
        )
        .unwrap();
        assert!(v["effort"].is_null());
        assert!(v["tools"].is_null());
    }

    #[test]
    fn which_scores_an_account_for_claude_and_none_for_codex() {
        use crate::testutil::TempDir;
        use crate::usage::{ProfileUsage, UsageReport, UsageSource, UsageWindow};
        use claude_usage_mqtt_core::{AuthState, FIVE_HOUR};

        let dir = TempDir::new("which");
        std::fs::create_dir_all(dir.path().join("agents")).unwrap();
        std::fs::create_dir_all(dir.path().join("tokens")).unwrap();
        std::fs::write(
            dir.path().join("agents/explorer.md"),
            "---\nname: explorer\ndescription: Investigates\nmodel: claude-haiku-4-5\ncodex_model: gpt-5.6-luna\n---\nbody\n",
        )
        .unwrap();
        let token = dir.path().join("tokens/team03.token");
        std::fs::write(&token, "sk-ant-oat-x\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&token, std::fs::Permissions::from_mode(0o600)).unwrap();
        }

        let mut cfg = Config {
            agents_dir: Some(dir.path().join("agents").display().to_string()),
            ..Config::default()
        };
        cfg.claude.auth = config::ClaudeAuth::Token;
        cfg.claude.token_dir = dir.path().join("tokens").display().to_string();
        cfg.claude.profiles = vec!["team03".into()];
        let xdg = dir.path().join("xdg");
        let env: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::from([(
            "XDG_CONFIG_HOME".to_string(),
            xdg.display().to_string(),
        )]);
        let mut sources = sources_in(&env);
        sources.site.read = false;
        let now = Timestamp::now();

        let usage = || {
            UsageSource::Report(UsageReport {
                profiles: vec![ProfileUsage {
                    profile: "team03".into(),
                    auth: AuthState::Ok,
                    fresh: true,
                    updated_at: Timestamp::now(),
                    windows: vec![UsageWindow {
                        key: FIVE_HOUR.to_string(),
                        utilization: 10.0,
                        resets_at: None,
                    }],
                }],
                daemon_status: Some("online".into()),
            })
        };
        let ctx = WhichCtx {
            cfg: &cfg,
            sources: &sources,
            env: &env,
            now,
            usage: &usage,
        };
        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        assert_eq!(
            which_with(&ctx, "explorer", None, false, &mut out, &mut err).unwrap(),
            0
        );
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("agent=explorer "), "{text}");
        assert_eq!(
            String::from_utf8(err).unwrap(),
            format!(
                "config: {} (default) (missing, defaults)\nagents config: {} (default)\n",
                xdg.join("ccc/config.toml").display(),
                xdg.join("ccc/agents.toml").display()
            )
        );
        assert!(text.contains("harness=claude"), "{text}");
        assert!(text.contains("profile=team03"), "{text}");
        assert!(text.contains("util5"), "{text}");

        // A Codex resolution has no account to pick, so it never reads usage.
        let boom = || panic!("codex resolution must not read usage");
        let ctx = WhichCtx {
            cfg: &cfg,
            sources: &sources,
            env: &env,
            now,
            usage: &boom,
        };
        let mut out: Vec<u8> = Vec::new();
        which_with(
            &ctx,
            "explorer",
            Some(Harness::Codex),
            false,
            &mut out,
            &mut std::io::sink(),
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("harness=codex"), "{text}");
        assert!(text.contains("profile=-"), "{text}");

        std::fs::write(
            dir.path().join("agents/explorer.codex.md"),
            "codex overlay\n",
        )
        .unwrap();
        let mut out = Vec::new();
        which_with(
            &ctx,
            "explorer",
            Some(Harness::Codex),
            false,
            &mut out,
            &mut std::io::sink(),
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("prompt=explorer.codex.md"), "{text}");
        let mut out = Vec::new();
        which_with(
            &ctx,
            "explorer",
            Some(Harness::Codex),
            true,
            &mut out,
            &mut std::io::sink(),
        )
        .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(json["prompt"], "explorer.codex.md");
        assert_eq!(
            json["config"],
            xdg.join("ccc/config.toml").display().to_string()
        );
        assert_eq!(
            json["agents_config"],
            xdg.join("ccc/agents.toml").display().to_string()
        );

        std::fs::write(
            dir.path().join("agents/explorer.md"),
            "---\nname: explorer\ndescription: Investigates\nharness: claude\nmodel: claude-haiku-4-5\n---\nbody\n",
        )
        .unwrap();
        let mut env = std::collections::BTreeMap::new();
        env.insert("CCC_CONFIG".into(), "/site.toml".into());
        env.insert("CCC_AGENTS_CONFIG".into(), "/agents.toml".into());
        env.insert("CCC_HARNESS_ALL".into(), "codex".into());
        let skipped_sources = sources_in(&env);
        {
            let skipped_ctx = WhichCtx {
                cfg: &cfg,
                sources: &skipped_sources,
                env: &env,
                now,
                usage: &usage,
            };
            let mut out = Vec::new();
            let mut err = Vec::new();
            which_with(&skipped_ctx, "explorer", None, true, &mut out, &mut err).unwrap();
            assert_eq!(
                String::from_utf8(err).unwrap(),
                "config: /site.toml (CCC_CONFIG)\n\
                 agents config: /agents.toml (CCC_AGENTS_CONFIG)\n\
                 note: CCC_HARNESS_ALL selected codex for agent \"explorer\", \
                 but allowed harnesses are: claude; skipped\n"
            );
            let json: serde_json::Value = serde_json::from_slice(&out).unwrap();
            assert_eq!(json["config"], "/site.toml");
            assert_eq!(json["agents_config"], "/agents.toml");
            assert_eq!(json["prompt"], "explorer.md");
            assert!(!String::from_utf8(out).unwrap().contains("skipped"));
        }

        env.insert("CCC_HARNESS_EXPLORER".into(), "codex".into());
        let skipped_ctx = WhichCtx {
            cfg: &cfg,
            sources: &skipped_sources,
            env: &env,
            now,
            usage: &usage,
        };
        let err = which_with(
            &skipped_ctx,
            "explorer",
            None,
            true,
            &mut Vec::new(),
            &mut std::io::sink(),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("explorer") && err.contains("codex"), "{err}");
        assert!(err.contains("CCC_HARNESS_EXPLORER"), "{err}");
    }

    #[test]
    fn which_under_the_ambient_login_prints_the_ambient_line_and_reads_no_usage() {
        use crate::testutil::TempDir;

        let dir = TempDir::new("which-ambient");
        std::fs::create_dir_all(dir.path().join("agents")).unwrap();
        std::fs::write(
            dir.path().join("agents/explorer.md"),
            "---\nname: explorer\ndescription: Investigates\nmodel: claude-haiku-4-5\ncodex_model: gpt-5.6-luna\n---\nbody\n",
        )
        .unwrap();
        let cfg = Config {
            agents_dir: Some(dir.path().join("agents").display().to_string()),
            ..Config::default()
        };
        let env: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::from([(
            "XDG_CONFIG_HOME".to_string(),
            dir.path().join("xdg").display().to_string(),
        )]);
        let sources = sources_in(&env);
        let boom = || panic!("the ambient login must not read usage");
        let ctx = WhichCtx {
            cfg: &cfg,
            sources: &sources,
            env: &env,
            now: Timestamp::now(),
            usage: &boom,
        };

        let mut out: Vec<u8> = Vec::new();
        let mut err: Vec<u8> = Vec::new();
        which_with(&ctx, "explorer", None, false, &mut out, &mut err).unwrap();
        let text = String::from_utf8(out).unwrap();
        let mut lines = text.lines();
        assert!(lines.next().unwrap().ends_with(" profile=-"), "{text}");
        assert_eq!(
            lines.next(),
            Some("ambient login ([claude] profiles is empty)"),
            "{text}"
        );
        assert_eq!(lines.next(), None, "{text}");
        let err = String::from_utf8(err).unwrap();
        assert!(!err.contains("usage data unavailable"), "{err}");
        assert!(!err.contains("falling back"), "{err}");

        let mut out: Vec<u8> = Vec::new();
        which_with(&ctx, "explorer", None, true, &mut out, &mut std::io::sink()).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(v["harness"], "claude");
        assert!(v["profile"].is_null());
        assert_eq!(v["candidates"], serde_json::json!([]));

        // A Codex resolution is not the ambient login: it has no Claude login.
        let mut out: Vec<u8> = Vec::new();
        which_with(
            &ctx,
            "explorer",
            Some(Harness::Codex),
            false,
            &mut out,
            &mut std::io::sink(),
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(!text.contains("ambient"), "{text}");
    }

    #[test]
    fn which_prints_the_candidate_table_when_an_account_was_chosen() {
        let choice = Choice {
            account: crate::harness::claude::Account {
                profile: "team03".into(),
                auth: crate::config::ClaudeAuth::Token,
                token_dir: "/t".into(),
                profiles_dir: "/p".into(),
            },
            scored: vec![
                select::Scored {
                    profile: "team03".into(),
                    util5: 31.0,
                    general: Some(select::Bucket {
                        util: 86.0,
                        hours: 84.0,
                    }),
                    fable: Some(select::Bucket {
                        util: 77.0,
                        hours: 84.0,
                    }),
                    five_hour_floor_pct: 1.0,
                    five_hour_knee_pct: 15.0,
                    gate5: 1.0,
                    score: 0.03,
                    usable: true,
                    reason: None,
                    notes: vec!["scored without the Fable reserve".into()],
                },
                select::Scored {
                    profile: "team04".into(),
                    util5: 99.9,
                    general: Some(select::Bucket {
                        util: 90.0,
                        hours: 84.0,
                    }),
                    fable: Some(select::Bucket {
                        util: 83.0,
                        hours: 84.0,
                    }),
                    five_hour_floor_pct: 1.0,
                    five_hour_knee_pct: 15.0,
                    gate5: 0.0,
                    score: 0.0,
                    usable: false,
                    reason: Some(
                        "the 5-hour window is at or below its floor (0 % left, floor 1 %)".into(),
                    ),
                    notes: Vec::new(),
                },
                select::Scored {
                    profile: "nofable".into(),
                    util5: 0.0,
                    general: Some(select::Bucket {
                        util: 7.0,
                        hours: 84.0,
                    }),
                    fable: None,
                    five_hour_floor_pct: 1.0,
                    five_hour_knee_pct: 15.0,
                    gate5: 1.0,
                    score: 0.0,
                    usable: false,
                    reason: Some("no Fable weekly window in the snapshot".into()),
                    notes: Vec::new(),
                },
                select::Scored {
                    profile: "ramped".into(),
                    util5: 50.0,
                    general: Some(select::Bucket {
                        util: 10.0,
                        hours: 84.0,
                    }),
                    fable: None,
                    five_hour_floor_pct: 20.0,
                    five_hour_knee_pct: 60.0,
                    gate5: 0.75,
                    score: 0.8,
                    usable: true,
                    reason: None,
                    notes: vec!["in the 5-hour ramp".into()],
                },
                select::Scored {
                    profile: "spent".into(),
                    util5: 0.0,
                    general: Some(select::Bucket {
                        util: 100.0,
                        hours: 84.0,
                    }),
                    fable: None,
                    five_hour_floor_pct: 1.0,
                    five_hour_knee_pct: 15.0,
                    gate5: 1.0,
                    score: 0.0,
                    usable: true,
                    reason: None,
                    notes: vec![
                        "scored without the Fable reserve".into(),
                        "weekly score is zero".into(),
                    ],
                },
            ],
            degraded: None,
            warnings: vec!["profile personal has no credential".into()],
        };

        let text = which_output(
            &resolved(),
            Some(&choice),
            "explorer.md",
            &test_sources(),
            false,
        )
        .unwrap();
        assert!(text.contains("profile=team03"), "{text}");
        assert!(text.contains("util5"), "{text}");
        assert!(text.contains("team04"), "{text}");
        assert!(
            text.contains("5-hour window is at or below its floor"),
            "{text}"
        );
        assert!(text.contains("scored without the Fable reserve"), "{text}");
        assert!(text.contains("usable"), "{text}");
        // The profile with no Fable bucket shows `-` in the util_f cell.
        let row = text.lines().find(|l| l.starts_with("nofable")).unwrap();
        let cells: Vec<&str> = row.split_whitespace().collect();
        assert_eq!(cells[2], "7.0", "{row}");
        assert_eq!(cells[3], "-", "{row}");
        // The tier column sits between gate5 and score; `-` for an unusable row.
        let header = text.lines().nth(1).unwrap();
        let heads: Vec<&str> = header.split_whitespace().collect();
        assert_eq!(&heads[5..8], &["gate5", "tier", "score"], "{header}");
        assert_eq!(cells[6], "-", "{row}");
        let row = text.lines().find(|l| l.starts_with("team03")).unwrap();
        assert_eq!(row.split_whitespace().nth(6), Some("1"), "{row}");
        let row = text.lines().find(|l| l.starts_with("ramped")).unwrap();
        assert_eq!(row.split_whitespace().nth(6), Some("2"), "{row}");
        assert!(row.ends_with("in the 5-hour ramp"), "{row}");
        let row = text.lines().find(|l| l.starts_with("spent")).unwrap();
        assert!(
            row.ends_with("scored without the Fable reserve; weekly score is zero"),
            "{row}"
        );
        // The warning is a stderr diagnostic, never part of the table.
        assert!(!text.contains("personal"), "{text}");

        let v: serde_json::Value = serde_json::from_str(
            &which_output(
                &resolved(),
                Some(&choice),
                "explorer.md",
                &test_sources(),
                true,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(v["profile"], "team03");
        assert!(v["degraded"].is_null());
        assert_eq!(v["candidates"].as_array().unwrap().len(), 5);
        assert_eq!(v["candidates"][0]["tier"], 1);
        assert!(v["candidates"][1]["tier"].is_null());
        assert_eq!(v["candidates"][3]["tier"], 2);
        assert_eq!(v["candidates"][3]["five_hour_floor_pct"], 20.0);
        assert_eq!(v["candidates"][3]["five_hour_knee_pct"], 60.0);
        assert_eq!(v["candidates"][0]["five_hour_knee_pct"], 15.0);
        assert_eq!(
            v["candidates"][4]["note"],
            "scored without the Fable reserve; weekly score is zero"
        );
        assert_eq!(v["candidates"][0]["score"], 0.03);
        assert_eq!(v["candidates"][1]["usable"], false);
        assert!(v["candidates"][0]["reason"].is_null());
        assert!(v["candidates"][1]["note"].is_null());
        // A bucket the account does not have is null, never a zero.
        assert_eq!(v["candidates"][0]["util_f"], 77.0);
        assert!(v["candidates"][2]["util_f"].is_null());
        assert!(v["candidates"][2]["hours_f"].is_null());
        assert_eq!(v["candidates"][2]["util_g"], 7.0);
    }
}
