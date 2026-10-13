//! The Codex driver: `codex exec --json` with the agent body passed as
//! `developer_instructions`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use serde_json::json;

use crate::audit::HookKind;
use crate::config::{Approval, EnvSource};
use crate::harness::{
    binary_in, check_arg_len, tail, Captured, Common, Driver, HarnessError, Invocation, Io, Origin,
    Outcome,
};
use crate::resolve::Resolved;

/// The Codex harness: an approval policy and the `-o` file it writes its final
/// message to.
pub struct CodexDriver<'a> {
    pub approval: Approval,
    /// Where `-o` writes the final message; unused by `launch`.
    pub output_file: Option<PathBuf>,
    pub env: &'a dyn EnvSource,
}

impl Driver for CodexDriver<'_> {
    fn build(&self, r: &Resolved, c: &Common) -> Result<Invocation> {
        build(r, c, self)
    }

    fn parse(&self, captured: &Captured) -> Result<Outcome, HarnessError> {
        parse(
            &captured.stdout,
            captured.output_file.as_deref(),
            &captured.stderr,
            captured.exit_code,
        )
    }

    fn early_session_id(&self, line: &str) -> Option<String> {
        thread_started_id(line)
    }
}

fn thread_started_id(line: &str) -> Option<String> {
    let ev = serde_json::from_str::<serde_json::Value>(line.trim()).ok()?;
    (ev.get("type").and_then(|v| v.as_str()) == Some("thread.started"))
        .then(|| {
            ev.get("thread_id")
                .or_else(|| ev.pointer("/thread/id"))
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .flatten()
}

/// Mints the path `-o` writes to: a fresh file, created empty and exclusively,
/// in a directory only this user can enter. Nothing else can pre-create or
/// substitute it, so whatever is read back was written by this child.
///
/// The caller must pass the path to [`take_output_file`] once the child has
/// exited; nothing else removes the file.
pub fn new_output_file(env: &dyn EnvSource) -> Result<PathBuf> {
    let dir = crate::config::state_home_in(env).join("ccc/tmp");
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700))
            .with_context(|| format!("securing {}", dir.display()))?;
    }

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let n = OUTPUT_SEQ.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!("out-{}-{nanos}-{n}.txt", std::process::id()));
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .with_context(|| format!("creating {}", path.display()))?;
    Ok(path)
}

static OUTPUT_SEQ: AtomicU64 = AtomicU64::new(0);

/// Reads the file [`new_output_file`] minted and removes it, so a subagent's
/// final message is not left on disk. Returns `None` when the child wrote
/// nothing or the file is already gone.
pub fn take_output_file(path: &std::path::Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok();
    let _ = std::fs::remove_file(path);
    content.filter(|s| !s.trim().is_empty())
}

/// Encodes `s` as a TOML basic string, quotes included, so it can never be
/// mis-parsed as some other TOML value.
pub fn toml_basic_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\u{:04X}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn push_config(args: &mut Vec<String>, key: &str, value: &str) {
    args.push("-c".into());
    args.push(format!("{key}={value}"));
}

/// Builds the argv, cwd and stdin for one Codex invocation.
fn build(r: &Resolved, c: &Common, d: &CodexDriver) -> Result<Invocation> {
    let instructions = toml_basic_string(&r.body);
    check_arg_len(r.label(), "developer_instructions", &instructions)?;

    let interactive = c.mode.is_launch();
    let one_shot_fresh = !interactive && c.mode.origin == Origin::Fresh;

    let mut args: Vec<String> = Vec::new();
    match (&c.mode.io, &c.mode.origin) {
        (Io::Interactive { self_exe }, origin) => {
            match origin {
                Origin::Fresh => {}
                // --include-non-interactive: admit exec-origin threads.
                Origin::Resume { id, fork: false } => {
                    args.push("resume".into());
                    args.push(id.clone());
                    args.push("--include-non-interactive".into());
                }
                Origin::Resume { id, fork: true } => {
                    args.push("fork".into());
                    args.push(id.clone());
                }
            }
            args.push("-m".into());
            args.push(r.model.clone());
            push_config(
                &mut args,
                "notify",
                &format!(
                    "[{}, {}, {}]",
                    toml_basic_string(&self_exe.display().to_string()),
                    toml_basic_string("hook"),
                    toml_basic_string(HookKind::CodexNotify.as_str())
                ),
            );
        }
        (Io::OneShot { .. }, Origin::Fresh) => {
            args.push("exec".into());
            args.push("--json".into());
            args.push("-m".into());
            args.push(r.model.clone());
        }
        (Io::OneShot { .. }, Origin::Resume { id, fork }) => {
            args.push("exec".into());
            args.push(if *fork { "fork" } else { "resume" }.into());
            args.push(id.clone());
            args.push("--json".into());
            // Assumed: `resume` does not inherit the model; pin it explicitly.
            push_config(&mut args, "model", &toml_basic_string(&r.model));
        }
    }

    if let Some(effort) = r.effort {
        push_config(
            &mut args,
            "model_reasoning_effort",
            &toml_basic_string(effort.as_str()),
        );
    }
    push_config(&mut args, "developer_instructions", &instructions);

    if interactive || one_shot_fresh {
        args.push("-s".into());
        args.push(r.sandbox.as_str().to_string());
    } else {
        // Assumed: `resume` does not accept `-s`; pin sandbox via config override.
        push_config(
            &mut args,
            "sandbox_mode",
            &toml_basic_string(r.sandbox.as_str()),
        );
    }

    if !interactive {
        // Assumed: `-a` is top-level only; use the config-key form for both
        // `exec` and `resume`.
        push_config(
            &mut args,
            "approval_policy",
            &toml_basic_string(d.approval.as_str()),
        );
    }

    // Assumed: `resume` does not accept `-C`; cwd is the process's own.
    if interactive || one_shot_fresh {
        args.push("-C".into());
        args.push(c.cwd.display().to_string());
    }

    if !interactive {
        args.push("--skip-git-repo-check".into());
        if let Some(out) = &d.output_file {
            args.push("-o".into());
            args.push(out.display().to_string());
        }
    }

    args.extend(c.extra_args.iter().cloned());

    if !interactive {
        args.push("-".into());
    }

    Ok(Invocation {
        program: binary_in(d.env, "CCC_CODEX_BIN", "codex"),
        args,
        env_set: Vec::new(),
        env_remove: Vec::new(),
        cwd: c.cwd.clone(),
        stdin: c.mode.stdin(),
    })
}

/// How many non-JSON stdout lines are kept for diagnostics.
const UNPARSABLE_KEPT: usize = 5;

/// Parses the `--json` event stream. `output_file` is the content of the `-o`
/// file when it exists; it wins over the streamed message if the two differ.
fn parse(
    stdout: &str,
    output_file: Option<&str>,
    stderr: &str,
    exit_code: Option<i32>,
) -> Result<Outcome, HarnessError> {
    let mut thread_id: Option<String> = None;
    let mut message: Option<String> = None;
    let mut usage = serde_json::Value::Null;
    let mut turn_failure: Option<String> = None;
    let mut top_level_error: Option<String> = None;
    let mut completed = false;
    let mut warnings: Vec<String> = Vec::new();
    let mut unparsable: Vec<&str> = Vec::new();

    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(ev) = serde_json::from_str::<serde_json::Value>(line) else {
            // Codex writes some fatal conditions to stdout as plain text; keep
            // the last few so a failure is not reported with no detail.
            unparsable.push(line);
            if unparsable.len() > UNPARSABLE_KEPT {
                unparsable.remove(0);
            }
            continue;
        };
        match ev.get("type").and_then(|v| v.as_str()).unwrap_or("") {
            "thread.started" => {
                thread_id = thread_started_id(line);
            }
            "item.completed" => {
                let item = ev.get("item").unwrap_or(&serde_json::Value::Null);
                match item.get("type").and_then(|v| v.as_str()).unwrap_or("") {
                    "agent_message" => {
                        message = item
                            .get("text")
                            .and_then(|v| v.as_str())
                            .map(|s| s.trim_end().to_string());
                    }
                    // A streamed error item is only a warning; a failed turn
                    // reports itself separately.
                    "error" => warnings.push(event_message(item)),
                    _ => {}
                }
            }
            "turn.completed" => {
                completed = true;
                if let Some(u) = ev.get("usage") {
                    usage = u.clone();
                }
            }
            "turn.failed" => turn_failure = Some(event_message(&ev)),
            "error" => top_level_error = Some(event_message(&ev)),
            _ => {}
        }
    }

    let file_message = output_file.map(str::trim_end).filter(|s| !s.is_empty());
    let result = match (file_message, message.as_deref()) {
        (Some(f), Some(m)) if f != m => {
            warnings
                .push("codex -o file and the streamed agent_message differ; using the file".into());
            f.to_string()
        }
        (Some(f), _) => f.to_string(),
        (None, Some(m)) => m.to_string(),
        (None, None) => String::new(),
    };

    let native = json!({ "usage": usage });

    // A top-level error that the turn recovered from (retry or backoff notices
    // look like this) is a warning; the paid-for message is still the answer.
    // Only a streamed message proves the turn produced one.
    let recovered = completed && message.is_some();
    let failure = turn_failure.or_else(|| top_level_error.clone().filter(|_| !recovered));
    if failure.is_none() {
        if let Some(e) = top_level_error {
            warnings.push(e);
        }
    }

    if let Some(message) = failure {
        return Err(HarnessError::Reported {
            session_id: thread_id,
            message,
            native,
            warnings,
        });
    }
    if result.is_empty() {
        let mut detail = tail(stderr, 5);
        if !unparsable.is_empty() {
            detail = format!("{detail}; stdout: {}", unparsable.join(" / "));
        }
        if let Some(code) = exit_code.filter(|c| *c != 0) {
            detail = format!("exit {code}; {detail}");
        }
        return Err(
            HarnessError::unusable(format!("codex produced no agent message ({detail})"))
                .with_session_id(thread_id),
        );
    }
    let Some(thread_id) = thread_id else {
        return Err(HarnessError::unusable(
            "codex produced no thread id".to_string(),
        ));
    };
    if !completed {
        warnings.push("codex ended without a turn.completed event".into());
    }
    if !unparsable.is_empty() {
        warnings.push(format!(
            "codex wrote non-JSON lines to stdout: {}",
            unparsable.join(" / ")
        ));
    }

    Ok(Outcome {
        session_id: thread_id,
        result,
        native,
        warnings,
    })
}

/// The human-readable text of an error-ish event, wherever it hides.
fn event_message(ev: &serde_json::Value) -> String {
    for p in ["/error/message", "/message", "/error", "/text"] {
        if let Some(s) = ev.pointer(p).and_then(|v| v.as_str()) {
            return s.to_string();
        }
    }
    ev.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::Sandbox;
    use crate::config::{Effort, Harness};
    use crate::harness::modes::{interactive, one_shot, resumed, run_mode};
    use crate::harness::Mode;

    fn resolved() -> Resolved {
        Resolved {
            agent: Some("scribe".into()),
            description: "Writes files".into(),
            harness: Harness::Codex,
            model: "gpt-5.6-luna".into(),
            effort: Some(Effort::Low),
            tools: None,
            sandbox: Sandbox::ReadOnly,
            body: "Write the file.".into(),
        }
    }

    fn common(mode: Mode) -> Common {
        Common {
            cwd: PathBuf::from("/srv/work"),
            extra_args: vec![],
            mode,
            preset_session_id: None,
        }
    }

    #[test]
    fn launch_notify_is_before_extras_and_one_shot_modes_have_no_callback_or_session_flag() {
        let mut launch = common(interactive("/tmp/ccc tool", Origin::Fresh));
        launch.preset_session_id = Some("ignored".into());
        launch.extra_args = vec!["--extra".into()];
        let invocation = driver().build(&resolved(), &launch).unwrap();
        assert_eq!(
            invocation.config_override("notify"),
            Some("[\"/tmp/ccc tool\", \"hook\", \"codex-notify\"]")
        );
        assert!(
            invocation
                .args
                .iter()
                .position(|arg| arg == "notify=[\"/tmp/ccc tool\", \"hook\", \"codex-notify\"]")
                .unwrap()
                < invocation
                    .args
                    .iter()
                    .position(|arg| arg == "--extra")
                    .unwrap()
        );
        let run = driver().build(&resolved(), &common(run_mode())).unwrap();
        assert!(!run.has_arg("notify"));
        assert!(!run.has_arg("--session-id"));
    }

    /// A stand-in for the private path `new_output_file` mints.
    const OUT_PATH: &str = "/run/state/ccc/tmp/out-1.txt";

    /// An environment naming nothing, so the driver falls back to `codex`.
    struct NoEnv;

    impl EnvSource for NoEnv {
        fn get(&self, _key: &str) -> Option<String> {
            None
        }

        fn vars_with_prefix(&self, _prefix: &str) -> Vec<(String, String)> {
            Vec::new()
        }
    }

    fn driver() -> CodexDriver<'static> {
        CodexDriver {
            approval: Approval::Never,
            output_file: Some(PathBuf::from(OUT_PATH)),
            env: &NoEnv,
        }
    }

    #[test]
    fn early_thread_ids_share_final_parser_shapes() {
        let d = driver();
        assert_eq!(
            d.early_session_id(r#"{"type":"thread.started","thread_id":"t9"}"#)
                .as_deref(),
            Some("t9")
        );
        assert_eq!(
            d.early_session_id(r#"{"type":"thread.started","thread":{"id":"t9"}}"#)
                .as_deref(),
            Some("t9")
        );
        assert_eq!(d.early_session_id(r#"{"type":"turn.completed"}"#), None);
        assert_eq!(d.early_session_id("not json"), None);
        assert_eq!(
            d.early_session_id(r#"{"type":"thread.started","thread_id":7}"#),
            None
        );
    }

    /// A state home under a scratch directory, so the minted `-o` files land
    /// there instead of the developer's own.
    fn state_env(dir: &crate::testutil::TempDir) -> std::collections::BTreeMap<String, String> {
        [(
            "XDG_STATE_HOME".to_string(),
            dir.path().display().to_string(),
        )]
        .into_iter()
        .collect()
    }

    fn captured(stdout: &str, output_file: Option<&str>, stderr: &str, exit: i32) -> Captured {
        Captured {
            stdout: stdout.to_string(),
            stderr: stderr.to_string(),
            output_file: output_file.map(String::from),
            exit_code: Some(exit),
        }
    }

    #[test]
    fn basic_string_escapes() {
        assert_eq!(toml_basic_string("plain"), "\"plain\"");
        assert_eq!(
            toml_basic_string("a\"b\\c\nd\te"),
            "\"a\\\"b\\\\c\\nd\\te\""
        );
        assert_eq!(toml_basic_string("\r\u{8}\u{c}"), "\"\\r\\b\\f\"");
        assert_eq!(toml_basic_string("\u{1}"), "\"\\u0001\"");
        assert_eq!(toml_basic_string("héllo ✂"), "\"héllo ✂\"");
        // Round-trips through a TOML parser as the original text.
        let body = "line1\n\"quoted\"\tand \\ back\u{7f}";
        let doc = format!("v = {}", toml_basic_string(body));
        let parsed: toml::Value = toml::from_str(&doc).unwrap();
        assert_eq!(parsed["v"].as_str().unwrap(), body);
    }

    #[test]
    fn run_argv() {
        let i = driver().build(&resolved(), &common(run_mode())).unwrap();
        assert_eq!(i.program, "codex");
        assert_eq!(&i.args[0..2], &["exec", "--json"]);
        assert_eq!(i.value_of("-m"), Some("gpt-5.6-luna"));
        assert_eq!(i.config_override("model_reasoning_effort"), Some("\"low\""));
        assert_eq!(
            i.config_override("developer_instructions"),
            Some("\"Write the file.\"")
        );
        assert_eq!(i.value_of("-s"), Some("read-only"));
        assert_eq!(i.config_override("approval_policy"), Some("\"never\""));
        assert_eq!(i.value_of("-C"), Some("/srv/work"));
        assert!(i.has_arg("--skip-git-repo-check"));
        assert_eq!(i.value_of("-o"), Some(OUT_PATH));
        assert_eq!(i.args.last().unwrap(), "-");
        // `-a` is never passed to `exec`.
        assert!(!i.has_arg("-a") && !i.has_arg("--ask-for-approval"));
        assert_eq!(i.stdin.as_deref(), Some("go"));
    }

    #[test]
    fn resume_argv_restates_everything_as_config_overrides() {
        let i = driver()
            .build(&resolved(), &common(one_shot(resumed("t-1234", false))))
            .unwrap();
        assert_eq!(&i.args[0..4], &["exec", "resume", "t-1234", "--json"]);
        assert_eq!(i.config_override("model"), Some("\"gpt-5.6-luna\""));
        assert_eq!(i.config_override("sandbox_mode"), Some("\"read-only\""));
        assert_eq!(i.config_override("approval_policy"), Some("\"never\""));
        assert!(!i.has_arg("-s"));
        assert!(!i.has_arg("-C"));
        assert!(!i.has_arg("-m"));
        // The cwd is the process's own, since `resume` cannot be told one.
        assert_eq!(i.cwd, PathBuf::from("/srv/work"));
    }

    #[test]
    fn launch_argv_is_interactive() {
        let i = driver()
            .build(&resolved(), &common(interactive("/ccc", Origin::Fresh)))
            .unwrap();
        assert!(!i.has_arg("exec"));
        assert!(!i.has_arg("--json"));
        assert!(!i.has_arg("-o"));
        assert_eq!(i.config_override("approval_policy"), None);
        assert_eq!(i.value_of("-m"), Some("gpt-5.6-luna"));
        assert_eq!(i.value_of("-s"), Some("read-only"));
        assert_eq!(i.value_of("-C"), Some("/srv/work"));
        assert_ne!(i.args.last().unwrap(), "-");
        assert_eq!(i.stdin, None);
    }

    #[test]
    fn resume_has_no_launch_callback_or_preset_session() {
        let mut c = common(one_shot(resumed("thread-1", false)));
        c.preset_session_id = Some("ignored".into());
        let i = driver().build(&resolved(), &c).unwrap();
        assert!(!i.has_arg("notify"));
        assert!(!i.has_arg("--session-id"));
    }

    #[test]
    fn six_cells_pick_the_subcommand_and_flag_set() {
        let id = "t-1234";
        let cells: [(Mode, &[&str]); 6] = [
            (interactive("/ccc", Origin::Fresh), &["-m"]),
            (
                interactive("/ccc", resumed(id, false)),
                &["resume", id, "--include-non-interactive", "-m"],
            ),
            (interactive("/ccc", resumed(id, true)), &["fork", id, "-m"]),
            (one_shot(Origin::Fresh), &["exec", "--json"]),
            (
                one_shot(resumed(id, false)),
                &["exec", "resume", id, "--json"],
            ),
            (one_shot(resumed(id, true)), &["exec", "fork", id, "--json"]),
        ];
        for (mode, head) in cells {
            let is_interactive = mode.is_launch();
            let one_shot_resumed = !is_interactive && mode.origin != Origin::Fresh;
            let mut c = common(mode.clone());
            c.extra_args = vec!["--extra".into()];
            let i = driver().build(&resolved(), &c).unwrap();

            assert_eq!(&i.args[..head.len()], head, "{mode:?}");
            assert_eq!(
                i.has_arg("--include-non-interactive"),
                is_interactive && mode.continues().is_some(),
                "{mode:?}"
            );
            assert_eq!(i.has_arg("-s"), !one_shot_resumed, "{mode:?}");
            assert_eq!(i.has_arg("-C"), !one_shot_resumed, "{mode:?}");
            assert_eq!(
                i.config_override("sandbox_mode").is_some(),
                !i.has_arg("-s"),
                "{mode:?}"
            );
            assert_eq!(
                i.config_override("model").is_some(),
                one_shot_resumed,
                "{mode:?}"
            );
            assert_eq!(
                i.config_override("approval_policy").is_some(),
                !is_interactive,
                "{mode:?}"
            );
            assert_eq!(
                i.has_arg("--skip-git-repo-check"),
                !is_interactive,
                "{mode:?}"
            );
            assert_eq!(i.has_arg("-o"), !is_interactive, "{mode:?}");
            assert_eq!(
                i.config_override("notify").is_some(),
                is_interactive,
                "{mode:?}"
            );
            let tail: &[&str] = if is_interactive {
                &["--extra"]
            } else {
                &["--extra", "-"]
            };
            assert_eq!(&i.args[i.args.len() - tail.len()..], tail, "{mode:?}");
        }
    }

    #[test]
    fn effort_and_extra_args() {
        let mut r = resolved();
        r.effort = None;
        let mut c = common(run_mode());
        c.extra_args = vec!["--include-plan-tool".into()];
        let i = driver().build(&r, &c).unwrap();
        assert_eq!(i.config_override("model_reasoning_effort"), None);
        assert_eq!(i.args[i.args.len() - 2], "--include-plan-tool");
    }

    const SUCCESS: &str = r#"
{"type":"thread.started","thread_id":"0199aa-bb"}
{"type":"turn.started"}
{"type":"item.completed","item":{"type":"agent_message","text":"DEVINSTR-OK done\n"}}
{"type":"turn.completed","usage":{"input_tokens":812,"output_tokens":44}}
"#;

    #[test]
    fn parses_a_successful_stream() {
        let out = driver().parse(&captured(SUCCESS, None, "", 0)).unwrap();
        assert_eq!(out.session_id, "0199aa-bb");
        assert_eq!(out.result, "DEVINSTR-OK done");
        assert_eq!(out.native["usage"]["input_tokens"], json!(812));
        assert!(out.warnings.is_empty());
    }

    #[test]
    fn the_output_file_wins_over_the_streamed_message() {
        let out = driver()
            .parse(&captured(SUCCESS, Some("final text\n"), "", 0))
            .unwrap();
        assert_eq!(out.result, "final text");
        assert_eq!(out.warnings.len(), 1);
        // An empty `-o` file falls back to the stream.
        let out = driver()
            .parse(&captured(SUCCESS, Some("\n"), "", 0))
            .unwrap();
        assert_eq!(out.result, "DEVINSTR-OK done");
        assert!(out.warnings.is_empty());
    }

    #[test]
    fn a_streamed_error_item_before_completion_is_only_a_warning() {
        let stream = r#"
{"type":"thread.started","thread_id":"t9"}
{"type":"item.completed","item":{"type":"error","message":"model mismatch; using default"}}
{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}
{"type":"turn.completed","usage":{}}
"#;
        let out = driver().parse(&captured(stream, None, "", 0)).unwrap();
        assert_eq!(out.result, "ok");
        assert_eq!(out.warnings, vec!["model mismatch; using default"]);
    }

    #[test]
    fn a_top_level_error_the_turn_recovered_from_is_only_a_warning() {
        let stream = r#"
{"type":"thread.started","thread_id":"t9"}
{"type":"error","message":"stream disconnected; retrying"}
{"type":"item.completed","item":{"type":"agent_message","text":"the answer"}}
{"type":"turn.completed","usage":{}}
"#;
        let out = driver().parse(&captured(stream, None, "", 0)).unwrap();
        assert_eq!(out.result, "the answer");
        assert_eq!(out.warnings, vec!["stream disconnected; retrying"]);

        // Without a completed turn it is still a failure.
        let unfinished = r#"
{"type":"thread.started","thread_id":"t9"}
{"type":"error","message":"stream disconnected; retrying"}
{"type":"item.completed","item":{"type":"agent_message","text":"partial"}}
"#;
        let err = driver()
            .parse(&captured(unfinished, None, "", 1))
            .unwrap_err();
        assert_eq!(err.exit_code(), 1);
        assert_eq!(err.message(), "stream disconnected; retrying");
    }

    #[test]
    fn a_failed_turn_is_exit_one() {
        let stream = r#"
{"type":"thread.started","thread_id":"t9"}
{"type":"error","message":"unknown model gpt-6-luna"}
{"type":"turn.failed","error":{"message":"unknown model gpt-6-luna"}}
"#;
        let err = driver().parse(&captured(stream, None, "", 1)).unwrap_err();
        assert_eq!(err.exit_code(), 1);
        assert_eq!(err.message(), "unknown model gpt-6-luna");
        assert_eq!(err.session_id(), Some("t9"));
    }

    #[test]
    fn a_failed_turn_keeps_the_warnings_that_explain_it() {
        let stream = r#"
{"type":"thread.started","thread_id":"t9"}
{"type":"item.completed","item":{"type":"error","message":"model mismatch; using default"}}
{"type":"item.completed","item":{"type":"error","message":"sandbox denied write to /etc"}}
{"type":"turn.failed","error":{"message":"the task could not be completed"}}
"#;
        let err = driver().parse(&captured(stream, None, "", 1)).unwrap_err();
        assert_eq!(err.message(), "the task could not be completed");
        assert_eq!(
            err.warnings(),
            [
                "model mismatch; using default",
                "sandbox denied write to /etc"
            ]
        );
    }

    #[test]
    fn no_agent_message_is_exit_three() {
        let stream = "{\"type\":\"thread.started\",\"thread_id\":\"t9\"}\ngarbage line\n";
        let err = driver()
            .parse(&captured(stream, None, "codex: killed\n", 137))
            .unwrap_err();
        assert_eq!(err.exit_code(), 3);
        assert!(err.message().contains("exit 137"), "{}", err.message());
        assert!(err.message().contains("codex: killed"));
    }

    #[test]
    fn a_message_without_thread_started_is_exit_three() {
        let stream = r#"{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}"#;
        assert_eq!(
            driver()
                .parse(&captured(stream, None, "", 0))
                .unwrap_err()
                .exit_code(),
            3
        );
    }

    #[test]
    fn an_output_file_alone_does_not_make_a_failed_turn_look_recovered() {
        let stream = r#"
{"type":"thread.started","thread_id":"t9"}
{"type":"error","message":"context window exceeded"}
{"type":"turn.completed","usage":{}}
"#;
        let err = driver()
            .parse(&captured(stream, Some("left over from before\n"), "", 1))
            .unwrap_err();
        assert_eq!(err.exit_code(), 1);
        assert_eq!(err.message(), "context window exceeded");
    }

    #[test]
    fn a_private_output_file_is_created_fresh_each_time() {
        let dir = crate::testutil::TempDir::new("codex-out");
        let env = state_env(&dir);
        let a = new_output_file(&env).unwrap();
        let b = new_output_file(&env).unwrap();
        assert!(a.starts_with(dir.path()), "{}", a.display());
        assert_ne!(a, b);
        assert_eq!(std::fs::read_to_string(&a).unwrap(), "");
        // The path is minted, never adopted: creating it again is refused.
        assert!(std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&a)
            .is_err());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(a.parent().unwrap())
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o700, "{mode:o}");
        }
        let _ = std::fs::remove_file(&a);
        let _ = std::fs::remove_file(&b);
    }

    #[test]
    fn reading_the_output_file_removes_it() {
        let dir = crate::testutil::TempDir::new("codex-take");
        let p = new_output_file(&state_env(&dir)).unwrap();
        std::fs::write(&p, "the final message\n").unwrap();
        assert_eq!(take_output_file(&p).as_deref(), Some("the final message\n"));
        assert!(!p.exists());
        assert_eq!(take_output_file(&p), None);
    }

    #[test]
    fn non_json_stdout_is_kept_for_the_diagnostic() {
        let stream = "codex: failed to parse config\nRUST_BACKTRACE panic\n";
        let err = driver().parse(&captured(stream, None, "", 1)).unwrap_err();
        assert_eq!(err.exit_code(), 3);
        assert!(
            err.message().contains("failed to parse config"),
            "{}",
            err.message()
        );
        assert!(
            err.message().contains("RUST_BACKTRACE panic"),
            "{}",
            err.message()
        );
    }

    #[test]
    fn an_error_message_is_found_in_every_shape_codex_uses() {
        for (event, expected) in [
            (r#"{"error":{"message":"nested"}}"#, "nested"),
            (r#"{"message":"flat"}"#, "flat"),
            (r#"{"error":"a string"}"#, "a string"),
            (r#"{"text":"as text"}"#, "as text"),
        ] {
            let ev: serde_json::Value = serde_json::from_str(event).unwrap();
            assert_eq!(event_message(&ev), expected, "{event}");
        }
        let ev: serde_json::Value = serde_json::from_str(r#"{"detail":"boom"}"#).unwrap();
        assert_eq!(event_message(&ev), r#"{"detail":"boom"}"#);
    }

    #[test]
    fn a_nested_thread_id_is_accepted() {
        let stream = r#"
{"type":"thread.started","thread":{"id":"t9"}}
{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}
{"type":"turn.completed","usage":{}}
"#;
        let out = driver().parse(&captured(stream, None, "", 0)).unwrap();
        assert_eq!(out.session_id, "t9");
    }

    #[test]
    fn a_missing_turn_completed_warns_but_succeeds() {
        let stream = r#"
{"type":"thread.started","thread_id":"t9"}
{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}
"#;
        let out = driver().parse(&captured(stream, None, "", 0)).unwrap();
        assert_eq!(out.warnings.len(), 1);
    }
}
