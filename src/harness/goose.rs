//! The Goose driver: `goose run` headless and `goose session` at a terminal,
//! with every Goose-provided extension off and ccc's tool servers as the only
//! extensions.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use crate::config::{EnvSource, GooseConfig};
use crate::goose_policy::{GoosePolicy, GooseTool};
use crate::harness::{
    binary_in, check_arg_len, tail, Captured, Common, Driver, HarnessError, Invocation, Origin,
    Outcome,
};
use crate::resolve::Resolved;

/// Inherited variables removed from Goose's environment besides every
/// `GOOSE_*`: Goose reads any config key from its upper-cased name first, and
/// these would add extensions, enable plugins, map recipe commands, change the
/// hint files or turn on confirmation prompts.
const STRIPPED: [&str; 5] = [
    "EXTENSIONS",
    "PLUGINS",
    "SLASH_COMMANDS",
    "CONTEXT_FILE_NAMES",
    "SECURITY_PROMPT_ENABLED",
];

/// Langfuse's credential variables. Goose uploads traces to Langfuse whenever
/// a public/secret pair is set and has no other off switch, so they are removed.
const LANGFUSE_KEYS: [&str; 4] = [
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_INIT_PROJECT_PUBLIC_KEY",
    "LANGFUSE_INIT_PROJECT_SECRET_KEY",
];

/// Set after the config env, so no config value replaces them: PostHog usage
/// events off, and every OpenTelemetry signal off ahead of any OTLP endpoint
/// in the environment or a Goose config file.
const TELEMETRY_OFF: [(&str, &str); 2] =
    [("GOOSE_TELEMETRY_OFF", "1"), ("OTEL_SDK_DISABLED", "true")];

/// The HTTPS proxy variables, both set to [`HTTPS_SINK`] for Goose.
const HTTPS_PROXY_KEYS: [&str; 2] = ["HTTPS_PROXY", "https_proxy"];

/// Goose's HTTPS proxy: the discard port on loopback, where nothing normally
/// listens. Goose fetches the models.dev catalog on every start with no off
/// switch, and falls back to its compiled-in catalog when the fetch fails.
const HTTPS_SINK: &str = "http://127.0.0.1:9";

/// Inherited proxy settings removed from Goose's environment: a `NO_PROXY`
/// would route around the sink, and an `HTTP_PROXY` or `ALL_PROXY` would send
/// model traffic through a proxy nobody chose for Goose.
const PROXY_STRIPPED: [&str; 6] = [
    "NO_PROXY",
    "no_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "ALL_PROXY",
    "all_proxy",
];

/// Carries ccc's own values of [`host_env_keys`] through Goose to the ccc
/// processes Goose starts, which put them back with [`restore_host_env`].
pub const HOST_ENV_VAR: &str = "CCC_GOOSE_HOST_ENV";

/// The variables ccc changes for Goose that its ccc children need as the host
/// has them.
fn host_env_keys() -> impl Iterator<Item = &'static str> {
    std::iter::once("HOME")
        .chain(HTTPS_PROXY_KEYS)
        .chain(PROXY_STRIPPED)
}

/// A variable that switches Goose telemetry (PostHog, OpenTelemetry export or
/// Langfuse upload).
pub fn is_telemetry_var(key: &str) -> bool {
    TELEMETRY_OFF.iter().any(|(k, _)| *k == key)
        || key == "GOOSE_TELEMETRY_ENABLED"
        || key.starts_with("LANGFUSE_")
}

/// A variable ccc sets for Goose's isolation: its empty home, its HTTPS sink,
/// and the host values carried to the ccc processes Goose starts.
pub fn is_isolation_var(key: &str) -> bool {
    key == "HOME" || HTTPS_PROXY_KEYS.contains(&key) || key == HOST_ENV_VAR
}

/// The binary a launch's hooks run; set only for `goose session`.
pub const HOOK_VAR: &str = "CCC_GOOSE_HOOK";
/// The ccc session id of a launch, for its `SessionStart` hook.
pub const SESSION_VAR: &str = "CCC_GOOSE_SESSION";

/// Goose's reply when `--max-turns` runs out. Goose ends such a turn with exit
/// 0 and this question as the answer; a string match is the only signal.
const MAX_TURNS_TEXT: &str = "I've reached the maximum number of actions I can do without user input. Would you like me to continue?";

/// The Goose harness for one invocation.
pub struct GooseDriver<'a> {
    pub cfg: &'a GooseConfig,
    pub policy: &'a GoosePolicy,
    /// The ccc binary the extensions and hooks run.
    pub self_exe: PathBuf,
    /// The ccc session id this invocation runs under, [`session_id`] of the
    /// `Common` passed to `build`. `build` names the session with it and
    /// `parse` reports it.
    pub session: String,
    pub env: &'a dyn EnvSource,
}

impl Driver for GooseDriver<'_> {
    fn build(&self, r: &Resolved, c: &Common) -> Result<Invocation> {
        build(r, c, self)
    }

    fn parse(&self, captured: &Captured) -> Result<Outcome, HarnessError> {
        parse(
            &captured.stdout,
            &captured.stderr,
            captured.exit_code,
            &self.session,
        )
    }
}

/// The directory every Goose process runs in: inside the Goose root, so
/// nothing a model writes is where Goose discovers plugins, settings, skills
/// or hints in its process cwd.
pub fn work_dir(root: &Path) -> PathBuf {
    root.join("work")
}

/// Goose's `HOME`: an empty directory in the Goose root, so Goose loads no
/// skills, agents or settings from the user's home.
pub fn home_dir(root: &Path) -> PathBuf {
    root.join("home")
}

/// The hex-encoded JSON object of `env`'s values of [`host_env_keys`], `null`
/// for an absent one.
fn host_env_token(env: &dyn EnvSource) -> String {
    let object: serde_json::Map<String, Value> = host_env_keys()
        .map(|k| (k.to_string(), env.get(k).map_or(Value::Null, Value::String)))
        .collect();
    crate::goose_policy::hex_encode(
        &serde_json::to_vec(&Value::Object(object)).expect("a JSON object serialises"),
    )
}

/// One environment change: set the key to the value, or remove it on `None`.
pub type EnvOp = (String, Option<String>);

/// The environment changes that restore the host's values from
/// [`HOST_ENV_VAR`] in `env`: set a key to its string, remove a `null` key,
/// leave a key the object lacks, and remove the variable itself. `None` when
/// the variable is absent.
pub fn host_env_ops(env: &dyn EnvSource) -> Result<Option<Vec<EnvOp>>> {
    let Some(token) = env.get(HOST_ENV_VAR) else {
        return Ok(None);
    };
    let decode = || -> Result<Vec<EnvOp>> {
        let bytes = crate::goose_policy::hex_decode(&token)?;
        let Value::Object(object) = serde_json::from_slice(&bytes)? else {
            bail!("not a JSON object");
        };
        let mut ops = Vec::new();
        for (key, value) in object {
            if !host_env_keys().any(|k| k == key) {
                bail!("unexpected key {key:?}");
            }
            match value {
                Value::Null => ops.push((key, None)),
                Value::String(v) if !v.contains('\0') => ops.push((key, Some(v))),
                _ => bail!("{key} is not a string or null"),
            }
        }
        ops.push((HOST_ENV_VAR.to_string(), None));
        Ok(ops)
    };
    decode()
        .map(Some)
        .with_context(|| format!("{HOST_ENV_VAR} does not decode"))
}

/// Puts back the host's values a Goose parent carried in [`HOST_ENV_VAR`].
/// Called first thing in a ccc process Goose starts, while it has one thread.
pub fn restore_host_env() -> Result<()> {
    for (key, value) in host_env_ops(&crate::config::ProcessEnv)?.unwrap_or_default() {
        match value {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }
    Ok(())
}

/// The plugin file holding the launch hooks.
pub fn plugin_file(root: &Path) -> PathBuf {
    root.join(".agents/plugins/ccc/hooks/hooks.json")
}

/// The `sh -c` command a hook rule runs. Without `CCC_GOOSE_HOOK`, which only
/// a launch sets, it drains the payload and allows.
pub(crate) fn hook_command(kind: &str) -> String {
    format!(
        "[ -n \"${HOOK_VAR}\" ] || {{ cat >/dev/null; exit 0; }}; exec \"${HOOK_VAR}\" hook {kind}"
    )
}

/// The exact content of [`plugin_file`].
pub fn plugin_json() -> String {
    let rule =
        |kind: &str| json!([{ "hooks": [{ "type": "command", "command": hook_command(kind) }] }]);
    let doc = json!({
        "hooks": {
            "SessionStart": rule("goose-session-start"),
            "Stop": rule("goose-stop"),
        }
    });
    let mut text = serde_json::to_string_pretty(&doc).expect("a fixed document serialises");
    text.push('\n');
    text
}

/// Creates the Goose root (0700, when missing), its work and home directories
/// (0700), and writes the plugin file when it is missing or differs. Returns
/// the work directory.
pub fn prepare_root(root: &Path) -> Result<PathBuf> {
    // An existing root keeps the mode its operator gave it.
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(root)
        .with_context(|| format!("creating {}", root.display()))?;
    let work = work_dir(root);
    crate::fsutil::ensure_private_dir(&work)?;
    crate::fsutil::ensure_private_dir(&home_dir(root))?;

    let plugin = plugin_file(root);
    let wanted = plugin_json();
    if std::fs::read_to_string(&plugin).ok().as_deref() != Some(wanted.as_str()) {
        let dir = plugin.parent().expect("the plugin file has a parent");
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        crate::fsutil::write_atomically(&plugin, &wanted)?;
    }
    Ok(work)
}

/// The `--system` text: a preamble naming the real working directory, since
/// Goose runs in [`work_dir`] and reports that, then the agent body.
fn system_prompt(cwd: &Path, body: &str) -> String {
    format!(
        "Your working directory is {}. The working directory Goose reports is a placeholder; ignore it.\n\n{body}",
        cwd.display()
    )
}

/// Refuses a `self_exe` that Goose's `--with-extension` tokenizer would
/// split or reinterpret: it splits on whitespace, honours `"` and `'`, and
/// takes a leading token holding `=` as an env assignment.
fn check_self_exe(self_exe: &Path) -> Result<&str> {
    let Some(text) = self_exe.to_str() else {
        bail!(
            "the ccc binary path {} is not UTF-8; Goose cannot start it as an extension. Install ccc at a path without such characters",
            self_exe.display()
        );
    };
    if text
        .chars()
        .any(|c| c.is_whitespace() || matches!(c, '"' | '\'' | '='))
    {
        bail!(
            "the ccc binary path {text:?} contains whitespace, a quote or `=`, which Goose's extension command parser splits on; install ccc at a path without those characters"
        );
    }
    Ok(text)
}

/// One `--with-extension` value per tool-server role the policy needs: `ccc`
/// for the subagent tools, `box` for the sandboxed file and shell tools.
fn extensions(agent: &str, self_exe: &str, policy: &GoosePolicy) -> Result<Vec<String>> {
    let hex = policy.to_hex();
    let mut out = Vec::new();
    if policy.tools.contains(&GooseTool::Ccc) {
        out.push(format!(
            "ccc:{self_exe} goose-tools --role host --policy {hex}"
        ));
    }
    if policy.needs_box() {
        out.push(format!(
            "box:{self_exe} goose-tools --role box --policy {hex}"
        ));
    }
    for ext in &out {
        check_arg_len(agent, "--with-extension", ext)?;
    }
    Ok(out)
}

/// The ccc session id a Goose invocation of `c` runs under: the preset id when
/// fresh, the resumed id otherwise. Goose's JSON output carries no session id,
/// so ccc names the session itself with `-n` and later resumes it by that
/// name. A fork is refused, and so is a fresh run with no preset id.
pub fn session_id(c: &Common) -> Result<String> {
    match &c.mode.origin {
        Origin::Resume { fork: true, .. } => {
            bail!("goose sessions cannot be forked: Goose has no fork that keeps ccc's session id")
        }
        Origin::Resume { id, fork: false } => Ok(id.clone()),
        Origin::Fresh => match &c.preset_session_id {
            Some(id) => Ok(id.clone()),
            None => bail!("a fresh goose run needs a preset session id"),
        },
    }
}

/// Builds the argv, environment, cwd and stdin for one Goose invocation.
fn build(r: &Resolved, c: &Common, d: &GooseDriver) -> Result<Invocation> {
    let agent = r.label();
    let interactive = c.mode.is_launch();
    let session = d.session.as_str();
    let fresh = c.mode.origin == Origin::Fresh;
    let self_exe = check_self_exe(&d.self_exe)?;
    let system = system_prompt(&c.cwd, &r.body);
    check_arg_len(agent, "--system", &system)?;

    let mut args: Vec<String> = Vec::new();
    if interactive {
        args.push("session".into());
    } else {
        args.extend(["run", "-q", "--output-format", "json"].map(String::from));
    }
    if fresh {
        args.push("--no-profile".into());
    } else {
        args.push("--resume".into());
    }
    args.push("--provider".into());
    args.push(d.cfg.provider.clone());
    args.push("--model".into());
    args.push(r.model.clone());
    args.push("-n".into());
    args.push(session.to_string());
    // A resumed session reloads the extensions saved with it; naming them
    // again would add a second extension under the same name.
    if fresh {
        for ext in extensions(agent, self_exe, d.policy)? {
            args.push("--with-extension".into());
            args.push(ext);
        }
    }
    args.push("--system".into());
    args.push(system);
    if let Some(n) = d.cfg.max_turns {
        args.push("--max-turns".into());
        args.push(n.to_string());
    }
    args.extend(c.extra_args.iter().cloned());
    if !interactive {
        args.extend(["-i", "-"].map(String::from));
    }

    let root = d.cfg.root_in(d.env);
    let mut env_set: Vec<(String, String)> = vec![
        ("GOOSE_PATH_ROOT".into(), root.display().to_string()),
        ("GOOSE_MODE".into(), "auto".into()),
        ("GOOSE_DISABLE_KEYRING".into(), "1".into()),
    ];
    if interactive {
        env_set.push((HOOK_VAR.into(), self_exe.to_string()));
        env_set.push((SESSION_VAR.into(), session.to_string()));
    }
    for (k, v) in d.cfg.env_for(&r.model) {
        set_env(&mut env_set, k, v);
    }
    for (k, v) in TELEMETRY_OFF {
        set_env(&mut env_set, k.into(), v.into());
    }
    set_env(
        &mut env_set,
        "HOME".into(),
        home_dir(&root).display().to_string(),
    );
    for k in HTTPS_PROXY_KEYS {
        set_env(&mut env_set, k.into(), HTTPS_SINK.into());
    }
    set_env(&mut env_set, HOST_ENV_VAR.into(), host_env_token(d.env));

    let mut env_remove: Vec<String> = d
        .env
        .vars_with_prefix("GOOSE_")
        .into_iter()
        .map(|(k, _)| k)
        .chain(STRIPPED.iter().map(|k| k.to_string()))
        .chain(LANGFUSE_KEYS.iter().map(|k| k.to_string()))
        .chain(PROXY_STRIPPED.iter().map(|k| k.to_string()))
        .collect();
    if !interactive {
        env_remove.extend([HOOK_VAR, SESSION_VAR].map(String::from));
    }
    // A key in both lists would be removed, so nothing ccc sets may be removed.
    env_remove.retain(|k| !env_set.iter().any(|(s, _)| s == k));
    env_remove.sort();
    env_remove.dedup();

    Ok(Invocation {
        program: binary_in(d.env, "CCC_GOOSE_BIN", "goose"),
        args,
        env_set,
        env_remove,
        cwd: work_dir(&root),
        stdin: c.mode.stdin(),
    })
}

/// Sets `key`, replacing an earlier value for it.
fn set_env(env: &mut Vec<(String, String)>, key: String, value: String) {
    match env.iter_mut().find(|(k, _)| *k == key) {
        Some(slot) => slot.1 = value,
        None => env.push((key, value)),
    }
}

/// The last top-level JSON object in `stdout`, parsed from the last line that
/// is exactly `{`, and the text before that line.
fn last_object(stdout: &str) -> Option<(serde_json::Map<String, Value>, &str)> {
    let mut start = None;
    let mut offset = 0;
    for line in stdout.split_inclusive('\n') {
        if line.trim_end_matches(['\n', '\r']) == "{" {
            start = Some(offset);
        }
        offset += line.len();
    }
    let start = start?;
    let mut stream = serde_json::Deserializer::from_str(&stdout[start..]).into_iter::<Value>();
    match stream.next() {
        Some(Ok(Value::Object(map))) => Some((map, &stdout[..start])),
        _ => None,
    }
}

/// The text of the last assistant message: its `text` items joined by `\n`.
fn last_assistant_text(object: &serde_json::Map<String, Value>) -> String {
    let Some(messages) = object.get("messages").and_then(Value::as_array) else {
        return String::new();
    };
    let Some(last) = messages
        .iter()
        .rev()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("assistant"))
    else {
        return String::new();
    };
    last.get("content")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter(|i| i.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|i| i.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

/// Parses `goose run --output-format json`: one pretty-printed
/// `{messages, metadata}` object.
fn parse(
    stdout: &str,
    stderr: &str,
    exit_code: Option<i32>,
    session: &str,
) -> Result<Outcome, HarnessError> {
    let exit_note = exit_code
        .map(|c| format!("exit {c}"))
        .unwrap_or_else(|| "no exit code".into());
    let Some((object, prefix)) = last_object(stdout) else {
        return Err(HarnessError::Unusable {
            session_id: Some(session.to_string()),
            message: format!(
                "goose produced no JSON result ({exit_note}; stderr: {}; stdout: {})",
                tail(stderr, 5),
                tail(stdout, 5)
            ),
        });
    };

    let mut warnings = Vec::new();
    if !prefix.trim().is_empty() {
        warnings.push(format!(
            "goose wrote text before its JSON result: {}",
            tail(prefix, 5)
        ));
    }

    let metadata = object.get("metadata").cloned().unwrap_or(Value::Null);
    let status_error = metadata.get("status").and_then(Value::as_str) == Some("error");
    let native = match metadata {
        Value::Object(mut m) => {
            m.remove("status");
            Value::Object(m)
        }
        other => other,
    };
    let result = last_assistant_text(&object).trim_end().to_string();

    if status_error || exit_code.is_some_and(|c| c != 0) {
        let stderr_tail = tail(stderr, 5);
        let message = if !stderr_tail.is_empty() {
            stderr_tail
        } else if !result.is_empty() {
            result
        } else {
            format!("goose reported a failed turn with no message ({exit_note})")
        };
        return Err(HarnessError::Reported {
            session_id: Some(session.to_string()),
            message,
            native,
            warnings,
        });
    }
    if result.is_empty() {
        return Err(HarnessError::Unusable {
            session_id: Some(session.to_string()),
            message: format!(
                "goose produced no assistant text ({exit_note}; stderr: {})",
                tail(stderr, 5)
            ),
        });
    }
    if result.trim() == MAX_TURNS_TEXT {
        return Err(HarnessError::Reported {
            session_id: Some(session.to_string()),
            message: "goose stopped at --max-turns".into(),
            native,
            warnings,
        });
    }
    Ok(Outcome {
        session_id: session.to_string(),
        result,
        native,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::*;
    use crate::agents::Sandbox;
    use crate::config::Harness;
    use crate::harness::modes::{interactive, one_shot, resumed};
    use crate::harness::Mode;
    use crate::testutil::TempDir;

    const SELF_EXE: &str = "/opt/ccc/bin/ccc";

    fn resolved() -> Resolved {
        Resolved {
            agent: Some("explorer".into()),
            description: "Reads".into(),
            harness: Harness::Goose,
            model: "qwen3-coder".into(),
            effort: None,
            tools: None,
            sandbox: Sandbox::ReadOnly,
            body: "Explore.".into(),
            goose: None,
        }
    }

    fn policy(tools: &[GooseTool]) -> GoosePolicy {
        GoosePolicy {
            tools: tools.iter().copied().collect::<BTreeSet<_>>(),
            agents: if tools.contains(&GooseTool::Ccc) {
                vec!["reviewer".into()]
            } else {
                vec![]
            },
            read_roots: vec!["/srv/work".into()],
            write_roots: vec![],
            exec_roots: vec![],
        }
    }

    fn env() -> BTreeMap<String, String> {
        [
            ("HOME", "/home/u"),
            ("XDG_STATE_HOME", "/home/u/.local/state"),
            ("GOOSE_MODE", "approve"),
            ("GOOSE_SYSTEM_PROMPT_FILE_PATH", "/x"),
            ("OPENAI_API_KEY", "sk"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
    }

    fn common(mode: Mode) -> Common {
        Common {
            cwd: PathBuf::from("/srv/work"),
            extra_args: vec![],
            mode,
            preset_session_id: Some("sess-1".into()),
        }
    }

    fn build_with(
        cfg: &GooseConfig,
        policy: &GoosePolicy,
        env: &BTreeMap<String, String>,
        c: &Common,
    ) -> Result<Invocation> {
        let d = GooseDriver {
            cfg,
            policy,
            self_exe: PathBuf::from(SELF_EXE),
            session: session_id(c)?,
            env,
        };
        d.build(&resolved(), c)
    }

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn system() -> String {
        system_prompt(Path::new("/srv/work"), "Explore.")
    }

    #[test]
    fn the_fresh_one_shot_argv_is_exact() {
        let p = policy(&[GooseTool::Read, GooseTool::Ccc]);
        let hex = p.to_hex();
        let inv = build_with(
            &GooseConfig::default(),
            &p,
            &env(),
            &common(one_shot(Origin::Fresh)),
        )
        .unwrap();
        assert_eq!(inv.program, "goose");
        let mut want = strs(&[
            "run",
            "-q",
            "--output-format",
            "json",
            "--no-profile",
            "--provider",
            "openai",
            "--model",
            "qwen3-coder",
            "-n",
            "sess-1",
            "--with-extension",
        ]);
        want.push(format!(
            "ccc:{SELF_EXE} goose-tools --role host --policy {hex}"
        ));
        want.push("--with-extension".into());
        want.push(format!(
            "box:{SELF_EXE} goose-tools --role box --policy {hex}"
        ));
        want.push("--system".into());
        want.push(system());
        want.extend(strs(&["-i", "-"]));
        assert_eq!(inv.args, want);
        assert_eq!(inv.stdin.as_deref(), Some("go"));
        assert_eq!(
            inv.cwd,
            PathBuf::from("/home/u/.local/state/ccc/goose/work")
        );
    }

    #[test]
    fn the_one_shot_resume_argv_names_no_extension_and_no_profile_flag() {
        let p = policy(&[GooseTool::Read]);
        let mut c = common(one_shot(resumed("sess-1", false)));
        c.preset_session_id = None;
        let inv = build_with(&GooseConfig::default(), &p, &env(), &c).unwrap();
        let mut want = strs(&[
            "run",
            "-q",
            "--output-format",
            "json",
            "--resume",
            "--provider",
            "openai",
            "--model",
            "qwen3-coder",
            "-n",
            "sess-1",
            "--system",
        ]);
        want.push(system());
        want.extend(strs(&["-i", "-"]));
        assert_eq!(inv.args, want);
        assert_eq!(
            inv.cwd,
            PathBuf::from("/home/u/.local/state/ccc/goose/work")
        );
    }

    #[test]
    fn the_interactive_argv_fresh_and_resumed() {
        let p = policy(&[GooseTool::Shell]);
        let cfg = GooseConfig {
            max_turns: Some(50),
            extra_args: vec!["--debug".into()],
            ..GooseConfig::default()
        };
        let mut c = common(interactive(SELF_EXE, Origin::Fresh));
        c.extra_args = vec!["--debug".into(), "--passthrough".into()];
        let inv = build_with(&cfg, &p, &env(), &c).unwrap();
        let mut want = strs(&[
            "session",
            "--no-profile",
            "--provider",
            "openai",
            "--model",
            "qwen3-coder",
            "-n",
            "sess-1",
            "--with-extension",
        ]);
        want.push(format!(
            "box:{SELF_EXE} goose-tools --role box --policy {}",
            p.to_hex()
        ));
        want.push("--system".into());
        want.push(system());
        want.extend(strs(&["--max-turns", "50", "--debug", "--passthrough"]));
        assert_eq!(inv.args, want);
        assert_eq!(inv.stdin, None);
        assert_eq!(
            inv.cwd,
            PathBuf::from("/home/u/.local/state/ccc/goose/work")
        );

        let mut c = common(interactive(SELF_EXE, resumed("sess-1", false)));
        c.preset_session_id = None;
        c.extra_args = vec!["--passthrough".into()];
        let inv = build_with(&cfg, &p, &env(), &c).unwrap();
        let mut want = strs(&[
            "session",
            "--resume",
            "--provider",
            "openai",
            "--model",
            "qwen3-coder",
            "-n",
            "sess-1",
            "--system",
        ]);
        want.push(system());
        want.extend(strs(&["--max-turns", "50", "--passthrough"]));
        assert_eq!(inv.args, want);
    }

    #[test]
    fn extensions_follow_the_tool_groups() {
        let cases: [(&[GooseTool], &[&str]); 5] = [
            (&[GooseTool::Ccc], &["ccc:"]),
            (&[GooseTool::Read], &["box:"]),
            (&[GooseTool::Write], &["box:"]),
            (&[GooseTool::Shell, GooseTool::Ccc], &["ccc:", "box:"]),
            (
                &[GooseTool::Read, GooseTool::Write, GooseTool::Shell],
                &["box:"],
            ),
        ];
        for (tools, prefixes) in cases {
            let p = policy(tools);
            let inv = build_with(
                &GooseConfig::default(),
                &p,
                &env(),
                &common(one_shot(Origin::Fresh)),
            )
            .unwrap();
            let exts: Vec<&str> = inv
                .args
                .iter()
                .enumerate()
                .filter(|(i, _)| *i > 0 && inv.args[i - 1] == "--with-extension")
                .map(|(_, a)| a.as_str())
                .collect();
            assert_eq!(exts.len(), prefixes.len(), "{tools:?}: {exts:?}");
            for (ext, prefix) in exts.iter().zip(prefixes) {
                assert!(ext.starts_with(prefix), "{tools:?}: {ext}");
            }
        }
    }

    #[test]
    fn max_turns_and_extra_args_sit_before_the_stdin_marker() {
        let p = policy(&[GooseTool::Read]);
        let mut cfg = GooseConfig::default();
        let inv = build_with(&cfg, &p, &env(), &common(one_shot(Origin::Fresh))).unwrap();
        assert!(!inv.has_arg("--max-turns"));

        cfg.max_turns = Some(7);
        let mut c = common(one_shot(Origin::Fresh));
        c.extra_args = vec!["--extra".into()];
        let inv = build_with(&cfg, &p, &env(), &c).unwrap();
        let n = inv.args.len();
        assert_eq!(
            inv.args[n - 5..],
            strs(&["--max-turns", "7", "--extra", "-i", "-"])
        );
    }

    #[test]
    fn the_preamble_names_the_real_cwd_on_fresh_and_resume() {
        let p = policy(&[GooseTool::Read]);
        for origin in [Origin::Fresh, resumed("sess-1", false)] {
            let mut c = common(one_shot(origin));
            c.cwd = PathBuf::from("/srv/elsewhere");
            let inv = build_with(&GooseConfig::default(), &p, &env(), &c).unwrap();
            let system = inv.value_of("--system").unwrap();
            assert!(
                system.starts_with("Your working directory is /srv/elsewhere. "),
                "{system}"
            );
            assert!(system.ends_with("\n\nExplore."), "{system}");
            assert_ne!(inv.cwd, c.cwd);
        }
    }

    #[test]
    fn the_environment_isolates_goose_and_overlays_the_model() {
        let p = policy(&[GooseTool::Read]);
        let mut cfg = GooseConfig {
            root: Some("/var/goose".into()),
            ..GooseConfig::default()
        };
        cfg.env = [
            ("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"),
            ("OPENAI_API_KEY", "base"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        cfg.models.insert(
            "qwen3-coder".into(),
            crate::config::GooseModelConfig {
                env: [(
                    "OPENAI_BASE_URL".to_string(),
                    "http://127.0.0.1:8081/v1".to_string(),
                )]
                .into_iter()
                .collect(),
            },
        );
        let inv = build_with(&cfg, &p, &env(), &common(one_shot(Origin::Fresh))).unwrap();
        let set: BTreeMap<&str, &str> = inv
            .env_set
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        assert_eq!(set["GOOSE_PATH_ROOT"], "/var/goose");
        assert_eq!(set["GOOSE_MODE"], "auto");
        assert_eq!(set["GOOSE_DISABLE_KEYRING"], "1");
        assert_eq!(set["GOOSE_TELEMETRY_OFF"], "1");
        assert_eq!(set["OPENAI_BASE_URL"], "http://127.0.0.1:8081/v1");
        assert_eq!(set["OPENAI_API_KEY"], "base");
        assert!(!set.contains_key(HOOK_VAR) && !set.contains_key(SESSION_VAR));
        assert_eq!(inv.cwd, PathBuf::from("/var/goose/work"));

        for k in [
            "GOOSE_SYSTEM_PROMPT_FILE_PATH",
            "EXTENSIONS",
            "PLUGINS",
            "SLASH_COMMANDS",
            "CONTEXT_FILE_NAMES",
            "SECURITY_PROMPT_ENABLED",
            HOOK_VAR,
            SESSION_VAR,
        ] {
            assert!(
                inv.env_remove.iter().any(|r| r == k),
                "{k}: {:?}",
                inv.env_remove
            );
        }
        assert!(!inv.env_remove.iter().any(|r| r == "GOOSE_MODE"));
        assert!(!inv.env_remove.iter().any(|r| r == "OPENAI_API_KEY"));
    }

    #[test]
    fn telemetry_stays_off_whatever_the_config_and_environment_say() {
        let p = policy(&[GooseTool::Read]);
        // `validate` refuses these keys; the driver must not depend on that.
        let cfg = GooseConfig {
            env: [("GOOSE_TELEMETRY_OFF", "0"), ("OTEL_SDK_DISABLED", "false")]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            ..GooseConfig::default()
        };
        let mut e = env();
        for k in LANGFUSE_KEYS {
            e.insert(k.into(), "lf".into());
        }
        e.insert("OTEL_SDK_DISABLED".into(), "false".into());
        e.insert("GOOSE_TELEMETRY_OFF".into(), "0".into());
        for mode in [
            one_shot(Origin::Fresh),
            interactive(SELF_EXE, Origin::Fresh),
        ] {
            let inv = build_with(&cfg, &p, &e, &common(mode)).unwrap();
            let set: BTreeMap<&str, &str> = inv
                .env_set
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            assert_eq!(set["GOOSE_TELEMETRY_OFF"], "1");
            assert_eq!(set["OTEL_SDK_DISABLED"], "true");
            assert_eq!(
                inv.env_set
                    .iter()
                    .filter(|(k, _)| k == "GOOSE_TELEMETRY_OFF" || k == "OTEL_SDK_DISABLED")
                    .count(),
                2
            );
            for k in LANGFUSE_KEYS {
                assert!(
                    inv.env_remove.iter().any(|r| r == k),
                    "{k}: {:?}",
                    inv.env_remove
                );
                assert!(!set.contains_key(k));
            }
            assert!(!inv
                .env_remove
                .iter()
                .any(|r| r == "GOOSE_TELEMETRY_OFF" || r == "OTEL_SDK_DISABLED"));
        }
    }

    const PROXIES: [&str; 6] = [
        "NO_PROXY",
        "no_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "ALL_PROXY",
        "all_proxy",
    ];

    #[test]
    fn goose_gets_an_empty_home_an_https_sink_and_no_inherited_proxy() {
        let p = policy(&[GooseTool::Read]);
        let cfg = GooseConfig {
            root: Some("/var/goose".into()),
            ..GooseConfig::default()
        };
        let mut e = env();
        e.insert("HTTP_PROXY".into(), "http://corp:3128".into());
        e.insert("NO_PROXY".into(), "192.168.0.0/16".into());
        e.insert("https_proxy".into(), "http://corp:3129".into());
        for k in ["no_proxy", "http_proxy", "ALL_PROXY", "all_proxy"] {
            e.insert(k.into(), format!("value of {k}"));
        }
        for mode in [
            one_shot(Origin::Fresh),
            interactive(SELF_EXE, Origin::Fresh),
        ] {
            let inv = build_with(&cfg, &p, &e, &common(mode)).unwrap();
            let set: BTreeMap<&str, &str> = inv
                .env_set
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            assert_eq!(set["HOME"], "/var/goose/home");
            assert_eq!(set["HTTPS_PROXY"], "http://127.0.0.1:9");
            assert_eq!(set["https_proxy"], "http://127.0.0.1:9");
            for k in PROXIES {
                assert!(!set.contains_key(k), "{k} set");
                assert!(
                    inv.env_remove.iter().any(|r| r == k),
                    "{k}: {:?}",
                    inv.env_remove
                );
            }

            let carried: BTreeMap<String, String> =
                [(HOST_ENV_VAR.to_string(), set[HOST_ENV_VAR].to_string())].into();
            let ops: BTreeMap<String, Option<String>> = host_env_ops(&carried)
                .unwrap()
                .unwrap()
                .into_iter()
                .collect();
            let mut want: BTreeMap<String, Option<String>> = host_env_keys()
                .map(|k| (k.to_string(), e.get(k).cloned()))
                .collect();
            want.insert(HOST_ENV_VAR.into(), None);
            assert_eq!(ops, want);
            assert_eq!(ops["HOME"].as_deref(), Some("/home/u"));
            assert_eq!(ops["HTTPS_PROXY"], None);
            assert_eq!(ops["HTTP_PROXY"].as_deref(), Some("http://corp:3128"));
        }
    }

    #[test]
    fn proxy_settings_from_the_config_reach_goose() {
        let p = policy(&[GooseTool::Read]);
        let cfg = GooseConfig {
            env: [
                ("NO_PROXY", "llm.example"),
                ("HTTP_PROXY", "http://chosen:3128"),
            ]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
            ..GooseConfig::default()
        };
        let mut e = env();
        e.insert("NO_PROXY".into(), "*".into());
        e.insert("HTTP_PROXY".into(), "http://corp:3128".into());
        let inv = build_with(&cfg, &p, &e, &common(one_shot(Origin::Fresh))).unwrap();
        let set: BTreeMap<&str, &str> = inv
            .env_set
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        assert_eq!(set["NO_PROXY"], "llm.example");
        assert_eq!(set["HTTP_PROXY"], "http://chosen:3128");
        for k in ["NO_PROXY", "HTTP_PROXY"] {
            assert!(!inv.env_remove.iter().any(|r| r == k), "{k} removed");
        }
        assert!(inv.env_remove.iter().any(|r| r == "ALL_PROXY"));
    }

    #[test]
    fn a_config_home_or_https_proxy_does_not_replace_ccc_s() {
        // `validate` refuses these keys; the driver must not depend on that.
        let p = policy(&[GooseTool::Read]);
        let cfg = GooseConfig {
            root: Some("/var/goose".into()),
            env: [("HOME", "/elsewhere"), ("HTTPS_PROXY", "http://corp:3128")]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            ..GooseConfig::default()
        };
        let inv = build_with(&cfg, &p, &env(), &common(one_shot(Origin::Fresh))).unwrap();
        let homes: Vec<&str> = inv
            .env_set
            .iter()
            .filter(|(k, _)| k == "HOME" || k == "HTTPS_PROXY")
            .map(|(_, v)| v.as_str())
            .collect();
        assert_eq!(homes, ["/var/goose/home", "http://127.0.0.1:9"]);
    }

    fn carrying(object: &str) -> BTreeMap<String, String> {
        [(
            HOST_ENV_VAR.to_string(),
            crate::goose_policy::hex_encode(object.as_bytes()),
        )]
        .into()
    }

    #[test]
    fn the_host_env_restore_sets_removes_and_leaves_keys() {
        assert_eq!(host_env_ops(&env()).unwrap(), None);
        let ops = host_env_ops(&carrying(
            r#"{"HOME":"/home/u","HTTPS_PROXY":null,"no_proxy":"a,b"}"#,
        ))
        .unwrap()
        .unwrap();
        let want: Vec<(String, Option<String>)> = vec![
            ("HOME".into(), Some("/home/u".into())),
            ("HTTPS_PROXY".into(), None),
            ("no_proxy".into(), Some("a,b".into())),
            (HOST_ENV_VAR.into(), None),
        ];
        assert_eq!(ops, want);
    }

    #[test]
    fn a_host_env_that_does_not_decode_is_an_error_naming_the_variable() {
        let mut odd = carrying("{}");
        odd.insert(HOST_ENV_VAR.into(), "abc".into());
        let mut not_hex = carrying("{}");
        not_hex.insert(HOST_ENV_VAR.into(), "zz".into());
        for env in [
            odd,
            not_hex,
            carrying("not json"),
            carrying(r#"["HOME"]"#),
            carrying(r#"{"PATH":"/bin"}"#),
            carrying(r#"{"HOME":3}"#),
        ] {
            let err = format!("{:#}", host_env_ops(&env).unwrap_err());
            assert!(err.contains(HOST_ENV_VAR), "{err}");
        }
    }

    #[test]
    fn a_host_env_value_holding_nul_is_refused() {
        let err = format!(
            "{:#}",
            host_env_ops(&carrying(r#"{"HOME":"a\u0000b"}"#)).unwrap_err()
        );
        assert!(
            err.contains(HOST_ENV_VAR) && err.contains("HOME is not a string or null"),
            "{err}"
        );
    }

    #[test]
    fn a_launch_sets_the_hook_variables() {
        let p = policy(&[GooseTool::Read]);
        let inv = build_with(
            &GooseConfig::default(),
            &p,
            &env(),
            &common(interactive(SELF_EXE, Origin::Fresh)),
        )
        .unwrap();
        let set: BTreeMap<&str, &str> = inv
            .env_set
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        assert_eq!(set[HOOK_VAR], SELF_EXE);
        assert_eq!(set[SESSION_VAR], "sess-1");
        assert!(!inv
            .env_remove
            .iter()
            .any(|r| r == HOOK_VAR || r == SESSION_VAR));
        assert!(inv.env_remove.iter().any(|r| r == "PLUGINS"));

        let mut c = common(interactive(SELF_EXE, resumed("sess-1", false)));
        c.preset_session_id = None;
        let inv = build_with(&GooseConfig::default(), &p, &env(), &c).unwrap();
        assert!(inv
            .env_set
            .contains(&(SESSION_VAR.to_string(), "sess-1".to_string())));
    }

    #[test]
    fn the_binary_override_is_honoured() {
        let p = policy(&[GooseTool::Read]);
        let mut e = env();
        e.insert("CCC_GOOSE_BIN".into(), "/opt/goose".into());
        let inv = build_with(
            &GooseConfig::default(),
            &p,
            &e,
            &common(one_shot(Origin::Fresh)),
        )
        .unwrap();
        assert_eq!(inv.program, "/opt/goose");
    }

    #[test]
    fn forks_and_unnamed_fresh_runs_are_refused() {
        let p = policy(&[GooseTool::Read]);
        for mode in [
            one_shot(resumed("sess-1", true)),
            interactive(SELF_EXE, resumed("sess-1", true)),
        ] {
            let err = build_with(&GooseConfig::default(), &p, &env(), &common(mode))
                .unwrap_err()
                .to_string();
            assert!(err.contains("cannot be forked"), "{err}");
        }
        let mut c = common(one_shot(Origin::Fresh));
        c.preset_session_id = None;
        assert!(build_with(&GooseConfig::default(), &p, &env(), &c).is_err());
    }

    #[test]
    fn an_oversized_policy_token_is_refused() {
        let mut p = policy(&[GooseTool::Ccc]);
        p.agents = (0..10_000).map(|i| format!("agent-{i}")).collect();
        let err = build_with(
            &GooseConfig::default(),
            &p,
            &env(),
            &common(one_shot(Origin::Fresh)),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("--with-extension"), "{err}");
        assert!(err.contains("command-line limit"), "{err}");
    }

    #[test]
    fn self_exe_paths_goose_would_split_are_refused() {
        let p = policy(&[GooseTool::Read]);
        let cfg = GooseConfig::default();
        let e = env();
        let c = common(one_shot(Origin::Fresh));
        for bad in [
            "/opt/my ccc/ccc",
            "/opt/\"q\"/ccc",
            "/opt/it's/ccc",
            "/opt/a=b/ccc",
        ] {
            let d = GooseDriver {
                cfg: &cfg,
                policy: &p,
                self_exe: PathBuf::from(bad),
                session: "sess-1".into(),
                env: &e,
            };
            let err = d.build(&resolved(), &c).unwrap_err().to_string();
            assert!(err.contains("install ccc at a path"), "{bad}: {err}");
        }
        let d = GooseDriver {
            cfg: &cfg,
            policy: &p,
            self_exe: PathBuf::from("/opt/c:c/ccc"),
            session: "sess-1".into(),
            env: &e,
        };
        let inv = d.build(&resolved(), &c).unwrap();
        assert!(inv
            .value_of("--with-extension")
            .unwrap()
            .starts_with("box:/opt/c:c/ccc goose-tools"));
    }

    #[test]
    fn prepare_root_creates_a_private_work_dir_and_maintains_the_plugin() {
        use std::os::unix::fs::PermissionsExt;
        let t = TempDir::new("goose-root");
        let root = t.path().join("goose");
        let work = prepare_root(&root).unwrap();
        assert_eq!(work, root.join("work"));
        let home = home_dir(&root);
        assert_eq!(std::fs::read_dir(&home).unwrap().count(), 0);
        for dir in [&root, &work, &home] {
            let mode = std::fs::metadata(dir).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o700, "{}", dir.display());
        }
        let plugin = plugin_file(&root);
        assert_eq!(std::fs::read_to_string(&plugin).unwrap(), plugin_json());

        // Identical content is left alone: an atomic rewrite would give the
        // file a new inode.
        use std::os::unix::fs::MetadataExt;
        let before = std::fs::metadata(&plugin).unwrap().ino();
        prepare_root(&root).unwrap();
        assert_eq!(std::fs::metadata(&plugin).unwrap().ino(), before);

        std::fs::write(&plugin, "{}").unwrap();
        std::fs::write(home.join("kept"), "x").unwrap();
        prepare_root(&root).unwrap();
        assert_eq!(std::fs::read_to_string(&plugin).unwrap(), plugin_json());
        assert_eq!(std::fs::read_to_string(home.join("kept")).unwrap(), "x");

        let other = t.path().join("existing");
        std::fs::create_dir(&other).unwrap();
        std::fs::set_permissions(&other, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::create_dir(other.join("work")).unwrap();
        std::fs::set_permissions(other.join("work"), std::fs::Permissions::from_mode(0o755))
            .unwrap();
        prepare_root(&other).unwrap();
        let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(&other), 0o755);
        assert_eq!(mode(&other.join("work")), 0o700);
    }

    #[test]
    fn the_plugin_holds_one_rule_per_event() {
        let doc: Value = serde_json::from_str(&plugin_json()).unwrap();
        for (event, kind) in [
            ("SessionStart", "goose-session-start"),
            ("Stop", "goose-stop"),
        ] {
            let rules = doc["hooks"][event].as_array().unwrap();
            assert_eq!(rules.len(), 1, "{event}");
            assert!(rules[0].get("matcher").is_none(), "{event}");
            let actions = rules[0]["hooks"].as_array().unwrap();
            assert_eq!(actions.len(), 1, "{event}");
            assert_eq!(actions[0]["type"], "command");
            assert_eq!(
                actions[0]["command"],
                format!(
                    "[ -n \"$CCC_GOOSE_HOOK\" ] || {{ cat >/dev/null; exit 0; }}; exec \"$CCC_GOOSE_HOOK\" hook {kind}"
                )
            );
        }
    }

    #[test]
    fn without_the_hook_variable_a_hook_drains_its_payload_and_allows() {
        use std::io::Write;
        use std::process::{Command, Stdio};
        for kind in ["goose-session-start", "goose-stop"] {
            let mut child = Command::new("/bin/sh")
                .arg("-c")
                .arg(hook_command(kind))
                .env_remove(HOOK_VAR)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            child
                .stdin
                .take()
                .unwrap()
                .write_all(br#"{"event":"Stop","session_id":"20261101_1"}"#)
                .unwrap();
            let out = child.wait_with_output().unwrap();
            assert_eq!(out.status.code(), Some(0), "{kind}");
            assert!(out.stdout.is_empty(), "{kind}");
        }
    }

    fn object(messages: Value, status: &str) -> String {
        let doc = json!({
            "messages": messages,
            "metadata": {"total_tokens": 42, "input_tokens": 30, "status": status},
        });
        serde_json::to_string_pretty(&doc).unwrap() + "\n"
    }

    fn assistant(texts: &[&str]) -> Value {
        let content: Vec<Value> = texts
            .iter()
            .map(|t| json!({"type": "text", "text": t}))
            .collect();
        json!({"role": "assistant", "content": content})
    }

    fn run_parse(stdout: &str, stderr: &str, exit: i32) -> Result<Outcome, HarnessError> {
        parse(stdout, stderr, Some(exit), "sess-1")
    }

    #[test]
    fn a_completed_object_parses_with_the_preset_session() {
        let out = run_parse(
            &object(json!([{"role":"user","content":[{"type":"text","text":"q"}]}, assistant(&["answer"])]), "completed"),
            "",
            0,
        )
        .unwrap();
        assert_eq!(out.session_id, "sess-1");
        assert_eq!(out.result, "answer");
        assert_eq!(out.native, json!({"total_tokens": 42, "input_tokens": 30}));
        assert!(out.warnings.is_empty());
    }

    #[test]
    fn the_last_assistant_wins_and_only_text_items_are_joined() {
        let history = json!([
            assistant(&["first"]),
            {"role":"user","content":[{"type":"toolResponse","id":"x"}]},
            {"role":"assistant","content":[
                {"type":"text","text":"one"},
                {"type":"toolRequest","id":"y"},
                {"type":"text","text":"two"}
            ]},
            {"role":"user","content":[{"type":"text","text":"trailing"}]}
        ]);
        let out = run_parse(&object(history, "completed"), "", 0).unwrap();
        assert_eq!(out.result, "one\ntwo");
    }

    #[test]
    fn text_before_the_object_becomes_one_warning() {
        let stdout = format!(
            "stray render\nmore\n{}",
            object(json!([assistant(&["ok"])]), "completed")
        );
        let out = run_parse(&stdout, "", 0).unwrap();
        assert_eq!(out.result, "ok");
        assert_eq!(out.warnings.len(), 1);
        assert!(out.warnings[0].contains("more"), "{:?}", out.warnings);
    }

    #[test]
    fn an_error_status_is_reported_with_the_stderr_tail() {
        let stdout = object(
            json!([
                assistant(&["partial"]),
                assistant(&["what would you like me to do?"])
            ]),
            "error",
        );
        let err = run_parse(&stdout, "Error: stream ended\n", 0).unwrap_err();
        assert!(matches!(err, HarnessError::Reported { .. }), "{err:?}");
        assert_eq!(err.message(), "Error: stream ended");
        assert_eq!(err.session_id(), Some("sess-1"));
        assert_eq!(
            err.reported().unwrap().1,
            &json!({"total_tokens": 42, "input_tokens": 30})
        );

        let err = run_parse(&stdout, "", 0).unwrap_err();
        assert_eq!(err.message(), "what would you like me to do?");
    }

    #[test]
    fn a_nonzero_exit_is_reported() {
        let stdout = object(json!([assistant(&["done"])]), "completed");
        let err = run_parse(&stdout, "Error: boom\n", 1).unwrap_err();
        assert_eq!(err.exit_code(), crate::harness::EXIT_HARNESS_FAILED);
        assert_eq!(err.message(), "Error: boom");

        let err = run_parse(&object(json!([]), "completed"), "", 1).unwrap_err();
        assert!(matches!(err, HarnessError::Reported { .. }), "{err:?}");
        assert!(err.message().contains("exit 1"), "{}", err.message());
    }

    #[test]
    fn no_json_is_unusable_with_both_tails() {
        let err = run_parse("Failed to create provider\n", "warn: x\n", 1).unwrap_err();
        assert_eq!(err.exit_code(), crate::harness::EXIT_NO_RESULT);
        let msg = err.message();
        for part in ["exit 1", "Failed to create provider", "warn: x"] {
            assert!(msg.contains(part), "{msg} lacks {part}");
        }
        // A `{` line that does not open an object is no object either.
        assert!(run_parse("{\nnot json\n", "", 0).is_err());
    }

    #[test]
    fn an_empty_result_is_unusable() {
        for history in [
            json!([]),
            json!([{"role":"assistant","content":[{"type":"toolRequest"}]}]),
        ] {
            let err = run_parse(&object(history, "completed"), "", 0).unwrap_err();
            assert_eq!(err.exit_code(), crate::harness::EXIT_NO_RESULT, "{err:?}");
        }
    }

    #[test]
    fn the_max_turns_question_is_a_reported_failure() {
        let err = run_parse(
            &object(json!([assistant(&[MAX_TURNS_TEXT])]), "completed"),
            "",
            0,
        )
        .unwrap_err();
        assert!(matches!(err, HarnessError::Reported { .. }), "{err:?}");
        assert_eq!(err.message(), "goose stopped at --max-turns");
    }
}
