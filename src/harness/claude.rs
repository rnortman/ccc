//! The Claude Code driver: `claude -p --output-format json` with the agent
//! supplied inline through `--agents`/`--agent`.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::json;

use crate::audit::HookKind;
use crate::config::{expand_tilde_in, ClaudeAuth, ClaudeConfig, EnvSource, PermissionMode};
use crate::harness::{
    binary_in, check_arg_len, tail, Captured, Common, Driver, HarnessError, Invocation, Io, Origin,
    Outcome,
};
use crate::resolve::Resolved;

/// Environment variables that must not be set when an OAuth token is used;
/// any of them would send the child to a different endpoint or credential.
const TOKEN_CONFLICTS: [&str; 5] = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
];

const TOKEN_VAR: &str = "CLAUDE_CODE_OAUTH_TOKEN";
const CONFIG_DIR_VAR: &str = "CLAUDE_CONFIG_DIR";

/// Variables to set, and variables to remove, for one account.
pub type EnvEdits = (Vec<(String, String)>, Vec<String>);

/// Which account the child runs as, and how it is pointed at it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Account {
    pub profile: String,
    pub auth: ClaudeAuth,
    pub token_dir: PathBuf,
    pub profiles_dir: PathBuf,
}

impl Account {
    /// The account described by `cfg` for `profile`, with paths expanded
    /// against `env`.
    pub fn from_config(cfg: &ClaudeConfig, profile: &str, env: &dyn EnvSource) -> Account {
        Account {
            profile: profile.to_string(),
            auth: cfg.auth,
            token_dir: expand_tilde_in(env, &cfg.token_dir),
            profiles_dir: expand_tilde_in(env, &cfg.profiles_dir),
        }
    }

    /// The directory exported as `CLAUDE_CONFIG_DIR` under `config-dir` auth,
    /// holding this account's credentials and `.claude.json`. Every path derived
    /// from the profile layout comes from here, so they cannot drift apart.
    pub fn config_dir(&self) -> PathBuf {
        self.profiles_dir.join(&self.profile)
    }

    /// The local file that proves this account can be used.
    pub fn credential_path(&self) -> PathBuf {
        match self.auth {
            ClaudeAuth::Token => self.token_dir.join(format!("{}.token", self.profile)),
            ClaudeAuth::ConfigDir => self.config_dir().join(".credentials.json"),
        }
    }

    /// Whether the credential file for this account exists on disk.
    pub fn has_credential(&self) -> bool {
        self.credential_path().exists()
    }

    /// `Ok` when this account has a local credential, otherwise the one
    /// sentence every caller reports; callers append their own context.
    pub fn require_credential(&self) -> Result<(), String> {
        if self.has_credential() {
            return Ok(());
        }
        Err(format!(
            "profile {} has no credential at {}",
            self.profile,
            self.credential_path().display()
        ))
    }

    /// The environment edits for this account: exactly one of the two account
    /// variables is set and the other is removed.
    pub fn env_edits(&self, env: &dyn EnvSource) -> Result<EnvEdits> {
        match self.auth {
            ClaudeAuth::Token => {
                if let Some(var) = conflicts_set(env).first() {
                    bail!("{var} is set; unset it before using token authentication");
                }
                let path = self.credential_path();
                let token = read_token(&path)?;
                let mut remove = conflict_removals();
                remove.push(CONFIG_DIR_VAR.to_string());
                Ok((vec![(TOKEN_VAR.to_string(), token)], remove))
            }
            ClaudeAuth::ConfigDir => {
                let dir = self.config_dir();
                // The same variables would override the config dir's own
                // credentials, sending the child to another account entirely.
                let mut remove = conflict_removals();
                remove.push(TOKEN_VAR.to_string());
                Ok((
                    vec![(CONFIG_DIR_VAR.to_string(), dir.display().to_string())],
                    remove,
                ))
            }
        }
    }
}

/// The environment edits for the caller's ambient login (`account` is
/// `None`): nothing is set, and only the variables that would send the child
/// to another endpoint or credential are removed.
pub fn ambient_env_edits() -> EnvEdits {
    (Vec::new(), conflict_removals())
}

/// The variables every Claude spawn removes, whatever the account mode.
fn conflict_removals() -> Vec<String> {
    TOKEN_CONFLICTS.iter().map(|v| v.to_string()).collect()
}

/// The endpoint/credential variables `env` sets to a non-blank value, which a
/// Claude spawn removes or, under token auth, refuses.
pub fn conflicts_set(env: &dyn EnvSource) -> Vec<&'static str> {
    TOKEN_CONFLICTS
        .into_iter()
        .filter(|var| env.get(var).is_some_and(|v| !v.trim().is_empty()))
        .collect()
}

/// The Claude Code config file a child started in `cwd` will read:
/// `.claude.json` inside the account's config dir; under the ambient login
/// (`None`), inside an inherited non-blank `CLAUDE_CONFIG_DIR`, which ccc
/// leaves in place and the child resolves against `cwd` when relative;
/// otherwise `$HOME/.claude.json`.
pub fn config_file(account: Option<&Account>, env: &dyn EnvSource, cwd: &Path) -> PathBuf {
    match account {
        Some(account) if account.auth == ClaudeAuth::ConfigDir => {
            account.config_dir().join(".claude.json")
        }
        None => match env.get(CONFIG_DIR_VAR) {
            Some(dir) if !dir.trim().is_empty() => cwd.join(dir).join(".claude.json"),
            _ => crate::config::home_dir_in(env).join(".claude.json"),
        },
        Some(_) => crate::config::home_dir_in(env).join(".claude.json"),
    }
}

/// Marks `cwd` trusted in the account's config file (`None`: the ambient
/// login's), so the child neither asks
/// (interactive) nor ignores the workspace's settings (`-p`).
/// `Ok(())` when already trusted; the file is not rewritten in that case.
///
/// Read-modify-write with a lost-update window: Claude Code rewrites the whole
/// file from its in-memory copy, so a session already running under the same
/// account can drop this entry, and this write can drop that session's own
/// updates. Accepted — several subagents share a handful of profiles, the entry
/// is rewritten before every spawn, and losing it once costs one spawn the
/// status-quo behaviour (the trust dialog, or ignored workspace settings).
pub fn trust_directory(account: Option<&Account>, env: &dyn EnvSource, cwd: &Path) -> Result<()> {
    // Claude Code keeps home-directory trust for the session only, so there is
    // nothing useful to persist. `cwd` is canonical (see `resolve_cwd`), so the
    // home path must be too before the comparison.
    let home = crate::config::home_dir_in(env);
    if cwd == std::fs::canonicalize(&home).unwrap_or(home) {
        return Ok(());
    }
    let path = config_file(account, env, cwd);

    let mut root = match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str::<serde_json::Value>(&text)
            .with_context(|| format!("parsing {}", path.display()))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => json!({}),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };

    // Claude Code keys `projects` by the literal path string; a lossy
    // conversion would write a key the child can never match.
    let key = cwd
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("{} is not valid UTF-8", cwd.display()))?;

    let Some(obj) = root.as_object_mut() else {
        bail!("{} is not a JSON object", path.display());
    };
    let projects = obj
        .entry("projects")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("projects in {} is not a JSON object", path.display()))?;
    let entry = projects
        .entry(key.to_string())
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| {
            anyhow::anyhow!("projects[{key}] in {} is not a JSON object", path.display())
        })?;
    if entry.get("hasTrustDialogAccepted") == Some(&json!(true)) {
        return Ok(());
    }
    entry.insert("hasTrustDialogAccepted".to_string(), json!(true));

    crate::fsutil::write_atomically(&path, &serde_json::to_string_pretty(&root)?)
}

/// Reads a `~/.claude-tokens/<profile>.token` file: mode 0600, one non-empty line.
fn read_token(path: &Path) -> Result<String> {
    // Permissions are checked before the secret is read, so a world-readable
    // token never enters this process.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(path)
            .with_context(|| format!("reading token file {}", path.display()))?
            .permissions()
            .mode()
            & 0o777;
        if mode & 0o077 != 0 {
            bail!(
                "token file {} is mode {mode:o}; it must not be readable by group or others (chmod 600)",
                path.display()
            );
        }
    }

    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading token file {}", path.display()))?;

    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let Some(token) = lines.next() else {
        bail!("token file {} is empty", path.display());
    };
    if lines.next().is_some() {
        bail!("token file {} has more than one line", path.display());
    }
    Ok(token.trim().to_string())
}

/// The Claude harness: an account and a permission mode.
pub struct ClaudeDriver<'a> {
    pub permission_mode: PermissionMode,
    pub account: Option<Account>,
    pub env: &'a dyn EnvSource,
}

impl Driver for ClaudeDriver<'_> {
    fn build(&self, r: &Resolved, c: &Common) -> Result<Invocation> {
        build(r, c, self)
    }

    fn parse(&self, captured: &Captured) -> Result<Outcome, HarnessError> {
        parse(&captured.stdout, &captured.stderr)
    }

    fn early_session_id(&self, _line: &str) -> Option<String> {
        None
    }
}

/// The `--agents` payload: one entry keyed by the agent type.
fn agents_json(agent: &str, r: &Resolved) -> String {
    let mut entry = json!({
        "description": r.description,
        "prompt": r.body,
    });
    if let Some(tools) = &r.tools {
        entry["tools"] = json!(tools);
    }
    json!({ agent: entry }).to_string()
}

/// Builds the argv, environment and stdin for one Claude invocation.
fn build(r: &Resolved, c: &Common, d: &ClaudeDriver) -> Result<Invocation> {
    let agents = match &r.agent {
        Some(agent) => {
            let json = agents_json(agent, r);
            check_arg_len(agent, "--agents JSON", &json)?;
            Some((agent, json))
        }
        None => None,
    };

    let mut args: Vec<String> = Vec::new();
    let one_shot = !c.mode.is_launch();
    if one_shot {
        args.extend(["-p", "--output-format", "json"].map(String::from));
    }
    args.push("--model".into());
    args.push(r.model.clone());
    if let Some(effort) = r.effort {
        args.push("--effort".into());
        args.push(effort.as_str().to_string());
    }
    if let Some((agent, json)) = agents {
        args.push("--agents".into());
        args.push(json);
        args.push("--agent".into());
        args.push(agent.clone());
    }
    if c.mode.origin == Origin::Fresh {
        if let Some(id) = &c.preset_session_id {
            args.push("--session-id".into());
            args.push(id.clone());
        }
    }
    // An interactive session keeps its own approval prompts, as Codex does.
    if one_shot {
        args.push("--permission-mode".into());
        args.push(d.permission_mode.as_str().to_string());
    }
    if let Some(id) = c.mode.starts_from() {
        args.push("--resume".into());
        args.push(id.to_string());
        if c.mode.is_fork() {
            args.push("--fork-session".into());
        }
    }
    if let Io::Interactive { self_exe } = &c.mode.io {
        let command = format!(
            "{} hook {}",
            shell_quote(&self_exe.to_string_lossy()),
            HookKind::ClaudeStop
        );
        let settings = serde_json::json!({
            "hooks": { "Stop": [{ "hooks": [{ "type": "command", "command": command }] }] }
        })
        .to_string();
        args.push("--settings".into());
        args.push(settings);
    }
    args.extend(c.extra_args.iter().cloned());

    let (env_set, env_remove) = match &d.account {
        Some(acct) => acct.env_edits(d.env)?,
        None => ambient_env_edits(),
    };

    Ok(Invocation {
        program: binary_in(d.env, "CCC_CLAUDE_BIN", "claude"),
        args,
        env_set,
        env_remove,
        cwd: c.cwd.clone(),
        stdin: c.mode.stdin(),
    })
}

fn shell_quote(path: &str) -> String {
    format!("'{}'", path.replace('\'', "'\\''"))
}

/// Whether a failed turn's message is Claude Code saying the account hit a
/// usage limit (`You've hit your session limit · resets …`, or a wording
/// naming a usage limit).
pub fn is_usage_limit(message: &str) -> bool {
    let message = message.trim();
    (message.starts_with("You've hit your") && message.contains("limit"))
        || message.contains("usage limit")
}

/// Parses one `--output-format json` object.
fn parse(stdout: &str, stderr: &str) -> Result<Outcome, HarnessError> {
    let value: serde_json::Value = match serde_json::from_str(stdout.trim()) {
        Ok(v) => v,
        Err(e) => {
            return Err(HarnessError::unusable(format!(
                "claude produced no JSON result ({e}); stderr: {}",
                tail(stderr, 5)
            )))
        }
    };

    let session_id = value
        .get("session_id")
        .and_then(|v| v.as_str())
        .map(String::from);
    let result = value
        .get("result")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let native = json!({
        "num_turns": value.get("num_turns"),
        "total_cost_usd": value.get("total_cost_usd"),
        "duration_ms": value.get("duration_ms"),
        "modelUsage": value.get("modelUsage"),
    });

    if value.get("is_error").and_then(|v| v.as_bool()) == Some(true) {
        let message = match (result.as_str(), tail(stderr, 5)) {
            (r, _) if !r.is_empty() => result,
            (_, s) if !s.is_empty() => s,
            // Neither channel said anything; report the structure instead of
            // handing the caller an empty failure.
            _ => {
                let subtype = value
                    .get("subtype")
                    .and_then(|v| v.as_str())
                    .unwrap_or("none");
                let turns = value
                    .get("num_turns")
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "?".to_string());
                format!(
                    "claude reported is_error=true with no message (subtype={subtype}, num_turns={turns})"
                )
            }
        };
        if is_usage_limit(&message) {
            return Err(HarnessError::Limited {
                session_id,
                message,
                native,
                warnings: Vec::new(),
            });
        }
        return Err(HarnessError::Reported {
            session_id,
            message,
            native,
            warnings: Vec::new(),
        });
    }

    let Some(session_id) = session_id else {
        return Err(HarnessError::unusable(
            "claude returned JSON without a session_id".to_string(),
        ));
    };
    if result.is_empty() {
        return Err(
            HarnessError::unusable("claude returned an empty result".to_string())
                .with_session_id(Some(session_id)),
        );
    }

    Ok(Outcome {
        session_id,
        result,
        native,
        warnings: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agents::Sandbox;
    use crate::config::{Effort, Harness};
    use crate::harness::modes::{interactive, one_shot, resumed, run_mode};
    use crate::harness::{Io, Mode, Origin};
    use crate::testutil::TempDir;
    use std::collections::BTreeMap;

    fn resolved() -> Resolved {
        Resolved {
            agent: Some("explorer".into()),
            description: "Investigates".into(),
            harness: Harness::Claude,
            model: "claude-haiku-4-5[1m]".into(),
            effort: Some(Effort::Low),
            tools: Some(vec!["Read".into(), "Grep".into()]),
            sandbox: Sandbox::ReadOnly,
            body: "You are an explorer.\nCite \"files\".".into(),
        }
    }

    fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
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
    fn run_and_launch_emit_preset_session_and_launch_stop_hook_before_extras() {
        let e = env(&[]);
        let d = driver(&e);
        let mut run = common(run_mode());
        run.preset_session_id = Some("preset-123".into());
        run.extra_args = vec!["--extra".into()];
        let invocation = d.build(&resolved(), &run).unwrap();
        assert_eq!(invocation.value_of("--session-id"), Some("preset-123"));
        assert!(
            invocation
                .args
                .iter()
                .position(|arg| arg == "--session-id")
                .unwrap()
                < invocation.args.len() - 1
        );

        let path = PathBuf::from("/tmp/path with 'quote'/ccc");
        let mut launch = common(Mode {
            io: Io::Interactive {
                self_exe: path.clone(),
            },
            origin: Origin::Fresh,
        });
        launch.extra_args = vec!["--extra".into()];
        launch.preset_session_id = Some("launch-preset".into());
        let invocation = d.build(&resolved(), &launch).unwrap();
        let settings: serde_json::Value =
            serde_json::from_str(invocation.value_of("--settings").unwrap()).unwrap();
        let command = settings["hooks"]["Stop"][0]["hooks"][0]["command"]
            .as_str()
            .unwrap();
        assert_eq!(
            command,
            "'/tmp/path with '\\''quote'\\''/ccc' hook claude-stop"
        );
        assert_eq!(invocation.value_of("--session-id"), Some("launch-preset"));
        assert!(
            invocation
                .args
                .iter()
                .position(|arg| arg == "--settings")
                .unwrap()
                < invocation
                    .args
                    .iter()
                    .position(|arg| arg == "--extra")
                    .unwrap()
        );
    }

    fn driver<'a>(env: &'a BTreeMap<String, String>) -> ClaudeDriver<'a> {
        ClaudeDriver {
            permission_mode: PermissionMode::DontAsk,
            account: None,
            env,
        }
    }

    #[test]
    fn run_argv_has_the_one_shot_flags() {
        let e = env(&[]);
        let i = driver(&e).build(&resolved(), &common(run_mode())).unwrap();
        assert_eq!(i.program, "claude");
        assert_eq!(&i.args[0..3], &["-p", "--output-format", "json"]);
        assert_eq!(i.value_of("--model"), Some("claude-haiku-4-5[1m]"));
        assert_eq!(i.value_of("--effort"), Some("low"));
        assert_eq!(i.value_of("--agent"), Some("explorer"));
        assert_eq!(i.value_of("--permission-mode"), Some("dontAsk"));
        assert!(!i.has_arg("--bare"));
        assert!(!i.has_arg("--resume"));
        assert_eq!(i.cwd, PathBuf::from("/srv/work"));
        assert_eq!(i.stdin.as_deref(), Some("go"));

        // The model lives on the CLI only, not in the agents JSON.
        let agents: serde_json::Value =
            serde_json::from_str(i.value_of("--agents").unwrap()).unwrap();
        let entry = &agents["explorer"];
        assert_eq!(entry["prompt"], "You are an explorer.\nCite \"files\".");
        assert_eq!(entry["tools"], json!(["Read", "Grep"]));
        assert!(entry.get("model").is_none());
    }

    #[test]
    fn resume_argv_is_run_plus_resume_and_launch_drops_the_one_shot_flags() {
        let e = env(&[]);
        let run = driver(&e).build(&resolved(), &common(run_mode())).unwrap();
        let resume = driver(&e)
            .build(&resolved(), &common(one_shot(resumed("9f22dd1b", false))))
            .unwrap();
        assert_eq!(resume.value_of("--resume"), Some("9f22dd1b"));
        let stripped: Vec<&String> = resume
            .args
            .iter()
            .filter(|a| a.as_str() != "--resume" && a.as_str() != "9f22dd1b")
            .collect();
        assert_eq!(stripped, run.args.iter().collect::<Vec<_>>());

        let launch = driver(&e)
            .build(&resolved(), &common(interactive("/ccc", Origin::Fresh)))
            .unwrap();
        assert!(!launch.has_arg("-p"));
        assert!(!launch.has_arg("--output-format"));
        assert_eq!(launch.value_of("--agent"), Some("explorer"));
        assert_eq!(launch.stdin, None);
        // A human is at the keyboard, so approvals stay interactive.
        assert!(!launch.has_arg("--permission-mode"));

        let mut resume = common(one_shot(resumed("thread-1", false)));
        resume.preset_session_id = Some("must-not-be-used".into());
        let resume = driver(&e).build(&resolved(), &resume).unwrap();
        assert!(!resume.has_arg("--session-id"));
    }

    #[test]
    fn a_roleless_launch_has_no_agent_flags_and_keeps_the_rest() {
        let e = env(&[]);
        let mut r = resolved();
        r.agent = None;
        r.description = String::new();
        r.body = String::new();
        r.tools = None;
        let mut launch = common(interactive("/ccc", Origin::Fresh));
        launch.preset_session_id = Some("launch-preset".into());
        launch.extra_args = vec!["--verbose".into()];
        let i = driver(&e).build(&r, &launch).unwrap();
        assert!(!i.has_arg("--agents"), "{:?}", i.args);
        assert!(!i.has_arg("--agent"), "{:?}", i.args);
        assert_eq!(i.value_of("--model"), Some("claude-haiku-4-5[1m]"));
        assert_eq!(i.value_of("--effort"), Some("low"));
        assert_eq!(i.value_of("--session-id"), Some("launch-preset"));
        assert!(i.has_arg("--settings"));
        assert_eq!(i.args.last().map(String::as_str), Some("--verbose"));

        let resume = driver(&e)
            .build(&r, &common(one_shot(resumed("9f22dd1b", false))))
            .unwrap();
        assert_eq!(&resume.args[0..3], &["-p", "--output-format", "json"]);
        assert_eq!(resume.value_of("--permission-mode"), Some("dontAsk"));
        assert_eq!(resume.value_of("--resume"), Some("9f22dd1b"));
        assert!(!resume.has_arg("--agent"), "{:?}", resume.args);
        assert!(!resume.has_arg("--agents"), "{:?}", resume.args);
    }

    #[test]
    fn quoted_launch_executable_survives_shell_parsing() {
        let e = env(&[]);
        let path = "/tmp/path with 'quote'/ccc";
        let i = driver(&e)
            .build(&resolved(), &common(interactive(path, Origin::Fresh)))
            .unwrap();
        let settings: serde_json::Value =
            serde_json::from_str(i.value_of("--settings").unwrap()).unwrap();
        let command = settings["hooks"]["Stop"][0]["hooks"][0]["command"]
            .as_str()
            .unwrap()
            .to_string();
        let script = format!("set -- {command}; printf '%s' \"$1\"");
        let output = std::process::Command::new("sh")
            .args(["-c", &script, "ccc"])
            .output()
            .unwrap();
        assert_eq!(String::from_utf8(output.stdout).unwrap(), path);
    }

    #[test]
    fn every_cell_carries_exactly_its_flags() {
        let e = env(&[]);
        let origins = [
            Origin::Fresh,
            resumed("9f22dd1b", false),
            resumed("9f22dd1b", true),
        ];
        for origin in origins {
            for mode in [
                one_shot(origin.clone()),
                interactive("/ccc", origin.clone()),
            ] {
                let is_one_shot = !mode.is_launch();
                let is_fresh = mode.origin == Origin::Fresh;
                let is_fork = mode.is_fork();
                let mut c = common(mode.clone());
                c.preset_session_id = Some("preset".into());
                c.extra_args = vec!["--extra".into()];
                let i = driver(&e).build(&resolved(), &c).unwrap();
                let at = |flag: &str| i.args.iter().position(|a| a == flag);

                assert_eq!(i.has_arg("-p"), is_one_shot, "{mode:?}");
                assert_eq!(i.has_arg("--output-format"), is_one_shot, "{mode:?}");
                assert_eq!(
                    i.value_of("--permission-mode"),
                    is_one_shot.then_some("dontAsk"),
                    "{mode:?}"
                );
                assert_eq!(
                    i.value_of("--session-id"),
                    is_fresh.then_some("preset"),
                    "{mode:?}"
                );
                assert_eq!(
                    i.value_of("--resume"),
                    (!is_fresh).then_some("9f22dd1b"),
                    "{mode:?}"
                );
                assert_eq!(i.has_arg("--fork-session"), is_fork, "{mode:?}");
                if is_fork {
                    assert_eq!(at("--fork-session"), at("--resume").map(|p| p + 2));
                }
                assert_eq!(i.has_arg("--settings"), !is_one_shot, "{mode:?}");
                assert_eq!(i.args.last().map(String::as_str), Some("--extra"));
            }
        }
    }

    #[test]
    fn agent_without_tools_or_effort_omits_them() {
        let mut r = resolved();
        r.tools = None;
        r.effort = None;
        let e = env(&[]);
        let i = driver(&e).build(&r, &common(run_mode())).unwrap();
        assert!(!i.has_arg("--effort"));
        let agents: serde_json::Value =
            serde_json::from_str(i.value_of("--agents").unwrap()).unwrap();
        assert!(agents["explorer"].get("tools").is_none());
    }

    #[test]
    fn extra_args_are_appended() {
        let mut c = common(run_mode());
        c.extra_args = vec!["--add-dir".into(), "/srv".into()];
        let e = env(&[]);
        let i = driver(&e).build(&resolved(), &c).unwrap();
        assert_eq!(&i.args[i.args.len() - 2..], &["--add-dir", "/srv"]);
    }

    #[test]
    fn token_auth_sets_the_token_and_removes_the_config_dir() {
        let dir = TempDir::new("claude-token");
        let path = dir.path().join("team03.token");
        dir.write_file("team03.token", "sk-ant-oat-secret\n", 0o600);

        let acct = Account {
            profile: "team03".into(),
            auth: ClaudeAuth::Token,
            token_dir: dir.path().to_path_buf(),
            profiles_dir: PathBuf::from("/nope"),
        };
        assert_eq!(acct.credential_path(), path);
        assert!(acct.has_credential());

        let e = env(&[]);
        let mut d = driver(&e);
        d.account = Some(acct);
        let i = d.build(&resolved(), &common(run_mode())).unwrap();
        assert_eq!(
            i.env_set,
            vec![(TOKEN_VAR.to_string(), "sk-ant-oat-secret".to_string())]
        );
        assert!(i.env_remove.contains(&CONFIG_DIR_VAR.to_string()));
        for var in TOKEN_CONFLICTS {
            assert!(i.env_remove.contains(&var.to_string()), "{var}");
        }
    }

    #[test]
    fn token_auth_refuses_conflicting_env_and_bad_token_files() {
        let dir = TempDir::new("claude-token-bad");
        dir.write_file("team03.token", "secret\n", 0o600);
        let acct = Account {
            profile: "team03".into(),
            auth: ClaudeAuth::Token,
            token_dir: dir.path().to_path_buf(),
            profiles_dir: PathBuf::from("/nope"),
        };
        for var in TOKEN_CONFLICTS {
            let err = acct.env_edits(&env(&[(var, "x")])).unwrap_err().to_string();
            assert!(err.contains(var), "{err}");
        }
        // An empty value is not a conflict.
        assert!(acct.env_edits(&env(&[("ANTHROPIC_API_KEY", "")])).is_ok());

        dir.write_file("loose.token", "secret\n", 0o644);
        let loose = Account {
            profile: "loose".into(),
            ..acct.clone()
        };
        assert!(loose
            .env_edits(&env(&[]))
            .unwrap_err()
            .to_string()
            .contains("mode"));

        dir.write_file("two.token", "a\nb\n", 0o600);
        let two = Account {
            profile: "two".into(),
            ..acct.clone()
        };
        assert!(two
            .env_edits(&env(&[]))
            .unwrap_err()
            .to_string()
            .contains("more than one line"));

        dir.write_file("empty.token", "\n", 0o600);
        let empty = Account {
            profile: "empty".into(),
            ..acct
        };
        assert!(empty
            .env_edits(&env(&[]))
            .unwrap_err()
            .to_string()
            .contains("empty"));
    }

    #[test]
    fn config_dir_auth_sets_the_config_dir_and_removes_the_token() {
        let acct = Account {
            profile: "personal".into(),
            auth: ClaudeAuth::ConfigDir,
            token_dir: PathBuf::from("/tokens"),
            profiles_dir: PathBuf::from("/profiles"),
        };
        assert_eq!(
            acct.credential_path(),
            PathBuf::from("/profiles/personal/.credentials.json")
        );
        let (set, remove) = acct.env_edits(&env(&[("ANTHROPIC_API_KEY", "x")])).unwrap();
        assert_eq!(
            set,
            vec![(CONFIG_DIR_VAR.to_string(), "/profiles/personal".to_string())]
        );
        assert!(remove.contains(&TOKEN_VAR.to_string()));
        // An API key in the parent's environment would outrank the config
        // dir's own credentials, so it is removed here too.
        for var in TOKEN_CONFLICTS {
            assert!(remove.contains(&var.to_string()), "{var}");
        }
    }

    #[test]
    fn account_paths_come_from_the_config_with_the_tilde_expanded() {
        let home = env(&[("HOME", "/home/tester")]);
        let cfg = ClaudeConfig::default();
        let acct = Account::from_config(&cfg, "personal", &home);
        assert_eq!(acct.auth, ClaudeAuth::ConfigDir);
        assert_eq!(
            acct.credential_path(),
            PathBuf::from("/home/tester/.claude-profiles/personal/.credentials.json")
        );
        assert!(!acct.has_credential());

        let cfg = ClaudeConfig {
            auth: ClaudeAuth::Token,
            ..ClaudeConfig::default()
        };
        let acct = Account::from_config(&cfg, "team03", &home);
        assert_eq!(
            acct.credential_path(),
            PathBuf::from("/home/tester/.claude-tokens/team03.token")
        );
    }

    #[test]
    fn parses_a_successful_result() {
        let stdout = r#"{"type":"result","subtype":"success","is_error":false,
            "session_id":"9f22dd1b-1111-2222-3333-444455556666","result":"PROBE-OK hello",
            "num_turns":1,"total_cost_usd":0.0031,"duration_ms":4200,
            "modelUsage":{"claude-haiku-4-5-20251001":{"inputTokens":10}}}"#;
        let out = parse(stdout, "").unwrap();
        assert_eq!(out.session_id, "9f22dd1b-1111-2222-3333-444455556666");
        assert_eq!(out.result, "PROBE-OK hello");
        assert_eq!(out.native["total_cost_usd"], json!(0.0031));
        assert!(out.native["modelUsage"]
            .get("claude-haiku-4-5-20251001")
            .is_some());
    }

    #[test]
    fn is_error_becomes_a_reported_failure() {
        let stdout =
            r#"{"is_error":true,"session_id":"abc","result":"Not logged in","num_turns":0}"#;
        let err = parse(stdout, "").unwrap_err();
        assert_eq!(err.exit_code(), 1);
        assert_eq!(err.message(), "Not logged in");
        assert_eq!(err.session_id(), Some("abc"));
    }

    const LIMIT_MESSAGE: &str = "You've hit your session limit · resets 3pm (America/New_York)";

    #[test]
    fn the_usage_limit_shape_matches_only_limit_failures() {
        assert!(is_usage_limit(LIMIT_MESSAGE));
        assert!(is_usage_limit(&format!("  {LIMIT_MESSAGE}\n")));
        assert!(is_usage_limit("You've hit your weekly limit"));
        assert!(is_usage_limit(
            "Claude usage limit reached. Try again later."
        ));
        assert!(!is_usage_limit("API Error: rate limit exceeded"));
        assert!(!is_usage_limit("Not logged in"));
        assert!(!is_usage_limit("You've hit your stride"));
        assert!(!is_usage_limit(
            "Done. The limit on retries is now enforced, as you've hit your goal."
        ));
    }

    #[test]
    fn a_usage_limit_failure_is_limited_and_keeps_its_session() {
        let stdout = json!({
            "is_error": true,
            "session_id": "abc",
            "result": LIMIT_MESSAGE,
            "num_turns": 1,
        })
        .to_string();
        let err = parse(&stdout, "").unwrap_err();
        assert!(matches!(err, HarnessError::Limited { .. }), "{err:?}");
        assert_eq!(err.exit_code(), crate::harness::EXIT_LIMITED);
        assert_eq!(err.message(), LIMIT_MESSAGE);
        assert_eq!(err.session_id(), Some("abc"));
    }

    #[test]
    fn a_successful_result_that_mentions_a_usage_limit_is_a_success() {
        let stdout = json!({
            "is_error": false,
            "session_id": "abc",
            "result": "You've hit your usage limit check in the parser; fixed.",
        })
        .to_string();
        assert!(parse(&stdout, "").is_ok());
    }

    #[test]
    fn a_failure_without_a_result_still_carries_a_reason() {
        let err = parse(
            r#"{"is_error":true,"session_id":"abc","result":""}"#,
            "auth failed\n",
        )
        .unwrap_err();
        assert_eq!(err.exit_code(), 1);
        assert!(err.message().contains("auth failed"), "{}", err.message());
        assert_eq!(err.session_id(), Some("abc"));

        let err = parse(
            r#"{"is_error":true,"session_id":"abc","result":"","subtype":"error_max_turns","num_turns":7}"#,
            "",
        )
        .unwrap_err();
        assert!(
            err.message().contains("error_max_turns"),
            "{}",
            err.message()
        );
        assert!(err.message().contains('7'), "{}", err.message());
    }

    #[test]
    fn unparsable_output_is_exit_three() {
        let err = parse("claude: command failed", "boom\n").unwrap_err();
        assert_eq!(err.exit_code(), 3);
        assert!(err.message().contains("boom"));
        assert_eq!(parse(r#"{"result":"hi"}"#, "").unwrap_err().exit_code(), 3);
        assert_eq!(
            parse(r#"{"session_id":"a","result":""}"#, "")
                .unwrap_err()
                .exit_code(),
            3
        );
    }

    /// An account of `auth` whose paths live under `dir`.
    fn trust_account(dir: &TempDir, auth: ClaudeAuth) -> Account {
        Account {
            profile: "team03".into(),
            auth,
            token_dir: dir.path().join("tokens"),
            profiles_dir: dir.path().join("profiles"),
        }
    }

    fn read_json(path: &Path) -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn the_config_file_follows_the_accounts_auth_mode() {
        let dir = TempDir::new("claude-config-file");
        let e = env(&[("HOME", dir.path().to_str().unwrap())]);
        let cwd = dir.path().join("work");

        assert_eq!(
            config_file(Some(&trust_account(&dir, ClaudeAuth::Token)), &e, &cwd),
            dir.path().join(".claude.json")
        );
        assert_eq!(
            config_file(Some(&trust_account(&dir, ClaudeAuth::ConfigDir)), &e, &cwd),
            dir.path().join("profiles/team03/.claude.json")
        );
        assert_eq!(config_file(None, &e, &cwd), dir.path().join(".claude.json"));

        // The ambient login keeps an inherited config dir, so its file is the
        // one the child reads; a blank value is no config dir.
        let inherited = dir.path().join("own-login");
        let e = env(&[
            ("HOME", dir.path().to_str().unwrap()),
            (CONFIG_DIR_VAR, inherited.to_str().unwrap()),
        ]);
        assert_eq!(config_file(None, &e, &cwd), inherited.join(".claude.json"));
        assert_eq!(
            config_file(Some(&trust_account(&dir, ClaudeAuth::Token)), &e, &cwd),
            dir.path().join(".claude.json")
        );
        let e = env(&[
            ("HOME", dir.path().to_str().unwrap()),
            (CONFIG_DIR_VAR, " "),
        ]);
        assert_eq!(config_file(None, &e, &cwd), dir.path().join(".claude.json"));

        // A relative inherited dir is the child's, so it resolves against the
        // child's cwd, not ccc's.
        let e = env(&[
            ("HOME", dir.path().to_str().unwrap()),
            (CONFIG_DIR_VAR, ".claude-local"),
        ]);
        assert_eq!(
            config_file(None, &e, &cwd),
            cwd.join(".claude-local/.claude.json")
        );
    }

    #[test]
    fn conflicts_set_names_only_the_non_blank_endpoint_variables() {
        let e = env(&[
            ("ANTHROPIC_API_KEY", "sk-test"),
            ("ANTHROPIC_AUTH_TOKEN", " "),
            ("CLAUDE_CODE_USE_VERTEX", "1"),
        ]);
        assert_eq!(
            conflicts_set(&e),
            vec!["ANTHROPIC_API_KEY", "CLAUDE_CODE_USE_VERTEX"]
        );
        assert!(conflicts_set(&env(&[])).is_empty());
    }

    #[test]
    fn the_ambient_login_marks_trust_in_the_home_config_file() {
        let dir = TempDir::new("claude-trust-ambient");
        let e = env(&[("HOME", dir.path().to_str().unwrap())]);
        let cwd = dir.path().join("work");
        trust_directory(None, &e, &cwd).unwrap();
        let v = read_json(&dir.path().join(".claude.json"));
        assert_eq!(
            v["projects"][cwd.display().to_string()]["hasTrustDialogAccepted"],
            serde_json::json!(true)
        );
    }

    #[test]
    fn trusting_a_directory_creates_the_config_file_when_there_is_none() {
        let dir = TempDir::new("claude-trust-new");
        let e = env(&[("HOME", dir.path().to_str().unwrap())]);
        let acct = trust_account(&dir, ClaudeAuth::ConfigDir);
        std::fs::create_dir_all(dir.path().join("profiles/team03")).unwrap();

        let cwd = dir.path().join("work");
        trust_directory(Some(&acct), &e, &cwd).unwrap();

        let path = dir.path().join("profiles/team03/.claude.json");
        let v = read_json(&path);
        assert_eq!(
            v["projects"][cwd.display().to_string()]["hasTrustDialogAccepted"],
            serde_json::json!(true)
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "{mode:o}");
        }
    }

    #[test]
    fn trusting_a_directory_preserves_everything_else_in_the_config() {
        let dir = TempDir::new("claude-trust-merge");
        let e = env(&[("HOME", dir.path().to_str().unwrap())]);
        let acct = trust_account(&dir, ClaudeAuth::Token);
        let path = dir.write_file(
            ".claude.json",
            &serde_json::to_string_pretty(&json!({
                "numStartups": 7,
                "projects": {
                    "/elsewhere": { "hasTrustDialogAccepted": true },
                    "/work": { "history": ["a"] }
                }
            }))
            .unwrap(),
            0o644,
        );

        trust_directory(Some(&acct), &e, Path::new("/work")).unwrap();

        let v = read_json(&path);
        assert_eq!(v["numStartups"], 7);
        assert_eq!(
            v["projects"]["/elsewhere"]["hasTrustDialogAccepted"],
            json!(true)
        );
        assert_eq!(
            v["projects"]["/work"]["hasTrustDialogAccepted"],
            json!(true)
        );
        assert_eq!(v["projects"]["/work"]["history"], json!(["a"]));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o644, "an existing file keeps its mode: {mode:o}");
        }
    }

    #[test]
    fn an_already_trusted_directory_leaves_the_file_untouched() {
        let dir = TempDir::new("claude-trust-noop");
        let e = env(&[("HOME", dir.path().to_str().unwrap())]);
        let acct = trust_account(&dir, ClaudeAuth::Token);
        // Compact, so a rewrite would be visible: `to_string_pretty` could not
        // produce these bytes.
        let text = "{\"projects\":{\"/work\":{\"hasTrustDialogAccepted\":true}}}";
        let path = dir.write_file(".claude.json", text, 0o600);

        trust_directory(Some(&acct), &e, Path::new("/work")).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
    }

    #[test]
    fn a_config_that_is_not_an_object_is_refused_rather_than_rewritten() {
        let dir = TempDir::new("claude-trust-bad");
        let e = env(&[("HOME", dir.path().to_str().unwrap())]);
        let acct = trust_account(&dir, ClaudeAuth::Token);

        for text in ["[1, 2]", "not json at all"] {
            let path = dir.write_file(".claude.json", text, 0o600);
            assert!(
                trust_directory(Some(&acct), &e, Path::new("/work")).is_err(),
                "{text}"
            );
            assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
        }

        // A `projects` key that is not an object is the same refusal.
        let path = dir.write_file(".claude.json", "{\"projects\": 3}", 0o600);
        let err = trust_directory(Some(&acct), &e, Path::new("/work"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("projects"), "{err}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{\"projects\": 3}");
    }

    #[cfg(unix)]
    #[test]
    fn a_path_that_is_not_utf8_is_refused_rather_than_written_lossily() {
        use std::os::unix::ffi::OsStrExt;

        let dir = TempDir::new("claude-trust-nonutf8");
        let e = env(&[("HOME", dir.path().to_str().unwrap())]);
        let acct = trust_account(&dir, ClaudeAuth::Token);
        let cwd = PathBuf::from(std::ffi::OsStr::from_bytes(b"/work/\xff"));

        let err = trust_directory(Some(&acct), &e, &cwd)
            .unwrap_err()
            .to_string();
        assert!(err.contains("UTF-8"), "{err}");
        assert!(!dir.path().join(".claude.json").exists());
    }

    #[test]
    fn the_home_directory_is_never_written_because_its_trust_is_session_only() {
        let dir = TempDir::new("claude-trust-home");
        let e = env(&[("HOME", dir.path().to_str().unwrap())]);
        let acct = trust_account(&dir, ClaudeAuth::Token);

        trust_directory(Some(&acct), &e, dir.path()).unwrap();
        assert!(!dir.path().join(".claude.json").exists());
    }

    #[cfg(unix)]
    #[test]
    fn a_home_reached_through_a_symlink_is_still_the_home_directory() {
        let dir = TempDir::new("claude-trust-home-link");
        let real = dir.path().join("real-home");
        std::fs::create_dir_all(&real).unwrap();
        let link = dir.path().join("linked-home");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let e = env(&[("HOME", link.to_str().unwrap())]);
        let acct = trust_account(&dir, ClaudeAuth::Token);

        // `cwd` arrives canonical (the contract of `resolve_cwd`), so it is
        // the link's target, not the literal value of `HOME`.
        trust_directory(Some(&acct), &e, &std::fs::canonicalize(&real).unwrap()).unwrap();
        assert!(!link.join(".claude.json").exists());
        assert!(!real.join(".claude.json").exists());
    }
}
