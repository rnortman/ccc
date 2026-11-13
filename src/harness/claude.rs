//! The Claude Code driver: `claude -p --output-format json` with the agent
//! supplied inline through `--agents`/`--agent`.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use crate::audit::HookKind;
use crate::config::{expand_tilde_in, ClaudeAuth, ClaudeConfig, EnvSource, PermissionMode};
use crate::harness::{
    binary_in, check_arg_len, tail, Captured, Common, Driver, HarnessError, Invocation, Io,
    LimitInfo, Origin, Outcome,
};
use crate::resolve::Resolved;

/// Environment variables that must not be set when an OAuth token is used;
/// any of them would send the child to a different endpoint or credential.
const TOKEN_CONFLICTS: [&str; 5] = [
    API_KEY_VAR,
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
];

const TOKEN_VAR: &str = "CLAUDE_CODE_OAUTH_TOKEN";
/// The environment variable that carries a Platform API key to a Claude child.
pub const API_KEY_VAR: &str = "ANTHROPIC_API_KEY";
const CONFIG_DIR_VAR: &str = "CLAUDE_CONFIG_DIR";

/// The native field `parse` writes with the process's cost in USD: the sum
/// of the costs its `result` events state, or `null` when none states one.
const COST_FIELD: &str = "turn_cost_usd";

/// The cost in USD a parsed turn's native fields report, or `None` when they
/// report none.
pub fn reported_cost(native: &serde_json::Value) -> Option<f64> {
    native.get(COST_FIELD)?.as_f64()
}

/// Variables to set, and variables to remove, for one account.
pub type EnvEdits = (Vec<(String, String)>, Vec<String>);

/// Which account the child runs as, and how it is pointed at it. `profile`
/// is the profile's name, or the key's name for an API key account.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Account {
    pub profile: String,
    pub auth: ClaudeAuth,
    pub token_dir: PathBuf,
    pub profiles_dir: PathBuf,
    /// The key file of an API key account; `None` for a login. Set only by
    /// [`Account::for_key`], together with `auth = ApiKey`.
    key_file: Option<PathBuf>,
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
            key_file: None,
        }
    }

    /// The API key account `[claude.api_keys.<name>]` describes, or `None`
    /// when no key of that name is configured.
    pub fn for_key(cfg: &ClaudeConfig, name: &str, env: &dyn EnvSource) -> Option<Account> {
        let key = cfg.api_keys.get(name)?;
        Some(Account {
            profile: name.to_string(),
            auth: ClaudeAuth::ApiKey,
            token_dir: expand_tilde_in(env, &cfg.token_dir),
            profiles_dir: expand_tilde_in(env, &cfg.profiles_dir),
            key_file: Some(expand_tilde_in(env, &key.key_file)),
        })
    }

    /// The key file of an API key account, `None` for a login: the one test
    /// of whether this account spends a key's budget.
    pub fn key_file(&self) -> Option<&Path> {
        self.key_file.as_deref()
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
            ClaudeAuth::ApiKey => self.key_file.clone().unwrap_or_default(),
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
        let what = match self.auth {
            ClaudeAuth::ApiKey => "API key",
            ClaudeAuth::Token | ClaudeAuth::ConfigDir => "profile",
        };
        Err(format!(
            "{what} {} has no credential at {}",
            self.profile,
            self.credential_path().display()
        ))
    }

    /// The environment edits for this account: exactly one account variable
    /// is set and every other way of naming an account or endpoint is removed.
    pub fn env_edits(&self) -> Result<EnvEdits> {
        match self.auth {
            // An inherited endpoint or credential variable is removed, not
            // refused: a key turn's child holds `ANTHROPIC_API_KEY`, and a
            // nested `ccc` started from it inherits that.
            ClaudeAuth::Token => {
                let path = self.credential_path();
                let token = read_token(&path, "token file")?;
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
            // TODO(api-key-assumptions-unconfirmed): A1 — that `claude -p` bills
            // this key, not the subscription login in `~/.claude`, has not been
            // checked against a real key.
            ClaudeAuth::ApiKey => {
                // An inherited key is overwritten: the config names the key.
                // Without `CLAUDE_CONFIG_DIR` the child uses `~/.claude`.
                let key = read_token(&self.credential_path(), "key file")?;
                let remove = TOKEN_CONFLICTS
                    .iter()
                    .filter(|v| **v != API_KEY_VAR)
                    .chain([TOKEN_VAR, CONFIG_DIR_VAR].iter())
                    .map(|v| v.to_string())
                    .collect();
                Ok((vec![(API_KEY_VAR.to_string(), key)], remove))
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
/// Claude spawn removes.
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

/// Reads a secret file — a `~/.claude-tokens/<profile>.token` or an API key
/// file, named `what` in errors: mode 0600, one non-empty line.
fn read_token(path: &Path, what: &str) -> Result<String> {
    // Permissions are checked before the secret is read, so a world-readable
    // token never enters this process.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(path)
            .with_context(|| format!("reading {what} {}", path.display()))?
            .permissions()
            .mode()
            & 0o777;
        if mode & 0o077 != 0 {
            bail!(
                "{what} {} is mode {mode:o}; it must not be readable by group or others (chmod 600)",
                path.display()
            );
        }
    }

    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {what} {}", path.display()))?;

    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let Some(token) = lines.next() else {
        bail!("{what} {} is empty", path.display());
    };
    if lines.next().is_some() {
        bail!("{what} {} has more than one line", path.display());
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
        args.extend(["-p", "--output-format", "stream-json", "--verbose"].map(String::from));
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
        Some(acct) => acct.env_edits()?,
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
/// usage limit (`You've hit your session limit · resets …`, `You've reached
/// your Fable limit. …`, or a wording naming a usage limit).
pub fn is_usage_limit(message: &str) -> bool {
    let message = message.trim();
    ((message.starts_with("You've hit your") || message.starts_with("You've reached your"))
        && message.contains("limit"))
        || message.contains("usage limit")
}

/// The events of one `--output-format stream-json` run that `parse` reads.
#[derive(Default)]
struct Events {
    /// The last `result` event: the turn's outcome.
    result: Option<Value>,
    /// The sum of the costs stated by the `result` events that state one
    /// ([`stated_cost`]), or `None` when none does.
    cost: Option<f64>,
    /// The `rate_limit_info` of the last `rate_limit_event` whose status is
    /// `rejected`.
    rejected: Option<Value>,
    /// The `api_error_params.rate_limit_info` of the last `assistant` event
    /// whose `error` is `rate_limit`, whatever its status.
    assistant_limit: Option<Value>,
}

impl Events {
    /// Reads every line that is a JSON object; skips the rest.
    fn read(stdout: &str) -> Events {
        let mut ev = Events::default();
        for line in stdout.lines() {
            let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(line.trim()) else {
                continue;
            };
            let value = Value::Object(obj);
            match value.get("type").and_then(Value::as_str) {
                Some("result") => {
                    if let Some(cost) = stated_cost(&value) {
                        ev.cost = Some(ev.cost.unwrap_or(0.0) + cost);
                    }
                    ev.result = Some(value);
                }
                Some("rate_limit_event") => {
                    if let Some(info) = value.get("rate_limit_info") {
                        if is_rejected(info) {
                            ev.rejected = Some(info.clone());
                        }
                    }
                }
                Some("assistant")
                    if value.get("error").and_then(Value::as_str) == Some("rate_limit") =>
                {
                    ev.assistant_limit =
                        value.pointer("/api_error_params/rate_limit_info").cloned();
                }
                _ => {}
            }
        }
        ev
    }

    /// The provider's refusal: the last rejected rate-limit event, else the
    /// rate-limit object of the last rate-limited assistant message when it
    /// is rejected.
    fn refusal(&self) -> Option<&Value> {
        self.rejected
            .as_ref()
            .or_else(|| self.assistant_limit.as_ref().filter(|i| is_rejected(i)))
    }
}

/// The cost a `result` event states for the calls its process made since its
/// previous result: the sum of `costUSD` over `model_usage_since_last_result`.
/// `{}` states 0; an absent field, or an entry without a numeric `costUSD` or
/// with a negative one, states none. Claude Code states each call on at most
/// one result, across resumes, so these sum without double counting;
/// `total_cost_usd` is a running total that a resumed process restores from
/// its transcript.
fn stated_cost(result: &Value) -> Option<f64> {
    result
        .get("model_usage_since_last_result")?
        .as_object()?
        .values()
        .try_fold(0.0, |total, entry| {
            let cost = entry.get("costUSD")?.as_f64()?;
            (cost.is_finite() && cost >= 0.0).then_some(total + cost)
        })
}

fn is_rejected(info: &Value) -> bool {
    info.get("status").and_then(Value::as_str) == Some("rejected")
}

/// The limit a refusal names: its `rateLimitType` and `resetsAt`.
fn limit_info(refusal: &Value) -> LimitInfo {
    let resets_at = refusal.get("resetsAt").and_then(|v| {
        v.as_i64()
            .or_else(|| v.as_f64().map(|f| f as i64))
            .and_then(|s| jiff::Timestamp::from_second(s).ok())
    });
    LimitInfo {
        kind: refusal
            .get("rateLimitType")
            .and_then(Value::as_str)
            .map(String::from),
        resets_at,
    }
}

// TODO(claude-stream-liveness): the events are read once, after the child
// exits; a refused turn waits for any background command the agent left
// running.
/// Parses the event lines of one `--output-format stream-json --verbose`
/// run. The last `result` event is the outcome; its raw fields are kept in
/// `native`, beside `turn_cost_usd`, the cost every result of the process
/// stated. A failed result is a usage
/// limit when it carries `api_error_status` 429 or its message reads as one
/// ([`is_usage_limit`]); the limit names the provider's refusal when the
/// stream reported one.
fn parse(stdout: &str, stderr: &str) -> Result<Outcome, HarnessError> {
    let events = Events::read(stdout);
    let Some(value) = &events.result else {
        return Err(HarnessError::unusable(format!(
            "claude produced no result event; stderr: {}",
            tail(stderr, 5)
        )));
    };
    let refusal = events.refusal();

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
        COST_FIELD: events.cost,
        "total_cost_usd": value.get("total_cost_usd"),
        "duration_ms": value.get("duration_ms"),
        "modelUsage": value.get("modelUsage"),
        "terminal_reason": value.get("terminal_reason"),
        "api_error_status": value.get("api_error_status"),
        "api_error": value.get("api_error"),
        "rate_limit_info": refusal,
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
        let status_429 = value.get("api_error_status").and_then(Value::as_u64) == Some(429);
        if status_429 || is_usage_limit(&message) {
            return Err(HarnessError::Limited {
                session_id,
                message,
                native,
                warnings: Vec::new(),
                limit: refusal.map(|r| Box::new(limit_info(r))),
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
            "claude returned a result without a session_id".to_string(),
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
            goose: None,
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
        assert_eq!(
            &i.args[0..4],
            &["-p", "--output-format", "stream-json", "--verbose"]
        );
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
        assert_eq!(
            &resume.args[0..4],
            &["-p", "--output-format", "stream-json", "--verbose"]
        );
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
            key_file: None,
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
    fn token_auth_removes_inherited_conflicts_and_refuses_bad_token_files() {
        let dir = TempDir::new("claude-token-bad");
        dir.write_file("team03.token", "secret\n", 0o600);
        let acct = Account {
            profile: "team03".into(),
            auth: ClaudeAuth::Token,
            token_dir: dir.path().to_path_buf(),
            profiles_dir: PathBuf::from("/nope"),
            key_file: None,
        };
        // As a nested run under a key turn inherits it: the child spawns on
        // the token with the variable removed.
        for var in TOKEN_CONFLICTS {
            let e = env(&[(var, "x")]);
            let mut d = driver(&e);
            d.account = Some(acct.clone());
            let i = d.build(&resolved(), &common(run_mode())).unwrap();
            assert_eq!(
                i.env_set,
                vec![(TOKEN_VAR.to_string(), "secret".to_string())]
            );
            assert!(i.env_remove.contains(&var.to_string()), "{var}");
        }

        dir.write_file("loose.token", "secret\n", 0o644);
        let loose = Account {
            profile: "loose".into(),
            ..acct.clone()
        };
        assert!(loose.env_edits().unwrap_err().to_string().contains("mode"));

        dir.write_file("two.token", "a\nb\n", 0o600);
        let two = Account {
            profile: "two".into(),
            ..acct.clone()
        };
        assert!(two
            .env_edits()
            .unwrap_err()
            .to_string()
            .contains("more than one line"));

        dir.write_file("empty.token", "\n", 0o600);
        let empty = Account {
            profile: "empty".into(),
            ..acct
        };
        assert!(empty.env_edits().unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn config_dir_auth_sets_the_config_dir_and_removes_the_token() {
        let acct = Account {
            profile: "personal".into(),
            auth: ClaudeAuth::ConfigDir,
            token_dir: PathBuf::from("/tokens"),
            profiles_dir: PathBuf::from("/profiles"),
            key_file: None,
        };
        assert_eq!(
            acct.credential_path(),
            PathBuf::from("/profiles/personal/.credentials.json")
        );
        let (set, remove) = acct.env_edits().unwrap();
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

    /// A config with one key, `team`, whose key file is `<dir>/team.key`.
    fn key_config(dir: &TempDir) -> ClaudeConfig {
        let mut cfg = ClaudeConfig {
            auth: ClaudeAuth::Token,
            ..ClaudeConfig::default()
        };
        let mut key = crate::config::ApiKeyConfig::new(
            &dir.path().join("team.key").display().to_string(),
            420.0,
            &["*fable*"],
        );
        key.exhausted_patterns = vec!["*credit*".into()];
        cfg.api_keys.insert("team".into(), key);
        cfg
    }

    #[test]
    fn a_key_account_comes_only_from_a_configured_key() {
        let dir = TempDir::new("claude-key-for");
        let cfg = key_config(&dir);
        let acct = Account::for_key(&cfg, "team", &env(&[])).unwrap();
        assert_eq!(acct.profile, "team");
        assert_eq!(acct.auth, ClaudeAuth::ApiKey);
        assert_eq!(acct.credential_path(), dir.path().join("team.key"));
        assert!(!acct.has_credential());
        let missing = acct.require_credential().unwrap_err();
        assert!(
            missing.starts_with("API key team has no credential at"),
            "{missing}"
        );
        assert_eq!(Account::for_key(&cfg, "team03", &env(&[])), None);

        let mut tilde = cfg.clone();
        tilde.api_keys.get_mut("team").unwrap().key_file = "~/keys/team.key".into();
        let acct = Account::for_key(&tilde, "team", &env(&[("HOME", "/home/tester")])).unwrap();
        assert_eq!(
            acct.credential_path(),
            PathBuf::from("/home/tester/keys/team.key")
        );
    }

    #[test]
    fn key_auth_sets_the_key_and_removes_every_other_account_variable() {
        let dir = TempDir::new("claude-key-env");
        dir.write_file("team.key", "sk-ant-api-secret\n", 0o600);
        let acct = Account::for_key(&key_config(&dir), "team", &env(&[])).unwrap();
        let (set, mut remove) = acct.env_edits().unwrap();
        assert_eq!(
            set,
            vec![(API_KEY_VAR.to_string(), "sk-ant-api-secret".to_string())]
        );
        remove.sort();
        assert_eq!(
            remove,
            [
                "ANTHROPIC_AUTH_TOKEN",
                TOKEN_VAR,
                "CLAUDE_CODE_USE_BEDROCK",
                "CLAUDE_CODE_USE_FOUNDRY",
                "CLAUDE_CODE_USE_VERTEX",
                CONFIG_DIR_VAR,
            ]
        );
    }

    #[test]
    fn key_auth_applies_the_secret_file_rules() {
        let dir = TempDir::new("claude-key-bad");
        let cfg = key_config(&dir);
        let acct = Account::for_key(&cfg, "team", &env(&[])).unwrap();
        for (contents, mode, expected) in [
            ("secret\n", 0o644, "key file"),
            ("secret\n", 0o644, "mode 644"),
            ("a\nb\n", 0o600, "more than one line"),
            ("\n", 0o600, "empty"),
        ] {
            dir.write_file("team.key", contents, mode);
            let err = format!("{:#}", acct.env_edits().unwrap_err());
            assert!(err.contains(expected), "{expected}: {err}");
        }
        std::fs::remove_file(dir.path().join("team.key")).unwrap();
        let err = format!("{:#}", acct.env_edits().unwrap_err());
        assert!(err.contains("reading key file"), "{err}");
    }

    #[test]
    fn a_key_account_trusts_the_directory_in_the_home_config_file() {
        let dir = TempDir::new("claude-key-trust");
        let acct = Account::for_key(&key_config(&dir), "team", &env(&[])).unwrap();
        let home = dir.path().join("home");
        let e = env(&[
            ("HOME", home.to_str().unwrap()),
            ("CLAUDE_CONFIG_DIR", "/elsewhere"),
        ]);
        assert_eq!(
            config_file(Some(&acct), &e, Path::new("/srv/work")),
            home.join(".claude.json")
        );
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
    fn reported_cost_reads_a_refused_result_and_is_none_without_one() {
        let refused = r#"{"type":"result","is_error":true,"result":"no","session_id":"9f22dd1b-1111-2222-3333-444455556666","total_cost_usd":0.5,"model_usage_since_last_result":{"m":{"costUSD":0.5}}}"#;
        let err = parse(refused, "").unwrap_err();
        assert_eq!(reported_cost(err.reported().unwrap().1), Some(0.5));
        let free = r#"{"type":"result","is_error":false,"result":"ok","session_id":"9f22dd1b-1111-2222-3333-444455556666"}"#;
        assert_eq!(reported_cost(&parse(free, "").unwrap().native), None);
    }

    #[test]
    fn parses_a_successful_result() {
        let stdout = json!({"type":"result","subtype":"success","is_error":false,
            "session_id":"9f22dd1b-1111-2222-3333-444455556666","result":"PROBE-OK hello",
            "num_turns":1,"total_cost_usd":4.2,"duration_ms":4200,
            "modelUsage":{"claude-haiku-4-5-20251001":{"inputTokens":10,"costUSD":4.2}},
            "model_usage_since_last_result":{"claude-haiku-4-5-20251001":{"costUSD":0.0031}}})
        .to_string();
        let stdout = stdout.as_str();
        let out = parse(stdout, "").unwrap();
        assert_eq!(out.session_id, "9f22dd1b-1111-2222-3333-444455556666");
        assert_eq!(out.result, "PROBE-OK hello");
        assert_eq!(out.native["turn_cost_usd"], json!(0.0031));
        assert_eq!(reported_cost(&out.native), Some(0.0031));
        assert_eq!(out.native["total_cost_usd"], json!(4.2));
        assert!(out.native["modelUsage"]
            .get("claude-haiku-4-5-20251001")
            .is_some());
    }

    #[test]
    fn the_stated_cost_sums_models_and_reads_an_empty_object_as_zero() {
        let two = result_line(json!({"is_error": false, "result": "ok",
            "model_usage_since_last_result": {"a": {"costUSD": 0.25}, "b": {"costUSD": 0.5}}}));
        assert_eq!(reported_cost(&parse(&two, "").unwrap().native), Some(0.75));
        let empty = result_line(json!({"is_error": false, "result": "ok",
            "total_cost_usd": 3.0, "model_usage_since_last_result": {}}));
        assert_eq!(reported_cost(&parse(&empty, "").unwrap().native), Some(0.0));
    }

    #[test]
    fn a_result_without_a_stated_cost_reports_none() {
        let absent = result_line(json!({"is_error": false, "result": "ok", "total_cost_usd": 0.4}));
        let out = parse(&absent, "").unwrap();
        assert_eq!(out.native["turn_cost_usd"], Value::Null);
        assert_eq!(reported_cost(&out.native), None);
        assert_eq!(out.native["total_cost_usd"], json!(0.4));
        let partial = result_line(json!({"is_error": false, "result": "ok",
            "model_usage_since_last_result": {"a": {"costUSD": 0.1}, "b": {"inputTokens": 3}}}));
        assert_eq!(reported_cost(&parse(&partial, "").unwrap().native), None);
        let negative = result_line(json!({"is_error": false, "result": "ok",
            "model_usage_since_last_result": {"a": {"costUSD": 0.5}, "b": {"costUSD": -0.3}}}));
        assert_eq!(reported_cost(&parse(&negative, "").unwrap().native), None);
    }

    #[test]
    fn every_result_of_a_process_is_summed_and_the_last_is_passed_through() {
        let first = result_line(json!({"is_error": false, "result": "one",
            "total_cost_usd": 7.02, "model_usage_since_last_result": {"m": {"costUSD": 0.02}}}));
        let second = result_line(json!({"is_error": false, "result": "two",
            "total_cost_usd": 7.05, "model_usage_since_last_result": {"m": {"costUSD": 0.03}},
            "modelUsage": {"m": {"costUSD": 7.05}}}));
        let out = parse(&format!("{first}\n{second}\n"), "").unwrap();
        assert_eq!(out.result, "two");
        let cost = reported_cost(&out.native).unwrap();
        assert!((cost - 0.05).abs() < 1e-12, "{cost}");
        assert_eq!(out.native["total_cost_usd"], json!(7.05));
        assert_eq!(out.native["modelUsage"]["m"]["costUSD"], json!(7.05));
    }

    #[test]
    fn a_result_stating_no_cost_adds_nothing_to_one_that_does() {
        let first = result_line(json!({"is_error": false, "result": "one",
            "model_usage_since_last_result": {"m": {"costUSD": 0.02}}}));
        let second =
            result_line(json!({"is_error": false, "result": "two", "total_cost_usd": 9.0}));
        let out = parse(&format!("{first}\n{second}\n"), "").unwrap();
        assert_eq!(reported_cost(&out.native), Some(0.02));
    }

    #[test]
    fn is_error_becomes_a_reported_failure() {
        let stdout = r#"{"type":"result","is_error":true,"session_id":"abc","result":"Not logged in","num_turns":0}"#;
        let err = parse(stdout, "").unwrap_err();
        assert_eq!(err.exit_code(), 1);
        assert_eq!(err.message(), "Not logged in");
        assert_eq!(err.session_id(), Some("abc"));
    }

    const LIMIT_MESSAGE: &str = "You've hit your session limit · resets 3pm (America/New_York)";
    const FABLE_MESSAGE: &str =
        "You've reached your Fable limit. Switch to another model to continue.";
    const FABLE_STREAM: &str = include_str!("testdata/claude-stream-fable-limit.jsonl");
    const OK_STREAM: &str = include_str!("testdata/claude-stream-ok.jsonl");

    fn limit_of(err: &HarnessError) -> Option<&LimitInfo> {
        match err {
            HarnessError::Limited { limit, .. } => limit.as_deref(),
            other => panic!("not limited: {other:?}"),
        }
    }

    #[test]
    fn a_refused_fable_stream_is_limited_with_the_providers_reset() {
        let err = parse(FABLE_STREAM, "").unwrap_err();
        assert_eq!(err.exit_code(), crate::harness::EXIT_LIMITED);
        assert_eq!(err.message(), FABLE_MESSAGE);
        assert_eq!(
            err.session_id(),
            Some("c7e95cd7-20b2-4454-9851-fbba6d6a9777")
        );
        let limit = limit_of(&err).expect("the stream reported its refusal");
        assert_eq!(limit.kind.as_deref(), Some("seven_day_overage_included"));
        assert_eq!(
            limit.resets_at,
            Some("2026-11-13T04:00:00Z".parse().unwrap())
        );
        let native = err.reported().unwrap().1;
        assert_eq!(reported_cost(native), Some(0.0));
        assert_eq!(native["api_error_status"], json!(429));
        assert_eq!(native["api_error"], json!("model_requires_usage_credits"));
        assert_eq!(native["terminal_reason"], json!("api_error"));
        assert_eq!(native["rate_limit_info"]["status"], json!("rejected"));
        assert_eq!(
            native["rate_limit_info"]["rateLimitType"],
            json!("seven_day_overage_included")
        );
    }

    #[test]
    fn a_successful_stream_is_ok_with_its_cost() {
        let out = parse(OK_STREAM, "").unwrap();
        assert_eq!(out.session_id, "9db845f0-22b5-4353-8899-4f2d2269c816");
        assert_eq!(out.result, "hi");
        let cost = reported_cost(&out.native).unwrap();
        assert!((cost - 0.0275102).abs() < 1e-12, "{cost}");
        assert!(out.native["total_cost_usd"].is_number());
        assert!(out.native["modelUsage"].is_object());
        assert_eq!(out.native["terminal_reason"], json!("completed"));
        assert_eq!(out.native["api_error_status"], Value::Null);
        // Its `allowed_warning` rate-limit event is not a refusal.
        assert_eq!(out.native["rate_limit_info"], Value::Null);
    }

    fn result_line(fields: Value) -> String {
        let mut v = json!({"type": "result", "session_id": "abc", "num_turns": 1});
        for (k, f) in fields.as_object().unwrap() {
            v[k] = f.clone();
        }
        v.to_string()
    }

    #[test]
    fn a_429_without_a_rate_limit_event_is_limited_with_no_limit() {
        let stdout = result_line(json!({
            "is_error": true, "result": "API Error: something", "api_error_status": 429,
        }));
        let err = parse(&stdout, "").unwrap_err();
        assert_eq!(limit_of(&err), None);
        assert_eq!(err.message(), "API Error: something");
    }

    #[test]
    fn a_limit_message_without_a_status_is_limited_with_no_limit() {
        for message in [FABLE_MESSAGE, LIMIT_MESSAGE] {
            let stdout = result_line(json!({"is_error": true, "result": message}));
            let err = parse(&stdout, "").unwrap_err();
            assert_eq!(limit_of(&err), None, "{message}");
        }
    }

    #[test]
    fn another_status_with_another_message_is_a_reported_failure() {
        let stdout = result_line(json!({
            "is_error": true, "result": "API Error: overloaded", "api_error_status": 529,
        }));
        let err = parse(&stdout, "").unwrap_err();
        assert!(matches!(err, HarnessError::Reported { .. }), "{err:?}");
        assert_eq!(err.reported().unwrap().1["api_error_status"], json!(529));
    }

    #[test]
    fn a_rejected_assistant_message_stands_in_for_a_missing_rate_limit_event() {
        let assistant = |status: &str| {
            json!({"type": "assistant", "error": "rate_limit", "api_error_params": {
                "rate_limit_info": {"status": status, "resetsAt": 1794448200,
                                    "rateLimitType": "five_hour"}}})
            .to_string()
        };
        let result = result_line(json!({
            "is_error": true, "result": "x", "api_error_status": 429,
        }));
        let err = parse(&format!("{}\n{result}\n", assistant("rejected")), "").unwrap_err();
        let limit = limit_of(&err).unwrap();
        assert_eq!(limit.kind.as_deref(), Some("five_hour"));
        assert_eq!(
            limit.resets_at,
            Some(jiff::Timestamp::from_second(1794448200).unwrap())
        );

        // Only the last such message counts, and only when it is rejected.
        let stdout = format!(
            "{}\n{}\n{result}",
            assistant("rejected"),
            assistant("allowed")
        );
        assert_eq!(limit_of(&parse(&stdout, "").unwrap_err()), None);

        // A rejected rate-limit event outranks the assistant message.
        let event = json!({"type": "rate_limit_event", "rate_limit_info":
            {"status": "rejected", "rateLimitType": "seven_day"}})
        .to_string();
        let stdout = format!("{event}\n{}\n{result}", assistant("rejected"));
        let limit = limit_of(&parse(&stdout, "").unwrap_err()).cloned().unwrap();
        assert_eq!(limit.kind.as_deref(), Some("seven_day"));
        assert_eq!(limit.resets_at, None);

        // Fractional seconds are truncated, not dropped.
        let event = json!({"type": "rate_limit_event", "rate_limit_info":
            {"status": "rejected", "resetsAt": 1794448200.5}})
        .to_string();
        let limit = limit_of(&parse(&format!("{event}\n{result}"), "").unwrap_err())
            .cloned()
            .unwrap();
        assert_eq!(
            limit.resets_at,
            Some(jiff::Timestamp::from_second(1794448200).unwrap())
        );
    }

    #[test]
    fn the_last_result_event_is_the_outcome_and_other_lines_are_ignored() {
        let first = result_line(json!({"is_error": true, "result": "first"}));
        let last = result_line(json!({"is_error": false, "result": "second", "result_index": 1}));
        let stdout = format!("not json\n{first}\n[1,2]\n\n{last}\n{{\"type\":\"system\"}}\n");
        let out = parse(&stdout, "").unwrap();
        assert_eq!(out.result, "second");
    }

    #[test]
    fn a_stream_without_a_result_event_is_unusable() {
        let stdout = json!({"type": "system", "subtype": "init", "session_id": "abc"}).to_string();
        let err = parse(&stdout, "boom\n").unwrap_err();
        assert_eq!(err.exit_code(), 3);
        assert!(
            err.message().contains("no result event"),
            "{}",
            err.message()
        );
        assert!(err.message().contains("boom"), "{}", err.message());
    }

    #[test]
    fn the_usage_limit_shape_matches_only_limit_failures() {
        assert!(is_usage_limit(LIMIT_MESSAGE));
        assert!(is_usage_limit(&format!("  {LIMIT_MESSAGE}\n")));
        assert!(is_usage_limit("You've hit your weekly limit"));
        assert!(is_usage_limit(FABLE_MESSAGE));
        assert!(!is_usage_limit("You've reached your destination"));
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
            "type": "result",
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
            "type": "result",
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
            r#"{"type":"result","is_error":true,"session_id":"abc","result":""}"#,
            "auth failed\n",
        )
        .unwrap_err();
        assert_eq!(err.exit_code(), 1);
        assert!(err.message().contains("auth failed"), "{}", err.message());
        assert_eq!(err.session_id(), Some("abc"));

        let err = parse(
            r#"{"type":"result","is_error":true,"session_id":"abc","result":"","subtype":"error_max_turns","num_turns":7}"#,
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
        assert_eq!(
            parse(r#"{"type":"result","result":"hi"}"#, "")
                .unwrap_err()
                .exit_code(),
            3
        );
        assert_eq!(
            parse(r#"{"type":"result","session_id":"a","result":""}"#, "")
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
            key_file: None,
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
