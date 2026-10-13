//! Session config: the site config (`CCC_CONFIG` or
//! `~/.config/ccc/config.toml`) and the agents config (`CCC_AGENTS_CONFIG` or
//! `~/.config/ccc/agents.toml`), which hold disjoint keys, plus the
//! `CCC_*_<AGENT>` / `CCC_*_ALL` environment overrides.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use claude_usage_mqtt_core::SEVEN_DAY;
use serde::Deserialize;

/// Which harness runs an agent.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Deserialize, serde::Serialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Harness {
    #[default]
    Claude,
    Codex,
}

impl Harness {
    pub const ALL: [Harness; 2] = [Harness::Claude, Harness::Codex];
    pub fn as_str(self) -> &'static str {
        match self {
            Harness::Claude => "claude",
            Harness::Codex => "codex",
        }
    }

    pub fn parse(s: &str) -> Result<Harness> {
        match s.trim().to_ascii_lowercase().as_str() {
            "claude" => Ok(Harness::Claude),
            "codex" => Ok(Harness::Codex),
            other => anyhow::bail!("unknown harness {other:?} (expected \"claude\" or \"codex\")"),
        }
    }
}

impl std::fmt::Display for Harness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How the Claude harness is pointed at an account.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ClaudeAuth {
    Token,
    ConfigDir,
}

impl ClaudeAuth {
    pub fn as_str(self) -> &'static str {
        match self {
            ClaudeAuth::Token => "token",
            ClaudeAuth::ConfigDir => "config-dir",
        }
    }
}

impl std::fmt::Display for ClaudeAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How much a Claude `-p` run may do without asking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub enum PermissionMode {
    AcceptEdits,
    Auto,
    BypassPermissions,
    Manual,
    DontAsk,
    Plan,
}

impl PermissionMode {
    pub fn as_str(self) -> &'static str {
        match self {
            PermissionMode::AcceptEdits => "acceptEdits",
            PermissionMode::Auto => "auto",
            PermissionMode::BypassPermissions => "bypassPermissions",
            PermissionMode::Manual => "manual",
            PermissionMode::DontAsk => "dontAsk",
            PermissionMode::Plan => "plan",
        }
    }
}

impl std::fmt::Display for PermissionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Codex `approval_policy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Approval {
    Untrusted,
    OnFailure,
    OnRequest,
    Never,
}

impl Approval {
    pub fn as_str(self) -> &'static str {
        match self {
            Approval::Untrusted => "untrusted",
            Approval::OnFailure => "on-failure",
            Approval::OnRequest => "on-request",
            Approval::Never => "never",
        }
    }
}

impl std::fmt::Display for Approval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Reasoning effort, as accepted by Claude's `--effort` and Codex's
/// `model_reasoning_effort`; the union of the two sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Effort {
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl Effort {
    pub fn as_str(self) -> &'static str {
        match self {
            Effort::Minimal => "minimal",
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::XHigh => "xhigh",
            Effort::Max => "max",
        }
    }

    pub fn parse(s: &str) -> Result<Effort> {
        match s.trim().to_ascii_lowercase().as_str() {
            "minimal" => Ok(Effort::Minimal),
            "low" => Ok(Effort::Low),
            "medium" => Ok(Effort::Medium),
            "high" => Ok(Effort::High),
            "xhigh" => Ok(Effort::XHigh),
            "max" => Ok(Effort::Max),
            other => anyhow::bail!(
                "unknown effort {other:?} (expected minimal, low, medium, high, xhigh or max)"
            ),
        }
    }

    /// The efforts one harness accepts: Claude's `--effort` takes `low` through
    /// `max`, Codex's `model_reasoning_effort` takes `minimal` through `xhigh`.
    pub fn accepted_by(harness: Harness) -> &'static [Effort] {
        match harness {
            Harness::Claude => &[
                Effort::Low,
                Effort::Medium,
                Effort::High,
                Effort::XHigh,
                Effort::Max,
            ],
            Harness::Codex => &[
                Effort::Minimal,
                Effort::Low,
                Effort::Medium,
                Effort::High,
                Effort::XHigh,
            ],
        }
    }

    /// Parses `s` and rejects a value the resolved harness does not accept,
    /// so a typo costs no spawn.
    pub fn parse_for(s: &str, harness: Harness) -> Result<Effort> {
        let e = Effort::parse(s)?;
        let accepted = Effort::accepted_by(harness);
        if !accepted.contains(&e) {
            let names: Vec<&str> = accepted.iter().map(|e| e.as_str()).collect();
            anyhow::bail!(
                "effort {:?} is not accepted by {harness} (accepted: {})",
                e.as_str(),
                names.join(", ")
            );
        }
        Ok(e)
    }
}

impl std::fmt::Display for Effort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The site config and the agents config composed, with defaults applied.
#[derive(Debug, Clone)]
pub struct Config {
    pub agents_dir: Option<String>,
    pub default_harness: Harness,
    pub agents: BTreeMap<String, AgentOverrides>,
    pub claude: ClaudeConfig,
    pub codex: CodexConfig,
    pub mqtt: MqttConfig,
    pub process: ProcessConfig,
    pub launch: LaunchConfig,
}

/// The site config file: accounts, Codex policy, the usage broker, kill grace
/// and roleless-launch defaults.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct SiteFile {
    claude: ClaudeConfig,
    codex: CodexConfig,
    mqtt: MqttConfig,
    process: ProcessConfig,
    launch: LaunchConfig,
}

/// Top-level keys of [`SiteFile`].
const SITE_KEYS: &[&str] = &["claude", "codex", "mqtt", "process", "launch"];

/// The agents config file: where agent definitions live and how to route them.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct AgentsFile {
    agents_dir: Option<String>,
    default_harness: Harness,
    agents: BTreeMap<String, AgentOverrides>,
}

/// Top-level keys of [`AgentsFile`].
const AGENTS_KEYS: &[&str] = &["agents_dir", "default_harness", "agents"];

/// Model aliases every roleless launch knows; a `[launch.aliases]` entry of the
/// same name wins.
pub const BUILTIN_MODEL_ALIASES: &[(&str, &str)] = &[
    ("opus", "claude-opus-5-5[1M]"),
    ("fable", "claude-fable-5-1[1M]"),
];

/// `[launch]`: a `ccc launch` with no agent type. Values are checked when such
/// a launch resolves, so a bad one fails only that launch.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LaunchConfig {
    /// An alias or a full Claude model string.
    pub model: String,
    /// Claude `--effort`; `None` leaves Claude's own default.
    pub effort: Option<String>,
    pub aliases: BTreeMap<String, String>,
}

impl Default for LaunchConfig {
    fn default() -> Self {
        LaunchConfig {
            model: "opus".into(),
            effort: None,
            aliases: BTreeMap::new(),
        }
    }
}

impl LaunchConfig {
    /// `name` through the user aliases, then the built-ins, else unchanged.
    /// One level deep: an alias value is never expanded again.
    pub fn expand(&self, name: &str) -> String {
        if let Some(model) = self.aliases.get(name) {
            return model.clone();
        }
        BUILTIN_MODEL_ALIASES
            .iter()
            .find(|(alias, _)| *alias == name)
            .map(|(_, model)| model.to_string())
            .unwrap_or_else(|| name.to_string())
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ProcessConfig {
    pub kill_grace_s: f64,
}

impl Default for ProcessConfig {
    fn default() -> Self {
        Self { kill_grace_s: 10.0 }
    }
}

/// Per-agent (or `"*"`) overrides from `[agents.<type>]`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AgentOverrides {
    pub harness: Option<Harness>,
    pub model: Option<String>,
    pub effort: Option<String>,
    pub codex_model: Option<String>,
    pub codex_effort: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ClaudeConfig {
    pub auth: ClaudeAuth,
    pub profiles: Vec<String>,
    pub permission_mode: PermissionMode,
    pub token_dir: String,
    pub profiles_dir: String,
    pub extra_args: Vec<String>,
    pub selection: SelectionConfig,
}

impl Default for ClaudeConfig {
    fn default() -> Self {
        ClaudeConfig {
            auth: ClaudeAuth::ConfigDir,
            profiles: Vec::new(),
            permission_mode: PermissionMode::BypassPermissions,
            token_dir: "~/.claude-tokens".to_string(),
            profiles_dir: "~/.claude-profiles".to_string(),
            extra_args: Vec::new(),
            selection: SelectionConfig::default(),
        }
    }
}

impl ClaudeConfig {
    /// Whether Claude runs under the caller's own login: no profiles are
    /// configured, so there is no account to choose and no usage to read.
    pub fn ambient(&self) -> bool {
        self.profiles.is_empty()
    }
}

/// Knobs for the account scorer.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SelectionConfig {
    pub five_hour_floor_pct: f64,
    pub five_hour_knee_pct: f64,
    pub min_time_left_h: f64,
    pub stale_grace_s: u64,
    pub fable_models: Vec<String>,
    pub general_windows: Vec<String>,
    pub fable_windows: Vec<String>,
    pub fable_general_ratio: f64,
    pub reserve_general_for_fable: bool,
    pub mqtt_quiesce_ms: u64,
    pub mqtt_timeout_ms: u64,
    /// `[claude.selection.profile.<name>]`: per-profile overrides of the
    /// 5-hour floor and knee.
    pub profile: BTreeMap<String, ProfileSelection>,
}

/// One profile's overrides of the global 5-hour floor and knee, in remaining
/// percent; an absent key falls back to the global value.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ProfileSelection {
    pub five_hour_floor_pct: Option<f64>,
    pub five_hour_knee_pct: Option<f64>,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        /// Below this much of the 5-hour window remaining, a profile is unusable.
        const FIVE_HOUR_FLOOR_PCT: f64 = 1.0;
        /// Above this much remaining, the 5-hour gate no longer damps the score.
        const FIVE_HOUR_KNEE_PCT: f64 = 15.0;
        /// Floor on hours-until-reset, so an imminent reset cannot explode a rate.
        const MIN_TIME_LEFT_H: f64 = 1.0;
        /// Snapshots older than this are not trusted.
        const STALE_GRACE_S: u64 = 3600;
        /// 1% of the Fable bucket costs this many % of the general bucket.
        const FABLE_GENERAL_RATIO: f64 = 0.5;
        const MQTT_QUIESCE_MS: u64 = 1500;
        const MQTT_TIMEOUT_MS: u64 = 8000;

        SelectionConfig {
            five_hour_floor_pct: FIVE_HOUR_FLOOR_PCT,
            five_hour_knee_pct: FIVE_HOUR_KNEE_PCT,
            min_time_left_h: MIN_TIME_LEFT_H,
            stale_grace_s: STALE_GRACE_S,
            fable_models: vec!["*fable*".to_string()],
            general_windows: vec![SEVEN_DAY.to_string(), "7-day_opus*".to_string()],
            fable_windows: vec!["7-day_fable*".to_string()],
            fable_general_ratio: FABLE_GENERAL_RATIO,
            reserve_general_for_fable: true,
            mqtt_quiesce_ms: MQTT_QUIESCE_MS,
            mqtt_timeout_ms: MQTT_TIMEOUT_MS,
            profile: BTreeMap::new(),
        }
    }
}

impl SelectionConfig {
    /// `profile`'s 5-hour `(floor, knee)` in remaining percent: its override
    /// where it has one, else the global value.
    pub fn effective(&self, profile: &str) -> (f64, f64) {
        let over = self.profile.get(profile);
        (
            over.and_then(|o| o.five_hour_floor_pct)
                .unwrap_or(self.five_hour_floor_pct),
            over.and_then(|o| o.five_hour_knee_pct)
                .unwrap_or(self.five_hour_knee_pct),
        )
    }

    /// Rejects knobs that would make a score meaningless. Every rate is
    /// divided by an hours-until-reset, and a score that comes out `NaN` or
    /// infinite silently collapses selection into configured order, so the
    /// values that could produce one are refused at load time instead.
    pub fn validate(&self) -> Result<()> {
        let finite = |name: &str, v: f64| -> Result<()> {
            if !v.is_finite() {
                anyhow::bail!("[claude.selection] {name} must be a finite number, not {v}");
            }
            Ok(())
        };
        finite("min_time_left_h", self.min_time_left_h)?;
        finite("fable_general_ratio", self.fable_general_ratio)?;

        if self.min_time_left_h <= 0.0 {
            anyhow::bail!(
                "[claude.selection] min_time_left_h must be greater than 0 (got {}); it is the floor that keeps an imminent reset from dividing by zero",
                self.min_time_left_h
            );
        }
        if self.fable_general_ratio < 0.0 {
            anyhow::bail!(
                "[claude.selection] fable_general_ratio must not be negative (got {})",
                self.fable_general_ratio
            );
        }
        check_ramp(
            "[claude.selection]",
            self.five_hour_floor_pct,
            self.five_hour_knee_pct,
        )?;
        if self.mqtt_quiesce_ms == 0 || self.mqtt_timeout_ms == 0 {
            anyhow::bail!(
                "[claude.selection] mqtt_quiesce_ms and mqtt_timeout_ms must be greater than 0 (got {} and {})",
                self.mqtt_quiesce_ms,
                self.mqtt_timeout_ms
            );
        }
        for name in self.profile.keys() {
            let (floor, knee) = self.effective(name);
            check_ramp(
                &format!(
                    "[claude.selection.profile.{name}] (unset keys fall back to [claude.selection])"
                ),
                floor,
                knee,
            )?;
        }
        Ok(())
    }
}

/// Refuses a 5-hour floor/knee pair the gate cannot use: each must be a number
/// in 0-100 (which also rules out NaN and infinities) and the floor must not
/// exceed the knee.
fn check_ramp(table: &str, floor: f64, knee: f64) -> Result<()> {
    if !(0.0..=100.0).contains(&floor) || !(0.0..=100.0).contains(&knee) || floor > knee {
        anyhow::bail!(
            "{table} needs 0 <= five_hour_floor_pct <= five_hour_knee_pct <= 100 (got floor {floor}, knee {knee})"
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct CodexConfig {
    pub approval: Approval,
    pub extra_args: Vec<String>,
}

impl Default for CodexConfig {
    fn default() -> Self {
        CodexConfig {
            approval: Approval::Never,
            extra_args: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct MqttConfig {
    pub config: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Config::compose(SiteFile::default(), AgentsFile::default())
    }
}

impl Config {
    /// The site config and the agents config, each from its env var or its
    /// default path; see [`ConfigPaths::in_env`].
    pub fn load() -> Result<Config> {
        Config::load_in(&ProcessEnv)
    }

    /// [`Config::load`] against an explicit environment. A missing default
    /// path yields that file's defaults; a missing path named by an env var
    /// is an error.
    pub fn load_in(env: &dyn EnvSource) -> Result<Config> {
        Ok(Config::load_with_sources_in(env)?.0)
    }

    /// [`Config::load_in`], also returning the two files it used and whether
    /// each was read.
    pub fn load_with_sources_in(env: &dyn EnvSource) -> Result<(Config, ConfigSources)> {
        let paths = ConfigPaths::in_env(env);
        let site = read_optional(&paths.site)?;
        let agents = read_optional(&paths.agents)?;
        let cfg = Config::from_texts(site.as_deref(), agents.as_deref(), &paths)?;
        let sources = ConfigSources {
            site: ConfigSource {
                read: site.is_some(),
                path: paths.site,
            },
            agents: ConfigSource {
                read: agents.is_some(),
                path: paths.agents,
            },
        };
        Ok((cfg, sources))
    }

    /// Parses the two files' texts (`None` for an absent file), refusing any
    /// key that belongs in the other file, and validates the result. `paths`
    /// names the files in error messages.
    pub fn from_texts(
        site: Option<&str>,
        agents: Option<&str>,
        paths: &ConfigPaths,
    ) -> Result<Config> {
        let site_file: SiteFile = match site {
            Some(text) => parse_file(text, ConfigFile::Site, paths)?,
            None => SiteFile::default(),
        };
        let agents_file: AgentsFile = match agents {
            Some(text) => parse_file(text, ConfigFile::Agents, paths)?,
            None => AgentsFile::default(),
        };
        let cfg = Config::compose(site_file, agents_file);
        cfg.validate()
            .with_context(|| format!("parsing config {}", paths.site))?;
        Ok(cfg)
    }

    fn compose(site: SiteFile, agents: AgentsFile) -> Config {
        Config {
            agents_dir: agents.agents_dir,
            default_harness: agents.default_harness,
            agents: agents.agents,
            claude: site.claude,
            codex: site.codex,
            mqtt: site.mqtt,
            process: site.process,
            launch: site.launch,
        }
    }

    fn validate(&self) -> Result<()> {
        self.claude.selection.validate()?;
        // A typo in an override name would silently leave that account on the
        // global knobs.
        if let Some(name) = self
            .claude
            .selection
            .profile
            .keys()
            .find(|name| !self.claude.profiles.contains(name))
        {
            anyhow::bail!(
                "[claude.selection.profile.{name}] names a profile that is not in [claude] profiles ({})",
                self.claude.profiles.join(", ")
            );
        }
        if !self.process.kill_grace_s.is_finite() || self.process.kill_grace_s <= 0.0 {
            anyhow::bail!(
                "[process] kill_grace_s must be a finite number greater than 0 (got {})",
                self.process.kill_grace_s
            );
        }
        Ok(())
    }

    /// Where `<type>.md` definitions live.
    pub fn agents_dir_in(&self, env: &dyn EnvSource) -> PathBuf {
        match &self.agents_dir {
            Some(d) => expand_tilde_in(env, d),
            None => config_home_in(env).join("ccc/agents"),
        }
    }

    /// `[agents.<type>]` then `[agents."*"]`, in lookup order.
    pub fn agent_overrides(&self, agent: &str) -> Vec<&AgentOverrides> {
        let mut out = Vec::new();
        if let Some(o) = self.agents.get(agent) {
            out.push(o);
        }
        if let Some(o) = self.agents.get("*") {
            out.push(o);
        }
        out
    }
}

/// One of the two config files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigFile {
    Site,
    Agents,
}

impl ConfigFile {
    /// The file's top-level keys.
    fn keys(self) -> &'static [&'static str] {
        match self {
            ConfigFile::Site => SITE_KEYS,
            ConfigFile::Agents => AGENTS_KEYS,
        }
    }

    fn other(self) -> ConfigFile {
        match self {
            ConfigFile::Site => ConfigFile::Agents,
            ConfigFile::Agents => ConfigFile::Site,
        }
    }
}

impl std::fmt::Display for ConfigFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ConfigFile::Site => "site config",
            ConfigFile::Agents => "agents config",
        })
    }
}

/// Parses `file`'s text into `T`, first refusing any top-level key of the
/// other file with a message naming the file it belongs in. Errors carry
/// `parsing config <path>` context.
fn parse_file<T: serde::de::DeserializeOwned>(
    text: &str,
    file: ConfigFile,
    paths: &ConfigPaths,
) -> Result<T> {
    let parse = || -> Result<T> {
        let table: toml::Table = toml::from_str(text)?;
        let other = file.other();
        if let Some(key) = table.keys().find(|k| other.keys().contains(&k.as_str())) {
            anyhow::bail!(
                "`{key}` belongs in the {other}, not the {file}.\n  {other}: {}",
                paths.get(other)
            );
        }
        Ok(toml::from_str(text)?)
    };
    parse().with_context(|| format!("parsing config {}", paths.get(file)))
}

/// Where a config path came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathSource {
    /// Named by this environment variable; the file must exist.
    Env(&'static str),
    /// The default location; an absent file means defaults.
    Default,
}

/// A config file path and where it came from. Displays as
/// `<path> (<VAR>)` or `<path> (default)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigPath {
    pub path: PathBuf,
    pub source: PathSource,
}

impl std::fmt::Display for ConfigPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.source {
            PathSource::Env(var) => write!(f, "{} ({var})", self.path.display()),
            PathSource::Default => write!(f, "{} (default)", self.path.display()),
        }
    }
}

/// The two config files ccc reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigPaths {
    pub site: ConfigPath,
    pub agents: ConfigPath,
}

impl ConfigPaths {
    pub fn in_env(env: &dyn EnvSource) -> ConfigPaths {
        ConfigPaths {
            site: config_path_in(env),
            agents: agents_config_path_in(env),
        }
    }

    pub fn get(&self, file: ConfigFile) -> &ConfigPath {
        match file {
            ConfigFile::Site => &self.site,
            ConfigFile::Agents => &self.agents,
        }
    }
}

/// A config file a load used, and whether it was read; an unread file is an
/// absent default path, so its half of the config is defaults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigSource {
    pub path: ConfigPath,
    pub read: bool,
}

/// The two files [`Config::load_with_sources_in`] used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigSources {
    pub site: ConfigSource,
    pub agents: ConfigSource,
}

/// `var` if set and non-empty, else `<config home>/ccc/<file>`.
fn path_in(env: &dyn EnvSource, var: &'static str, file: &str) -> ConfigPath {
    match env.get(var).filter(|p| !p.is_empty()) {
        Some(p) => ConfigPath {
            path: PathBuf::from(p),
            source: PathSource::Env(var),
        },
        None => ConfigPath {
            path: config_home_in(env).join("ccc").join(file),
            source: PathSource::Default,
        },
    }
}

/// The site config: `CCC_CONFIG`, else `<config home>/ccc/config.toml`.
pub fn config_path_in(env: &dyn EnvSource) -> ConfigPath {
    path_in(env, "CCC_CONFIG", "config.toml")
}

/// The agents config: `CCC_AGENTS_CONFIG`, else `<config home>/ccc/agents.toml`.
pub fn agents_config_path_in(env: &dyn EnvSource) -> ConfigPath {
    path_in(env, "CCC_AGENTS_CONFIG", "agents.toml")
}

/// The file's text; `None` when a default path does not exist. A missing path
/// named by an env var is an error.
fn read_optional(p: &ConfigPath) -> Result<Option<String>> {
    match std::fs::read_to_string(&p.path) {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => match p.source {
            PathSource::Default => Ok(None),
            PathSource::Env(var) => anyhow::bail!(
                "{var} names {}, which does not exist (unset {var} to use the default path)",
                p.path.display()
            ),
        },
        Err(e) => Err(e).with_context(|| format!("reading config {p}")),
    }
}

/// `$XDG_CONFIG_HOME` or `~/.config`.
pub fn config_home_in(env: &dyn EnvSource) -> PathBuf {
    match env.get("XDG_CONFIG_HOME").filter(|x| !x.is_empty()) {
        Some(x) => PathBuf::from(x),
        None => home_dir_in(env).join(".config"),
    }
}

/// `$XDG_STATE_HOME` or `~/.local/state`.
pub fn state_home_in(env: &dyn EnvSource) -> PathBuf {
    match env.get("XDG_STATE_HOME").filter(|x| !x.is_empty()) {
        Some(x) => PathBuf::from(x),
        None => home_dir_in(env).join(".local/state"),
    }
}

/// `$HOME`, or `/` when it is unset or empty.
pub fn home_dir_in(env: &dyn EnvSource) -> PathBuf {
    env.get("HOME")
        .filter(|h| !h.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

/// Expands a leading `~` to `$HOME`. Only the `~/…` form is handled.
pub fn expand_tilde_in(env: &dyn EnvSource, s: &str) -> PathBuf {
    if s == "~" {
        return home_dir_in(env);
    }
    if let Some(rest) = s.strip_prefix("~/") {
        return home_dir_in(env).join(rest);
    }
    PathBuf::from(s)
}

/// Treats a blank string as an absent value, as the env sentinels do.
pub fn blank_to_none(v: Option<String>) -> Option<String> {
    v.filter(|s| !s.trim().is_empty())
}

/// The env-key fragment for an agent type: upper-cased, non-alphanumerics to `_`.
pub fn env_agent_key(agent: &str) -> String {
    agent
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

/// Sentinel values that mean "no override".
fn is_sentinel(v: &str) -> bool {
    let t = v.trim();
    t.is_empty()
        || t.eq_ignore_ascii_case("inherit")
        || t.eq_ignore_ascii_case("default")
        || t == "-"
}

/// Reads environment variables by key.
pub trait EnvSource: Sync {
    fn get(&self, key: &str) -> Option<String>;
    fn vars_with_prefix(&self, prefix: &str) -> Vec<(String, String)>;
}

/// The real process environment.
pub struct ProcessEnv;

impl EnvSource for ProcessEnv {
    fn get(&self, key: &str) -> Option<String> {
        std::env::var(key).ok()
    }

    fn vars_with_prefix(&self, prefix: &str) -> Vec<(String, String)> {
        std::env::vars_os()
            .filter_map(|(key, value)| {
                let key = key.into_string().ok()?;
                let value = value.into_string().ok()?;
                key.starts_with(prefix).then_some((key, value))
            })
            .collect()
    }
}

impl EnvSource for BTreeMap<String, String> {
    fn get(&self, key: &str) -> Option<String> {
        BTreeMap::get(self, key).cloned()
    }

    fn vars_with_prefix(&self, prefix: &str) -> Vec<(String, String)> {
        self.range(prefix.to_string()..)
            .take_while(|(key, _)| key.starts_with(prefix))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }
}

/// `CCC_<SUFFIX>_<AGENT>` then `CCC_<SUFFIX>_ALL`, skipping sentinel values.
pub fn env_override(env: &dyn EnvSource, suffix: &str, agent: &str) -> Option<String> {
    env_value(env, &format!("CCC_{suffix}_{}", env_agent_key(agent)))
        .or_else(|| env_value(env, &format!("CCC_{suffix}_ALL")))
}

/// Reads one environment override, applying the common blank/sentinel rules.
pub fn env_value(env: &dyn EnvSource, key: &str) -> Option<String> {
    env.get(key)
        .filter(|v| !is_sentinel(v))
        .map(|v| v.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn paths() -> ConfigPaths {
        ConfigPaths::in_env(&env(&[("HOME", "/home/u")]))
    }

    fn site_cfg(text: &str) -> Result<Config> {
        Config::from_texts(Some(text), None, &paths())
    }

    fn agents_cfg(text: &str) -> Result<Config> {
        Config::from_texts(None, Some(text), &paths())
    }

    #[test]
    fn enumerates_prefixes_in_map_order() {
        let e = env(&[("CCC_Z", "z"), ("CCC_A", "a"), ("NO", "n")]);
        assert_eq!(
            e.vars_with_prefix("CCC_"),
            vec![("CCC_A".into(), "a".into()), ("CCC_Z".into(), "z".into())]
        );
        assert_eq!(e.vars_with_prefix(""), e.into_iter().collect::<Vec<_>>());
    }

    #[test]
    fn launch_defaults_apply_without_a_section() {
        let l = site_cfg("").unwrap().launch;
        assert_eq!(l.model, "opus");
        assert_eq!(l.effort, None);
        assert!(l.aliases.is_empty());
        assert_eq!(l.expand(&l.model), "claude-opus-5-5[1M]");
    }

    #[test]
    fn launch_aliases_layer_over_the_built_ins_one_level_deep() {
        let l = site_cfg(
            "[launch.aliases]\nfable = \"my-fable\"\nhaiku = \"claude-haiku-4-5\"\nchain = \"opus\"\n",
        )
        .unwrap()
        .launch;
        assert_eq!(l.expand("opus"), "claude-opus-5-5[1M]");
        assert_eq!(l.expand("fable"), "my-fable");
        assert_eq!(l.expand("haiku"), "claude-haiku-4-5");
        assert_eq!(l.expand("sonnet"), "sonnet");
        assert_eq!(l.expand("claude-opus-4-6[1M]"), "claude-opus-4-6[1M]");
        assert_eq!(l.expand("chain"), "opus");
        assert_eq!(
            LaunchConfig::default().expand("fable"),
            "claude-fable-5-1[1M]"
        );
    }

    #[test]
    fn launch_values_are_checked_at_resolve_time_not_parse_time() {
        for text in [
            "[launch]\neffort = \"minimal\"\n",
            "[launch]\neffort = \"lo\"\n",
            "[launch]\nmodel = \"\"\n",
            "[launch.aliases]\nopus = \"\"\n",
        ] {
            site_cfg(text).unwrap_or_else(|e| panic!("{text}: {e:#}"));
        }
        let err = format!(
            "{:#}",
            site_cfg("[launch]\nmodle = \"opus\"\n").unwrap_err()
        );
        assert!(err.contains("modle"), "{err}");
    }

    #[test]
    fn empty_config_is_all_defaults() {
        let c = site_cfg("").unwrap();
        assert_eq!(c.default_harness, Harness::Claude);
        assert_eq!(c.claude.auth, ClaudeAuth::ConfigDir);
        assert_eq!(c.claude.permission_mode, PermissionMode::BypassPermissions);
        assert!(c.claude.profiles.is_empty());
        assert!(c.claude.ambient());
        assert_eq!(c.claude.profiles_dir, "~/.claude-profiles");
        assert_eq!(c.claude.token_dir, "~/.claude-tokens");
        assert_eq!(c.codex.approval, Approval::Never);
        assert_eq!(c.claude.selection.fable_general_ratio, 0.5);
        assert!(c.claude.selection.reserve_general_for_fable);
        assert_eq!(
            c.claude.selection.general_windows,
            vec!["seven_day", "7-day_opus*"]
        );
    }

    #[test]
    fn parses_the_documented_config() {
        let site = r#"
[claude]
auth = "config-dir"
profiles = ["team03", "personal"]
extra_args = ["--add-dir", "/srv"]

[claude.selection]
five_hour_knee_pct = 20.0
fable_windows = ["7-day_fable*", "weekly_fable"]

[claude.selection.profile.personal]
five_hour_knee_pct = 50.0
five_hour_floor_pct = 20.0

[codex]
approval = "on-failure"

[mqtt]
config = "~/.config/claude-usage-mqtt/config.toml"
"#;
        let agents = r#"
agents_dir = "~/.config/ccc/agents"
default_harness = "codex"

[agents.explorer]
harness = "codex"
codex_model = "gpt-5.6-terra"

[agents."*"]
effort = "low"
"#;
        let c = Config::from_texts(Some(site), Some(agents), &paths()).unwrap();
        assert_eq!(c.default_harness, Harness::Codex);
        assert_eq!(c.agents_dir.as_deref(), Some("~/.config/ccc/agents"));
        assert_eq!(c.claude.auth, ClaudeAuth::ConfigDir);
        assert_eq!(c.claude.profiles, vec!["team03", "personal"]);
        assert_eq!(c.claude.extra_args, vec!["--add-dir", "/srv"]);
        assert_eq!(c.claude.selection.five_hour_knee_pct, 20.0);
        assert_eq!(c.claude.selection.five_hour_floor_pct, 1.0);
        assert_eq!(c.claude.selection.effective("personal"), (20.0, 50.0));
        assert_eq!(c.claude.selection.fable_windows.len(), 2);
        assert_eq!(c.codex.approval, Approval::OnFailure);
        assert_eq!(
            c.mqtt.config.as_deref(),
            Some("~/.config/claude-usage-mqtt/config.toml")
        );
        let ov = c.agent_overrides("explorer");
        assert_eq!(ov.len(), 2);
        assert_eq!(ov[0].harness, Some(Harness::Codex));
        assert_eq!(ov[1].effort.as_deref(), Some("low"));

        // Each half alone leaves the other half at its defaults.
        let only_site = site_cfg(site).unwrap();
        assert_eq!(only_site.default_harness, Harness::Claude);
        assert!(only_site.agents.is_empty());
        let only_agents = agents_cfg(agents).unwrap();
        assert_eq!(only_agents.default_harness, Harness::Codex);
        assert_eq!(only_agents.claude.auth, ClaudeAuth::ConfigDir);
    }

    #[test]
    fn an_absent_file_and_an_empty_file_agree_on_every_default() {
        let absent = Config::default();
        for (site, agents) in [(None, None), (Some(""), Some(""))] {
            let parsed = Config::from_texts(site, agents, &paths()).unwrap();
            assert_eq!(
                format!("{parsed:?}"),
                format!("{absent:?}"),
                "serde defaults and Default::default disagree"
            );
        }
    }

    #[test]
    fn closed_value_sets_reject_typos_at_parse_time() {
        assert!(site_cfg("[claude]\npermission_mode = \"bypassPermission\"").is_err());
        assert!(site_cfg("[codex]\napproval = \"nver\"").is_err());
        assert_eq!(
            site_cfg("[claude]\npermission_mode = \"dontAsk\"")
                .unwrap()
                .claude
                .permission_mode,
            PermissionMode::DontAsk
        );
        assert!(Effort::parse("lo").is_err());
        assert_eq!(Effort::parse(" XHigh ").unwrap(), Effort::XHigh);
        assert_eq!(Effort::Max.as_str(), "max");
    }

    #[test]
    fn unknown_config_keys_are_rejected() {
        for result in [
            site_cfg("defualt_harness = \"claude\""),
            agents_cfg("defualt_harness = \"claude\""),
            site_cfg("[claude]\nprofile = \"team01\""),
            agents_cfg("[agents.x]\ncodex_efort = \"low\""),
        ] {
            let err = format!("{:#}", result.unwrap_err());
            assert!(err.contains("unknown field"), "{err}");
        }
    }

    #[test]
    fn missing_config_file_is_defaults() {
        let home = crate::testutil::TempDir::new("config-home");
        let e = env(&[("HOME", home.path().to_str().unwrap())]);
        let c = Config::load_in(&e).unwrap();
        assert_eq!(
            format!("{c:?}"),
            format!("{:?}", Config::default()),
            "absent default paths are not all defaults"
        );
    }

    #[test]
    fn both_default_paths_are_read() {
        let home = crate::testutil::TempDir::new("config-home");
        std::fs::create_dir_all(home.path().join(".config/ccc")).unwrap();
        home.write_file(
            ".config/ccc/config.toml",
            "[codex]\napproval = \"on-request\"\n",
            0o600,
        );
        home.write_file(
            ".config/ccc/agents.toml",
            "default_harness = \"codex\"\n",
            0o600,
        );
        let e = env(&[("HOME", home.path().to_str().unwrap())]);
        let c = Config::load_in(&e).unwrap();
        assert_eq!(c.codex.approval, Approval::OnRequest);
        assert_eq!(c.default_harness, Harness::Codex);
    }

    #[test]
    fn sources_report_which_default_paths_were_read() {
        for (present, absent) in [
            ("config.toml", "agents.toml"),
            ("agents.toml", "config.toml"),
        ] {
            let home = crate::testutil::TempDir::new("config-home");
            std::fs::create_dir_all(home.path().join(".config/ccc")).unwrap();
            home.write_file(&format!(".config/ccc/{present}"), "", 0o600);
            let e = env(&[("HOME", home.path().to_str().unwrap())]);
            let (_, sources) = Config::load_with_sources_in(&e).unwrap();
            let (read, unread) = if present == "config.toml" {
                (&sources.site, &sources.agents)
            } else {
                (&sources.agents, &sources.site)
            };
            assert!(read.read, "{present}");
            assert!(!unread.read, "{absent}");
            assert_eq!(unread.path.source, PathSource::Default);
            assert_eq!(
                unread.path.path,
                home.path().join(".config/ccc").join(absent)
            );
        }
    }

    #[test]
    fn a_missing_path_named_by_an_env_var_is_an_error() {
        let home = crate::testutil::TempDir::new("config-home");
        let missing = home.path().join("gone.toml");
        for var in ["CCC_CONFIG", "CCC_AGENTS_CONFIG"] {
            let e = env(&[
                ("HOME", home.path().to_str().unwrap()),
                (var, missing.to_str().unwrap()),
            ]);
            let err = format!("{:#}", Config::load_in(&e).unwrap_err());
            assert!(err.contains(var), "{err}");
            assert!(err.contains(&missing.display().to_string()), "{err}");
        }
        // An empty variable is unset.
        let e = env(&[
            ("HOME", home.path().to_str().unwrap()),
            ("CCC_CONFIG", ""),
            ("CCC_AGENTS_CONFIG", ""),
        ]);
        Config::load_in(&e).unwrap();
    }

    /// A minimal valid text whose only top-level key is `key`.
    fn text_for_key(key: &str) -> String {
        match key {
            "agents_dir" => "agents_dir = \"/srv/agents\"\n".to_string(),
            "default_harness" => "default_harness = \"codex\"\n".to_string(),
            table => format!("[{table}]\n"),
        }
    }

    #[test]
    fn every_key_parses_in_its_own_file_and_is_refused_in_the_other() {
        let paths = ConfigPaths::in_env(&env(&[
            ("HOME", "/home/u"),
            ("CCC_AGENTS_CONFIG", "/srv/agents.toml"),
        ]));
        for key in SITE_KEYS {
            let text = text_for_key(key);
            Config::from_texts(Some(&text), None, &paths)
                .unwrap_or_else(|e| panic!("{key} as site config: {e:#}"));
            let err = format!(
                "{:#}",
                Config::from_texts(None, Some(&text), &paths).unwrap_err()
            );
            assert!(
                err.contains("parsing config /srv/agents.toml (CCC_AGENTS_CONFIG)"),
                "{err}"
            );
            assert!(
                err.contains(&format!(
                    "`{key}` belongs in the site config, not the agents config."
                )),
                "{err}"
            );
            assert!(
                err.contains("site config: /home/u/.config/ccc/config.toml (default)"),
                "{err}"
            );
        }
        let paths = ConfigPaths::in_env(&env(&[
            ("HOME", "/home/u"),
            ("CCC_CONFIG", "/srv/site.toml"),
        ]));
        for key in AGENTS_KEYS {
            let text = text_for_key(key);
            Config::from_texts(None, Some(&text), &paths)
                .unwrap_or_else(|e| panic!("{key} as agents config: {e:#}"));
            let err = format!(
                "{:#}",
                Config::from_texts(Some(&text), None, &paths).unwrap_err()
            );
            assert!(
                err.contains("parsing config /srv/site.toml (CCC_CONFIG)"),
                "{err}"
            );
            assert!(
                err.contains(&format!(
                    "`{key}` belongs in the agents config, not the site config."
                )),
                "{err}"
            );
            assert!(
                err.contains("agents config: /home/u/.config/ccc/agents.toml (default)"),
                "{err}"
            );
        }
    }

    #[test]
    fn an_old_layout_combined_file_names_one_misplaced_key() {
        // Top-level keys are checked in sorted order, so `agents` comes first.
        let text = "agents_dir = \"/a\"\n[agents.x]\nharness = \"codex\"\n[claude]\n";
        let err = format!("{:#}", site_cfg(text).unwrap_err());
        assert!(
            err.contains("`agents` belongs in the agents config"),
            "{err}"
        );
        let err = format!(
            "{:#}",
            site_cfg("agents_dir = \"/a\"\n[claude]\n").unwrap_err()
        );
        assert!(
            err.contains("`agents_dir` belongs in the agents config"),
            "{err}"
        );
        let err = format!("{:#}", agents_cfg("[claude]\n[agents.x]\n").unwrap_err());
        assert!(err.contains("`claude` belongs in the site config"), "{err}");
    }

    #[test]
    fn agent_env_key_derivation() {
        assert_eq!(env_agent_key("comment-rewriter"), "COMMENT_REWRITER");
        assert_eq!(env_agent_key("explorer"), "EXPLORER");
        assert_eq!(env_agent_key("a.b c"), "A_B_C");
    }

    #[test]
    fn per_agent_env_beats_all() {
        let e = env(&[("CCC_MODEL_EXPLORER", "haiku"), ("CCC_MODEL_ALL", "opus")]);
        assert_eq!(
            env_override(&e, "MODEL", "explorer").as_deref(),
            Some("haiku")
        );
        assert_eq!(env_override(&e, "MODEL", "scribe").as_deref(), Some("opus"));
    }

    #[test]
    fn sentinel_env_values_are_no_ops() {
        for v in ["", "  ", "inherit", "INHERIT", "default", "-"] {
            let e = env(&[("CCC_HARNESS_EXPLORER", v)]);
            assert_eq!(env_override(&e, "HARNESS", "explorer"), None, "value {v:?}");
        }
        // A sentinel on the per-agent key falls through to the _ALL key.
        let e = env(&[("CCC_HARNESS_EXPLORER", "-"), ("CCC_HARNESS_ALL", "codex")]);
        assert_eq!(
            env_override(&e, "HARNESS", "explorer").as_deref(),
            Some("codex")
        );
    }

    #[test]
    fn harness_parsing() {
        assert_eq!(Harness::parse("Claude").unwrap(), Harness::Claude);
        assert_eq!(Harness::parse(" codex ").unwrap(), Harness::Codex);
        assert!(Harness::parse("gemini").is_err());
    }

    #[test]
    fn tilde_expansion() {
        let e = env(&[("HOME", "/home/tester")]);
        assert_eq!(
            expand_tilde_in(&e, "~/.claude-tokens"),
            PathBuf::from("/home/tester/.claude-tokens")
        );
        assert_eq!(expand_tilde_in(&e, "~"), PathBuf::from("/home/tester"));
        assert_eq!(expand_tilde_in(&e, "/abs/path"), PathBuf::from("/abs/path"));
        assert_eq!(expand_tilde_in(&e, "rel/path"), PathBuf::from("rel/path"));
        assert_eq!(expand_tilde_in(&env(&[]), "~/x"), PathBuf::from("/x"));
    }

    #[test]
    fn xdg_paths_fall_back_when_unset_or_empty() {
        let home = env(&[("HOME", "/home/tester")]);
        assert_eq!(config_home_in(&home), PathBuf::from("/home/tester/.config"));
        assert_eq!(
            state_home_in(&home),
            PathBuf::from("/home/tester/.local/state")
        );

        let xdg = env(&[
            ("HOME", "/home/tester"),
            ("XDG_CONFIG_HOME", "/xdg/config"),
            ("XDG_STATE_HOME", "/xdg/state"),
        ]);
        assert_eq!(config_home_in(&xdg), PathBuf::from("/xdg/config"));
        assert_eq!(state_home_in(&xdg), PathBuf::from("/xdg/state"));

        let empty = env(&[
            ("HOME", "/home/tester"),
            ("XDG_CONFIG_HOME", ""),
            ("XDG_STATE_HOME", ""),
        ]);
        assert_eq!(
            config_home_in(&empty),
            PathBuf::from("/home/tester/.config")
        );
        assert_eq!(
            state_home_in(&empty),
            PathBuf::from("/home/tester/.local/state")
        );
    }

    #[test]
    fn config_and_agents_paths() {
        let base = env(&[("HOME", "/home/tester")]);
        assert_eq!(
            config_path_in(&base),
            ConfigPath {
                path: PathBuf::from("/home/tester/.config/ccc/config.toml"),
                source: PathSource::Default,
            }
        );
        assert_eq!(
            agents_config_path_in(&base).to_string(),
            "/home/tester/.config/ccc/agents.toml (default)"
        );
        let explicit = env(&[
            ("HOME", "/home/tester"),
            ("CCC_CONFIG", "/etc/ccc.toml"),
            ("CCC_AGENTS_CONFIG", "/etc/agents.toml"),
        ]);
        assert_eq!(
            config_path_in(&explicit).to_string(),
            "/etc/ccc.toml (CCC_CONFIG)"
        );
        assert_eq!(
            agents_config_path_in(&explicit),
            ConfigPath {
                path: PathBuf::from("/etc/agents.toml"),
                source: PathSource::Env("CCC_AGENTS_CONFIG"),
            }
        );

        let cfg = agents_cfg("").unwrap();
        assert_eq!(
            cfg.agents_dir_in(&base),
            PathBuf::from("/home/tester/.config/ccc/agents")
        );
        let cfg = agents_cfg("agents_dir = \"~/agents\"").unwrap();
        assert_eq!(
            cfg.agents_dir_in(&base),
            PathBuf::from("/home/tester/agents")
        );
    }

    #[test]
    fn a_config_file_that_does_not_parse_names_the_path_and_the_value() {
        let dir = crate::testutil::TempDir::new("config-bad");
        for (var, name, body) in [
            (
                "CCC_CONFIG",
                "site.toml",
                "[codex]\napproval = \"gemini\"\n",
            ),
            (
                "CCC_AGENTS_CONFIG",
                "agents.toml",
                "default_harness = \"gemini\"\n",
            ),
        ] {
            let path = dir.write_file(name, body, 0o600);
            let e = env(&[
                ("HOME", dir.path().to_str().unwrap()),
                (var, path.to_str().unwrap()),
            ]);
            let text = format!("{:#}", Config::load_in(&e).unwrap_err());
            assert!(text.contains("parsing config"), "{text}");
            assert!(text.contains(&path.display().to_string()), "{text}");
            assert!(text.contains("gemini"), "{text}");
        }
    }

    #[test]
    fn effort_is_checked_against_the_harness_that_will_receive_it() {
        assert_eq!(
            Effort::parse_for("max", Harness::Claude).unwrap(),
            Effort::Max
        );
        assert_eq!(
            Effort::parse_for("minimal", Harness::Codex).unwrap(),
            Effort::Minimal
        );
        let err = Effort::parse_for("minimal", Harness::Claude)
            .unwrap_err()
            .root_cause()
            .to_string();
        assert!(err.contains("claude"), "{err}");
        assert!(err.contains("minimal"), "{err}");
        let err = Effort::parse_for("max", Harness::Codex)
            .unwrap_err()
            .root_cause()
            .to_string();
        assert!(err.contains("codex"), "{err}");
    }

    #[test]
    fn blank_values_are_absent_values() {
        assert_eq!(blank_to_none(Some("  ".into())), None);
        assert_eq!(blank_to_none(Some("".into())), None);
        assert_eq!(blank_to_none(Some("m".into())).as_deref(), Some("m"));
        assert_eq!(blank_to_none(None), None);
    }

    #[test]
    fn the_defaults_are_valid_selection_knobs() {
        SelectionConfig::default().validate().unwrap();
        site_cfg("").unwrap();
    }

    #[test]
    fn selection_knobs_that_would_produce_nan_or_inf_are_refused() {
        let cases = [
            ("min_time_left_h = 0.0", "min_time_left_h"),
            ("min_time_left_h = -1.0", "min_time_left_h"),
            ("min_time_left_h = nan", "min_time_left_h"),
            ("fable_general_ratio = -0.5", "fable_general_ratio"),
            ("five_hour_floor_pct = 20.0", "five_hour_floor_pct"),
            ("five_hour_knee_pct = 140.0", "five_hour_knee_pct"),
            ("five_hour_floor_pct = -1.0", "five_hour_floor_pct"),
            ("mqtt_timeout_ms = 0", "mqtt_timeout_ms"),
            ("mqtt_quiesce_ms = 0", "mqtt_quiesce_ms"),
        ];
        for (line, key) in cases {
            let text = format!("[claude.selection]\n{line}\n");
            let err = site_cfg(&text).unwrap_err().root_cause().to_string();
            assert!(err.contains(key), "{line} gave {err}");
            assert!(err.contains("[claude.selection]"), "{line} gave {err}");
        }

        // The ordering constraint holds even when both values are in range.
        let err =
            site_cfg("[claude.selection]\nfive_hour_floor_pct = 15.0\nfive_hour_knee_pct = 5.0\n")
                .unwrap_err()
                .root_cause()
                .to_string();
        assert!(
            err.contains("five_hour_floor_pct <= five_hour_knee_pct"),
            "{err}"
        );

        // A knee equal to the floor is a step, not an error.
        site_cfg("[claude.selection]\nfive_hour_floor_pct = 5.0\nfive_hour_knee_pct = 5.0\n")
            .unwrap();
    }

    #[test]
    fn per_profile_floor_and_knee_parse_and_resolve_against_the_globals() {
        assert!(site_cfg("").unwrap().claude.selection.profile.is_empty());

        let c = site_cfg(
            "[claude]\nprofiles = [\"personal\", \"team01\"]\n\n\
             [claude.selection]\nfive_hour_knee_pct = 20.0\n\n\
             [claude.selection.profile.personal]\nfive_hour_knee_pct = 60.0\nfive_hour_floor_pct = 20.0\n\n\
             [claude.selection.profile.team01]\nfive_hour_floor_pct = 5.0\n",
        )
        .unwrap();
        let sel = &c.claude.selection;
        assert_eq!(
            sel.profile["personal"],
            ProfileSelection {
                five_hour_floor_pct: Some(20.0),
                five_hour_knee_pct: Some(60.0),
            }
        );
        assert_eq!(sel.effective("personal"), (20.0, 60.0));
        // Only the floor is overridden; the knee is the global one.
        assert_eq!(sel.effective("team01"), (5.0, 20.0));
        // No override at all: both globals.
        assert_eq!(sel.effective("team02"), (1.0, 20.0));

        let err = site_cfg("[claude.selection.profile.personal]\nknee = 60.0\n")
            .unwrap_err()
            .root_cause()
            .to_string();
        assert!(err.contains("unknown field"), "{err}");
    }

    #[test]
    fn per_profile_overrides_that_cannot_work_are_refused_by_name() {
        let err = site_cfg(
            "[claude]\nprofiles = [\"team01\", \"personal\"]\n\n\
             [claude.selection.profile.persnal]\nfive_hour_knee_pct = 60.0\n",
        )
        .unwrap_err()
        .root_cause()
        .to_string();
        assert!(err.contains("[claude.selection.profile.persnal]"), "{err}");
        assert!(err.contains("not in [claude] profiles"), "{err}");

        for (lines, key) in [
            ("five_hour_knee_pct = 140.0", "five_hour_knee_pct"),
            ("five_hour_floor_pct = -1.0", "five_hour_floor_pct"),
            ("five_hour_knee_pct = nan", "five_hour_knee_pct"),
            ("five_hour_floor_pct = inf", "five_hour_floor_pct"),
            // Overriding only the floor cannot cross the global knee (15).
            (
                "five_hour_floor_pct = 20.0",
                "five_hour_floor_pct <= five_hour_knee_pct",
            ),
            // Nor only the knee below the global floor (1).
            (
                "five_hour_knee_pct = 0.5",
                "five_hour_floor_pct <= five_hour_knee_pct",
            ),
            (
                "five_hour_floor_pct = 30.0\nfive_hour_knee_pct = 20.0",
                "five_hour_floor_pct <= five_hour_knee_pct",
            ),
        ] {
            let text = format!("[claude.selection.profile.personal]\n{lines}\n");
            let err = site_cfg(&text).unwrap_err().root_cause().to_string();
            assert!(err.contains(key), "{lines} gave {err}");
            assert!(
                err.contains("[claude.selection.profile.personal]"),
                "{lines} gave {err}"
            );
        }

        // The message shows the resolved pair, so the bad value is visible.
        let err = site_cfg("[claude.selection.profile.personal]\nfive_hour_knee_pct = 160.0\n")
            .unwrap_err()
            .root_cause()
            .to_string();
        assert_eq!(
            err,
            "[claude.selection.profile.personal] (unset keys fall back to [claude.selection]) \
             needs 0 <= five_hour_floor_pct <= five_hour_knee_pct <= 100 (got floor 1, knee 160)"
        );

        // A floor raised together with the knee is fine.
        site_cfg(
            "[claude]\nprofiles = [\"personal\"]\n\n\
             [claude.selection.profile.personal]\nfive_hour_floor_pct = 20.0\nfive_hour_knee_pct = 60.0\n",
        )
        .unwrap();
    }

    #[test]
    fn a_per_profile_table_is_refused_when_profiles_is_empty() {
        let err = site_cfg("[claude.selection.profile.work]\nfive_hour_knee_pct = 60.0\n")
            .unwrap_err()
            .root_cause()
            .to_string();
        assert_eq!(
            err,
            "[claude.selection.profile.work] names a profile that is not in [claude] profiles ()"
        );
    }

    #[test]
    fn process_kill_grace_defaults_validates_and_rejects_bad_tables() {
        assert_eq!(site_cfg("").unwrap().process.kill_grace_s, 10.0);
        assert_eq!(
            site_cfg("[process]\nkill_grace_s = 0.25\n")
                .unwrap()
                .process
                .kill_grace_s,
            0.25
        );
        for value in ["0.0", "-1.0", "nan", "inf", "-inf"] {
            let error = site_cfg(&format!("[process]\nkill_grace_s = {value}\n"))
                .unwrap_err()
                .root_cause()
                .to_string();
            assert!(error.contains("[process] kill_grace_s"), "{value}: {error}");
        }
        let error = site_cfg("[process]\nother = 1\n")
            .unwrap_err()
            .root_cause()
            .to_string();
        assert!(error.contains("unknown field"), "{error}");
    }
}
