//! Session config: the site config (`CCC_CONFIG` or
//! `~/.config/ccc/config.toml`) and the agents config (`CCC_AGENTS_CONFIG` or
//! `~/.config/ccc/agents.toml`), which hold disjoint keys, plus the
//! `CCC_*_<AGENT>` / `CCC_*_ALL` environment overrides.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use claude_usage_mqtt_core::SEVEN_DAY;
use serde::Deserialize;

use crate::tuning::{self, AccountLayer, ModelRule, Priority, Tuning, TuningLayer, WeeklyReset};

/// Which harness runs an agent.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Deserialize, serde::Serialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Harness {
    #[default]
    Claude,
    Codex,
    Goose,
}

impl Harness {
    pub const ALL: [Harness; 3] = [Harness::Claude, Harness::Codex, Harness::Goose];
    pub fn as_str(self) -> &'static str {
        match self {
            Harness::Claude => "claude",
            Harness::Codex => "codex",
            Harness::Goose => "goose",
        }
    }

    pub fn parse(s: &str) -> Result<Harness> {
        match s.trim().to_ascii_lowercase().as_str() {
            "claude" => Ok(Harness::Claude),
            "codex" => Ok(Harness::Codex),
            "goose" => Ok(Harness::Goose),
            other => anyhow::bail!(
                "unknown harness {other:?} (expected \"claude\", \"codex\" or \"goose\")"
            ),
        }
    }
}

impl std::fmt::Display for Harness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How the Claude harness is pointed at an account. `ApiKey` names what a
/// turn ran under; it is not a way to authenticate `[claude] profiles`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ClaudeAuth {
    Token,
    ConfigDir,
    ApiKey,
}

impl ClaudeAuth {
    pub fn as_str(self) -> &'static str {
        match self {
            ClaudeAuth::Token => "token",
            ClaudeAuth::ConfigDir => "config-dir",
            ClaudeAuth::ApiKey => "api-key",
        }
    }
}

impl std::fmt::Display for ClaudeAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How much a Claude `-p` run may do without asking. The default is `auto`:
/// Claude's classifier approves or denies each action, and under `-p` a
/// denial reaches the agent as a failed tool call.
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
    /// `max`, Codex's `model_reasoning_effort` takes `minimal` through `xhigh`,
    /// and Goose has no reasoning-effort setting.
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
            Harness::Goose => &[],
        }
    }

    /// Parses `s` and rejects a value the resolved harness does not accept,
    /// so a typo costs no spawn.
    pub fn parse_for(s: &str, harness: Harness) -> Result<Effort> {
        let e = Effort::parse(s)?;
        let accepted = Effort::accepted_by(harness);
        if accepted.is_empty() {
            anyhow::bail!("{harness} accepts no effort (got {:?})", e.as_str());
        }
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
    pub goose: GooseConfig,
    pub mqtt: MqttConfig,
    pub bus: BusConfig,
    pub process: ProcessConfig,
    pub launch: LaunchConfig,
    /// The account-tuning knobs selection reads, resolved from the site config.
    pub tuning: Tuning,
}

/// The site config file: accounts, Codex policy, the usage broker, kill grace
/// and roleless-launch defaults.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct SiteFile {
    claude: ClaudeConfig,
    codex: CodexConfig,
    goose: GooseConfig,
    mqtt: MqttConfig,
    bus: BusConfig,
    process: ProcessConfig,
    launch: LaunchConfig,
}

/// Top-level keys of [`SiteFile`].
const SITE_KEYS: &[&str] = &[
    "claude", "codex", "goose", "mqtt", "bus", "process", "launch",
];

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
/// same name wins. Each maps to Claude's own alias plus the 1M-context suffix,
/// so Claude resolves the current model version and nothing here goes stale.
pub const BUILTIN_MODEL_ALIASES: &[(&str, &str)] = &[("opus", "opus[1m]"), ("fable", "fable[1m]")];

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
    pub goose_model: Option<String>,
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
    /// `[claude.api_keys.<name>]`: Claude Platform API key accounts.
    pub api_keys: BTreeMap<String, ApiKeyConfig>,
}

impl Default for ClaudeConfig {
    fn default() -> Self {
        ClaudeConfig {
            auth: ClaudeAuth::ConfigDir,
            profiles: Vec::new(),
            permission_mode: PermissionMode::Auto,
            token_dir: "~/.claude-tokens".to_string(),
            profiles_dir: "~/.claude-profiles".to_string(),
            extra_args: Vec::new(),
            selection: SelectionConfig::default(),
            api_keys: BTreeMap::new(),
        }
    }
}

/// The message globs that mark a refused key turn as the credit running out.
// TODO(api-key-assumptions-unconfirmed): A2 — this pattern is a guess at the
// Platform's credit-gone message, and no default matches the spend-limit
// refusal; neither has been seen from a real key.
pub const DEFAULT_EXHAUSTED_PATTERNS: &[&str] = &["*credit balance is too low*"];

/// One `[claude.api_keys.<name>]`: a Claude Platform API key with a monthly
/// dollar budget. `key_file`, `budget_usd` and `models` have no default.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApiKeyConfig {
    /// Mode 0600, one non-empty line; `~` is expanded.
    pub key_file: String,
    pub budget_usd: f64,
    /// Globs over the resolved model; a key never takes a model none matches.
    pub models: Vec<String>,
    /// How far short of `budget_usd` selection stops spending the key.
    #[serde(default)]
    pub reserve_usd: f64,
    /// IANA zone whose calendar month is the budget period.
    #[serde(default = "default_reset_tz")]
    pub reset_tz: String,
    /// Globs over a failed turn's message that mean the key is out of credit.
    #[serde(default = "default_exhausted_patterns")]
    pub exhausted_patterns: Vec<String>,
    /// Orders this key among the keys eligible at overflow. Private, like
    /// every knob: it is read into [`Config::tuning`] at load, and selection
    /// reads only that.
    #[serde(default)]
    weight: Option<f64>,
    /// Per-model weights, first match wins; read into [`Config::tuning`].
    #[serde(default)]
    by_model: Vec<KeyModelPriorityToml>,
}

/// One `[[claude.api_keys.<name>.by_model]]` entry.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct KeyModelPriorityToml {
    models: Vec<String>,
    weight: f64,
}

fn default_reset_tz() -> String {
    "UTC".to_string()
}

fn default_exhausted_patterns() -> Vec<String> {
    DEFAULT_EXHAUSTED_PATTERNS
        .iter()
        .map(|p| p.to_string())
        .collect()
}

impl ApiKeyConfig {
    /// A key with the given required fields, the defaults for the rest, and
    /// no knobs.
    #[cfg(test)]
    pub fn new(key_file: &str, budget_usd: f64, models: &[&str]) -> ApiKeyConfig {
        ApiKeyConfig {
            key_file: key_file.to_string(),
            budget_usd,
            models: models.iter().map(|m| m.to_string()).collect(),
            reserve_usd: 0.0,
            reset_tz: default_reset_tz(),
            exhausted_patterns: default_exhausted_patterns(),
            weight: None,
            by_model: Vec::new(),
        }
    }

    /// Refuses a key table no budget can be kept against; `name` is the
    /// table's key.
    fn validate(&self, name: &str, profiles: &[String]) -> Result<()> {
        let table = format!("[claude.api_keys.{name}]");
        if !claude_usage_mqtt_core::is_topic_safe(name) {
            anyhow::bail!("{table}: the name may use only A-Z, a-z, 0-9, '.', '_' and '-'");
        }
        if profiles.iter().any(|p| p == name) {
            anyhow::bail!(
                "{table}: {name} is also in [claude] profiles; a key and a profile cannot share a name"
            );
        }
        if !self.budget_usd.is_finite() || self.budget_usd <= 0.0 {
            anyhow::bail!(
                "{table} budget_usd must be a finite number greater than 0 (got {})",
                self.budget_usd
            );
        }
        if !self.reserve_usd.is_finite()
            || self.reserve_usd < 0.0
            || self.reserve_usd >= self.budget_usd
        {
            anyhow::bail!(
                "{table} reserve_usd must be a finite number, at least 0 and below budget_usd (got {})",
                self.reserve_usd
            );
        }
        if self.models.is_empty() {
            anyhow::bail!("{table} models must name at least one model glob");
        }
        if self.exhausted_patterns.is_empty() {
            anyhow::bail!("{table} exhausted_patterns must name at least one message glob");
        }
        if let Err(e) = jiff::tz::TimeZone::get(&self.reset_tz) {
            anyhow::bail!(
                "{table} reset_tz {:?} is not a known time zone: {e}",
                self.reset_tz
            );
        }
        Ok(())
    }
}

impl ClaudeConfig {
    /// Whether Claude runs under the caller's own login: no profiles are
    /// configured, so there is no account to choose and no usage to read.
    pub fn ambient(&self) -> bool {
        self.profiles.is_empty()
    }

    /// The site config's account-tuning knobs as one layer, each value
    /// checked and refused with its table's name.
    fn tuning_layer(&self) -> Result<TuningLayer> {
        let sel = &self.selection;
        let mut accounts = BTreeMap::new();
        for (name, ps) in &sel.profile {
            let table = format!("[claude.selection.profile.{name}]");
            let priority = Priority {
                weight: checked_weight(&table, ps.weight)?,
                weekly_reset: weekly_reset(
                    &table,
                    ps.weekly_reset_at.as_ref(),
                    ps.weekly_reset_earlier_h,
                )?,
            };
            let mut rules = Vec::new();
            for (i, r) in ps.by_model.iter().enumerate() {
                let table = format!(
                    "[[claude.selection.profile.{name}.by_model]] entry {}",
                    i + 1
                );
                checked_models(&table, &r.models)?;
                let priority = Priority {
                    weight: checked_weight(&table, r.weight)?,
                    weekly_reset: weekly_reset(
                        &table,
                        r.weekly_reset_at.as_ref(),
                        r.weekly_reset_earlier_h,
                    )?,
                };
                tuning::check_rule_sets(&priority).map_err(|e| anyhow::anyhow!("{table} {e}"))?;
                rules.push(ModelRule {
                    models: r.models.clone(),
                    priority,
                });
            }
            accounts.insert(
                name.clone(),
                AccountLayer {
                    five_hour_floor_pct: ps.five_hour_floor_pct,
                    five_hour_knee_pct: ps.five_hour_knee_pct,
                    priority,
                    by_model: Some(rules),
                },
            );
        }
        for (name, key) in &self.api_keys {
            let table = format!("[claude.api_keys.{name}]");
            let mut rules = Vec::new();
            for (i, r) in key.by_model.iter().enumerate() {
                let table = format!("[[claude.api_keys.{name}.by_model]] entry {}", i + 1);
                checked_models(&table, &r.models)?;
                rules.push(ModelRule {
                    models: r.models.clone(),
                    priority: Priority {
                        weight: checked_weight(&table, Some(r.weight))?,
                        weekly_reset: None,
                    },
                });
            }
            accounts.insert(
                name.clone(),
                AccountLayer {
                    priority: Priority {
                        weight: checked_weight(&table, key.weight)?,
                        weekly_reset: None,
                    },
                    by_model: Some(rules),
                    ..AccountLayer::default()
                },
            );
        }
        Ok(TuningLayer {
            five_hour_floor_pct: Some(sel.five_hour_floor_pct),
            five_hour_knee_pct: Some(sel.five_hour_knee_pct),
            accounts,
        })
    }
}

/// Knobs for the account scorer. The 5-hour floor and knee and the
/// `profile` tables are private: they are read into [`Config::tuning`] at
/// load, and selection reads only that.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SelectionConfig {
    five_hour_floor_pct: f64,
    five_hour_knee_pct: f64,
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
    /// 5-hour floor and knee, and the profile's priority knobs.
    profile: BTreeMap<String, ProfileSelection>,
}

/// One profile's overrides of the global 5-hour floor and knee, in remaining
/// percent (an absent key falls back to the global value), and its priority
/// knobs. Read into [`Tuning`] at load.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct ProfileSelection {
    five_hour_floor_pct: Option<f64>,
    five_hour_knee_pct: Option<f64>,
    weight: Option<f64>,
    weekly_reset_at: Option<TomlTime>,
    weekly_reset_earlier_h: Option<f64>,
    by_model: Vec<ModelPriorityToml>,
}

/// One `[[claude.selection.profile.<name>.by_model]]` entry.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelPriorityToml {
    models: Vec<String>,
    weight: Option<f64>,
    weekly_reset_at: Option<TomlTime>,
    weekly_reset_earlier_h: Option<f64>,
}

/// A TOML offset date-time, unquoted or as an RFC 3339 string.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
enum TomlTime {
    Datetime(toml::value::Datetime),
    Text(String),
}

impl TomlTime {
    fn timestamp(&self, table: &str) -> Result<jiff::Timestamp> {
        let text = match self {
            TomlTime::Datetime(d) => d.to_string(),
            TomlTime::Text(s) => s.clone(),
        };
        text.parse().map_err(|_| {
            anyhow::anyhow!(
                "{table} weekly_reset_at must be a date and time with an offset, such as 2026-11-14T08:00:00Z (got {text})"
            )
        })
    }
}

/// The weekly-reset override one table sets, refusing both forms at once.
fn weekly_reset(
    table: &str,
    at: Option<&TomlTime>,
    earlier_h: Option<f64>,
) -> Result<Option<WeeklyReset>> {
    match (at, earlier_h) {
        (Some(_), Some(_)) => {
            anyhow::bail!("{table} sets both weekly_reset_at and weekly_reset_earlier_h; set one")
        }
        (Some(t), None) => Ok(Some(WeeklyReset::At(t.timestamp(table)?))),
        (None, Some(h)) => {
            tuning::check_earlier_h(h).map_err(|e| anyhow::anyhow!("{table} {e}"))?;
            Ok(Some(WeeklyReset::EarlierBy(
                jiff::SignedDuration::from_secs_f64(h * 3600.0),
            )))
        }
        (None, None) => Ok(None),
    }
}

/// `table`'s weight, refused unless it is a finite number above 0.
fn checked_weight(table: &str, weight: Option<f64>) -> Result<Option<f64>> {
    if let Some(w) = weight {
        tuning::check_weight(w).map_err(|e| anyhow::anyhow!("{table} {e}"))?;
    }
    Ok(weight)
}

fn checked_models(table: &str, models: &[String]) -> Result<()> {
    tuning::check_rule_models(models).map_err(|e| anyhow::anyhow!("{table} {e}"))
}

impl Default for SelectionConfig {
    fn default() -> Self {
        use crate::tuning::{FIVE_HOUR_FLOOR_PCT, FIVE_HOUR_KNEE_PCT};
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
        if self.mqtt_quiesce_ms == 0 || self.mqtt_timeout_ms == 0 {
            anyhow::bail!(
                "[claude.selection] mqtt_quiesce_ms and mqtt_timeout_ms must be greater than 0 (got {} and {})",
                self.mqtt_quiesce_ms,
                self.mqtt_timeout_ms
            );
        }
        Ok(())
    }
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

/// The longest `[goose] tool_wait_s`: Goose fails a stdio-extension call
/// after a hard-coded 300 s, and a tool reply must reach it before then.
pub const GOOSE_MAX_TOOL_WAIT_S: u64 = 280;

/// `[goose]`: how Goose is run against a self-hosted model endpoint. ccc does
/// not model Goose providers; endpoint settings are Goose's own environment
/// variables, passed through `env`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct GooseConfig {
    /// Goose `--provider`.
    pub provider: String,
    /// `GOOSE_PATH_ROOT`; `None` is `<state home>/ccc/goose`.
    pub root: Option<String>,
    /// How long one `ccc_run`, `ccc_wait` or `shell` call waits, in seconds.
    pub tool_wait_s: u64,
    /// Goose `--max-turns`; `None` leaves Goose's own default.
    pub max_turns: Option<u32>,
    /// Appended to every goose argv.
    pub extra_args: Vec<String>,
    /// Set in Goose's environment, verbatim.
    pub env: BTreeMap<String, String>,
    /// Per-model settings, keyed by the resolved model name.
    pub models: BTreeMap<String, GooseModelConfig>,
    /// The bubblewrap binary: a bare name looked up on `PATH`, or an absolute
    /// path (`~/` allowed).
    pub bwrap: String,
    /// Extra directories mounted read-only in the box, each at its own path.
    pub box_ro: Vec<String>,
    /// `PATH` inside the box.
    pub box_path: String,
}

/// `[goose.models."<model>"]`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct GooseModelConfig {
    /// Set in Goose's environment after `[goose.env]`, winning over it.
    pub env: BTreeMap<String, String>,
}

impl Default for GooseConfig {
    fn default() -> Self {
        GooseConfig {
            provider: "openai".to_string(),
            root: None,
            tool_wait_s: 240,
            max_turns: None,
            extra_args: Vec::new(),
            env: BTreeMap::new(),
            models: BTreeMap::new(),
            bwrap: "bwrap".to_string(),
            box_ro: Vec::new(),
            box_path: "/usr/local/bin:/usr/bin:/bin".to_string(),
        }
    }
}

impl GooseConfig {
    fn validate(&self) -> Result<()> {
        if self.provider.trim().is_empty() {
            anyhow::bail!("[goose] provider is blank");
        }
        if !(1..=GOOSE_MAX_TOOL_WAIT_S).contains(&self.tool_wait_s) {
            anyhow::bail!(
                "[goose] tool_wait_s must be between 1 and {GOOSE_MAX_TOOL_WAIT_S} (got {}); Goose fails a tool call after 300 s",
                self.tool_wait_s
            );
        }
        if self.max_turns == Some(0) {
            anyhow::bail!("[goose] max_turns must be at least 1");
        }
        if self.bwrap.trim().is_empty() {
            anyhow::bail!("[goose] bwrap is blank");
        }
        // A relative path would resolve against a different cwd in each process.
        if self.bwrap.contains('/') && !absolute_after_tilde(&self.bwrap) {
            anyhow::bail!(
                "[goose] bwrap must be a bare name looked up on PATH, or an absolute path (got {:?})",
                self.bwrap
            );
        }
        // Goose ignores a relative GOOSE_PATH_ROOT.
        if let Some(root) = &self.root {
            if !absolute_after_tilde(root) {
                anyhow::bail!("[goose] root must be absolute or start with `~/` (got {root:?})");
            }
        }
        for entry in &self.box_ro {
            if !absolute_after_tilde(entry) {
                anyhow::bail!(
                    "[goose] box_ro entries must be absolute or start with `~/` (got {entry:?})"
                );
            }
        }
        let tables = std::iter::once(("[goose.env]".to_string(), &self.env)).chain(
            self.models
                .iter()
                .map(|(m, c)| (format!("[goose.models.{m:?}] env"), &c.env)),
        );
        for (table, vars) in tables {
            if let Some(key) = vars
                .keys()
                .find(|k| k.is_empty() || k.contains('=') || k.contains('\0'))
            {
                anyhow::bail!("{table} has an invalid variable name {key:?}");
            }
            if let Some(key) = vars
                .keys()
                .find(|k| crate::harness::goose::is_telemetry_var(k))
            {
                anyhow::bail!(
                    "{table} sets {key:?}, which ccc controls: Goose telemetry is always off"
                );
            }
            if let Some(key) = vars
                .keys()
                .find(|k| crate::harness::goose::is_isolation_var(k))
            {
                anyhow::bail!("{table} sets {key:?}, which ccc sets for Goose's isolation");
            }
        }
        Ok(())
    }

    /// The Goose root: `root` with `~` expanded, else `<state home>/ccc/goose`.
    pub fn root_in(&self, env: &dyn EnvSource) -> PathBuf {
        match &self.root {
            Some(root) => expand_tilde_in(env, root),
            None => state_home_in(env).join("ccc/goose"),
        }
    }

    /// The variables set in Goose's environment for `model`: `[goose.env]`,
    /// then that model's `env`, later winning.
    pub fn env_for(&self, model: &str) -> BTreeMap<String, String> {
        let mut out = self.env.clone();
        if let Some(m) = self.models.get(model) {
            out.extend(m.env.iter().map(|(k, v)| (k.clone(), v.clone())));
        }
        out
    }

    /// The `box_ro` directories, `~` expanded and canonicalised. An entry
    /// that does not exist or is not a directory is an error naming it.
    pub fn box_ro_in(&self, env: &dyn EnvSource) -> Result<Vec<PathBuf>> {
        self.box_ro
            .iter()
            .map(|entry| {
                let path = expand_tilde_in(env, entry);
                let canonical = std::fs::canonicalize(&path)
                    .with_context(|| format!("[goose] box_ro entry {}", path.display()))?;
                if !canonical.is_dir() {
                    anyhow::bail!("[goose] box_ro entry {} is not a directory", path.display());
                }
                Ok(canonical)
            })
            .collect()
    }
}

/// Whether `s` is an absolute path, or one that `~` expansion makes absolute.
fn absolute_after_tilde(s: &str) -> bool {
    s == "~" || s.starts_with("~/") || std::path::Path::new(s).is_absolute()
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct MqttConfig {
    pub config: Option<String>,
}

/// `[bus]`: publishing this host's invocations and API key spend into the
/// agent tree of the usage broker named by `[mqtt]`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct BusConfig {
    /// Off: no connection, nothing published, budgets read this host only.
    pub enabled: bool,
    /// The `<pub>` segment of every topic. Topic-safe.
    pub publisher: String,
    /// The `<h>` segment; `None` takes the system hostname.
    pub host: Option<String>,
    /// How many leading characters of the prompt go into a session's
    /// `title`; 0 publishes no prompt text.
    pub prompt_chars: usize,
    pub connect_timeout_ms: u64,
    /// The most a clean exit waits for queued publishes.
    pub flush_timeout_ms: u64,
}

impl Default for BusConfig {
    fn default() -> Self {
        BusConfig {
            enabled: false,
            publisher: "ccc".to_string(),
            host: None,
            prompt_chars: 0,
            connect_timeout_ms: 3000,
            flush_timeout_ms: 2000,
        }
    }
}

impl BusConfig {
    fn validate(&self) -> Result<()> {
        if !claude_usage_mqtt_core::is_topic_safe(&self.publisher) {
            anyhow::bail!(
                "[bus] publisher {:?} is not topic-safe (letters, digits, '.', '_', '-')",
                self.publisher
            );
        }
        if let Some(host) = &self.host {
            if !claude_usage_mqtt_core::is_topic_safe(host) {
                anyhow::bail!(
                    "[bus] host {host:?} is not topic-safe (letters, digits, '.', '_', '-')"
                );
            }
        }
        if self.connect_timeout_ms == 0 || self.flush_timeout_ms == 0 {
            anyhow::bail!(
                "[bus] connect_timeout_ms and flush_timeout_ms must be greater than 0 (got {} and {})",
                self.connect_timeout_ms,
                self.flush_timeout_ms
            );
        }
        Ok(())
    }
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
        let mut cfg = Config::compose(site_file, agents_file);
        cfg.tuning = cfg
            .validate()
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
            goose: site.goose,
            mqtt: site.mqtt,
            bus: site.bus,
            process: site.process,
            launch: site.launch,
            tuning: Tuning::default(),
        }
    }

    /// Refuses a config that cannot be run, and resolves its account-tuning
    /// knobs.
    fn validate(&self) -> Result<Tuning> {
        if self.claude.auth == ClaudeAuth::ApiKey {
            anyhow::bail!(
                "[claude] auth = \"api-key\" is not a way to authenticate profiles; configure API keys under [claude.api_keys.<name>]"
            );
        }
        for (name, key) in &self.claude.api_keys {
            key.validate(name, &self.claude.profiles)?;
        }
        // With the bus on, each profile's exhausted flags are published under
        // its name as a topic segment.
        if self.bus.enabled {
            if let Some(name) = self
                .claude
                .profiles
                .iter()
                .find(|p| !claude_usage_mqtt_core::is_topic_safe(p))
            {
                anyhow::bail!(
                    "[claude] profiles: {name:?} is not topic-safe (letters, digits, '.', '_', '-'), which [bus] enabled = true needs: the profile's exhausted flags are published under its name"
                );
            }
        }
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
        self.goose.validate()?;
        self.bus.validate()?;
        let tuning = Tuning::default().overlay(&self.claude.tuning_layer()?);
        if let Err(fault) = tuning.check_ramps() {
            match &fault.account {
                None => anyhow::bail!("[claude.selection] {fault}"),
                Some(name) => anyhow::bail!(
                    "[claude.selection.profile.{name}] (unset keys fall back to [claude.selection]) {fault}"
                ),
            }
        }
        Ok(tuning)
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
        assert_eq!(l.expand(&l.model), "opus[1m]");
    }

    #[test]
    fn launch_aliases_layer_over_the_built_ins_one_level_deep() {
        let l = site_cfg(
            "[launch.aliases]\nfable = \"my-fable\"\nhaiku = \"claude-haiku-4-5\"\nchain = \"opus\"\n",
        )
        .unwrap()
        .launch;
        assert_eq!(l.expand("opus"), "opus[1m]");
        assert_eq!(l.expand("fable"), "my-fable");
        assert_eq!(l.expand("haiku"), "claude-haiku-4-5");
        assert_eq!(l.expand("sonnet"), "sonnet");
        assert_eq!(l.expand("claude-opus-4-6[1M]"), "claude-opus-4-6[1M]");
        assert_eq!(l.expand("chain"), "opus");
        assert_eq!(LaunchConfig::default().expand("fable"), "fable[1m]");
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

    const KEY: &str = "[claude.api_keys.team]\nkey_file = \"~/keys/team.key\"\nbudget_usd = 420.0\nmodels = [\"*fable*\"]\n";

    #[test]
    fn api_keys_parse_with_their_defaults() {
        assert!(site_cfg("").unwrap().claude.api_keys.is_empty());
        let c = site_cfg(KEY).unwrap();
        // A key needs no profiles: with none it is reachable through --profile.
        assert!(c.claude.ambient());
        let key = &c.claude.api_keys["team"];
        assert_eq!(key.key_file, "~/keys/team.key");
        assert_eq!(key.budget_usd, 420.0);
        assert_eq!(key.models, vec!["*fable*".to_string()]);
        assert_eq!(key.reserve_usd, 0.0);
        assert_eq!(key.reset_tz, "UTC");
        assert_eq!(
            key.exhausted_patterns,
            vec!["*credit balance is too low*".to_string()]
        );
        let c = site_cfg(&format!(
            "{KEY}reserve_usd = 20.0\nreset_tz = \"America/New_York\"\nexhausted_patterns = [\"*spend limit*\"]\n"
        ))
        .unwrap();
        let key = &c.claude.api_keys["team"];
        assert_eq!(key.reserve_usd, 20.0);
        assert_eq!(key.reset_tz, "America/New_York");
        assert_eq!(key.exhausted_patterns, vec!["*spend limit*".to_string()]);
    }

    #[test]
    fn api_key_tables_are_refused_by_name() {
        let refused = |text: &str, expected: &str| {
            let err = format!("{:#}", site_cfg(text).unwrap_err());
            assert!(err.contains(expected), "{text}\n=> {err}");
        };
        for field in ["key_file", "budget_usd", "models"] {
            let text: String = KEY
                .lines()
                .filter(|l| !l.starts_with(field))
                .map(|l| format!("{l}\n"))
                .collect();
            refused(&text, field);
        }
        refused(&format!("{KEY}budgte_usd = 1.0\n"), "budgte_usd");
        refused(
            &KEY.replace("api_keys.team]", "api_keys.\"te am\"]"),
            "[claude.api_keys.te am]: the name may use only",
        );
        refused(
            &format!("[claude]\nprofiles = [\"team\"]\n{KEY}"),
            "team is also in [claude] profiles",
        );
        for budget in ["0.0", "-1.0", "nan", "inf"] {
            refused(
                &KEY.replace("420.0", budget),
                "[claude.api_keys.team] budget_usd must be",
            );
        }
        for reserve in ["-1.0", "420.0", "500.0", "nan"] {
            refused(
                &format!("{KEY}reserve_usd = {reserve}\n"),
                "[claude.api_keys.team] reserve_usd must be",
            );
        }
        refused(
            &KEY.replace("[\"*fable*\"]", "[]"),
            "[claude.api_keys.team] models must",
        );
        refused(
            &format!("{KEY}exhausted_patterns = []\n"),
            "[claude.api_keys.team] exhausted_patterns must",
        );
        refused(
            &format!("{KEY}reset_tz = \"Mars/Olympus_Mons\"\n"),
            "[claude.api_keys.team] reset_tz \"Mars/Olympus_Mons\" is not a known time zone",
        );
    }

    #[test]
    fn api_key_auth_is_refused_for_profiles() {
        let err = format!(
            "{:#}",
            site_cfg("[claude]\nauth = \"api-key\"\n").unwrap_err()
        );
        assert!(
            err.contains("[claude] auth = \"api-key\" is not a way"),
            "{err}"
        );
        assert_eq!(ClaudeAuth::ApiKey.as_str(), "api-key");
    }

    #[test]
    fn empty_config_is_all_defaults() {
        let c = site_cfg("").unwrap();
        assert_eq!(c.default_harness, Harness::Claude);
        assert_eq!(c.claude.auth, ClaudeAuth::ConfigDir);
        assert_eq!(c.claude.permission_mode, PermissionMode::Auto);
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
weight = 1.5
weekly_reset_earlier_h = 36.0

[[claude.selection.profile.personal.by_model]]
models = ["*fable*"]
weight = 3.0
weekly_reset_at = 2026-11-14T08:00:00Z

[claude.api_keys.team]
key_file = "~/.config/ccc/team.key"
budget_usd = 420.0
models = ["*"]
weight = 2.0

[[claude.api_keys.team.by_model]]
models = ["*opus*"]
weight = 0.5

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
        assert_eq!(c.tuning.ramp("personal"), (20.0, 50.0));
        assert_eq!(c.tuning.ramp("team03"), (1.0, 20.0));
        let personal = c.tuning.priority("personal", "claude-opus-5[1M]");
        assert_eq!(personal.weight, 1.5);
        assert_eq!(
            personal.weekly_reset,
            Some(WeeklyReset::EarlierBy(jiff::SignedDuration::from_hours(36)))
        );
        let fable = c.tuning.priority("personal", "claude-fable-5-1[1m]");
        assert_eq!((fable.weight, fable.weight_rule), (3.0, Some(0)));
        assert_eq!(
            (fable.weekly_reset, fable.reset_rule),
            (
                Some(WeeklyReset::At("2026-11-14T08:00:00Z".parse().unwrap())),
                Some(0)
            )
        );
        assert_eq!(
            c.tuning.priority("team", "claude-fable-5-1[1m]").weight,
            2.0
        );
        let opus = c.tuning.priority("team", "claude-opus-5[1M]");
        assert_eq!((opus.weight, opus.weight_rule), (0.5, Some(0)));
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
        for m in [
            PermissionMode::AcceptEdits,
            PermissionMode::Auto,
            PermissionMode::BypassPermissions,
            PermissionMode::Manual,
            PermissionMode::DontAsk,
            PermissionMode::Plan,
        ] {
            assert_eq!(
                site_cfg(&format!("[claude]\npermission_mode = \"{}\"", m.as_str()))
                    .unwrap()
                    .claude
                    .permission_mode,
                m
            );
        }
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
            agents_cfg("[agents.x]\ngoose_modl = \"m\""),
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
        assert_eq!(Harness::parse("GOOSE").unwrap(), Harness::Goose);
        let err = Harness::parse("gemini").unwrap_err().to_string();
        assert!(err.contains("\"claude\", \"codex\" or \"goose\""), "{err}");
        for h in Harness::ALL {
            assert_eq!(Harness::parse(h.as_str()).unwrap(), h);
        }
    }

    #[test]
    fn goose_accepts_no_effort() {
        assert!(Effort::accepted_by(Harness::Goose).is_empty());
        let err = Effort::parse_for("low", Harness::Goose)
            .unwrap_err()
            .to_string();
        assert!(err.contains("goose accepts no effort"), "{err}");
    }

    #[test]
    fn goose_defaults_apply_without_a_section() {
        let g = site_cfg("").unwrap().goose;
        assert_eq!(g.provider, "openai");
        assert_eq!(g.root, None);
        assert_eq!(g.tool_wait_s, 240);
        assert_eq!(g.max_turns, None);
        assert!(g.extra_args.is_empty() && g.env.is_empty() && g.models.is_empty());
        assert_eq!(g.bwrap, "bwrap");
        assert!(g.box_ro.is_empty());
        assert_eq!(g.box_path, "/usr/local/bin:/usr/bin:/bin");
        let e = env(&[("HOME", "/home/u")]);
        assert_eq!(
            g.root_in(&e),
            PathBuf::from("/home/u/.local/state/ccc/goose")
        );
        let e = env(&[("HOME", "/home/u"), ("XDG_STATE_HOME", "/xdg/state")]);
        assert_eq!(g.root_in(&e), PathBuf::from("/xdg/state/ccc/goose"));
    }

    #[test]
    fn parses_the_documented_goose_section() {
        let g = site_cfg(
            r#"
[goose]
provider = "ollama"
root = "~/goose-root"
tool_wait_s = 30
max_turns = 200
extra_args = ["--debug"]
bwrap = "/usr/bin/bwrap"
box_ro = ["~/.cargo", "/opt/rust"]
box_path = "/usr/bin"

[goose.env]
OPENAI_BASE_URL = "http://127.0.0.1:8080/v1"
OPENAI_API_KEY = "none"

[goose.models."qwen3-coder-30b"]
env = { OPENAI_BASE_URL = "http://127.0.0.1:8081/v1" }
"#,
        )
        .unwrap()
        .goose;
        assert_eq!(g.provider, "ollama");
        assert_eq!(g.tool_wait_s, 30);
        assert_eq!(g.max_turns, Some(200));
        assert_eq!(g.extra_args, vec!["--debug"]);
        assert_eq!(g.bwrap, "/usr/bin/bwrap");
        assert_eq!(g.box_ro, vec!["~/.cargo", "/opt/rust"]);
        assert_eq!(g.box_path, "/usr/bin");
        let e = env(&[("HOME", "/home/u")]);
        assert_eq!(g.root_in(&e), PathBuf::from("/home/u/goose-root"));

        let base = g.env_for("other-model");
        assert_eq!(base["OPENAI_BASE_URL"], "http://127.0.0.1:8080/v1");
        assert_eq!(base.len(), 2);
        let qwen = g.env_for("qwen3-coder-30b");
        assert_eq!(qwen["OPENAI_BASE_URL"], "http://127.0.0.1:8081/v1");
        assert_eq!(qwen["OPENAI_API_KEY"], "none");
    }

    #[test]
    fn goose_values_that_cannot_work_are_refused_at_load() {
        for (text, needle) in [
            (
                "[goose]
tool_wait_s = 0
",
                "tool_wait_s",
            ),
            (
                "[goose]
tool_wait_s = 281
",
                "tool_wait_s",
            ),
            (
                "[goose]
max_turns = 0
",
                "max_turns",
            ),
            (
                "[goose]
provider = \" \"
",
                "provider",
            ),
            (
                "[goose]
bwrap = \"\"
",
                "bwrap",
            ),
            (
                "[goose]
bwrap = \"tools/bwrap\"
",
                "[goose] bwrap must be a bare name",
            ),
            (
                "[goose]
bwrap = \"./bwrap\"
",
                "[goose] bwrap must be a bare name",
            ),
            (
                "[goose]
root = \"rel/goose\"
",
                "root",
            ),
            (
                "[goose]
box_ro = [\"rel\"]
",
                "box_ro",
            ),
            (
                "[goose.env]
\"A=B\" = \"x\"
",
                "[goose.env]",
            ),
            (
                "[goose.models.m]
env = { \"\" = \"x\" }
",
                "[goose.models.\"m\"] env",
            ),
            (
                "[goose]
model = \"x\"
",
                "unknown field",
            ),
            (
                "[goose.models.m]
envv = {}
",
                "unknown field",
            ),
        ] {
            let err = format!("{:#}", site_cfg(text).unwrap_err());
            assert!(err.contains(needle), "{text}: {err}");
        }
        for ok in [
            "[goose]
tool_wait_s = 1
",
            "[goose]
tool_wait_s = 280
",
            "[goose]
root = \"/srv/goose\"
",
            "[goose]
root = \"~\"
",
        ] {
            site_cfg(ok).unwrap_or_else(|e| panic!("{ok}: {e:#}"));
        }
    }

    #[test]
    fn goose_telemetry_variables_are_refused_in_every_env_table() {
        for key in [
            "GOOSE_TELEMETRY_OFF",
            "GOOSE_TELEMETRY_ENABLED",
            "OTEL_SDK_DISABLED",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_INIT_PROJECT_PUBLIC_KEY",
            "LANGFUSE_INIT_PROJECT_SECRET_KEY",
            "LANGFUSE_HOST",
        ] {
            for (text, table) in [
                (format!("[goose.env]\n{key} = \"x\"\n"), "[goose.env]"),
                (
                    format!("[goose.models.m]\nenv = {{ {key} = \"x\" }}\n"),
                    "[goose.models.\"m\"] env",
                ),
            ] {
                let err = format!("{:#}", site_cfg(&text).unwrap_err());
                assert!(err.contains(table), "{text}: {err}");
                assert!(err.contains(&format!("{key:?}")), "{text}: {err}");
                assert!(err.contains("telemetry is always off"), "{text}: {err}");
            }
        }
        let g = site_cfg(
            "[goose.env]
OPENAI_BASE_URL = \"http://127.0.0.1:8090/v1\"
OTEL_EXPORTER_OTLP_ENDPOINT = \"http://127.0.0.1:4318\"
[goose.models.m]
env = { OPENAI_API_KEY = \"none\" }
",
        )
        .unwrap()
        .goose;
        assert_eq!(g.env["OPENAI_BASE_URL"], "http://127.0.0.1:8090/v1");
    }

    #[test]
    fn goose_isolation_variables_are_refused_in_every_env_table() {
        for key in ["HOME", "HTTPS_PROXY", "https_proxy", "CCC_GOOSE_HOST_ENV"] {
            for (text, table) in [
                (format!("[goose.env]\n{key} = \"x\"\n"), "[goose.env]"),
                (
                    format!("[goose.models.m]\nenv = {{ {key} = \"x\" }}\n"),
                    "[goose.models.\"m\"] env",
                ),
            ] {
                let err = format!("{:#}", site_cfg(&text).unwrap_err());
                assert!(err.contains(table), "{text}: {err}");
                assert!(err.contains(&format!("{key:?}")), "{text}: {err}");
                assert!(err.contains("Goose's isolation"), "{text}: {err}");
            }
        }
        let g = site_cfg(
            "[goose.env]\nNO_PROXY = \"llm.example\"\nno_proxy = \"llm.example\"\nHTTP_PROXY = \"http://p:3128\"\nALL_PROXY = \"http://p:3128\"\n",
        )
        .unwrap()
        .goose;
        assert_eq!(g.env["NO_PROXY"], "llm.example");
    }

    #[test]
    fn goose_in_the_agents_config_is_refused_with_the_wrong_file_message() {
        let err = format!(
            "{:#}",
            agents_cfg(
                "[goose]
"
            )
            .unwrap_err()
        );
        assert!(
            err.contains("`goose` belongs in the site config, not the agents config."),
            "{err}"
        );
    }

    #[test]
    fn box_ro_entries_are_expanded_and_canonicalised_when_resolved() {
        let home = crate::testutil::TempDir::new("goose-box-ro");
        std::fs::create_dir_all(home.path().join("real/.cargo")).unwrap();
        std::os::unix::fs::symlink(home.path().join("real"), home.path().join("link")).unwrap();
        home.write_file("file", "", 0o600);
        let canonical_home = std::fs::canonicalize(home.path()).unwrap();
        let e = env(&[("HOME", home.path().to_str().unwrap())]);
        let g = site_cfg(
            "[goose]
box_ro = [\"~/link/.cargo\"]
",
        )
        .unwrap()
        .goose;
        assert_eq!(
            g.box_ro_in(&e).unwrap(),
            vec![canonical_home.join("real/.cargo")]
        );
        for (entry, needle) in [("~/missing", "missing"), ("~/file", "not a directory")] {
            let g = site_cfg(&format!(
                "[goose]
box_ro = [\"{entry}\"]
"
            ))
            .unwrap()
            .goose;
            let err = format!("{:#}", g.box_ro_in(&e).unwrap_err());
            assert!(err.contains("box_ro") && err.contains(needle), "{err}");
        }
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
    fn bus_defaults_when_the_section_is_absent() {
        let cfg = site_cfg("").unwrap();
        assert!(!cfg.bus.enabled);
        assert_eq!(cfg.bus.publisher, "ccc");
        assert_eq!(cfg.bus.host, None);
        assert_eq!(cfg.bus.prompt_chars, 0);
        assert_eq!(cfg.bus.connect_timeout_ms, 3000);
        assert_eq!(cfg.bus.flush_timeout_ms, 2000);
        assert!(cfg.claude.api_keys.is_empty());

        let cfg = site_cfg(
            "[bus]\nenabled = true\npublisher = \"ccc-lab\"\nhost = \"atlas\"\nprompt_chars = 40\nconnect_timeout_ms = 500\nflush_timeout_ms = 100\n",
        )
        .unwrap();
        assert!(cfg.bus.enabled);
        assert_eq!(cfg.bus.publisher, "ccc-lab");
        assert_eq!(cfg.bus.host.as_deref(), Some("atlas"));
        assert_eq!(cfg.bus.prompt_chars, 40);
        assert_eq!(cfg.bus.connect_timeout_ms, 500);
        assert_eq!(cfg.bus.flush_timeout_ms, 100);
    }

    #[test]
    fn bus_values_that_cannot_name_a_topic_or_bound_a_wait_are_refused() {
        let cases = [
            ("publisher = \"c c\"", "publisher"),
            ("publisher = \"\"", "publisher"),
            ("publisher = \"ccc/x\"", "publisher"),
            ("host = \"at+las\"", "host"),
            ("connect_timeout_ms = 0", "connect_timeout_ms"),
            ("flush_timeout_ms = 0", "flush_timeout_ms"),
        ];
        for (line, key) in cases {
            let text = format!("[bus]\n{line}\n");
            let err = site_cfg(&text).unwrap_err().root_cause().to_string();
            assert!(err.contains(key), "{line} gave {err}");
            assert!(err.contains("[bus]"), "{line} gave {err}");
        }
        let err = format!("{:#}", site_cfg("[bus]\ntopic = \"x\"\n").unwrap_err());
        assert!(err.contains("topic"), "{err}");
    }

    #[test]
    fn a_profile_name_that_cannot_name_a_topic_is_refused_only_with_the_bus_on() {
        let profiles = "[claude]\nprofiles = [\"team01\", \"rn+work\"]\n";
        let c = site_cfg(profiles).unwrap();
        assert_eq!(c.claude.profiles, vec!["team01", "rn+work"]);
        let err = site_cfg(&format!("{profiles}[bus]\nenabled = true\n"))
            .unwrap_err()
            .root_cause()
            .to_string();
        assert!(
            err.contains("[claude] profiles: \"rn+work\" is not topic-safe"),
            "{err}"
        );
        site_cfg("[claude]\nprofiles = [\"team-01.a_b\"]\n[bus]\nenabled = true\n").unwrap();
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
                ..ProfileSelection::default()
            }
        );
        assert_eq!(c.tuning.ramp("personal"), (20.0, 60.0));
        // Only the floor is overridden; the knee is the global one.
        assert_eq!(c.tuning.ramp("team01"), (5.0, 20.0));
        // No override at all: both globals.
        assert_eq!(c.tuning.ramp("team02"), (1.0, 20.0));

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
            let text = format!(
                "[claude]\nprofiles = [\"personal\"]\n\n[claude.selection.profile.personal]\n{lines}\n"
            );
            let err = site_cfg(&text).unwrap_err().root_cause().to_string();
            assert!(err.contains(key), "{lines} gave {err}");
            assert!(
                err.contains("[claude.selection.profile.personal]"),
                "{lines} gave {err}"
            );
        }

        // The message shows the resolved pair, so the bad value is visible.
        let err = site_cfg(
            "[claude]\nprofiles = [\"personal\"]\n\n\
             [claude.selection.profile.personal]\nfive_hour_knee_pct = 160.0\n",
        )
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
    fn priority_knobs_parse_both_time_forms_and_refuse_bad_values_by_table() {
        let head = "[claude]\nprofiles = [\"p\"]\n\n[claude.selection.profile.p]\n";
        let at = |v: &str| site_cfg(&format!("{head}weekly_reset_at = {v}\n"));
        let want = Some(WeeklyReset::At("2026-11-14T08:00:00Z".parse().unwrap()));
        for v in [
            "2026-11-14T08:00:00Z",
            "\"2026-11-14T08:00:00Z\"",
            "2026-11-14T09:00:00+01:00",
        ] {
            assert_eq!(
                at(v).unwrap().tuning.priority("p", "m").weekly_reset,
                want,
                "{v}"
            );
        }
        for v in ["2026-11-14T08:00:00", "2026-11-14", "\"soon\""] {
            let err = at(v).unwrap_err().root_cause().to_string();
            assert!(
                err.starts_with(
                    "[claude.selection.profile.p] weekly_reset_at must be a date and time with an offset, such as 2026-11-14T08:00:00Z (got "
                ),
                "{v} gave {err}"
            );
        }

        let key = "[claude.api_keys.k]\nkey_file = \"/k\"\nbudget_usd = 1.0\nmodels = [\"*\"]\n";
        let rule = "[[claude.selection.profile.p.by_model]]\n";
        let key_rule = "[[claude.api_keys.k.by_model]]\n";
        let cases = [
            (format!("{head}weight = 0.0\n"), "[claude.selection.profile.p] weight must be from 0.000001 to 1000000 (got 0)"),
            (format!("{head}weekly_reset_earlier_h = 168.0\n"), "[claude.selection.profile.p] weekly_reset_earlier_h must be a finite number, at least 0 and below 168 (got 168)"),
            (format!("{head}weekly_reset_earlier_h = 1.0\nweekly_reset_at = 2026-11-14T08:00:00Z\n"), "[claude.selection.profile.p] sets both weekly_reset_at and weekly_reset_earlier_h; set one"),
            (format!("{head}\n{rule}models = [\"*\"]\nweight = 1.0\n\n{rule}models = []\nweight = 1.0\n"), "[[claude.selection.profile.p.by_model]] entry 2 models must name at least one model glob"),
            (format!("{head}\n{rule}models = [\"*\"]\n"), "[[claude.selection.profile.p.by_model]] entry 1 sets neither weight nor a weekly reset; a rule that sets nothing still stops later rules from matching"),
            (format!("{head}\n{rule}models = [\"*\"]\nweight = -1.0\n"), "[[claude.selection.profile.p.by_model]] entry 1 weight must be from 0.000001 to 1000000 (got -1)"),
            (format!("{key}weight = inf\n"), "[claude.api_keys.k] weight must be from 0.000001 to 1000000 (got inf)"),
            (format!("{key}\n{key_rule}models = []\nweight = 1.0\n"), "[[claude.api_keys.k.by_model]] entry 1 models must name at least one model glob"),
            (format!("{key}\n{key_rule}models = [\"*\"]\nweight = nan\n"), "[[claude.api_keys.k.by_model]] entry 1 weight must be from 0.000001 to 1000000 (got NaN)"),
        ];
        for (text, want) in cases {
            let err = site_cfg(&text).unwrap_err().root_cause().to_string();
            assert_eq!(err, want, "{text}");
        }
        // The smallest positive f64 would underflow any score to zero.
        let err = site_cfg(&format!("{head}weight = 5e-324\n"))
            .unwrap_err()
            .root_cause()
            .to_string();
        assert_eq!(
            err,
            format!(
                "[claude.selection.profile.p] weight must be from 0.000001 to 1000000 (got {})",
                5e-324_f64
            )
        );

        for (text, want) in [
            (
                format!("{head}\n{rule}models = [\"*\"]\nwieght = 1.0\n"),
                "unknown field",
            ),
            (
                format!("{key}weekly_reset_earlier_h = 1.0\n"),
                "unknown field",
            ),
            (
                format!(
                    "{key}\n{key_rule}models = [\"*\"]\nweekly_reset_at = 2026-11-14T08:00:00Z\n"
                ),
                "unknown field",
            ),
            (
                format!("{key}\n{key_rule}models = [\"*\"]\n"),
                "missing field",
            ),
        ] {
            let err = format!("{:#}", site_cfg(&text).unwrap_err());
            assert!(err.contains(want), "{text} gave {err}");
        }
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
