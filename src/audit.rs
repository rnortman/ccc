//! Append-only invocation audit records.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use jiff::Timestamp;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::agents::Sandbox;
use crate::config::{self, ClaudeAuth, Effort, EnvSource, Harness};
use crate::procs::ProcId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexPolicy {
    Append,
    AppendIfAbsent,
    None,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lineage {
    pub trace_id: String,
    pub invocation_id: String,
    pub parent: Option<String>,
    pub index_policy: IndexPolicy,
}

/// The inherited audit variables, parsed without allocating an invocation or
/// emitting a diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InheritedLineage {
    Absent,
    TraceOnly(String),
    TraceParent { trace: String, parent: String },
    Malformed,
}

pub fn inherited_lineage(env: &dyn EnvSource) -> InheritedLineage {
    let trace = env.get("CCC_TRACE_ID");
    let parent = env.get("CCC_PARENT_INVOCATION_ID");
    match (trace, parent) {
        (None, None) => InheritedLineage::Absent,
        (Some(t), None) if parse_uuid(&t).is_ok() => InheritedLineage::TraceOnly(t),
        (Some(t), Some(p)) if parse_uuid(&t).is_ok() && parse_uuid(&p).is_ok() => {
            InheritedLineage::TraceParent {
                trace: t,
                parent: p,
            }
        }
        _ => InheritedLineage::Malformed,
    }
}

impl Lineage {
    pub fn from_env(env: &dyn EnvSource) -> Lineage {
        let invocation_id = Uuid::new_v4().to_string();
        match inherited_lineage(env) {
            InheritedLineage::Absent => Lineage {
                trace_id: invocation_id.clone(),
                invocation_id,
                parent: None,
                index_policy: IndexPolicy::Append,
            },
            InheritedLineage::TraceParent {
                trace: t,
                parent: p,
            } => Lineage {
                trace_id: t,
                invocation_id,
                parent: Some(p),
                index_policy: IndexPolicy::None,
            },
            InheritedLineage::TraceOnly(t) => Lineage {
                trace_id: t,
                invocation_id,
                parent: None,
                index_policy: IndexPolicy::AppendIfAbsent,
            },
            _ => {
                crate::harness::diag("malformed inherited audit lineage is ignored");
                Lineage {
                    trace_id: invocation_id.clone(),
                    invocation_id,
                    parent: None,
                    index_policy: IndexPolicy::Append,
                }
            }
        }
    }

    pub fn child_env(&self) -> Vec<(String, String)> {
        vec![
            ("CCC_TRACE_ID".into(), self.trace_id.clone()),
            (
                "CCC_PARENT_INVOCATION_ID".into(),
                self.invocation_id.clone(),
            ),
        ]
    }
}

fn parse_uuid(id: &str) -> Result<Uuid> {
    Uuid::parse_str(id).map_err(|e| anyhow::anyhow!(e))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Command {
    Run,
    Resume,
    Launch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptSource {
    Arg,
    File,
    Stdin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookKind {
    CodexNotify,
    ClaudeStop,
}

impl HookKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CodexNotify => "codex-notify",
            Self::ClaudeStop => "claude-stop",
        }
    }
}

impl std::fmt::Display for HookKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for HookKind {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "codex-notify" => Ok(Self::CodexNotify),
            "claude-stop" => Ok(Self::ClaudeStop),
            _ => Err(format!(
                "unknown hook kind {value:?}; expected codex-notify or claude-stop"
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartPayload {
    pub pid: u32,
    pub cmd: Command,
    pub agent: Option<String>,
    pub session: Option<String>,
    pub cwd_arg: Option<String>,
    pub profile_arg: Option<String>,
    pub harness_arg: Option<Harness>,
    pub json: bool,
    pub prompt_source: Option<PromptSource>,
    pub prompt_file: Option<String>,
    pub prompt: Option<String>,
    pub passthrough: Vec<String>,
    /// Whether the resumed `session` is forked into a new one.
    #[serde(default)]
    pub fork: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnPayload {
    pub harness: Harness,
    pub model: String,
    pub effort: Option<Effort>,
    pub profile: Option<String>,
    pub auth: Option<ClaudeAuth>,
    pub sandbox: Sandbox,
    pub tools: Option<Vec<String>>,
    /// `None` for a roleless launch.
    #[serde(default)]
    pub agent: Option<String>,
    pub cwd: String,
    pub session: Option<String>,
    pub selection_notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Turn {
    pub ts: Timestamp,
    pub inv: String,
    pub harness: Harness,
    pub session: String,
    pub turn: Option<String>,
    pub result: Option<String>,
    pub transcript: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndPayload {
    pub exit: Option<i32>,
    pub signal: Option<i32>,
    pub session: Option<String>,
    pub result: Option<String>,
    pub error: Option<String>,
    pub warnings: Vec<String>,
    pub native: Option<serde_json::Value>,
    pub turns: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChildPayload {
    pub proc: ProcId,
    pub pgid: u32,
    pub supervisor: ProcId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChildRecord {
    pub ts: Timestamp,
    pub inv: String,
    #[serde(flatten)]
    pub payload: ChildPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    pub ts: Timestamp,
    pub inv: String,
    pub session: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SignalDestination {
    Supervisor,
    Child,
    Descendants,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalPayload {
    pub signal: i32,
    pub to: SignalDestination,
    pub by: ProcId,
    pub by_inv: Option<String>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalRecord {
    pub ts: Timestamp,
    pub inv: String,
    #[serde(flatten)]
    pub payload: SignalPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartRecord {
    pub ts: Timestamp,
    pub inv: String,
    pub parent: Option<String>,
    #[serde(flatten)]
    pub payload: StartPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnRecord {
    pub ts: Timestamp,
    pub inv: String,
    #[serde(flatten)]
    pub payload: SpawnPayload,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndRecord {
    pub ts: Timestamp,
    pub inv: String,
    #[serde(flatten)]
    pub payload: EndPayload,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "t")]
pub enum Record {
    #[serde(rename = "start")]
    Start(StartRecord),
    #[serde(rename = "spawn")]
    Spawn(SpawnRecord),
    #[serde(rename = "turn")]
    Turn(Turn),
    #[serde(rename = "end")]
    End(EndRecord),
    #[serde(rename = "child")]
    Child(ChildRecord),
    #[serde(rename = "session")]
    Session(SessionRecord),
    #[serde(rename = "signal")]
    Signal(SignalRecord),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexLine {
    pub trace: String,
    pub ts: Timestamp,
    pub inv: String,
    pub pid: u32,
    pub cmd: Command,
    pub agent: Option<String>,
    pub cwd: String,
    pub config: String,
    pub config_exists: bool,
    /// Agents-config path; empty on index lines written before it was recorded.
    #[serde(default)]
    pub agents_config: String,
    #[serde(default)]
    pub agents_config_exists: bool,
    pub env: BTreeMap<String, String>,
    pub env_other: Vec<String>,
}

pub fn classify_env(vars: &[(String, String)]) -> (BTreeMap<String, String>, Vec<String>) {
    let mut known = BTreeMap::new();
    let mut other = BTreeSet::new();
    for (key, value) in vars {
        if !key.starts_with("CCC_") || key == "CCC_TRACE_ID" || key == "CCC_PARENT_INVOCATION_ID" {
            continue;
        }
        let recognised = matches!(
            key.as_str(),
            "CCC_CONFIG" | "CCC_AGENTS_CONFIG" | "CCC_CLAUDE_BIN" | "CCC_CODEX_BIN"
        ) || override_name(key);
        if recognised {
            known.insert(key.clone(), value.clone());
        } else {
            other.insert(key.clone());
        }
    }
    (known, other.into_iter().collect())
}

fn override_name(key: &str) -> bool {
    let Some(rest) = key.strip_prefix("CCC_") else {
        return false;
    };
    let suffix = [
        "CODEX_MODEL_",
        "CODEX_EFFORT_",
        "HARNESS_",
        "MODEL_",
        "EFFORT_",
    ]
    .iter()
    .find_map(|prefix| rest.strip_prefix(prefix))
    .filter(|suffix| !suffix.is_empty());
    let Some(suffix) = suffix else { return false };
    suffix == "ALL"
        || suffix
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
}

fn audit_dir(env: &dyn EnvSource) -> PathBuf {
    config::state_home_in(env).join("ccc/audit")
}

fn ensure_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating audit directory {}", dir.display()))?;
    #[cfg(unix)]
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
        .with_context(|| format!("setting mode on {}", dir.display()))?;
    Ok(())
}

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

fn open_append(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).append(true);
    #[cfg(unix)]
    options.mode(0o600);
    options
        .open(path)
        .with_context(|| format!("opening audit file {}", path.display()))
}

fn write_locked(file: &mut File, path: &Path, line: &str) -> Result<()> {
    file.lock()
        .with_context(|| format!("locking audit file {}", path.display()))?;
    let result = (|| {
        let bytes = append_bytes(file, path, line)?;
        file.write_all(&bytes)
            .with_context(|| format!("writing audit file {}", path.display()))?;
        file.flush()
            .with_context(|| format!("flushing audit file {}", path.display()))
    })();
    let unlock = file
        .unlock()
        .with_context(|| format!("unlocking audit file {}", path.display()));
    match (result, unlock) {
        (Ok(wrote), Ok(())) => Ok(wrote),
        (Err(e), _) | (_, Err(e)) => Err(e),
    }
}

fn append_bytes(file: &mut File, path: &Path, line: &str) -> Result<Vec<u8>> {
    let end = file
        .seek(std::io::SeekFrom::End(0))
        .with_context(|| format!("seeking audit file {}", path.display()))?;
    let needs_separator = if end == 0 {
        false
    } else {
        let mut tail = [0u8; 1];
        file.seek(std::io::SeekFrom::End(-1))
            .with_context(|| format!("seeking audit file {}", path.display()))?;
        file.read_exact(&mut tail)
            .with_context(|| format!("reading audit file {}", path.display()))?;
        tail[0] != b'\n'
    };
    file.seek(std::io::SeekFrom::End(0))
        .with_context(|| format!("seeking audit file {}", path.display()))?;
    let mut bytes = Vec::with_capacity(line.len() + 2);
    if needs_separator {
        bytes.push(b'\n');
    }
    bytes.extend_from_slice(line.as_bytes());
    bytes.push(b'\n');
    Ok(bytes)
}

pub fn append_line(path: &Path, line: &str) -> Result<()> {
    let mut file = open_append(path)?;
    write_locked(&mut file, path, line)
}

pub fn append_index_if_absent(path: &Path, line: &str) -> Result<bool> {
    let mut file = open_append(path)?;
    file.lock()
        .with_context(|| format!("locking audit file {}", path.display()))?;
    let result = (|| {
        let trace = serde_json::from_str::<serde_json::Value>(line)?
            .get("trace")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("index line has no trace"))?
            .to_string();
        file.rewind()
            .with_context(|| format!("rewinding audit file {}", path.display()))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .with_context(|| format!("reading audit file {}", path.display()))?;
        for (n, existing) in bytes.split(|b| *b == b'\n').enumerate() {
            let Ok(existing) = serde_json::from_slice::<serde_json::Value>(existing) else {
                if !existing.is_empty() {
                    crate::harness::diag(format_args!(
                        "skipping malformed index {} line {}",
                        path.display(),
                        n + 1
                    ));
                }
                continue;
            };
            if existing
                .get("trace")
                .and_then(|x| x.as_str())
                .map(str::to_owned)
                .as_deref()
                == Some(&trace)
            {
                return Ok(false);
            }
        }
        let bytes = append_bytes(&mut file, path, line)?;
        file.write_all(&bytes)
            .with_context(|| format!("writing audit file {}", path.display()))?;
        file.flush()?;
        Ok(true)
    })();
    let unlock = file
        .unlock()
        .with_context(|| format!("unlocking audit file {}", path.display()));
    match (result, unlock) {
        (Ok(wrote), Ok(())) => Ok(wrote),
        (Err(e), _) | (_, Err(e)) => Err(e),
    }
}

fn path_for(dir: &Path, id: &str) -> Result<PathBuf> {
    parse_uuid(id).with_context(|| format!("invalid audit trace id {id:?}"))?;
    Ok(dir.join(format!("{id}.jsonl")))
}

pub struct Recorder {
    lineage: Lineage,
    started_at: Timestamp,
    dir: PathBuf,
    config: String,
    config_exists: bool,
    agents_config: String,
    agents_config_exists: bool,
    env: BTreeMap<String, String>,
    env_other: Vec<String>,
    ended: Mutex<bool>,
}

impl Recorder {
    pub fn dir_in(env: &dyn EnvSource) -> PathBuf {
        audit_dir(env)
    }

    pub fn new(env: &dyn EnvSource) -> Result<Self> {
        let started_at = Timestamp::now();
        let dir = audit_dir(env);
        ensure_dir(&dir)?;
        let config_path = config::config_path_in(env).path;
        let agents_config_path = config::agents_config_path_in(env).path;
        let (classified, other) = classify_env(&env.vars_with_prefix("CCC_"));
        Ok(Self {
            lineage: Lineage::from_env(env),
            started_at,
            dir,
            config: config_path.to_string_lossy().into_owned(),
            config_exists: config_path.exists(),
            agents_config: agents_config_path.to_string_lossy().into_owned(),
            agents_config_exists: agents_config_path.exists(),
            env: classified,
            env_other: other,
            ended: Mutex::new(false),
        })
    }

    pub fn lineage(&self) -> &Lineage {
        &self.lineage
    }

    pub fn trace_path(&self) -> PathBuf {
        self.dir.join(format!("{}.jsonl", self.lineage.trace_id))
    }

    pub fn start(&self, payload: StartPayload) -> Result<()> {
        let trace_path = path_for(&self.dir, &self.lineage.trace_id)?;
        let record = Record::Start(StartRecord {
            ts: self.started_at,
            inv: self.lineage.invocation_id.clone(),
            parent: self.lineage.parent.clone(),
            payload: payload.clone(),
        });
        append_record(&trace_path, &record)?;
        if self.lineage.index_policy != IndexPolicy::None {
            let cwd = payload.cwd_arg.clone().unwrap_or_else(|| {
                std::env::current_dir()
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .to_string_lossy()
                    .into_owned()
            });
            let index = IndexLine {
                trace: self.lineage.trace_id.clone(),
                ts: self.started_at,
                inv: self.lineage.invocation_id.clone(),
                pid: payload.pid,
                cmd: payload.cmd,
                agent: payload.agent.clone(),
                cwd,
                config: self.config.clone(),
                config_exists: self.config_exists,
                agents_config: self.agents_config.clone(),
                agents_config_exists: self.agents_config_exists,
                env: self.env.clone(),
                env_other: self.env_other.clone(),
            };
            let index_path = self.dir.join("traces.jsonl");
            let line = serde_json::to_string(&index)?;
            let result = match self.lineage.index_policy {
                IndexPolicy::Append => append_line(&index_path, &line),
                IndexPolicy::AppendIfAbsent => {
                    append_index_if_absent(&index_path, &line).map(|_| ())
                }
                IndexPolicy::None => Ok(()),
            };
            if let Err(e) = result {
                bail!(
                    "index {} failed after trace {} was written: {e:#}",
                    index_path.display(),
                    trace_path.display()
                );
            }
        }
        Ok(())
    }

    pub fn spawn(&self, payload: SpawnPayload) -> Result<()> {
        self.append(Record::Spawn(SpawnRecord {
            ts: Timestamp::now(),
            inv: self.lineage.invocation_id.clone(),
            payload,
        }))
    }

    pub fn child(&self, payload: ChildPayload) -> Result<()> {
        self.append(Record::Child(ChildRecord {
            ts: Timestamp::now(),
            inv: self.lineage.invocation_id.clone(),
            payload,
        }))
    }

    pub fn session(&self, session: &str) -> Result<()> {
        self.append(Record::Session(SessionRecord {
            ts: Timestamp::now(),
            inv: self.lineage.invocation_id.clone(),
            session: session.to_string(),
        }))
    }

    pub fn signal(&self, payload: SignalPayload) -> Result<()> {
        self.append(Record::Signal(SignalRecord {
            ts: Timestamp::now(),
            inv: self.lineage.invocation_id.clone(),
            payload,
        }))
    }

    fn append(&self, record: Record) -> Result<()> {
        let path = path_for(&self.dir, &self.lineage.trace_id)?;
        append_record(&path, &record)
    }

    pub fn end(&self, payload: EndPayload) {
        let mut ended = self.ended.lock().expect("recorder end mutex poisoned");
        if *ended {
            return;
        }
        *ended = true;
        let duration_ms = Timestamp::now()
            .since(self.started_at)
            .ok()
            .and_then(|d| d.total(jiff::Unit::Millisecond).ok())
            .filter(|ms| *ms >= 0.0)
            .map(|ms| ms as u64)
            .unwrap_or(0);
        if let Err(e) = self.append(Record::End(EndRecord {
            ts: Timestamp::now(),
            inv: self.lineage.invocation_id.clone(),
            payload,
            duration_ms,
        })) {
            crate::harness::diag(format_args!(
                "audit end failed for {}: {e:#}",
                self.dir
                    .join(format!("{}.jsonl", self.lineage.trace_id))
                    .display()
            ));
        }
    }

    pub fn turns(&self) -> Vec<Turn> {
        match read_trace(&self.dir, &self.lineage.trace_id) {
            Ok(records) => records
                .into_iter()
                .filter_map(|r| match r {
                    Record::Turn(t) if t.inv == self.lineage.invocation_id => Some(t),
                    _ => None,
                })
                .collect(),
            Err(e) => {
                crate::harness::diag(format_args!("audit trace read failed: {e:#}"));
                Vec::new()
            }
        }
    }
}

fn append_record(path: &Path, record: &Record) -> Result<()> {
    append_line(path, &serde_json::to_string(record)?)
}

pub fn append_turn(dir: &Path, trace_id: &str, turn: Turn) -> Result<()> {
    append_record(&path_for(dir, trace_id)?, &Record::Turn(turn))
}

pub fn append_signal_for(
    dir: &Path,
    trace_id: &str,
    inv: &str,
    payload: SignalPayload,
) -> Result<()> {
    parse_uuid(inv).context("invalid audit invocation id")?;
    append_record(
        &path_for(dir, trace_id)?,
        &Record::Signal(SignalRecord {
            ts: Timestamp::now(),
            inv: inv.to_string(),
            payload,
        }),
    )
}

pub fn append_end_for(dir: &Path, trace_id: &str, inv: &str, payload: EndPayload) -> Result<bool> {
    parse_uuid(inv).context("invalid audit invocation id")?;
    let path = path_for(dir, trace_id)?;
    let mut file = open_append(&path)?;
    file.lock()
        .with_context(|| format!("locking audit file {}", path.display()))?;
    let result = (|| {
        file.rewind()?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let mut started = None;
        let mut ended = false;
        for line in bytes.split(|b| *b == b'\n').filter(|line| !line.is_empty()) {
            let Ok(value) = serde_json::from_slice::<serde_json::Value>(line) else {
                continue;
            };
            let Ok(record) = serde_json::from_value::<Record>(value) else {
                continue;
            };
            match record {
                Record::Start(s) if s.inv == inv && started.is_none() => started = Some(s.ts),
                Record::End(e) if e.inv == inv => ended = true,
                _ => {}
            }
        }
        if ended {
            return Ok(false);
        }
        let now = Timestamp::now();
        let duration_ms = started
            .and_then(|start| now.since(start).ok())
            .and_then(|d| d.total(jiff::Unit::Millisecond).ok())
            .filter(|ms| *ms >= 0.0)
            .map(|ms| ms as u64)
            .unwrap_or(0);
        let record = Record::End(EndRecord {
            ts: now,
            inv: inv.to_string(),
            payload,
            duration_ms,
        });
        let bytes = append_bytes(&mut file, &path, &serde_json::to_string(&record)?)?;
        file.write_all(&bytes)?;
        file.flush()?;
        Ok(true)
    })();
    let unlock = file
        .unlock()
        .with_context(|| format!("unlocking audit file {}", path.display()));
    match (result, unlock) {
        (Ok(wrote), Ok(())) => Ok(wrote),
        (Err(e), _) | (_, Err(e)) => Err(e),
    }
}

/// Validate and append one callback emitted by an interactive harness.
pub fn ingest_hook(
    dir: &Path,
    kind: HookKind,
    payload: &str,
    env: &dyn EnvSource,
) -> std::result::Result<(), String> {
    let trace = env.get("CCC_TRACE_ID").ok_or("missing CCC_TRACE_ID")?;
    let inv = env
        .get("CCC_PARENT_INVOCATION_ID")
        .ok_or("missing CCC_PARENT_INVOCATION_ID")?;
    parse_uuid(&trace).map_err(|_| "malformed CCC_TRACE_ID".to_string())?;
    parse_uuid(&inv).map_err(|_| "malformed CCC_PARENT_INVOCATION_ID".to_string())?;
    let value: serde_json::Value =
        serde_json::from_str(payload).map_err(|e| format!("malformed {kind} payload: {e}"))?;
    let (harness, session, turn, result, transcript) = match kind {
        HookKind::CodexNotify => {
            if value.get("type").and_then(|v| v.as_str()) != Some("agent-turn-complete") {
                return Ok(());
            }
            let session = value
                .get("thread-id")
                .and_then(|v| v.as_str())
                .ok_or("codex payload missing thread-id")?;
            let turn = value
                .get("turn-id")
                .and_then(|v| v.as_str())
                .ok_or("codex payload missing turn-id")?;
            let result = value
                .get("last-assistant-message")
                .and_then(|v| v.as_str())
                .map(String::from);
            (
                Harness::Codex,
                session.to_string(),
                Some(turn.to_string()),
                result,
                None,
            )
        }
        HookKind::ClaudeStop => {
            let session = value
                .get("session_id")
                .and_then(|v| v.as_str())
                .ok_or("claude payload missing session_id")?;
            let result = value
                .get("last_assistant_message")
                .and_then(|v| v.as_str())
                .map(String::from);
            let transcript = value
                .get("transcript_path")
                .and_then(|v| v.as_str())
                .map(String::from);
            (
                Harness::Claude,
                session.to_string(),
                None,
                result,
                transcript,
            )
        }
    };
    append_turn(
        dir,
        &trace,
        Turn {
            ts: Timestamp::now(),
            inv,
            harness,
            session,
            turn,
            result,
            transcript,
        },
    )
    .map_err(|e| format!("appending audit turn: {e:#}"))
}

pub fn read_trace(dir: &Path, trace_id: &str) -> Result<Vec<Record>> {
    read_records(&path_for(dir, trace_id)?, true)
}

pub fn read_trace_with_diag(
    dir: &Path,
    trace_id: &str,
    diagnostic: &mut dyn Write,
) -> Result<Vec<Record>> {
    read_records_with_diag(&path_for(dir, trace_id)?, true, Some(diagnostic))
}

/// Reads a trace while treating malformed individual records as diagnostics.
/// A status snapshot can therefore retain the usable part of a concurrently
/// written or partially corrupted trace.
pub fn read_trace_tolerant(
    dir: &Path,
    trace_id: &str,
    diagnostic: &mut dyn Write,
) -> Result<Vec<Record>> {
    let path = path_for(dir, trace_id)?;
    let records = read_records_values(&path, true, Some(diagnostic))?
        .into_iter()
        .filter_map(|value| match serde_json::from_value(value) {
            Ok(record) => Some(record),
            Err(e) => {
                let _ = writeln!(
                    diagnostic,
                    "ccc: skipping malformed audit record in {}: {e}",
                    path.display()
                );
                None
            }
        })
        .collect::<Vec<_>>();
    Ok(records)
}

pub fn read_index(dir: &Path) -> Result<Vec<IndexLine>> {
    read_index_with_diag(dir, None)
}

pub fn read_index_with_diag(
    dir: &Path,
    diagnostic: Option<&mut dyn Write>,
) -> Result<Vec<IndexLine>> {
    let path = dir.join("traces.jsonl");
    let records = read_records_values(&path, false, diagnostic)?;
    let mut seen = BTreeSet::new();
    records
        .into_iter()
        .filter_map(|value| {
            if seen.insert(
                value
                    .get("trace")
                    .and_then(|x| x.as_str())
                    .unwrap_or_default()
                    .to_string(),
            ) {
                Some(serde_json::from_value(value))
            } else {
                None
            }
        })
        .collect::<std::result::Result<_, _>>()
        .with_context(|| format!("parsing index {}", path.display()))
}

fn read_records(path: &Path, trace: bool) -> Result<Vec<Record>> {
    read_records_with_diag(path, trace, None)
}

fn read_records_with_diag(
    path: &Path,
    trace: bool,
    diagnostic: Option<&mut dyn Write>,
) -> Result<Vec<Record>> {
    read_records_values(path, trace, diagnostic)?
        .into_iter()
        .map(|value| {
            serde_json::from_value(value)
                .with_context(|| format!("parsing record {}", path.display()))
        })
        .collect()
}

fn read_records_values(
    path: &Path,
    trace: bool,
    mut diagnostic: Option<&mut dyn Write>,
) -> Result<Vec<serde_json::Value>> {
    let bytes =
        std::fs::read(path).with_context(|| format!("reading audit file {}", path.display()))?;
    let mut values = Vec::new();
    let chunks = bytes.split(|b| *b == b'\n').collect::<Vec<_>>();
    for (n, line) in chunks.iter().enumerate() {
        if line.is_empty() {
            continue;
        }
        match serde_json::from_slice::<serde_json::Value>(line) {
            Ok(value) => {
                if trace
                    && value.get("t").and_then(|x| x.as_str()).is_some_and(|t| {
                        !matches!(
                            t,
                            "start" | "spawn" | "turn" | "end" | "child" | "session" | "signal"
                        )
                    })
                {
                    report_diag(
                        &mut diagnostic,
                        format_args!(
                            "skipping unknown audit record type in {} line {}",
                            path.display(),
                            n + 1
                        ),
                    );
                    continue;
                }
                values.push(value);
            }
            Err(e) if n + 1 == chunks.len() => {
                report_diag(
                    &mut diagnostic,
                    format_args!(
                        "skipping truncated audit line in {} line {}: {e}",
                        path.display(),
                        n + 1
                    ),
                );
            }
            Err(e) => report_diag(
                &mut diagnostic,
                format_args!(
                    "skipping malformed audit line in {} line {}: {e}",
                    path.display(),
                    n + 1
                ),
            ),
        }
    }
    Ok(values)
}

fn report_diag(diagnostic: &mut Option<&mut dyn Write>, message: impl std::fmt::Display) {
    if let Some(sink) = diagnostic.as_deref_mut() {
        let _ = writeln!(sink, "ccc: {message}");
    } else {
        crate::harness::diag(message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;
    use std::thread;

    fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).into(), (*v).into()))
            .collect()
    }

    fn start() -> StartPayload {
        StartPayload {
            pid: 42,
            cmd: Command::Run,
            agent: Some("scribe".into()),
            session: None,
            cwd_arg: Some("/work".into()),
            profile_arg: None,
            harness_arg: Some(Harness::Claude),
            json: false,
            prompt_source: Some(PromptSource::Arg),
            prompt_file: None,
            prompt: Some("hello".into()),
            passthrough: vec![],
            fork: false,
        }
    }

    #[test]
    fn start_payload_without_fork_parses_as_not_forked() {
        let mut value = serde_json::to_value(start()).unwrap();
        value.as_object_mut().unwrap().remove("fork");
        let text = value.to_string();
        assert!(!text.contains("fork"), "{text}");
        let parsed: StartPayload = serde_json::from_str(&text).unwrap();
        assert!(!parsed.fork);

        let mut forked = start();
        forked.fork = true;
        let back: StartPayload =
            serde_json::from_str(&serde_json::to_string(&forked).unwrap()).unwrap();
        assert!(back.fork);
    }

    #[test]
    fn lineage_covers_root_trace_only_and_child_cases() {
        let root = Lineage::from_env(&env(&[]));
        assert_eq!(root.index_policy, IndexPolicy::Append);
        assert_eq!(root.trace_id, root.invocation_id);
        assert_eq!(
            root.child_env(),
            vec![
                ("CCC_TRACE_ID".into(), root.trace_id.clone()),
                (
                    "CCC_PARENT_INVOCATION_ID".into(),
                    root.invocation_id.clone()
                )
            ]
        );

        let trace = Uuid::new_v4().to_string();
        let trace_only = Lineage::from_env(&env(&[("CCC_TRACE_ID", &trace)]));
        assert_eq!(trace_only.trace_id, trace);
        assert_eq!(trace_only.index_policy, IndexPolicy::AppendIfAbsent);
        let parent = Uuid::new_v4().to_string();
        let child = Lineage::from_env(&env(&[
            ("CCC_TRACE_ID", &trace),
            ("CCC_PARENT_INVOCATION_ID", &parent),
        ]));
        assert_eq!(child.parent.as_deref(), Some(parent.as_str()));
        assert_eq!(child.index_policy, IndexPolicy::None);
        let malformed = Lineage::from_env(&env(&[
            ("CCC_TRACE_ID", ""),
            ("CCC_PARENT_INVOCATION_ID", &parent),
        ]));
        assert_eq!(malformed.index_policy, IndexPolicy::Append);
        assert_eq!(malformed.trace_id, malformed.invocation_id);
    }

    #[test]
    fn inherited_lineage_is_side_effect_free_and_distinguishes_every_pairing() {
        let trace = Uuid::new_v4().to_string();
        let parent = Uuid::new_v4().to_string();
        assert_eq!(inherited_lineage(&env(&[])), InheritedLineage::Absent);
        assert_eq!(
            inherited_lineage(&env(&[("CCC_TRACE_ID", &trace)])),
            InheritedLineage::TraceOnly(trace.clone())
        );
        assert_eq!(
            inherited_lineage(&env(&[
                ("CCC_TRACE_ID", &trace),
                ("CCC_PARENT_INVOCATION_ID", &parent),
            ])),
            InheritedLineage::TraceParent {
                trace: trace.clone(),
                parent: parent.clone(),
            }
        );
        for pairs in [
            vec![("CCC_PARENT_INVOCATION_ID", parent.as_str())],
            vec![("CCC_TRACE_ID", "not-a-uuid")],
            vec![
                ("CCC_TRACE_ID", trace.as_str()),
                ("CCC_PARENT_INVOCATION_ID", "bad"),
            ],
        ] {
            assert_eq!(inherited_lineage(&env(&pairs)), InheritedLineage::Malformed);
        }
    }

    fn hook_env(trace: &str, parent: &str) -> BTreeMap<String, String> {
        env(&[
            ("CCC_TRACE_ID", trace),
            ("CCC_PARENT_INVOCATION_ID", parent),
        ])
    }

    #[test]
    fn codex_hook_maps_one_completed_turn_and_ignores_other_events() {
        let dir = TempDir::new("hook-codex");
        let trace = Uuid::new_v4().to_string();
        let parent = Uuid::new_v4().to_string();
        let environment = hook_env(&trace, &parent);
        let payload = r#"{"type":"agent-turn-complete","thread-id":"thread-1","turn-id":"turn-1","last-assistant-message":"done"}"#;
        ingest_hook(dir.path(), HookKind::CodexNotify, payload, &environment).unwrap();
        assert!(ingest_hook(
            dir.path(),
            HookKind::CodexNotify,
            r#"{"type":"other"}"#,
            &environment
        )
        .is_ok());
        let records = read_trace(dir.path(), &trace).unwrap();
        assert_eq!(records.len(), 1);
        match &records[0] {
            Record::Turn(turn) => {
                assert_eq!(turn.inv, parent);
                assert_eq!(turn.session, "thread-1");
                assert_eq!(turn.turn.as_deref(), Some("turn-1"));
                assert_eq!(turn.result.as_deref(), Some("done"));
            }
            _ => panic!("expected turn"),
        }
    }

    #[test]
    fn claude_hook_maps_stop_payload_and_rejects_bad_inputs_without_writing() {
        let dir = TempDir::new("hook-claude");
        let trace = Uuid::new_v4().to_string();
        let parent = Uuid::new_v4().to_string();
        let environment = hook_env(&trace, &parent);
        let payload = r#"{"session_id":"session-1","transcript_path":"/tmp/t.jsonl","last_assistant_message":"answer"}"#;
        ingest_hook(dir.path(), HookKind::ClaudeStop, payload, &environment).unwrap();
        match &read_trace(dir.path(), &trace).unwrap()[0] {
            Record::Turn(turn) => {
                assert_eq!(turn.harness, Harness::Claude);
                assert_eq!(turn.session, "session-1");
                assert_eq!(turn.transcript.as_deref(), Some("/tmp/t.jsonl"));
                assert!(turn.turn.is_none());
            }
            _ => panic!("expected turn"),
        }
        for (kind, body) in [
            (HookKind::ClaudeStop, "{"),
            (HookKind::CodexNotify, r#"{"type":"agent-turn-complete"}"#),
        ] {
            assert!(ingest_hook(dir.path(), kind, body, &environment).is_err());
        }
        ingest_hook(
            dir.path(),
            HookKind::ClaudeStop,
            r#"{"session_id":"s","last_assistant_message":null}"#,
            &environment,
        )
        .unwrap();
        assert_eq!(read_trace(dir.path(), &trace).unwrap().len(), 2);
    }

    #[test]
    fn errhandling_hook_absent_and_null_last_messages_append_null_results() {
        let dir = TempDir::new("hook-null-and-absent");
        let trace = Uuid::new_v4().to_string();
        let parent = Uuid::new_v4().to_string();
        let environment = hook_env(&trace, &parent);
        for (kind, body) in [
            (
                HookKind::CodexNotify,
                r#"{"type":"agent-turn-complete","thread-id":"thread-a","turn-id":"turn-a"}"#,
            ),
            (
                HookKind::CodexNotify,
                r#"{"type":"agent-turn-complete","thread-id":"thread-n","turn-id":"turn-n","last-assistant-message":null}"#,
            ),
            (
                HookKind::ClaudeStop,
                r#"{"session_id":"session-a","transcript_path":"/tmp/a"}"#,
            ),
            (
                HookKind::ClaudeStop,
                r#"{"session_id":"session-n","transcript_path":"/tmp/n","last_assistant_message":null}"#,
            ),
        ] {
            ingest_hook(dir.path(), kind, body, &environment).unwrap();
        }
        let turns = read_trace(dir.path(), &trace)
            .unwrap()
            .into_iter()
            .filter_map(|r| match r {
                Record::Turn(t) => Some(t),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(turns.len(), 4);
        assert!(turns.iter().all(|t| t.result.is_none()));
        assert_eq!(turns[0].session, "thread-a");
        assert_eq!(turns[0].turn.as_deref(), Some("turn-a"));
        assert_eq!(turns[1].turn.as_deref(), Some("turn-n"));
        assert_eq!(turns[2].session, "session-a");
        assert_eq!(turns[2].transcript.as_deref(), Some("/tmp/a"));
        assert_eq!(turns[3].session, "session-n");
        assert_eq!(turns[3].transcript.as_deref(), Some("/tmp/n"));
    }

    #[test]
    fn hook_requires_valid_inherited_lineage_and_reports_unwritable_targets() {
        let dir = TempDir::new("hook-invalid");
        let payload = r#"{"type":"agent-turn-complete","thread-id":"t","turn-id":"u","last-assistant-message":"m"}"#;
        for environment in [
            env(&[]),
            env(&[("CCC_TRACE_ID", "bad"), ("CCC_PARENT_INVOCATION_ID", "bad")]),
        ] {
            assert!(ingest_hook(dir.path(), HookKind::CodexNotify, payload, &environment).is_err());
        }
        let trace = Uuid::new_v4().to_string();
        let parent = Uuid::new_v4().to_string();
        let file = dir.path().join("not-a-directory");
        std::fs::write(&file, "x").unwrap();
        assert!(ingest_hook(
            &file,
            HookKind::CodexNotify,
            payload,
            &hook_env(&trace, &parent)
        )
        .is_err());
    }

    #[test]
    fn hook_kind_has_canonical_spellings_and_rejects_unknown_values() {
        assert_eq!(HookKind::CodexNotify.as_str(), "codex-notify");
        assert_eq!(HookKind::ClaudeStop.to_string(), "claude-stop");
        assert!("unknown"
            .parse::<HookKind>()
            .unwrap_err()
            .contains("unknown hook kind"));
    }

    #[test]
    fn inherited_lineage_reuses_trace_without_adding_an_index_entry() {
        let dir = TempDir::new("hook-child");
        let trace = Uuid::new_v4().to_string();
        let parent = Uuid::new_v4().to_string();
        let mut environment = hook_env(&trace, &parent);
        environment.insert("XDG_STATE_HOME".into(), dir.path().display().to_string());
        let recorder = Recorder::new(&environment).unwrap();
        recorder.start(start()).unwrap();
        assert_eq!(recorder.lineage().trace_id, trace);
        assert!(!Recorder::dir_in(&environment).join("traces.jsonl").exists());
    }

    #[test]
    fn provenance_is_structural_and_does_not_keep_unknown_values() {
        let vars = vec![
            ("CCC_CONFIG".into(), "/etc/c.toml".into()),
            ("CCC_AGENTS_CONFIG".into(), "/etc/a.toml".into()),
            ("CCC_CLAUDE_BIN".into(), "claude".into()),
            ("CCC_MODEL_ALL".into(), "opus".into()),
            ("CCC_EFFORT_COMMENT_REWRITER".into(), "high".into()),
            ("CCC_CODEX_MODEL_ALL".into(), "gpt".into()),
            ("CCC_MODEL_bad".into(), "secret".into()),
            ("CCC_SECRET".into(), "top-secret".into()),
            ("CCC_TRACE_ID".into(), "trace-secret".into()),
            ("PATH".into(), "ignored".into()),
        ];
        let (known, other) = classify_env(&vars);
        assert_eq!(known.get("CCC_CONFIG").unwrap(), "/etc/c.toml");
        assert_eq!(known.get("CCC_AGENTS_CONFIG").unwrap(), "/etc/a.toml");
        assert!(known.contains_key("CCC_CODEX_MODEL_ALL"));
        assert!(other.contains(&"CCC_MODEL_bad".into()));
        assert!(other.contains(&"CCC_SECRET".into()));
        assert!(!other.iter().any(|name| name == "CCC_TRACE_ID"));
        let json = serde_json::to_string(&IndexLine {
            trace: Uuid::new_v4().to_string(),
            ts: Timestamp::now(),
            inv: Uuid::new_v4().to_string(),
            pid: 1,
            cmd: Command::Run,
            agent: None,
            cwd: "/".into(),
            config: "/etc/c.toml".into(),
            config_exists: false,
            agents_config: "/etc/a.toml".into(),
            agents_config_exists: false,
            env: known,
            env_other: other,
        })
        .unwrap();
        assert!(!json.contains("top-secret"));
        assert!(!json.contains("secret"));
    }

    #[test]
    fn record_json_is_flat_and_round_trips() {
        let record = Record::Start(StartRecord {
            ts: Timestamp::now(),
            inv: Uuid::new_v4().to_string(),
            parent: None,
            payload: start(),
        });
        let value = serde_json::to_value(&record).unwrap();
        assert_eq!(value.get("t").unwrap(), "start");
        assert!(value.get("payload").is_none());
        assert!(value.get("pid").is_some());
        assert_eq!(
            serde_json::to_value(serde_json::from_value::<Record>(value.clone()).unwrap()).unwrap(),
            value
        );
    }

    #[test]
    fn concurrent_appends_are_complete_lines() {
        let dir = TempDir::new("audit-append");
        let path = dir.path().join("trace.jsonl");
        let mut handles = Vec::new();
        for n in 0..12 {
            let path = path.clone();
            handles.push(thread::spawn(move || {
                append_line(&path, &format!(r#"{{"n":{n}}}"#)).unwrap()
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        let lines = std::fs::read_to_string(path).unwrap();
        assert_eq!(lines.lines().count(), 12);
        for line in lines.lines() {
            serde_json::from_str::<serde_json::Value>(line).unwrap();
        }
    }

    #[test]
    fn recorder_writes_start_index_and_one_end_and_filters_turns() {
        let dir = TempDir::new("audit-recorder");
        let e = env(&[("XDG_STATE_HOME", dir.path().to_str().unwrap())]);
        let recorder = Recorder::new(&e).unwrap();
        recorder.start(start()).unwrap();
        let other = Uuid::new_v4().to_string();
        append_turn(
            &Recorder::dir_in(&e),
            &recorder.lineage().trace_id,
            Turn {
                ts: Timestamp::now(),
                inv: other,
                harness: Harness::Claude,
                session: "s".into(),
                turn: None,
                result: None,
                transcript: None,
            },
        )
        .unwrap();
        append_turn(
            &Recorder::dir_in(&e),
            &recorder.lineage().trace_id,
            Turn {
                ts: Timestamp::now(),
                inv: recorder.lineage().invocation_id.clone(),
                harness: Harness::Claude,
                session: "s".into(),
                turn: Some("t".into()),
                result: None,
                transcript: None,
            },
        )
        .unwrap();
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
        recorder.end(EndPayload {
            exit: Some(1),
            signal: None,
            session: None,
            result: None,
            error: None,
            warnings: vec![],
            native: None,
            turns: None,
        });
        let records = read_trace(&Recorder::dir_in(&e), &recorder.lineage().trace_id).unwrap();
        assert_eq!(
            records
                .iter()
                .filter(|r| matches!(r, Record::End(_)))
                .count(),
            1
        );
        assert_eq!(recorder.turns().len(), 1);
        assert_eq!(read_index(&Recorder::dir_in(&e)).unwrap().len(), 1);
    }

    #[test]
    fn readers_tolerate_a_truncated_final_line_and_reject_bad_ids() {
        let dir = TempDir::new("audit-read");
        let id = Uuid::new_v4().to_string();
        let path = dir.path().join(format!("{id}.jsonl"));
        let line = serde_json::to_string(&Record::Start(StartRecord {
            ts: Timestamp::now(),
            inv: id.clone(),
            parent: None,
            payload: start(),
        }))
        .unwrap();
        std::fs::write(&path, format!("{line}\n{{\"t\":\"future\"}}\n{{\"t\":")).unwrap();
        assert_eq!(read_trace(dir.path(), &id).unwrap().len(), 1);
        assert!(read_trace(dir.path(), "../escape").is_err());
    }

    fn index_line(trace: &str, inv: &str) -> String {
        serde_json::to_string(&IndexLine {
            trace: trace.into(),
            ts: Timestamp::now(),
            inv: inv.into(),
            pid: 1,
            cmd: Command::Run,
            agent: Some("agent".into()),
            cwd: "/work".into(),
            config: "/config".into(),
            config_exists: true,
            agents_config: "/agents".into(),
            agents_config_exists: true,
            env: BTreeMap::new(),
            env_other: vec![],
        })
        .unwrap()
    }

    #[test]
    fn delta_3_torn_tails_and_split_utf8_are_delimited_before_append() {
        let dir = TempDir::new("audit-delta3-tail");
        let path = dir.path().join("tail.jsonl");
        std::fs::write(&path, b"{\"broken\":\"x").unwrap();
        append_line(&path, r#"{"ok":1}"#).unwrap();
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"{\"broken\":\"x\n{\"ok\":1}\n"
        );

        std::fs::write(&path, b"{\"broken\":\"\xc3").unwrap();
        append_line(&path, r#"{"ok":2}"#).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[bytes.len() - 9..], b"{\"ok\":2}\n");
        assert_eq!(bytes.iter().filter(|b| **b == b'\n').count(), 2);
    }

    #[test]
    fn delta_3_readers_keep_surrounding_records_and_turns_around_damage() {
        let dir = TempDir::new("audit-delta3-read");
        let id = Uuid::new_v4().to_string();
        let path = dir.path().join(format!("{id}.jsonl"));
        let first = serde_json::to_string(&Record::Start(StartRecord {
            ts: Timestamp::now(),
            inv: id.clone(),
            parent: None,
            payload: start(),
        }))
        .unwrap();
        let turn = serde_json::to_string(&Record::Turn(Turn {
            ts: Timestamp::now(),
            inv: id.clone(),
            harness: Harness::Claude,
            session: "s".into(),
            turn: None,
            result: Some("r".into()),
            transcript: None,
        }))
        .unwrap();
        let last = serde_json::to_string(&Record::End(EndRecord {
            ts: Timestamp::now(),
            inv: id.clone(),
            payload: EndPayload {
                exit: Some(0),
                signal: None,
                session: Some("s".into()),
                result: None,
                error: None,
                warnings: vec![],
                native: None,
                turns: None,
            },
            duration_ms: 1,
        }))
        .unwrap();
        let mut damaged = format!("{first}\nnot-json\n{turn}\n").into_bytes();
        damaged.extend_from_slice(&[0xff, b'\n']);
        damaged.extend_from_slice(format!("{last}\n").as_bytes());
        std::fs::write(&path, damaged).unwrap();
        let records = read_trace(dir.path(), &id).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(
            records
                .iter()
                .filter(|r| matches!(r, Record::Turn(_)))
                .count(),
            1
        );
        let recorder = Recorder {
            lineage: Lineage {
                trace_id: id,
                invocation_id: records
                    .iter()
                    .find_map(|r| match r {
                        Record::Start(s) => Some(s.inv.clone()),
                        _ => None,
                    })
                    .unwrap(),
                parent: None,
                index_policy: IndexPolicy::None,
            },
            started_at: Timestamp::now(),
            dir: dir.path().into(),
            config: String::new(),
            config_exists: false,
            agents_config: String::new(),
            agents_config_exists: false,
            env: BTreeMap::new(),
            env_other: vec![],
            ended: Mutex::new(false),
        };
        assert_eq!(recorder.turns().len(), 1);
    }

    #[test]
    fn test_missing_append_index_if_absent() {
        let dir = TempDir::new("audit-index-lock");
        let path = dir.path().join("traces.jsonl");
        let trace = Uuid::new_v4().to_string();
        let line = index_line(&trace, "first");
        assert!(append_index_if_absent(&path, &line).unwrap());
        assert!(!append_index_if_absent(&path, &index_line(&trace, "second")).unwrap());
        assert_eq!(read_index(dir.path()).unwrap()[0].inv, "first");

        let path2 = dir.path().join("concurrent.jsonl");
        let shared_trace = Uuid::new_v4().to_string();
        let handles = (0..8)
            .map(|n| {
                let path = path2.clone();
                let line = index_line(&shared_trace, &n.to_string());
                std::thread::spawn(move || append_index_if_absent(&path, &line).unwrap())
            })
            .collect::<Vec<_>>();
        let results = handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(results.iter().filter(|x| **x).count(), 1);
        assert_eq!(results.iter().filter(|x| !**x).count(), 7);
        let lines = std::fs::read_to_string(&path2).unwrap();
        assert_eq!(lines.lines().count(), 1);
        assert!(lines.ends_with('\n'));
        let stored: IndexLine = serde_json::from_str(lines.trim_end()).unwrap();
        assert_eq!(stored.trace, shared_trace);
        assert!((0..8).map(|n| n.to_string()).any(|n| stored.inv == n));

        let path3 = dir.path().join("damaged.jsonl");
        std::fs::write(&path3, b"garbage\n").unwrap();
        assert!(append_index_if_absent(&path3, &index_line(&trace, "later")).unwrap());
        assert!(std::fs::read_to_string(path3).unwrap().contains("later"));

        let absent_dir = TempDir::new("audit-index-recorder-absent");
        let trace = Uuid::new_v4().to_string();
        let mut absent_env = env(&[("XDG_STATE_HOME", absent_dir.path().to_str().unwrap())]);
        absent_env.insert("CCC_TRACE_ID".into(), trace.clone());
        let recorder = Recorder::new(&absent_env).unwrap();
        recorder.start(start()).unwrap();
        let index = read_index(&Recorder::dir_in(&absent_env)).unwrap();
        assert_eq!(index.len(), 1);
        assert_eq!(index[0].trace, trace);
        assert_eq!(index[0].inv, recorder.lineage().invocation_id);

        let present_dir = TempDir::new("audit-index-recorder-present");
        let trace = Uuid::new_v4().to_string();
        let mut present_env = env(&[("XDG_STATE_HOME", present_dir.path().to_str().unwrap())]);
        present_env.insert("CCC_TRACE_ID".into(), trace.clone());
        let dir = Recorder::dir_in(&present_env);
        std::fs::create_dir_all(&dir).unwrap();
        let index_path = dir.join("traces.jsonl");
        std::fs::write(&index_path, format!("{}\n", index_line(&trace, "existing"))).unwrap();
        let before = std::fs::read(&index_path).unwrap();
        let recorder = Recorder::new(&present_env).unwrap();
        recorder.start(start()).unwrap();
        assert_eq!(std::fs::read(&index_path).unwrap(), before);
        assert_eq!(read_trace(&dir, &trace).unwrap().len(), 1);
    }

    #[test]
    fn test_missing_start_write_order_failure() {
        let dir = TempDir::new("audit-shapes");
        let environment = env(&[("XDG_STATE_HOME", dir.path().to_str().unwrap())]);
        let recorder = Recorder::new(&environment).unwrap();
        let index = Recorder::dir_in(&environment).join("traces.jsonl");
        std::fs::create_dir(&index).unwrap();
        let err = recorder.start(start()).unwrap_err().to_string();
        assert!(err.contains("traces.jsonl") && err.contains(".jsonl"));
        let trace =
            Recorder::dir_in(&environment).join(format!("{}.jsonl", recorder.lineage().trace_id));
        assert_eq!(
            read_trace(
                &Recorder::dir_in(&environment),
                &recorder.lineage().trace_id
            )
            .unwrap()
            .len(),
            1
        );
        std::fs::remove_dir(&index).unwrap();

        let dir2 = TempDir::new("audit-trace-dir");
        let env2 = env(&[("XDG_STATE_HOME", dir2.path().to_str().unwrap())]);
        let recorder2 = Recorder::new(&env2).unwrap();
        std::fs::create_dir(
            Recorder::dir_in(&env2).join(format!("{}.jsonl", recorder2.lineage().trace_id)),
        )
        .unwrap();
        assert!(recorder2
            .start(start())
            .unwrap_err()
            .to_string()
            .contains(".jsonl"));
        assert!(!Recorder::dir_in(&env2).join("traces.jsonl").exists());

        let backup = trace.with_extension("jsonl.backup");
        std::fs::rename(&trace, &backup).unwrap();
        std::fs::create_dir(&trace).unwrap();
        recorder.end(EndPayload {
            exit: Some(0),
            signal: None,
            session: Some("s".into()),
            result: Some("r".into()),
            error: None,
            warnings: vec![],
            native: None,
            turns: None,
        });
        std::fs::remove_dir(&trace).unwrap();
        std::fs::rename(&backup, &trace).unwrap();
        recorder.end(EndPayload {
            exit: Some(0),
            signal: None,
            session: Some("s".into()),
            result: Some("r".into()),
            error: None,
            warnings: vec![],
            native: None,
            turns: None,
        });
        assert_eq!(
            read_trace(
                &Recorder::dir_in(&environment),
                &recorder.lineage().trace_id
            )
            .unwrap()
            .len(),
            1
        );
    }

    #[test]
    fn test_missing_read_index_duplicate_tolerance() {
        let dir = TempDir::new("audit-index-duplicates");
        let a = Uuid::new_v4().to_string();
        let b = Uuid::new_v4().to_string();
        std::fs::write(
            dir.path().join("traces.jsonl"),
            format!(
                "{}\n{}\n{}\n",
                index_line(&a, "a1"),
                index_line(&b, "b"),
                index_line(&a, "a2")
            ),
        )
        .unwrap();
        let entries = read_index(dir.path()).unwrap();
        assert_eq!(
            entries.iter().map(|x| x.trace.clone()).collect::<Vec<_>>(),
            vec![a, b]
        );
        assert_eq!(entries[0].inv, "a1");
    }

    #[cfg(unix)]
    #[test]
    fn process_env_non_utf8_values_are_ignored_without_panicking() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;
        static LOCK: std::sync::OnceLock<Mutex<()>> = std::sync::OnceLock::new();
        let _guard = LOCK.get_or_init(|| Mutex::new(())).lock().unwrap();
        unsafe {
            std::env::set_var(
                OsString::from_vec(b"NOT_CCC\xff".to_vec()),
                OsString::from_vec(b"x\xff".to_vec()),
            );
            std::env::set_var("CCC_BAD_UTF8", OsString::from_vec(b"x\xff".to_vec()));
        }
        let vars = crate::config::ProcessEnv.vars_with_prefix("");
        assert!(!vars.iter().any(|(k, _)| k == "CCC_BAD_UTF8"));
        unsafe {
            std::env::remove_var(OsString::from_vec(b"NOT_CCC\xff".to_vec()));
            std::env::remove_var("CCC_BAD_UTF8");
        }
    }

    #[test]
    fn test_vacuous_launch_duration_assert() {
        let dir = TempDir::new("audit-duration");
        let e = env(&[("XDG_STATE_HOME", dir.path().to_str().unwrap())]);
        let r = Recorder::new(&e).unwrap();
        r.start(start()).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(3));
        r.end(EndPayload {
            exit: Some(0),
            signal: None,
            session: None,
            result: None,
            error: None,
            warnings: vec![],
            native: None,
            turns: None,
        });
        let end = read_trace(&Recorder::dir_in(&e), &r.lineage().trace_id)
            .unwrap()
            .into_iter()
            .find_map(|x| match x {
                Record::End(e) => Some(e),
                _ => None,
            })
            .unwrap();
        assert!(end.duration_ms > 0);
    }

    #[test]
    fn process_records_have_flat_wire_shapes_and_external_end_is_atomic() {
        let id = ProcId {
            pid: 7,
            start: 8,
            boot_id: "boot".into(),
        };
        let child = Record::Child(ChildRecord {
            ts: Timestamp::UNIX_EPOCH,
            inv: Uuid::new_v4().to_string(),
            payload: ChildPayload {
                proc: id.clone(),
                pgid: 7,
                supervisor: id.clone(),
            },
        });
        let signal = Record::Signal(SignalRecord {
            ts: Timestamp::UNIX_EPOCH,
            inv: Uuid::new_v4().to_string(),
            payload: SignalPayload {
                signal: 15,
                to: SignalDestination::Descendants,
                by: id,
                by_inv: None,
                reason: "timeout".into(),
            },
        });
        let session = Record::Session(SessionRecord {
            ts: Timestamp::UNIX_EPOCH,
            inv: Uuid::new_v4().to_string(),
            session: "s".into(),
        });
        assert_eq!(serde_json::to_value(&child).unwrap()["t"], "child");
        assert_eq!(serde_json::to_value(&session).unwrap()["t"], "session");
        assert_eq!(serde_json::to_value(&signal).unwrap()["to"], "descendants");
        for record in [child, session, signal] {
            let value = serde_json::to_value(&record).unwrap();
            assert_eq!(
                serde_json::to_value(serde_json::from_value::<Record>(value.clone()).unwrap())
                    .unwrap(),
                value
            );
        }

        let dir = TempDir::new("audit-external-end");
        let e = env(&[("XDG_STATE_HOME", dir.path().to_str().unwrap())]);
        let recorder = Recorder::new(&e).unwrap();
        recorder.start(start()).unwrap();
        let trace = recorder.lineage().trace_id.clone();
        let inv = recorder.lineage().invocation_id.clone();
        let payload = EndPayload {
            exit: Some(0),
            signal: None,
            session: None,
            result: None,
            error: None,
            warnings: vec![],
            native: None,
            turns: None,
        };
        let dir = Recorder::dir_in(&e);
        let a = {
            let dir = dir.clone();
            let trace = trace.clone();
            let inv = inv.clone();
            let payload = payload.clone();
            std::thread::spawn(move || append_end_for(&dir, &trace, &inv, payload).unwrap())
        };
        let b = {
            let dir = dir.clone();
            let trace = trace.clone();
            let inv = inv.clone();
            std::thread::spawn(move || append_end_for(&dir, &trace, &inv, payload).unwrap())
        };
        assert_ne!(a.join().unwrap(), b.join().unwrap());
        assert_eq!(
            read_trace(&dir, &trace)
                .unwrap()
                .iter()
                .filter(|r| matches!(r, Record::End(_)))
                .count(),
            1
        );
    }

    #[test]
    fn recorder_and_external_signal_writers_use_the_requested_invocation() {
        let dir = TempDir::new("audit-process-writers");
        let e = env(&[("XDG_STATE_HOME", dir.path().to_str().unwrap())]);
        let recorder = Recorder::new(&e).unwrap();
        recorder.start(start()).unwrap();
        let trace = recorder.lineage().trace_id.clone();
        let own_inv = recorder.lineage().invocation_id.clone();
        let other_inv = Uuid::new_v4().to_string();
        let identity = ProcId {
            pid: 7,
            start: 8,
            boot_id: "boot".into(),
        };
        recorder
            .child(ChildPayload {
                proc: identity.clone(),
                pgid: 7,
                supervisor: identity.clone(),
            })
            .unwrap();
        recorder.session("own-session").unwrap();
        recorder
            .signal(SignalPayload {
                signal: 15,
                to: SignalDestination::Child,
                by: identity.clone(),
                by_inv: Some(own_inv.clone()),
                reason: "own".into(),
            })
            .unwrap();
        append_signal_for(
            &Recorder::dir_in(&e),
            &trace,
            &other_inv,
            SignalPayload {
                signal: 9,
                to: SignalDestination::Descendants,
                by: identity,
                by_inv: None,
                reason: "external".into(),
            },
        )
        .unwrap();
        let records = read_trace(&Recorder::dir_in(&e), &trace).unwrap();
        assert!(records
            .iter()
            .any(|record| matches!(record, Record::Child(child) if child.inv == own_inv)));
        assert!(records
            .iter()
            .any(|record| matches!(record, Record::Session(session) if session.inv == own_inv)));
        assert!(records.iter().any(|record| matches!(record, Record::Signal(signal) if signal.inv == own_inv && signal.payload.signal == 15)));
        assert!(records.iter().any(|record| matches!(record, Record::Signal(signal) if signal.inv == other_inv && signal.payload.signal == 9)));
    }

    #[test]
    fn external_end_uses_the_target_start_and_rejects_a_repeat() {
        let dir = TempDir::new("audit-target-end");
        let trace = Uuid::new_v4().to_string();
        let target = Uuid::new_v4().to_string();
        let other = Uuid::new_v4().to_string();
        let path = dir.path().join(format!("{trace}.jsonl"));
        append_record(
            &path,
            &Record::Start(StartRecord {
                ts: Timestamp::now(),
                inv: other,
                parent: None,
                payload: start(),
            }),
        )
        .unwrap();
        append_record(
            &path,
            &Record::Start(StartRecord {
                ts: Timestamp::UNIX_EPOCH,
                inv: target.clone(),
                parent: None,
                payload: start(),
            }),
        )
        .unwrap();
        let payload = EndPayload {
            exit: Some(0),
            signal: None,
            session: None,
            result: None,
            error: None,
            warnings: vec![],
            native: None,
            turns: None,
        };
        assert!(append_end_for(dir.path(), &trace, &target, payload.clone()).unwrap());
        assert!(!append_end_for(dir.path(), &trace, &target, payload).unwrap());
        let end = read_trace(dir.path(), &trace)
            .unwrap()
            .into_iter()
            .find_map(|record| match record {
                Record::End(end) if end.inv == target => Some(end),
                _ => None,
            })
            .unwrap();
        assert!(end.duration_ms > 1_000);
    }
}
