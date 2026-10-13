//! Read-only audit status: authority, process observations, and projections.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Result;
use jiff::{tz::TimeZone, SignedDuration, Timestamp};
use serde::Serialize;

use crate::audit::{self, InheritedLineage, Record, SignalDestination};
use crate::audit_view::{self, Entry, InvocationJson};
use crate::config::{self, ClaudeAuth, Config, EnvSource, Harness};
use crate::procs::{self, Liveness, Presence, ProcId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    Subtree,
    Trace,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeSource {
    Ancestry,
    Env,
    None,
}

#[derive(Debug, Clone)]
struct Scope {
    kind: ScopeKind,
    source: ScopeSource,
    trace: Option<String>,
    root: Option<String>,
}

#[derive(Debug, Clone)]
struct TraceData {
    entry: Entry,
    invocations: Vec<InvocationJson>,
}

#[derive(Debug, Clone, Serialize)]
struct ProcessObservation {
    pid: u32,
    identity: Option<ProcId>,
    alive: Option<bool>,
    verified: bool,
}

#[derive(Debug, Clone)]
struct Row {
    trace: String,
    invocation: InvocationJson,
}

struct IdleQuery<'a> {
    cfg: &'a Config,
    env: &'a dyn EnvSource,
    harness: Harness,
    profile: Option<&'a str>,
    auth: Option<ClaudeAuth>,
    session: Option<&'a str>,
    started: Option<Timestamp>,
    now: Timestamp,
}

pub struct StatusContext<'a> {
    pub dir: &'a Path,
    pub cfg: &'a Config,
    pub env: &'a dyn EnvSource,
    pub now: Timestamp,
    pub width: usize,
    pub trace: Option<String>,
    pub json: bool,
}

pub trait ProcessOps {
    fn current(&self) -> Result<ProcId>;
    fn alive(&self, pid: &ProcId) -> Liveness;
    fn descendants(&self, pid: u32) -> Vec<ProcId>;
    fn signal(&self, pid: u32, signal: i32) -> std::io::Result<()>;
    fn signal_group(&self, pgid: u32, signal: i32) -> std::io::Result<()>;
    fn sleep(&self, duration: std::time::Duration);
    fn now(&self) -> std::time::Instant;
}

pub struct SystemProcessOps;

impl ProcessOps for SystemProcessOps {
    fn current(&self) -> Result<ProcId> {
        ProcId::current()
    }
    fn alive(&self, pid: &ProcId) -> Liveness {
        pid.alive()
    }
    fn descendants(&self, pid: u32) -> Vec<ProcId> {
        procs::descendants(pid)
    }
    fn signal(&self, pid: u32, signal: i32) -> std::io::Result<()> {
        procs::signal(pid, signal)
    }
    fn signal_group(&self, pgid: u32, signal: i32) -> std::io::Result<()> {
        procs::signal_group(pgid, signal)
    }
    fn sleep(&self, duration: std::time::Duration) {
        std::thread::sleep(duration)
    }
    fn now(&self) -> std::time::Instant {
        std::time::Instant::now()
    }
}

pub struct KillContext<'a> {
    pub dir: &'a Path,
    pub cfg: &'a Config,
    pub env: &'a dyn EnvSource,
    pub now: Timestamp,
    pub trace: Option<String>,
    pub targets: Vec<String>,
    pub all: bool,
    pub immediate: bool,
    pub grace: Option<f64>,
    pub no_wait: bool,
    pub json: bool,
    pub ops: &'a dyn ProcessOps,
    pub scope_root: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct StatusRow {
    inv: String,
    parent: Option<String>,
    trace: String,
    cmd: String,
    agent: String,
    harness: Option<String>,
    model: Option<String>,
    session: Option<String>,
    state: String,
    exit: Option<i32>,
    ended: Option<String>,
    duration_s: Option<f64>,
    error: Option<String>,
    supervisor: Option<ProcessObservation>,
    harness_proc: Option<ProcessObservation>,
    idle_s: Option<u64>,
    started: Option<String>,
    signals: Vec<Record>,
    #[serde(skip)]
    terminal_signal: Option<i32>,
}

fn scope_names(scope: &Scope) -> (&'static str, &'static str) {
    (
        match scope.kind {
            ScopeKind::Subtree => "subtree",
            ScopeKind::Trace => "trace",
            ScopeKind::All => "all",
        },
        match scope.source {
            ScopeSource::Ancestry => "ancestry",
            ScopeSource::Env => "env",
            ScopeSource::None => "none",
        },
    )
}

fn parse_scope(
    entries: &[TraceData],
    env: &dyn EnvSource,
    selector: Option<&str>,
    err: &mut dyn Write,
) -> Result<std::result::Result<Scope, ()>> {
    parse_scope_from_ancestors_named(entries, env, selector, &procs::ancestors(), "status", err)
}

#[cfg(test)]
fn parse_scope_from_ancestors(
    entries: &[TraceData],
    env: &dyn EnvSource,
    selector: Option<&str>,
    ancestors: &[ProcId],
    err: &mut dyn Write,
) -> Result<std::result::Result<Scope, ()>> {
    parse_scope_from_ancestors_named(entries, env, selector, ancestors, "status", err)
}

fn parse_scope_from_ancestors_named(
    entries: &[TraceData],
    env: &dyn EnvSource,
    selector: Option<&str>,
    ancestors: &[ProcId],
    command: &str,
    err: &mut dyn Write,
) -> Result<std::result::Result<Scope, ()>> {
    let mut identities: HashMap<ProcId, (String, String)> = HashMap::new();
    for trace in entries {
        for inv in &trace.invocations {
            if let Some(Record::Child(c)) = &inv.child {
                identities.insert(
                    c.payload.proc.clone(),
                    (trace.entry.id.clone(), inv.inv.clone()),
                );
                identities.insert(
                    c.payload.supervisor.clone(),
                    (trace.entry.id.clone(), inv.inv.clone()),
                );
            }
        }
    }
    let anchor = ancestors.iter().find_map(|p| identities.get(p).cloned());
    let inherited = audit::inherited_lineage(env);
    let inherited_kind = match &inherited {
        InheritedLineage::Absent => "absent",
        InheritedLineage::Malformed => "malformed",
        _ => "inconsistent",
    };
    let (mut scope, recovery) = if let Some((trace, root)) = anchor {
        let valid = matches!(&inherited, InheritedLineage::TraceParent { trace: t, parent: p } if t == &trace && p == &root);
        (
            Scope {
                kind: ScopeKind::Subtree,
                source: ScopeSource::Ancestry,
                trace: Some(trace),
                root: Some(root),
            },
            !valid,
        )
    } else {
        match inherited {
            InheritedLineage::TraceParent { trace, parent } => (
                Scope {
                    kind: ScopeKind::Subtree,
                    source: ScopeSource::Env,
                    trace: Some(trace),
                    root: Some(parent),
                },
                false,
            ),
            InheritedLineage::TraceOnly(trace) => (
                Scope {
                    kind: ScopeKind::Trace,
                    source: ScopeSource::Env,
                    trace: Some(trace),
                    root: None,
                },
                false,
            ),
            InheritedLineage::Absent | InheritedLineage::Malformed => (
                Scope {
                    kind: ScopeKind::All,
                    source: ScopeSource::None,
                    trace: None,
                    root: None,
                },
                false,
            ),
        }
    };
    if recovery {
        writeln!(err, "ccc: lineage recovered from process ancestry (invocation {}); inherited CCC_* variables were {inherited_kind}", scope.root.as_deref().unwrap_or("-"))?;
    }
    if let Some(prefix) = selector {
        if scope.kind == ScopeKind::Subtree {
            let detail = if command == "kill" {
                "ccc kill operates only on your own subagents"
            } else {
                "ccc status shows your own subagents"
            };
            writeln!(
                err,
                "ccc: --trace is not available to a spawned agent; {detail}"
            )?;
            return Ok(Err(()));
        }
        let matches = entries
            .iter()
            .filter(|t| t.entry.id.starts_with(prefix))
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            if matches.is_empty() {
                writeln!(err, "ccc: no trace matches {prefix}")?;
            } else {
                writeln!(err, "ccc: ambiguous trace selector {prefix}")?;
                for t in matches {
                    writeln!(err, "ccc: {}", t.entry.id)?;
                }
            }
            return Ok(Err(()));
        }
        scope = Scope {
            kind: ScopeKind::Trace,
            source: ScopeSource::None,
            trace: Some(matches[0].entry.id.clone()),
            root: None,
        };
    }
    Ok(Ok(scope))
}

fn load_traces(dir: &Path, err: &mut dyn Write) -> Result<Vec<TraceData>> {
    let entries = audit_view::load_catalog(dir, err)?;
    Ok(entries
        .into_iter()
        .map(|entry| {
            let invocations = if entry.exists {
                audit::read_trace_tolerant(dir, &entry.id, err)
                    .map(|r| audit_view::join_invocations(&r).0)
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            TraceData { entry, invocations }
        })
        .collect())
}

fn selected(entries: &[TraceData], scope: &Scope) -> Vec<Row> {
    let mut out = Vec::new();
    for trace in entries {
        if scope.trace.as_ref().is_some_and(|id| id != &trace.entry.id) {
            continue;
        }
        let by_id: HashMap<String, &InvocationJson> = trace
            .invocations
            .iter()
            .map(|i| (i.inv.clone(), i))
            .collect();
        let ids: HashSet<String> = match scope.kind {
            ScopeKind::Trace => trace.invocations.iter().map(|i| i.inv.clone()).collect(),
            ScopeKind::Subtree => {
                let root = scope.root.as_deref().unwrap_or("");
                trace
                    .invocations
                    .iter()
                    .filter(|i| {
                        let mut p = i.parent.as_deref();
                        let mut seen = HashSet::new();
                        while let Some(id) = p {
                            if id == root {
                                return true;
                            }
                            if !seen.insert(id) {
                                break;
                            }
                            p = by_id.get(id).and_then(|x| x.parent.as_deref());
                        }
                        false
                    })
                    .map(|i| i.inv.clone())
                    .collect()
            }
            ScopeKind::All => {
                let mut keep = HashSet::new();
                for i in &trace.invocations {
                    if i.end.is_none() {
                        keep.insert(i.inv.clone());
                        let mut p = i.parent.as_deref();
                        let mut seen = HashSet::new();
                        while let Some(id) = p {
                            if !seen.insert(id) {
                                break;
                            }
                            keep.insert(id.to_string());
                            p = by_id.get(id).and_then(|x| x.parent.as_deref());
                        }
                    }
                }
                keep
            }
        };
        for i in &trace.invocations {
            if ids.contains(&i.inv) {
                out.push(Row {
                    trace: trace.entry.id.clone(),
                    invocation: i.clone(),
                });
            }
        }
    }
    out
}

fn observed(pid: &ProcId) -> ProcessObservation {
    ProcessObservation {
        pid: pid.pid,
        identity: Some(pid.clone()),
        alive: match pid.alive() {
            Liveness::Alive => Some(true),
            Liveness::Dead => Some(false),
            Liveness::Unknown(_) => None,
        },
        verified: true,
    }
}

fn legacy(pid: u32) -> ProcessObservation {
    ProcessObservation {
        pid,
        identity: None,
        alive: match procs::presence(pid) {
            Presence::Present => Some(true),
            Presence::Absent => Some(false),
            Presence::Unknown(_) => None,
        },
        verified: false,
    }
}

struct IdleIndex {
    claude: HashMap<PathBuf, HashMap<String, Timestamp>>,
    codex: HashMap<PathBuf, HashMap<String, Timestamp>>,
    reads: HashMap<PathBuf, usize>,
}

impl IdleIndex {
    fn new(_cfg: &Config, _env: &dyn EnvSource) -> Self {
        Self {
            claude: HashMap::new(),
            codex: HashMap::new(),
            reads: HashMap::new(),
        }
    }

    fn scan_claude(&mut self, root: &Path) -> &HashMap<String, Timestamp> {
        if !self.claude.contains_key(root) {
            *self.reads.entry(root.to_path_buf()).or_default() += 1;
            let found = {
                let mut found = HashMap::new();
                if let Ok(items) = fs::read_dir(root.join("projects")) {
                    for item in items.flatten() {
                        if !item.file_type().is_ok_and(|t| t.is_dir()) {
                            continue;
                        }
                        let Ok(files) = fs::read_dir(item.path()) else {
                            continue;
                        };
                        for file in files.flatten() {
                            let path = file.path();
                            if path.extension().and_then(|x| x.to_str()) != Some("jsonl") {
                                continue;
                            }
                            let Some(session) = path.file_stem().and_then(|x| x.to_str()) else {
                                continue;
                            };
                            let Ok(modified) = file.metadata().and_then(|m| m.modified()) else {
                                continue;
                            };
                            let Ok(ts) = Timestamp::try_from(modified) else {
                                continue;
                            };
                            found
                                .entry(session.to_string())
                                .and_modify(|old| {
                                    if ts > *old {
                                        *old = ts;
                                    }
                                })
                                .or_insert(ts);
                        }
                    }
                }
                found
            };
            self.claude.insert(root.to_path_buf(), found);
        }
        self.claude.get(root).unwrap()
    }

    fn scan_codex(&mut self, dir: &Path) -> &HashMap<String, Timestamp> {
        if !self.codex.contains_key(dir) {
            *self.reads.entry(dir.to_path_buf()).or_default() += 1;
            let found = {
                let mut found = HashMap::new();
                if let Ok(items) = fs::read_dir(dir) {
                    for item in items.flatten() {
                        let name = item.file_name().to_string_lossy().to_string();
                        let Some(rest) = name.strip_prefix("rollout-") else {
                            continue;
                        };
                        let Some(session) = rest
                            .strip_suffix(".jsonl")
                            .or_else(|| rest.strip_suffix(".jsonl.zst"))
                        else {
                            continue;
                        };
                        let Some(session) = session.get(20..) else {
                            continue;
                        };
                        let Ok(modified) = item.metadata().and_then(|m| m.modified()) else {
                            continue;
                        };
                        let Ok(ts) = Timestamp::try_from(modified) else {
                            continue;
                        };
                        found
                            .entry(session.to_string())
                            .and_modify(|old| {
                                if ts > *old {
                                    *old = ts;
                                }
                            })
                            .or_insert(ts);
                    }
                }
                found
            };
            self.codex.insert(dir.to_path_buf(), found);
        }
        self.codex.get(dir).unwrap()
    }

    #[cfg(test)]
    fn reads(&self, path: &Path) -> usize {
        self.reads.get(path).copied().unwrap_or(0)
    }

    fn idle_for(&mut self, query: IdleQuery<'_>) -> Option<u64> {
        let IdleQuery {
            cfg,
            env,
            harness,
            profile,
            auth,
            session,
            started,
            now,
        } = query;
        let session = session?;
        let mut paths = Vec::new();
        match harness {
            Harness::Claude => {
                let home = config::home_dir_in(env);
                let first = match (auth, profile) {
                    (Some(ClaudeAuth::ConfigDir), Some(profile)) => {
                        crate::harness::claude::Account::from_config(&cfg.claude, profile, env)
                            .config_dir()
                    }
                    _ if auth == Some(ClaudeAuth::Token) => home.join(".claude"),
                    _ => env
                        .get("CLAUDE_CONFIG_DIR")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| home.join(".claude")),
                };
                paths.push(first.clone());
                if first != home.join(".claude") {
                    paths.push(home.join(".claude"));
                }
                for root in paths {
                    if let Some(t) = self.scan_claude(&root).get(session) {
                        return Some(age(now, *t));
                    }
                }
            }
            Harness::Codex => {
                let root = env
                    .get("CODEX_HOME")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| config::home_dir_in(env).join(".codex"));
                let start = started?;
                for delta in [-1i64, 0, 1] {
                    let date = start
                        .checked_add(SignedDuration::from_secs(delta * 86400))
                        .ok()?
                        .to_zoned(TimeZone::system());
                    let dir = root
                        .join("sessions")
                        .join(date.strftime("%Y/%m/%d").to_string());
                    if let Some(t) = self.scan_codex(&dir).get(session) {
                        return Some(age(now, *t));
                    }
                }
            }
        }
        None
    }
}

fn age(now: Timestamp, then: Timestamp) -> u64 {
    now.since(then)
        .ok()
        .and_then(|d| d.total(jiff::Unit::Second).ok())
        .unwrap_or(0.0)
        .max(0.0) as u64
}

fn row_status(
    cfg: &Config,
    env: &dyn EnvSource,
    row: &Row,
    now: Timestamp,
    idle: &mut IdleIndex,
) -> StatusRow {
    let i = &row.invocation;
    let start = audit_view::start_of(i);
    let spawn = audit_view::spawn_of(i);
    let end = audit_view::end_of(i);
    let started = start.map(|s| s.ts);
    let ended = end.map(|e| e.ts);
    let supervisor = if end.is_some() {
        None
    } else {
        match &i.child {
            Some(Record::Child(c)) => Some(observed(&c.payload.supervisor)),
            _ => start.map(|s| legacy(s.payload.pid)),
        }
    };
    let harness_proc = if end.is_some() {
        None
    } else {
        i.child.as_ref().and_then(|r| match r {
            Record::Child(c) => Some(observed(&c.payload.proc)),
            _ => None,
        })
    };
    let cmd = start
        .map(|s| {
            match s.payload.cmd {
                audit::Command::Run => "run",
                audit::Command::Resume => "resume",
                audit::Command::Launch => "launch",
            }
            .to_string()
        })
        .unwrap_or_else(|| "incomplete".into());
    let fact = |p: &ProcessObservation| match p.alive {
        Some(true) => audit_view::ProcessFact::Alive,
        Some(false) => audit_view::ProcessFact::Dead,
        None => audit_view::ProcessFact::Unknown,
    };
    let state = audit_view::invocation_state_from_facts(
        i,
        supervisor.as_ref().map(fact),
        harness_proc.as_ref().map(fact),
    )
    .to_string();
    StatusRow {
        inv: i.inv.clone(),
        parent: i.parent.clone(),
        trace: row.trace.clone(),
        cmd,
        agent: start
            .and_then(|s| s.payload.agent.clone())
            .or_else(|| spawn.and_then(|s| s.payload.agent.clone()))
            .unwrap_or_else(|| "-".into()),
        harness: spawn.map(|s| s.payload.harness.to_string()),
        model: spawn.map(|s| s.payload.model.clone()),
        session: audit_view::effective_session(i).map(str::to_string),
        state,
        exit: end.and_then(|e| e.payload.exit),
        ended: ended.map(|t| t.to_string()),
        duration_s: end.map(|e| e.duration_ms as f64 / 1000.0),
        error: end.and_then(|e| e.payload.error.clone()),
        supervisor,
        harness_proc,
        idle_s: if end.is_some() {
            None
        } else {
            idle.idle_for(IdleQuery {
                cfg,
                env,
                harness: spawn.map(|s| s.payload.harness).unwrap_or(Harness::Claude),
                profile: spawn.and_then(|s| s.payload.profile.as_deref()),
                auth: spawn.and_then(|s| s.payload.auth),
                session: audit_view::effective_session(i),
                started,
                now,
            })
        },
        started: started.map(|t| t.to_string()),
        signals: i.signals.clone(),
        terminal_signal: end.and_then(|e| e.payload.signal),
    }
}

pub fn status_with(
    ctx: StatusContext<'_>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<i32> {
    let StatusContext {
        dir,
        cfg,
        env,
        now,
        width,
        trace,
        json,
    } = ctx;
    let catalog = load_traces(dir, err)?;
    let scope = match parse_scope(&catalog, env, trace.as_deref(), err)? {
        Ok(s) => s,
        Err(()) => return Ok(2),
    };
    let rows = selected(&catalog, &scope);
    let mut idle = IdleIndex::new(cfg, env);
    let projected = rows
        .iter()
        .map(|r| row_status(cfg, env, r, now, &mut idle))
        .collect::<Vec<_>>();
    if json {
        let (kind, source) = scope_names(&scope);
        serde_json::to_writer(
            &mut *out,
            &serde_json::json!({ "scope": { "kind": kind, "source": source, "trace": scope.trace, "root": scope.root }, "invocations": projected }),
        )?;
        writeln!(out)?;
    } else {
        for row in projected {
            let routing = match (&row.harness, &row.model) {
                (Some(h), Some(m)) => format!(" {h} {m}"),
                (Some(h), None) => format!(" {h}"),
                _ => String::new(),
            };
            let tail = if row.ended.is_some() {
                let outcome = row
                    .exit
                    .map(|x| format!("exit {x}"))
                    .or_else(|| row.terminal_signal.map(|x| format!("signal {x}")))
                    .or_else(|| {
                        row.signals.last().and_then(|s| match s {
                            Record::Signal(x) => Some(format!("signal {}", x.payload.signal)),
                            _ => None,
                        })
                    })
                    .unwrap_or_else(|| row.state.clone());
                format!(
                    " {} {}{}",
                    outcome,
                    row.duration_s
                        .map(|x| format!("{x:.0}s"))
                        .unwrap_or_else(|| "-".into()),
                    row.error
                        .as_deref()
                        .map(|x| format!(" ‹{}›", audit_view::sanitize(x)))
                        .unwrap_or_default()
                )
            } else {
                format!(
                    " supervisor {} harness {} idle {}",
                    marker(row.supervisor.as_ref()),
                    marker(row.harness_proc.as_ref()),
                    row.idle_s
                        .map(|x| format!("{x}s"))
                        .unwrap_or_else(|| "-".into())
                )
            };
            let line = format!(
                "└─ {} {} {} {} {}{} session={}{}",
                audit_view::short(&row.inv, false),
                row.cmd,
                audit_view::sanitize(&row.agent),
                row.started.as_deref().unwrap_or("-"),
                row.state,
                routing,
                audit_view::sanitize(row.session.as_deref().unwrap_or("-")),
                tail
            );
            writeln!(out, "{}", audit_view::clip(&line, width))?;
        }
    }
    Ok(0)
}

fn marker(p: Option<&ProcessObservation>) -> String {
    p.map(|p| {
        format!(
            "{} {}",
            p.pid,
            if !p.verified {
                "?"
            } else if p.alive == Some(true) {
                "✓"
            } else if p.alive == Some(false) {
                "dead"
            } else {
                "?"
            }
        )
    })
    .unwrap_or_else(|| "-".into())
}

#[derive(Debug, Serialize)]
struct KillResult {
    inv: String,
    trace: String,
    session: Option<String>,
    state: String,
    swept: usize,
    error: Option<String>,
}

fn invocation_matches(rows: &[Row], prefix: &str) -> Vec<Row> {
    rows.iter()
        .filter(|r| r.invocation.inv.starts_with(prefix))
        .cloned()
        .collect()
}

fn depth(row: &Row, by_id: &HashMap<String, &InvocationJson>) -> usize {
    let mut current = row.invocation.parent.as_deref();
    let mut seen = HashSet::new();
    let mut result = 0;
    while let Some(id) = current {
        if !seen.insert(id) {
            break;
        }
        result += 1;
        current = by_id.get(id).and_then(|i| i.parent.as_deref());
    }
    result
}

fn resolve_kill_targets(
    catalog: &[TraceData],
    scope: &Scope,
    targets: &[String],
    all: bool,
    err: &mut dyn Write,
) -> Result<std::result::Result<Vec<Row>, ()>> {
    if all && scope.kind == ScopeKind::All {
        writeln!(
            err,
            "ccc: ccc kill --all requires --trace when operating from a human shell"
        )?;
        return Ok(Err(()));
    }
    let rows = selected(catalog, scope);
    let mut resolved = Vec::new();
    if all {
        resolved.extend(rows.into_iter().filter(|r| r.invocation.end.is_none()));
    } else {
        let all_rows = catalog
            .iter()
            .flat_map(|t| {
                t.invocations.iter().map(move |i| Row {
                    trace: t.entry.id.clone(),
                    invocation: i.clone(),
                })
            })
            .collect::<Vec<_>>();
        for target in targets {
            let matches = invocation_matches(&rows, target);
            if matches.len() > 1 {
                writeln!(err, "ccc: ambiguous target {target}")?;
                for row in matches {
                    writeln!(err, "ccc: {}", row.invocation.inv)?;
                }
                return Ok(Err(()));
            }
            if let Some(row) = matches.into_iter().next() {
                if row.invocation.end.is_some() {
                    writeln!(err, "ccc: {target} is not running")?;
                    return Ok(Err(()));
                }
                resolved.push(row);
                continue;
            }
            let outside = invocation_matches(&all_rows, target);
            if scope.kind == ScopeKind::Subtree && !outside.is_empty() {
                writeln!(
                    err,
                    "ccc: {target} is not a subordinate of this agent (use a human shell)"
                )?;
                return Ok(Err(()));
            }
            let sessions = rows
                .iter()
                .filter(|r| {
                    r.invocation.end.is_none()
                        && audit_view::effective_session(&r.invocation) == Some(target.as_str())
                })
                .cloned()
                .collect::<Vec<_>>();
            if sessions.len() > 1 {
                writeln!(err, "ccc: ambiguous target {target}")?;
                for row in sessions {
                    writeln!(err, "ccc: {}", row.invocation.inv)?;
                }
                return Ok(Err(()));
            }
            if let Some(row) = sessions.into_iter().next() {
                resolved.push(row);
                continue;
            }
            if !outside.is_empty()
                || all_rows
                    .iter()
                    .any(|r| r.invocation.inv.starts_with(target) && r.invocation.end.is_some())
            {
                writeln!(err, "ccc: {target} is not running")?;
            } else {
                writeln!(err, "ccc: no running invocation matches {target}")?;
            }
            return Ok(Err(()));
        }
    }
    let mut seen = HashSet::new();
    resolved.retain(|r| seen.insert((r.trace.clone(), r.invocation.inv.clone())));
    let mut order = HashMap::new();
    for (n, row) in resolved.iter().enumerate() {
        order.insert((row.trace.clone(), row.invocation.inv.clone()), n);
    }
    let depths = catalog
        .iter()
        .flat_map(|trace| {
            let by_id = trace
                .invocations
                .iter()
                .map(|i| (i.inv.clone(), i))
                .collect::<HashMap<_, _>>();
            trace
                .invocations
                .iter()
                .map(|i| {
                    (
                        (trace.entry.id.clone(), i.inv.clone()),
                        depth(
                            &Row {
                                trace: trace.entry.id.clone(),
                                invocation: i.clone(),
                            },
                            &by_id,
                        ),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<HashMap<_, _>>();
    resolved.sort_by(|a, b| {
        depths[&(b.trace.clone(), b.invocation.inv.clone())]
            .cmp(&depths[&(a.trace.clone(), a.invocation.inv.clone())])
            .then_with(|| {
                order[&(a.trace.clone(), a.invocation.inv.clone())]
                    .cmp(&order[&(b.trace.clone(), b.invocation.inv.clone())])
            })
    });
    Ok(Ok(resolved))
}

fn send_identity(
    ops: &dyn ProcessOps,
    identity: &ProcId,
    group: Option<u32>,
    signal: i32,
) -> std::result::Result<bool, String> {
    match ops.alive(identity) {
        Liveness::Dead => Ok(false),
        Liveness::Unknown(e) => Err(format!("cannot safely signal an unverified process: {e}")),
        Liveness::Alive => {
            if let Some(pgid) = group {
                if pgid != identity.pid {
                    return Err("cannot safely signal an unverified process group".into());
                }
            }
            let result = match group {
                Some(pgid) => ops.signal_group(pgid, signal),
                None => ops.signal(identity.pid, signal),
            };
            match result {
                Ok(()) => Ok(true),
                Err(e) if e.raw_os_error() == Some(libc::ESRCH) => Ok(false),
                Err(e) => Err(e.to_string()),
            }
        }
    }
}

fn signal_record(
    ctx: &KillContext<'_>,
    row: &Row,
    by: &ProcId,
    to: SignalDestination,
    signal: i32,
    reason: &str,
) -> Result<()> {
    audit::append_signal_for(
        ctx.dir,
        &row.trace,
        &row.invocation.inv,
        audit::SignalPayload {
            signal,
            to,
            by: by.clone(),
            by_inv: ctx.scope_root.clone(),
            reason: reason.into(),
        },
    )
}

fn append_orphan_end(
    ctx: &KillContext<'_>,
    row: &Row,
    swept: usize,
    immediate: bool,
    residual: &[String],
) -> Result<bool> {
    let session = audit_view::effective_session(&row.invocation).map(str::to_string);
    let supervisor = match &row.invocation.child {
        Some(Record::Child(c)) => c.payload.supervisor.pid,
        _ => 0,
    };
    let detail = if residual.is_empty() {
        String::new()
    } else {
        format!("; cleanup residuals: {}", residual.join(", "))
    };
    audit::append_end_for(
        ctx.dir,
        &row.trace,
        &row.invocation.inv,
        audit::EndPayload {
            exit: Some(3),
            signal: None,
            session,
            result: None,
            native: None,
            turns: None,
            warnings: residual.to_vec(),
            error: Some(if immediate {
                format!("orphaned harness terminated by ccc kill --now; supervisor pid {supervisor}; {swept} descendants swept{detail}")
            } else {
                format!("orphaned harness terminated by ccc kill; supervisor pid {supervisor} was already dead; {swept} descendants swept{detail}")
            }),
        },
    )
}

fn sweep_descendants(ops: &dyn ProcessOps, descendants: Vec<ProcId>) -> (usize, Vec<String>) {
    let mut signalled = 0;
    let mut residual_errors = Vec::new();
    for identity in descendants {
        match ops.alive(&identity) {
            Liveness::Dead => {}
            Liveness::Unknown(reason) => {
                residual_errors.push(format!("PID {}: unverified ({reason})", identity.pid))
            }
            Liveness::Alive => match ops.signal(identity.pid, 9) {
                Ok(()) => signalled += 1,
                Err(e) if e.raw_os_error() == Some(libc::ESRCH) => {}
                Err(e) => residual_errors.push(format!("PID {}: {e}", identity.pid)),
            },
        }
    }
    (signalled, residual_errors)
}

fn wait_dead(
    ops: &dyn ProcessOps,
    identity: &ProcId,
    seconds: f64,
) -> std::result::Result<bool, String> {
    let deadline = ops.now() + std::time::Duration::from_secs_f64(seconds.max(0.0));
    loop {
        match ops.alive(identity) {
            Liveness::Dead => return Ok(true),
            Liveness::Unknown(e) => return Err(e),
            Liveness::Alive if ops.now() >= deadline => return Ok(false),
            Liveness::Alive => ops.sleep(std::time::Duration::from_millis(200)),
        }
    }
}

fn act_kill(ctx: &KillContext<'_>, row: &Row, by: &ProcId) -> KillResult {
    const DIRECT_KILL_CONFIRMATION_S: f64 = 5.0;
    let session = audit_view::effective_session(&row.invocation).map(str::to_string);
    let base = || KillResult {
        inv: row.invocation.inv.clone(),
        trace: row.trace.clone(),
        session: session.clone(),
        state: "refused".into(),
        swept: 0,
        error: None,
    };
    let child = match &row.invocation.child {
        Some(Record::Child(c)) => c,
        _ => {
            return {
                let mut r = base();
                r.error = Some("ccc cannot safely signal an unverified process; handle the legacy PID manually".into());
                r
            }
        }
    };
    let start_cmd = audit_view::start_of(&row.invocation).map(|s| s.payload.cmd);
    let group = matches!(
        start_cmd,
        Some(audit::Command::Run | audit::Command::Resume)
    )
    .then_some(child.payload.pgid)
    .filter(|pgid| *pgid == child.payload.proc.pid);
    let supervisor_live = ctx.ops.alive(&child.payload.supervisor);
    let child_live = ctx.ops.alive(&child.payload.proc);
    if matches!(supervisor_live, Liveness::Unknown(_)) || matches!(child_live, Liveness::Unknown(_))
    {
        let mut r = base();
        r.error =
            Some("ccc cannot safely signal an unverified process; handle the PID manually".into());
        return r;
    }
    if supervisor_live == Liveness::Dead && child_live == Liveness::Dead {
        let mut r = base();
        r.state = "lost".into();
        return r;
    }
    let descendants = if ctx.immediate && child_live == Liveness::Alive {
        ctx.ops.descendants(child.payload.proc.pid)
    } else {
        Vec::new()
    };
    if ctx.immediate {
        let mut r = base();
        if supervisor_live == Liveness::Alive {
            if let Err(e) = signal_record(
                ctx,
                row,
                by,
                SignalDestination::Supervisor,
                9,
                "ccc kill --now",
            ) {
                r.error = Some(e.to_string());
                return r;
            }
            if let Err(e) = send_identity(ctx.ops, &child.payload.supervisor, None, 9) {
                r.error = Some(e);
                return r;
            }
        }
        if child_live == Liveness::Alive {
            if let Err(e) =
                signal_record(ctx, row, by, SignalDestination::Child, 9, "ccc kill --now")
            {
                r.error = Some(e.to_string());
                return r;
            }
            if let Err(e) = send_identity(ctx.ops, &child.payload.proc, group, 9) {
                r.error = Some(e);
                return r;
            }
            match wait_dead(ctx.ops, &child.payload.proc, DIRECT_KILL_CONFIRMATION_S) {
                Ok(true) => {}
                Ok(false) => {
                    r.state = "survived".into();
                    r.error = Some("harness survived direct SIGKILL".into());
                    return r;
                }
                Err(reason) => {
                    r.state = "survived".into();
                    r.error = Some(format!(
                        "could not verify harness after direct SIGKILL: {reason}"
                    ));
                    return r;
                }
            }
        }
        let (swept, residual) = sweep_descendants(ctx.ops, descendants);
        match append_orphan_end(ctx, row, swept, true, &residual) {
            Ok(_) => {
                r.state = if residual.is_empty() {
                    "killed"
                } else {
                    "survived"
                }
                .into();
                r.swept = swept;
                if !residual.is_empty() {
                    r.error = Some(residual.join(", "));
                }
                r
            }
            Err(e) => {
                r.error = Some(e.to_string());
                r
            }
        }
    } else if supervisor_live == Liveness::Alive {
        let mut r = base();
        if let Err(e) = signal_record(ctx, row, by, SignalDestination::Supervisor, 15, "ccc kill") {
            r.error = Some(e.to_string());
            return r;
        }
        if let Err(e) = send_identity(ctx.ops, &child.payload.supervisor, None, 15) {
            r.error = Some(e);
            return r;
        }
        if ctx.no_wait {
            r.state = "signalled".into();
            return r;
        }
        match wait_dead(
            ctx.ops,
            &child.payload.supervisor,
            ctx.cfg.process.kill_grace_s + 5.0,
        ) {
            Ok(true) => {
                r.state = "killed".into();
                r
            }
            Ok(false) => {
                r.state = "survived".into();
                r.error = Some("supervisor survived the grace period".into());
                r
            }
            Err(e) => {
                r.state = "survived".into();
                r.error = Some(format!("could not verify supervisor after SIGTERM: {e}"));
                r
            }
        }
    } else {
        let mut r = base();
        let descendants = if child_live == Liveness::Alive {
            ctx.ops.descendants(child.payload.proc.pid)
        } else {
            Vec::new()
        };
        if child_live == Liveness::Alive {
            if let Err(e) = signal_record(ctx, row, by, SignalDestination::Child, 15, "ccc kill") {
                r.error = Some(e.to_string());
                return r;
            }
            if let Err(e) = send_identity(ctx.ops, &child.payload.proc, group, 15) {
                r.state = "survived".into();
                r.error = Some(e);
                return r;
            }
        }
        let mut escalated = false;
        if child_live == Liveness::Alive {
            match wait_dead(
                ctx.ops,
                &child.payload.proc,
                ctx.grace.unwrap_or(ctx.cfg.process.kill_grace_s),
            ) {
                Ok(true) => {}
                Ok(false) => escalated = true,
                Err(e) => {
                    r.state = "survived".into();
                    r.error = Some(format!(
                        "could not verify harness during SIGTERM grace: {e}"
                    ));
                    return r;
                }
            }
        }
        let mut sweep = descendants;
        if escalated {
            sweep.extend(ctx.ops.descendants(child.payload.proc.pid));
            if let Err(e) =
                signal_record(ctx, row, by, SignalDestination::Child, 9, "grace expired")
            {
                r.error = Some(e.to_string());
                return r;
            }
            if let Err(e) = send_identity(ctx.ops, &child.payload.proc, group, 9) {
                r.error = Some(e);
                return r;
            }
            match wait_dead(ctx.ops, &child.payload.proc, DIRECT_KILL_CONFIRMATION_S) {
                Ok(true) => {}
                Ok(false) => {
                    r.state = "survived".into();
                    r.error = Some("harness survived direct SIGKILL".into());
                    return r;
                }
                Err(reason) => {
                    r.state = "survived".into();
                    r.error = Some(format!(
                        "could not verify harness after direct SIGKILL: {reason}"
                    ));
                    return r;
                }
            }
        }
        sweep.sort_by_key(|p| (p.pid, p.start, p.boot_id.clone()));
        sweep.dedup();
        let (swept, residual) = sweep_descendants(ctx.ops, sweep);
        match append_orphan_end(ctx, row, swept, false, &residual) {
            Ok(_) => {
                r.state = if residual.is_empty() {
                    "killed"
                } else {
                    "survived"
                }
                .into();
                r.swept = swept;
                if !residual.is_empty() {
                    r.error = Some(residual.join(", "));
                }
                r
            }
            Err(e) => {
                r.error = Some(e.to_string());
                r
            }
        }
    }
}

pub fn kill_with(
    mut ctx: KillContext<'_>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<i32> {
    let _now = ctx.now;
    let catalog = load_traces(ctx.dir, err)?;
    let scope = match parse_scope_from_ancestors_named(
        &catalog,
        ctx.env,
        ctx.trace.as_deref(),
        &procs::ancestors(),
        "kill",
        err,
    )? {
        Ok(s) => s,
        Err(()) => return Ok(2),
    };
    ctx.scope_root = scope.root.clone();
    let rows = match resolve_kill_targets(&catalog, &scope, &ctx.targets, ctx.all, err)? {
        Ok(rows) => rows,
        Err(()) => return Ok(2),
    };
    let by = match ctx.ops.current() {
        Ok(by) => by,
        Err(e) => {
            writeln!(err, "ccc: {e:#}")?;
            return Ok(1);
        }
    };
    let results = rows
        .iter()
        .map(|row| act_kill(&ctx, row, &by))
        .collect::<Vec<_>>();
    if ctx.json {
        serde_json::to_writer(&mut *out, &serde_json::json!({"targets": results}))?;
        writeln!(out)?;
    } else {
        for r in &results {
            writeln!(
                out,
                "{} {} session={}{}",
                audit_view::short(&r.inv, false),
                r.state,
                r.session.as_deref().unwrap_or("-"),
                r.error
                    .as_deref()
                    .map(|e| format!(" ‹{}›", audit_view::sanitize(e)))
                    .unwrap_or_default()
            )?;
        }
    }
    Ok(
        if results
            .iter()
            .all(|r| matches!(r.state.as_str(), "killed" | "lost" | "signalled"))
        {
            0
        } else {
            1
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::{
        ChildPayload, Command, EndPayload, EndRecord, SignalDestination, SignalPayload,
        SignalRecord, StartPayload, StartRecord,
    };
    use crate::testutil::TempDir;
    use std::collections::{BTreeMap, HashMap, VecDeque};
    use std::sync::Mutex;
    use std::time::Instant;

    const TRACE: &str = "11111111-1111-4111-8111-111111111111";
    const ROOT: &str = "22222222-2222-4222-8222-222222222222";
    const CHILD: &str = "33333333-3333-4333-8333-333333333333";
    const SIBLING: &str = "44444444-4444-4444-8444-444444444444";

    fn proc(pid: u32) -> ProcId {
        ProcId {
            pid,
            start: pid as u64 + 100,
            boot_id: "test-boot".into(),
        }
    }

    fn child_record(inv: &str, supervisor: ProcId, harness: ProcId, pgid: u32) -> Record {
        Record::Child(crate::audit::ChildRecord {
            ts: "2026-01-01T00:00:01Z".parse().unwrap(),
            inv: inv.into(),
            payload: ChildPayload {
                proc: harness,
                pgid,
                supervisor,
            },
        })
    }

    fn write_records(dir: &Path, trace: &str, records: &[Record]) {
        let text = records
            .iter()
            .map(|r| serde_json::to_string(r).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        std::fs::write(dir.join(format!("{trace}.jsonl")), text).unwrap();
    }

    struct TestOps {
        current: Option<ProcId>,
        live: Mutex<HashMap<ProcId, VecDeque<Liveness>>>,
        descendants: Mutex<VecDeque<Vec<ProcId>>>,
        signals: Mutex<Vec<(bool, u32, i32)>>,
        signal_errors: Mutex<HashMap<u32, VecDeque<i32>>>,
        sleeps: Mutex<usize>,
        esrch: bool,
        signal_error: Mutex<Option<i32>>,
        events: Mutex<Vec<&'static str>>,
        now: Mutex<Instant>,
    }

    impl Default for TestOps {
        fn default() -> Self {
            Self {
                current: None,
                live: Mutex::new(HashMap::new()),
                descendants: Mutex::new(VecDeque::new()),
                signals: Mutex::new(Vec::new()),
                signal_errors: Mutex::new(HashMap::new()),
                sleeps: Mutex::new(0),
                esrch: false,
                signal_error: Mutex::new(None),
                events: Mutex::new(Vec::new()),
                now: Mutex::new(Instant::now()),
            }
        }
    }

    impl TestOps {
        fn set(&self, id: ProcId, states: &[Liveness]) {
            self.live
                .lock()
                .unwrap()
                .insert(id, states.iter().cloned().collect());
        }
        fn signal_log(&self) -> Vec<(bool, u32, i32)> {
            self.signals.lock().unwrap().clone()
        }
        fn fail_signal(&self, pid: u32, errors: &[i32]) {
            self.signal_errors
                .lock()
                .unwrap()
                .insert(pid, errors.iter().copied().collect());
        }
    }

    impl ProcessOps for TestOps {
        fn current(&self) -> Result<ProcId> {
            self.current
                .clone()
                .ok_or_else(|| anyhow::anyhow!("no current identity"))
        }
        fn alive(&self, id: &ProcId) -> Liveness {
            let mut live = self.live.lock().unwrap();
            let states = live.entry(id.clone()).or_default();
            states
                .pop_front()
                .or_else(|| states.back().cloned())
                .unwrap_or(Liveness::Dead)
        }
        fn descendants(&self, _pid: u32) -> Vec<ProcId> {
            self.events.lock().unwrap().push("descendants");
            self.descendants
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_default()
        }
        fn signal(&self, pid: u32, signal: i32) -> std::io::Result<()> {
            self.events.lock().unwrap().push("signal");
            self.signals.lock().unwrap().push((false, pid, signal));
            if let Some(error) = self
                .signal_errors
                .lock()
                .unwrap()
                .get_mut(&pid)
                .and_then(VecDeque::pop_front)
            {
                return Err(std::io::Error::from_raw_os_error(error));
            }
            if let Some(error) = *self.signal_error.lock().unwrap() {
                return Err(std::io::Error::from_raw_os_error(error));
            }
            if self.esrch {
                Err(std::io::Error::from_raw_os_error(libc::ESRCH))
            } else {
                Ok(())
            }
        }
        fn signal_group(&self, pgid: u32, signal: i32) -> std::io::Result<()> {
            self.events.lock().unwrap().push("signal");
            self.signals.lock().unwrap().push((true, pgid, signal));
            if let Some(error) = self
                .signal_errors
                .lock()
                .unwrap()
                .get_mut(&pgid)
                .and_then(VecDeque::pop_front)
            {
                return Err(std::io::Error::from_raw_os_error(error));
            }
            if let Some(error) = *self.signal_error.lock().unwrap() {
                return Err(std::io::Error::from_raw_os_error(error));
            }
            if self.esrch {
                Err(std::io::Error::from_raw_os_error(libc::ESRCH))
            } else {
                Ok(())
            }
        }
        fn sleep(&self, duration: std::time::Duration) {
            *self.sleeps.lock().unwrap() += 1;
            *self.now.lock().unwrap() += duration;
        }
        fn now(&self) -> Instant {
            *self.now.lock().unwrap()
        }
    }

    fn action_context<'a>(
        dir: &'a Path,
        ops: &'a TestOps,
        immediate: bool,
        no_wait: bool,
        grace: Option<f64>,
        root: Option<String>,
    ) -> KillContext<'a> {
        KillContext {
            dir,
            cfg: Box::leak(Box::new(Config::default())),
            env: Box::leak(Box::new(BTreeMap::new())),
            now: Timestamp::now(),
            trace: Some(TRACE.into()),
            targets: vec![],
            all: false,
            immediate,
            grace,
            no_wait,
            json: false,
            ops,
            scope_root: root,
        }
    }

    fn command_kill(
        dir: &Path,
        ops: &TestOps,
        immediate: bool,
        json: bool,
    ) -> (i32, Vec<u8>, Vec<u8>) {
        let mut ctx = action_context(dir, ops, immediate, false, Some(0.4), Some(ROOT.into()));
        ctx.trace = Some(TRACE.into());
        ctx.targets = vec![ROOT.into()];
        ctx.json = json;
        let mut out = Vec::new();
        let mut err = Vec::new();
        let code = kill_with(ctx, &mut out, &mut err).unwrap();
        (code, out, err)
    }

    fn orphan_fixture(
        label: &str,
        supervisor: u32,
        harness: u32,
        descendant: Option<u32>,
        harness_states: &[Liveness],
    ) -> (TempDir, Row, TestOps, ProcId) {
        let dir = TempDir::new(label);
        let supervisor = proc(supervisor);
        let harness = proc(harness);
        let by = proc(harness.pid + 100);
        let row = action_row(
            dir.path(),
            ROOT,
            Command::Run,
            supervisor.clone(),
            harness.clone(),
            harness.pid,
            Some("session"),
        );
        let ops = TestOps {
            current: Some(by.clone()),
            ..Default::default()
        };
        ops.set(supervisor, &[Liveness::Dead]);
        ops.set(harness, harness_states);
        if let Some(pid) = descendant {
            let descendant = proc(pid);
            ops.set(descendant.clone(), &[Liveness::Alive]);
            *ops.descendants.lock().unwrap() =
                VecDeque::from(vec![vec![descendant.clone()], vec![descendant]]);
        }
        (dir, row, ops, by)
    }

    fn start(inv: &str, parent: Option<&str>, ts: &str) -> Record {
        Record::Start(StartRecord {
            ts: ts.parse().unwrap(),
            inv: inv.into(),
            parent: parent.map(str::to_string),
            payload: StartPayload {
                pid: u32::MAX,
                cmd: Command::Run,
                agent: Some("worker".into()),
                session: None,
                cwd_arg: Some("/work".into()),
                profile_arg: None,
                harness_arg: Some(Harness::Codex),
                json: false,
                prompt_source: None,
                prompt_file: None,
                prompt: Some("prompt".into()),
                passthrough: vec![],
                fork: false,
            },
        })
    }

    fn start_kind(inv: &str, parent: Option<&str>, ts: &str, cmd: Command) -> Record {
        let mut record = start(inv, parent, ts);
        if let Record::Start(s) = &mut record {
            s.payload.cmd = cmd;
        }
        record
    }

    fn action_row(
        dir: &Path,
        inv: &str,
        cmd: Command,
        supervisor: ProcId,
        harness: ProcId,
        pgid: u32,
        session: Option<&str>,
    ) -> Row {
        let mut records = vec![
            start_kind(inv, None, "2026-01-01T00:00:00Z", cmd),
            child_record(inv, supervisor, harness, pgid),
        ];
        if let Some(session) = session {
            records.push(Record::Session(crate::audit::SessionRecord {
                ts: "2026-01-01T00:00:02Z".parse().unwrap(),
                inv: inv.into(),
                session: session.into(),
            }));
        }
        write_records(dir, TRACE, &records);
        let data = trace(records);
        Row {
            trace: TRACE.into(),
            invocation: data.invocations.into_iter().next().unwrap(),
        }
    }

    fn end(inv: &str, exit: Option<i32>, signal: Option<i32>) -> Record {
        Record::End(EndRecord {
            ts: "2026-01-01T00:01:00Z".parse().unwrap(),
            inv: inv.into(),
            payload: EndPayload {
                exit,
                signal,
                session: None,
                result: None,
                error: None,
                warnings: vec![],
                native: None,
                turns: None,
            },
            duration_ms: 1200,
        })
    }

    fn trace(records: Vec<Record>) -> TraceData {
        let (invocations, _) = audit_view::join_invocations(&records);
        TraceData {
            entry: Entry {
                id: TRACE.into(),
                index: None,
                exists: true,
                start: None,
            },
            invocations,
        }
    }

    fn named_trace(id: &str, records: Vec<Record>) -> TraceData {
        let mut data = trace(records);
        data.entry.id = id.into();
        data
    }

    #[test]
    fn scope_ancestry_and_lineage_seams_are_authoritative() {
        let near = proc(101);
        let far = proc(102);
        let trace_a = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
        let trace_b = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
        let near_inv = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbc";
        let far_inv = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaab";
        let entries = || {
            vec![
                named_trace(
                    trace_a,
                    vec![
                        start(far_inv, None, "2026-01-01T00:00:00Z"),
                        child_record(far_inv, proc(103), far.clone(), 103),
                    ],
                ),
                named_trace(
                    trace_b,
                    vec![
                        start(near_inv, None, "2026-01-01T00:00:00Z"),
                        child_record(near_inv, proc(104), near.clone(), 104),
                    ],
                ),
            ]
        };

        let mut err = Vec::new();
        assert!(parse_scope_from_ancestors(
            &entries(),
            &BTreeMap::from([
                ("CCC_TRACE_ID".into(), trace_b.into()),
                ("CCC_PARENT_INVOCATION_ID".into(), near_inv.into()),
            ]),
            Some(trace_a),
            std::slice::from_ref(&near),
            &mut err,
        )
        .unwrap()
        .is_err());
        assert_eq!(
            String::from_utf8(err).unwrap(),
            "ccc: --trace is not available to a spawned agent; ccc status shows your own subagents\n"
        );

        let mut err = Vec::new();
        let scope = parse_scope_from_ancestors(
            &entries(),
            &BTreeMap::from([
                ("CCC_TRACE_ID".into(), trace_a.into()),
                ("CCC_PARENT_INVOCATION_ID".into(), far_inv.into()),
            ]),
            None,
            &[near.clone(), far.clone()],
            &mut err,
        )
        .unwrap()
        .unwrap();
        assert_eq!(scope.source, ScopeSource::Ancestry);
        assert_eq!(scope.trace.as_deref(), Some(trace_b));
        assert_eq!(scope.root.as_deref(), Some(near_inv));
        assert!(String::from_utf8(err).unwrap().contains("inconsistent"));
        let mut err = Vec::new();
        assert!(
            resolve_kill_targets(&entries(), &scope, &[far_inv.into()], false, &mut err,)
                .unwrap()
                .is_err()
        );
        assert_eq!(
            String::from_utf8(err).unwrap(),
            format!("ccc: {far_inv} is not a subordinate of this agent (use a human shell)\n")
        );

        let mut err = Vec::new();
        let scope = parse_scope_from_ancestors(
            &entries(),
            &BTreeMap::from([
                ("CCC_TRACE_ID".into(), trace_b.into()),
                ("CCC_PARENT_INVOCATION_ID".into(), near_inv.into()),
            ]),
            None,
            &[near.clone(), far.clone()],
            &mut err,
        )
        .unwrap()
        .unwrap();
        assert_eq!(scope.source, ScopeSource::Ancestry);
        assert!(err.is_empty());

        let mut no_near_child = entries();
        no_near_child[1] =
            named_trace(trace_b, vec![start(near_inv, None, "2026-01-01T00:00:00Z")]);
        let mut err = Vec::new();
        let scope = parse_scope_from_ancestors(
            &no_near_child,
            &BTreeMap::from([
                ("CCC_TRACE_ID".into(), trace_b.into()),
                ("CCC_PARENT_INVOCATION_ID".into(), near_inv.into()),
            ]),
            None,
            std::slice::from_ref(&near),
            &mut err,
        )
        .unwrap()
        .unwrap();
        assert_eq!(scope.source, ScopeSource::Env);
        assert_eq!(scope.root.as_deref(), Some(near_inv));
        assert!(err.is_empty());
    }

    #[test]
    fn kill_resolution_is_scoped_deduplicated_and_leaves_first() {
        let second = "33333333-3333-4333-8333-333333333334";
        let data = trace(vec![
            start(ROOT, None, "2026-01-01T00:00:00Z"),
            start(CHILD, Some(ROOT), "2026-01-01T00:00:01Z"),
            Record::Session(crate::audit::SessionRecord {
                ts: "2026-01-01T00:00:02Z".parse().unwrap(),
                inv: CHILD.into(),
                session: "sess".into(),
            }),
            start(second, Some(CHILD), "2026-01-01T00:00:03Z"),
            start(SIBLING, None, "2026-01-01T00:00:04Z"),
            end(SIBLING, Some(0), None),
        ]);
        let scope = Scope {
            kind: ScopeKind::Trace,
            source: ScopeSource::None,
            trace: Some(TRACE.into()),
            root: None,
        };
        let mut err = Vec::new();
        let resolved = resolve_kill_targets(
            &[data],
            &scope,
            &["sess".into(), second.to_string(), CHILD.to_string()],
            false,
            &mut err,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            resolved
                .iter()
                .map(|r| r.invocation.inv.as_str())
                .collect::<Vec<_>>(),
            vec![second, CHILD]
        );
        let ambiguous = "33333333-3333-4333-8333-333333333335";
        let mut err = Vec::new();
        assert!(resolve_kill_targets(
            &[trace(vec![
                start(ROOT, None, "2026-01-01T00:00:00Z"),
                start(CHILD, Some(ROOT), "2026-01-01T00:00:01Z"),
                start(second, Some(ROOT), "2026-01-01T00:00:02Z"),
                start(ambiguous, Some(ROOT), "2026-01-01T00:00:03Z")
            ])],
            &scope,
            &["33333333".into()],
            false,
            &mut err
        )
        .unwrap()
        .is_err());
        assert_eq!(
            String::from_utf8(err).unwrap(),
            format!(
                "ccc: ambiguous target 33333333\nccc: {CHILD}\nccc: {second}\nccc: {ambiguous}\n"
            )
        );
        let duplicate = "44444444-4444-4444-8444-444444444444";
        let mut err = Vec::new();
        assert!(resolve_kill_targets(
            &[trace(vec![
                start(CHILD, None, "2026-01-01T00:00:00Z"),
                Record::Session(crate::audit::SessionRecord {
                    ts: "2026-01-01T00:00:01Z".parse().unwrap(),
                    inv: CHILD.into(),
                    session: "duplicate".into(),
                }),
                start(duplicate, None, "2026-01-01T00:00:02Z"),
                Record::Session(crate::audit::SessionRecord {
                    ts: "2026-01-01T00:00:03Z".parse().unwrap(),
                    inv: duplicate.into(),
                    session: "duplicate".into(),
                }),
            ])],
            &scope,
            &["duplicate".into()],
            false,
            &mut err,
        )
        .unwrap()
        .is_err());
        assert_eq!(
            String::from_utf8(err).unwrap(),
            format!("ccc: ambiguous target duplicate\nccc: {CHILD}\nccc: {duplicate}\n")
        );
        let mut err = Vec::new();
        assert!(resolve_kill_targets(
            &[trace(vec![
                start(ROOT, None, "2026-01-01T00:00:00Z"),
                start(CHILD, Some(ROOT), "2026-01-01T00:00:01Z"),
                end(CHILD, Some(0), None)
            ])],
            &scope,
            &[CHILD[..35].into()],
            false,
            &mut err
        )
        .unwrap()
        .is_err());
        assert_eq!(
            String::from_utf8(err).unwrap(),
            format!("ccc: {} is not running\n", &CHILD[..35])
        );
        let subtree = Scope {
            kind: ScopeKind::Subtree,
            source: ScopeSource::Env,
            trace: Some(TRACE.into()),
            root: Some(ROOT.into()),
        };
        let mut err = Vec::new();
        assert!(resolve_kill_targets(
            &[trace(vec![
                start(ROOT, None, "2026-01-01T00:00:00Z"),
                start(CHILD, Some(ROOT), "2026-01-01T00:00:01Z"),
                start(SIBLING, None, "2026-01-01T00:00:02Z")
            ])],
            &subtree,
            &[SIBLING[..8].into()],
            false,
            &mut err
        )
        .unwrap()
        .is_err());
        assert_eq!(
            String::from_utf8(err).unwrap(),
            format!(
                "ccc: {} is not a subordinate of this agent (use a human shell)\n",
                &SIBLING[..8]
            )
        );
        let mut err = Vec::new();
        assert!(resolve_kill_targets(
            &[trace(vec![start(ROOT, None, "2026-01-01T00:00:00Z")])],
            &Scope {
                kind: ScopeKind::All,
                source: ScopeSource::None,
                trace: None,
                root: None
            },
            &[],
            true,
            &mut err
        )
        .unwrap()
        .is_err());
        assert_eq!(
            String::from_utf8(err).unwrap(),
            "ccc: ccc kill --all requires --trace when operating from a human shell\n"
        );
        let rows = resolve_kill_targets(
            &[trace(vec![
                start(ROOT, None, "2026-01-01T00:00:00Z"),
                start(CHILD, Some(ROOT), "2026-01-01T00:00:01Z"),
            ])],
            &subtree,
            &[],
            true,
            &mut Vec::new(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            rows.iter()
                .map(|r| r.invocation.inv.as_str())
                .collect::<Vec<_>>(),
            vec![CHILD]
        );
    }

    #[test]
    fn kill_resolution_orders_each_trace_by_its_own_depth_map() {
        let a_root = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
        let a_child = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaab";
        let a_leaf = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaac";
        let b_root = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
        let b_child = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbc";
        let catalog = vec![
            named_trace(
                TRACE,
                vec![
                    start(a_root, None, "2026-01-01T00:00:00Z"),
                    start(a_child, Some(a_root), "2026-01-01T00:00:01Z"),
                    start(a_leaf, Some(a_child), "2026-01-01T00:00:02Z"),
                ],
            ),
            named_trace(
                "55555555-5555-4555-8555-555555555555",
                vec![
                    start(b_root, None, "2026-01-01T00:00:00Z"),
                    start(b_child, Some(b_root), "2026-01-01T00:00:01Z"),
                ],
            ),
        ];
        let scope = Scope {
            kind: ScopeKind::All,
            source: ScopeSource::None,
            trace: None,
            root: None,
        };
        let targets = vec![b_child.into(), a_root.into(), a_leaf.into(), b_root.into()];
        let resolved = resolve_kill_targets(&catalog, &scope, &targets, false, &mut Vec::new())
            .unwrap()
            .unwrap();
        assert_eq!(
            resolved
                .iter()
                .map(|r| r.invocation.inv.as_str())
                .collect::<Vec<_>>(),
            vec![a_leaf, b_child, a_root, b_root]
        );
        let reversed = vec![b_root.into(), a_leaf.into(), a_root.into(), b_child.into()];
        let resolved_reversed =
            resolve_kill_targets(&catalog, &scope, &reversed, false, &mut Vec::new())
                .unwrap()
                .unwrap();
        assert_eq!(
            resolved_reversed
                .iter()
                .map(|r| r.invocation.inv.as_str())
                .collect::<Vec<_>>(),
            vec![a_leaf, b_child, b_root, a_root]
        );
    }

    #[test]
    fn kill_live_supervisor_records_term_and_waits_only_when_requested() {
        for (no_wait, states, expected) in [
            (true, vec![Liveness::Alive], "signalled"),
            (false, vec![Liveness::Alive, Liveness::Dead], "killed"),
            (false, vec![Liveness::Alive; 100], "survived"),
        ] {
            let dir = TempDir::new("kill-supervisor");
            let supervisor = proc(10);
            let harness = proc(11);
            let by = proc(12);
            let row = action_row(
                dir.path(),
                ROOT,
                Command::Run,
                supervisor.clone(),
                harness.clone(),
                77,
                Some("session"),
            );
            let ops = TestOps {
                current: Some(by.clone()),
                ..Default::default()
            };
            let mut supervisor_states = vec![Liveness::Alive];
            supervisor_states.extend(states.clone());
            ops.set(supervisor.clone(), &supervisor_states);
            ops.set(harness, &[Liveness::Alive]);
            let ctx = action_context(dir.path(), &ops, false, no_wait, None, Some("root".into()));
            let result = act_kill(&ctx, &row, &by);
            assert_eq!(result.state, expected);
            assert_eq!(ops.signal_log(), vec![(false, 10, 15)]);
            let elapsed = *ops.sleeps.lock().unwrap() as f64 * 0.2;
            if no_wait {
                assert_eq!(elapsed, 0.0);
            } else if expected == "survived" {
                assert_eq!(
                    result.error.as_deref(),
                    Some("supervisor survived the grace period")
                );
                assert!(!audit::read_trace(dir.path(), TRACE)
                    .unwrap()
                    .iter()
                    .any(|r| matches!(r, Record::End(_))));
                assert!(elapsed >= Config::default().process.kill_grace_s + 5.0);
                assert!(elapsed <= Config::default().process.kill_grace_s + 5.2);
            } else {
                assert!(elapsed >= 0.0);
            }
            let records = audit::read_trace(dir.path(), TRACE).unwrap();
            let signals = records
                .iter()
                .filter_map(|r| match r {
                    Record::Signal(s) => Some(s),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(signals.len(), 1);
            assert!(matches!(
                signals[0].payload.to,
                SignalDestination::Supervisor
            ));
            assert_eq!(signals[0].payload.signal, 15);
            assert_eq!(signals[0].payload.by_inv.as_deref(), Some("root"));
            assert_eq!(signals[0].payload.reason, "ccc kill");
        }
    }

    #[test]
    fn kill_live_supervisor_survival_is_a_json_command_failure() {
        let dir = TempDir::new("kill-supervisor-survived-json");
        let supervisor = proc(180);
        let harness = proc(181);
        let by = proc(182);
        action_row(
            dir.path(),
            ROOT,
            Command::Run,
            supervisor.clone(),
            harness.clone(),
            181,
            None,
        );
        let ops = TestOps {
            current: Some(by),
            ..Default::default()
        };
        ops.set(supervisor, &vec![Liveness::Alive; 100]);
        ops.set(harness, &[Liveness::Alive]);
        let (code, out, err) = command_kill(dir.path(), &ops, false, true);
        assert_eq!(code, 1, "{}", String::from_utf8_lossy(&err));
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(value["targets"][0]["state"], "survived");
        assert_eq!(
            value["targets"][0]["error"],
            "supervisor survived the grace period"
        );
        assert_eq!(value["targets"][0]["swept"], 0);
        assert!(!audit::read_trace(dir.path(), TRACE)
            .unwrap()
            .iter()
            .any(|r| matches!(r, Record::End(_))));
    }

    #[test]
    fn kill_orphan_escalates_by_identity_and_uses_launch_pid_or_harness_group() {
        for (cmd, group) in [
            (Command::Launch, false),
            (Command::Run, true),
            (Command::Resume, true),
        ] {
            let dir = TempDir::new("kill-orphan");
            let supervisor = proc(20);
            let harness = proc(21);
            let descendant = proc(22);
            let by = proc(23);
            let row = action_row(
                dir.path(),
                ROOT,
                cmd,
                supervisor.clone(),
                harness.clone(),
                if group { harness.pid } else { 88 },
                Some("session"),
            );
            let ops = TestOps {
                current: Some(by.clone()),
                ..Default::default()
            };
            ops.set(supervisor, &[Liveness::Dead]);
            ops.set(
                harness.clone(),
                &[
                    Liveness::Alive,
                    Liveness::Alive,
                    Liveness::Alive,
                    Liveness::Alive,
                    Liveness::Alive,
                    Liveness::Dead,
                ],
            );
            ops.set(descendant.clone(), &[Liveness::Alive]);
            *ops.descendants.lock().unwrap() =
                VecDeque::from(vec![vec![descendant.clone()], vec![descendant.clone()]]);
            let ctx = action_context(
                dir.path(),
                &ops,
                false,
                true,
                Some(0.2),
                Some("root".into()),
            );
            let result = act_kill(&ctx, &row, &by);
            assert_eq!(result.state, "killed");
            assert_eq!(result.session.as_deref(), Some("session"));
            assert_eq!(result.swept, 1);
            let log = ops.signal_log();
            assert_eq!(log[0], (group, 21, 15));
            assert_eq!(log[1], (group, 21, 9));
            assert_eq!(log[2], (false, 22, 9));
            let records = audit::read_trace(dir.path(), TRACE).unwrap();
            let signals = records
                .iter()
                .filter_map(|r| match r {
                    Record::Signal(s) => Some(s),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert_eq!(signals.len(), 2);
            assert!(signals
                .iter()
                .all(|s| matches!(s.payload.to, SignalDestination::Child)));
            assert!(signals
                .iter()
                .all(|s| s.payload.by_inv.as_deref() == Some("root")));
            assert_eq!(
                signals
                    .iter()
                    .map(|s| (s.payload.signal, s.payload.reason.as_str()))
                    .collect::<Vec<_>>(),
                vec![(15, "ccc kill"), (9, "grace expired")]
            );
            assert_eq!(
                records
                    .iter()
                    .filter(|r| matches!(r, Record::End(_)))
                    .count(),
                1
            );
            let end = records
                .iter()
                .find_map(|r| match r {
                    Record::End(e) => Some(e),
                    _ => None,
                })
                .unwrap();
            assert_eq!(end.payload.exit, Some(3));
            assert!(end
                .payload
                .error
                .as_deref()
                .unwrap()
                .contains("1 descendants swept"));
        }
        let dir = TempDir::new("kill-esrch");
        let supervisor = proc(24);
        let harness = proc(25);
        let by = proc(26);
        let row = action_row(
            dir.path(),
            ROOT,
            Command::Launch,
            supervisor.clone(),
            harness.clone(),
            1,
            None,
        );
        let ops = TestOps {
            current: Some(by.clone()),
            esrch: true,
            ..Default::default()
        };
        ops.set(supervisor, &[Liveness::Dead]);
        ops.set(harness.clone(), &vec![Liveness::Alive; 4]);
        let result = act_kill(
            &action_context(dir.path(), &ops, false, true, Some(0.2), None),
            &row,
            &by,
        );
        assert_eq!(result.state, "killed");
        assert_eq!(result.swept, 0);
        let dir = TempDir::new("kill-union");
        let harness = proc(27);
        let first = proc(28);
        let second = proc(29);
        let row = action_row(
            dir.path(),
            ROOT,
            Command::Run,
            proc(26),
            harness.clone(),
            2,
            None,
        );
        let ops = TestOps {
            current: Some(by),
            ..Default::default()
        };
        ops.set(proc(26), &[Liveness::Dead]);
        ops.set(
            harness,
            &[
                Liveness::Alive,
                Liveness::Alive,
                Liveness::Alive,
                Liveness::Alive,
                Liveness::Alive,
                Liveness::Dead,
            ],
        );
        ops.set(first.clone(), &[Liveness::Alive]);
        ops.set(second.clone(), &[Liveness::Alive]);
        *ops.descendants.lock().unwrap() = VecDeque::from(vec![vec![first], vec![second]]);
        let result = act_kill(
            &action_context(dir.path(), &ops, false, true, Some(0.2), None),
            &row,
            &proc(26),
        );
        assert_eq!(result.swept, 2);
    }

    #[test]
    fn orphan_sweep_reports_dead_esrch_unknown_permission_and_success_in_both_modes() {
        for immediate in [true, false] {
            let dead = proc(70);
            let esrch = proc(71);
            let unknown = proc(72);
            let denied = proc(73);
            let success = proc(74);
            let mut ops = TestOps::default();
            ops.set(dead.clone(), &[Liveness::Dead]);
            ops.set(esrch.clone(), &[Liveness::Alive]);
            ops.set(unknown.clone(), &[Liveness::Unknown("procfs".into())]);
            ops.set(denied.clone(), &[Liveness::Alive]);
            ops.set(success.clone(), &[Liveness::Alive]);
            let (swept, residual) = if immediate {
                ops.esrch = true;
                sweep_descendants(&ops, vec![dead, esrch, unknown, denied, success])
            } else {
                *ops.signal_error.lock().unwrap() = Some(libc::EPERM);
                sweep_descendants(&ops, vec![dead, esrch, unknown, denied, success])
            };
            assert_eq!(swept, 0);
            assert_eq!(residual.len(), if immediate { 1 } else { 4 });
            assert!(residual.iter().any(|e| e.contains("PID 72")));
            if !immediate {
                assert!(residual.iter().any(|e| e.contains("PID 73")));
            }
            let success_ops = TestOps::default();
            let success = proc(75);
            success_ops.set(success.clone(), &[Liveness::Alive]);
            assert_eq!(
                sweep_descendants(&success_ops, vec![success]),
                (1, Vec::new())
            );
        }
    }

    #[test]
    fn kill_now_lost_and_unverified_paths_are_safe_and_json_is_stable() {
        let dir = TempDir::new("kill-now");
        let supervisor = proc(30);
        let harness = proc(31);
        let desc = proc(32);
        let by = proc(33);
        let row = action_row(
            dir.path(),
            ROOT,
            Command::Run,
            supervisor.clone(),
            harness.clone(),
            99,
            Some("session"),
        );
        let ops = TestOps {
            current: Some(by.clone()),
            ..Default::default()
        };
        ops.set(supervisor.clone(), &vec![Liveness::Alive; 2]);
        ops.set(harness.clone(), &vec![Liveness::Alive; 2]);
        ops.set(desc.clone(), &[Liveness::Alive]);
        *ops.descendants.lock().unwrap() = VecDeque::from([vec![desc.clone()]]);
        let ctx = action_context(dir.path(), &ops, true, false, None, None);
        let result = act_kill(&ctx, &row, &by);
        assert_eq!(result.state, "killed");
        assert_eq!(
            ops.events.lock().unwrap().first().copied(),
            Some("descendants")
        );
        assert_eq!(
            ops.signal_log(),
            vec![(false, 30, 9), (false, 31, 9), (false, 32, 9)]
        );
        let records = audit::read_trace(dir.path(), TRACE).unwrap();
        assert_eq!(
            records
                .iter()
                .filter(|r| matches!(r, Record::End(_)))
                .count(),
            1
        );
        assert!(records
            .iter()
            .filter_map(|r| match r {
                Record::Signal(s) => Some(s.payload.reason.as_str()),
                _ => None,
            })
            .all(|reason| reason == "ccc kill --now"));
        let dead_dir = TempDir::new("kill-lost");
        let dead = proc(40);
        let dead_row = action_row(
            dead_dir.path(),
            ROOT,
            Command::Launch,
            dead.clone(),
            dead.clone(),
            1,
            None,
        );
        let dead_ops = TestOps {
            current: Some(by.clone()),
            ..Default::default()
        };
        dead_ops.set(dead, &[Liveness::Dead]);
        assert_eq!(
            act_kill(
                &action_context(dead_dir.path(), &dead_ops, false, false, None, None),
                &dead_row,
                &by
            )
            .state,
            "lost"
        );
        assert!(dead_ops.signal_log().is_empty());
        let legacy_dir = TempDir::new("kill-legacy");
        let legacy_records = vec![start(ROOT, None, "2026-01-01T00:00:00Z")];
        write_records(legacy_dir.path(), TRACE, &legacy_records);
        let legacy_row = Row {
            trace: TRACE.into(),
            invocation: trace(legacy_records)
                .invocations
                .into_iter()
                .next()
                .unwrap(),
        };
        let legacy_ops = TestOps {
            current: Some(by.clone()),
            ..Default::default()
        };
        let refused = act_kill(
            &action_context(legacy_dir.path(), &legacy_ops, false, false, None, None),
            &legacy_row,
            &by,
        );
        assert_eq!(refused.state, "refused");
        assert!(legacy_ops.signal_log().is_empty());
        let unknown_dir = TempDir::new("kill-unknown");
        let unknown_row = action_row(
            unknown_dir.path(),
            ROOT,
            Command::Launch,
            proc(50),
            proc(51),
            1,
            None,
        );
        let unknown_ops = TestOps {
            current: Some(by),
            ..Default::default()
        };
        unknown_ops.set(proc(50), &[Liveness::Unknown("probe".into())]);
        unknown_ops.set(proc(51), &[Liveness::Alive]);
        let refused = act_kill(
            &action_context(unknown_dir.path(), &unknown_ops, false, false, None, None),
            &unknown_row,
            &proc(33),
        );
        assert_eq!(refused.state, "refused");
        assert!(unknown_ops.signal_log().is_empty());
    }

    #[test]
    fn kill_post_signal_confirmation_is_fixed_and_reasoned() {
        let dir = TempDir::new("kill-confirm");
        let supervisor = proc(80);
        let harness = proc(81);
        let by = proc(82);
        let row = action_row(
            dir.path(),
            ROOT,
            Command::Run,
            supervisor.clone(),
            harness.clone(),
            harness.pid,
            None,
        );
        let ops = TestOps {
            current: Some(by.clone()),
            ..Default::default()
        };
        ops.set(supervisor, &[Liveness::Alive, Liveness::Alive]);
        ops.set(
            harness.clone(),
            &[
                Liveness::Alive,
                Liveness::Alive,
                Liveness::Alive,
                Liveness::Alive,
                Liveness::Dead,
            ],
        );
        let result = act_kill(
            &action_context(dir.path(), &ops, true, false, Some(0.01), None),
            &row,
            &by,
        );
        assert_eq!(result.state, "killed");
        assert_eq!(*ops.sleeps.lock().unwrap(), 2);
        assert_eq!(
            audit::read_trace(dir.path(), TRACE)
                .unwrap()
                .iter()
                .filter(|r| matches!(r, Record::End(_)))
                .count(),
            1
        );
        let dir = TempDir::new("kill-confirm-survived");
        let supervisor = proc(83);
        let harness = proc(84);
        let row = action_row(
            dir.path(),
            ROOT,
            Command::Run,
            supervisor.clone(),
            harness.clone(),
            harness.pid,
            None,
        );
        let ops = TestOps {
            current: Some(by.clone()),
            ..Default::default()
        };
        ops.set(supervisor, &[Liveness::Alive, Liveness::Alive]);
        ops.set(harness.clone(), &vec![Liveness::Alive; 40]);
        let result = act_kill(
            &action_context(dir.path(), &ops, true, false, Some(0.01), None),
            &row,
            &by,
        );
        assert_eq!(result.state, "survived");
        assert_eq!(
            result.error.as_deref(),
            Some("harness survived direct SIGKILL")
        );
        let elapsed = *ops.sleeps.lock().unwrap() as f64 * 0.2;
        assert!(
            (5.0..=5.2).contains(&elapsed),
            "virtual elapsed time: {elapsed}"
        );
        assert_eq!(
            audit::read_trace(dir.path(), TRACE)
                .unwrap()
                .iter()
                .filter(|r| matches!(r, Record::End(_)))
                .count(),
            0
        );
    }

    #[test]
    fn kill_orphan_confirmation_precedes_descendant_sweep() {
        let states = vec![Liveness::Alive; 10]
            .into_iter()
            .chain([Liveness::Dead])
            .collect::<Vec<_>>();
        let (dir, row, ops, by) =
            orphan_fixture("kill-orphan-command-direct", 90, 91, Some(92), &states);
        let result = act_kill(
            &action_context(dir.path(), &ops, false, false, Some(0.4), Some(ROOT.into())),
            &row,
            &by,
        );
        assert_eq!(result.state, "killed");
        assert_eq!(result.swept, 1);
        assert_eq!(result.error, None);
        assert_eq!(
            ops.signal_log(),
            vec![(true, 91, 15), (true, 91, 9), (false, 92, 9)]
        );
        assert_eq!(
            audit::read_trace(dir.path(), TRACE)
                .unwrap()
                .iter()
                .filter(|r| matches!(r, Record::End(_)))
                .count(),
            1
        );
        let end = audit::read_trace(dir.path(), TRACE)
            .unwrap()
            .into_iter()
            .find_map(|r| match r {
                Record::End(end) => Some(end),
                _ => None,
            })
            .unwrap();
        assert!(end
            .payload
            .error
            .as_deref()
            .unwrap()
            .contains("1 descendants swept"));

        let (dir, _row, ops, _by) =
            orphan_fixture("kill-orphan-command-text", 93, 94, Some(95), &states);
        let (code, out, err) = command_kill(dir.path(), &ops, false, false);
        assert_eq!(code, 0, "{}", String::from_utf8_lossy(&err));
        assert!(String::from_utf8(out)
            .unwrap()
            .ends_with("killed session=session\n"));

        let (dir, _row, ops, _by) =
            orphan_fixture("kill-orphan-command-json", 96, 97, Some(98), &states);
        let (code, out, err) = command_kill(dir.path(), &ops, false, true);
        assert_eq!(code, 0, "{}", String::from_utf8_lossy(&err));
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(value["targets"][0]["state"], "killed");
        assert_eq!(value["targets"][0]["swept"], 1);
        assert_eq!(value["targets"][0]["error"], serde_json::Value::Null);
    }

    #[test]
    fn kill_orphan_direct_confirmation_survival_is_a_command_failure() {
        let states = vec![Liveness::Alive; 40];
        let (dir, row, ops, by) =
            orphan_fixture("kill-orphan-survived-direct", 100, 101, Some(102), &states);
        let result = act_kill(
            &action_context(dir.path(), &ops, false, false, Some(0.4), Some(ROOT.into())),
            &row,
            &by,
        );
        assert_eq!(result.state, "survived");
        assert_eq!(result.swept, 0);
        assert_eq!(
            result.error.as_deref(),
            Some("harness survived direct SIGKILL")
        );
        assert_eq!(ops.signal_log(), vec![(true, 101, 15), (true, 101, 9)]);
        let elapsed = *ops.sleeps.lock().unwrap() as f64 * 0.2;
        assert!(
            (5.0..=5.2).contains(&(elapsed - 0.4)),
            "virtual confirmation time: {elapsed}"
        );
        assert!(!audit::read_trace(dir.path(), TRACE)
            .unwrap()
            .iter()
            .any(|r| matches!(r, Record::End(_))));

        let (dir, _row, ops, _by) =
            orphan_fixture("kill-orphan-survived-text", 103, 104, Some(105), &states);
        let (code, out, err) = command_kill(dir.path(), &ops, false, false);
        assert_eq!(code, 1, "{}", String::from_utf8_lossy(&err));
        assert!(String::from_utf8(out)
            .unwrap()
            .ends_with("survived session=session ‹harness survived direct SIGKILL›\n"));

        let (dir, _row, ops, _by) =
            orphan_fixture("kill-orphan-survived-json", 106, 107, Some(108), &states);
        let (code, out, err) = command_kill(dir.path(), &ops, false, true);
        assert_eq!(code, 1, "{}", String::from_utf8_lossy(&err));
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(value["targets"][0]["state"], "survived");
        assert_eq!(
            value["targets"][0]["error"],
            "harness survived direct SIGKILL"
        );
        assert_eq!(value["targets"][0]["swept"], 0);
    }

    #[test]
    fn kill_now_always_alive_confirmation_is_a_command_failure() {
        let make = |label: &str| {
            let dir = TempDir::new(label);
            let supervisor = proc(170);
            let harness = proc(171);
            let by = proc(172);
            let row = action_row(
                dir.path(),
                ROOT,
                Command::Run,
                supervisor.clone(),
                harness.clone(),
                171,
                None,
            );
            let ops = TestOps {
                current: Some(by.clone()),
                ..Default::default()
            };
            ops.set(supervisor, &[Liveness::Alive, Liveness::Alive]);
            ops.set(harness, &vec![Liveness::Alive; 40]);
            (dir, row, ops, by)
        };
        let (dir, row, ops, by) = make("kill-now-survived-direct");
        let result = act_kill(
            &action_context(dir.path(), &ops, true, false, None, None),
            &row,
            &by,
        );
        assert_eq!(result.state, "survived");
        assert_eq!(
            result.error.as_deref(),
            Some("harness survived direct SIGKILL")
        );
        assert_eq!(result.swept, 0);
        assert!(!audit::read_trace(dir.path(), TRACE)
            .unwrap()
            .iter()
            .any(|r| matches!(r, Record::End(_))));

        let (dir, _row, ops, _by) = make("kill-now-survived-text");
        let (code, out, err) = command_kill(dir.path(), &ops, true, false);
        assert_eq!(code, 1, "{}", String::from_utf8_lossy(&err));
        assert!(String::from_utf8(out)
            .unwrap()
            .ends_with("survived session=- ‹harness survived direct SIGKILL›\n"));
        assert!(!audit::read_trace(dir.path(), TRACE)
            .unwrap()
            .iter()
            .any(|r| matches!(r, Record::End(_))));

        let (dir, _row, ops, _by) = make("kill-now-survived-json");
        let (code, out, err) = command_kill(dir.path(), &ops, true, true);
        assert_eq!(code, 1, "{}", String::from_utf8_lossy(&err));
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(value["targets"][0]["state"], "survived");
        assert_eq!(
            value["targets"][0]["error"],
            "harness survived direct SIGKILL"
        );
        assert_eq!(value["targets"][0]["swept"], 0);
        assert!(!audit::read_trace(dir.path(), TRACE)
            .unwrap()
            .iter()
            .any(|r| matches!(r, Record::End(_))));
    }

    #[test]
    fn kill_orphan_post_kill_verification_error_prevents_sweep() {
        let states = vec![Liveness::Alive; 10]
            .into_iter()
            .chain([Liveness::Unknown("proc hidden".into())])
            .collect::<Vec<_>>();
        let (dir, row, ops, by) =
            orphan_fixture("kill-orphan-unknown-direct", 110, 111, Some(112), &states);
        let result = act_kill(
            &action_context(dir.path(), &ops, false, false, Some(0.4), Some(ROOT.into())),
            &row,
            &by,
        );
        assert_eq!(result.state, "survived");
        assert_eq!(
            result.error.as_deref(),
            Some("could not verify harness after direct SIGKILL: proc hidden")
        );
        assert_eq!(ops.signal_log(), vec![(true, 111, 15), (true, 111, 9)]);
        assert!(!audit::read_trace(dir.path(), TRACE)
            .unwrap()
            .iter()
            .any(|r| matches!(r, Record::End(_))));

        let (dir, _row, ops, _by) =
            orphan_fixture("kill-orphan-unknown-text", 113, 114, Some(115), &states);
        let (code, out, err) = command_kill(dir.path(), &ops, false, false);
        assert_eq!(code, 1, "{}", String::from_utf8_lossy(&err));
        assert!(String::from_utf8(out)
            .unwrap()
            .contains("‹could not verify harness after direct SIGKILL: proc hidden›"));
        assert!(!ops.signal_log().iter().any(|(_, pid, _)| *pid == 115));
        assert!(!audit::read_trace(dir.path(), TRACE)
            .unwrap()
            .iter()
            .any(|r| matches!(r, Record::End(_))));

        let (dir, _row, ops, _by) =
            orphan_fixture("kill-orphan-unknown-json", 116, 117, Some(118), &states);
        let (code, out, err) = command_kill(dir.path(), &ops, false, true);
        assert_eq!(code, 1, "{}", String::from_utf8_lossy(&err));
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(value["targets"][0]["state"], result.state);
        assert_eq!(value["targets"][0]["error"], result.error.clone().unwrap());
        assert_eq!(value["targets"][0]["swept"], result.swept);
        assert!(!ops.signal_log().iter().any(|(_, pid, _)| *pid == 118));
        assert!(!audit::read_trace(dir.path(), TRACE)
            .unwrap()
            .iter()
            .any(|r| matches!(r, Record::End(_))));
    }

    #[test]
    fn kill_supervisor_verification_loss_is_reported_at_the_command_boundary() {
        let make = |label: &str| {
            let dir = TempDir::new(label);
            let supervisor = proc(120);
            let harness = proc(121);
            let by = proc(122);
            let row = action_row(
                dir.path(),
                ROOT,
                Command::Run,
                supervisor.clone(),
                harness.clone(),
                121,
                None,
            );
            let ops = TestOps {
                current: Some(by.clone()),
                ..Default::default()
            };
            ops.set(
                supervisor,
                &[
                    Liveness::Alive,
                    Liveness::Alive,
                    Liveness::Unknown("proc hidden".into()),
                ],
            );
            ops.set(harness, &[Liveness::Alive]);
            (dir, row, ops, by)
        };
        let (dir, row, ops, by) = make("kill-supervisor-unknown-direct");
        let result = act_kill(
            &action_context(dir.path(), &ops, false, false, None, Some(ROOT.into())),
            &row,
            &by,
        );
        assert_eq!(result.state, "survived");
        assert_eq!(
            result.error.as_deref(),
            Some("could not verify supervisor after SIGTERM: proc hidden")
        );
        assert_eq!(ops.signal_log(), vec![(false, 120, 15)]);
        assert!(!audit::read_trace(dir.path(), TRACE)
            .unwrap()
            .iter()
            .any(|r| matches!(r, Record::End(_))));

        let (dir, _row, ops, _by) = make("kill-supervisor-unknown-text");
        let (code, out, _) = command_kill(dir.path(), &ops, false, false);
        assert_eq!(code, 1);
        assert!(String::from_utf8(out)
            .unwrap()
            .contains("‹could not verify supervisor after SIGTERM: proc hidden›"));
        let (dir, _row, ops, _by) = make("kill-supervisor-unknown-json");
        let (code, out, _) = command_kill(dir.path(), &ops, false, true);
        assert_eq!(code, 1);
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(
            value["targets"][0]["error"],
            "could not verify supervisor after SIGTERM: proc hidden"
        );
        assert_eq!(value["targets"][0]["state"], "survived");
        assert_eq!(value["targets"][0]["swept"], 0);
    }

    #[test]
    fn kill_term_grace_verification_loss_stops_before_escalation() {
        let dir = TempDir::new("kill-grace-unknown");
        let supervisor = proc(130);
        let harness = proc(131);
        let by = proc(132);
        let row = action_row(
            dir.path(),
            ROOT,
            Command::Run,
            supervisor.clone(),
            harness.clone(),
            131,
            None,
        );
        let ops = TestOps {
            current: Some(by.clone()),
            ..Default::default()
        };
        ops.set(supervisor, &[Liveness::Dead]);
        ops.set(
            harness.clone(),
            &[
                Liveness::Alive,
                Liveness::Alive,
                Liveness::Unknown("proc hidden".into()),
            ],
        );
        let result = act_kill(
            &action_context(dir.path(), &ops, false, false, Some(1.0), Some(ROOT.into())),
            &row,
            &by,
        );
        assert_eq!(result.state, "survived");
        assert_eq!(
            result.error.as_deref(),
            Some("could not verify harness during SIGTERM grace: proc hidden")
        );
        assert_eq!(ops.signal_log(), vec![(true, 131, 15)]);
        assert!(!audit::read_trace(dir.path(), TRACE)
            .unwrap()
            .iter()
            .any(|r| matches!(r, Record::End(_))));

        let make = |label: &str| {
            let dir = TempDir::new(label);
            let supervisor = proc(133);
            let harness = proc(134);
            let by = proc(135);
            let row = action_row(
                dir.path(),
                ROOT,
                Command::Run,
                supervisor.clone(),
                harness.clone(),
                134,
                None,
            );
            let ops = TestOps {
                current: Some(by.clone()),
                ..Default::default()
            };
            ops.set(supervisor, &[Liveness::Dead]);
            ops.set(
                harness,
                &[
                    Liveness::Alive,
                    Liveness::Alive,
                    Liveness::Unknown("proc hidden".into()),
                ],
            );
            (dir, row, ops, by)
        };
        let (dir, _row, ops, _by) = make("kill-grace-unknown-text");
        let (code, out, err) = command_kill(dir.path(), &ops, false, false);
        assert_eq!(code, 1, "{}", String::from_utf8_lossy(&err));
        assert!(String::from_utf8(out)
            .unwrap()
            .contains("‹could not verify harness during SIGTERM grace: proc hidden›"));
        assert!(!ops
            .signal_log()
            .iter()
            .any(|(_, pid, signal)| *pid == 134 && *signal == 9));
        assert!(!audit::read_trace(dir.path(), TRACE)
            .unwrap()
            .iter()
            .any(|r| matches!(r, Record::End(_))));
        let (dir, _row, ops, _by) = make("kill-grace-unknown-json");
        let (code, out, err) = command_kill(dir.path(), &ops, false, true);
        assert_eq!(code, 1, "{}", String::from_utf8_lossy(&err));
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(value["targets"][0]["state"], result.state);
        assert_eq!(value["targets"][0]["error"], result.error.clone().unwrap());
        assert_eq!(value["targets"][0]["swept"], result.swept);
        assert!(!audit::read_trace(dir.path(), TRACE)
            .unwrap()
            .iter()
            .any(|r| matches!(r, Record::End(_))));
    }

    #[test]
    fn kill_now_confirmation_verification_loss_stops_before_descendant_sweep() {
        let dir = TempDir::new("kill-now-unknown-confirm");
        let supervisor = proc(140);
        let harness = proc(141);
        let descendant = proc(142);
        let by = proc(143);
        let row = action_row(
            dir.path(),
            ROOT,
            Command::Run,
            supervisor.clone(),
            harness.clone(),
            141,
            None,
        );
        let ops = TestOps {
            current: Some(by.clone()),
            ..Default::default()
        };
        ops.set(supervisor, &[Liveness::Alive, Liveness::Alive]);
        ops.set(
            harness.clone(),
            &[
                Liveness::Alive,
                Liveness::Alive,
                Liveness::Unknown("proc hidden".into()),
            ],
        );
        ops.set(descendant.clone(), &[Liveness::Alive]);
        *ops.descendants.lock().unwrap() = VecDeque::from([vec![descendant]]);
        let result = act_kill(
            &action_context(dir.path(), &ops, true, false, None, None),
            &row,
            &by,
        );
        assert_eq!(result.state, "survived");
        assert_eq!(
            result.error.as_deref(),
            Some("could not verify harness after direct SIGKILL: proc hidden")
        );
        assert_eq!(ops.signal_log(), vec![(false, 140, 9), (true, 141, 9)]);
        assert!(!ops.signal_log().iter().any(|(_, pid, _)| *pid == 142));
        assert!(!audit::read_trace(dir.path(), TRACE)
            .unwrap()
            .iter()
            .any(|r| matches!(r, Record::End(_))));

        let make = |label: &str| {
            let dir = TempDir::new(label);
            let supervisor = proc(144);
            let harness = proc(145);
            let descendant = proc(146);
            let by = proc(147);
            let row = action_row(
                dir.path(),
                ROOT,
                Command::Run,
                supervisor.clone(),
                harness.clone(),
                145,
                None,
            );
            let ops = TestOps {
                current: Some(by.clone()),
                ..Default::default()
            };
            ops.set(supervisor, &[Liveness::Alive, Liveness::Alive]);
            ops.set(
                harness,
                &[
                    Liveness::Alive,
                    Liveness::Alive,
                    Liveness::Unknown("proc hidden".into()),
                ],
            );
            ops.set(descendant.clone(), &[Liveness::Alive]);
            *ops.descendants.lock().unwrap() = VecDeque::from([vec![descendant]]);
            (dir, row, ops, by)
        };
        for json in [false, true] {
            let (dir, _row, ops, _by) = make(if json {
                "kill-now-unknown-json"
            } else {
                "kill-now-unknown-text"
            });
            let (code, out, err) = command_kill(dir.path(), &ops, true, json);
            assert_eq!(code, 1, "{}", String::from_utf8_lossy(&err));
            if json {
                let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
                assert_eq!(value["targets"][0]["state"], result.state);
                assert_eq!(value["targets"][0]["error"], result.error.clone().unwrap());
                assert_eq!(value["targets"][0]["swept"], result.swept);
            } else {
                assert!(String::from_utf8(out)
                    .unwrap()
                    .contains("‹could not verify harness after direct SIGKILL: proc hidden›"));
            }
            assert!(!ops.signal_log().iter().any(|(_, pid, _)| *pid == 146));
            assert!(!audit::read_trace(dir.path(), TRACE)
                .unwrap()
                .iter()
                .any(|r| matches!(r, Record::End(_))));
        }
    }

    #[test]
    fn kill_orphan_residual_signal_failure_is_survived_and_audited() {
        let dir = TempDir::new("kill-residual");
        let supervisor = proc(150);
        let harness = proc(151);
        let first = proc(152);
        let second = proc(153);
        let by = proc(154);
        let row = action_row(
            dir.path(),
            ROOT,
            Command::Run,
            supervisor.clone(),
            harness.clone(),
            151,
            None,
        );
        let ops = TestOps {
            current: Some(by.clone()),
            ..Default::default()
        };
        ops.set(supervisor, &[Liveness::Dead]);
        ops.set(harness, &[Liveness::Alive, Liveness::Alive, Liveness::Dead]);
        ops.set(first.clone(), &[Liveness::Alive]);
        ops.set(second.clone(), &[Liveness::Alive]);
        ops.fail_signal(second.pid, &[libc::EPERM]);
        *ops.descendants.lock().unwrap() = VecDeque::from([vec![first, second]]);
        let result = act_kill(
            &action_context(dir.path(), &ops, false, false, Some(0.4), Some(ROOT.into())),
            &row,
            &by,
        );
        let expected = format!(
            "PID 153: {}",
            std::io::Error::from_raw_os_error(libc::EPERM)
        );
        assert_eq!(result.state, "survived");
        assert_eq!(result.swept, 1);
        assert!(result.error.as_deref().unwrap().contains(&expected));
        let direct_error = result.error.clone();
        let direct_swept = result.swept;
        let child_signals = ops
            .signal_log()
            .into_iter()
            .filter(|(_, pid, _)| *pid == 151)
            .collect::<Vec<_>>();
        assert_eq!(child_signals, vec![(true, 151, 15)]);
        let end = audit::read_trace(dir.path(), TRACE)
            .unwrap()
            .into_iter()
            .find_map(|r| match r {
                Record::End(e) => Some(e),
                _ => None,
            })
            .unwrap();
        assert_eq!(end.payload.exit, Some(3));
        assert!(end
            .payload
            .warnings
            .iter()
            .any(|warning| warning.contains(&expected)));
        assert!(end.payload.error.as_deref().unwrap().ends_with(&expected));

        let (dir, _row, ops, _by) = orphan_fixture(
            "kill-residual-json",
            155,
            156,
            Some(152),
            &[Liveness::Alive, Liveness::Alive, Liveness::Dead],
        );
        ops.fail_signal(153, &[libc::EPERM]);
        let first = proc(152);
        let second = proc(153);
        ops.set(first.clone(), &[Liveness::Alive]);
        ops.set(second.clone(), &[Liveness::Alive]);
        *ops.descendants.lock().unwrap() = VecDeque::from([vec![first, second]]);
        let (code, out, err) = command_kill(dir.path(), &ops, false, true);
        assert_eq!(code, 1, "{}", String::from_utf8_lossy(&err));
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(value["targets"][0]["state"], "survived");
        assert_eq!(value["targets"][0]["error"], direct_error.unwrap());
        assert_eq!(value["targets"][0]["swept"], direct_swept);
    }

    #[test]
    fn kill_now_residual_signal_failure_is_survived_and_audited() {
        let dir = TempDir::new("kill-now-residual");
        let supervisor = proc(160);
        let harness = proc(161);
        let first = proc(162);
        let second = proc(163);
        let by = proc(164);
        let row = action_row(
            dir.path(),
            ROOT,
            Command::Run,
            supervisor.clone(),
            harness.clone(),
            161,
            None,
        );
        let ops = TestOps {
            current: Some(by.clone()),
            ..Default::default()
        };
        ops.set(supervisor, &[Liveness::Alive, Liveness::Alive]);
        ops.set(
            harness.clone(),
            &[Liveness::Alive, Liveness::Alive, Liveness::Dead],
        );
        ops.set(first.clone(), &[Liveness::Alive]);
        ops.set(second.clone(), &[Liveness::Alive]);
        ops.fail_signal(second.pid, &[libc::EPERM]);
        *ops.descendants.lock().unwrap() = VecDeque::from([vec![first, second]]);
        let result = act_kill(
            &action_context(dir.path(), &ops, true, false, None, None),
            &row,
            &by,
        );
        let expected = format!(
            "PID 163: {}",
            std::io::Error::from_raw_os_error(libc::EPERM)
        );
        assert_eq!(result.state, "survived");
        assert_eq!(result.swept, 1);
        assert!(result.error.as_deref().unwrap().contains(&expected));
        let direct_error = result.error.clone();
        let direct_swept = result.swept;
        let end = audit::read_trace(dir.path(), TRACE)
            .unwrap()
            .into_iter()
            .find_map(|r| match r {
                Record::End(e) => Some(e),
                _ => None,
            })
            .unwrap();
        assert_eq!(end.payload.exit, Some(3));
        assert!(end.payload.warnings.iter().any(|w| w.contains(&expected)));
        assert!(end.payload.error.as_deref().unwrap().ends_with(&expected));

        let dir = TempDir::new("kill-now-residual-json");
        let supervisor = proc(165);
        let harness = proc(166);
        let first = proc(162);
        let second = proc(163);
        let by = proc(169);
        let _row = action_row(
            dir.path(),
            ROOT,
            Command::Run,
            supervisor.clone(),
            harness.clone(),
            166,
            None,
        );
        let ops = TestOps {
            current: Some(by.clone()),
            ..Default::default()
        };
        ops.set(supervisor, &[Liveness::Alive, Liveness::Alive]);
        ops.set(harness, &[Liveness::Alive, Liveness::Alive, Liveness::Dead]);
        ops.set(first.clone(), &[Liveness::Alive]);
        ops.set(second.clone(), &[Liveness::Alive]);
        ops.fail_signal(second.pid, &[libc::EPERM]);
        *ops.descendants.lock().unwrap() = VecDeque::from([vec![first, second]]);
        let (code, out, err) = command_kill(dir.path(), &ops, true, true);
        assert_eq!(code, 1, "{}", String::from_utf8_lossy(&err));
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(value["targets"][0]["state"], "survived");
        assert_eq!(value["targets"][0]["error"], direct_error.unwrap());
        assert_eq!(value["targets"][0]["swept"], direct_swept);
    }

    #[test]
    fn kill_json_and_preflight_failures_are_side_effect_free() {
        let dir = TempDir::new("kill-json");
        let dead = proc(60);
        let by = proc(61);
        action_row(
            dir.path(),
            ROOT,
            Command::Launch,
            dead.clone(),
            dead.clone(),
            1,
            None,
        );
        let ops = TestOps {
            current: Some(by.clone()),
            ..Default::default()
        };
        ops.set(dead, &[Liveness::Dead]);
        let mut ctx = action_context(dir.path(), &ops, false, false, None, None);
        ctx.trace = Some(TRACE[..8].into());
        ctx.targets = vec![ROOT.into()];
        ctx.json = true;
        let mut out = Vec::new();
        let mut err = Vec::new();
        assert_eq!(kill_with(ctx, &mut out, &mut err).unwrap(), 0);
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        let target = &value["targets"][0];
        assert_eq!(target["inv"], ROOT);
        assert_eq!(target["trace"], TRACE);
        assert_eq!(target["session"], serde_json::Value::Null);
        assert_eq!(target["state"], "lost");
        assert_eq!(target["swept"], 0);
        assert_eq!(target["error"], serde_json::Value::Null);
        assert!(ops.signal_log().is_empty());
        let before = audit::read_trace(dir.path(), TRACE).unwrap().len();
        let mut bad = action_context(dir.path(), &ops, false, false, None, None);
        bad.trace = Some(TRACE[..8].into());
        bad.targets = vec!["missing".into()];
        let mut out = Vec::new();
        let mut err = Vec::new();
        assert_eq!(kill_with(bad, &mut out, &mut err).unwrap(), 2);
        assert!(out.is_empty());
        assert_eq!(audit::read_trace(dir.path(), TRACE).unwrap().len(), before);
        assert!(ops.signal_log().is_empty());
        let mut all = action_context(dir.path(), &ops, false, false, None, None);
        all.trace = None;
        all.targets = vec![];
        all.all = true;
        let mut out = Vec::new();
        let mut err = Vec::new();
        assert_eq!(kill_with(all, &mut out, &mut err).unwrap(), 2);
        assert!(out.is_empty());
    }

    #[test]
    fn subtree_trace_and_all_selection_are_bounded_and_authoritative() {
        let data = trace(vec![
            start(ROOT, None, "2026-01-01T00:00:00Z"),
            start(CHILD, Some(ROOT), "2026-01-01T00:00:01Z"),
            end(CHILD, Some(0), None),
            start(SIBLING, None, "2026-01-01T00:00:02Z"),
            end(SIBLING, Some(0), None),
        ]);
        let subtree = Scope {
            kind: ScopeKind::Subtree,
            source: ScopeSource::Env,
            trace: Some(TRACE.into()),
            root: Some(ROOT.into()),
        };
        let rows = selected(std::slice::from_ref(&data), &subtree);
        assert_eq!(
            rows.iter()
                .map(|r| r.invocation.inv.as_str())
                .collect::<Vec<_>>(),
            vec![CHILD]
        );

        let all = Scope {
            kind: ScopeKind::All,
            source: ScopeSource::None,
            trace: None,
            root: None,
        };
        let rows = selected(std::slice::from_ref(&data), &all);
        assert_eq!(
            rows.iter()
                .map(|r| r.invocation.inv.as_str())
                .collect::<Vec<_>>(),
            vec![ROOT]
        );
        let cycle = trace(vec![
            start(ROOT, Some(CHILD), "2026-01-01T00:00:00Z"),
            start(CHILD, Some(ROOT), "2026-01-01T00:00:01Z"),
        ]);
        let rows = selected(&[cycle], &subtree);
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().any(|row| row.invocation.inv == CHILD));
    }

    #[test]
    fn state_projection_covers_ended_outcomes_without_probing_them() {
        let done = audit_view::join_invocations(&[
            start(ROOT, None, "2026-01-01T00:00:00Z"),
            end(ROOT, Some(0), None),
        ])
        .0
        .remove(0);
        let failed = audit_view::join_invocations(&[
            start(ROOT, None, "2026-01-01T00:00:00Z"),
            end(ROOT, Some(2), None),
        ])
        .0
        .remove(0);
        let killed = audit_view::join_invocations(&[
            start(ROOT, None, "2026-01-01T00:00:00Z"),
            end(ROOT, None, Some(15)),
        ])
        .0
        .remove(0);
        assert_eq!(
            audit_view::invocation_state_from_facts(&done, None, None),
            "done"
        );
        assert_eq!(
            audit_view::invocation_state_from_facts(&failed, None, None),
            "failed"
        );
        assert_eq!(
            audit_view::invocation_state_from_facts(&killed, None, None),
            "killed"
        );
    }

    #[test]
    fn state_projection_covers_live_orphan_lost_stopping_and_legacy() {
        let live = ProcId::current().unwrap();
        let dead = ProcId {
            pid: u32::MAX,
            start: 0,
            boot_id: live.boot_id.clone(),
        };
        let child = |supervisor: ProcId, harness: ProcId, signal: bool| {
            let mut records = vec![
                start(ROOT, None, "2026-01-01T00:00:00Z"),
                Record::Child(crate::audit::ChildRecord {
                    ts: "2026-01-01T00:00:01Z".parse().unwrap(),
                    inv: ROOT.into(),
                    payload: ChildPayload {
                        proc: harness,
                        pgid: 1,
                        supervisor,
                    },
                }),
            ];
            if signal {
                records.push(Record::Signal(SignalRecord {
                    ts: "2026-01-01T00:00:02Z".parse().unwrap(),
                    inv: ROOT.into(),
                    payload: SignalPayload {
                        signal: 15,
                        to: SignalDestination::Supervisor,
                        by: live.clone(),
                        by_inv: None,
                        reason: "test".into(),
                    },
                }));
            }
            audit_view::join_invocations(&records).0.remove(0)
        };
        assert_eq!(
            audit_view::invocation_state_from_facts(
                &child(live.clone(), live.clone(), false),
                Some(audit_view::ProcessFact::Alive),
                Some(audit_view::ProcessFact::Alive)
            ),
            "running"
        );
        assert_eq!(
            audit_view::invocation_state_from_facts(
                &child(live.clone(), live.clone(), true),
                Some(audit_view::ProcessFact::Alive),
                Some(audit_view::ProcessFact::Alive)
            ),
            "stopping"
        );
        assert_eq!(
            audit_view::invocation_state_from_facts(
                &child(dead.clone(), live.clone(), false),
                Some(audit_view::ProcessFact::Dead),
                Some(audit_view::ProcessFact::Alive)
            ),
            "orphaned"
        );
        assert_eq!(
            audit_view::invocation_state_from_facts(
                &child(dead.clone(), dead, false),
                Some(audit_view::ProcessFact::Dead),
                Some(audit_view::ProcessFact::Dead)
            ),
            "lost"
        );
        let mut legacy_record = start(ROOT, None, "2026-01-01T00:00:00Z");
        if let Record::Start(record) = &mut legacy_record {
            record.payload.pid = live.pid;
        }
        let legacy = audit_view::join_invocations(&[legacy_record]).0.remove(0);
        assert_eq!(
            audit_view::invocation_state_from_facts(
                &legacy,
                Some(audit_view::ProcessFact::Alive),
                None
            ),
            "running"
        );
    }

    #[test]
    fn trace_prefix_selection_is_unique_and_ambiguous_selectors_fail() {
        let a = named_trace(
            "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            vec![start(ROOT, None, "2026-01-01T00:00:00Z")],
        );
        let b = named_trace(
            "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            vec![start(ROOT, None, "2026-01-01T00:00:00Z")],
        );
        let env = BTreeMap::new();
        let mut err = Vec::new();
        let selected =
            parse_scope_from_ancestors(&[a.clone(), b.clone()], &env, Some("aaaa"), &[], &mut err)
                .unwrap()
                .unwrap();
        assert_eq!(selected.kind, ScopeKind::Trace);
        assert_eq!(selected.trace.as_deref(), Some(a.entry.id.as_str()));
        let mut err = Vec::new();
        assert!(parse_scope_from_ancestors(
            &[
                named_trace("aaaa1111-aaaa-4aaa-8aaa-aaaaaaaaaaaa", vec![]),
                named_trace("aaaa2222-aaaa-4aaa-8aaa-aaaaaaaaaaaa", vec![])
            ],
            &env,
            Some("aaaa"),
            &[],
            &mut err,
        )
        .unwrap()
        .is_err());
        assert!(String::from_utf8_lossy(&err).contains("ambiguous trace selector"));
    }

    #[test]
    fn ancestry_anchor_wins_over_malformed_or_spoofed_environment() {
        let identity = ProcId::current().unwrap();
        let records = vec![
            start(ROOT, None, "2026-01-01T00:00:00Z"),
            Record::Child(crate::audit::ChildRecord {
                ts: "2026-01-01T00:00:01Z".parse().unwrap(),
                inv: ROOT.into(),
                payload: ChildPayload {
                    proc: identity.clone(),
                    pgid: 1,
                    supervisor: identity.clone(),
                },
            }),
        ];
        let data = trace(records);
        let env: BTreeMap<String, String> = [
            ("CCC_TRACE_ID".into(), "not-a-trace".into()),
            (
                "CCC_PARENT_INVOCATION_ID".into(),
                "not-an-invocation".into(),
            ),
        ]
        .into_iter()
        .collect();
        let mut err = Vec::new();
        let scope = parse_scope_from_ancestors(
            &[data],
            &env,
            None,
            std::slice::from_ref(&identity),
            &mut err,
        )
        .unwrap()
        .unwrap();
        assert_eq!(scope.source, ScopeSource::Ancestry);
        assert_eq!(scope.root.as_deref(), Some(ROOT));
        assert!(String::from_utf8_lossy(&err).contains("variables were malformed"));
    }

    #[test]
    fn opaque_claude_project_names_supply_advisory_idle_seconds() {
        let home = TempDir::new("status-idle");
        let project = home.path().join(".claude/projects/opaque._project");
        std::fs::create_dir_all(&project).unwrap();
        let session = "session-opaque";
        std::fs::write(project.join(format!("{session}.jsonl")), "not parsed").unwrap();
        let env: BTreeMap<String, String> = [("HOME".into(), home.path().display().to_string())]
            .into_iter()
            .collect();
        let now = Timestamp::now();
        let cfg = Config::default();
        let mut index = IdleIndex::new(&cfg, &env);
        let idle = index.idle_for(IdleQuery {
            cfg: &cfg,
            env: &env,
            harness: Harness::Claude,
            profile: None,
            auth: Some(ClaudeAuth::Token),
            session: Some(session),
            started: None,
            now,
        });
        assert!(idle.is_some());
    }

    #[test]
    fn codex_idle_scans_adjacent_dates_and_compressed_rollouts() {
        let home = TempDir::new("status-codex-idle");
        let dir = home.path().join(".codex/sessions/2026/01/03");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("rollout-2026-01-03T05-41-02-z.jsonl.zst"),
            "opaque",
        )
        .unwrap();
        let env: BTreeMap<String, String> = [("HOME".into(), home.path().display().to_string())]
            .into_iter()
            .collect();
        let cfg = Config::default();
        let mut index = IdleIndex::new(&cfg, &env);
        let idle = index.idle_for(IdleQuery {
            cfg: &cfg,
            env: &env,
            harness: Harness::Codex,
            profile: None,
            auth: None,
            session: Some("z"),
            started: Some("2026-01-02T12:00:00Z".parse().unwrap()),
            now: "2026-01-02T12:00:00Z".parse().unwrap(),
        });
        assert!(idle.is_some());
    }

    #[test]
    fn idle_index_reuses_shared_claude_roots_and_codex_dates() {
        let home = TempDir::new("status-idle-index");
        let configured = home.path().join("configured/projects/project");
        let fallback = home.path().join(".claude/projects/project");
        std::fs::create_dir_all(&configured).unwrap();
        std::fs::create_dir_all(&fallback).unwrap();
        std::fs::write(configured.join("configured-session.jsonl"), "x").unwrap();
        std::fs::write(fallback.join("fallback-session.jsonl"), "x").unwrap();
        let codex = home.path().join(".codex/sessions/2026/01/02");
        std::fs::create_dir_all(&codex).unwrap();
        std::fs::write(codex.join("rollout-2026-01-02T05-41-02-codex-a.jsonl"), "x").unwrap();
        std::fs::write(
            codex.join("rollout-2026-01-02T05-42-02-codex-b.jsonl.zst"),
            "x",
        )
        .unwrap();
        let env: BTreeMap<String, String> = [
            ("HOME".into(), home.path().display().to_string()),
            (
                "CLAUDE_CONFIG_DIR".into(),
                home.path().join("configured").display().to_string(),
            ),
        ]
        .into_iter()
        .collect();
        let now = Timestamp::now();
        let cfg = Config::default();
        let mut index = IdleIndex::new(&cfg, &env);
        assert!(index
            .idle_for(IdleQuery {
                cfg: &cfg,
                env: &env,
                harness: Harness::Claude,
                profile: None,
                auth: None,
                session: Some("configured-session"),
                started: None,
                now
            })
            .is_some());
        assert!(index
            .idle_for(IdleQuery {
                cfg: &cfg,
                env: &env,
                harness: Harness::Claude,
                profile: None,
                auth: None,
                session: Some("fallback-session"),
                started: None,
                now
            })
            .is_some());
        assert_eq!(index.reads(&home.path().join("configured")), 1);
        assert_eq!(index.reads(&home.path().join(".claude")), 1);
        let started = "2026-01-02T12:00:00Z".parse().unwrap();
        assert!(index
            .idle_for(IdleQuery {
                cfg: &cfg,
                env: &env,
                harness: Harness::Codex,
                profile: None,
                auth: None,
                session: Some("codex-a"),
                started: Some(started),
                now
            })
            .is_some());
        assert!(index
            .idle_for(IdleQuery {
                cfg: &cfg,
                env: &env,
                harness: Harness::Codex,
                profile: None,
                auth: None,
                session: Some("codex-b"),
                started: Some(started),
                now
            })
            .is_some());
        assert_eq!(index.reads(&codex), 1);
    }

    #[test]
    fn a_null_start_agent_renders_the_spawn_agent_or_a_dash() {
        let dir = TempDir::new("status-roleless");
        let mut start = start_kind(ROOT, None, "2026-01-01T00:00:00Z", Command::Launch);
        if let Record::Start(s) = &mut start {
            s.payload.agent = None;
        }
        let env = BTreeMap::new();
        for (spawn_agent, shown, json) in [
            (None, "-", true),
            (None, "-", false),
            (Some("worker"), "worker", true),
            (Some("worker"), "worker", false),
        ] {
            let mut spawn = serde_json::json!({
                "t":"spawn", "ts":"2026-01-01T00:00:01Z", "inv":ROOT, "harness":"claude",
                "model":"claude-opus-5-5[1M]", "effort":null, "profile":null, "auth":null,
                "sandbox":"read-only", "tools":null, "cwd":"/work", "session":"s-1",
                "selection_notes":[]
            });
            if let Some(agent) = spawn_agent {
                spawn["agent"] = agent.into();
            }
            let text = serde_json::to_string(&start).unwrap() + "\n" + &spawn.to_string() + "\n";
            std::fs::write(dir.path().join(format!("{TRACE}.jsonl")), text).unwrap();
            let mut out = Vec::new();
            let mut err = Vec::new();
            status_with(
                StatusContext {
                    dir: dir.path(),
                    cfg: &Config::default(),
                    env: &env,
                    now: "2026-01-01T00:02:00Z".parse().unwrap(),
                    width: 120,
                    trace: Some(TRACE[..8].into()),
                    json,
                },
                &mut out,
                &mut err,
            )
            .unwrap();
            if json {
                let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
                assert_eq!(value["invocations"][0]["agent"], shown, "{value}");
                assert_eq!(value["invocations"][0]["model"], "claude-opus-5-5[1M]");
            } else {
                let text = String::from_utf8(out).unwrap();
                assert!(text.contains(&format!(" launch {shown} ")), "{text}");
            }
        }
    }

    #[test]
    fn status_json_keeps_scope_and_legacy_process_nullability() {
        let dir = TempDir::new("status-json");
        let text =
            serde_json::to_string(&start(ROOT, None, "2026-01-01T00:00:00Z")).unwrap() + "\n";
        std::fs::write(dir.path().join(format!("{TRACE}.jsonl")), text).unwrap();
        let env = BTreeMap::new();
        let mut out = Vec::new();
        let mut err = Vec::new();
        assert_eq!(
            status_with(
                StatusContext {
                    dir: dir.path(),
                    cfg: &Config::default(),
                    env: &env,
                    now: "2026-01-01T00:02:00Z".parse().unwrap(),
                    width: 120,
                    trace: Some(TRACE[..8].into()),
                    json: true,
                },
                &mut out,
                &mut err,
            )
            .unwrap(),
            0
        );
        let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(value["scope"]["kind"], "trace");
        assert_eq!(value["invocations"][0]["supervisor"]["verified"], false);
        assert!(value["invocations"][0]["harness_proc"].is_null());
    }
}
