//! Discovery and presentation of invocation audit traces.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use jiff::{tz::TimeZone, Timestamp};
use serde::Serialize;
use uuid::Uuid;

use crate::audit::{self, Command, IndexLine, Record};
use crate::config::Harness;

#[derive(Debug, Clone, Default)]
pub struct LogOptions {
    pub trace: Option<String>,
    pub latest: bool,
    pub flat: bool,
    pub verbose: bool,
    pub json: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct Entry {
    pub(crate) id: String,
    pub(crate) index: Option<IndexLine>,
    pub(crate) exists: bool,
    pub(crate) start: Option<audit::StartRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TraceJson {
    pub trace: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<IndexLine>,
    pub invocations: Vec<InvocationJson>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InvocationJson {
    pub inv: String,
    pub parent: Option<String>,
    pub start: Option<Record>,
    pub spawn: Option<Record>,
    pub child: Option<Record>,
    pub sessions: Vec<Record>,
    pub signals: Vec<Record>,
    pub turns: Vec<Record>,
    pub end: Option<Record>,
}

#[derive(Debug, Clone, Copy)]
struct InvocationMeta {
    first: usize,
    earliest: Option<Timestamp>,
}

type InvocationGroup = (
    InvocationMeta,
    Option<Record>,
    Option<Record>,
    Option<Record>,
    Vec<Record>,
    Vec<Record>,
    Vec<Record>,
    Option<Record>,
);

fn diag(err: &mut dyn Write, message: impl std::fmt::Display) -> Result<()> {
    writeln!(err, "ccc: {message}")?;
    Ok(())
}

fn record_ts(r: &Record) -> Timestamp {
    match r {
        Record::Start(x) => x.ts,
        Record::Spawn(x) => x.ts,
        Record::Turn(x) => x.ts,
        Record::End(x) => x.ts,
        Record::Child(x) => x.ts,
        Record::Session(x) => x.ts,
        Record::Signal(x) => x.ts,
    }
}

fn record_inv(r: &Record) -> &str {
    match r {
        Record::Start(x) => &x.inv,
        Record::Spawn(x) => &x.inv,
        Record::Turn(x) => &x.inv,
        Record::End(x) => &x.inv,
        Record::Child(x) => &x.inv,
        Record::Session(x) => &x.inv,
        Record::Signal(x) => &x.inv,
    }
}

fn record_kind(r: &Record) -> &'static str {
    match r {
        Record::Start(_) => "start",
        Record::Spawn(_) => "spawn",
        Record::Turn(_) => "turn",
        Record::End(_) => "end",
        Record::Child(_) => "child",
        Record::Session(_) => "session",
        Record::Signal(_) => "signal",
    }
}

pub fn join_invocations(records: &[Record]) -> (Vec<InvocationJson>, Vec<String>) {
    let mut groups: BTreeMap<String, InvocationGroup> = BTreeMap::new();
    let mut duplicates = BTreeSet::new();
    for (pos, record) in records.iter().enumerate() {
        let inv = record_inv(record).to_string();
        let entry = groups.entry(inv.clone()).or_insert_with(|| {
            (
                InvocationMeta {
                    first: pos,
                    earliest: None,
                },
                None,
                None,
                None,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                None,
            )
        });
        let meta = &mut entry.0;
        meta.earliest = Some(
            meta.earliest
                .map_or(record_ts(record), |t| t.min(record_ts(record))),
        );
        match record {
            Record::Start(_x) => {
                if entry.1.is_some() {
                    duplicates.insert((inv.clone(), "start"));
                } else {
                    entry.1 = Some(record.clone());
                }
            }
            Record::Spawn(_) => {
                if entry.2.is_some() {
                    duplicates.insert((inv.clone(), "spawn"));
                } else {
                    entry.2 = Some(record.clone());
                }
            }
            Record::Turn(_) => entry.6.push(record.clone()),
            Record::End(_) => {
                if entry.7.is_some() {
                    duplicates.insert((inv.clone(), "end"));
                } else {
                    entry.7 = Some(record.clone());
                }
            }
            Record::Child(_) => {
                if entry.3.is_some() {
                    duplicates.insert((inv.clone(), "child"));
                } else {
                    entry.3 = Some(record.clone());
                }
            }
            Record::Session(_) => entry.4.push(record.clone()),
            Record::Signal(_) => entry.5.push(record.clone()),
        }
    }
    let mut out = groups
        .into_values()
        .map(
            |(_meta, start, spawn, child, mut sessions, mut signals, mut turns, end)| {
                sessions.sort_by_key(record_ts);
                signals.sort_by_key(record_ts);
                turns.sort_by_key(record_ts);
                let parent = start.as_ref().and_then(|r| match r {
                    Record::Start(s) => s.parent.clone(),
                    _ => None,
                });
                let invocation = InvocationJson {
                    inv: start
                        .as_ref()
                        .or(spawn.as_ref())
                        .or(child.as_ref())
                        .or(turns.first())
                        .or(sessions.first())
                        .or(signals.first())
                        .or(end.as_ref())
                        .map(record_inv)
                        .unwrap_or("")
                        .to_string(),
                    parent,
                    start,
                    spawn,
                    child,
                    sessions,
                    signals,
                    turns,
                    end,
                };
                (_meta, invocation)
            },
        )
        .collect::<Vec<_>>();
    out.sort_by_key(|(meta, _)| (meta.earliest, meta.first));
    let diagnostics = duplicates
        .into_iter()
        .map(|(inv, kind)| format!("duplicate {kind} record for invocation {inv}"))
        .collect();
    let out = out.into_iter().map(|(_, i)| i).collect();
    (out, diagnostics)
}

pub fn render_json(
    trace: &str,
    index: Option<&IndexLine>,
    records: &[Record],
) -> Result<(String, Vec<String>)> {
    let (invocations, diagnostics) = join_invocations(records);
    Ok((
        serde_json::to_string_pretty(&TraceJson {
            trace: trace.to_string(),
            index: index.cloned(),
            invocations,
        })?,
        diagnostics,
    ))
}

fn valid_id(name: &str) -> Option<String> {
    name.strip_suffix(".jsonl")
        .filter(|s| Uuid::parse_str(s).is_ok())
        .map(str::to_string)
}

pub(crate) fn load_catalog(dir: &Path, err: &mut dyn Write) -> Result<Vec<Entry>> {
    let index_path = dir.join("traces.jsonl");
    let indexes = match audit::read_index_with_diag(dir, Some(err)) {
        Ok(v) => v,
        // TODO(log-missing-index-is-reported-as-an-error): a NotFound here is the empty log before the first run and should be silent; see TODO.md.
        Err(e) => {
            diag(err, format_args!("{}: {e:#}", index_path.display()))?;
            Vec::new()
        }
    };
    let mut files = BTreeSet::new();
    if dir.exists() {
        for item in fs::read_dir(dir)
            .with_context(|| format!("reading audit directory {}", dir.display()))?
        {
            let item = item?;
            if item.file_type()?.is_file() {
                if let Some(name) = item.file_name().to_str().and_then(valid_id) {
                    files.insert(name);
                }
            }
        }
    }
    let mut indexed = indexes
        .into_iter()
        .enumerate()
        .map(|(n, x)| (x.trace.clone(), x, n))
        .collect::<Vec<_>>();
    indexed.sort_by(|a, b| a.1.ts.cmp(&b.1.ts).then(a.2.cmp(&b.2)));
    let mut used = HashSet::new();
    let mut result = Vec::new();
    for (id, index, _) in indexed {
        used.insert(id.clone());
        let exists = files.contains(&id);
        result.push(Entry {
            id,
            index: Some(index),
            exists,
            start: None,
        });
    }
    for id in files.into_iter().filter(|id| !used.contains(id)) {
        let path = dir.join(format!("{id}.jsonl"));
        let start = first_start(&path, err)?;
        result.push(Entry {
            id,
            index: None,
            exists: true,
            start,
        });
    }
    Ok(result)
}

fn first_start(path: &Path, err: &mut dyn Write) -> Result<Option<audit::StartRecord>> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) => {
            diag(err, format_args!("{}: {e:#}", path.display()))?;
            return Ok(None);
        }
    };
    first_start_from_reader(BufReader::new(file), path, err)
}

fn first_start_from_reader<R: BufRead>(
    mut reader: R,
    path: &Path,
    err: &mut dyn Write,
) -> Result<Option<audit::StartRecord>> {
    let mut line = Vec::new();
    loop {
        line.clear();
        match reader.read_until(b'\n', &mut line) {
            Ok(0) => return Ok(None),
            Ok(_) if line.iter().all(u8::is_ascii_whitespace) => continue,
            Ok(_) => break,
            Err(e) => {
                diag(err, format_args!("{}: {e:#}", path.display()))?;
                return Ok(None);
            }
        }
    }
    let record = match serde_json::from_slice::<Record>(&line) {
        Ok(record) => record,
        Err(e) => {
            diag(
                err,
                format_args!("{}: malformed JSONL: {e}", path.display()),
            )?;
            return Ok(None);
        }
    };
    Ok(match record {
        Record::Start(start) => Some(start),
        _ => None,
    })
}

pub(crate) fn sanitize(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{7f}' => out.push_str("\\x7f"),
            '\u{1b}' => out.push_str("\\x1b"),
            c if c.is_control() => out.push_str(&format!("\\u{{{:04x}}}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

pub(crate) fn short(id: &str, verbose: bool) -> String {
    let id = sanitize(id);
    if verbose {
        id
    } else {
        id.chars().take(8).collect()
    }
}

pub(crate) fn time(ts: Timestamp, fmt: &str) -> String {
    let z = ts.to_zoned(TimeZone::system());
    z.strftime(fmt).to_string()
}

pub(crate) fn compact(ms: u64) -> String {
    let sec = ms / 1000;
    if sec == 0 {
        "<1s".into()
    } else if sec < 60 {
        format!("{sec}s")
    } else if sec < 3600 {
        format!("{}m{:02}s", sec / 60, sec % 60)
    } else {
        format!("{}h{:02}m", sec / 3600, (sec / 60) % 60)
    }
}

pub(crate) fn elapsed(now: Timestamp, then: Timestamp) -> String {
    let ms = u64::try_from(now.duration_since(then).as_millis()).unwrap_or(0);
    compact(ms)
}

fn home_contract(s: &str, home: Option<&Path>) -> String {
    if let Some(home) = home.and_then(|p| p.to_str()) {
        if s == home {
            return "~".into();
        }
        if let Some(rest) = s.strip_prefix(&(home.to_string() + "/")) {
            return format!("~/{}", sanitize(rest));
        }
    }
    sanitize(s)
}

fn path_text(path: &str, exists: bool, home: Option<&Path>) -> String {
    home_contract(path, home) + if exists { "" } else { " (missing)" }
}

/// The agents-config path for display, or `None` for an index line written
/// before the agents config was recorded.
fn agents_config_text(i: &IndexLine, home: Option<&Path>) -> Option<String> {
    (!i.agents_config.is_empty()).then(|| path_text(&i.agents_config, i.agents_config_exists, home))
}

fn env_text(i: &IndexLine) -> String {
    let mut v = i
        .env
        .iter()
        .map(|(k, v)| format!("{}={}", sanitize(k), sanitize(v)))
        .collect::<Vec<_>>();
    v.extend(
        i.env_other
            .iter()
            .map(|k| format!("{}=<unrecognized>", sanitize(k))),
    );
    if v.is_empty() {
        "-".into()
    } else {
        v.join(", ")
    }
}

fn render_index(entries: &[Entry], verbose: bool, home: Option<&Path>) -> String {
    let mut out = String::from("TRACE STARTED ROOT CWD CONFIG AGENTS ENV\n");
    for e in entries {
        let (started, root, cwd, config, agents, env) = if let Some(i) = &e.index {
            (
                time(i.ts, "%m-%d %H:%M:%S"),
                format!(
                    "{} {}",
                    match i.cmd {
                        Command::Run => "run",
                        Command::Resume => "resume",
                        Command::Launch => "launch",
                    },
                    sanitize(i.agent.as_deref().unwrap_or("-"))
                ),
                home_contract(&i.cwd, home),
                path_text(&i.config, i.config_exists, home),
                agents_config_text(i, home).unwrap_or_else(|| "-".into()),
                env_text(i),
            )
        } else if let Some(s) = &e.start {
            (
                time(s.ts, "%m-%d %H:%M:%S"),
                format!(
                    "{} {} (unindexed)",
                    match s.payload.cmd {
                        Command::Run => "run",
                        Command::Resume => "resume",
                        Command::Launch => "launch",
                    },
                    sanitize(s.payload.agent.as_deref().unwrap_or("-"))
                ),
                home_contract(s.payload.cwd_arg.as_deref().unwrap_or("-"), home),
                "-".into(),
                "-".into(),
                "-".into(),
            )
        } else {
            (
                "-".into(),
                "-".into(),
                "-".into(),
                "-".into(),
                "-".into(),
                "-".into(),
            )
        };
        out.push_str(&format!(
            "{} {started} {root} {cwd} {config} {agents} {env}\n",
            short(&e.id, verbose)
        ));
    }
    out
}

fn prompt_line(s: Option<&str>) -> String {
    let line = s
        .unwrap_or("")
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    sanitize(line)
}
pub(crate) fn clip(s: &str, width: usize) -> String {
    let s = sanitize(s);
    if s.chars().count() <= width {
        s
    } else {
        s.chars().take(width.saturating_sub(1)).collect::<String>() + "…"
    }
}

pub(crate) fn start_of(i: &InvocationJson) -> Option<&audit::StartRecord> {
    i.start.as_ref().and_then(|r| match r {
        Record::Start(s) => Some(s),
        _ => None,
    })
}

pub(crate) fn spawn_of(i: &InvocationJson) -> Option<&audit::SpawnRecord> {
    i.spawn.as_ref().and_then(|r| match r {
        Record::Spawn(s) => Some(s),
        _ => None,
    })
}

pub(crate) fn end_of(i: &InvocationJson) -> Option<&audit::EndRecord> {
    i.end.as_ref().and_then(|r| match r {
        Record::End(s) => Some(s),
        _ => None,
    })
}

pub(crate) fn effective_session(i: &InvocationJson) -> Option<&str> {
    end_of(i)
        .and_then(|e| e.payload.session.as_deref())
        .or_else(|| {
            i.turns.iter().rev().find_map(|r| match r {
                Record::Turn(t) => Some(t.session.as_str()),
                _ => None,
            })
        })
        .or_else(|| {
            i.sessions.iter().rev().find_map(|r| match r {
                Record::Session(s) => Some(s.session.as_str()),
                _ => None,
            })
        })
        .or_else(|| spawn_of(i).and_then(|s| s.payload.session.as_deref()))
}

pub(crate) trait ProcessProbe {
    fn identity_liveness(&self, process: &crate::procs::ProcId) -> crate::procs::Liveness;
    fn pid_presence(&self, pid: u32) -> crate::procs::Presence;
}

struct HostProcessProbe;

impl ProcessProbe for HostProcessProbe {
    fn identity_liveness(&self, process: &crate::procs::ProcId) -> crate::procs::Liveness {
        process.alive()
    }

    fn pid_presence(&self, pid: u32) -> crate::procs::Presence {
        crate::procs::presence(pid)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ProcessFact {
    Alive,
    Dead,
    Unknown,
}

pub(crate) fn invocation_state_from_facts(
    i: &InvocationJson,
    supervisor: Option<ProcessFact>,
    harness: Option<ProcessFact>,
) -> &'static str {
    if let Some(end) = end_of(i) {
        if end.payload.exit == Some(0) {
            return "done";
        }
        if end.payload.signal.is_some() || !i.signals.is_empty() {
            return "killed";
        }
        return "failed";
    }
    let sup = supervisor.map(fact_value).unwrap_or(-1);
    let child = harness.map(fact_value).unwrap_or(0);
    if sup < 0 {
        return "unknown";
    }
    if sup == 1 && child == 1 {
        if !i.signals.is_empty() {
            "stopping"
        } else {
            "running"
        }
    } else if sup == 1 {
        "running"
    } else if child == 1 {
        "orphaned"
    } else {
        "lost"
    }
}

fn fact_value(fact: ProcessFact) -> i8 {
    match fact {
        ProcessFact::Alive => 1,
        ProcessFact::Dead => 0,
        ProcessFact::Unknown => -1,
    }
}

fn invocation_state_with(i: &InvocationJson, probe: &dyn ProcessProbe) -> &'static str {
    let supervisor = i
        .child
        .as_ref()
        .and_then(|r| match r {
            Record::Child(c) => Some(match probe.identity_liveness(&c.payload.supervisor) {
                crate::procs::Liveness::Alive => ProcessFact::Alive,
                crate::procs::Liveness::Dead => ProcessFact::Dead,
                crate::procs::Liveness::Unknown(_) => ProcessFact::Unknown,
            }),
            _ => None,
        })
        .or_else(|| {
            start_of(i).map(|s| match probe.pid_presence(s.payload.pid) {
                crate::procs::Presence::Present => ProcessFact::Alive,
                crate::procs::Presence::Absent => ProcessFact::Dead,
                crate::procs::Presence::Unknown(_) => ProcessFact::Unknown,
            })
        });
    let harness = i.child.as_ref().and_then(|r| match r {
        Record::Child(c) => Some(match probe.identity_liveness(&c.payload.proc) {
            crate::procs::Liveness::Alive => ProcessFact::Alive,
            crate::procs::Liveness::Dead => ProcessFact::Dead,
            crate::procs::Liveness::Unknown(_) => ProcessFact::Unknown,
        }),
        _ => None,
    });
    invocation_state_from_facts(i, supervisor, harness)
}

fn result_or_error(i: &InvocationJson) -> Option<&str> {
    end_of(i)
        .and_then(|e| e.payload.error.as_deref().or(e.payload.result.as_deref()))
        .or_else(|| {
            i.turns.iter().rev().find_map(|r| match r {
                Record::Turn(t) => t.result.as_deref(),
                _ => None,
            })
        })
}

fn append_verbose(out: &mut String, label: &str, text: Option<&str>, indent: &str) {
    if let Some(text) = text {
        out.push_str(&format!("{indent}{label}\n"));
        for line in text.lines() {
            out.push_str(&format!("{indent}  {}\n", sanitize(line)));
        }
    }
}

fn invocation_line(
    i: &InvocationJson,
    all: &[InvocationJson],
    now: Timestamp,
    verbose: bool,
    width: usize,
    detail: &mut String,
    probe: &dyn ProcessProbe,
) -> String {
    let start = start_of(i);
    let spawn = spawn_of(i);
    let end = end_of(i);
    let cmd = start
        .map(|s| match s.payload.cmd {
            Command::Run => "run",
            Command::Resume => "resume",
            Command::Launch => "launch",
        })
        .unwrap_or("incomplete");
    let agent = start
        .and_then(|s| s.payload.agent.as_deref())
        .or_else(|| spawn.and_then(|s| s.payload.agent.as_deref()))
        .unwrap_or("-");
    let session = effective_session(i).unwrap_or("-");
    let status = if let Some(e) = end {
        if let Some(n) = e.payload.exit {
            format!("exit {n} {}", compact(e.duration_ms))
        } else if let Some(n) = e.payload.signal {
            format!("signal {n} {}", compact(e.duration_ms))
        } else {
            format!("done {}", compact(e.duration_ms))
        }
    } else {
        format!(
            "{} {}",
            invocation_state_with(i, probe),
            start
                .map(|s| elapsed(now, s.ts))
                .unwrap_or_else(|| "<1s".to_string())
        )
    };
    let turns = if cmd == "launch" {
        format!(
            " {} turns",
            end.and_then(|e| e.payload.turns).unwrap_or(i.turns.len())
        )
    } else {
        String::new()
    };
    let routing = spawn
        .map(|s| {
            format!(
                " {} {} {} {}",
                s.payload.harness,
                s.payload.model,
                s.payload
                    .effort
                    .map(|x| x.to_string())
                    .unwrap_or_else(|| "-".into()),
                s.payload.profile.as_deref().unwrap_or("-")
            )
        })
        .unwrap_or_default();
    let label = if spawn.map(|s| s.payload.harness) == Some(Harness::Codex) {
        "thread"
    } else {
        "session"
    };
    let mut line = format!(
        "{} {} {} {} {}{} {} {}={}",
        short(&i.inv, verbose),
        cmd,
        sanitize(agent),
        start
            .map(|s| time(s.ts, "%H:%M:%S"))
            .unwrap_or_else(|| "-".into()),
        status,
        turns,
        sanitize(&routing),
        label,
        sanitize(session)
    );
    if let Some(parent) = start.and_then(|s| s.payload.session.as_deref()) {
        let same = all
            .iter()
            .filter(|other| other.inv != i.inv && effective_session(other) == Some(parent))
            .min_by_key(|other| {
                start_of(other)
                    .map(|s| s.ts)
                    .or_else(|| other.turns.iter().map(record_ts).min())
                    .unwrap_or(Timestamp::MAX)
            });
        let label = if start.is_some_and(|s| s.payload.fork) {
            "forks"
        } else {
            "resumes"
        };
        match same {
            Some(other) => line.push_str(&format!(" ({label} {})", short(&other.inv, verbose))),
            None => line.push_str(" (session created outside this trace)"),
        }
    }
    if !verbose {
        if cmd != "launch" {
            if let Some(prompt) = start.and_then(|s| s.payload.prompt.as_deref()) {
                line.push_str(&format!(
                    " ›{}",
                    clip(
                        &prompt_line(Some(prompt)),
                        width.saturating_sub(line.chars().count() + 3)
                    )
                ));
            }
        }
        if let Some(text) = result_or_error(i) {
            line.push_str(&format!(
                " ‹{}›",
                clip(
                    &prompt_line(Some(text)),
                    width.saturating_sub(line.chars().count() + 3)
                )
            ));
        }
    } else {
        if let Some(s) = start {
            detail.push_str(&format!("  pid={}\n", s.payload.pid));
            append_verbose(detail, "  prompt:", s.payload.prompt.as_deref(), "  ");
        }
        if cmd == "launch" {
            for r in &i.turns {
                if let Record::Turn(t) = r {
                    detail.push_str(&format!(
                        "  turn {} {} session={}{}\n",
                        time(t.ts, "%H:%M:%S"),
                        sanitize(t.turn.as_deref().unwrap_or("-")),
                        sanitize(&t.session),
                        t.transcript
                            .as_deref()
                            .map(|p| format!(" transcript={}", sanitize(p)))
                            .unwrap_or_default()
                    ));
                    append_verbose(detail, "    result:", t.result.as_deref(), "    ");
                }
            }
        }
        append_verbose(detail, "  result/error:", result_or_error(i), "  ");
        if let Some(s) = spawn {
            for note in &s.payload.selection_notes {
                detail.push_str(&format!("  selection: {}\n", sanitize(note)));
            }
        }
        if let Some(e) = end {
            for warning in &e.payload.warnings {
                detail.push_str(&format!("  warning: {}\n", sanitize(warning)));
            }
            if let Some(native) = &e.payload.native {
                detail.push_str(&format!("  native: {}\n", sanitize(&native.to_string())));
            }
        }
    }
    if !verbose {
        line = clip(&line, width);
    }
    line
}

pub fn render_tree(
    trace: &str,
    index: Option<&IndexLine>,
    records: &[Record],
    now: Timestamp,
    verbose: bool,
    width: usize,
    home: Option<&Path>,
) -> (String, Vec<String>) {
    render_tree_with_probe(
        trace,
        index,
        records,
        now,
        verbose,
        width,
        home,
        &HostProcessProbe,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn render_tree_with_probe(
    trace: &str,
    index: Option<&IndexLine>,
    records: &[Record],
    now: Timestamp,
    verbose: bool,
    width: usize,
    home: Option<&Path>,
    probe: &dyn ProcessProbe,
) -> (String, Vec<String>) {
    let (invocations, diagnostics) = join_invocations(records);
    let mut out = String::new();
    if let Some(i) = index {
        out.push_str(&format!(
            "trace {} root {} {} started {} config {}{} env {}\n",
            sanitize(trace),
            match i.cmd {
                Command::Run => "run",
                Command::Resume => "resume",
                Command::Launch => "launch",
            },
            sanitize(i.agent.as_deref().unwrap_or("-")),
            time(i.ts, "%Y-%m-%d %H:%M:%S"),
            path_text(&i.config, i.config_exists, home),
            agents_config_text(i, home)
                .map(|agents| format!(" agents {agents}"))
                .unwrap_or_default(),
            env_text(i)
        ));
    } else {
        out.push_str(&format!("trace {} (unindexed)\n", sanitize(trace)));
    }
    let mut children: HashMap<Option<String>, Vec<usize>> = HashMap::new();
    for (n, i) in invocations.iter().enumerate() {
        children.entry(i.parent.clone()).or_default().push(n);
    }
    let mut seen = HashSet::new();
    #[allow(clippy::too_many_arguments)]
    fn walk(
        out: &mut String,
        ids: &[usize],
        children: &HashMap<Option<String>, Vec<usize>>,
        inv: &[InvocationJson],
        seen: &mut HashSet<String>,
        now: Timestamp,
        verbose: bool,
        width: usize,
        prefix: &str,
        probe: &dyn ProcessProbe,
    ) {
        for (pos, n) in ids.iter().enumerate() {
            if *n >= inv.len() || !seen.insert(inv[*n].inv.clone()) {
                continue;
            }
            let last = pos + 1 == ids.len();
            let branch = if last { "└─ " } else { "├─ " };
            out.push_str(prefix);
            out.push_str(branch);
            let mut detail = String::new();
            out.push_str(&invocation_line(
                &inv[*n],
                inv,
                now,
                verbose,
                width.saturating_sub(prefix.chars().count() + 3),
                &mut detail,
                probe,
            ));
            out.push('\n');
            out.push_str(&detail);
            let next = format!("{prefix}{}", if last { "   " } else { "│  " });
            if let Some(kids) = children.get(&Some(inv[*n].inv.clone())) {
                walk(
                    out, kids, children, inv, seen, now, verbose, width, &next, probe,
                );
            }
        }
    }
    let ids: HashSet<String> = invocations.iter().map(|i| i.inv.clone()).collect();
    let roots = children.get(&None).cloned().unwrap_or_default();
    let mut orphans: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (n, i) in invocations.iter().enumerate() {
        if let Some(parent) = i.parent.as_ref().filter(|p| !ids.contains(*p)) {
            orphans.entry(parent.clone()).or_default().push(n);
        }
    }
    walk(
        &mut out,
        &roots,
        &children,
        &invocations,
        &mut seen,
        now,
        verbose,
        width,
        "",
        probe,
    );
    for (parent, orphan_ids) in &orphans {
        out.push_str(&format!("?? unknown parent {}\n", short(parent, verbose)));
        walk(
            &mut out,
            orphan_ids,
            &children,
            &invocations,
            &mut seen,
            now,
            verbose,
            width,
            "",
            probe,
        );
    }
    for i in &invocations {
        if !seen.contains(&i.inv) {
            let mut detail = String::new();
            out.push_str(&format!(
                "?? invalid parent chain\n└─ {}\n",
                invocation_line(
                    i,
                    &invocations,
                    now,
                    verbose,
                    width.saturating_sub(3),
                    &mut detail,
                    probe,
                )
            ));
            out.push_str(&detail);
        }
    }
    (out, diagnostics)
}

pub fn render_flat(records: &[Record], verbose: bool, width: usize) -> (String, Vec<String>) {
    let (inv, diagnostics) = join_invocations(records);
    let parents: HashMap<String, Option<String>> = inv
        .iter()
        .map(|i| (i.inv.clone(), i.parent.clone()))
        .collect();
    fn depth(
        id: &str,
        parents: &HashMap<String, Option<String>>,
        memo: &mut HashMap<String, usize>,
        visiting: &mut HashSet<String>,
    ) -> usize {
        if let Some(d) = memo.get(id) {
            return *d;
        }
        if !visiting.insert(id.to_string()) {
            return usize::MAX;
        }
        let d = parents
            .get(id)
            .and_then(|p| p.as_deref())
            .map(|p| {
                if parents.contains_key(p) {
                    let parent_depth = depth(p, parents, memo, visiting);
                    if parent_depth == usize::MAX {
                        usize::MAX
                    } else {
                        parent_depth + 1
                    }
                } else {
                    0
                }
            })
            .unwrap_or(0);
        visiting.remove(id);
        if d != usize::MAX {
            memo.insert(id.to_string(), d);
        }
        d
    }
    let mut depths = HashMap::new();
    for i in &inv {
        let mut visiting = HashSet::new();
        let d = depth(&i.inv, &parents, &mut depths, &mut visiting);
        depths.insert(i.inv.clone(), if d == usize::MAX { 0 } else { d });
    }
    let mut ordered = records.iter().enumerate().collect::<Vec<_>>();
    ordered.sort_by_key(|(n, r)| (record_ts(r), *n));
    let mut out = String::new();
    for (_, r) in ordered {
        let depth = depths.get(record_inv(r)).copied().unwrap_or(0);
        let detail = match r {
            Record::Start(s) => format!(
                "{} {}",
                match s.payload.cmd {
                    Command::Run => "run",
                    Command::Resume => "resume",
                    Command::Launch => "launch",
                },
                prompt_line(s.payload.prompt.as_deref())
            ),
            Record::Spawn(s) => format!(
                "{} {} {}",
                s.payload.harness,
                s.payload.model,
                s.payload.session.as_deref().unwrap_or("-")
            ),
            Record::Turn(t) => format!("{} {}", t.session, prompt_line(t.result.as_deref())),
            Record::End(e) => format!(
                "{} {} session={} {}",
                e.payload
                    .exit
                    .map(|x| x.to_string())
                    .or_else(|| e.payload.signal.map(|x| format!("signal {x}")))
                    .unwrap_or_else(|| "-".into()),
                compact(e.duration_ms),
                e.payload.session.as_deref().unwrap_or("-"),
                e.payload
                    .error
                    .as_deref()
                    .or(e.payload.result.as_deref())
                    .unwrap_or("")
            ),
            Record::Child(c) => format!(
                "pid={} pgid={} supervisor={}",
                c.payload.proc.pid, c.payload.pgid, c.payload.supervisor.pid
            ),
            Record::Session(s) => s.session.clone(),
            Record::Signal(s) => format!(
                "{} → {} by pid {} [{}]",
                s.payload.signal,
                match s.payload.to {
                    audit::SignalDestination::Supervisor => "supervisor",
                    audit::SignalDestination::Child => "child",
                    audit::SignalDestination::Descendants => "descendants",
                },
                s.payload.by.pid,
                s.payload.reason
            ),
        };
        let detail = sanitize(&detail);
        out.push_str(&format!(
            "{}{} {} {} {}\n",
            time(record_ts(r), "%H:%M:%S"),
            "  ".repeat(depth),
            short(record_inv(r), verbose),
            record_kind(r),
            clip(&detail, width)
        ));
        if verbose {
            let full = match r {
                Record::Start(s) => Some(s.payload.prompt.as_deref().unwrap_or("")),
                Record::Turn(t) => Some(t.result.as_deref().unwrap_or("")),
                Record::End(e) => Some(
                    e.payload
                        .error
                        .as_deref()
                        .or(e.payload.result.as_deref())
                        .unwrap_or(""),
                ),
                Record::Spawn(s) => {
                    for note in &s.payload.selection_notes {
                        out.push_str(&format!("    selection: {}\n", sanitize(note)));
                    }
                    None
                }
                Record::Child(_) | Record::Session(_) | Record::Signal(_) => None,
            };
            if let Some(full) = full {
                for line in full.lines() {
                    out.push_str(&format!("    {}\n", sanitize(line)));
                }
            }
            if let Record::End(e) = r {
                for warning in &e.payload.warnings {
                    out.push_str(&format!("    warning: {}\n", sanitize(warning)));
                }
                if let Some(native) = &e.payload.native {
                    out.push_str(&format!("    native: {native}\n"));
                }
            }
        }
    }
    (out, diagnostics)
}

/// The catalog entries a trace selector names.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum TraceMatch {
    Unique(usize),
    None,
    Ambiguous(Vec<usize>),
}

/// Matches `prefix` against the catalog's trace IDs by string prefix.
pub(crate) fn match_trace(entries: &[Entry], prefix: &str) -> TraceMatch {
    let matches = entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.id.starts_with(prefix))
        .map(|(n, _)| n)
        .collect::<Vec<_>>();
    match matches.len() {
        0 => TraceMatch::None,
        1 => TraceMatch::Unique(matches[0]),
        _ => TraceMatch::Ambiguous(matches),
    }
}

fn choose(entries: &[Entry], options: &LogOptions, err: &mut dyn Write) -> Result<Option<usize>> {
    if let Some(prefix) = &options.trace {
        match match_trace(entries, prefix) {
            TraceMatch::Unique(n) => return Ok(Some(n)),
            TraceMatch::None => diag(err, format_args!("no trace matches {prefix}"))?,
            TraceMatch::Ambiguous(matches) => {
                diag(err, format_args!("ambiguous trace selector {prefix}"))?;
                for n in matches {
                    diag(err, &entries[n].id)?;
                }
            }
        }
        return Ok(None);
    }
    if options.latest {
        let mut best: Option<(Timestamp, usize)> = None;
        for (n, e) in entries.iter().enumerate() {
            if let Some(ts) = e
                .index
                .as_ref()
                .map(|i| i.ts)
                .or_else(|| e.start.as_ref().map(|s| s.ts))
            {
                if best.is_none_or(|(b, _)| ts > b) {
                    best = Some((ts, n));
                }
            }
        }
        if let Some((_, n)) = best {
            return Ok(Some(n));
        }
        diag(err, "no audit traces found")?;
        return Ok(None);
    }
    Ok(None)
}

/// Resolves a trace selector to the full harness session ID of the selected
/// trace's root invocation. `Ok(None)` means no trace matches `selector`.
/// Only the selected trace is read; reader and duplicate-record diagnostics,
/// and on success a note naming the trace and session, go to `err`.
pub(crate) fn resolve_root_session(
    dir: &Path,
    selector: &str,
    err: &mut dyn Write,
) -> Result<Option<String>> {
    let entries = load_catalog(dir, err)?;
    match match_trace(&entries, selector) {
        TraceMatch::None => Ok(None),
        TraceMatch::Ambiguous(matches) => {
            let ids = matches
                .iter()
                .map(|&n| sanitize(&entries[n].id))
                .collect::<Vec<_>>()
                .join(", ");
            anyhow::bail!(
                "ambiguous trace selector {}; matching traces: {ids}; use a full trace ID from `ccc log --verbose`",
                sanitize(selector)
            )
        }
        TraceMatch::Unique(n) => {
            let id = sanitize(&entries[n].id);
            let session = root_session_of(dir, &entries[n], err).map_err(|e| {
                anyhow::anyhow!(
                    "cannot resume trace {id}: {}; inspect it with `ccc log {id}`",
                    sanitize(&format!("{e:#}"))
                )
            })?;
            writeln!(err, "note: trace {id} resolves to root session {session}")?;
            Ok(Some(session))
        }
    }
}

/// The root invocation's effective session for one catalog entry. An indexed
/// entry's root is exactly its index `inv`; an unindexed entry needs exactly
/// one invocation with a parentless start record.
fn root_session_of(dir: &Path, entry: &Entry, err: &mut dyn Write) -> Result<String> {
    if !entry.exists {
        anyhow::bail!(
            "no trace file {}",
            dir.join(format!("{}.jsonl", entry.id)).display()
        );
    }
    let records = audit::read_trace_with_diag(dir, &entry.id, err)?;
    let (invocations, duplicates) = join_invocations(&records);
    for d in duplicates {
        diag(err, sanitize(&d))?;
    }
    let root = match &entry.index {
        Some(index) => {
            let root = invocations
                .iter()
                .find(|i| i.inv == index.inv)
                .filter(|i| start_of(i).is_some())
                .ok_or_else(|| {
                    anyhow::anyhow!("indexed root invocation {} has no start record", index.inv)
                })?;
            if let Some(parent) = start_of(root).and_then(|s| s.parent.as_deref()) {
                anyhow::bail!(
                    "indexed root invocation {} is a child of {parent}",
                    index.inv
                );
            }
            root
        }
        None => {
            let roots = invocations
                .iter()
                .filter(|i| start_of(i).is_some_and(|s| s.parent.is_none()))
                .collect::<Vec<_>>();
            match roots.as_slice() {
                [root] => *root,
                [] => anyhow::bail!("no root invocation with a start record"),
                many => anyhow::bail!(
                    "multiple root invocations: {}",
                    many.iter()
                        .map(|i| i.inv.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }
        }
    };
    let session = effective_session(root)
        .ok_or_else(|| anyhow::anyhow!("root invocation {} has no recorded session", root.inv))?;
    crate::session::check_id(session)
        .with_context(|| format!("root invocation {} recorded a session id", root.inv))?;
    Ok(session.to_string())
}

pub fn log_with(
    dir: &Path,
    options: &LogOptions,
    now: Timestamp,
    width: usize,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<i32> {
    let entries = load_catalog(dir, err)?;
    if options.trace.is_none() && !options.latest {
        if options.json || options.flat {
            return Ok(2);
        }
        write!(
            out,
            "{}",
            render_index(&entries, options.verbose, dirs_home().as_deref())
        )?;
        return Ok(0);
    }
    let Some(n) = choose(&entries, options, err)? else {
        return Ok(2);
    };
    let e = &entries[n];
    if !e.exists {
        if !options.json {
            let (header, diagnostics) = render_tree(
                &e.id,
                e.index.as_ref(),
                &[],
                now,
                options.verbose,
                width,
                dirs_home().as_deref(),
            );
            write!(out, "{header}")?;
            for diagnostic in diagnostics {
                diag(err, diagnostic)?;
            }
        }
        diag(
            err,
            format_args!(
                "no trace file: {}",
                dir.join(format!("{}.jsonl", e.id)).display()
            ),
        )?;
        return Ok(2);
    }
    let records = audit::read_trace_with_diag(dir, &e.id, err)
        .with_context(|| format!("reading trace {}", e.id))?;
    if options.json {
        let (text, ds) = render_json(&e.id, e.index.as_ref(), &records)?;
        for d in ds {
            diag(err, d)?;
        }
        writeln!(out, "{text}")?;
    } else if options.flat {
        let (text, diagnostics) = render_flat(&records, options.verbose, width);
        write!(out, "{text}")?;
        for diagnostic in diagnostics {
            diag(err, diagnostic)?;
        }
    } else {
        let (text, ds) = render_tree(
            &e.id,
            e.index.as_ref(),
            &records,
            now,
            options.verbose,
            width,
            dirs_home().as_deref(),
        );
        for d in ds {
            diag(err, d)?;
        }
        write!(out, "{text}")?;
    }
    Ok(0)
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

pub fn terminal_width() -> usize {
    if !std::io::IsTerminal::is_terminal(&std::io::stdout()) {
        return 120;
    }
    #[cfg(unix)]
    {
        let mut ws = unsafe { std::mem::zeroed::<libc::winsize>() };
        if unsafe { libc::ioctl(std::io::stdout().as_raw_fd(), libc::TIOCGWINSZ, &mut ws) } == 0
            && ws.ws_col > 0
        {
            return ws.ws_col as usize;
        }
    }
    120
}

#[cfg(unix)]
use std::os::fd::AsRawFd;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;
    use std::io;
    use std::io::BufRead;

    struct AbsentProcessProbe;

    impl ProcessProbe for AbsentProcessProbe {
        fn identity_liveness(&self, _process: &crate::procs::ProcId) -> crate::procs::Liveness {
            crate::procs::Liveness::Dead
        }

        fn pid_presence(&self, _pid: u32) -> crate::procs::Presence {
            crate::procs::Presence::Absent
        }
    }

    struct CountingProcessProbe {
        identity: std::sync::Mutex<usize>,
        presence: std::sync::Mutex<usize>,
    }

    impl ProcessProbe for CountingProcessProbe {
        fn identity_liveness(&self, _process: &crate::procs::ProcId) -> crate::procs::Liveness {
            *self.identity.lock().unwrap() += 1;
            crate::procs::Liveness::Alive
        }

        fn pid_presence(&self, _pid: u32) -> crate::procs::Presence {
            *self.presence.lock().unwrap() += 1;
            crate::procs::Presence::Present
        }
    }

    const ROOT: &str = "11111111-1111-4111-8111-111111111111";
    const CHILD: &str = "22222222-2222-4222-8222-222222222222";
    const ORPHAN: &str = "33333333-3333-4333-8333-333333333333";
    const OTHER: &str = "44444444-4444-4444-8444-444444444444";

    fn value(t: &str, ts: &str, inv: &str, parent: Option<&str>) -> serde_json::Value {
        match t {
            "start" => serde_json::json!({
                "t":"start", "ts":ts, "inv":inv, "parent":parent,
                "pid":42, "cmd":"run", "agent":"worker", "session":null,
                "cwd_arg":"/work", "profile_arg":null, "harness_arg":"codex",
                "json":false, "prompt_source":"arg", "prompt_file":null,
                "prompt":"\nfirst prompt line\nsecond line", "passthrough":[]
            }),
            "spawn" => serde_json::json!({
                "t":"spawn", "ts":ts, "inv":inv, "harness":"codex", "model":"m",
                "effort":"low", "profile":null, "auth":null, "sandbox":"read-only",
                "tools":null, "agent":"worker", "cwd":"/work", "session":"session-1",
                "selection_notes":["selected by test"]
            }),
            "turn" => serde_json::json!({
                "t":"turn", "ts":ts, "inv":inv, "harness":"codex", "session":"session-1",
                "turn":"turn-1", "result":"\nturn result\nmore", "transcript":"/tmp/t"
            }),
            "end" => serde_json::json!({
                "t":"end", "ts":ts, "inv":inv, "exit":0, "signal":null,
                "session":"session-1", "result":"done", "error":null,
                "warnings":["warning detail"], "native":{"detail":true}, "turns":1,
                "duration_ms":1200
            }),
            "child" => serde_json::json!({
                "t":"child", "ts":ts, "inv":inv,
                "proc":{"pid":101,"start":202,"boot_id":"boot"}, "pgid":101,
                "supervisor":{"pid":99,"start":303,"boot_id":"boot"}
            }),
            "session" => serde_json::json!({
                "t":"session", "ts":ts, "inv":inv, "session":"session-1"
            }),
            "signal" => serde_json::json!({
                "t":"signal", "ts":ts, "inv":inv, "signal":15,
                "to":"descendants", "by":{"pid":99,"start":303,"boot_id":"boot"},
                "by_inv":null, "reason":"timeout"
            }),
            _ => unreachable!(),
        }
    }

    #[test]
    fn invocation_state_probe_reads_each_represented_process_once() {
        let invocation = join_invocations(&records(&[
            ("start", "2026-01-01T00:00:00Z", ROOT, None),
            ("child", "2026-01-01T00:00:01Z", ROOT, None),
        ]))
        .0
        .remove(0);
        let probe = CountingProcessProbe {
            identity: std::sync::Mutex::new(0),
            presence: std::sync::Mutex::new(0),
        };
        assert_eq!(invocation_state_with(&invocation, &probe), "running");
        assert_eq!(*probe.identity.lock().unwrap(), 2);
        assert_eq!(*probe.presence.lock().unwrap(), 0);
        assert_eq!(
            invocation_state_from_facts(
                &invocation,
                Some(ProcessFact::Dead),
                Some(ProcessFact::Alive),
            ),
            "orphaned"
        );
    }

    fn records(spec: &[(&str, &str, &str, Option<&str>)]) -> Vec<Record> {
        spec.iter()
            .map(|(t, ts, inv, parent)| serde_json::from_value(value(t, ts, inv, *parent)).unwrap())
            .collect()
    }

    fn write_trace(dir: &Path, id: &str, records: &[Record]) {
        let text = records
            .iter()
            .map(|r| serde_json::to_string(r).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        fs::write(dir.join(format!("{id}.jsonl")), text).unwrap();
    }

    fn index(id: &str, ts: &str) -> serde_json::Value {
        serde_json::json!({"trace":id,"ts":ts,"inv":"root","pid":42,"cmd":"run","agent":"worker","cwd":"/work","config":"/config","config_exists":true,"env":{"CCC_MODEL_ALL":"m"},"env_other":["CCC_UNKNOWN"]})
    }

    fn record(json: &str) -> Record {
        serde_json::from_str(json).unwrap()
    }

    fn record_with_field(t: &str, ts: &str, inv: &str, field: &str, field_value: &str) -> Record {
        let mut json = value(t, ts, inv, None);
        json[field] = serde_json::Value::String(field_value.into());
        serde_json::from_value(json).unwrap()
    }

    struct OneChunk {
        chunk: Vec<u8>,
        used: bool,
    }

    impl BufRead for OneChunk {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            if self.used {
                Err(io::Error::other("tail was read"))
            } else {
                Ok(&self.chunk)
            }
        }

        fn consume(&mut self, amount: usize) {
            assert!(amount <= self.chunk.len());
            self.used = true;
        }
    }

    impl std::io::Read for OneChunk {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            let bytes = self.fill_buf()?.to_vec();
            let n = bytes.len().min(output.len());
            output[..n].copy_from_slice(&bytes[..n]);
            self.consume(n);
            Ok(n)
        }
    }

    #[test]
    fn first_start_reads_only_the_first_nonempty_chunk() {
        let id = "77777777-7777-4777-8777-777777777777";
        let line = serde_json::to_vec(&records(&[("start", "2026-01-01T00:00:00Z", id, None)])[0])
            .unwrap();
        let mut err = Vec::new();
        let mut first = line.clone();
        first.push(b'\n');
        let start = first_start_from_reader(
            OneChunk {
                chunk: first,
                used: false,
            },
            Path::new("bounded.jsonl"),
            &mut err,
        )
        .unwrap();
        assert!(start.is_some());
        assert!(err.is_empty());

        let mut err = Vec::new();
        assert!(first_start_from_reader(
            OneChunk {
                chunk: vec![0xff, b'\n'],
                used: false
            },
            Path::new("bad.jsonl"),
            &mut err,
        )
        .unwrap()
        .is_none());
        assert!(String::from_utf8(err).unwrap().contains("bad.jsonl"));

        let catalog = TempDir::new("indexed-trace-not-opened");
        let indexed = "88888888-8888-4888-8888-888888888888";
        fs::write(
            catalog.path().join("traces.jsonl"),
            format!("{}\n", index(indexed, "2026-01-01T00:00:00Z")),
        )
        .unwrap();
        let mut err = Vec::new();
        let entries = load_catalog(catalog.path(), &mut err).unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].index.is_some());
        assert!(err.is_empty());
    }

    #[test]
    fn json_keeps_tags_sorts_turns_and_reports_singleton_duplicates() {
        let records = vec![
            record(
                r#"{"t":"start","ts":"2026-01-01T00:00:00Z","inv":"a","parent":null,"pid":1,"cmd":"run","agent":"x","session":null,"cwd_arg":null,"profile_arg":null,"harness_arg":null,"json":false,"prompt_source":null,"prompt_file":null,"prompt":"p","passthrough":[],"extra":"ignored"}"#,
            ),
            record(
                r#"{"t":"turn","ts":"2026-01-01T00:00:02Z","inv":"a","harness":"codex","session":"s","turn":"2","result":"two","transcript":null}"#,
            ),
            record(
                r#"{"t":"turn","ts":"2026-01-01T00:00:01Z","inv":"a","harness":"codex","session":"s","turn":"1","result":"one","transcript":null}"#,
            ),
            record(
                r#"{"t":"start","ts":"2026-01-01T00:00:03Z","inv":"a","parent":null,"pid":1,"cmd":"run","agent":"x","session":null,"cwd_arg":null,"profile_arg":null,"harness_arg":null,"json":false,"prompt_source":null,"prompt_file":null,"prompt":"p","passthrough":[]}"#,
            ),
        ];
        let (json, warnings) = render_json("a", None, &records).unwrap();
        assert_eq!(warnings, vec!["duplicate start record for invocation a"]);
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["trace"], "a");
        assert!(value.get("index").is_none());
        assert_eq!(value["invocations"][0]["turns"][0]["t"], "turn");
        assert_eq!(value["invocations"][0]["turns"][0]["result"], "one");
        assert!(value.to_string().find("ignored").is_none());
        assert!(!json.ends_with('\n'));

        let indexed: IndexLine =
            serde_json::from_value(index(ROOT, "2026-01-01T00:00:00Z")).unwrap();
        let (indexed_json, _) = render_json(ROOT, Some(&indexed), &records).unwrap();
        let indexed_value: serde_json::Value = serde_json::from_str(&indexed_json).unwrap();
        assert_eq!(
            indexed_value["index"],
            serde_json::to_value(indexed).unwrap()
        );
    }

    #[test]
    fn empty_json_has_only_trace_and_invocations() {
        let (json, warnings) = render_json("a", None, &[]).unwrap();
        assert!(warnings.is_empty());
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value, serde_json::json!({"trace":"a","invocations":[]}));
    }

    #[test]
    fn json_keeps_startless_invocations_equal_time_turn_order_and_named_duplicates() {
        let mut spawn1 = value("spawn", "2026-01-01T00:00:02Z", ROOT, None);
        let spawn2 = spawn1.clone();
        let mut end1 = value("end", "2026-01-01T00:00:04Z", ROOT, None);
        let end2 = end1.clone();
        spawn1["selection_notes"] = serde_json::json!(["first"]);
        end1["result"] = serde_json::json!("first");
        let records = vec![
            record(
                &serde_json::to_string(&value("start", "2026-01-01T00:00:01Z", ROOT, None))
                    .unwrap(),
            ),
            record(&serde_json::to_string(&spawn1).unwrap()),
            record(&serde_json::to_string(&spawn2).unwrap()),
            record(
                &serde_json::to_string(&value("turn", "2026-01-01T00:00:03Z", OTHER, None))
                    .unwrap(),
            ),
            record(
                &serde_json::to_string(&value("turn", "2026-01-01T00:00:03Z", OTHER, None))
                    .unwrap(),
            ),
            record(&serde_json::to_string(&end1).unwrap()),
            record(&serde_json::to_string(&end2).unwrap()),
        ];
        let (json, warnings) = render_json(ROOT, None, &records).unwrap();
        assert_eq!(warnings.len(), 2);
        assert!(warnings
            .iter()
            .any(|w| w.contains("duplicate spawn") && w.contains(ROOT)));
        assert!(warnings
            .iter()
            .any(|w| w.contains("duplicate end") && w.contains(ROOT)));
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["invocations"].as_array().unwrap().len(), 2);
        assert_eq!(value["invocations"][0]["inv"], ROOT);
        assert_eq!(value["invocations"][1]["start"], serde_json::Value::Null);
        assert_eq!(
            value["invocations"][1]["turns"].as_array().unwrap().len(),
            2
        );
    }

    #[test]
    fn tree_covers_prompt_turns_resume_verbose_details_orphans_and_cycles() {
        let mut all = records(&[
            ("start", "2026-01-01T00:00:00Z", ROOT, None),
            ("spawn", "2026-01-01T00:00:01Z", ROOT, None),
            ("turn", "2026-01-01T00:00:02Z", ROOT, None),
            ("end", "2026-01-01T00:00:03Z", ROOT, None),
            ("start", "2026-01-01T00:00:04Z", CHILD, Some(ROOT)),
            ("spawn", "2026-01-01T00:00:05Z", CHILD, None),
            ("end", "2026-01-01T00:00:06Z", CHILD, None),
            ("start", "2026-01-01T00:00:07Z", ORPHAN, Some(OTHER)),
        ]);
        if let Some(Record::Start(s)) = all
            .iter_mut()
            .find(|r| matches!(r, Record::Start(x) if x.inv == CHILD))
        {
            s.payload.cmd = Command::Resume;
            s.payload.session = Some("session-1".into());
        }
        let now: Timestamp = "2026-01-01T00:01:00Z".parse().unwrap();
        let (text, _) = render_tree(ROOT, None, &all, now, false, 200, None);
        assert!(text.contains("›first prompt line") && text.contains("‹done›"));
        assert!(text.contains("resumes") || text.contains("session created outside"));
        assert!(text.contains("?? unknown parent"));
        let (verbose, _) = render_tree(ROOT, None, &all, now, true, 200, None);
        assert!(verbose.contains("pid=42") && verbose.contains("selection: selected by test"));
        assert!(verbose.contains("warning: warning detail") && verbose.contains("native:"));

        let mut launch = records(&[
            ("start", "2026-01-01T00:00:00Z", ROOT, None),
            ("spawn", "2026-01-01T00:00:01Z", ROOT, None),
            ("turn", "2026-01-01T00:00:02Z", ROOT, None),
            ("end", "2026-01-01T00:00:03Z", ROOT, None),
        ]);
        if let Some(Record::Start(s)) = launch.first_mut() {
            s.payload.cmd = Command::Launch;
        }
        if let Some(Record::End(e)) = launch.last_mut() {
            e.payload.exit = None;
            e.payload.signal = Some(9);
        }
        let (launch_text, _) = render_tree(ROOT, None, &launch, now, false, 200, None);
        assert!(
            launch_text.contains("signal 9")
                && launch_text.contains("1 turns")
                && launch_text.contains("‹done›")
        );

        let cycle = records(&[
            ("start", "2026-01-01T00:00:00Z", ROOT, Some(CHILD)),
            ("start", "2026-01-01T00:00:01Z", CHILD, Some(ROOT)),
        ]);
        let (cycled, _) = render_tree(ROOT, None, &cycle, now, false, 200, None);
        assert_eq!(cycled.matches(ROOT).count(), 1);
        assert!(cycled.contains("invalid parent chain"));
    }

    #[test]
    fn flat_is_timestamp_ordered_depth_safe_and_verbose() {
        let rs = records(&[
            ("start", "2026-01-01T00:00:02Z", CHILD, Some(ROOT)),
            ("end", "2026-01-01T00:00:04Z", CHILD, None),
            ("start", "2026-01-01T00:00:01Z", ROOT, None),
            ("turn", "2026-01-01T00:00:03Z", ROOT, None),
        ]);
        let (text, diagnostics) = render_flat(&rs, false, 200);
        assert!(diagnostics.is_empty());
        assert!(text.find(&ROOT[..8]).unwrap() < text.find(&CHILD[..8]).unwrap());
        assert!(text.contains("end 0 1s session=session-1"));
        assert!(text.contains("turn session-1"));
        let (verbose, diagnostics) = render_flat(&rs, true, 200);
        assert!(diagnostics.is_empty());
        assert!(verbose.contains("first prompt line") && verbose.contains("turn result"));
    }

    #[test]
    fn flat_command_renders_first_duplicates_and_reports_each_duplicate_kind() {
        let td = TempDir::new("audit-view-flat-duplicates");
        let mut rs = records(&[
            ("start", "2026-01-01T00:00:00Z", ROOT, None),
            ("spawn", "2026-01-01T00:00:01Z", ROOT, None),
            ("end", "2026-01-01T00:00:02Z", ROOT, None),
        ]);
        rs.extend(rs.clone());
        write_trace(td.path(), ROOT, &rs);
        let mut out = Vec::new();
        let mut err = Vec::new();
        assert_eq!(
            log_with(
                td.path(),
                &LogOptions {
                    trace: Some(ROOT.into()),
                    flat: true,
                    ..Default::default()
                },
                "2026-01-01T00:00:03Z".parse().unwrap(),
                200,
                &mut out,
                &mut err,
            )
            .unwrap(),
            0
        );
        let stdout = String::from_utf8(out).unwrap();
        let stderr = String::from_utf8(err).unwrap();
        assert!(stdout.contains("run") && stdout.contains("codex") && stdout.contains("0 1s"));
        assert_eq!(stderr.matches("duplicate start record").count(), 1);
        assert_eq!(stderr.matches("duplicate spawn record").count(), 1);
        assert_eq!(stderr.matches("duplicate end record").count(), 1);
    }

    #[test]
    fn fixture_catalog_listing_selection_latest_and_broken_index_are_sink_tested() {
        let td = TempDir::new("audit-view-catalog");
        write_trace(
            td.path(),
            ROOT,
            &records(&[("start", "2026-01-01T00:00:00Z", ROOT, None)]),
        );
        write_trace(
            td.path(),
            CHILD,
            &records(&[("start", "2026-01-03T00:00:00Z", CHILD, None)]),
        );
        fs::write(td.path().join(format!("{OTHER}.jsonl")), "").unwrap();
        fs::write(td.path().join(format!("{ORPHAN}.jsonl")), "not json\n").unwrap();
        fs::write(
            td.path().join("traces.jsonl"),
            format!(
                "{}\n",
                serde_json::to_string(&index(ROOT, "2026-01-02T00:00:00Z")).unwrap()
            ),
        )
        .unwrap();
        let mut out = Vec::new();
        let mut err = Vec::new();
        assert_eq!(
            log_with(
                td.path(),
                &LogOptions::default(),
                Timestamp::UNIX_EPOCH,
                120,
                &mut out,
                &mut err
            )
            .unwrap(),
            0
        );
        let listing = String::from_utf8_lossy(&out);
        assert!(listing.contains(&ROOT[..8]) && listing.contains("(unindexed)"));
        assert!(listing.contains(&OTHER[..8]) && listing.contains(&ORPHAN[..8]));
        assert!(String::from_utf8_lossy(&err).contains(&format!("{ORPHAN}.jsonl")));
        out.clear();
        err.clear();
        assert_eq!(
            log_with(
                td.path(),
                &LogOptions {
                    latest: true,
                    ..Default::default()
                },
                Timestamp::UNIX_EPOCH,
                120,
                &mut out,
                &mut err
            )
            .unwrap(),
            0
        );
        assert!(String::from_utf8_lossy(&out).contains(CHILD));
        fs::write(td.path().join("traces.jsonl"), "garbage\n").unwrap();
        out.clear();
        err.clear();
        assert_eq!(
            log_with(
                td.path(),
                &LogOptions {
                    trace: Some(CHILD[..8].into()),
                    json: true,
                    ..Default::default()
                },
                Timestamp::UNIX_EPOCH,
                120,
                &mut out,
                &mut err
            )
            .unwrap(),
            0
        );
        assert!(String::from_utf8(out).unwrap().contains("\"invocations\""));
        assert!(String::from_utf8_lossy(&err).contains("traces.jsonl"));
    }

    #[test]
    fn fixture_selector_ambiguity_unmatched_and_dangling_keep_json_stdout_empty() {
        let td = TempDir::new("audit-view-select");
        write_trace(
            td.path(),
            ROOT,
            &records(&[("start", "2026-01-01T00:00:00Z", ROOT, None)]),
        );
        write_trace(
            td.path(),
            OTHER,
            &records(&[("start", "2026-01-01T00:00:00Z", OTHER, None)]),
        );
        fs::write(
            td.path().join("traces.jsonl"),
            format!(
                "{}\n",
                serde_json::to_string(&index(ROOT, "2026-01-01T00:00:00Z")).unwrap()
            ),
        )
        .unwrap();
        let mut out = Vec::new();
        let mut err = Vec::new();
        assert_eq!(
            log_with(
                td.path(),
                &LogOptions {
                    trace: Some("not-found".into()),
                    json: true,
                    ..Default::default()
                },
                Timestamp::UNIX_EPOCH,
                120,
                &mut out,
                &mut err
            )
            .unwrap(),
            2
        );
        assert!(out.is_empty() && String::from_utf8_lossy(&err).contains("no trace matches"));
        out.clear();
        err.clear();
        assert_eq!(
            log_with(
                td.path(),
                &LogOptions {
                    trace: Some("".into()),
                    json: true,
                    ..Default::default()
                },
                Timestamp::UNIX_EPOCH,
                120,
                &mut out,
                &mut err
            )
            .unwrap(),
            2
        );
        assert!(out.is_empty() && String::from_utf8_lossy(&err).contains("ambiguous"));
        let mut dangling = index(ORPHAN, "2026-01-01T00:00:00Z");
        dangling["config_exists"] = serde_json::json!(false);
        fs::write(
            td.path().join("traces.jsonl"),
            serde_json::to_string(&dangling).unwrap(),
        )
        .unwrap();
        out.clear();
        err.clear();
        assert_eq!(
            log_with(
                td.path(),
                &LogOptions {
                    trace: Some(ORPHAN.into()),
                    ..Default::default()
                },
                Timestamp::UNIX_EPOCH,
                120,
                &mut out,
                &mut err
            )
            .unwrap(),
            2
        );
        let header = String::from_utf8_lossy(&out);
        assert!(header.contains(&format!("trace {ORPHAN} root run worker started")));
        assert!(header.contains("config /config (missing)"));
        assert!(header.contains("env CCC_MODEL_ALL=m, CCC_UNKNOWN=<unrecognized>"));
        out.clear();
        err.clear();
        assert_eq!(
            log_with(
                td.path(),
                &LogOptions {
                    trace: Some(ORPHAN.into()),
                    json: true,
                    ..Default::default()
                },
                Timestamp::UNIX_EPOCH,
                120,
                &mut out,
                &mut err
            )
            .unwrap(),
            2
        );
        assert!(out.is_empty() && String::from_utf8_lossy(&err).contains("no trace file"));
    }

    #[test]
    fn agents_config_shows_in_listing_header_and_json_and_is_dash_when_unrecorded() {
        let td = TempDir::new("audit-view-agents-config");
        for (id, ts) in [
            (ROOT, "2026-01-01T00:00:00Z"),
            (OTHER, "2026-01-02T00:00:00Z"),
        ] {
            write_trace(td.path(), id, &records(&[("start", ts, id, None)]));
        }
        let legacy = index(ROOT, "2026-01-01T00:00:00Z");
        let legacy_line: IndexLine = serde_json::from_value(legacy.clone()).unwrap();
        assert_eq!(legacy_line.agents_config, "");
        assert!(!legacy_line.agents_config_exists);
        let mut current = index(OTHER, "2026-01-02T00:00:00Z");
        current["agents_config"] = serde_json::json!("/agents.toml");
        current["agents_config_exists"] = serde_json::json!(false);
        fs::write(
            td.path().join("traces.jsonl"),
            format!(
                "{}\n{}\n",
                serde_json::to_string(&legacy).unwrap(),
                serde_json::to_string(&current).unwrap()
            ),
        )
        .unwrap();
        let run = |options: LogOptions| {
            let mut out = Vec::new();
            let mut err = Vec::new();
            let code = log_with(
                td.path(),
                &options,
                Timestamp::UNIX_EPOCH,
                120,
                &mut out,
                &mut err,
            )
            .unwrap();
            assert_eq!(code, 0, "{}", String::from_utf8_lossy(&err));
            String::from_utf8(out).unwrap()
        };

        let listing = run(LogOptions::default());
        assert!(listing.starts_with("TRACE STARTED ROOT CWD CONFIG AGENTS ENV\n"));
        let row = |id: &str| {
            listing
                .lines()
                .find(|line| line.starts_with(&id[..8]))
                .unwrap()
                .to_string()
        };
        assert!(
            row(ROOT).contains(" /config - CCC_MODEL_ALL=m"),
            "{listing}"
        );
        assert!(
            row(OTHER).contains(" /config /agents.toml (missing) CCC_MODEL_ALL=m"),
            "{listing}"
        );

        let header = |id: &str| {
            run(LogOptions {
                trace: Some(id.into()),
                ..Default::default()
            })
            .lines()
            .next()
            .unwrap()
            .to_string()
        };
        let legacy_header = header(ROOT);
        assert!(
            legacy_header.contains("config /config env "),
            "{legacy_header}"
        );
        assert!(!legacy_header.contains(" agents "), "{legacy_header}");
        let current_header = header(OTHER);
        assert!(
            current_header.contains("config /config agents /agents.toml (missing) env "),
            "{current_header}"
        );

        let json: serde_json::Value = serde_json::from_str(&run(LogOptions {
            trace: Some(OTHER.into()),
            json: true,
            ..Default::default()
        }))
        .unwrap();
        assert_eq!(json["index"]["agents_config"], "/agents.toml");
        assert_eq!(json["index"]["agents_config_exists"], false);
        assert_eq!(json["index"]["config"], "/config");
    }

    #[test]
    fn test_missing_tree_nesting_and_sibling_order() {
        let a = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
        let b = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
        let c = "cccccccc-cccc-4ccc-8ccc-cccccccccccc";
        let d = "dddddddd-dddd-4ddd-8ddd-dddddddddddd";
        let rs = records(&[
            ("start", "2026-01-01T00:00:02Z", b, None),
            ("start", "2026-01-01T00:00:01Z", a, None),
            ("start", "2026-01-01T00:00:04Z", d, Some(b)),
            ("start", "2026-01-01T00:00:03Z", c, Some(a)),
        ]);
        let (tree, _) = render_tree_with_probe(
            a,
            None,
            &rs,
            Timestamp::UNIX_EPOCH,
            false,
            200,
            None,
            &AbsentProcessProbe,
        );
        // Rendered times are in the system zone.
        let hms = |sec: u8| {
            time(
                format!("2026-01-01T00:00:0{sec}Z").parse().unwrap(),
                "%H:%M:%S",
            )
        };
        let (t1, t2, t3, t4) = (hms(1), hms(2), hms(3), hms(4));
        assert_eq!(
            tree.lines().collect::<Vec<_>>(),
            vec![
                "trace aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa (unindexed)".to_string(),
                format!("├─ aaaaaaaa run worker {t1} lost <1s  session=- ›first prompt line"),
                format!("│  └─ cccccccc run worker {t3} lost <1s  session=- ›first prompt line"),
                format!("└─ bbbbbbbb run worker {t2} lost <1s  session=- ›first prompt line"),
                format!("   └─ dddddddd run worker {t4} lost <1s  session=- ›first prompt line"),
            ]
        );
        let (flat, _) = render_flat(&rs, false, 200);
        assert_eq!(
            flat.lines().collect::<Vec<_>>(),
            vec![
                format!("{t1} aaaaaaaa start run first prompt line"),
                format!("{t2} bbbbbbbb start run first prompt line"),
                format!("{t3}   cccccccc start run first prompt line"),
                format!("{t4}   dddddddd start run first prompt line"),
            ]
        );
    }

    #[test]
    fn test_vacuous_resume_link_disjunction() {
        let owner = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
        let earlier = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
        let later = "cccccccc-cccc-4ccc-8ccc-cccccccccccc";
        let outside = "dddddddd-dddd-4ddd-8ddd-dddddddddddd";
        let mut rs = records(&[
            ("start", "2026-01-01T00:00:03Z", later, None),
            ("start", "2026-01-01T00:00:01Z", earlier, None),
            ("start", "2026-01-01T00:00:04Z", outside, None),
        ]);
        for r in &mut rs {
            if let Record::Start(s) = r {
                s.payload.cmd = Command::Resume;
            }
        }
        let session = "session-shared";
        for r in &mut rs {
            if let Record::Start(s) = r {
                s.payload.session = Some(session.into());
            }
        }
        for (id, ts) in [
            (earlier, "2026-01-01T00:00:01Z"),
            (later, "2026-01-01T00:00:03Z"),
            (outside, "2026-01-01T00:00:04Z"),
        ] {
            let mut turn = value("turn", ts, id, None);
            turn["session"] = serde_json::json!(session);
            rs.push(record(&turn.to_string()));
        }
        let (text, _) = render_tree(owner, None, &rs, Timestamp::UNIX_EPOCH, false, 200, None);
        assert!(text.contains(&format!("resumes {}", &earlier[..8])));
        let mut no_match = rs;
        for r in &mut no_match {
            if let Record::Start(s) = r {
                if s.inv == outside {
                    s.payload.session = Some("other".into());
                }
            }
        }
        let (text, _) = render_tree(
            owner,
            None,
            &no_match,
            Timestamp::UNIX_EPOCH,
            false,
            200,
            None,
        );
        assert!(text.contains("session created outside this trace"));
    }

    #[test]
    fn lineage_suffix_keys_on_the_start_records_session() {
        let owner = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
        let earlier = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb";
        let fork = "cccccccc-cccc-4ccc-8ccc-cccccccccccc";
        let launched = "dddddddd-dddd-4ddd-8ddd-dddddddddddd";
        let orphan_fork = "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee";
        let fresh = "ffffffff-ffff-4fff-8fff-ffffffffffff";
        let mut rs = records(&[
            ("start", "2026-01-01T00:00:01Z", earlier, None),
            ("turn", "2026-01-01T00:00:02Z", earlier, None),
            ("start", "2026-01-01T00:00:03Z", fork, None),
            ("start", "2026-01-01T00:00:05Z", launched, None),
            ("turn", "2026-01-01T00:00:06Z", launched, None),
            ("start", "2026-01-01T00:00:07Z", orphan_fork, None),
            ("start", "2026-01-01T00:00:09Z", fresh, None),
        ]);
        for r in &mut rs {
            if let Record::Start(s) = r {
                let (cmd, session, forked) = match s.inv.as_str() {
                    i if i == fork => (Command::Resume, Some("session-1"), true),
                    i if i == launched => (Command::Launch, Some("session-1"), false),
                    i if i == orphan_fork => (Command::Resume, Some("nowhere"), true),
                    i if i == fresh => (Command::Launch, None, false),
                    _ => (Command::Run, None, false),
                };
                s.payload.cmd = cmd;
                s.payload.session = session.map(str::to_string);
                s.payload.fork = forked;
            }
        }
        rs.push(record_with_field(
            "end",
            "2026-01-01T00:00:04Z",
            fork,
            "session",
            "fork-9",
        ));
        let (text, _) = render_tree(owner, None, &rs, Timestamp::UNIX_EPOCH, false, 200, None);
        let line_of = |inv: &str| {
            text.lines()
                .find(|l| l.contains(&inv[..8]))
                .map(str::to_string)
                .unwrap_or_else(|| panic!("no line for {inv} in\n{text}"))
        };
        let fork_line = line_of(fork);
        assert!(
            fork_line.contains(&format!("forks {}", &earlier[..8])),
            "{text}"
        );
        let launched_line = line_of(launched);
        assert!(
            launched_line.contains(&format!("resumes {}", &earlier[..8])),
            "{text}"
        );
        assert!(
            line_of(orphan_fork).contains("session created outside this trace"),
            "{text}"
        );
        let fresh_line = line_of(fresh);
        assert!(
            !fresh_line.contains("resumes")
                && !fresh_line.contains("forks")
                && !fresh_line.contains("outside this trace"),
            "{text}"
        );
    }

    #[test]
    fn test_turn_tiebreak_unobservable() {
        let mut b = value("turn", "2026-01-01T00:00:03Z", ROOT, None);
        let mut a = b.clone();
        b["turn"] = serde_json::json!("B");
        a["turn"] = serde_json::json!("A");
        let rs = vec![record(&b.to_string()), record(&a.to_string())];
        let (json, _) = render_json(ROOT, None, &rs).unwrap();
        let turns = serde_json::from_str::<serde_json::Value>(&json).unwrap()["invocations"][0]
            ["turns"]
            .clone();
        assert_eq!(turns[0]["turn"], "B");
        assert_eq!(turns[1]["turn"], "A");
    }

    #[test]
    fn test_security_terminal_escape_injection_through_log_rendering() {
        let mut rs = records(&[
            ("start", "2026-01-01T00:00:00Z", ROOT, None),
            ("spawn", "2026-01-01T00:00:01Z", ROOT, None),
            ("end", "2026-01-01T00:00:02Z", ROOT, None),
        ]);
        if let Record::Start(s) = &mut rs[0] {
            s.payload.cmd = Command::Launch;
            s.payload.agent = Some("agent\r\x1b[31m\n".into());
            s.payload.prompt = Some("hello\n\r\t\x1b[2J\x1b]0;title\x07\x7f\u{0080}".into());
        }
        if let Record::Spawn(s) = &mut rs[1] {
            s.payload.model = "model\x1b[31m".into();
            s.payload.selection_notes = vec!["note\nnext\r\x1b[2J".into()];
        }
        rs.push(record(
            &value("turn", "2026-01-01T00:00:03Z", ROOT, None).to_string(),
        ));
        if let Some(Record::Turn(t)) = rs.last_mut() {
            t.turn = Some("turn\r\x1b[1m".into());
            t.session = "session\n\x1b]0;title\x07".into();
            t.transcript = Some("/tmp/transcript\r\x1b[2J".into());
        }
        if let Record::End(e) = &mut rs[2] {
            e.payload.error = Some("error\r\x1b[0m".into());
            e.payload.warnings = vec!["warn\u{0080}".into()];
            e.payload.native = Some(serde_json::json!({"x":"\x1b[1m"}));
        }
        for (text, _) in [
            render_tree(ROOT, None, &rs, Timestamp::UNIX_EPOCH, false, 200, None),
            render_flat(&rs, false, 200),
            render_flat(&rs, true, 200),
        ] {
            assert!(text.contains("\\x1b") || text.contains("\\r") || text.contains("\\u{"));
            assert!(
                text.bytes().all(|b| b == b'\n' || !(b < 0x20 || b == 0x7f)),
                "{text:?}"
            );
        }
        let mut indexed: IndexLine =
            serde_json::from_value(index(ROOT, "2026-01-01T00:00:00Z")).unwrap();
        indexed.agent = Some("indexed-agent\r\x1b[31m".into());
        indexed.config = "/config\n\x1b]0;config\x07".into();
        indexed
            .env
            .insert("CCC_CONFIG".into(), "value\r\x1b[2J".into());
        let (verbose_tree, _) = render_tree(
            ROOT,
            Some(&indexed),
            &rs,
            Timestamp::UNIX_EPOCH,
            true,
            200,
            None,
        );
        assert!(verbose_tree.contains("indexed-agent\\r\\x1b[31m"));
        assert!(verbose_tree.contains("transcript=/tmp/transcript\\r\\x1b[2J"));
        assert!(verbose_tree.contains("selection: note\\nnext\\r\\x1b[2J"));
        assert!(verbose_tree.contains("warning: warn\\u{0080}"));
        assert!(verbose_tree.contains("native: {\"x\":\"\\u001b[1m\"}"));
        assert!(verbose_tree.contains("\\n\\x1b]0;config\\u{0007}"));
        assert!(verbose_tree
            .bytes()
            .all(|b| b == b'\n' || !(b < 0x20 || b == 0x7f)));
    }

    #[test]
    fn delta_3_catalog_lists_corrupt_first_line_and_intact_unindexed_trace() {
        let td = TempDir::new("audit-view-corrupt-catalog");
        let bad = "55555555-5555-4555-8555-555555555555";
        let good = "66666666-6666-4666-8666-666666666666";
        fs::write(td.path().join(format!("{bad}.jsonl")), [0xff, b'\n']).unwrap();
        write_trace(
            td.path(),
            good,
            &records(&[("start", "2026-01-01T00:00:00Z", good, None)]),
        );
        fs::write(td.path().join("traces.jsonl"), b"").unwrap();
        let mut out = Vec::new();
        let mut err = Vec::new();
        assert_eq!(
            log_with(
                td.path(),
                &LogOptions::default(),
                Timestamp::UNIX_EPOCH,
                120,
                &mut out,
                &mut err
            )
            .unwrap(),
            0
        );
        let stdout = String::from_utf8(out).unwrap();
        let stderr = String::from_utf8(err).unwrap();
        assert!(stdout.contains(&bad[..8]) && stdout.contains(&good[..8]));
        assert!(stdout.lines().any(|line| line.starts_with(&bad[..8])
            && line.split_whitespace().skip(1).all(|field| field == "-")));
        assert!(stderr.contains(&format!("{bad}.jsonl")));
        assert_eq!(stderr.matches("ccc:").count(), 1);
    }

    #[test]
    fn delta_3_renderers_skip_one_interior_line_with_one_path_line_diagnostic() {
        let td = TempDir::new("audit-view-damaged-interior");
        let trace = ROOT;
        let rs = records(&[
            ("start", "2026-01-01T00:00:00Z", ROOT, None),
            ("start", "2026-01-01T00:00:02Z", CHILD, Some(ROOT)),
            ("end", "2026-01-01T00:00:03Z", ROOT, None),
        ]);
        let encoded = rs
            .iter()
            .map(|r| serde_json::to_string(r).unwrap())
            .collect::<Vec<_>>();
        fs::write(
            td.path().join(format!("{trace}.jsonl")),
            format!("{}\nnot-json\n{}\n{}\n", encoded[0], encoded[1], encoded[2]),
        )
        .unwrap();
        fs::write(
            td.path().join("traces.jsonl"),
            serde_json::to_string(&index(ROOT, "2026-01-01T00:00:00Z")).unwrap(),
        )
        .unwrap();
        for options in [
            LogOptions {
                trace: Some(trace.into()),
                ..Default::default()
            },
            LogOptions {
                trace: Some(trace.into()),
                flat: true,
                ..Default::default()
            },
            LogOptions {
                trace: Some(trace.into()),
                json: true,
                ..Default::default()
            },
        ] {
            let mut out = Vec::new();
            let mut err = Vec::new();
            assert_eq!(
                log_with(
                    td.path(),
                    &options,
                    Timestamp::UNIX_EPOCH,
                    200,
                    &mut out,
                    &mut err
                )
                .unwrap(),
                0
            );
            let stdout = String::from_utf8(out).unwrap();
            let stderr = String::from_utf8(err).unwrap();
            assert!(stdout.contains(&ROOT[..8]) && stdout.contains(&CHILD[..8]));
            assert_eq!(stderr.matches("ccc:").count(), 1);
            assert!(stderr.contains("line 2"));
            assert!(!stdout.contains("skipping malformed"));
        }
    }

    #[test]
    fn test_multibyte_ids_and_unknown_parents_are_clipped_by_characters() {
        let inv = "😀😀😀😀😀😀😀😀tail";
        let parent = "ééééééééparent";
        let rs = vec![serde_json::from_value::<Record>(serde_json::json!({
            "t":"start", "ts":"2026-01-01T00:00:00Z", "inv":inv, "parent":parent,
            "pid":1,"cmd":"run","agent":"x","session":null,"cwd_arg":null,"profile_arg":null,
            "harness_arg":null,"json":false,"prompt_source":null,"prompt_file":null,"prompt":null,"passthrough":[]
        })).unwrap()];
        let (tree, _) = render_tree("trace", None, &rs, Timestamp::UNIX_EPOCH, false, 200, None);
        assert!(tree.contains("😀😀😀😀😀😀😀😀"));
        assert!(tree.contains("éééééééé"));
    }

    #[test]
    fn bug_resume_suffix_is_inside_the_requested_width() {
        let mut rs = records(&[
            ("start", "2026-01-01T00:00:00Z", ROOT, None),
            ("start", "2026-01-01T00:00:01Z", CHILD, Some(ROOT)),
        ]);
        if let Record::Start(s) = &mut rs[1] {
            s.payload.cmd = Command::Resume;
            s.payload.session = Some("same".into());
        }
        let (tree, _) = render_tree(ROOT, None, &rs, Timestamp::UNIX_EPOCH, false, 48, None);
        for line in tree
            .lines()
            .filter(|l| l.contains("start") || l.contains("resume"))
        {
            assert!(line.chars().count() <= 48, "{line}");
        }
    }

    #[test]
    fn new_events_are_projected_sorted_and_only_duplicate_children_are_diagnosed() {
        let rs = vec![
            record(&value("child", "2026-01-01T00:00:03Z", ROOT, None).to_string()),
            record(&value("child", "2026-01-01T00:00:04Z", ROOT, None).to_string()),
            record(&value("session", "2026-01-01T00:00:02Z", ROOT, None).to_string()),
            record(&value("session", "2026-01-01T00:00:01Z", ROOT, None).to_string()),
            record(&value("signal", "2026-01-01T00:00:06Z", ROOT, None).to_string()),
            record(&value("signal", "2026-01-01T00:00:05Z", ROOT, None).to_string()),
        ];
        let (invocations, diagnostics) = join_invocations(&rs);
        assert_eq!(invocations.len(), 1);
        assert!(invocations[0].child.is_some());
        assert_eq!(invocations[0].sessions.len(), 2);
        assert_eq!(invocations[0].signals.len(), 2);
        assert!(
            matches!(&invocations[0].sessions[0], Record::Session(session) if session.ts < match &invocations[0].sessions[1] { Record::Session(later) => later.ts, _ => Timestamp::MAX })
        );
        assert!(
            matches!(&invocations[0].signals[0], Record::Signal(signal) if signal.ts < match &invocations[0].signals[1] { Record::Signal(later) => later.ts, _ => Timestamp::MAX })
        );
        assert_eq!(
            diagnostics,
            vec![format!("duplicate child record for invocation {ROOT}")]
        );
    }

    #[test]
    fn a_spawn_agent_renders_behind_a_null_start_agent_or_as_a_dash() {
        let now: Timestamp = "2026-01-01T00:01:00Z".parse().unwrap();
        let spawn = |agent: Option<serde_json::Value>| {
            let mut json = value("spawn", "2026-01-01T00:00:01Z", ROOT, None);
            match agent {
                Some(a) => json["agent"] = a,
                None => {
                    json.as_object_mut().unwrap().remove("agent");
                }
            }
            serde_json::from_value::<Record>(json).unwrap()
        };
        let mut start = value("start", "2026-01-01T00:00:00Z", ROOT, None);
        start["cmd"] = "launch".into();
        start["agent"] = serde_json::Value::Null;
        let start: Record = serde_json::from_value(start).unwrap();

        for roleless in [spawn(None), spawn(Some(serde_json::Value::Null))] {
            let Record::Spawn(s) = &roleless else {
                unreachable!()
            };
            assert_eq!(s.payload.agent, None);
            let (text, _) = render_tree(
                ROOT,
                None,
                &[start.clone(), roleless],
                now,
                false,
                200,
                None,
            );
            assert!(text.contains(" launch - "), "{text}");
            assert!(!text.contains("worker"), "{text}");
        }

        let old = spawn(Some("worker".into()));
        let Record::Spawn(s) = &old else {
            unreachable!()
        };
        assert_eq!(s.payload.agent.as_deref(), Some("worker"));
        let (text, _) = render_tree(ROOT, None, &[start, old], now, false, 200, None);
        assert!(text.contains(" launch worker "), "{text}");
    }

    #[test]
    fn effective_session_precedence_is_independent_at_each_level() {
        let spawn = record_with_field("spawn", "2026-01-01T00:00:00Z", ROOT, "session", "spawn");
        let session_a = record_with_field(
            "session",
            "2026-01-01T00:00:01Z",
            ROOT,
            "session",
            "audit-a",
        );
        let session_b = record_with_field(
            "session",
            "2026-01-01T00:00:02Z",
            ROOT,
            "session",
            "audit-b",
        );
        let turn = record_with_field("turn", "2026-01-01T00:00:03Z", ROOT, "session", "turn");
        let end = record_with_field("end", "2026-01-01T00:00:04Z", ROOT, "session", "end");
        let one = |records: Vec<Record>| join_invocations(&records).0.remove(0);
        assert_eq!(effective_session(&one(vec![spawn.clone()])), Some("spawn"));
        assert_eq!(
            effective_session(&one(vec![spawn.clone(), session_a, session_b.clone()])),
            Some("audit-b")
        );
        assert_eq!(
            effective_session(&one(vec![spawn.clone(), session_b.clone(), turn])),
            Some("turn")
        );
        assert_eq!(
            effective_session(&one(vec![spawn, session_b, end])),
            Some("end")
        );
    }

    #[test]
    fn flat_and_json_outputs_include_new_events() {
        let rs = vec![
            record(&value("child", "2026-01-01T00:00:01Z", ROOT, None).to_string()),
            record(&value("session", "2026-01-01T00:00:02Z", ROOT, None).to_string()),
            record(&value("signal", "2026-01-01T00:00:03Z", ROOT, None).to_string()),
        ];
        let (flat, diagnostics) = render_flat(&rs, false, 200);
        assert!(diagnostics.is_empty());
        assert!(flat.contains("child pid=101 pgid=101 supervisor=99"));
        assert!(flat.contains("session session-1"));
        assert!(flat.contains("signal 15 → descendants by pid 99 [timeout]"));
        let (json, _) = render_json(ROOT, None, &rs).unwrap();
        let invocation =
            &serde_json::from_str::<serde_json::Value>(&json).unwrap()["invocations"][0];
        assert!(invocation.get("child").is_some());
        assert!(invocation.get("sessions").is_some());
        assert!(invocation.get("signals").is_some());
    }

    /// A record of kind `t` whose `session` field is `session` (JSON null
    /// when `None`); a start additionally gets `parent`.
    fn sess(t: &str, ts: &str, inv: &str, parent: Option<&str>, session: Option<&str>) -> Record {
        let mut json = value(t, ts, inv, parent);
        json["session"] = serde_json::json!(session);
        serde_json::from_value(json).unwrap()
    }

    fn start(ts: &str, inv: &str, parent: Option<&str>) -> Record {
        sess("start", ts, inv, parent, None)
    }

    fn write_index(dir: &Path, lines: &[(&str, &str)]) {
        let text = lines
            .iter()
            .map(|(trace, inv)| {
                let mut line = index(trace, "2026-01-01T00:00:00Z");
                line["inv"] = serde_json::json!(inv);
                serde_json::to_string(&line).unwrap() + "\n"
            })
            .collect::<String>();
        fs::write(dir.join("traces.jsonl"), text).unwrap();
    }

    fn resolve(dir: &Path, selector: &str) -> (Result<Option<String>>, String) {
        let mut err = Vec::new();
        let result = resolve_root_session(dir, selector, &mut err);
        (result, String::from_utf8(err).unwrap())
    }

    fn resolve_err(dir: &Path, selector: &str) -> String {
        format!("{:#}", resolve(dir, selector).0.unwrap_err())
    }

    const T0: &str = "2026-01-01T00:00:00Z";
    const T1: &str = "2026-01-01T00:00:01Z";
    const T2: &str = "2026-01-01T00:00:02Z";
    const T3: &str = "2026-01-01T00:00:03Z";

    #[test]
    fn match_trace_reports_unique_none_and_every_ambiguous_candidate() {
        let entry = |id: &str| Entry {
            id: id.into(),
            index: None,
            exists: true,
            start: None,
        };
        let near = "11111111-1111-4111-8111-111111111112";
        let entries = [entry(ROOT), entry(near), entry(CHILD)];
        assert_eq!(match_trace(&entries, &CHILD[..8]), TraceMatch::Unique(2));
        assert_eq!(match_trace(&entries, ROOT), TraceMatch::Unique(0));
        assert_eq!(match_trace(&entries, "9"), TraceMatch::None);
        assert_eq!(
            match_trace(&entries, &ROOT[..8]),
            TraceMatch::Ambiguous(vec![0, 1])
        );
    }

    #[test]
    fn indexed_root_session_ignores_trace_id_record_order_children_and_other_roots() {
        let td = TempDir::new("resolve-indexed");
        write_trace(
            td.path(),
            ROOT,
            &[
                start(T0, "kid", Some("inv-a")),
                sess("spawn", T0, "kid", None, Some("kid-session")),
                start(T1, "inv-b", None),
                sess("spawn", T1, "inv-b", None, Some("b-session")),
                start(T2, "inv-a", None),
                sess("spawn", T2, "inv-a", None, Some("a-session")),
                sess("turn", T3, "kid", None, Some("kid-later")),
            ],
        );
        write_index(td.path(), &[(ROOT, "inv-a"), (ROOT, "inv-b")]);
        let (result, err) = resolve(td.path(), &ROOT[..8]);
        assert_eq!(result.unwrap().as_deref(), Some("a-session"));
        assert_eq!(
            err,
            format!("note: trace {ROOT} resolves to root session a-session\n")
        );
        assert_eq!(
            resolve(td.path(), ROOT).0.unwrap().as_deref(),
            Some("a-session")
        );
        assert_eq!(resolve(td.path(), "5").0.unwrap(), None);
    }

    #[test]
    fn inconsistent_indexed_roots_fail_without_falling_back() {
        let td = TempDir::new("resolve-index-bad");
        let base = [
            start(T0, "inv-b", None),
            sess("spawn", T0, "inv-b", None, Some("b-session")),
            start(T1, "kid", Some("inv-b")),
            sess("spawn", T1, "kid", None, Some("kid-session")),
            sess("spawn", T2, "inv-a", None, Some("a-session")),
        ];
        write_trace(td.path(), ROOT, &base);
        write_index(td.path(), &[(ROOT, "inv-a")]);
        let e = resolve_err(td.path(), &ROOT[..8]);
        assert!(
            e.contains("indexed root invocation inv-a has no start record"),
            "{e}"
        );
        assert!(e.contains(&format!("cannot resume trace {ROOT}")), "{e}");
        assert!(e.contains(&format!("`ccc log {ROOT}`")), "{e}");
        write_index(td.path(), &[(ROOT, "kid")]);
        let e = resolve_err(td.path(), &ROOT[..8]);
        assert!(
            e.contains("indexed root invocation kid is a child of inv-b"),
            "{e}"
        );
        write_index(td.path(), &[(ROOT, "absent")]);
        let e = resolve_err(td.path(), &ROOT[..8]);
        assert!(e.contains("absent has no start record"), "{e}");
    }

    #[test]
    fn a_root_without_a_session_never_borrows_a_child_or_its_fork_source() {
        let td = TempDir::new("resolve-sessionless");
        write_trace(
            td.path(),
            ROOT,
            &[
                sess("start", T0, "root", None, Some("fork-source")),
                start(T1, "kid", Some("root")),
                sess("spawn", T1, "kid", None, Some("kid-session")),
            ],
        );
        write_index(td.path(), &[(ROOT, "root")]);
        let e = resolve_err(td.path(), &ROOT[..8]);
        assert!(
            e.contains("root invocation root has no recorded session"),
            "{e}"
        );
        assert!(
            !e.contains("kid-session") && !e.contains("fork-source"),
            "{e}"
        );
    }

    #[test]
    fn unindexed_traces_need_exactly_one_parentless_start() {
        let td = TempDir::new("resolve-unindexed");
        write_trace(
            td.path(),
            ROOT,
            &[
                sess("spawn", T0, "orphan", None, Some("orphan-session")),
                start(T1, "kid", Some("inv-a")),
                sess("spawn", T1, "kid", None, Some("kid-session")),
                start(T2, "inv-a", None),
                sess("session", T2, "inv-a", None, Some("a-session")),
            ],
        );
        write_trace(
            td.path(),
            CHILD,
            &[
                start(T0, "one", None),
                sess("spawn", T0, "one", None, Some("one-session")),
                start(T1, "two", None),
                sess("spawn", T1, "two", None, Some("two-session")),
            ],
        );
        write_trace(
            td.path(),
            OTHER,
            &[sess("spawn", T0, "startless", None, Some("s-session"))],
        );
        fs::write(td.path().join(format!("{ORPHAN}.jsonl")), "").unwrap();
        assert_eq!(
            resolve(td.path(), &ROOT[..8]).0.unwrap().as_deref(),
            Some("a-session")
        );
        let e = resolve_err(td.path(), &CHILD[..8]);
        assert!(e.contains("multiple root invocations: one, two"), "{e}");
        for id in [OTHER, ORPHAN] {
            let e = resolve_err(td.path(), &id[..8]);
            assert!(e.contains("no root invocation with a start record"), "{e}");
        }
    }

    #[test]
    fn root_session_follows_effective_session_precedence_and_rejects_invalid_ids() {
        let td = TempDir::new("resolve-precedence");
        let mut trace = vec![
            start(T0, "root", None),
            sess("spawn", T0, "root", None, Some("from-spawn")),
        ];
        write_index(td.path(), &[(ROOT, "root")]);
        write_trace(td.path(), ROOT, &trace);
        let got = |td: &TempDir| resolve(td.path(), &ROOT[..8]).0.unwrap().unwrap();
        assert_eq!(got(&td), "from-spawn");
        trace.push(sess("session", T1, "root", None, Some("from-session")));
        write_trace(td.path(), ROOT, &trace);
        assert_eq!(got(&td), "from-session");
        trace.push(sess("turn", T2, "root", None, Some("from-turn")));
        trace.push(sess("end", T3, "root", None, None));
        write_trace(td.path(), ROOT, &trace);
        assert_eq!(got(&td), "from-turn");
        trace.pop();
        trace.push(sess("end", T3, "root", None, Some("from-end")));
        write_trace(td.path(), ROOT, &trace);
        assert_eq!(got(&td), "from-end");
        trace.pop();
        trace.push(sess("end", T3, "root", None, Some("../escape")));
        write_trace(td.path(), ROOT, &trace);
        let e = resolve_err(td.path(), &ROOT[..8]);
        assert!(
            e.contains("recorded a session id: invalid session id \"../escape\""),
            "{e}"
        );
        assert!(!e.contains("from-turn"), "{e}");
    }

    #[test]
    fn selector_ambiguity_and_dangling_entries_fail_with_full_trace_ids() {
        let td = TempDir::new("resolve-select");
        let near = "11111111-1111-4111-8111-111111111112";
        for id in [ROOT, near] {
            write_trace(
                td.path(),
                id,
                &[
                    start(T0, "root", None),
                    sess("spawn", T0, "root", None, Some("s")),
                ],
            );
        }
        write_index(td.path(), &[(ORPHAN, "root")]);
        let e = resolve_err(td.path(), &ROOT[..8]);
        assert!(
            e.contains(&format!("matching traces: {ROOT}, {near}")),
            "{e}"
        );
        assert!(e.contains("ccc log --verbose"), "{e}");
        let e = resolve_err(td.path(), &ORPHAN[..8]);
        assert!(e.contains("no trace file"), "{e}");
        assert!(e.contains(&format!("cannot resume trace {ORPHAN}")), "{e}");
    }

    #[test]
    fn damaged_lines_are_diagnosed_while_schema_and_read_failures_reject() {
        let td = TempDir::new("resolve-reader");
        let line = |r: &Record| serde_json::to_string(r).unwrap();
        let mut bytes = Vec::new();
        for text in [
            line(&start(T0, "root", None)),
            "{not json".into(),
            r#"{"t":"mystery","ts":"2026-01-01T00:00:00Z","inv":"root"}"#.into(),
            line(&start(T1, "root", Some("dup-parent"))),
            line(&sess("spawn", T1, "root", None, Some("kept"))),
        ] {
            bytes.extend_from_slice(text.as_bytes());
            bytes.push(b'\n');
        }
        bytes.extend_from_slice(b"\xff\xfe\n{\"t\":\"turn\"");
        fs::write(td.path().join(format!("{ROOT}.jsonl")), &bytes).unwrap();
        write_index(td.path(), &[(ROOT, "root")]);
        let (result, err) = resolve(td.path(), &ROOT[..8]);
        assert_eq!(result.unwrap().as_deref(), Some("kept"));
        assert!(err.contains("skipping malformed audit line"), "{err}");
        assert!(err.contains("line 6"), "{err}");
        assert!(err.contains("skipping unknown audit record type"), "{err}");
        assert!(err.contains("skipping truncated audit line"), "{err}");
        assert!(
            err.contains("duplicate start record for invocation root"),
            "{err}"
        );

        fs::write(
            td.path().join(format!("{ROOT}.jsonl")),
            format!(
                "{}\n{{\"t\":\"spawn\",\"ts\":\"bad\",\"inv\":\"root\"}}\n",
                line(&start(T0, "root", None))
            ),
        )
        .unwrap();
        let e = resolve_err(td.path(), &ROOT[..8]);
        assert!(e.contains("parsing record"), "{e}");

        let entry = Entry {
            id: CHILD.into(),
            index: None,
            exists: true,
            start: None,
        };
        let e = root_session_of(td.path(), &entry, &mut Vec::new()).unwrap_err();
        assert!(format!("{e:#}").contains("reading audit file"), "{e:#}");
    }

    #[test]
    fn broken_and_duplicate_indices_follow_catalog_recovery() {
        let td = TempDir::new("resolve-index-recovery");
        write_trace(
            td.path(),
            ROOT,
            &[
                start(T0, "first", None),
                sess("spawn", T0, "first", None, Some("first-session")),
                start(T1, "second", None),
                sess("spawn", T1, "second", None, Some("second-session")),
            ],
        );
        write_index(td.path(), &[(ROOT, "second"), (ROOT, "first")]);
        assert_eq!(
            resolve(td.path(), &ROOT[..8]).0.unwrap().as_deref(),
            Some("second-session")
        );
        fs::write(td.path().join("traces.jsonl"), "garbage\n").unwrap();
        let (result, err) = resolve(td.path(), &ROOT[..8]);
        let e = format!("{:#}", result.unwrap_err());
        assert!(
            e.contains("multiple root invocations: first, second"),
            "{e}"
        );
        assert!(err.contains("traces.jsonl"), "{err}");
    }

    #[test]
    fn resolver_diagnostic_text_is_sanitized() {
        let td = TempDir::new("resolve-sanitize");
        write_trace(td.path(), ROOT, &[start(T0, "evil\u{1b}[2J", None)]);
        let e = resolve_err(td.path(), &ROOT[..8]);
        assert!(!e.contains('\u{1b}'), "{e:?}");
    }
}
