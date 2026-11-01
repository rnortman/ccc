//! The `host` role: `ccc_run`, `ccc_resume` and `ccc_wait`. They start
//! subagents as host `ccc` processes, admitted against the run's policy:
//! the agent allowlist, the exec roots and their execution surfaces, and the
//! parent's own read, write and exec roots for anything delegated.
//!
//! Every path a model hands over, and every path read back from a session
//! record, is acted on only when it is already canonical; a path holding a
//! symlink is refused, never resolved.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use super::{arg_str, ToolDef, ToolReply, Toolset};
use crate::goose_policy::{under_any, ExecSurface, GoosePolicy, GooseTool};
use crate::session::{self, SessionRecord};

/// How often a waiting call checks whether its child has exited.
const POLL: Duration = Duration::from_millis(50);

/// The stderr tail a failed child's tool error carries.
const STDERR_TAIL: usize = 4096;

/// How long a finished child's output pipes may stay open before what was
/// read is used.
const READER_GRACE: Duration = Duration::from_secs(1);

/// The `ccc` tools of one Goose run.
pub struct HostTools {
    policy: GoosePolicy,
    /// The `ccc` binary children run as.
    self_exe: PathBuf,
    sessions_dir: PathBuf,
    /// The longest one `ccc_run`, `ccc_resume` or `ccc_wait` call waits.
    tool_wait: Duration,
    /// How long shutdown waits for terminated children.
    kill_grace: Duration,
    /// The children's environment when set; inherited otherwise.
    child_env: Option<BTreeMap<String, String>>,
    runs: Mutex<Runs>,
}

#[derive(Default)]
struct Runs {
    next: u64,
    live: BTreeMap<u64, Arc<Spawned>>,
    closed: bool,
}

/// One child `ccc` and the readers draining its output.
struct Spawned {
    what: String,
    child: Mutex<Child>,
    stdout: Drained,
    stderr: Drained,
    /// Set while a call is waiting on this child.
    claimed: AtomicBool,
    /// Set once shutdown has sent SIGTERM.
    terminated: AtomicBool,
}

impl HostTools {
    /// The tools for `policy`, starting children as `self_exe` and resuming
    /// sessions recorded in `sessions_dir`.
    pub fn new(
        policy: GoosePolicy,
        self_exe: PathBuf,
        sessions_dir: PathBuf,
        tool_wait: Duration,
        kill_grace: Duration,
    ) -> HostTools {
        HostTools {
            policy,
            self_exe,
            sessions_dir,
            tool_wait,
            kill_grace,
            child_env: None,
            runs: Mutex::new(Runs::default()),
        }
    }

    /// Children get exactly `env` instead of this process's environment.
    #[cfg(test)]
    pub fn with_child_env(mut self, env: BTreeMap<String, String>) -> HostTools {
        self.child_env = Some(env);
        self
    }

    fn runs(&self) -> std::sync::MutexGuard<'_, Runs> {
        self.runs.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn closed(&self) -> bool {
        self.runs().closed
    }

    fn roots_text(roots: &[PathBuf]) -> String {
        if roots.is_empty() {
            return "none".into();
        }
        roots
            .iter()
            .map(|r| r.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }

    // ---- admission -------------------------------------------------------

    /// The union of the exec roots' execution surfaces, read now: a `.git`
    /// file under an exec root is checked as it is at the moment of the call.
    fn surfaces(&self) -> Result<Vec<PathBuf>, ToolReply> {
        self.policy
            .exec_surfaces()
            .map_err(|e| ToolReply::error(format!("reading the exec roots' surfaces: {e:#}")))
    }

    /// Admits `dir` as a child's working directory or exec root: under an exec
    /// root, with its execution surface inside `surfaces`.
    fn check_exec_dir(&self, dir: &Path, what: &str, surfaces: &[PathBuf]) -> Result<(), String> {
        if !under_any(dir, &self.policy.exec_roots) {
            return Err(format!(
                "{what} {} is not under an exec root ({})",
                dir.display(),
                Self::roots_text(&self.policy.exec_roots)
            ));
        }
        let surface =
            ExecSurface::of(dir).map_err(|e| format!("{what} {}: {e:#}", dir.display()))?;
        if !surface.lies_inside(surfaces) {
            let outside: Vec<String> = surface
                .elements()
                .into_iter()
                .filter(|e| !under_any(e, surfaces))
                .map(|e| e.display().to_string())
                .collect();
            return Err(format!(
                "{what} {} belongs to a git checkout whose {} lies outside every exec root's \
                 project; an operator can grant that tree with --allow-exec on the top-level run",
                dir.display(),
                outside.join(" and ")
            ));
        }
        Ok(())
    }

    fn check_agent(&self, agent: &str) -> Result<(), String> {
        if self.policy.agents.iter().any(|a| a == agent) {
            Ok(())
        } else {
            Err(format!(
                "agent {agent:?} is not allowed; allowed agents: {}",
                self.policy.agents.join(", ")
            ))
        }
    }

    /// The `ccc run` argv for an admitted call, or the reason it is refused.
    /// Nothing is spawned here.
    fn admit_run(&self, args: &Value) -> Result<(String, Vec<String>), ToolReply> {
        let agent = arg_str(args, "agent")?;
        self.check_agent(agent).map_err(ToolReply::error)?;
        let cwd = match super::arg_opt_str(args, "cwd")? {
            Some(c) => canonical_arg(c, "cwd")?,
            None => match self.policy.exec_roots.first() {
                Some(root) => root.clone(),
                None => return Err(ToolReply::error("this run has no exec root")),
            },
        };
        let allow_read = canonical_list(args, "allow_read")?;
        let allow_write = canonical_list(args, "allow_write")?;
        let allow_exec = canonical_list(args, "allow_exec")?;

        let surfaces = self.surfaces()?;
        self.check_exec_dir(&cwd, "cwd", &surfaces)
            .map_err(ToolReply::error)?;
        for p in &allow_read {
            if !under_any(p, &self.policy.read_roots) {
                return Err(ToolReply::error(format!(
                    "allow_read {} is not under a read root of this run ({})",
                    p.display(),
                    Self::roots_text(&self.policy.read_roots)
                )));
            }
        }
        for p in &allow_write {
            if !under_any(p, &self.policy.write_roots) {
                return Err(ToolReply::error(format!(
                    "allow_write {} is not under a write root of this run ({})",
                    p.display(),
                    Self::roots_text(&self.policy.write_roots)
                )));
            }
        }
        for p in &allow_exec {
            self.check_exec_dir(p, "allow_exec", &surfaces)
                .map_err(ToolReply::error)?;
        }

        let mut argv = vec![
            "run".to_string(),
            agent.to_string(),
            "--goose-nested".to_string(),
            "--cwd".to_string(),
            path_arg(&cwd)?,
        ];
        for (flag, paths) in [
            ("--allow-read", &allow_read),
            ("--allow-write", &allow_write),
            ("--allow-exec", &allow_exec),
        ] {
            for p in paths {
                argv.push(flag.to_string());
                argv.push(path_arg(p)?);
            }
        }
        argv.push("--json".to_string());
        argv.push("-".to_string());
        Ok((format!("agent {agent}"), argv))
    }

    /// The `ccc resume` argv for an admitted session, or the reason it is
    /// refused. Nothing is spawned here.
    fn admit_resume(&self, args: &Value) -> Result<(String, Vec<String>), ToolReply> {
        let id = arg_str(args, "session")?;
        let full_id = || {
            ToolReply::error(format!(
                "no session {id:?}: ccc_resume needs the full session id that ccc_run or \
                 ccc_resume returned"
            ))
        };
        if session::check_id(id).is_err() {
            return Err(full_id());
        }
        let path = session::path_in(&self.sessions_dir, id).map_err(|_| full_id())?;
        if !path.is_file() {
            return Err(full_id());
        }
        let record = session::load_from(&self.sessions_dir, id)
            .map_err(|e| ToolReply::error(format!("{e:#}")))?;
        self.admit_record(id, &record).map_err(ToolReply::error)?;
        let argv = vec![
            "resume".to_string(),
            id.to_string(),
            "--json".to_string(),
            "-".to_string(),
        ];
        Ok((format!("session {id}"), argv))
    }

    /// Admits a loaded record for `ccc_resume`, or says why it is refused.
    fn admit_record(&self, id: &str, record: &SessionRecord) -> Result<(), String> {
        let agent = record
            .agent
            .as_deref()
            .ok_or_else(|| format!("session {id} has no agent type"))?;
        self.check_agent(agent)
            .map_err(|e| format!("session {id}: {e}"))?;
        let surfaces = self.surfaces().map_err(|r| r.text)?;
        let cwd = recorded_canonical(&record.cwd, "cwd")?;
        self.check_exec_dir(&cwd, "the session's cwd", &surfaces)?;
        if let Some(p) = &record.goose_policy {
            for w in &p.write_roots {
                let w = recorded_canonical(w, "write root")?;
                if !under_any(&w, &self.policy.write_roots) {
                    return Err(format!(
                        "session {id} may write {}, which this run may not",
                        w.display()
                    ));
                }
            }
            for e in &p.exec_roots {
                let e = recorded_canonical(e, "exec root")?;
                self.check_exec_dir(&e, "the session's exec root", &surfaces)?;
            }
            for r in &p.read_roots {
                let r = recorded_canonical(r, "read root")?;
                if !under_any(&r, &self.policy.read_roots)
                    && !under_any(&r, &self.policy.exec_roots)
                {
                    return Err(format!(
                        "session {id} may read {}, which lies outside this run's read and exec \
                         roots",
                        r.display()
                    ));
                }
            }
        }
        if record.opened_at_terminal {
            return Err(format!(
                "session {id} was opened at a terminal, where tools may have been added to it; \
                 a human can resume it with ccc resume"
            ));
        }
        Ok(())
    }

    // ---- running ----------------------------------------------------------

    /// Starts `<self_exe> <argv>` with `prompt` on its stdin and waits one
    /// slice for it.
    fn start(&self, what: String, argv: &[String], prompt: &str) -> ToolReply {
        let (n, spawned) = {
            let mut runs = self.runs();
            if runs.closed {
                return ToolReply::error("the ccc tool server is shutting down");
            }
            let mut cmd = Command::new(&self.self_exe);
            cmd.args(argv)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            if let Some(env) = &self.child_env {
                cmd.env_clear().envs(env);
            }
            crate::supervise::unmask_signals_on_exec(&mut cmd);
            let mut child = match cmd.spawn() {
                Ok(c) => c,
                Err(e) => {
                    return ToolReply::error(format!("starting {}: {e}", self.self_exe.display()))
                }
            };
            let stdout = Drained::start(child.stdout.take());
            let stderr = Drained::start(child.stderr.take());
            let stdin = child.stdin.take();
            runs.next += 1;
            let n = runs.next;
            let spawned = Arc::new(Spawned {
                what,
                child: Mutex::new(child),
                stdout,
                stderr,
                claimed: AtomicBool::new(true),
                terminated: AtomicBool::new(false),
            });
            runs.live.insert(n, Arc::clone(&spawned));
            (n, (spawned, stdin))
        };
        let (spawned, stdin) = spawned;
        if let Some(mut stdin) = stdin {
            // A child that exits without reading reports why through its exit
            // status and stderr; a broken pipe here adds nothing.
            let _ = stdin.write_all(prompt.as_bytes());
        }
        self.wait_slice(n, &spawned)
    }

    /// Waits up to `tool_wait` for run `n`.
    fn wait_slice(&self, n: u64, spawned: &Spawned) -> ToolReply {
        // Goose fails a call to a stdio extension after 300 s and leaves the
        // child running; one slice stays under that, and `ccc_wait` continues.
        let deadline = Instant::now() + self.tool_wait;
        loop {
            let status = {
                let mut child = spawned.child.lock().unwrap_or_else(|e| e.into_inner());
                match child.try_wait() {
                    Ok(s) => s,
                    Err(e) => {
                        spawned.claimed.store(false, Ordering::SeqCst);
                        return ToolReply::error(format!("waiting for run {n}: {e}"));
                    }
                }
            };
            if let Some(status) = status {
                self.runs().live.remove(&n);
                return finished(spawned, status);
            }
            if self.closed() {
                return ToolReply::error("the ccc tool server is shutting down");
            }
            let now = Instant::now();
            if now >= deadline {
                spawned.claimed.store(false, Ordering::SeqCst);
                return ToolReply::ok(
                    json!({
                        "status": "running",
                        "run": n.to_string(),
                        "note": format!(
                            "{} is still running; call ccc_wait with run \"{n}\" for its result",
                            spawned.what
                        ),
                    })
                    .to_string(),
                );
            }
            std::thread::sleep(POLL.min(deadline - now));
        }
    }

    fn wait(&self, args: &Value) -> ToolReply {
        let n = match args.get("run") {
            Some(Value::String(s)) => s.trim().parse::<u64>().ok(),
            Some(Value::Number(v)) => v.as_u64(),
            _ => None,
        };
        let Some(n) = n else {
            return ToolReply::error(
                "ccc_wait needs `run`, the number ccc_run or ccc_resume returned",
            );
        };
        let spawned = {
            let runs = self.runs();
            match runs.live.get(&n) {
                Some(s) => Arc::clone(s),
                None => {
                    return ToolReply::error(format!(
                        "no running run {n}; a run's result is returned once"
                    ))
                }
            }
        };
        if spawned.claimed.swap(true, Ordering::SeqCst) {
            return ToolReply::error(format!(
                "run {n} is already being waited on by another call"
            ));
        }
        self.wait_slice(n, &spawned)
    }

    /// Sends SIGTERM to every child still running, once each, and waits for
    /// them up to `kill_grace`.
    pub fn terminate_children(&self) {
        let live: Vec<Arc<Spawned>> = {
            let mut runs = self.runs();
            runs.closed = true;
            runs.live.values().cloned().collect()
        };
        for s in &live {
            let mut child = s.child.lock().unwrap_or_else(|e| e.into_inner());
            // Signalled only while unreaped, so the pid is still this child's.
            if matches!(child.try_wait(), Ok(None)) && !s.terminated.swap(true, Ordering::SeqCst) {
                let _ = crate::procs::signal(child.id(), libc::SIGTERM);
            }
        }
        let deadline = Instant::now() + self.kill_grace;
        for s in &live {
            loop {
                let done = {
                    let mut child = s.child.lock().unwrap_or_else(|e| e.into_inner());
                    !matches!(child.try_wait(), Ok(None))
                };
                let now = Instant::now();
                if done || now >= deadline {
                    break;
                }
                std::thread::sleep(POLL.min(deadline - now));
            }
        }
    }

    fn describe_exec(&self) -> String {
        format!(
            "Allowed agents: {}. Working directories allowed: under {}.",
            self.policy.agents.join(", "),
            Self::roots_text(&self.policy.exec_roots)
        )
    }
}

impl Toolset for HostTools {
    fn list(&self) -> Vec<ToolDef> {
        if !self.policy.tools.contains(&GooseTool::Ccc) {
            return Vec::new();
        }
        let paths = json!({"type": "array", "items": {"type": "string"}});
        vec![
            ToolDef {
                name: "ccc_run",
                description: format!(
                    "Runs a subagent with `prompt` and returns its answer and session id. {} \
                     `cwd` defaults to {}. Paths must be absolute with no symlink. allow_read \
                     must lie under {}, allow_write under {}, allow_exec under the allowed \
                     working directories. If the reply says running, call ccc_wait.",
                    self.describe_exec(),
                    self.policy
                        .exec_roots
                        .first()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "none".into()),
                    Self::roots_text(&self.policy.read_roots),
                    Self::roots_text(&self.policy.write_roots),
                ),
                schema: json!({
                    "type": "object",
                    "properties": {
                        "agent": {"type": "string"},
                        "prompt": {"type": "string"},
                        "cwd": {"type": "string"},
                        "allow_read": paths,
                        "allow_write": paths,
                        "allow_exec": paths,
                    },
                    "required": ["agent", "prompt"],
                }),
            },
            ToolDef {
                name: "ccc_resume",
                description: format!(
                    "Continues a subagent session with `prompt`. `session` is the full id \
                     ccc_run or ccc_resume returned. {} If the reply says running, call ccc_wait.",
                    self.describe_exec()
                ),
                schema: json!({
                    "type": "object",
                    "properties": {"session": {"type": "string"}, "prompt": {"type": "string"}},
                    "required": ["session", "prompt"],
                }),
            },
            ToolDef {
                name: "ccc_wait",
                description: "Waits for a subagent that ccc_run or ccc_resume reported as \
                              running, and returns its answer, or running again."
                    .into(),
                schema: json!({
                    "type": "object",
                    "properties": {"run": {"type": "string"}},
                    "required": ["run"],
                }),
            },
        ]
    }

    fn call(&self, name: &str, args: &Value) -> ToolReply {
        if !self.policy.tools.contains(&GooseTool::Ccc) {
            return ToolReply::error(format!("unknown tool {name}"));
        }
        let admitted = match name {
            "ccc_run" => self.admit_run(args),
            "ccc_resume" => self.admit_resume(args),
            "ccc_wait" => return self.wait(args),
            _ => return ToolReply::error(format!("unknown tool {name}")),
        };
        let (what, argv) = match admitted {
            Ok(a) => a,
            Err(reply) => return reply,
        };
        let prompt = match arg_str(args, "prompt") {
            Ok(p) => p,
            Err(reply) => return reply,
        };
        self.start(what, &argv, prompt)
    }

    fn shutdown(&self) {
        self.terminate_children();
    }
}

/// A model-supplied path, admitted only when it is already canonical.
fn canonical_arg(p: &str, what: &str) -> Result<PathBuf, ToolReply> {
    let path = PathBuf::from(p);
    let canonical =
        std::fs::canonicalize(&path).map_err(|e| ToolReply::error(format!("{what} {p}: {e}")))?;
    if canonical != path {
        return Err(ToolReply::error(format!(
            "{what} {p} contains a symlink or a relative component; pass the real directory, \
             an absolute path with no symlink"
        )));
    }
    if !canonical.is_dir() {
        return Err(ToolReply::error(format!("{what} {p} is not a directory")));
    }
    Ok(canonical)
}

/// An optional list of paths: an array of strings, or one string.
fn canonical_list(args: &Value, key: &str) -> Result<Vec<PathBuf>, ToolReply> {
    let items: Vec<&str> = match args.get(key) {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::String(s)) => vec![s],
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| {
                v.as_str().ok_or_else(|| {
                    ToolReply::error(format!("argument `{key}` must be a list of strings"))
                })
            })
            .collect::<Result<_, _>>()?,
        Some(_) => {
            return Err(ToolReply::error(format!(
                "argument `{key}` must be a list of strings"
            )))
        }
    };
    items.into_iter().map(|p| canonical_arg(p, key)).collect()
}

/// A path read from a session record, admitted only when still canonical.
fn recorded_canonical(p: &Path, what: &str) -> Result<PathBuf, String> {
    match std::fs::canonicalize(p) {
        Ok(c) if c == p => Ok(c),
        Ok(_) => Err(format!(
            "the session's {what} {} now contains a symlink",
            p.display()
        )),
        Err(e) => Err(format!("the session's {what} {}: {e}", p.display())),
    }
}

/// A path as an argv element; ccc's argv parser takes UTF-8 only.
fn path_arg(p: &Path) -> Result<String, ToolReply> {
    p.to_str()
        .map(str::to_string)
        .ok_or_else(|| ToolReply::error(format!("{} is not UTF-8", p.display())))
}

/// A pipe read to its end on a thread of its own.
struct Drained {
    buf: Arc<Mutex<Vec<u8>>>,
    done: Arc<AtomicBool>,
}

impl Drained {
    fn start(r: Option<impl Read + Send + 'static>) -> Drained {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let done = Arc::new(AtomicBool::new(r.is_none()));
        if let Some(mut r) = r {
            let (buf, done) = (Arc::clone(&buf), Arc::clone(&done));
            std::thread::spawn(move || {
                let mut chunk = [0u8; 8192];
                loop {
                    match r.read(&mut chunk) {
                        Ok(0) => break,
                        Ok(n) => buf
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .extend_from_slice(&chunk[..n]),
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                        Err(_) => break,
                    }
                }
                done.store(true, Ordering::SeqCst);
            });
        }
        Drained { buf, done }
    }

    /// What was read, once the pipe closed or `grace` passed: a process the
    /// child left behind can hold the pipe open after the child exits.
    fn collect(&self, grace: Duration) -> Vec<u8> {
        let deadline = Instant::now() + grace;
        while !self.done.load(Ordering::SeqCst) {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            std::thread::sleep(POLL.min(deadline - now));
        }
        self.buf.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

fn tail(bytes: &[u8], cap: usize) -> String {
    let text = String::from_utf8_lossy(bytes);
    let text = text.trim_end();
    if text.len() <= cap {
        return text.to_string();
    }
    let mut start = text.len() - cap;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    format!("…{}", &text[start..])
}

/// The reply for a child that has exited.
fn finished(spawned: &Spawned, status: ExitStatus) -> ToolReply {
    let stdout = spawned.stdout.collect(READER_GRACE);
    let stderr = spawned.stderr.collect(READER_GRACE);
    if !status.success() {
        let how = match status.code() {
            Some(code) => format!("exited with code {code}"),
            None => {
                use std::os::unix::process::ExitStatusExt;
                format!("was killed by signal {}", status.signal().unwrap_or(0))
            }
        };
        return ToolReply::error(format!(
            "{} {how}\n{}",
            spawned.what,
            tail(&stderr, STDERR_TAIL)
        ));
    }
    let parsed: Option<Value> = serde_json::from_slice(&stdout).ok();
    match parsed {
        Some(v) if v.get("session").is_some() => ToolReply::ok(
            serde_json::to_string_pretty(&json!({
                "status": "done",
                "session": v["session"],
                "harness": v["harness"],
                "model": v["model"],
                "result": v["result"],
            }))
            .expect("a JSON value always serialises"),
        ),
        _ => ToolReply::error(format!(
            "{} exited 0 without a result object\nstdout: {}\nstderr: {}",
            spawned.what,
            tail(&stdout, STDERR_TAIL),
            tail(&stderr, STDERR_TAIL)
        )),
    }
}

/// The `host` role: serves the `ccc` tools on stdio until stdin closes or
/// SIGTERM or SIGHUP arrives, then terminates the children still running.
pub fn serve_host(policy: GoosePolicy, env: &dyn crate::config::EnvSource) -> Result<i32> {
    if !policy.tools.contains(&GooseTool::Ccc) {
        bail!("the host role serves only a policy whose tools hold ccc");
    }
    let cfg = crate::config::Config::load_in(env)?;
    let self_exe = std::env::current_exe().context("locating the ccc binary")?;
    let tools = Arc::new(HostTools::new(
        policy,
        self_exe,
        session::sessions_dir_in(env),
        Duration::from_secs(cfg.goose.tool_wait_s),
        Duration::from_secs_f64(cfg.process.kill_grace_s),
    ));

    // SIGTERM and SIGHUP, blocked before any thread starts, so every thread
    // inherits the mask and only the waiter below receives them.
    crate::supervise::block_signals(false)?;
    let set = crate::supervise::managed_set(false);
    {
        let tools = Arc::clone(&tools);
        std::thread::spawn(move || {
            let mut sig: libc::c_int = 0;
            // SAFETY: `set` and `sig` are live for the call.
            let rc = unsafe { libc::sigwait(&set, &mut sig) };
            if rc != 0 {
                eprintln!(
                    "ccc: waiting for SIGTERM failed, stopping: {}",
                    std::io::Error::from_raw_os_error(rc)
                );
            }
            tools.terminate_children();
            std::process::exit(if rc == 0 { 128 + sig } else { 1 });
        });
    }
    let stdin = std::io::stdin();
    super::serve(tools, stdin.lock(), std::io::stdout())?;
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goose_tools::tests::{exchange, reply_for};
    use crate::testutil::TempDir;

    /// A temp tree whose exec root `exec` is its own git top, so the host's
    /// `/tmp` does not become every surface's project top.
    struct Tree {
        dir: TempDir,
    }

    impl Tree {
        fn new(label: &str) -> Tree {
            let dir = TempDir::new(label);
            for d in ["exec/.git", "exec/sub", "read", "write", "other", "bin"] {
                std::fs::create_dir_all(dir.path().join(d)).unwrap();
            }
            Tree { dir }
        }

        fn p(&self, rel: &str) -> PathBuf {
            std::fs::canonicalize(self.dir.path()).unwrap().join(rel)
        }

        fn s(&self, rel: &str) -> String {
            self.p(rel).display().to_string()
        }

        fn policy(&self) -> GoosePolicy {
            GoosePolicy {
                tools: [GooseTool::Ccc, GooseTool::Write].into(),
                agents: vec!["explorer".into(), "gwriter".into()],
                read_roots: vec![self.p("read"), self.p("write")],
                write_roots: vec![self.p("write")],
                exec_roots: vec![self.p("exec")],
            }
        }

        /// A stand-in `ccc` that records its argv and stdin, then runs `body`.
        fn fake_ccc(&self, body: &str) -> PathBuf {
            let calls = self.p("calls");
            let script = format!(
                "#!/bin/sh\nmkdir -p {c}\nprintf '%s\\n' \"$@\" > {c}/argv\ncat > {c}/prompt\n{body}\n",
                c = calls.display()
            );
            let path = self.p("bin/fake-ccc");
            std::fs::write(&path, script).unwrap();
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            path
        }

        fn tools(&self, policy: GoosePolicy, body: &str, wait: Duration) -> HostTools {
            HostTools::new(
                policy,
                self.fake_ccc(body),
                self.p("sessions"),
                wait,
                Duration::from_secs(5),
            )
        }

        fn spawned(&self) -> bool {
            self.p("calls/argv").exists()
        }

        fn argv(&self) -> Vec<String> {
            std::fs::read_to_string(self.p("calls/argv"))
                .unwrap()
                .lines()
                .map(str::to_string)
                .collect()
        }
    }

    const DONE: &str = r#"printf '%s' '{"session":"s-1","harness":"claude","model":"m","result":"the answer","native":{}}'"#;

    fn text(reply: &ToolReply) -> &str {
        &reply.text
    }

    fn refused(reply: ToolReply, needle: &str) {
        assert!(reply.is_error, "{reply:?}");
        assert!(reply.text.contains(needle), "{reply:?}");
    }

    #[test]
    fn an_admitted_run_spawns_a_nested_ccc_and_returns_its_result() {
        let t = Tree::new("host-run");
        std::fs::create_dir_all(t.p("write/out")).unwrap();
        let tools = t.tools(t.policy(), DONE, Duration::from_secs(9));
        let reply = tools.call(
            "ccc_run",
            &json!({"agent": "explorer", "prompt": "look", "cwd": t.s("exec/sub"),
                    "allow_read": [t.s("read")], "allow_write": t.s("write/out"),
                    "allow_exec": [t.s("exec")]}),
        );
        assert!(!reply.is_error, "{reply:?}");
        let v: Value = serde_json::from_str(text(&reply)).unwrap();
        assert_eq!(v["status"], "done");
        assert_eq!(v["session"], "s-1");
        assert_eq!(v["result"], "the answer");
        assert_eq!(v["harness"], "claude");
        assert_eq!(
            t.argv(),
            [
                "run",
                "explorer",
                "--goose-nested",
                "--cwd",
                &t.s("exec/sub"),
                "--allow-read",
                &t.s("read"),
                "--allow-write",
                &t.s("write/out"),
                "--allow-exec",
                &t.s("exec"),
                "--json",
                "-",
            ]
        );
        assert_eq!(
            std::fs::read_to_string(t.p("calls/prompt")).unwrap(),
            "look"
        );
    }

    #[test]
    fn cwd_defaults_to_the_first_exec_root() {
        let t = Tree::new("host-default-cwd");
        let tools = t.tools(t.policy(), DONE, Duration::from_secs(9));
        let reply = tools.call("ccc_run", &json!({"agent": "explorer", "prompt": "p"}));
        assert!(!reply.is_error, "{reply:?}");
        assert_eq!(t.argv()[4], t.s("exec"));
    }

    #[test]
    fn run_admission_refusals_spawn_nothing() {
        let t = Tree::new("host-refuse");
        std::os::unix::fs::symlink(t.p("other"), t.p("write/link")).unwrap();
        std::os::unix::fs::symlink(t.p("other"), t.p("exec/link")).unwrap();
        std::os::unix::fs::symlink(t.p("other"), t.p("read/ssh")).unwrap();
        let tools = t.tools(t.policy(), DONE, Duration::from_secs(9));
        let run = |extra: Value| {
            let mut args = json!({"agent": "explorer", "prompt": "p"});
            for (k, v) in extra.as_object().unwrap() {
                args[k] = v.clone();
            }
            tools.call("ccc_run", &args)
        };
        refused(
            tools.call("ccc_run", &json!({"agent": "intruder", "prompt": "p"})),
            "allowed agents: explorer, gwriter",
        );
        refused(run(json!({"cwd": t.s("other")})), "not under an exec root");
        refused(run(json!({"cwd": t.s("exec/link")})), "contains a symlink");
        refused(
            run(json!({"cwd": format!("{}/..", t.s("exec/sub"))})),
            "contains a symlink",
        );
        refused(run(json!({"cwd": "exec"})), "cwd exec");
        refused(run(json!({"allow_read": [t.s("other")]})), "allow_read");
        refused(
            run(json!({"allow_read": [t.s("read/ssh")]})),
            "contains a symlink",
        );
        refused(
            run(json!({"allow_write": [t.s("read")]})),
            "not under a write root",
        );
        refused(
            run(json!({"allow_write": [t.s("write/link")]})),
            "contains a symlink",
        );
        refused(
            run(json!({"allow_exec": [t.s("write")]})),
            "not under an exec root",
        );
        refused(
            run(json!({"allow_exec": [t.s("exec/link")]})),
            "contains a symlink",
        );
        refused(run(json!({"allow_read": 7})), "list of strings");
        assert!(!t.spawned(), "a refused call spawned a child");
    }

    /// A worktree under the exec root whose git directory lies elsewhere.
    fn worktree(t: &Tree, gitdir_parent: &str) -> PathBuf {
        let main_git = t.p(&format!("{gitdir_parent}/main/.git"));
        let wt_git = main_git.join("worktrees/wt");
        std::fs::create_dir_all(&wt_git).unwrap();
        std::fs::write(wt_git.join("commondir"), "../..\n").unwrap();
        let wt = t.p("exec/wt");
        std::fs::create_dir_all(&wt).unwrap();
        std::fs::write(wt.join(".git"), format!("gitdir: {}\n", wt_git.display())).unwrap();
        wt
    }

    #[test]
    fn containment_refuses_a_worktree_whose_git_dir_leaves_the_exec_surfaces() {
        for parent in ["write", "other"] {
            let t = Tree::new("host-worktree");
            let wt = worktree(&t, parent);
            let tools = t.tools(t.policy(), DONE, Duration::from_secs(9));
            let cwd = wt.display().to_string();
            refused(
                tools.call(
                    "ccc_run",
                    &json!({"agent": "explorer", "prompt": "p", "cwd": cwd}),
                ),
                "lies outside every exec root's project",
            );
            refused(
                tools.call(
                    "ccc_run",
                    &json!({"agent": "explorer", "prompt": "p", "allow_exec": [cwd]}),
                ),
                "allow_exec",
            );
            assert!(!t.spawned(), "{parent}");
        }
    }

    #[test]
    fn containment_admits_a_submodule_of_the_exec_root() {
        let t = Tree::new("host-submodule");
        std::fs::create_dir_all(t.p("exec/.git/modules/sm")).unwrap();
        std::fs::create_dir_all(t.p("exec/sm")).unwrap();
        // Relative, as git writes it.
        std::fs::write(t.p("exec/sm/.git"), "gitdir: ../.git/modules/sm\n").unwrap();
        let tools = t.tools(t.policy(), DONE, Duration::from_secs(9));
        let reply = tools.call(
            "ccc_run",
            &json!({"agent": "explorer", "prompt": "p", "cwd": t.s("exec/sm")}),
        );
        assert!(!reply.is_error, "{reply:?}");
    }

    fn record(agent: &str, cwd: PathBuf, policy: Option<GoosePolicy>) -> SessionRecord {
        SessionRecord {
            harness: if policy.is_some() {
                crate::config::Harness::Goose
            } else {
                crate::config::Harness::Claude
            },
            agent: Some(agent.into()),
            model: "m".into(),
            effort: None,
            profile: None,
            auth: None,
            cwd,
            sandbox: crate::agents::Sandbox::ReadOnly,
            tools: None,
            created_at: jiff::Timestamp::now(),
            last_used_at: jiff::Timestamp::now(),
            trace: None,
            inv: None,
            forked_from: None,
            goose_policy: policy,
            opened_at_terminal: false,
        }
    }

    fn save(t: &Tree, id: &str, r: &SessionRecord) {
        session::save_in(&t.p("sessions"), id, r).unwrap();
    }

    fn resume(tools: &HostTools, id: &str) -> ToolReply {
        tools.call("ccc_resume", &json!({"session": id, "prompt": "again"}))
    }

    #[test]
    fn an_admitted_resume_spawns_ccc_resume() {
        let t = Tree::new("host-resume");
        let nested = GoosePolicy {
            tools: [GooseTool::Read].into(),
            agents: vec![],
            read_roots: vec![t.p("exec/sub")],
            write_roots: vec![],
            exec_roots: vec![],
        };
        save(
            &t,
            "nested-reader",
            &record("gwriter", t.p("exec/sub"), Some(nested)),
        );
        let tools = t.tools(t.policy(), DONE, Duration::from_secs(9));
        let reply = resume(&tools, "nested-reader");
        assert!(!reply.is_error, "{reply:?}");
        assert_eq!(t.argv(), ["resume", "nested-reader", "--json", "-"]);
        assert_eq!(
            std::fs::read_to_string(t.p("calls/prompt")).unwrap(),
            "again"
        );
    }

    #[test]
    fn resume_admission_refusals_spawn_nothing() {
        let t = Tree::new("host-resume-refuse");
        let wt = worktree(&t, "other");
        std::os::unix::fs::symlink(t.p("exec/sub"), t.p("exec/sublink")).unwrap();
        // Recorded roots that lexically sit inside this run's roots but now
        // lead to `other`.
        for root in ["read", "write", "exec"] {
            std::os::unix::fs::symlink(t.p("other"), t.p(&format!("{root}/linked"))).unwrap();
        }
        let goose = |read: Vec<PathBuf>, write: Vec<PathBuf>, exec: Vec<PathBuf>| GoosePolicy {
            tools: [GooseTool::Read, GooseTool::Write, GooseTool::Ccc].into(),
            agents: vec!["explorer".into()],
            read_roots: read,
            write_roots: write,
            exec_roots: exec,
        };
        let cases: Vec<(&str, SessionRecord, &str)> = vec![
            (
                "foreign-agent",
                record("intruder", t.p("exec"), None),
                "not allowed",
            ),
            (
                "foreign-cwd",
                record("explorer", t.p("other"), None),
                "not under an exec root",
            ),
            (
                "linked-cwd",
                record("explorer", t.p("exec/sublink"), None),
                "now contains a symlink",
            ),
            (
                "worktree-cwd",
                record("explorer", wt.clone(), None),
                "outside every exec root's project",
            ),
            (
                "top-level-writer",
                record(
                    "gwriter",
                    t.p("exec/sub"),
                    Some(goose(vec![t.p("exec/sub")], vec![t.p("exec/sub")], vec![])),
                ),
                "may write",
            ),
            (
                "wider-read",
                record(
                    "gwriter",
                    t.p("exec/sub"),
                    Some(goose(vec![t.p("other")], vec![], vec![])),
                ),
                "may read",
            ),
            (
                "worktree-exec",
                record(
                    "gwriter",
                    t.p("exec/sub"),
                    Some(goose(vec![t.p("exec/sub")], vec![], vec![wt.clone()])),
                ),
                "exec root",
            ),
            (
                "linked-write",
                record(
                    "gwriter",
                    t.p("exec/sub"),
                    Some(goose(
                        vec![t.p("write/linked")],
                        vec![t.p("write/linked")],
                        vec![],
                    )),
                ),
                "write root",
            ),
            (
                "linked-exec",
                record(
                    "gwriter",
                    t.p("exec/sub"),
                    Some(goose(
                        vec![t.p("exec/sub")],
                        vec![],
                        vec![t.p("exec/linked")],
                    )),
                ),
                "exec root",
            ),
            (
                "linked-read",
                record(
                    "gwriter",
                    t.p("exec/sub"),
                    Some(goose(vec![t.p("read/linked")], vec![], vec![])),
                ),
                "read root",
            ),
        ];
        let tools = t.tools(t.policy(), DONE, Duration::from_secs(9));
        for (id, r, needle) in &cases {
            save(&t, id, r);
            let reply = resume(&tools, id);
            if id.starts_with("linked-") {
                refused(reply.clone(), "now contains a symlink");
            }
            refused(reply, needle);
        }
        let mut terminal = record("explorer", t.p("exec"), None);
        terminal.opened_at_terminal = true;
        save(&t, "terminal", &terminal);
        refused(resume(&tools, "terminal"), "opened at a terminal");
        for partial in ["foreign", "", "../x", "2026"] {
            refused(resume(&tools, partial), "full session id");
        }
        assert!(!t.spawned(), "a refused resume spawned a child");
    }

    #[test]
    fn a_long_run_returns_running_then_ccc_wait_returns_the_result() {
        let t = Tree::new("host-slices");
        let tools = t.tools(
            t.policy(),
            &format!("sleep 3\n{DONE}"),
            Duration::from_secs(1),
        );
        let first = tools.call("ccc_run", &json!({"agent": "explorer", "prompt": "p"}));
        assert!(!first.is_error, "{first:?}");
        let v: Value = serde_json::from_str(text(&first)).unwrap();
        assert_eq!(v["status"], "running");
        let run = v["run"].as_str().unwrap().to_string();
        refused(
            tools.call("ccc_wait", &json!({"run": "99"})),
            "no running run 99",
        );
        let mut last = Value::Null;
        for _ in 0..6 {
            let r = tools.call("ccc_wait", &json!({"run": run}));
            assert!(!r.is_error, "{r:?}");
            last = serde_json::from_str(text(&r)).unwrap();
            if last["status"] == "done" {
                break;
            }
        }
        assert_eq!(last["result"], "the answer");
        refused(
            tools.call("ccc_wait", &json!({"run": run})),
            "returned once",
        );
    }

    #[test]
    fn one_ccc_wait_at_a_time_holds_a_run_and_a_running_slice_releases_it() {
        let t = Tree::new("host-claim");
        let tools = Arc::new(t.tools(
            t.policy(),
            &format!("sleep 4\n{DONE}"),
            Duration::from_secs(1),
        ));
        let first = tools.call("ccc_run", &json!({"agent": "explorer", "prompt": "p"}));
        let v: Value = serde_json::from_str(text(&first)).unwrap();
        assert_eq!(v["status"], "running", "{v}");
        let run = v["run"].as_str().unwrap().to_string();
        // Two calls at once: whichever claims the run first holds it for its
        // whole one-second slice, so the other is refused.
        let other = {
            let tools = Arc::clone(&tools);
            let run = run.clone();
            std::thread::spawn(move || tools.call("ccc_wait", &json!({"run": run})))
        };
        let mine = tools.call("ccc_wait", &json!({"run": run}));
        let other = other.join().unwrap();
        let (held, turned_away) = if mine.is_error {
            (other, mine)
        } else {
            (mine, other)
        };
        refused(
            turned_away,
            &format!("run {run} is already being waited on"),
        );
        assert!(!held.is_error, "{held:?}");
        let v: Value = serde_json::from_str(text(&held)).unwrap();
        assert_eq!(v["status"], "running", "{v}");
        // Released after `running`; `run` may also be a JSON number.
        let n: u64 = run.parse().unwrap();
        let mut last = Value::Null;
        for _ in 0..6 {
            let r = tools.call("ccc_wait", &json!({"run": n}));
            assert!(!r.is_error, "{r:?}");
            last = serde_json::from_str(text(&r)).unwrap();
            if last["status"] == "done" {
                break;
            }
        }
        assert_eq!(last["result"], "the answer", "{last}");
    }

    #[test]
    fn a_failing_child_is_a_tool_error_with_its_code_and_stderr() {
        let t = Tree::new("host-fail");
        let tools = t.tools(
            t.policy(),
            "echo 'error: nested refusal' >&2\nexit 2",
            Duration::from_secs(9),
        );
        let reply = tools.call("ccc_run", &json!({"agent": "explorer", "prompt": "p"}));
        refused(reply.clone(), "exited with code 2");
        assert!(reply.text.contains("nested refusal"), "{reply:?}");
        let tools = t.tools(t.policy(), "echo not-json", Duration::from_secs(9));
        refused(
            tools.call("ccc_run", &json!({"agent": "explorer", "prompt": "p"})),
            "without a result object",
        );
    }

    #[test]
    fn shutdown_sends_sigterm_to_a_running_child_and_refuses_new_runs() {
        let t = Tree::new("host-shutdown");
        let marker = t.p("termed");
        let body = format!(
            "trap 'echo term > {}; exit 143' TERM\nsleep 5 >/dev/null 2>&1 &\nwait",
            marker.display()
        );
        let tools = Arc::new(t.tools(t.policy(), &body, Duration::from_secs(9)));
        let call = {
            let tools = Arc::clone(&tools);
            std::thread::spawn(move || {
                tools.call("ccc_run", &json!({"agent": "explorer", "prompt": "p"}))
            })
        };
        // The fake writes its argv before it reads the prompt; the trap is set
        // once the prompt is read.
        for _ in 0..100 {
            if t.p("calls/prompt").exists() {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        std::thread::sleep(Duration::from_millis(200));
        tools.shutdown();
        assert_eq!(std::fs::read_to_string(&marker).unwrap().trim(), "term");
        // Either the slice saw the child end, or it saw the shutdown first.
        assert!(call.join().unwrap().is_error);
        refused(
            tools.call("ccc_run", &json!({"agent": "explorer", "prompt": "p"})),
            "shutting down",
        );
    }

    #[test]
    fn served_over_mcp_the_host_lists_and_refuses() {
        let t = Tree::new("host-mcp");
        let tools: Arc<dyn Toolset> = Arc::new(t.tools(t.policy(), DONE, Duration::from_secs(9)));
        let replies = exchange(
            tools,
            &[
                json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
                json!({"jsonrpc":"2.0","id":2,"method":"tools/call",
                       "params":{"name":"ccc_run","arguments":{"agent":"intruder","prompt":"p"}}}),
            ],
        );
        assert_eq!(
            reply_for(&replies, 1)["result"]["tools"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        let r = &reply_for(&replies, 2)["result"];
        assert_eq!(r["isError"], true, "{replies:?}");
        assert!(!t.spawned());
    }

    #[test]
    fn tools_list_offers_the_three_ccc_tools_only_with_ccc() {
        let t = Tree::new("host-list");
        let tools = t.tools(t.policy(), DONE, Duration::from_secs(1));
        let names: Vec<&str> = tools.list().iter().map(|d| d.name).collect();
        assert_eq!(names, ["ccc_run", "ccc_resume", "ccc_wait"]);
        let run = &tools.list()[0].description;
        assert!(run.contains("explorer, gwriter"), "{run}");
        assert!(run.contains(&t.s("exec")), "{run}");
        let mut no_ccc = t.policy();
        no_ccc.tools = [GooseTool::Read].into();
        let tools = t.tools(no_ccc.clone(), DONE, Duration::from_secs(1));
        assert!(tools.list().is_empty());
        refused(tools.call("ccc_run", &json!({})), "unknown tool");
        assert!(serve_host(no_ccc, &BTreeMap::<String, String>::new()).is_err());
    }

    /// `ccc goose-tools --role host` as Goose starts it: its stdin is a pipe
    /// whose only writer is this test, closed after the requests.
    fn host_process(t: &Tree, policy: &str, requests: &str) -> std::process::Output {
        host_process_with(t, policy, requests, &[])
    }

    fn host_process_with(
        t: &Tree,
        policy: &str,
        requests: &str,
        extra_env: &[(&str, &str)],
    ) -> std::process::Output {
        std::fs::write(t.p("config.toml"), "[goose]\ntool_wait_s = 5\n").unwrap();
        let mut child = Command::new(crate::testutil::ccc_binary())
            .args(["goose-tools", "--role", "host", "--policy", policy])
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", t.p("home"))
            .env("XDG_STATE_HOME", t.p("state"))
            .env("CCC_CONFIG", t.p("config.toml"))
            .envs(extra_env.iter().copied())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(requests.as_bytes())
            .unwrap();
        child.wait_with_output().unwrap()
    }

    #[test]
    fn the_host_role_serves_the_ccc_tools_over_stdio_and_exits_at_eof() {
        let t = Tree::new("host-process");
        let requests = [
            json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}),
            json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
            json!({"jsonrpc": "2.0", "id": 2, "method": "resources/list"}),
            json!({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "ccc_run", "arguments": {"agent": "intruder", "prompt": "p"}}}),
        ]
        .iter()
        .map(|r| format!("{r}\n"))
        .collect::<String>();
        let out = host_process(&t, &t.policy().to_hex(), &requests);
        assert_eq!(out.status.code(), Some(0), "{out:?}");
        let replies: Vec<Value> = String::from_utf8(out.stdout)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(
            reply_for(&replies, 0)["result"]["capabilities"],
            json!({"tools": {}}),
            "{replies:?}"
        );
        let names: Vec<&str> = reply_for(&replies, 1)["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["ccc_run", "ccc_resume", "ccc_wait"]);
        assert_eq!(reply_for(&replies, 2)["error"]["code"], -32601);
        assert_eq!(reply_for(&replies, 3)["result"]["isError"], true);
        assert!(!t.p("state/ccc/sessions").exists(), "a child was started");
    }

    #[test]
    fn the_host_role_refuses_a_bad_policy_with_exit_2() {
        let t = Tree::new("host-process-bad");
        let out = host_process(&t, "zz", "");
        assert_eq!(out.status.code(), Some(2), "{out:?}");
        assert!(out.stdout.is_empty(), "{out:?}");
    }

    #[test]
    fn the_host_role_refuses_a_host_env_that_does_not_decode_with_exit_2() {
        let t = Tree::new("host-process-bad-env");
        let requests = format!(
            "{}\n",
            json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
        );
        let out = host_process_with(
            &t,
            &t.policy().to_hex(),
            &requests,
            &[(crate::harness::goose::HOST_ENV_VAR, "zz")],
        );
        assert_eq!(out.status.code(), Some(2), "{out:?}");
        assert!(out.stdout.is_empty(), "{out:?}");
        let stderr = String::from_utf8(out.stderr).unwrap();
        assert!(
            stderr.contains(crate::harness::goose::HOST_ENV_VAR),
            "{stderr}"
        );
    }
}
