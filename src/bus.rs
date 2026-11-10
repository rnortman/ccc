//! The bus: this process's connection to the agent tree of the usage broker
//! (`<topic_prefix>/agents/<publisher>/<host>/…`, specified in the
//! `claude-usage-mqtt` README), through `claude-usage-mqtt-client`'s
//! `AgentClient`.
//!
//! Nothing here can block or fail a spawn. One background thread owns the
//! client; the rest of `ccc` talks to it through a channel. Publishing never
//! waits and never fails. Only the budget read and `close` wait, each with a
//! deadline. A connection that does not come up within `connect_timeout_ms`
//! marks the bus dead for the rest of the process: later publishes are
//! dropped, reads answer "unavailable" at once, and one note says so.
//!
//! TODO(agent-tree-session-coordination): nothing in `ccc` reads the sessions
//! other processes publish; coordinating through them needs a policy first.

use std::future::Future;
use std::sync::{mpsc as std_mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use claude_usage_mqtt_client::{AgentClient, AgentClientConfig, AgentError, ClientConfig};
use claude_usage_mqtt_core::{AccountKind, AccountSpend, SessionRecord, SessionState};
use jiff::Timestamp;
use tokio::sync::mpsc;

use crate::audit::{EndPayload, Lineage, SpawnPayload, StartPayload};
use crate::budget::Identity;
use crate::config::{ClaudeAuth, Config, EnvSource};

/// What a budget read found on the broker.
#[derive(Debug, Clone, PartialEq)]
pub enum Remote {
    /// The bus is off; this host's ledgers are the whole picture.
    Off,
    /// The bus is on but the broker could not be read, for this reason.
    Unavailable(String),
    /// Every publisher's and host's account records.
    Records(Vec<AccountSpend>),
}

/// This process's view of the agent tree. Every method takes `&self`, so a
/// bus can sit behind the `Sync` context the supervisor's callback captures.
pub trait Bus: Sync {
    /// The publisher and host this bus writes as; `None` when it is off.
    fn identity(&self) -> Option<&Identity>;

    /// Publishes or updates this invocation's running record; the client
    /// heartbeats it from then on. Never waits.
    fn put_session(&self, record: SessionRecord);

    /// Publishes the final record on `ended` and withdraws the running one.
    /// Never waits.
    fn end_session(&self, record: SessionRecord);

    /// Publishes this host's level for one account. Never waits.
    fn put_account(&self, spend: AccountSpend);

    /// The budget read: puts `levels` (this host's level for every
    /// configured key), waits briefly for them to be acknowledged, then
    /// reads every account record. Bounded by `timeout` as a whole. The
    /// reason a read is unavailable is among the bus's notes.
    fn fetch_accounts(
        &self,
        levels: Vec<AccountSpend>,
        quiesce: Duration,
        timeout: Duration,
    ) -> Remote;

    /// Waits at most `flush_timeout_ms` for what is queued to reach the
    /// broker, then disconnects. Returns every note the bus raised in this
    /// process; none means everything published was acknowledged. Nothing
    /// else reports a note: the caller delivers these, on every exit.
    fn close(&self) -> Vec<String>;
}

/// The bus when `[bus] enabled` is false.
pub struct NoBus;

impl Bus for NoBus {
    fn identity(&self) -> Option<&Identity> {
        None
    }

    fn put_session(&self, _record: SessionRecord) {}

    fn end_session(&self, _record: SessionRecord) {}

    fn put_account(&self, _spend: AccountSpend) {}

    fn fetch_accounts(&self, _: Vec<AccountSpend>, _: Duration, _: Duration) -> Remote {
        Remote::Off
    }

    fn close(&self) -> Vec<String> {
        Vec::new()
    }
}

/// The bus for `cfg`: the no-op when `[bus]` is off, otherwise a
/// connection to the `[mqtt]` broker under `client_id`. Broker settings that
/// cannot be resolved give a bus that is dead from the start, with the note.
pub fn open(cfg: &Config, env: &dyn EnvSource, client_id: String) -> Box<dyn Bus> {
    if !cfg.bus.enabled {
        return Box::new(NoBus);
    }
    let identity = Identity {
        publisher: cfg.bus.publisher.clone(),
        host: cfg.bus.host.clone().unwrap_or_else(system_host),
    };
    let timeouts = Timeouts {
        connect: Duration::from_millis(cfg.bus.connect_timeout_ms),
        flush: Duration::from_millis(cfg.bus.flush_timeout_ms),
    };
    match crate::usage::broker_config(&cfg.mqtt, env) {
        Ok(mut broker) => {
            broker.client_id = client_id;
            Box::new(MqttBus::start(broker, identity, timeouts))
        }
        Err(e) => Box::new(MqttBus::dead(
            identity,
            timeouts,
            format!("the bus has no broker settings ({e:#}); nothing is published by this process"),
        )),
    }
}

/// The system hostname made topic-safe: every character outside
/// `[A-Za-z0-9._-]` becomes `_`.
pub fn system_host() -> String {
    let mut buf = [0u8; 256];
    // SAFETY: the buffer is valid for its length; gethostname writes at most
    // that many bytes.
    let rc = unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) };
    let raw = if rc == 0 {
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        String::from_utf8_lossy(&buf[..end]).into_owned()
    } else {
        String::new()
    };
    sanitise_host(&raw)
}

fn sanitise_host(raw: &str) -> String {
    let host: String = raw
        .chars()
        .map(|c| {
            if claude_usage_mqtt_core::is_topic_safe(c.encode_utf8(&mut [0; 4])) {
                c
            } else {
                '_'
            }
        })
        .collect();
    if host.is_empty() {
        "unknown-host".to_string()
    } else {
        host
    }
}

/// This invocation's session record, as last published. Every change goes
/// to the bus at once; with the bus off there is no record and nothing is
/// published.
pub struct Live {
    record: Mutex<Option<SessionRecord>>,
}

impl Live {
    /// No record: the bus is off.
    #[cfg(test)]
    pub const fn off() -> Live {
        Live {
            record: Mutex::new(None),
        }
    }

    /// The record of the invocation `start` opens, before anything is
    /// resolved: `id` the invocation id, `parent` and `root` its lineage,
    /// `title` the first `prompt_chars` characters of the prompt. `None` for
    /// `identity` is the bus off.
    pub fn new(
        identity: Option<&Identity>,
        lineage: &Lineage,
        start: &StartPayload,
        agents_config: &str,
        prompt_chars: usize,
        now: Timestamp,
    ) -> Live {
        let record = identity.map(|me| {
            let mut record = SessionRecord::running(
                lineage.invocation_id.clone(),
                me.publisher.clone(),
                me.host.clone(),
                now,
            );
            record.parent = lineage.parent.clone();
            record.root = Some(lineage.trace_id.clone());
            record.pid = Some(start.pid);
            record.agent = start.agent.clone();
            record.title = start
                .prompt
                .as_deref()
                .filter(|_| prompt_chars > 0)
                .map(|prompt| prompt.chars().take(prompt_chars).collect());
            record.meta.insert("cmd".into(), json(&start.cmd));
            record
                .meta
                .insert("agents_config".into(), agents_config.into());
            record
        });
        Live {
            record: Mutex::new(record),
        }
    }

    /// Applies `change` to the record, stamps it and returns a copy to
    /// publish; `None` when the bus is off.
    fn update(
        &self,
        now: Timestamp,
        change: impl FnOnce(&mut SessionRecord),
    ) -> Option<SessionRecord> {
        let mut guard = self.record.lock().unwrap();
        let record = guard.as_mut()?;
        change(record);
        record.updated_at = now;
        Some(record.clone())
    }

    /// A harness is about to be spawned as `spawn` describes: the record
    /// takes its harness, model, agent, cwd, account and session, and is put.
    pub fn spawned(&self, bus: &dyn Bus, spawn: &SpawnPayload, now: Timestamp) {
        let put = self.update(now, |record| {
            record.harness = Some(spawn.harness.as_str().to_string());
            record.model = Some(spawn.model.clone());
            record.agent = spawn.agent.clone();
            record.cwd = Some(spawn.cwd.clone());
            record.account = spawn.profile.clone();
            record.account_kind = spawn.auth.map(|auth| match auth {
                ClaudeAuth::ApiKey => AccountKind::ApiKey,
                ClaudeAuth::ConfigDir | ClaudeAuth::Token => AccountKind::Subscription,
            });
            if spawn.session.is_some() {
                record.session = spawn.session.clone();
            }
            set_meta(record, "effort", spawn.effort.map(|e| json(&e)));
            set_meta(record, "auth", spawn.auth.map(|a| json(&a)));
        });
        if let Some(record) = put {
            bus.put_session(record);
        }
    }

    /// A turn reported `cost`: it is added to the invocation's `cost_usd`
    /// and the record is put. A negative or non-finite figure is no cost.
    pub fn charged(&self, bus: &dyn Bus, cost: Option<f64>, now: Timestamp) {
        let Some(cost) = cost.filter(|c| c.is_finite() && *c >= 0.0) else {
            return;
        };
        let put = self.update(now, |record| {
            record.cost_usd = Some(record.cost_usd.unwrap_or(0.0) + cost);
        });
        if let Some(record) = put {
            bus.put_session(record);
        }
    }

    /// The invocation ended as `end` records: the final record goes to
    /// `ended` and the running one is withdrawn.
    pub fn ended(&self, bus: &dyn Bus, end: &EndPayload, now: Timestamp) {
        let put = self.update(now, |record| {
            record.state = SessionState::Ended;
            record.ended_at = Some(now);
            record.duration_ms = Some(
                now.duration_since(record.started_at)
                    .as_millis()
                    .try_into()
                    .unwrap_or(0),
            );
            record.ok = Some(end.exit == Some(0));
            if end.session.is_some() {
                record.session = end.session.clone();
            }
            set_meta(record, "exit", end.exit.map(Into::into));
            set_meta(record, "turns", end.turns.map(Into::into));
        });
        if let Some(record) = put {
            bus.end_session(record);
        }
    }

    #[cfg(test)]
    pub fn record(&self) -> Option<SessionRecord> {
        self.record.lock().unwrap().clone()
    }
}

fn json<T: serde::Serialize>(value: &T) -> serde_json::Value {
    serde_json::to_value(value).unwrap_or(serde_json::Value::Null)
}

/// Sets `key` under `meta`, or removes it for `None`.
fn set_meta(record: &mut SessionRecord, key: &str, value: Option<serde_json::Value>) {
    match value {
        Some(value) => record.meta.insert(key.into(), value),
        None => record.meta.remove(key),
    };
}

#[derive(Debug, Clone, Copy)]
pub struct Timeouts {
    pub connect: Duration,
    pub flush: Duration,
}

/// The notes a bus has raised, and whether it is dead. Shared by the
/// caller's side and the bus thread.
#[derive(Default)]
struct Shared {
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    /// Why the bus is dead, once it is.
    dead: Option<String>,
    notes: Vec<String>,
}

impl Shared {
    /// Records `note`, unless the same note was raised before.
    fn raise(&self, note: String) {
        let mut state = self.state.lock().unwrap();
        if !state.notes.contains(&note) {
            state.notes.push(note);
        }
    }

    fn die(&self, note: String) {
        self.state.lock().unwrap().dead = Some(note.clone());
        self.raise(note);
    }

    fn dead(&self) -> Option<String> {
        self.state.lock().unwrap().dead.clone()
    }
}

enum Cmd {
    PutSession(Box<SessionRecord>),
    EndSession(Box<SessionRecord>),
    PutAccount(Box<AccountSpend>),
    Fetch {
        levels: Vec<AccountSpend>,
        quiesce: Duration,
        timeout: Duration,
        reply: std_mpsc::Sender<Remote>,
    },
    Close {
        deadline: Instant,
        reply: std_mpsc::Sender<()>,
    },
    #[cfg(test)]
    SignalsBlocked(std_mpsc::Sender<bool>),
}

/// The bus over a real broker connection.
pub struct MqttBus {
    identity: Identity,
    /// `None` when the bus was dead before a thread was started.
    tx: Option<mpsc::UnboundedSender<Cmd>>,
    shared: Arc<Shared>,
    timeouts: Timeouts,
}

impl MqttBus {
    /// Starts the bus thread, which connects to `broker` as `identity`.
    pub fn start(broker: ClientConfig, identity: Identity, timeouts: Timeouts) -> MqttBus {
        let prefix = broker.topic_prefix.clone();
        let cfg = AgentClientConfig::new(broker, &identity.publisher, &identity.host);
        MqttBus::start_with(identity, prefix, timeouts, move || {
            AgentClient::connect(cfg)
        })
    }

    fn dead(identity: Identity, timeouts: Timeouts, note: String) -> MqttBus {
        let shared = Arc::new(Shared::default());
        shared.die(note);
        MqttBus {
            identity,
            tx: None,
            shared,
            timeouts,
        }
    }

    fn start_with<P, F, Fut>(
        identity: Identity,
        prefix: String,
        timeouts: Timeouts,
        connect: F,
    ) -> MqttBus
    where
        P: Port + 'static,
        F: FnOnce() -> Fut + Send + 'static,
        Fut: Future<Output = Result<P, AgentError>> + 'static,
    {
        let shared = Arc::new(Shared::default());
        let (tx, rx) = mpsc::unbounded_channel();
        let acl = format!("{prefix}/agents/{}/#", identity.publisher);
        let thread_shared = Arc::clone(&shared);
        let spawned = std::thread::Builder::new()
            .name("ccc-bus".into())
            .spawn(move || run(connect, timeouts, acl, rx, thread_shared));
        let tx = match spawned {
            Ok(_) => Some(tx),
            Err(e) => {
                shared.die(format!(
                    "the bus thread could not start ({e}); nothing is published by this process"
                ));
                None
            }
        };
        MqttBus {
            identity,
            tx,
            shared,
            timeouts,
        }
    }

    /// The channel to a live bus thread, or `None` when the bus is dead.
    fn live(&self) -> Option<&mpsc::UnboundedSender<Cmd>> {
        match self.shared.dead() {
            Some(_) => None,
            None => self.tx.as_ref(),
        }
    }

    /// The budget read itself, without its note.
    fn read_accounts(
        &self,
        levels: Vec<AccountSpend>,
        quiesce: Duration,
        timeout: Duration,
    ) -> Remote {
        let Some(tx) = self.live() else {
            return Remote::Unavailable(self.shared.dead().unwrap_or_default());
        };
        let (reply, answer) = std_mpsc::channel();
        let sent = tx.send(Cmd::Fetch {
            levels,
            quiesce,
            timeout,
            reply,
        });
        if sent.is_err() {
            return Remote::Unavailable("the bus thread has stopped".into());
        }
        // The thread bounds the read by `timeout` once it starts it; this
        // bound also covers a thread still waiting for its connection.
        match answer.recv_timeout(timeout) {
            Ok(remote) => remote,
            Err(_) => match self.shared.dead() {
                Some(why) => Remote::Unavailable(why),
                None => Remote::Unavailable(format!(
                    "the broker did not answer within {} ms",
                    timeout.as_millis()
                )),
            },
        }
    }

    /// Hands a publish to a live bus thread; a dead bus drops it.
    fn send(&self, cmd: Cmd) {
        if let Some(tx) = self.live() {
            // A send fails only once the thread has gone, and the bus is
            // then as good as dead: the publish is dropped.
            let _ = tx.send(cmd);
        }
    }

    /// Whether the bus thread has `SIGTERM`, `SIGINT` and `SIGHUP` blocked,
    /// as read from inside it.
    #[cfg(test)]
    fn thread_blocks_signals(&self) -> bool {
        let (reply, answer) = std_mpsc::channel();
        self.tx
            .as_ref()
            .unwrap()
            .send(Cmd::SignalsBlocked(reply))
            .unwrap();
        answer.recv_timeout(Duration::from_secs(5)).unwrap()
    }
}

impl Bus for MqttBus {
    fn identity(&self) -> Option<&Identity> {
        Some(&self.identity)
    }

    fn put_session(&self, record: SessionRecord) {
        self.send(Cmd::PutSession(Box::new(record)));
    }

    fn end_session(&self, record: SessionRecord) {
        self.send(Cmd::EndSession(Box::new(record)));
    }

    fn put_account(&self, spend: AccountSpend) {
        self.send(Cmd::PutAccount(Box::new(spend)));
    }

    fn fetch_accounts(
        &self,
        levels: Vec<AccountSpend>,
        quiesce: Duration,
        timeout: Duration,
    ) -> Remote {
        let remote = self.read_accounts(levels, quiesce, timeout);
        if let Remote::Unavailable(why) = &remote {
            // A dead bus's reason was raised when it died.
            if self.shared.dead().as_ref() != Some(why) {
                self.shared
                    .raise(format!("account spend not read from the broker: {why}"));
            }
        }
        remote
    }

    fn close(&self) -> Vec<String> {
        if let Some(tx) = self.live() {
            let deadline = Instant::now() + self.timeouts.flush;
            let (reply, answer) = std_mpsc::channel();
            if tx.send(Cmd::Close { deadline, reply }).is_ok() {
                let wait = deadline.saturating_duration_since(Instant::now());
                if answer.recv_timeout(wait).is_err() && self.shared.dead().is_none() {
                    self.shared.raise(format!(
                        "the bus did not finish publishing within {} ms; what was still queued is lost",
                        self.timeouts.flush.as_millis()
                    ));
                }
            }
        }
        self.shared.state.lock().unwrap().notes.clone()
    }
}

/// The broker operations the bus thread uses: [`AgentClient`]'s, and a fake's
/// in tests.
trait Port {
    async fn put_session(&self, record: SessionRecord) -> Result<(), AgentError>;
    async fn end_session(&self, record: SessionRecord) -> Result<(), AgentError>;
    async fn put_account(&self, spend: AccountSpend) -> Result<(), AgentError>;
    async fn flush(&self, timeout: Duration) -> Result<(), AgentError>;
    async fn fetch_accounts(
        &mut self,
        quiesce: Duration,
        timeout: Duration,
    ) -> Result<Vec<AccountSpend>, AgentError>;
    async fn disconnect(&self);
}

impl Port for AgentClient {
    async fn put_session(&self, record: SessionRecord) -> Result<(), AgentError> {
        AgentClient::put_session(self, record).await
    }

    async fn end_session(&self, record: SessionRecord) -> Result<(), AgentError> {
        AgentClient::end_session(self, record).await
    }

    async fn put_account(&self, spend: AccountSpend) -> Result<(), AgentError> {
        AgentClient::put_account(self, spend).await
    }

    async fn flush(&self, timeout: Duration) -> Result<(), AgentError> {
        AgentClient::flush(self, timeout).await
    }

    async fn fetch_accounts(
        &mut self,
        quiesce: Duration,
        timeout: Duration,
    ) -> Result<Vec<AccountSpend>, AgentError> {
        AgentClient::fetch_accounts(self, quiesce, timeout).await
    }

    async fn disconnect(&self) {
        AgentClient::disconnect(self).await
    }
}

/// The bus thread: blocks the supervisor's signals, connects within the
/// connect timeout, then serves commands until `close` or until every sender
/// is gone.
fn run<P, F, Fut>(
    connect: F,
    timeouts: Timeouts,
    acl: String,
    rx: mpsc::UnboundedReceiver<Cmd>,
    shared: Arc<Shared>,
) where
    P: Port,
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<P, AgentError>>,
{
    // A process-directed signal goes to any thread that has it unblocked; on
    // this one it would take the default action and kill `ccc` past the
    // supervisor. Threads the runtime starts inherit the mask.
    if let Err(e) = crate::supervise::block_signals(true) {
        shared.die(format!(
            "the bus could not block signals ({e:#}); nothing is published by this process"
        ));
        return;
    }
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            shared.die(format!(
                "the bus could not start its runtime ({e}); nothing is published by this process"
            ));
            return;
        }
    };
    runtime.block_on(async move {
        let mut rx = rx;
        let port = match tokio::time::timeout(timeouts.connect, connect()).await {
            Ok(Ok(port)) => port,
            Ok(Err(e)) => {
                shared.die(format!(
                    "the bus could not connect to the broker ({e}); nothing is published by this process"
                ));
                return drain(&mut rx, &shared).await;
            }
            Err(_) => {
                shared.die(format!(
                    "the bus did not connect to the broker within {} ms; nothing is published by this process",
                    timeouts.connect.as_millis()
                ));
                return drain(&mut rx, &shared).await;
            }
        };
        serve(port, &mut rx, &shared, &acl).await
    });
}

/// Answers a dead bus's commands: reads are unavailable, publishes dropped.
async fn drain(rx: &mut mpsc::UnboundedReceiver<Cmd>, shared: &Shared) {
    while let Some(cmd) = rx.recv().await {
        match cmd {
            Cmd::PutSession(_) | Cmd::EndSession(_) | Cmd::PutAccount(_) => {}
            Cmd::Fetch { reply, .. } => {
                let _ = reply.send(Remote::Unavailable(shared.dead().unwrap_or_default()));
            }
            Cmd::Close { reply, .. } => {
                let _ = reply.send(());
                return;
            }
            #[cfg(test)]
            Cmd::SignalsBlocked(reply) => {
                let _ = reply.send(signals_blocked());
            }
        }
    }
}

async fn serve<P: Port>(
    mut port: P,
    rx: &mut mpsc::UnboundedReceiver<Cmd>,
    shared: &Shared,
    acl: &str,
) {
    let flushed = |result: Result<(), AgentError>, timeout: Duration| match result {
        Ok(()) => {}
        Err(AgentError::Refused { reason }) => shared.raise(format!(
            "the broker refused a publish ({reason}); check the ACL for {acl}"
        )),
        Err(AgentError::Timeout) => shared.raise(format!(
            "the broker did not acknowledge every publish within {} ms",
            timeout.as_millis()
        )),
        Err(e) => shared.raise(format!("the bus: {e}")),
    };
    while let Some(cmd) = rx.recv().await {
        let published = |result: Result<(), AgentError>| {
            if let Err(e) = result {
                shared.raise(format!("the bus could not publish ({e})"));
            }
        };
        match cmd {
            Cmd::PutSession(record) => published(port.put_session(*record).await),
            Cmd::EndSession(record) => published(port.end_session(*record).await),
            Cmd::PutAccount(spend) => published(port.put_account(*spend).await),
            Cmd::Fetch {
                levels,
                quiesce,
                timeout,
                reply,
            } => {
                let read = async {
                    for level in levels {
                        if let Err(e) = port.put_account(level).await {
                            shared.raise(format!("the bus could not publish ({e})"));
                        }
                    }
                    // Half the budget at most, so the read still runs when
                    // the levels are not acknowledged in time.
                    let flush_for = timeout / 2;
                    match port.flush(flush_for).await {
                        Err(AgentError::Timeout) => {}
                        other => flushed(other, flush_for),
                    }
                    port.fetch_accounts(quiesce, timeout).await
                };
                let remote = match tokio::time::timeout(timeout, read).await {
                    Ok(Ok(records)) => Remote::Records(records),
                    Ok(Err(e)) => Remote::Unavailable(format!("reading account spend: {e}")),
                    Err(_) => Remote::Unavailable(format!(
                        "the broker did not answer within {} ms",
                        timeout.as_millis()
                    )),
                };
                let _ = reply.send(remote);
            }
            Cmd::Close { deadline, reply } => {
                let left = deadline.saturating_duration_since(Instant::now());
                // Most of what is left for the flush, the rest for the
                // disconnect, so the reply beats the caller's deadline.
                let flush_for = left.mul_f64(0.8);
                flushed(port.flush(flush_for).await, flush_for);
                let left = deadline.saturating_duration_since(Instant::now());
                let _ = tokio::time::timeout(left, port.disconnect()).await;
                let _ = reply.send(());
                return;
            }
            #[cfg(test)]
            Cmd::SignalsBlocked(reply) => {
                let _ = reply.send(signals_blocked());
            }
        }
    }
}

#[cfg(test)]
fn signals_blocked() -> bool {
    // SAFETY: a zeroed sigset_t is valid storage for pthread_sigmask to fill.
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        if libc::pthread_sigmask(libc::SIG_BLOCK, std::ptr::null(), &mut set) != 0 {
            return false;
        }
        [libc::SIGTERM, libc::SIGINT, libc::SIGHUP]
            .iter()
            .all(|&sig| libc::sigismember(&set, sig) == 1)
    }
}

/// A bus that records what it was asked to do, for tests.
#[cfg(test)]
pub struct RecordingBus {
    identity: Identity,
    /// What `fetch_accounts` answers.
    pub remote: Remote,
    /// What `close` answers.
    pub close_notes: Vec<String>,
    pub calls: Mutex<Vec<Recorded>>,
}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub enum Recorded {
    PutSession(Box<SessionRecord>),
    EndSession(Box<SessionRecord>),
    PutAccount(Box<AccountSpend>),
    Fetch(Vec<AccountSpend>),
    Close,
}

#[cfg(test)]
impl RecordingBus {
    pub fn new(publisher: &str, host: &str, remote: Remote) -> RecordingBus {
        RecordingBus {
            identity: Identity {
                publisher: publisher.into(),
                host: host.into(),
            },
            remote,
            close_notes: Vec::new(),
            calls: Mutex::default(),
        }
    }

    pub fn calls(&self) -> Vec<Recorded> {
        self.calls.lock().unwrap().clone()
    }

    /// The session records put and ended, in order, each with whether it
    /// was the `end_session`.
    pub fn sessions(&self) -> Vec<(bool, SessionRecord)> {
        self.calls()
            .into_iter()
            .filter_map(|call| match call {
                Recorded::PutSession(r) => Some((false, *r)),
                Recorded::EndSession(r) => Some((true, *r)),
                _ => None,
            })
            .collect()
    }
}

/// A shared test bus, so a test keeps a handle to what it hands out.
#[cfg(test)]
impl<B: Bus + Send + ?Sized> Bus for Arc<B> {
    fn identity(&self) -> Option<&Identity> {
        (**self).identity()
    }

    fn put_session(&self, record: SessionRecord) {
        (**self).put_session(record)
    }

    fn end_session(&self, record: SessionRecord) {
        (**self).end_session(record)
    }

    fn put_account(&self, spend: AccountSpend) {
        (**self).put_account(spend)
    }

    fn fetch_accounts(
        &self,
        levels: Vec<AccountSpend>,
        quiesce: Duration,
        timeout: Duration,
    ) -> Remote {
        (**self).fetch_accounts(levels, quiesce, timeout)
    }

    fn close(&self) -> Vec<String> {
        (**self).close()
    }
}

#[cfg(test)]
impl Bus for RecordingBus {
    fn identity(&self) -> Option<&Identity> {
        Some(&self.identity)
    }

    fn put_session(&self, record: SessionRecord) {
        self.calls
            .lock()
            .unwrap()
            .push(Recorded::PutSession(Box::new(record)));
    }

    fn end_session(&self, record: SessionRecord) {
        self.calls
            .lock()
            .unwrap()
            .push(Recorded::EndSession(Box::new(record)));
    }

    fn put_account(&self, spend: AccountSpend) {
        self.calls
            .lock()
            .unwrap()
            .push(Recorded::PutAccount(Box::new(spend)));
    }

    fn fetch_accounts(&self, levels: Vec<AccountSpend>, _: Duration, _: Duration) -> Remote {
        self.calls.lock().unwrap().push(Recorded::Fetch(levels));
        self.remote.clone()
    }

    fn close(&self) -> Vec<String> {
        self.calls.lock().unwrap().push(Recorded::Close);
        self.close_notes.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use claude_usage_mqtt_core::AccountKind;

    fn me() -> Identity {
        Identity {
            publisher: "ccc".into(),
            host: "atlas".into(),
        }
    }

    fn level(account: &str, usd: f64) -> AccountSpend {
        AccountSpend::new(
            account,
            AccountKind::ApiKey,
            "ccc",
            "atlas",
            "2026-11",
            usd,
            "2026-11-10T00:00:00Z".parse().unwrap(),
        )
    }

    fn broker_at(port: u16) -> ClientConfig {
        ClientConfig {
            host: "127.0.0.1".into(),
            port,
            tls: false,
            client_id: "ccc-bus-test".into(),
            ..ClientConfig::default()
        }
    }

    const QUICK: Duration = Duration::from_millis(50);

    /// Waits, at most a bounded number of short sleeps, for the bus to die.
    fn wait_dead(bus: &MqttBus, within: Duration) -> bool {
        let step = Duration::from_millis(20);
        for _ in 0..(within.as_millis() / step.as_millis()).max(1) {
            if bus.shared.dead().is_some() {
                return true;
            }
            std::thread::sleep(step);
        }
        bus.shared.dead().is_some()
    }

    #[test]
    fn hostnames_are_made_topic_safe() {
        assert_eq!(sanitise_host("atlas.lan"), "atlas.lan");
        assert_eq!(sanitise_host("my host/1+x"), "my_host_1_x");
        assert_eq!(sanitise_host("hôte"), "h_te");
        assert_eq!(sanitise_host(""), "unknown-host");
        assert!(claude_usage_mqtt_core::is_topic_safe(&system_host()));
    }

    #[test]
    fn a_refused_connection_kills_the_bus_with_one_note() {
        let port = {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().port()
        };
        let timeouts = Timeouts {
            connect: Duration::from_secs(5),
            flush: Duration::from_secs(1),
        };
        let bus = MqttBus::start(broker_at(port), me(), timeouts);
        assert!(wait_dead(&bus, Duration::from_secs(5)));
        assert!(bus.thread_blocks_signals());

        let started = Instant::now();
        bus.put_account(level("team", 1.0));
        match bus.fetch_accounts(vec![level("team", 1.0)], QUICK, Duration::from_secs(5)) {
            Remote::Unavailable(why) => assert!(why.contains("could not connect"), "{why}"),
            other => panic!("{other:?}"),
        }
        let notes = bus.close();
        assert!(started.elapsed() < Duration::from_millis(500));
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].contains("nothing is published"), "{notes:?}");
    }

    #[test]
    fn a_silent_broker_kills_the_bus_at_the_connect_timeout_and_close_is_bounded() {
        // Bound and kept open, never accepted: the connect gets no ConnAck.
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let timeouts = Timeouts {
            connect: Duration::from_millis(200),
            flush: Duration::from_millis(300),
        };
        let started = Instant::now();
        let bus = MqttBus::start(broker_at(port), me(), timeouts);
        assert!(wait_dead(&bus, Duration::from_secs(1)));
        assert!(started.elapsed() < Duration::from_secs(1));
        let notes = bus.close();
        assert!(notes[0].contains("within 200 ms"), "{notes:?}");

        // Closed while still connecting: back within the flush timeout.
        let timeouts = Timeouts {
            connect: Duration::from_secs(3),
            flush: Duration::from_millis(300),
        };
        let bus = MqttBus::start(broker_at(port), me(), timeouts);
        let started = Instant::now();
        let notes = bus.close();
        let took = started.elapsed();
        assert!(took < Duration::from_millis(450), "{took:?}");
        assert!(notes[0].contains("did not finish publishing"), "{notes:?}");
        drop(listener);
    }

    type Log = Arc<Mutex<Vec<String>>>;

    /// A broker stand-in behind the real bus thread.
    struct FakePort {
        log: Arc<Mutex<Vec<String>>>,
        refuse: Option<String>,
        records: Vec<AccountSpend>,
        /// The read fails.
        unreadable: bool,
    }

    impl Port for FakePort {
        async fn put_session(&self, record: SessionRecord) -> Result<(), AgentError> {
            self.log
                .lock()
                .unwrap()
                .push(format!("session {}", record.id));
            Ok(())
        }

        async fn end_session(&self, record: SessionRecord) -> Result<(), AgentError> {
            self.log.lock().unwrap().push(format!("end {}", record.id));
            Ok(())
        }

        async fn put_account(&self, spend: AccountSpend) -> Result<(), AgentError> {
            self.log
                .lock()
                .unwrap()
                .push(format!("put {} {}", spend.account, spend.usd));
            Ok(())
        }

        async fn flush(&self, _: Duration) -> Result<(), AgentError> {
            self.log.lock().unwrap().push("flush".into());
            match &self.refuse {
                Some(reason) => Err(AgentError::Refused {
                    reason: reason.clone(),
                }),
                None => Ok(()),
            }
        }

        async fn fetch_accounts(
            &mut self,
            _: Duration,
            _: Duration,
        ) -> Result<Vec<AccountSpend>, AgentError> {
            self.log.lock().unwrap().push("fetch".into());
            if self.unreadable {
                return Err(AgentError::Timeout);
            }
            Ok(self.records.clone())
        }

        async fn disconnect(&self) {
            self.log.lock().unwrap().push("disconnect".into());
        }
    }

    fn fake_bus(refuse: Option<&str>, records: Vec<AccountSpend>) -> (MqttBus, Log) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let port = FakePort {
            log: Arc::clone(&log),
            refuse: refuse.map(str::to_string),
            records,
            unreadable: false,
        };
        (start_fake(port), log)
    }

    fn start_fake(port: FakePort) -> MqttBus {
        let timeouts = Timeouts {
            connect: Duration::from_secs(5),
            flush: Duration::from_secs(5),
        };
        MqttBus::start_with(me(), "claude/usage".into(), timeouts, move || async move {
            Ok::<_, AgentError>(port)
        })
    }

    #[test]
    fn an_unreadable_broker_on_a_live_bus_is_one_note_however_often_it_is_read() {
        let port = FakePort {
            log: Arc::new(Mutex::new(Vec::new())),
            refuse: None,
            records: Vec::new(),
            unreadable: true,
        };
        let bus = start_fake(port);
        for _ in 0..2 {
            match bus.fetch_accounts(vec![level("team", 1.0)], QUICK, Duration::from_secs(5)) {
                Remote::Unavailable(why) => {
                    assert!(why.starts_with("reading account spend"), "{why}")
                }
                other => panic!("{other:?}"),
            }
        }
        let notes = bus.close();
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(
            notes[0].starts_with("account spend not read from the broker: reading account spend"),
            "{notes:?}"
        );
    }

    #[test]
    fn the_budget_read_puts_every_level_before_it_fetches_and_publishes_reach_the_client_in_order()
    {
        let other = AccountSpend::new(
            "team",
            AccountKind::ApiKey,
            "ccc",
            "zeus",
            "2026-11",
            40.0,
            "2026-11-10T00:00:00Z".parse().unwrap(),
        );
        let (bus, log) = fake_bus(None, vec![other.clone()]);
        let remote = bus.fetch_accounts(
            vec![level("team", 1.5), level("lab", 0.0)],
            QUICK,
            Duration::from_secs(5),
        );
        assert_eq!(remote, Remote::Records(vec![other]));
        bus.put_account(level("team", 2.0));
        let record = SessionRecord::running("inv-1", "ccc", "atlas", now());
        bus.put_session(record.clone());
        bus.end_session(record);
        assert_eq!(bus.close(), Vec::<String>::new());
        assert_eq!(
            *log.lock().unwrap(),
            [
                "put team 1.5",
                "put lab 0",
                "flush",
                "fetch",
                "put team 2",
                "session inv-1",
                "end inv-1",
                "flush",
                "disconnect"
            ]
        );
    }

    #[test]
    fn a_refused_publish_is_one_acl_note_and_the_read_still_runs() {
        let (bus, log) = fake_bus(Some("NotAuthorized"), Vec::new());
        for _ in 0..2 {
            let remote =
                bus.fetch_accounts(vec![level("team", 1.0)], QUICK, Duration::from_secs(5));
            assert_eq!(remote, Remote::Records(Vec::new()));
        }
        let notes = bus.close();
        assert_eq!(
            notes,
            ["the broker refused a publish (NotAuthorized); check the ACL for claude/usage/agents/ccc/#"]
        );
        assert_eq!(
            log.lock().unwrap().iter().filter(|l| *l == "fetch").count(),
            2
        );
    }

    fn lineage(trace: &str, parent: Option<&str>) -> Lineage {
        Lineage {
            trace_id: trace.into(),
            invocation_id: "inv-1".into(),
            parent: parent.map(str::to_string),
            index_policy: crate::audit::IndexPolicy::Append,
        }
    }

    fn start(prompt: Option<&str>) -> StartPayload {
        StartPayload {
            pid: 4242,
            cmd: crate::audit::Command::Resume,
            agent: None,
            session: None,
            cwd_arg: None,
            profile_arg: None,
            harness_arg: None,
            json: false,
            prompt_source: None,
            prompt_file: None,
            prompt: prompt.map(str::to_string),
            passthrough: vec![],
            fork: false,
        }
    }

    fn now() -> Timestamp {
        "2026-11-10T12:00:00Z".parse().unwrap()
    }

    #[test]
    fn the_record_takes_its_tree_from_the_lineage() {
        let me = me();
        let cases = [
            // No inherited lineage: the trace is the invocation's own id.
            (lineage("inv-1", None), Some("inv-1"), None),
            // Trace-only: the grouping comes from outside, no parent.
            (lineage("outside", None), Some("outside"), None),
            // Trace and parent inherited from a parent invocation.
            (lineage("tr", Some("inv-0")), Some("tr"), Some("inv-0")),
        ];
        for (lineage, root, parent) in cases {
            let live = Live::new(Some(&me), &lineage, &start(None), "/a.toml", 0, now());
            let record = live.record().unwrap();
            assert_eq!(record.id, "inv-1");
            assert_eq!(record.root.as_deref(), root);
            assert_eq!(record.parent.as_deref(), parent);
            assert_eq!(record.pid, Some(4242));
            assert_eq!(record.meta["cmd"], "resume");
            assert_eq!(record.meta["agents_config"], "/a.toml");
        }
        let off = Live::new(None, &lineage("inv-1", None), &start(None), "/a", 0, now());
        assert!(off.record().is_none());
    }

    #[test]
    fn the_title_is_the_prompt_cut_on_a_character_boundary_and_absent_at_zero() {
        let title = |prompt: &str, n: usize| {
            Live::new(
                Some(&me()),
                &lineage("t", None),
                &start(Some(prompt)),
                "/a",
                n,
                now(),
            )
            .record()
            .unwrap()
            .title
        };
        assert_eq!(title("résumé the plan", 0), None);
        assert_eq!(title("résumé the plan", 2).as_deref(), Some("ré"));
        assert_eq!(title("日本語のテキスト", 3).as_deref(), Some("日本語"));
        assert_eq!(title("short", 50).as_deref(), Some("short"));
    }

    #[test]
    fn a_spawn_sets_the_resolved_fields_and_drops_the_account_of_a_codex_turn() {
        let bus = RecordingBus::new("ccc", "atlas", Remote::Off);
        let live = Live::new(
            Some(&me()),
            &lineage("t", None),
            &start(None),
            "/a",
            0,
            now(),
        );
        let spawn = |harness, profile: Option<&str>, auth| SpawnPayload {
            harness,
            model: "m".into(),
            effort: Some(crate::config::Effort::High),
            profile: profile.map(str::to_string),
            auth,
            sandbox: crate::agents::Sandbox::ReadOnly,
            tools: None,
            agent: Some("explorer".into()),
            cwd: "/work".into(),
            session: Some("s-1".into()),
            selection_notes: vec![],
            goose_policy: None,
        };
        live.spawned(
            &bus,
            &spawn(
                crate::config::Harness::Claude,
                Some("team03"),
                Some(ClaudeAuth::Token),
            ),
            now(),
        );
        live.spawned(
            &bus,
            &spawn(crate::config::Harness::Codex, None, None),
            now(),
        );
        let puts = bus.sessions();
        let (claude, codex) = (&puts[0].1, &puts[1].1);
        assert_eq!(claude.account.as_deref(), Some("team03"));
        assert_eq!(claude.account_kind, Some(AccountKind::Subscription));
        assert_eq!(claude.meta["auth"], "token");
        assert_eq!(claude.meta["effort"], "high");
        assert_eq!(codex.harness.as_deref(), Some("codex"));
        assert_eq!(
            (codex.account.as_ref(), codex.account_kind.as_ref()),
            (None, None)
        );
        assert!(!codex.meta.contains_key("auth"));
        assert_eq!(codex.session.as_deref(), Some("s-1"));

        // No cost reported, or a nonsensical one, leaves cost_usd absent.
        live.charged(&bus, None, now());
        live.charged(&bus, Some(f64::NAN), now());
        assert_eq!(bus.sessions().len(), 2);
        live.charged(&bus, Some(0.5), now());
        live.charged(&bus, Some(0.25), now());
        assert_eq!(bus.sessions().last().unwrap().1.cost_usd, Some(0.75));

        let end = EndPayload {
            exit: Some(3),
            signal: Some(15),
            session: None,
            result: None,
            error: Some("terminated".into()),
            warnings: vec![],
            native: None,
            turns: Some(2),
        };
        let later: Timestamp = "2026-11-10T12:00:01.5Z".parse().unwrap();
        live.ended(&bus, &end, later);
        let (is_end, ended) = bus.sessions().pop().unwrap();
        assert!(is_end);
        assert_eq!(ended.state, SessionState::Ended);
        assert_eq!(ended.ok, Some(false));
        assert_eq!(ended.duration_ms, Some(1500));
        assert_eq!(ended.ended_at, Some(later));
        assert_eq!(ended.session.as_deref(), Some("s-1"));
        assert_eq!(
            (ended.meta["exit"].clone(), ended.meta["turns"].clone()),
            (3.into(), 2.into())
        );
    }

    #[test]
    fn the_no_op_bus_is_off() {
        let bus = NoBus;
        assert!(bus.identity().is_none());
        bus.put_account(level("team", 1.0));
        assert_eq!(bus.fetch_accounts(Vec::new(), QUICK, QUICK), Remote::Off);
        assert!(bus.close().is_empty());
    }

    #[test]
    fn a_disabled_bus_opens_as_the_no_op_and_missing_broker_settings_kill_it() {
        let env: std::collections::BTreeMap<String, String> =
            [("HOME".to_string(), "/nonexistent-home".to_string())]
                .into_iter()
                .collect();
        let mut cfg = Config::default();
        assert!(open(&cfg, &env, "ccc-x".into()).identity().is_none());

        cfg.bus.enabled = true;
        cfg.bus.host = Some("atlas".into());
        cfg.mqtt.config = Some("/nonexistent/client.toml".into());
        let bus = open(&cfg, &env, "ccc-x".into());
        assert_eq!(bus.identity(), Some(&me()));
        match bus.fetch_accounts(Vec::new(), QUICK, QUICK) {
            Remote::Unavailable(why) => assert!(why.contains("no broker settings"), "{why}"),
            other => panic!("{other:?}"),
        }
        let notes = bus.close();
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].contains("no broker settings"), "{notes:?}");
    }
}
