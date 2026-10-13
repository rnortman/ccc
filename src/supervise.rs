//! Signal-safe ownership and bounded I/O for harness children.

use std::io::{Read, Write};
use std::process::{Child, ExitStatus};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::harness::{Captured, Invocation};
use crate::procs::{self, Liveness, ProcId};

pub const PIPE_DRAIN: Duration = Duration::from_secs(1);

#[cfg(test)]
thread_local! { static TEST_SIGNAL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

#[cfg(test)]
pub(crate) fn set_test_signal(enabled: bool) {
    TEST_SIGNAL.with(|signal| signal.set(enabled));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TestIoStream {
    Stdout,
    Stderr,
    Prompt,
    #[cfg(test)]
    Waiter,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TestIoFault {
    Setup,
    Poll,
    Read,
    ReadAfterBytes,
    Write,
    Panic,
    SpawnPanic,
}

#[cfg(test)]
static TEST_IO_FAULT: Mutex<Option<(std::thread::ThreadId, TestIoStream, TestIoFault)>> =
    Mutex::new(None);

#[cfg(test)]
pub(crate) static TEST_IO_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
pub(crate) fn set_test_io_fault(stream: TestIoStream, fault: TestIoFault) {
    *TEST_IO_FAULT.lock().unwrap() = Some((std::thread::current().id(), stream, fault));
}

#[cfg(test)]
fn take_test_io_fault(
    owner: std::thread::ThreadId,
    stream: TestIoStream,
    fault: TestIoFault,
) -> bool {
    let mut configured = TEST_IO_FAULT.lock().unwrap();
    if *configured == Some((owner, stream, fault)) {
        *configured = None;
        true
    } else {
        false
    }
}

#[derive(Debug)]
pub struct Supervised {
    pub captured: Captured,
    pub status: ExitStatus,
    pub warnings: Vec<String>,
    pub termination: Option<Termination>,
    pub prompt: Option<PromptDelivery>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptDelivery {
    pub written: usize,
    pub total: usize,
    pub ended: PromptEnd,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromptEnd {
    Complete,
    ChildClosed,
    Stopped,
    Failed(String),
}

#[derive(Debug)]
pub struct Termination {
    pub requested: i32,
    pub by: Option<u32>,
    pub reason: &'static str,
    pub escalated: Option<&'static str>,
    pub swept: usize,
}

pub struct SuperviseOpts<'a> {
    pub detach_group: bool,
    pub grace: Duration,
    pub capture: bool,
    pub on_spawned: &'a (dyn Fn(ProcId, u32) + Send + Sync),
    pub on_signal: Option<&'a (dyn Fn(crate::audit::SignalPayload) + Send + Sync)>,
}

struct Target {
    child: ProcId,
    pgid: u32,
    group: bool,
}

struct TermState {
    termination: Option<Termination>,
    descendants: Vec<ProcId>,
    warnings: Vec<String>,
}

fn finish_term_state(
    term_state: TermState,
    opts: &SuperviseOpts<'_>,
    supervisor: &ProcId,
) -> (Option<Termination>, Vec<String>) {
    let TermState {
        mut termination,
        descendants,
        mut warnings,
    } = term_state;
    if !descendants.is_empty() {
        let (swept, sweep_warnings) = sweep_descendants_with(
            &descendants,
            |p| p.alive(),
            |p| procs::signal(p.pid, libc::SIGKILL),
        );
        if swept > 0 {
            signal_record(
                opts,
                libc::SIGKILL,
                crate::audit::SignalDestination::Descendants,
                supervisor,
                &format!("swept {swept} descendants"),
            );
        }
        warnings.extend(
            sweep_warnings
                .into_iter()
                .map(|w| format!("descendant cleanup failed: {w}")),
        );
        if let Some(term) = termination.as_mut() {
            term.swept = swept;
        }
    }
    (termination, warnings)
}

#[cfg(unix)]
fn managed_signals(group: bool) -> [libc::c_int; 3] {
    if group {
        [libc::SIGTERM, libc::SIGINT, libc::SIGHUP]
    } else {
        [libc::SIGTERM, libc::SIGHUP, 0]
    }
}

#[cfg(unix)]
fn block_signals(group: bool) -> Result<()> {
    unsafe {
        let signals = managed_signals(group);
        let mut set = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        for sig in signals {
            if sig != 0 {
                libc::sigaddset(&mut set, sig);
            }
        }
        if libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut()) != 0 {
            return Err(std::io::Error::last_os_error()).context("blocking harness signals");
        }
    }
    Ok(())
}

#[cfg(not(unix))]
fn block_signals(_group: bool) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn spawn_child(inv: &Invocation, opts: &SuperviseOpts<'_>) -> Result<Child> {
    use std::os::unix::process::CommandExt;
    let mut cmd = inv.command();
    if opts.capture {
        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
    }
    unsafe {
        cmd.pre_exec(|| {
            let mut set = std::mem::zeroed();
            if libc::sigemptyset(&mut set) != 0
                || libc::pthread_sigmask(libc::SIG_SETMASK, &set, std::ptr::null_mut()) != 0
            {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    if opts.detach_group {
        cmd.process_group(0);
    }
    cmd.spawn()
        .with_context(|| format!("spawning {}", inv.program))
}

#[cfg(not(unix))]
fn spawn_child(inv: &Invocation, opts: &SuperviseOpts<'_>) -> Result<Child> {
    let mut cmd = inv.command();
    if opts.capture {
        cmd.stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
    }
    cmd.spawn()
        .with_context(|| format!("spawning {}", inv.program))
}

#[cfg(unix)]
fn pgid(child: u32, group: bool) -> Result<u32> {
    if group {
        return Ok(child);
    }
    let value = unsafe { libc::getpgid(child as libc::pid_t) };
    if value < 0 {
        Err(std::io::Error::last_os_error()).context("reading harness process group")
    } else {
        Ok(value as u32)
    }
}

#[cfg(not(unix))]
fn pgid(child: u32, _group: bool) -> Result<u32> {
    Ok(child)
}

fn signal_record(
    opts: &SuperviseOpts<'_>,
    signal: i32,
    to: crate::audit::SignalDestination,
    by: &ProcId,
    reason: &str,
) {
    signal_record_with(opts.on_signal, signal, to, by, reason);
}

fn signal_record_with(
    on_signal: Option<&(dyn Fn(crate::audit::SignalPayload) + Send + Sync)>,
    signal: i32,
    to: crate::audit::SignalDestination,
    by: &ProcId,
    reason: &str,
) {
    if let Some(cb) = on_signal {
        cb(crate::audit::SignalPayload {
            signal,
            to,
            by: by.clone(),
            by_inv: None,
            reason: reason.to_string(),
        });
    }
}

#[cfg(unix)]
fn send(target: &Target, signal: i32) -> std::io::Result<()> {
    if target.group {
        procs::signal_group(target.pgid, signal)
    } else {
        procs::signal(target.child.pid, signal)
    }
}

fn send_outcome(result: std::io::Result<()>) -> std::result::Result<bool, String> {
    match result {
        Ok(()) => Ok(true),
        Err(error) if error.raw_os_error() == Some(libc::ESRCH) => Ok(false),
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(not(unix))]
fn send(_target: &Target, _signal: i32) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "signals unsupported",
    ))
}

#[cfg(unix)]
fn waiter(
    target: Arc<Mutex<Option<Target>>>,
    grace: Duration,
    opts: &SuperviseOpts<'_>,
    supervisor: ProcId,
    #[cfg(test)] owner: std::thread::ThreadId,
) -> TermState {
    let mut state = TermState {
        termination: None,
        descendants: Vec::new(),
        warnings: Vec::new(),
    };
    let mut deadline = None;
    loop {
        let timeout = deadline
            .map(|d: Instant| d.saturating_duration_since(Instant::now()))
            .unwrap_or(Duration::from_secs(86400));
        let mut set = unsafe { std::mem::zeroed() };
        unsafe {
            libc::sigemptyset(&mut set);
            for sig in managed_signals(opts.detach_group) {
                if sig != 0 {
                    libc::sigaddset(&mut set, sig);
                }
            }
        }
        let ts = libc::timespec {
            tv_sec: timeout.as_secs() as _,
            tv_nsec: timeout.subsec_nanos() as _,
        };
        let mut info = unsafe { std::mem::zeroed() };
        let got = unsafe { libc::sigtimedwait(&set, &mut info, &ts) };
        let current = target.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(t) = current.as_ref() else {
            #[cfg(test)]
            if take_test_io_fault(owner, TestIoStream::Waiter, TestIoFault::Panic) {
                panic!("injected waiter panic");
            }
            break;
        };
        let expired = deadline.is_some_and(|d| Instant::now() >= d);
        let incoming = if got > 0 { Some(got) } else { None };
        if incoming.is_some() && state.termination.is_none() {
            match t.child.alive() {
                Liveness::Alive => {
                    state.descendants = procs::descendants(t.child.pid);
                    match send_outcome(send(t, libc::SIGTERM)) {
                        Ok(true) => {
                            signal_record(
                                opts,
                                libc::SIGTERM,
                                crate::audit::SignalDestination::Child,
                                &supervisor,
                                "signal to ccc",
                            );
                            state.termination = Some(Termination {
                                requested: libc::SIGTERM,
                                by: Some(unsafe { info.si_pid() } as u32).filter(|p| *p != 0),
                                reason: "signal to ccc",
                                escalated: None,
                                swept: 0,
                            });
                            deadline = Some(Instant::now() + grace);
                        }
                        Ok(false) => {}
                        Err(error) => state
                            .warnings
                            .push(format!("failed to signal harness with SIGTERM: {error}")),
                    }
                }
                Liveness::Dead => {}
                Liveness::Unknown(reason) => state
                    .warnings
                    .push(format!("could not verify harness liveness: {reason}")),
            }
        } else if (expired || (incoming.is_some() && state.termination.is_some()))
            && state
                .termination
                .as_ref()
                .is_some_and(|t| t.escalated.is_none())
        {
            let reason = if expired {
                "grace expired"
            } else {
                "second signal"
            };
            match t.child.alive() {
                Liveness::Dead => deadline = None,
                Liveness::Unknown(detail) => {
                    deadline = None;
                    state
                        .warnings
                        .push(format!("could not verify harness liveness: {detail}"));
                }
                Liveness::Alive => {
                    state.descendants.extend(procs::descendants(t.child.pid));
                    state
                        .descendants
                        .sort_by_key(|p| (p.pid, p.start, p.boot_id.clone()));
                    state.descendants.dedup();
                    match send_outcome(send(t, libc::SIGKILL)) {
                        Ok(true) => {
                            signal_record(
                                opts,
                                libc::SIGKILL,
                                crate::audit::SignalDestination::Child,
                                &supervisor,
                                reason,
                            );
                            if let Some(term) = state.termination.as_mut() {
                                term.escalated = Some(reason);
                            }
                        }
                        Ok(false) => {}
                        Err(error) => state
                            .warnings
                            .push(format!("failed to signal harness with SIGKILL: {error}")),
                    }
                    deadline = None;
                }
            }
        }
    }
    state
}

#[cfg(not(unix))]
fn waiter(
    _target: Arc<Mutex<Option<Target>>>,
    _grace: Duration,
    _opts: &SuperviseOpts<'_>,
    _supervisor: ProcId,
    #[cfg(test)] _owner: std::thread::ThreadId,
) -> TermState {
    TermState {
        termination: None,
        descendants: Vec::new(),
        warnings: Vec::new(),
    }
}

#[cfg(unix)]
fn ignore_terminal_signals() -> (libc::sighandler_t, libc::sighandler_t) {
    unsafe {
        (
            libc::signal(libc::SIGINT, libc::SIG_IGN),
            libc::signal(libc::SIGQUIT, libc::SIG_IGN),
        )
    }
}

#[cfg(unix)]
fn restore_terminal_signals(old: (libc::sighandler_t, libc::sighandler_t)) {
    unsafe {
        libc::signal(libc::SIGINT, old.0);
        libc::signal(libc::SIGQUIT, old.1);
    }
}

#[cfg(unix)]
fn set_nonblocking(fd: std::os::unix::io::RawFd) -> std::io::Result<()> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if flags < 0 {
        return Err(std::io::Error::last_os_error());
    }
    if unsafe { libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(unix)]
type LineCallback<'a> = Arc<Mutex<&'a mut (dyn FnMut(&str) + Send)>>;

#[derive(Debug)]
enum ReadPipeResult {
    Eof,
    Stopped,
    Io(std::io::Error),
}

fn read_pipe<T: Read + std::os::unix::io::AsRawFd>(
    mut file: T,
    stop: Arc<AtomicBool>,
    result: Arc<Mutex<Vec<u8>>>,
    lines: Option<LineCallback<'_>>,
    eof: Arc<AtomicBool>,
    stream: TestIoStream,
    owner: std::thread::ThreadId,
) -> ReadPipeResult {
    #[cfg(not(test))]
    let _ = (stream, owner);
    #[cfg(test)]
    if take_test_io_fault(owner, stream, TestIoFault::Panic) {
        panic!("injected capture panic");
    }
    #[cfg(test)]
    if take_test_io_fault(owner, stream, TestIoFault::Setup) {
        return ReadPipeResult::Io(std::io::Error::other("injected nonblocking setup failure"));
    }
    if let Err(error) = set_nonblocking(file.as_raw_fd()) {
        return ReadPipeResult::Io(std::io::Error::new(
            error.kind(),
            format!("setting nonblocking mode: {error}"),
        ));
    }
    let mut pending = Vec::new();
    let mut buf = [0u8; 16 * 1024];
    loop {
        let mut pfd = libc::pollfd {
            fd: file.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        #[cfg(test)]
        if take_test_io_fault(owner, stream, TestIoFault::Poll) {
            return finish_read_pipe(
                pending,
                lines.as_ref(),
                ReadPipeResult::Io(std::io::Error::other("polling pipe: injected poll failure")),
            );
        }
        let n = unsafe { libc::poll(&mut pfd, 1, 100) };
        if n < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return finish_read_pipe(
                pending,
                lines.as_ref(),
                ReadPipeResult::Io(std::io::Error::new(
                    error.kind(),
                    format!("polling pipe: {error}"),
                )),
            );
        }
        if n == 0 && stop.load(Ordering::Acquire) {
            return finish_read_pipe(pending, lines.as_ref(), ReadPipeResult::Stopped);
        }
        if n == 0 {
            continue;
        }
        if stop.load(Ordering::Acquire) {
            return finish_read_pipe(pending, lines.as_ref(), ReadPipeResult::Stopped);
        }
        loop {
            #[cfg(test)]
            if take_test_io_fault(owner, stream, TestIoFault::Read) {
                return finish_read_pipe(
                    pending,
                    lines.as_ref(),
                    ReadPipeResult::Io(std::io::Error::other(
                        "reading pipe: injected read failure",
                    )),
                );
            }
            match file.read(&mut buf) {
                Ok(0) => {
                    eof.store(true, Ordering::Release);
                    if let Some(cb) = &lines {
                        if !pending.is_empty() {
                            let s = String::from_utf8_lossy(&pending);
                            (cb.lock().unwrap())(&s);
                        }
                    }
                    return ReadPipeResult::Eof;
                }
                Ok(n) => {
                    result.lock().unwrap().extend_from_slice(&buf[..n]);
                    if let Some(line_cb) = &lines {
                        pending.extend_from_slice(&buf[..n]);
                        while let Some(i) = pending.iter().position(|b| *b == b'\n') {
                            let line = pending.drain(..=i).collect::<Vec<_>>();
                            let s = String::from_utf8_lossy(&line);
                            (line_cb.lock().unwrap())(&s);
                        }
                    }
                    #[cfg(test)]
                    if take_test_io_fault(owner, stream, TestIoFault::ReadAfterBytes) {
                        return finish_read_pipe(
                            pending,
                            lines.as_ref(),
                            ReadPipeResult::Io(std::io::Error::other(
                                "reading pipe after captured bytes: injected read failure",
                            )),
                        );
                    }
                    if stop.load(Ordering::Acquire) {
                        return finish_read_pipe(pending, lines.as_ref(), ReadPipeResult::Stopped);
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    return finish_read_pipe(
                        pending,
                        lines.as_ref(),
                        ReadPipeResult::Io(std::io::Error::new(
                            e.kind(),
                            format!("reading pipe: {e}"),
                        )),
                    )
                }
            }
        }
    }
}

fn finish_read_pipe(
    pending: Vec<u8>,
    lines: Option<&LineCallback<'_>>,
    result: ReadPipeResult,
) -> ReadPipeResult {
    if let Some(cb) = lines {
        if !pending.is_empty() {
            let s = String::from_utf8_lossy(&pending);
            (cb.lock().unwrap())(&s);
        }
    }
    result
}

#[cfg(unix)]
fn write_prompt<T: Write + std::os::unix::io::AsRawFd>(
    mut file: T,
    prompt: String,
    stop: Arc<AtomicBool>,
    stream: TestIoStream,
    owner: std::thread::ThreadId,
) -> PromptDelivery {
    let total = prompt.len();
    let failed = |written, message: String| PromptDelivery {
        written,
        total,
        ended: PromptEnd::Failed(message),
    };
    #[cfg(not(test))]
    let _ = (stream, owner);
    #[cfg(test)]
    if take_test_io_fault(owner, stream, TestIoFault::Panic) {
        panic!("injected prompt writer panic");
    }
    #[cfg(test)]
    if take_test_io_fault(owner, stream, TestIoFault::Setup) {
        return failed(
            0,
            "prompt writer setup failed: injected nonblocking setup failure".into(),
        );
    }
    if let Err(error) = set_nonblocking(file.as_raw_fd()) {
        return failed(0, format!("prompt writer setup failed: {error}"));
    }
    let bytes = prompt.as_bytes();
    let mut pos = 0;
    while pos < bytes.len() {
        let mut pfd = libc::pollfd {
            fd: file.as_raw_fd(),
            events: libc::POLLOUT,
            revents: 0,
        };
        #[cfg(test)]
        if take_test_io_fault(owner, stream, TestIoFault::Poll) {
            return failed(
                pos,
                "prompt writer poll failed: injected poll failure".into(),
            );
        }
        let n = unsafe { libc::poll(&mut pfd, 1, 100) };
        if n < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return failed(pos, format!("prompt writer poll failed: {error}"));
        }
        if stop.load(Ordering::Acquire) {
            return PromptDelivery {
                written: pos,
                total,
                ended: PromptEnd::Stopped,
            };
        }
        if n == 0 {
            continue;
        }
        #[cfg(test)]
        if take_test_io_fault(owner, stream, TestIoFault::Write) {
            return failed(
                pos,
                "prompt writer write failed: injected write failure".into(),
            );
        }
        match file.write(&bytes[pos..]) {
            Ok(0) => {
                return PromptDelivery {
                    written: pos,
                    total,
                    ended: PromptEnd::ChildClosed,
                }
            }
            Ok(n) => pos += n,
            Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => {
                return PromptDelivery {
                    written: pos,
                    total,
                    ended: PromptEnd::ChildClosed,
                };
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => {
                return failed(pos, format!("prompt writer write failed: {e}"));
            }
        }
    }
    PromptDelivery {
        written: pos,
        total,
        ended: PromptEnd::Complete,
    }
}

pub fn run(
    inv: &Invocation,
    opts: &SuperviseOpts<'_>,
    on_line: &mut (dyn FnMut(&str) + Send),
) -> Result<Supervised> {
    #[cfg(test)]
    let test_signal = TEST_SIGNAL.with(|signal| {
        signal
            .replace(false)
            .then_some(vec![(libc::SIGTERM, Duration::from_millis(50))])
    });
    #[cfg(not(test))]
    let test_signal = None;
    run_impl(inv, opts, on_line, test_signal)
}

#[cfg(test)]
fn run_signalled(
    inv: &Invocation,
    opts: &SuperviseOpts<'_>,
    on_line: &mut (dyn FnMut(&str) + Send),
    signal: i32,
) -> Result<Supervised> {
    run_impl(
        inv,
        opts,
        on_line,
        Some(vec![(signal, Duration::from_millis(50))]),
    )
}

#[cfg(test)]
fn run_signalled_schedule(
    inv: &Invocation,
    opts: &SuperviseOpts<'_>,
    on_line: &mut (dyn FnMut(&str) + Send),
    schedule: Vec<(i32, Duration)>,
) -> Result<Supervised> {
    run_impl(inv, opts, on_line, Some(schedule))
}

struct SpawnedHarness<'a> {
    child: Child,
    detach_group: bool,
    on_signal: Option<&'a (dyn Fn(crate::audit::SignalPayload) + Send + Sync)>,
    child_id: Option<ProcId>,
    supervisor: Option<ProcId>,
    reaped: bool,
    #[cfg(unix)]
    old_terminal: Option<(libc::sighandler_t, libc::sighandler_t)>,
}

impl<'a> SpawnedHarness<'a> {
    fn wait(&mut self, program: &str) -> Result<ExitStatus> {
        let status = self
            .child
            .wait()
            .with_context(|| format!("waiting for {program}"))?;
        self.reaped = true;
        Ok(status)
    }
}

impl Drop for SpawnedHarness<'_> {
    fn drop(&mut self) {
        if !self.reaped {
            #[cfg(unix)]
            {
                let result = if self.detach_group {
                    procs::signal_group(self.child.id(), libc::SIGKILL)
                } else {
                    procs::signal(self.child.id(), libc::SIGKILL)
                };
                let _ = result;
                let _ = self.child.wait();
                self.reaped = true;
            }
            #[cfg(not(unix))]
            {
                let _ = self.child.kill();
                let _ = self.child.wait();
                self.reaped = true;
            }
            if let (Some(cb), Some(by), Some(child)) = (
                self.on_signal,
                self.supervisor.as_ref(),
                self.child_id.as_ref(),
            ) {
                cb(crate::audit::SignalPayload {
                    signal: libc::SIGKILL,
                    to: crate::audit::SignalDestination::Child,
                    by: by.clone(),
                    by_inv: None,
                    reason: "supervisor failure".into(),
                });
                let _ = child;
            }
        }
        #[cfg(unix)]
        if let Some(old) = self.old_terminal.take() {
            restore_terminal_signals(old);
        }
    }
}

struct ScopeTeardown<'scope, 'a> {
    waiter: Option<std::thread::ScopedJoinHandle<'scope, TermState>>,
    waiter_thread: libc::pthread_t,
    target: Arc<Mutex<Option<Target>>>,
    stop: Arc<AtomicBool>,
    supervisor: ProcId,
    on_signal: Option<&'a (dyn Fn(crate::audit::SignalPayload) + Send + Sync)>,
}

impl ScopeTeardown<'_, '_> {
    fn release_target(&self) {
        self.target
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        #[cfg(unix)]
        unsafe {
            libc::pthread_kill(self.waiter_thread, libc::SIGTERM);
        }
    }

    fn join_waiter(&mut self) -> Result<TermState> {
        self.release_target();
        let waiter = self.waiter.take().expect("waiter already joined");
        waiter
            .join()
            .map_err(|_| anyhow::anyhow!("signal waiter panicked"))
    }
}

impl Drop for ScopeTeardown<'_, '_> {
    fn drop(&mut self) {
        if let Some(waiter) = self.waiter.take() {
            self.release_target();
            if let Ok(state) = waiter.join() {
                if !state.descendants.is_empty() {
                    let (swept, _) = sweep_descendants_with(
                        &state.descendants,
                        |p| p.alive(),
                        |p| procs::signal(p.pid, libc::SIGKILL),
                    );
                    if swept > 0 {
                        signal_record_with(
                            self.on_signal,
                            libc::SIGKILL,
                            crate::audit::SignalDestination::Descendants,
                            &self.supervisor,
                            &format!("swept {swept} descendants"),
                        );
                    }
                }
            }
        }
        self.stop.store(true, Ordering::Release);
    }
}

fn run_impl(
    inv: &Invocation,
    opts: &SuperviseOpts<'_>,
    on_line: &mut (dyn FnMut(&str) + Send),
    test_signal: Option<Vec<(i32, Duration)>>,
) -> Result<Supervised> {
    let test_io_owner = std::thread::current().id();
    #[cfg(not(test))]
    let _ = test_signal;
    block_signals(opts.detach_group)?;
    let mut harness = SpawnedHarness {
        child: spawn_child(inv, opts)?,
        detach_group: opts.detach_group,
        on_signal: opts.on_signal,
        child_id: None,
        supervisor: None,
        reaped: false,
        #[cfg(unix)]
        old_terminal: None,
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> Result<Supervised> {
            #[cfg(unix)]
            if !opts.capture {
                harness.old_terminal = Some(ignore_terminal_signals());
            }
            let child_id = ProcId::of(harness.child.id()).ok_or_else(|| {
                anyhow::anyhow!("capturing identity of harness {}", harness.child.id())
            })?;
            let supervisor = ProcId::current().context("capturing supervisor identity")?;
            harness.child_id = Some(child_id.clone());
            harness.supervisor = Some(supervisor.clone());
            let pgid = pgid(harness.child.id(), opts.detach_group)?;
            (opts.on_spawned)(child_id.clone(), pgid);
            let target = Arc::new(Mutex::new(Some(Target {
                child: child_id,
                pgid,
                group: opts.detach_group,
            })));
            let waiter_target = Arc::clone(&target);
            let grace = opts.grace;
            std::thread::scope(|scope| -> Result<Supervised> {
                let (waiter_thread, waiter_thread_rx) = std::sync::mpsc::sync_channel(1);
                #[cfg(test)]
                let (test_signal_thread, test_signal_rx) = std::sync::mpsc::sync_channel(1);
                let waiter_supervisor = supervisor.clone();
                let waiter_handle = scope.spawn(move || {
                    let thread = unsafe { libc::pthread_self() };
                    waiter_thread.send(thread).ok();
                    #[cfg(test)]
                    test_signal_thread.send(thread).ok();
                    waiter(
                        waiter_target,
                        grace,
                        opts,
                        waiter_supervisor,
                        #[cfg(test)]
                        test_io_owner,
                    )
                });
                let waiter_thread = waiter_thread_rx.recv().context("starting signal waiter")?;
                let mut teardown = ScopeTeardown {
                    waiter: Some(waiter_handle),
                    waiter_thread,
                    target: Arc::clone(&target),
                    stop: Arc::new(AtomicBool::new(false)),
                    supervisor: supervisor.clone(),
                    on_signal: opts.on_signal,
                };
                #[cfg(test)]
                let test_signal_handle = test_signal.map(|schedule| {
                    scope.spawn(move || {
                        if let Ok(thread) = test_signal_rx.recv() {
                            for (signal, delay) in schedule {
                                std::thread::sleep(delay);
                                unsafe {
                                    libc::pthread_kill(thread, signal);
                                }
                            }
                        }
                    })
                });
                let stop = Arc::clone(&teardown.stop);
                let stdout_result = Arc::new(Mutex::new(Vec::new()));
                let stderr_result = Arc::new(Mutex::new(Vec::new()));
                let out_eof = Arc::new(AtomicBool::new(!opts.capture));
                let err_eof = Arc::new(AtomicBool::new(!opts.capture));
                let mut stdout_handle = None;
                let mut stderr_handle = None;
                let mut prompt_handle = None;
                let mut prompt = None;
                let line_cb = Arc::new(Mutex::new(on_line));
                if opts.capture {
                    let out = harness
                        .child
                        .stdout
                        .take()
                        .context("stdout was not piped")?;
                    let err = harness
                        .child
                        .stderr
                        .take()
                        .context("stderr was not piped")?;
                    let s = Arc::clone(&stop);
                    let r = Arc::clone(&stdout_result);
                    let e = Arc::clone(&out_eof);
                    let cb = Arc::clone(&line_cb);
                    stdout_handle = Some(scope.spawn(move || {
                        read_pipe(out, s, r, Some(cb), e, TestIoStream::Stdout, test_io_owner)
                    }));
                    let s = Arc::clone(&stop);
                    let r = Arc::clone(&stderr_result);
                    let e = Arc::clone(&err_eof);
                    stderr_handle = Some(scope.spawn(move || {
                        read_pipe(err, s, r, None, e, TestIoStream::Stderr, test_io_owner)
                    }));
                    #[cfg(test)]
                    if take_test_io_fault(
                        test_io_owner,
                        TestIoStream::Prompt,
                        TestIoFault::SpawnPanic,
                    ) {
                        panic!("injected thread spawn failure");
                    }
                    if let Some(prompt) = inv.stdin.clone() {
                        let input = harness
                            .child
                            .stdin
                            .take()
                            .context("the prompt pipe was not opened")?;
                        let s = Arc::clone(&stop);
                        prompt_handle = Some(scope.spawn(move || {
                            write_prompt(input, prompt, s, TestIoStream::Prompt, test_io_owner)
                        }));
                    }
                }
                let status = harness.wait(&inv.program)?;
                let term_state = teardown.join_waiter()?;
                let (termination, mut warnings) = finish_term_state(term_state, opts, &supervisor);
                #[cfg(test)]
                if let Some(handle) = test_signal_handle {
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("test signal thread panicked"))?;
                }
                let deadline = Instant::now() + PIPE_DRAIN;
                while opts.capture
                    && ((!out_eof.load(Ordering::Acquire)
                        && stdout_handle.as_ref().is_some_and(|h| !h.is_finished()))
                        || (!err_eof.load(Ordering::Acquire)
                            && stderr_handle.as_ref().is_some_and(|h| !h.is_finished())))
                    && Instant::now() < deadline
                {
                    std::thread::sleep(Duration::from_millis(10));
                }
                if opts.capture
                    && !out_eof.load(Ordering::Acquire)
                    && stdout_handle.as_ref().is_some_and(|h| !h.is_finished())
                {
                    warnings.push("stdout was still held open by another process after the harness exited; capture closed after 1 s".to_string());
                }
                if opts.capture
                    && !err_eof.load(Ordering::Acquire)
                    && stderr_handle.as_ref().is_some_and(|h| !h.is_finished())
                {
                    warnings.push("stderr was still held open by another process after the harness exited; capture closed after 1 s".to_string());
                }
                stop.store(true, Ordering::Release);
                if let Some(h) = stdout_handle {
                    match h.join() {
                        Ok(ReadPipeResult::Io(error)) => {
                            warnings.push(format!("stdout capture failed: {error}"));
                        }
                        Ok(ReadPipeResult::Eof | ReadPipeResult::Stopped) => {}
                        Err(_) => warnings.push("stdout capture thread panicked".to_string()),
                    }
                }
                if let Some(h) = stderr_handle {
                    match h.join() {
                        Ok(ReadPipeResult::Io(error)) => {
                            warnings.push(format!("stderr capture failed: {error}"));
                        }
                        Ok(ReadPipeResult::Eof | ReadPipeResult::Stopped) => {}
                        Err(_) => warnings.push("stderr capture thread panicked".to_string()),
                    }
                }
                if let Some(h) = prompt_handle {
                    match h.join() {
                        Ok(delivery) => prompt = Some(delivery),
                        Err(_) => {
                            prompt = Some(PromptDelivery {
                                written: 0,
                                total: inv.stdin.as_ref().map_or(0, String::len),
                                ended: PromptEnd::Failed("prompt writer thread panicked".into()),
                            })
                        }
                    }
                }
                let stdout = String::from_utf8_lossy(&stdout_result.lock().unwrap()).into_owned();
                let stderr = String::from_utf8_lossy(&stderr_result.lock().unwrap()).into_owned();
                Ok(Supervised {
                    captured: Captured {
                        stdout,
                        stderr,
                        output_file: None,
                        exit_code: status.code(),
                    },
                    status,
                    warnings,
                    termination,
                    prompt,
                })
            })
        },
    ));
    match result {
        Ok(value) => value,
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "non-string panic payload".into());
            Err(anyhow::anyhow!("supervisor failed: {message}"))
        }
    }
}

fn sweep_descendants_with<I, S>(
    descendants: &[ProcId],
    mut identity: I,
    mut signal: S,
) -> (usize, Vec<String>)
where
    I: FnMut(&ProcId) -> Liveness,
    S: FnMut(&ProcId) -> std::io::Result<()>,
{
    let mut swept = 0;
    let mut warnings = Vec::new();
    for p in descendants {
        match identity(p) {
            Liveness::Dead => {}
            Liveness::Unknown(reason) => {
                warnings.push(format!("PID {}: unverified ({reason})", p.pid))
            }
            Liveness::Alive => match signal(p) {
                Ok(()) => swept += 1,
                Err(e) if e.raw_os_error() == Some(libc::ESRCH) => {}
                Err(e) => warnings.push(format!("PID {}: {e}", p.pid)),
            },
        }
    }
    (swept, warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::Mutex;
    use std::time::Instant;

    fn invocation(script: &str, stdin: Option<&str>) -> Invocation {
        Invocation {
            program: "/bin/sh".into(),
            args: vec!["-c".into(), script.into()],
            env_set: vec![],
            env_remove: vec![],
            cwd: PathBuf::from("/tmp"),
            stdin: stdin.map(str::to_string),
        }
    }

    #[test]
    fn supervisor_sweep_reports_unknown_and_signal_failures() {
        let descendants = vec![
            ProcId {
                pid: 10,
                start: 1,
                boot_id: "boot".into(),
            },
            ProcId {
                pid: 11,
                start: 1,
                boot_id: "boot".into(),
            },
            ProcId {
                pid: 12,
                start: 1,
                boot_id: "boot".into(),
            },
            ProcId {
                pid: 13,
                start: 1,
                boot_id: "boot".into(),
            },
            ProcId {
                pid: 14,
                start: 1,
                boot_id: "boot".into(),
            },
        ];
        let (swept, warnings) = sweep_descendants_with(
            &descendants,
            |p| match p.pid {
                10 => Liveness::Dead,
                11 => Liveness::Unknown("permission denied".into()),
                _ => Liveness::Alive,
            },
            |p| match p.pid {
                12 => Err(std::io::Error::from_raw_os_error(libc::ESRCH)),
                13 => Err(std::io::Error::from_raw_os_error(libc::EPERM)),
                _ => Ok(()),
            },
        );
        assert_eq!(swept, 1);
        assert_eq!(warnings.len(), 2);
        assert!(warnings
            .iter()
            .any(|w| w.contains("PID 11") && w.contains("unverified")));
        assert!(warnings.iter().any(|w| w.contains("PID 13")));
    }

    #[cfg(unix)]
    #[test]
    fn signal_send_outcomes_distinguish_success_race_and_failure() {
        assert_eq!(send_outcome(Ok(())), Ok(true));
        assert_eq!(
            send_outcome(Err(std::io::Error::from_raw_os_error(libc::ESRCH))),
            Ok(false)
        );
        assert_eq!(
            send_outcome(Err(std::io::Error::from_raw_os_error(libc::EPERM))),
            Err("Operation not permitted (os error 1)".into())
        );
    }

    fn captured(inv: &Invocation, grace: Duration) -> (Supervised, Vec<String>, Vec<String>) {
        let _lock = TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        captured_inner(inv, grace, None)
    }

    fn captured_signalled(
        inv: &Invocation,
        grace: Duration,
        signal: i32,
    ) -> (Supervised, Vec<String>, Vec<String>) {
        let _lock = TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        captured_inner(inv, grace, Some(signal))
    }

    fn captured_inner(
        inv: &Invocation,
        grace: Duration,
        signal: Option<i32>,
    ) -> (Supervised, Vec<String>, Vec<String>) {
        let child = Mutex::new(Vec::new());
        let signals = Mutex::new(Vec::new());
        let on_spawned = |proc: ProcId, pgid| child.lock().unwrap().push((proc, pgid));
        let on_signal = |signal: crate::audit::SignalPayload| signals.lock().unwrap().push(signal);
        let lines = Mutex::new(Vec::new());
        let mut on_line = |line: &str| lines.lock().unwrap().push(line.to_string());
        let result = match signal {
            Some(signal) => run_signalled(
                inv,
                &SuperviseOpts {
                    detach_group: true,
                    grace,
                    capture: true,
                    on_spawned: &on_spawned,
                    on_signal: Some(&on_signal),
                },
                &mut on_line,
                signal,
            ),
            None => run(
                inv,
                &SuperviseOpts {
                    detach_group: true,
                    grace,
                    capture: true,
                    on_spawned: &on_spawned,
                    on_signal: Some(&on_signal),
                },
                &mut on_line,
            ),
        }
        .unwrap();
        assert_eq!(child.lock().unwrap().len(), 1);
        let captured_lines = lines.lock().unwrap().clone();
        let signal_reasons = signals
            .lock()
            .unwrap()
            .iter()
            .map(|s| s.reason.clone())
            .collect();
        (result, captured_lines, signal_reasons)
    }

    #[test]
    fn normal_capture_streams_lines_and_reaps_without_termination() {
        let (result, lines, signals) = captured(
            &invocation("printf 'one\\ntwo\\n'; printf err >&2; exit 7", None),
            Duration::from_secs(1),
        );
        assert_eq!(result.captured.stdout, "one\ntwo\n");
        assert_eq!(result.captured.stderr, "err");
        assert_eq!(result.captured.exit_code, Some(7));
        assert_eq!(lines, vec!["one\n", "two\n"]);
        assert!(signals.is_empty());
        assert!(result.termination.is_none());
    }

    #[test]
    fn missing_executable_fails_before_a_waiter_is_created() {
        let inv = Invocation {
            program: "/no/such/ccc-harness".into(),
            args: vec![],
            env_set: vec![],
            env_remove: vec![],
            cwd: "/tmp".into(),
            stdin: None,
        };
        let on_spawned = |_: ProcId, _: u32| panic!("spawn callback called");
        let mut on_line = |_line: &str| {};
        let error = run(
            &inv,
            &SuperviseOpts {
                detach_group: true,
                grace: Duration::from_millis(200),
                capture: true,
                on_spawned: &on_spawned,
                on_signal: None,
            },
            &mut on_line,
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("spawning /no/such/ccc-harness"));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn child_signal_mask_is_empty() {
        let (result, _, _) = captured(
            &invocation(
                "grep '^SigBlk:\\s*0000000000000000' /proc/self/status",
                None,
            ),
            Duration::from_secs(1),
        );
        assert_eq!(
            result.captured.exit_code,
            Some(0),
            "{}",
            result.captured.stdout
        );
    }

    #[cfg(all(unix, target_os = "linux"))]
    fn run_launch_inner(
        name: &str,
        script: &str,
        grace: Duration,
        expected: fn(&Supervised, &[crate::audit::SignalPayload]),
    ) {
        if std::env::var("CCC_TEST_INNER").ok().as_deref() != Some(name) {
            return;
        }
        let verdict = PathBuf::from(std::env::var_os("CCC_TEST_VERDICT").expect("verdict path"));
        let _lock = TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        unsafe {
            libc::signal(libc::SIGINT, libc::SIG_DFL);
            libc::signal(libc::SIGQUIT, libc::SIG_DFL);
        }
        let mut sibling = std::process::Command::new("/bin/sleep")
            .arg("10")
            .spawn()
            .expect("spawning same-group sibling");
        let sibling_id = ProcId::of(sibling.id()).expect("capturing sibling identity");
        let helper_group = unsafe { libc::getpgrp() as u32 };
        let sibling_pgid = unsafe { libc::getpgid(sibling.id() as libc::pid_t) };
        assert_eq!(sibling_pgid, helper_group as libc::pid_t);

        let child = Mutex::new(Vec::new());
        let observed_masks = Mutex::new(Vec::new());
        let signals = Mutex::new(Vec::new());
        let on_spawned = |proc: ProcId, pgid| {
            let mask = std::fs::read_to_string("/proc/self/status")
                .expect("reading signal dispositions during supervision")
                .lines()
                .find_map(|line| line.strip_prefix("SigIgn:").map(str::trim))
                .and_then(|value| u64::from_str_radix(value, 16).ok())
                .expect("parsing signal dispositions during supervision");
            child.lock().unwrap().push((proc.clone(), pgid));
            observed_masks.lock().unwrap().push((proc, mask));
        };
        let on_signal =
            |payload: crate::audit::SignalPayload| signals.lock().unwrap().push(payload);
        let mut on_line = |_line: &str| {};
        let started = Instant::now();
        let result = run_signalled_schedule(
            &invocation(script, None),
            &SuperviseOpts {
                detach_group: false,
                grace,
                capture: false,
                on_spawned: &on_spawned,
                on_signal: Some(&on_signal),
            },
            &mut on_line,
            vec![(libc::SIGTERM, Duration::from_millis(50))],
        );
        let sibling_liveness = sibling_id.alive();
        if sibling_liveness == Liveness::Alive {
            procs::signal(sibling_id.pid, libc::SIGKILL).expect("killing same-group sibling");
        }
        sibling.wait().expect("reaping same-group sibling");
        assert_eq!(sibling_liveness, Liveness::Alive);

        let result = result.expect("launch-mode supervision");
        assert!(started.elapsed() < Duration::from_secs(3));
        let observed_masks = observed_masks.into_inner().unwrap();
        assert_eq!(observed_masks.len(), 1);
        assert_eq!(observed_masks[0].1 & 0x6, 0x6);
        let child = child.into_inner().unwrap();
        assert_eq!(child.len(), 1);
        assert_eq!(observed_masks[0].0, child[0].0);
        assert_eq!(child[0].1, helper_group);
        assert_ne!(child[0].0.pid, helper_group);
        let signal_records = signals.into_inner().unwrap();
        expected(&result, &signal_records);

        let status =
            std::fs::read_to_string("/proc/self/status").expect("reading signal dispositions");
        let sig_ign = status
            .lines()
            .find_map(|line| {
                line.strip_prefix("SigIgn:")?
                    .trim()
                    .split_once(' ')
                    .map(|(value, _)| value)
            })
            .or_else(|| {
                status
                    .lines()
                    .find_map(|line| line.strip_prefix("SigIgn:").map(str::trim))
            })
            .expect("SigIgn disposition");
        let sig_ign = u64::from_str_radix(sig_ign, 16).expect("hexadecimal SigIgn");
        assert_eq!(sig_ign & 0x6, 0);
        std::fs::write(verdict, "PASS\n").expect("writing verdict");
    }

    #[cfg(all(unix, target_os = "linux"))]
    fn launch_outer(inner: &str) {
        use std::fs::File;
        use std::process::Stdio;

        let dir = crate::testutil::TempDir::new("launch-supervision");
        let stdout = dir.path().join("stdout");
        let stderr = dir.path().join("stderr");
        let verdict = dir.path().join("verdict");
        let mut command = std::process::Command::new("timeout");
        command
            .args(["-k", "1", "9"])
            .arg(std::env::current_exe().expect("current test executable"))
            .args(["--exact", inner, "--nocapture"])
            .env("CCC_TEST_INNER", inner)
            .env("CCC_TEST_VERDICT", &verdict)
            .stdout(Stdio::from(
                File::create(&stdout).expect("creating stdout capture"),
            ))
            .stderr(Stdio::from(
                File::create(&stderr).expect("creating stderr capture"),
            ));
        let mut child =
            crate::testutil::BoundedChild::spawn(&mut command).expect("spawning bounded outer");
        let outcome = child.finish_by(Instant::now() + Duration::from_secs(10));
        let status = match outcome {
            Ok(status) => status,
            Err(error) => {
                drop(child);
                panic!(
                    "outer launch test failed: {error}; stdout={}; stderr={}",
                    std::fs::read_to_string(&stdout).unwrap_or_default(),
                    std::fs::read_to_string(&stderr).unwrap_or_default()
                );
            }
        };
        if !status.success() {
            drop(child);
            panic!(
                "outer launch test exited {status}; stdout={}; stderr={}",
                std::fs::read_to_string(&stdout).unwrap_or_default(),
                std::fs::read_to_string(&stderr).unwrap_or_default()
            );
        }
        drop(child);
        let actual = std::fs::read_to_string(&verdict).unwrap_or_default();
        assert_eq!(
            actual,
            "PASS\n",
            "unexpected verdict; stdout={}; stderr={}",
            std::fs::read_to_string(&stdout).unwrap_or_default(),
            std::fs::read_to_string(&stderr).unwrap_or_default()
        );
    }

    #[cfg(all(unix, target_os = "linux"))]
    fn handled_launch_result(result: &Supervised, signals: &[crate::audit::SignalPayload]) {
        use std::os::unix::process::ExitStatusExt;

        assert_eq!(result.status.code(), Some(1));
        assert_eq!(result.status.signal(), None);
        let term = result.termination.as_ref().expect("termination record");
        assert_eq!(term.requested, libc::SIGTERM);
        assert_eq!(term.escalated, None);
        assert_eq!(signals.len(), 2, "signals: {signals:?}");
        assert_eq!(signals[0].signal, libc::SIGTERM);
        assert!(matches!(
            signals[0].to,
            crate::audit::SignalDestination::Child
        ));
        assert_eq!(signals[0].reason, "signal to ccc");
        assert_eq!(signals[1].signal, libc::SIGKILL);
        assert!(matches!(
            signals[1].to,
            crate::audit::SignalDestination::Descendants
        ));
        assert!(signals[1].reason.starts_with("swept "));
    }

    #[cfg(all(unix, target_os = "linux"))]
    fn ignored_launch_result(result: &Supervised, signals: &[crate::audit::SignalPayload]) {
        use std::os::unix::process::ExitStatusExt;

        assert_eq!(result.status.signal(), Some(libc::SIGKILL));
        assert_eq!(result.captured.exit_code, None);
        let term = result.termination.as_ref().expect("termination record");
        assert_eq!(term.requested, libc::SIGTERM);
        assert_eq!(term.escalated, Some("grace expired"));
        assert_eq!(signals.len(), 2);
        assert_eq!(signals[0].signal, libc::SIGTERM);
        assert!(matches!(
            signals[0].to,
            crate::audit::SignalDestination::Child
        ));
        assert_eq!(signals[0].reason, "signal to ccc");
        assert_eq!(signals[1].signal, libc::SIGKILL);
        assert!(matches!(
            signals[1].to,
            crate::audit::SignalDestination::Child
        ));
        assert_eq!(signals[1].reason, "grace expired");
    }

    #[cfg(all(unix, target_os = "linux"))]
    #[test]
    fn launch_mode_forwards_term_to_child_without_signalling_group_outer() {
        launch_outer(
            "supervise::tests::launch_mode_forwards_term_to_child_without_signalling_group_inner",
        );
    }

    #[cfg(all(unix, target_os = "linux"))]
    #[test]
    fn launch_mode_forwards_term_to_child_without_signalling_group_inner() {
        run_launch_inner(
            "supervise::tests::launch_mode_forwards_term_to_child_without_signalling_group_inner",
            "trap 'exit 1' TERM; sleep 10 & wait",
            Duration::from_secs(2),
            handled_launch_result,
        );
    }

    #[cfg(all(unix, target_os = "linux"))]
    #[test]
    fn launch_mode_escalates_ignored_term_without_signalling_group_outer() {
        launch_outer(
            "supervise::tests::launch_mode_escalates_ignored_term_without_signalling_group_inner",
        );
    }

    #[cfg(all(unix, target_os = "linux"))]
    #[test]
    fn launch_mode_escalates_ignored_term_without_signalling_group_inner() {
        run_launch_inner(
            "supervise::tests::launch_mode_escalates_ignored_term_without_signalling_group_inner",
            // `exec` keeps the child a single process under every `/bin/sh`;
            // dash forks `sleep`, which the sweep would then also signal.
            "trap '' TERM; exec sleep 10",
            Duration::from_millis(200),
            ignored_launch_result,
        );
    }

    #[cfg(unix)]
    #[test]
    fn handled_signal_exits_before_grace_and_is_recorded() {
        let _lock = TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let started = Instant::now();
        let (result, _, reasons) = captured_inner(
            &invocation("trap 'exit 1' TERM; sleep 10 & wait", None),
            Duration::from_secs(2),
            Some(libc::SIGTERM),
        );
        assert!(started.elapsed() < Duration::from_secs(3));
        assert_eq!(result.captured.exit_code, Some(1));
        assert!(result.termination.is_some());
        assert!(reasons.iter().any(|r| r == "signal to ccc"));
    }

    #[cfg(unix)]
    #[test]
    fn ignored_term_is_escalated_after_grace() {
        let (result, _, reasons) = captured_signalled(
            &invocation("trap '' TERM; sleep 10", None),
            Duration::from_millis(200),
            libc::SIGTERM,
        );
        let term = result.termination.unwrap();
        assert_eq!(term.escalated, Some("grace expired"));
        assert_eq!(result.captured.exit_code, None);
        assert!(reasons.iter().any(|r| r == "grace expired"));
    }

    #[cfg(unix)]
    #[test]
    fn a_second_signal_shortens_grace_and_names_the_reason() {
        use std::os::unix::process::ExitStatusExt;

        let _lock = TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let inv = invocation("trap '' TERM; sleep 10", None);
        let on_spawned = |_: ProcId, _: u32| {};
        let mut on_line = |_line: &str| {};
        let audit = std::sync::Mutex::new(Vec::new());
        let on_signal = |payload: crate::audit::SignalPayload| {
            audit.lock().unwrap().push(payload);
        };
        let started = Instant::now();
        let result = run_signalled_schedule(
            &inv,
            &SuperviseOpts {
                detach_group: true,
                grace: Duration::from_secs(5),
                capture: true,
                on_spawned: &on_spawned,
                on_signal: Some(&on_signal),
            },
            &mut on_line,
            vec![
                (libc::SIGTERM, Duration::from_millis(50)),
                (libc::SIGTERM, Duration::from_millis(300)),
            ],
        )
        .unwrap();
        assert!(started.elapsed() < Duration::from_secs(2));
        let term = result.termination.unwrap();
        assert_eq!(term.escalated, Some("second signal"));
        assert_eq!(result.captured.exit_code, None);
        assert_eq!(result.status.signal(), Some(libc::SIGKILL));
        let audit = audit.into_inner().unwrap();
        assert_eq!(audit.len(), 2);
        assert_eq!(audit[0].signal, libc::SIGTERM);
        assert_eq!(audit[0].reason, "signal to ccc");
        assert_eq!(audit[1].signal, libc::SIGKILL);
        assert_eq!(audit[1].reason, "second signal");
    }

    #[cfg(unix)]
    #[test]
    fn out_of_group_descendant_is_identity_swept() {
        let dir = crate::testutil::TempDir::new("supervise-sweep");
        let pid_file = dir.path().join("pid");
        let ready_file = dir.path().join("ready");
        let script = format!(
            "setsid sh -c 'echo ready > {}; trap \"\" TERM; exec sleep 10' & echo $! > {}; n=0; while [ ! -f {} ] && [ $n -lt 500 ]; do sleep .01; n=$((n+1)); done; if [ ! -f {} ]; then exit 1; fi; trap 'exit 0' TERM; wait",
            ready_file.display(),
            pid_file.display(),
            ready_file.display(),
            ready_file.display(),
        );
        let (result, _, _) = captured_signalled(
            &invocation(&script, None),
            Duration::from_millis(200),
            libc::SIGTERM,
        );
        let pid: u32 = std::fs::read_to_string(&pid_file)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let term = result.termination.unwrap();
        assert_eq!(term.swept, 1);
        // The sweep's SIGKILL reaches a process the supervisor never waits for;
        // its death is asynchronous. Poll, bounded; `sleep 10` dies on its own.
        if let Some(swept) = ProcId::of(pid) {
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut liveness = swept.alive();
            while liveness != Liveness::Dead && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(10));
                liveness = swept.alive();
            }
            assert_eq!(
                liveness,
                Liveness::Dead,
                "swept out-of-group descendant {pid} (last seen {liveness:?}) survived 5 s after run returned"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn normal_finish_sweeps_descendants_without_termination() {
        let _lock = TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let mut child = std::process::Command::new("sleep")
            .arg("10")
            .spawn()
            .unwrap();
        let identity = ProcId::of(child.id()).unwrap();
        let supervisor = ProcId::current().unwrap();
        let signals = Mutex::new(Vec::new());
        let on_spawned = |_: ProcId, _: u32| {};
        let on_signal = |signal: crate::audit::SignalPayload| signals.lock().unwrap().push(signal);
        let opts = SuperviseOpts {
            detach_group: false,
            grace: Duration::from_secs(1),
            capture: false,
            on_spawned: &on_spawned,
            on_signal: Some(&on_signal),
        };
        let (termination, warnings) = finish_term_state(
            TermState {
                termination: None,
                descendants: vec![identity.clone()],
                warnings: vec!["sentinel".into()],
            },
            &opts,
            &supervisor,
        );
        assert!(termination.is_none());
        assert_eq!(warnings, vec!["sentinel"]);
        let deadline = Instant::now() + Duration::from_secs(1);
        while identity.alive() != Liveness::Dead && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(identity.alive(), Liveness::Dead);
        child.wait().unwrap();
        let signals = signals.lock().unwrap();
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].signal, libc::SIGKILL);
        assert!(matches!(
            signals[0].to,
            crate::audit::SignalDestination::Descendants
        ));
        assert_eq!(signals[0].reason, "swept 1 descendants");
    }

    #[cfg(unix)]
    #[test]
    fn inherited_pipes_are_bounded_and_report_held_streams() {
        let dir = crate::testutil::TempDir::new("supervise-pipes");
        let pid_file = dir.path().join("pid");
        let script = format!(
            "setsid sleep 10 & echo $! > {}; printf '{{\"ok\":true}}'; exit 0",
            pid_file.display()
        );
        let _lock = TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let started = Instant::now();
        let (result, _, _) =
            captured_inner(&invocation(&script, None), Duration::from_millis(200), None);
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(result
            .warnings
            .iter()
            .any(|w| w.starts_with("stdout was still held open")));
        assert!(result
            .warnings
            .iter()
            .any(|w| w.starts_with("stderr was still held open")));
        let pid: u32 = std::fs::read_to_string(&pid_file)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let identity = ProcId::of(pid).unwrap();
        assert_eq!(identity.alive(), Liveness::Alive);
        procs::signal(pid, libc::SIGKILL).unwrap();
    }

    #[test]
    fn stderr_only_holder_and_nonreading_prompt_are_bounded() {
        let dir = crate::testutil::TempDir::new("supervise-stderr");
        let pid_file = dir.path().join("pid");
        let script = format!(
            "setsid sh -c 'exec 1>&-; sleep 10' & echo $! > {}; printf ok; exit 0",
            pid_file.display()
        );
        let (result, _, _) = captured(&invocation(&script, None), Duration::from_millis(200));
        assert!(!result
            .warnings
            .iter()
            .any(|w| w.starts_with("stdout was still held open")));
        assert!(result
            .warnings
            .iter()
            .any(|w| w.starts_with("stderr was still held open")));
        let pid: u32 = std::fs::read_to_string(&pid_file)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        procs::signal(pid, libc::SIGKILL).unwrap();

        // dash points an async list's stdin at /dev/null before applying its
        // redirections, so the holder takes the prompt pipe from fd 3.
        let holder = dir.path().join("prompt-pid");
        let ready = dir.path().join("prompt-ready");
        let script = format!(
            "exec 3<&0; sh -c 'echo ready > \"$1\"; exec sleep 10' sh {} <&3 3<&- & pid=$!; n=0; while [ ! -f {} ] && [ $n -lt 500 ]; do sleep .01; n=$((n+1)); done; echo $pid > {}; exit 0",
            ready.display(),
            ready.display(),
            holder.display()
        );
        let _lock = TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let started = Instant::now();
        let (result, _, _) = captured_inner(
            &invocation(&script, Some(&"p".repeat(1024 * 1024))),
            Duration::from_millis(200),
            None,
        );
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(matches!(
            result.prompt.as_ref().map(|p| &p.ended),
            Some(PromptEnd::Stopped)
        ));
        assert!(result.prompt.as_ref().is_some_and(|p| p.written < p.total));
        let pid: u32 = std::fs::read_to_string(holder)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        procs::signal(pid, libc::SIGKILL).unwrap();
    }

    #[test]
    fn capture_worker_failures_are_explicit_and_not_held_open() {
        let _lock = TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        for (stream, fault, needle) in [
            (
                TestIoStream::Stdout,
                TestIoFault::Setup,
                "stdout capture failed:",
            ),
            (
                TestIoStream::Stderr,
                TestIoFault::Poll,
                "stderr capture failed:",
            ),
            (
                TestIoStream::Stdout,
                TestIoFault::Read,
                "stdout capture failed:",
            ),
            (
                TestIoStream::Stderr,
                TestIoFault::Panic,
                "stderr capture thread panicked",
            ),
        ] {
            set_test_io_fault(stream, fault);
            let (result, _, _) = captured_inner(
                &invocation("printf partial; exit 0", None),
                Duration::from_secs(1),
                None,
            );
            assert!(
                result
                    .warnings
                    .iter()
                    .any(|warning| warning.contains(needle)),
                "{needle}: {:?}",
                result.warnings
            );
            assert!(!result
                .warnings
                .iter()
                .any(|warning| warning.starts_with("stdout was still held open")));
            assert!(!result
                .warnings
                .iter()
                .any(|warning| warning.starts_with("stderr was still held open")));
        }
        set_test_io_fault(TestIoStream::Stdout, TestIoFault::ReadAfterBytes);
        let (result, lines, _) = captured_inner(
            &invocation("printf partial; exit 0", None),
            Duration::from_secs(1),
            None,
        );
        assert_eq!(result.captured.stdout, "partial");
        assert_eq!(lines, vec!["partial"]);
        assert!(result.warnings.iter().any(|warning| warning
            .starts_with("stdout capture failed: reading pipe after captured bytes:")));
        assert!(!result
            .warnings
            .iter()
            .any(|warning| warning.starts_with("stdout was still held open")));
        for (fault, needle) in [
            (TestIoFault::Setup, "prompt writer setup failed:"),
            (TestIoFault::Poll, "prompt writer poll failed:"),
            (TestIoFault::Write, "prompt writer write failed:"),
            (TestIoFault::Panic, "prompt writer thread panicked"),
        ] {
            set_test_io_fault(TestIoStream::Prompt, fault);
            let (result, _, _) = captured_inner(
                &invocation("cat >/dev/null", Some("prompt")),
                Duration::from_secs(1),
                None,
            );
            assert!(
                matches!(
                    result.prompt.as_ref().map(|p| &p.ended),
                    Some(PromptEnd::Failed(message)) if message.contains(needle)
                ),
                "{needle}: {:?}",
                result.prompt
            );
            if fault == TestIoFault::Panic {
                let delivery = result.prompt.as_ref().expect("prompt delivery");
                assert_eq!(delivery.written, 0);
                assert_eq!(delivery.total, "prompt".len());
                assert!(matches!(
                    delivery.ended,
                    PromptEnd::Failed(ref message) if message == "prompt writer thread panicked"
                ));
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn every_return_reader_stops_when_pipe_remains_ready() {
        use std::os::unix::io::{AsRawFd, FromRawFd};

        let mut fds = [0; 2];
        assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
        let reader = unsafe { std::fs::File::from_raw_fd(fds[0]) };
        let writer = unsafe { std::fs::File::from_raw_fd(fds[1]) };
        let probe_fd = unsafe { libc::dup(fds[0]) };
        assert!(probe_fd >= 0);
        let probe = unsafe { std::fs::File::from_raw_fd(probe_fd) };
        let stop = Arc::new(AtomicBool::new(false));
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let eof = Arc::new(AtomicBool::new(false));
        let wrote = Arc::new(AtomicBool::new(false));
        let writer_wrote = Arc::clone(&wrote);
        let writer_thread = std::thread::spawn(move || {
            let mut writer = writer;
            let mut first = vec![b'x'; 16 * 1024];
            first[..6].copy_from_slice(b"first\n");
            let mut second = vec![b'y'; 16 * 1024];
            second[..6].copy_from_slice(b"later\n");
            writer.write_all(&first).unwrap();
            writer.write_all(&second).unwrap();
            writer_wrote.store(true, Ordering::Release);
        });
        let reader_stop = Arc::clone(&stop);
        let reader_bytes = Arc::clone(&bytes);
        let reader_eof = Arc::clone(&eof);
        let (line_seen_tx, line_seen_rx) = std::sync::mpsc::sync_channel(1);
        let reader_stop_for_line = Arc::clone(&stop);
        let reader_bytes_for_line = Arc::clone(&bytes);
        let mut published = false;
        let on_line = move |_line: &str| {
            if published {
                return;
            }
            published = true;
            let captured = reader_bytes_for_line.lock().unwrap().len();
            reader_stop_for_line.store(true, Ordering::Release);
            line_seen_tx.send(captured).unwrap();
        };
        let on_line: &'static mut (dyn FnMut(&str) + Send) = Box::leak(Box::new(on_line));
        let lines = Arc::new(Mutex::new(on_line));
        let (reader_result_tx, reader_result_rx) = std::sync::mpsc::sync_channel(1);
        let ready_deadline = Instant::now() + Duration::from_secs(5);
        let writer_ready = loop {
            if wrote.load(Ordering::Acquire) {
                break true;
            }
            if Instant::now() >= ready_deadline {
                break false;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        let pipe_ready = loop {
            if Instant::now() >= ready_deadline {
                break false;
            }
            let mut pollfd = libc::pollfd {
                fd: probe.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            let result = unsafe { libc::poll(&mut pollfd, 1, 10) };
            if result > 0 && pollfd.revents & libc::POLLIN != 0 {
                break true;
            }
        };
        let reader_thread = std::thread::spawn(move || {
            let result = read_pipe(
                reader,
                reader_stop,
                reader_bytes,
                Some(lines),
                reader_eof,
                TestIoStream::Stdout,
                std::thread::current().id(),
            );
            reader_result_tx.send(result).unwrap();
        });
        let captured_at_stop = line_seen_rx.recv_timeout(Duration::from_secs(1));
        let reader_result = reader_result_rx.recv_timeout(Duration::from_secs(1));
        let captured_after = bytes.lock().unwrap().len();
        drop(probe);
        let writer_join = writer_thread.join();
        let reader_join = reader_thread.join();
        assert!(writer_ready, "writer never became ready");
        assert!(pipe_ready, "pipe never became readable");
        assert!(
            captured_at_stop.is_ok(),
            "line callback did not publish stop"
        );
        assert_eq!(captured_after, captured_at_stop.unwrap());
        assert!(
            matches!(reader_result, Ok(ReadPipeResult::Stopped)),
            "reader did not stop within one second: {reader_result:?}"
        );
        assert!(writer_join.is_ok(), "writer thread panicked");
        assert!(reader_join.is_ok(), "reader thread panicked");
    }

    #[cfg(unix)]
    #[test]
    fn spawn_panic_kills_reaps_and_audits_the_child() {
        let _lock = TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let dir = crate::testutil::TempDir::new("spawn-panic-group");
        let marker = dir.path().join("descendant");
        let child = Mutex::new(Vec::new());
        let descendant = Mutex::new(None);
        let signals = Mutex::new(Vec::new());
        let descendant_for_callback = &descendant;
        let on_spawned = |proc: ProcId, _: u32| {
            child.lock().unwrap().push(proc);
            let deadline = Instant::now() + Duration::from_secs(5);
            let pid = loop {
                if let Ok(text) = std::fs::read_to_string(&marker) {
                    if let Ok(pid) = text.trim().parse::<u32>() {
                        break pid;
                    }
                }
                assert!(Instant::now() < deadline, "descendant readiness failed");
                std::thread::sleep(Duration::from_millis(10));
            };
            *descendant_for_callback.lock().unwrap() =
                Some(ProcId::of(pid).expect("descendant identity"));
        };
        let on_signal = |signal: crate::audit::SignalPayload| signals.lock().unwrap().push(signal);
        let mut on_line = |_line: &str| {};
        set_test_io_fault(TestIoStream::Prompt, TestIoFault::SpawnPanic);
        let started = Instant::now();
        let error = run(
            &invocation(
                &format!(
                    "trap '' TERM; sleep 10 & echo $! > '{}'; wait",
                    marker.display()
                ),
                Some(&"p".repeat(1024 * 1024)),
            ),
            &SuperviseOpts {
                detach_group: true,
                grace: Duration::from_millis(200),
                capture: true,
                on_spawned: &on_spawned,
                on_signal: Some(&on_signal),
            },
            &mut on_line,
        )
        .unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(format!("{error:#}").contains("supervisor failed: injected thread spawn failure"));
        assert_eq!(child.lock().unwrap().len(), 1);
        assert_eq!(child.lock().unwrap()[0].alive(), Liveness::Dead);
        // The grandchild is not the supervisor's child: nothing waits for it,
        // so its death is asynchronous. Poll, bounded; `sleep 10` dies on its own.
        let descendant_id = descendant
            .lock()
            .unwrap()
            .take()
            .expect("descendant identity captured at readiness");
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut liveness = descendant_id.alive();
        while liveness != Liveness::Dead && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
            liveness = descendant_id.alive();
        }
        assert_eq!(
            liveness,
            Liveness::Dead,
            "in-group descendant {} (last seen {liveness:?}) survived the group SIGKILL 5 s after run returned",
            descendant_id.pid
        );
        let signals = signals.lock().unwrap();
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].signal, libc::SIGKILL);
        assert!(matches!(
            signals[0].to,
            crate::audit::SignalDestination::Child
        ));
        assert_eq!(signals[0].reason, "supervisor failure");
    }

    #[cfg(unix)]
    #[test]
    fn teardown_sweeps_descendants_without_termination_and_audits_the_sweep() {
        let _lock = TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        block_signals(false).unwrap();
        let mut child = std::process::Command::new("sleep")
            .arg("10")
            .spawn()
            .unwrap();
        let identity = ProcId::of(child.id()).unwrap();
        let supervisor = ProcId::current().unwrap();
        let target = Arc::new(Mutex::new(Some(Target {
            child: identity.clone(),
            pgid: child.id(),
            group: false,
        })));
        let signals = Mutex::new(Vec::new());
        let on_signal = |signal: crate::audit::SignalPayload| signals.lock().unwrap().push(signal);
        std::thread::scope(|scope| {
            let (thread_tx, thread_rx) = std::sync::mpsc::sync_channel(1);
            let identity_for_waiter = identity.clone();
            let waiter = scope.spawn(move || {
                thread_tx.send(unsafe { libc::pthread_self() }).unwrap();
                TermState {
                    termination: None,
                    descendants: vec![identity_for_waiter],
                    warnings: vec![],
                }
            });
            let mut teardown = ScopeTeardown {
                waiter: Some(waiter),
                waiter_thread: thread_rx.recv().unwrap(),
                target: Arc::clone(&target),
                stop: Arc::new(AtomicBool::new(false)),
                supervisor,
                on_signal: Some(&on_signal),
            };
            let _ = &mut teardown;
        });
        let deadline = Instant::now() + Duration::from_secs(1);
        while identity.alive() != Liveness::Dead && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(identity.alive(), Liveness::Dead);
        child.wait().unwrap();
        unsafe {
            libc::pthread_sigmask(libc::SIG_SETMASK, std::ptr::null(), std::ptr::null_mut());
        }
        let signals = signals.lock().unwrap();
        assert_eq!(signals.len(), 1);
        assert_eq!(signals[0].signal, libc::SIGKILL);
        assert!(matches!(
            signals[0].to,
            crate::audit::SignalDestination::Descendants
        ));
        assert_eq!(signals[0].reason, "swept 1 descendants");
    }

    #[cfg(unix)]
    #[test]
    fn on_spawned_panic_is_bounded_in_run_and_launch_and_restores_masks() {
        let _lock = TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        for (detach_group, capture) in [(true, true), (false, false)] {
            let child = Mutex::new(Vec::new());
            let on_spawned = |proc: ProcId, _: u32| {
                child.lock().unwrap().push(proc);
                panic!("injected on_spawned panic");
            };
            let mut on_line = |_line: &str| {};
            let started = Instant::now();
            let error = run(
                &invocation("sleep 10", None),
                &SuperviseOpts {
                    detach_group,
                    grace: Duration::from_millis(200),
                    capture,
                    on_spawned: &on_spawned,
                    on_signal: None,
                },
                &mut on_line,
            )
            .unwrap_err();
            assert!(started.elapsed() < Duration::from_secs(2));
            assert!(format!("{error:#}").contains("supervisor failed"));
            assert_eq!(child.lock().unwrap().len(), 1);
            assert_eq!(child.lock().unwrap()[0].alive(), Liveness::Dead);
            if !capture {
                let ignored = std::fs::read_to_string("/proc/self/status")
                    .unwrap()
                    .lines()
                    .find_map(|line| line.strip_prefix("SigIgn:").map(str::trim))
                    .and_then(|value| u64::from_str_radix(value, 16).ok())
                    .unwrap();
                assert_eq!(ignored & 0x6, 0);
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn waiter_panic_after_reap_stops_workers_without_signalling_the_child() {
        let _lock = TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let dir = crate::testutil::TempDir::new("waiter-panic");
        let holder = dir.path().join("holder");
        let script = format!(
            "setsid sh -c 'exec 1>&-; sleep 10' & echo $! > {}; exit 0",
            holder.display()
        );
        let child = Mutex::new(Vec::new());
        let signals = Mutex::new(Vec::new());
        let on_spawned = |proc: ProcId, _: u32| child.lock().unwrap().push(proc);
        let on_signal = |signal: crate::audit::SignalPayload| signals.lock().unwrap().push(signal);
        let mut on_line = |_line: &str| {};
        set_test_io_fault(TestIoStream::Waiter, TestIoFault::Panic);
        let started = Instant::now();
        let error = run(
            &invocation(&script, None),
            &SuperviseOpts {
                detach_group: true,
                grace: Duration::from_millis(200),
                capture: true,
                on_spawned: &on_spawned,
                on_signal: Some(&on_signal),
            },
            &mut on_line,
        )
        .unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(format!("{error:#}").contains("signal waiter panicked"));
        assert_eq!(child.lock().unwrap().len(), 1);
        assert_eq!(child.lock().unwrap()[0].alive(), Liveness::Dead);
        assert!(signals.lock().unwrap().is_empty());
        let pid: u32 = std::fs::read_to_string(holder)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        procs::signal(pid, libc::SIGKILL).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn waiter_panic_in_launch_restores_dispositions_without_signalling() {
        let _lock = TEST_IO_TEST_LOCK
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let child = Mutex::new(Vec::new());
        let signals = Mutex::new(Vec::new());
        let on_spawned = |proc: ProcId, _: u32| child.lock().unwrap().push(proc);
        let on_signal = |signal: crate::audit::SignalPayload| signals.lock().unwrap().push(signal);
        let mut on_line = |_line: &str| {};
        set_test_io_fault(TestIoStream::Waiter, TestIoFault::Panic);
        let error = run(
            &invocation("exit 0", None),
            &SuperviseOpts {
                detach_group: false,
                grace: Duration::from_millis(200),
                capture: false,
                on_spawned: &on_spawned,
                on_signal: Some(&on_signal),
            },
            &mut on_line,
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("signal waiter panicked"));
        assert_eq!(child.lock().unwrap().len(), 1);
        assert_eq!(child.lock().unwrap()[0].alive(), Liveness::Dead);
        assert!(signals.lock().unwrap().is_empty());
        let ignored = std::fs::read_to_string("/proc/self/status")
            .unwrap()
            .lines()
            .find_map(|line| line.strip_prefix("SigIgn:").map(str::trim))
            .and_then(|value| u64::from_str_radix(value, 16).ok())
            .unwrap();
        assert_eq!(ignored & 0x6, 0);
    }
}
