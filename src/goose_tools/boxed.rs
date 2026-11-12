//! The `box` role: builds the bubblewrap sandbox a Goose run's file and shell
//! tools live in, and execs into it. The preflight builds the same sandbox
//! with `/bin/true` as its command before Goose is spawned.
//!
//! Every bind source a model could have touched (the policy's roots and the
//! `[goose] box_ro` entries) is opened one component at a time without
//! following symlinks and handed to bwrap as a descriptor, so what is mounted
//! is the directory that sat at exactly the recorded path.

use std::ffi::{OsStr, OsString};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use anyhow::{bail, Context, Result};

use crate::config::{expand_tilde_in, Config, EnvSource};
use crate::goose_policy::GoosePolicy;

/// Directories mirrored from the host as it has them: a symlink stays a
/// symlink, a directory is bound read-only, an absent one is left out.
const MIRRORED: [&str; 4] = ["bin", "sbin", "lib", "lib64"];
/// System directories bound read-only whole.
const SYSTEM: [&str; 2] = ["/usr", "/etc"];
/// `HOME` inside the box, on its private `/tmp`.
const BOX_HOME: &str = "/tmp/home";

/// The site settings a box is built from.
#[derive(Debug, Clone)]
pub struct BoxSettings {
    /// The bubblewrap binary, resolved.
    pub bwrap: PathBuf,
    /// `[goose] box_ro`, `~` expanded and canonical.
    pub box_ro: Vec<PathBuf>,
    /// `PATH` inside the box.
    pub box_path: String,
    /// `[goose] tool_wait_s`, passed to `box-inner`.
    pub tool_wait_s: u64,
}

impl BoxSettings {
    /// Reads `[goose]`: resolves `bwrap` (a bare name on `env`'s `PATH`) and
    /// canonicalises `box_ro`. A missing binary or entry is an error naming it.
    pub fn from_config(cfg: &Config, env: &dyn EnvSource) -> Result<BoxSettings> {
        Ok(BoxSettings {
            bwrap: resolve_bwrap(&cfg.goose.bwrap, env)?,
            box_ro: cfg.goose.box_ro_in(env)?,
            box_path: cfg.goose.box_path.clone(),
            tool_wait_s: cfg.goose.tool_wait_s,
        })
    }
}

/// `[goose] bwrap` as a path: one holding a `/` is used as written (after `~`
/// expansion); a bare name is looked up on `env`'s `PATH`.
fn resolve_bwrap(name: &str, env: &dyn EnvSource) -> Result<PathBuf> {
    if name.contains('/') {
        return Ok(expand_tilde_in(env, name));
    }
    let path = env.get("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .filter(|dir| dir.is_absolute())
        .map(|dir| dir.join(name))
        .find(|candidate| is_executable(candidate))
        .with_context(|| {
            format!("the Goose box needs bubblewrap: [goose] bwrap {name:?} is not on PATH")
        })
}

fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

/// The command `box-inner` runs as inside the box. Its policy has no exec
/// roots or agents: exec roots are not mounted, and every root `box-inner` is
/// given must exist inside the box.
pub fn inner_command(policy: &GoosePolicy, self_exe: &Path, tool_wait_s: u64) -> Vec<OsString> {
    let inner = GoosePolicy {
        agents: Vec::new(),
        exec_roots: Vec::new(),
        ..policy.clone()
    };
    let mut cmd: Vec<OsString> = vec![self_exe.into()];
    cmd.extend(
        [
            "goose-tools".to_string(),
            "--role".into(),
            "box-inner".into(),
            "--policy".into(),
            inner.to_hex(),
            "--tool-wait-s".into(),
            tool_wait_s.to_string(),
        ]
        .map(OsString::from),
    );
    cmd
}

/// A built bwrap invocation and the bind-source descriptors it needs open.
pub struct BoxCommand {
    /// The bubblewrap binary.
    pub program: PathBuf,
    /// bwrap's arguments; each `*-fd` flag names a descriptor this value owns.
    pub args: Vec<OsString>,
    fds: Vec<std::os::fd::OwnedFd>,
}

impl BoxCommand {
    /// A `Command` for the invocation. The bind-source descriptors lose
    /// close-on-exec in the child only, so no other spawn inherits them.
    pub fn command(&self) -> Command {
        use std::os::fd::AsRawFd;
        use std::os::unix::process::CommandExt;
        let mut command = Command::new(&self.program);
        command.args(&self.args);
        let fds: Vec<i32> = self.fds.iter().map(AsRawFd::as_raw_fd).collect();
        // SAFETY: the closure calls only fcntl, which is async-signal-safe, on
        // descriptors this BoxCommand owns and keeps open across the spawn.
        unsafe {
            command.pre_exec(move || {
                for &fd in &fds {
                    if libc::fcntl(fd, libc::F_SETFD, 0) == -1 {
                        return Err(std::io::Error::last_os_error());
                    }
                }
                Ok(())
            });
        }
        command
    }
}

/// Builds bwrap's argv for `policy`, with `command` run inside. `host` is the
/// directory the `/bin`-style entries are inspected in (`/` outside tests).
/// The working directory is the policy's first read root, which is always the
/// run's cwd.
pub fn build(
    policy: &GoosePolicy,
    settings: &BoxSettings,
    self_exe: &Path,
    command: &[OsString],
    host: &Path,
) -> Result<BoxCommand> {
    let Some(cwd) = policy.read_roots.first() else {
        bail!("the Goose policy has no read root to run the box in");
    };
    let mut args: Vec<OsString> = [
        "--unshare-all",
        "--unshare-user",
        "--disable-userns",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
    ]
    .map(OsString::from)
    .to_vec();
    let mut fds = Vec::new();
    let mut covered: Vec<PathBuf> = Vec::new();

    for dir in SYSTEM {
        push_all(&mut args, ["--ro-bind", dir, dir]);
        covered.push(dir.into());
    }
    for name in MIRRORED {
        let source = host.join(name);
        let dest = format!("/{name}");
        match std::fs::symlink_metadata(&source) {
            Ok(m) if m.file_type().is_symlink() => {
                let target = std::fs::read_link(&source)
                    .with_context(|| format!("reading the link {}", source.display()))?;
                args.push("--symlink".into());
                args.push(target.into());
                args.push(dest.into());
            }
            Ok(m) if m.is_dir() => {
                args.push("--ro-bind".into());
                args.push(source.into());
                args.push(dest.clone().into());
                covered.push(dest.into());
            }
            _ => {}
        }
    }

    push_all(
        &mut args,
        [
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", BOX_HOME,
        ],
    );

    for dir in &settings.box_ro {
        let fd = open_nofollow(dir)?;
        push_fd_bind(&mut args, "--ro-bind-fd", &fd, dir);
        fds.push(fd);
        covered.push(dir.clone());
    }

    // The ccc binary is bound by path, not descriptor: it lies in a protected
    // directory no write root overlaps, and current_exe() is canonical.
    if !covered.iter().any(|dir| self_exe.starts_with(dir)) {
        args.push("--ro-bind".into());
        args.push(self_exe.into());
        args.push(self_exe.into());
    }

    for (root, writable) in mount_order(policy) {
        let fd = open_nofollow(root)?;
        let flag = if writable {
            "--bind-fd"
        } else {
            "--ro-bind-fd"
        };
        push_fd_bind(&mut args, flag, &fd, root);
        fds.push(fd);
    }

    // Last mount step: outside its mounts and `/tmp`, nothing in the box is
    // writable. Submounts keep their own flags.
    push_all(&mut args, ["--remount-ro", "/"]);
    args.push("--chdir".into());
    args.push(cwd.into());
    for (key, value) in [
        ("PATH", settings.box_path.as_str()),
        ("HOME", BOX_HOME),
        ("TMPDIR", "/tmp"),
        ("LANG", "C.UTF-8"),
        ("TERM", "dumb"),
    ] {
        push_all(&mut args, ["--setenv", key, value]);
    }
    let filter = seccomp_fd()?;
    args.push("--add-seccomp-fd".into());
    args.push(std::os::fd::AsRawFd::as_raw_fd(&filter).to_string().into());
    fds.push(filter);
    args.push("--".into());
    args.extend(command.iter().cloned());
    Ok(BoxCommand {
        program: settings.bwrap.clone(),
        args,
        fds,
    })
}

fn push_all<const N: usize>(args: &mut Vec<OsString>, items: [&str; N]) {
    args.extend(items.map(OsString::from));
}

fn push_fd_bind(args: &mut Vec<OsString>, flag: &str, fd: &std::os::fd::OwnedFd, dest: &Path) {
    use std::os::fd::AsRawFd;
    args.push(flag.into());
    args.push(fd.as_raw_fd().to_string().into());
    args.push(dest.into());
}

/// The roots to mount, each with whether it is writable: shallowest first,
/// read before write at equal depth, so a write root nested in a read root
/// stays writable. A read root inside a write root is left out; it is already
/// visible.
fn mount_order(policy: &GoosePolicy) -> Vec<(&Path, bool)> {
    let writes = &policy.write_roots;
    let mut roots: Vec<(&Path, bool)> = policy
        .read_roots
        .iter()
        .filter(|r| !writes.iter().any(|w| r.starts_with(w)))
        .map(|r| (r.as_path(), false))
        .chain(writes.iter().map(|w| (w.as_path(), true)))
        .collect();
    roots.sort_by_key(|(path, writable)| (path.components().count(), *writable));
    roots
}

/// `AUDIT_ARCH_X86_64` from `linux/audit.h`; libc does not define it.
#[cfg(all(target_arch = "x86_64", target_endian = "little"))]
const NATIVE_AUDIT_ARCH: Option<u32> = Some(0xC000_003E);
/// `AUDIT_ARCH_AARCH64` from `linux/audit.h`; libc does not define it.
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
const NATIVE_AUDIT_ARCH: Option<u32> = Some(0xC000_00B7);
#[cfg(not(any(
    all(target_arch = "x86_64", target_endian = "little"),
    all(target_arch = "aarch64", target_endian = "little")
)))]
const NATIVE_AUDIT_ARCH: Option<u32> = None;

/// The x32 ABI bit in an x86_64 syscall number.
const X32_SYSCALL_BIT: u32 = 0x4000_0000;
/// The kernel's `SOCK_TYPE_MASK`: the bits of a socket type left once
/// `SOCK_NONBLOCK` and `SOCK_CLOEXEC` are masked off.
const SOCK_TYPE_MASK: u32 = 0xf;

// Offsets into `struct seccomp_data`; on a little-endian target an
// argument's offset addresses its low 32 bits.
const SECCOMP_NR: u32 = 0;
const SECCOMP_ARCH: u32 = 4;
const SECCOMP_ARG0: u32 = 16;
const SECCOMP_ARG1: u32 = 24;

/// A jump target in the filter program.
#[derive(Clone, Copy, PartialEq)]
enum To {
    /// The next instruction.
    Next,
    /// The instruction after `Label(name)`.
    Label(&'static str),
    Allow,
    NoFamily,
    NoSys,
    NoPerm,
}

enum Op {
    Load(u32),
    Jeq(u32, To, To),
    Jge(u32, To, To),
    And(u32),
    Label(&'static str),
}

/// The classic-BPF program the box loads, as `sock_filter` entries in native
/// byte order:
///
/// - a syscall of any ABI but the native one, and on x86_64 an x32 syscall,
///   fails with `EPERM`;
/// - `socket` is allowed only for `AF_INET`, `AF_INET6` and `AF_NETLINK`;
///   any other family fails with `EAFNOSUPPORT`;
/// - `socketpair` is allowed only for `AF_UNIX` stream and seqpacket pairs;
///   any other domain or type fails with `EAFNOSUPPORT`;
/// - the three io_uring syscalls fail with `ENOSYS`, since io_uring creates
///   and connects sockets without passing through seccomp;
/// - every other syscall is allowed.
///
/// On a target other than little-endian x86_64 or aarch64 there is no
/// program, and the box is refused.
fn seccomp_program() -> Result<Vec<u8>> {
    let Some(arch) = NATIVE_AUDIT_ARCH else {
        bail!(
            "the Goose box has no syscall filter for this architecture ({}, {}-endian); \
             it runs only on little-endian x86_64 or aarch64",
            std::env::consts::ARCH,
            if cfg!(target_endian = "little") {
                "little"
            } else {
                "big"
            }
        );
    };
    let nr = |n: libc::c_long| n as u32;
    let family = |f: libc::c_int| f as u32;
    let mut ops = vec![
        Op::Load(SECCOMP_ARCH),
        Op::Jeq(arch, To::Next, To::NoPerm),
        Op::Load(SECCOMP_NR),
    ];
    if cfg!(target_arch = "x86_64") {
        ops.push(Op::Jge(X32_SYSCALL_BIT, To::NoPerm, To::Next));
    }
    ops.extend([
        Op::Jeq(nr(libc::SYS_socket), To::Next, To::Label("socket")),
        Op::Load(SECCOMP_ARG0),
        Op::Jeq(family(libc::AF_INET), To::Allow, To::Next),
        Op::Jeq(family(libc::AF_INET6), To::Allow, To::Next),
        Op::Jeq(family(libc::AF_NETLINK), To::Allow, To::NoFamily),
        Op::Label("socket"),
        Op::Jeq(nr(libc::SYS_socketpair), To::Next, To::Label("socketpair")),
        Op::Load(SECCOMP_ARG0),
        Op::Jeq(family(libc::AF_UNIX), To::Next, To::NoFamily),
        Op::Load(SECCOMP_ARG1),
        Op::And(SOCK_TYPE_MASK),
        Op::Jeq(family(libc::SOCK_STREAM), To::Allow, To::Next),
        Op::Jeq(family(libc::SOCK_SEQPACKET), To::Allow, To::NoFamily),
        Op::Label("socketpair"),
        Op::Jeq(nr(libc::SYS_io_uring_setup), To::NoSys, To::Next),
        Op::Jeq(nr(libc::SYS_io_uring_enter), To::NoSys, To::Next),
        Op::Jeq(nr(libc::SYS_io_uring_register), To::NoSys, To::Allow),
    ]);
    Ok(assemble(&ops))
}

/// Lays `ops` out with the four return instructions last, resolving every
/// jump to a forward offset.
fn assemble(ops: &[Op]) -> Vec<u8> {
    let mut labels = Vec::new();
    let mut count = 0usize;
    for op in ops {
        match op {
            Op::Label(name) => labels.push((*name, count)),
            _ => count += 1,
        }
    }
    let returns = [
        (To::Allow, libc::SECCOMP_RET_ALLOW),
        (
            To::NoFamily,
            libc::SECCOMP_RET_ERRNO | libc::EAFNOSUPPORT as u32,
        ),
        (To::NoSys, libc::SECCOMP_RET_ERRNO | libc::ENOSYS as u32),
        (To::NoPerm, libc::SECCOMP_RET_ERRNO | libc::EPERM as u32),
    ];
    let target = |to: To| -> usize {
        match to {
            To::Label(name) => labels.iter().find(|(n, _)| *n == name).expect("label").1,
            To::Next => unreachable!("resolved by the caller"),
            ret => count + returns.iter().position(|(r, _)| *r == ret).expect("return"),
        }
    };
    let mut out = Vec::new();
    let mut emit = |code: u32, jt: u8, jf: u8, k: u32| {
        out.extend_from_slice(&(code as u16).to_ne_bytes());
        out.push(jt);
        out.push(jf);
        out.extend_from_slice(&k.to_ne_bytes());
    };
    let mut at = 0usize;
    for op in ops {
        let offset = |to: To| -> u8 {
            match to {
                To::Next => 0,
                to => u8::try_from(target(to) - (at + 1)).expect("a jump fits in 8 bits"),
            }
        };
        match *op {
            Op::Label(_) => continue,
            Op::Load(off) => emit(libc::BPF_LD | libc::BPF_W | libc::BPF_ABS, 0, 0, off),
            Op::Jeq(k, t, f) => emit(
                libc::BPF_JMP | libc::BPF_JEQ | libc::BPF_K,
                offset(t),
                offset(f),
                k,
            ),
            Op::Jge(k, t, f) => emit(
                libc::BPF_JMP | libc::BPF_JGE | libc::BPF_K,
                offset(t),
                offset(f),
                k,
            ),
            Op::And(k) => emit(libc::BPF_ALU | libc::BPF_AND | libc::BPF_K, 0, 0, k),
        }
        at += 1;
    }
    for (_, ret) in returns {
        emit(libc::BPF_RET | libc::BPF_K, 0, 0, ret);
    }
    out
}

/// The filter program in a `memfd`, at offset 0, for bwrap's
/// `--add-seccomp-fd`.
fn seccomp_fd() -> Result<std::os::fd::OwnedFd> {
    use std::io::{Seek, Write};
    use std::os::fd::FromRawFd;
    let program = seccomp_program()?;
    // SAFETY: a NUL-terminated literal name; the result is checked.
    let fd = unsafe { libc::memfd_create(c"ccc-box-seccomp".as_ptr(), libc::MFD_CLOEXEC) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error())
            .context("creating the Goose box's syscall filter");
    }
    // SAFETY: `fd` was just created and is owned by nothing else.
    let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
    file.write_all(&program)
        .and_then(|()| file.rewind())
        .context("writing the Goose box's syscall filter")?;
    Ok(file.into())
}

/// Opens the directory at exactly `path` as an `O_PATH` descriptor, walking
/// from `/` one component at a time and refusing any component that is a
/// symlink. `path` must be absolute with no `.` or `..` component.
pub fn open_nofollow(path: &Path) -> Result<std::os::fd::OwnedFd> {
    use std::os::fd::{FromRawFd, OwnedFd};
    use std::os::unix::ffi::OsStrExt;

    let refuse = |what: String| -> anyhow::Error {
        anyhow::anyhow!(
            "the Goose box cannot mount {}: {what}; it is never mounted through a symlink",
            path.display()
        )
    };
    let flags = libc::O_PATH | libc::O_DIRECTORY | libc::O_CLOEXEC;
    // SAFETY: a NUL-terminated literal path; the result is checked.
    let fd = unsafe { libc::open(c"/".as_ptr(), flags) };
    if fd < 0 {
        return Err(refuse(format!(
            "opening /: {}",
            std::io::Error::last_os_error()
        )));
    }
    // SAFETY: `fd` was just opened and is owned by nothing else.
    let mut current = unsafe { OwnedFd::from_raw_fd(fd) };
    let mut components = path.components();
    if components.next() != Some(Component::RootDir) {
        return Err(refuse("the path is not absolute".into()));
    }
    for component in components {
        let Component::Normal(name) = component else {
            return Err(refuse(format!(
                "it has a {:?} component",
                component.as_os_str()
            )));
        };
        let cname = std::ffi::CString::new(name.as_bytes())
            .map_err(|_| refuse("it contains a NUL byte".into()))?;
        use std::os::fd::AsRawFd;
        // SAFETY: `current` is an open directory descriptor and `cname` a
        // NUL-terminated name; the result is checked.
        let next = unsafe {
            libc::openat(
                current.as_raw_fd(),
                cname.as_ptr(),
                flags | libc::O_NOFOLLOW,
            )
        };
        if next < 0 {
            let err = std::io::Error::last_os_error();
            return Err(refuse(if is_symlink_at(&current, &cname) {
                format!("component {} is a symlink", OsStr::new(name).display())
            } else {
                format!("component {}: {err}", OsStr::new(name).display())
            }));
        }
        // SAFETY: `next` was just opened and is owned by nothing else.
        current = unsafe { OwnedFd::from_raw_fd(next) };
    }
    Ok(current)
}

fn is_symlink_at(dir: &std::os::fd::OwnedFd, name: &std::ffi::CStr) -> bool {
    use std::os::fd::AsRawFd;
    let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `dir` is open, `name` NUL-terminated, `st` sized for the call.
    let rc = unsafe {
        libc::fstatat(
            dir.as_raw_fd(),
            name.as_ptr(),
            st.as_mut_ptr(),
            libc::AT_SYMLINK_NOFOLLOW,
        )
    };
    // SAFETY: fstatat filled `st` when it returned 0.
    rc == 0 && unsafe { st.assume_init() }.st_mode & libc::S_IFMT == libc::S_IFLNK
}

/// The `box` role: builds the box for `policy` from the site config and
/// execs bwrap, so Goose's stdio reaches `box-inner` directly. Returns only
/// on failure.
pub fn exec_box(policy: &GoosePolicy, env: &dyn EnvSource) -> Result<i32> {
    use std::os::unix::process::CommandExt;
    let cfg = Config::load_in(env)?;
    let settings = BoxSettings::from_config(&cfg, env)?;
    let self_exe = std::env::current_exe().context("locating the ccc binary")?;
    // Nothing this process inherited reaches bwrap or the box; the bind
    // sources `build` opens next are passed explicitly.
    super::inner::sweep_fds_from(3, super::inner::Sweep::Cloexec);
    let inner = inner_command(policy, &self_exe, settings.tool_wait_s);
    let built = build(policy, &settings, &self_exe, &inner, Path::new("/"))?;
    let err = built.command().exec();
    Err(err).with_context(|| format!("running bubblewrap {}", built.program.display()))
}

/// Builds the box for `policy` exactly as the `box` role does and runs it with
/// `/bin/true`, bounded by `grace`. A failure names bwrap and quotes its
/// stderr, or names the path whose walk failed.
pub fn preflight(
    policy: &GoosePolicy,
    settings: &BoxSettings,
    self_exe: &Path,
    grace: std::time::Duration,
) -> Result<()> {
    use std::io::Read;
    let built = build(
        policy,
        settings,
        self_exe,
        &[OsString::from("/bin/true")],
        Path::new("/"),
    )?;
    let bwrap = built.program.display().to_string();
    let mut command = built.command();
    command
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped());
    let mut child = command
        .spawn()
        .with_context(|| format!("the Goose box cannot start: running bubblewrap {bwrap}"))?;
    let mut stderr = child.stderr.take().expect("stderr is piped");
    let reader = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = stderr.read_to_string(&mut text);
        text
    });
    let deadline = std::time::Instant::now() + grace;
    let status = loop {
        if let Some(status) = child.try_wait().context("waiting for bubblewrap")? {
            break Some(status);
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    let text = reader.join().unwrap_or_default();
    match status {
        Some(s) if s.success() => Ok(()),
        Some(s) => bail!(
            "the Goose box cannot start: bubblewrap {bwrap} failed ({s}): {}",
            text.trim()
        ),
        None => bail!(
            "the Goose box cannot start: bubblewrap {bwrap} did not finish within {:.1}s",
            grace.as_secs_f64()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goose_policy::GooseTool;
    use crate::testutil::TempDir;
    use std::collections::BTreeMap;
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::fs::symlink;
    use std::sync::mpsc;
    use std::time::Duration;

    use serde_json::{json, Value};

    fn mkdir(base: &Path, rel: &str) -> PathBuf {
        let p = base.join(rel);
        std::fs::create_dir_all(&p).unwrap();
        std::fs::canonicalize(p).unwrap()
    }

    fn policy(read: &[&Path], write: &[&Path]) -> GoosePolicy {
        let mut tools = std::collections::BTreeSet::from([GooseTool::Read, GooseTool::Shell]);
        if !write.is_empty() {
            tools.insert(GooseTool::Write);
        }
        GoosePolicy {
            tools,
            agents: vec![],
            read_roots: read.iter().map(|p| p.to_path_buf()).collect(),
            write_roots: write.iter().map(|p| p.to_path_buf()).collect(),
            exec_roots: vec![],
        }
    }

    fn settings(box_ro: Vec<PathBuf>) -> BoxSettings {
        BoxSettings {
            bwrap: PathBuf::from("/usr/bin/bwrap"),
            box_ro,
            box_path: "/usr/bin:/bin".into(),
            tool_wait_s: 5,
        }
    }

    /// The argv with every descriptor number after a `*-fd` flag replaced by
    /// `FD`, as strings.
    fn shown(built: &BoxCommand) -> Vec<String> {
        let mut out = Vec::new();
        let mut fd_next = false;
        for a in &built.args {
            let a = a.to_string_lossy().into_owned();
            out.push(if fd_next { "FD".to_string() } else { a.clone() });
            fd_next = a.ends_with("-fd");
        }
        out
    }

    #[test]
    fn the_argv_is_exact() {
        let t = TempDir::new("box-argv");
        let host = mkdir(t.path(), "host");
        mkdir(&host, "usr/bin");
        symlink("usr/bin", host.join("bin")).unwrap();
        mkdir(&host, "lib");
        let ro = mkdir(t.path(), "cargo");
        let cwd = mkdir(t.path(), "r");
        let w = mkdir(t.path(), "w");
        let p = policy(&[&cwd, &w], &[&w]);
        let exe = Path::new("/opt/ccc/bin/ccc");
        let built = build(
            &p,
            &settings(vec![ro.clone()]),
            exe,
            &["/bin/true".into()],
            &host,
        )
        .unwrap();
        assert_eq!(built.program, Path::new("/usr/bin/bwrap"));
        assert_eq!(built.fds.len(), 4);
        let s = |p: &Path| p.display().to_string();
        let mut want: Vec<String> = [
            "--unshare-all",
            "--unshare-user",
            "--disable-userns",
            "--die-with-parent",
            "--new-session",
            "--clearenv",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/etc",
            "/etc",
            "--symlink",
            "usr/bin",
            "/bin",
        ]
        .map(String::from)
        .to_vec();
        want.extend(["--ro-bind".into(), s(&host.join("lib")), "/lib".into()]);
        want.extend(
            [
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--dir",
                "/tmp/home",
            ]
            .map(String::from),
        );
        want.extend(["--ro-bind-fd".into(), "FD".into(), s(&ro)]);
        want.extend(["--ro-bind".into(), s(exe), s(exe)]);
        want.extend(["--ro-bind-fd".into(), "FD".into(), s(&cwd)]);
        want.extend(["--bind-fd".into(), "FD".into(), s(&w)]);
        want.extend(["--remount-ro".into(), "/".into()]);
        want.extend(["--chdir".into(), s(&cwd)]);
        for (k, v) in [
            ("PATH", "/usr/bin:/bin"),
            ("HOME", "/tmp/home"),
            ("TMPDIR", "/tmp"),
            ("LANG", "C.UTF-8"),
            ("TERM", "dumb"),
        ] {
            want.extend(["--setenv".into(), k.into(), v.into()]);
        }
        want.extend(["--add-seccomp-fd".into(), "FD".into()]);
        want.extend(["--".into(), "/bin/true".into()]);
        assert_eq!(shown(&built), want);

        use std::os::fd::AsRawFd;
        let filter = built.fds.last().unwrap().as_raw_fd();
        let named = &built.args[built.args.len() - 3];
        assert_eq!(named.to_str().unwrap(), filter.to_string());
        // SAFETY: `filter` is open for the life of `built`.
        assert_eq!(unsafe { libc::lseek(filter, 0, libc::SEEK_CUR) }, 0);
        assert_eq!(
            std::fs::read(format!("/proc/self/fd/{filter}")).unwrap(),
            seccomp_program().unwrap()
        );
    }

    /// The fields of `struct seccomp_data` the program reads.
    struct Data {
        nr: u32,
        arch: u32,
        args: [u64; 2],
    }

    impl Data {
        fn native(nr: libc::c_long, args: [u64; 2]) -> Data {
            Data {
                nr: nr as u32,
                arch: NATIVE_AUDIT_ARCH.unwrap(),
                args,
            }
        }

        /// The word at `offset` as the kernel lays the structure out on a
        /// little-endian target.
        fn word(&self, offset: u32) -> u32 {
            match offset {
                SECCOMP_NR => self.nr,
                SECCOMP_ARCH => self.arch,
                SECCOMP_ARG0 => self.args[0] as u32,
                SECCOMP_ARG1 => self.args[1] as u32,
                other => panic!("the program loads seccomp_data offset {other}"),
            }
        }
    }

    /// Runs the program's bytes over `data`, modelling only the opcodes the
    /// program is allowed to use; any other opcode fails the test.
    fn evaluate(program: &[u8], data: &Data) -> u32 {
        const LD_W_ABS: u32 = libc::BPF_LD | libc::BPF_W | libc::BPF_ABS;
        const JEQ_K: u32 = libc::BPF_JMP | libc::BPF_JEQ | libc::BPF_K;
        const JGE_K: u32 = libc::BPF_JMP | libc::BPF_JGE | libc::BPF_K;
        const AND_K: u32 = libc::BPF_ALU | libc::BPF_AND | libc::BPF_K;
        const RET_K: u32 = libc::BPF_RET | libc::BPF_K;
        assert_eq!(program.len() % 8, 0);
        let insns: Vec<(u32, usize, usize, u32)> = program
            .chunks(8)
            .map(|c| {
                (
                    u32::from(u16::from_ne_bytes([c[0], c[1]])),
                    usize::from(c[2]),
                    usize::from(c[3]),
                    u32::from_ne_bytes([c[4], c[5], c[6], c[7]]),
                )
            })
            .collect();
        let mut acc = 0u32;
        let mut pc = 0usize;
        loop {
            let (code, jt, jf, k) = insns[pc];
            pc += 1;
            match code {
                LD_W_ABS => acc = data.word(k),
                AND_K => acc &= k,
                JEQ_K => pc += if acc == k { jt } else { jf },
                JGE_K => pc += if acc >= k { jt } else { jf },
                RET_K => return k,
                other => panic!("opcode {other:#x} at {} is not modelled", pc - 1),
            }
        }
    }

    #[test]
    fn the_syscall_filter_limits_sockets_and_refuses_io_uring_and_foreign_abis() {
        let program = seccomp_program().unwrap();
        let allow = libc::SECCOMP_RET_ALLOW;
        let errno = |e: i32| libc::SECCOMP_RET_ERRNO | e as u32;
        let no_family = errno(libc::EAFNOSUPPORT);
        let a = |v: libc::c_int| v as u64;
        let socket = |family: u64| Data::native(libc::SYS_socket, [family, a(libc::SOCK_STREAM)]);
        let pair = |family: libc::c_int, ty: libc::c_int| {
            Data::native(libc::SYS_socketpair, [a(family), a(ty)])
        };
        let mut cases: Vec<(&str, Data, u32)> = vec![
            ("socket AF_UNIX", socket(a(libc::AF_UNIX)), no_family),
            (
                "socket AF_UNIX, high bits set",
                socket(0xdead_beef_0000_0000 | a(libc::AF_UNIX)),
                no_family,
            ),
            ("socket AF_VSOCK", socket(a(libc::AF_VSOCK)), no_family),
            ("socket AF_PACKET", socket(a(libc::AF_PACKET)), no_family),
            ("socket AF_ALG", socket(a(libc::AF_ALG)), no_family),
            ("socket AF_INET", socket(a(libc::AF_INET)), allow),
            ("socket AF_INET6", socket(a(libc::AF_INET6)), allow),
            ("socket AF_NETLINK", socket(a(libc::AF_NETLINK)), allow),
            ("pair stream", pair(libc::AF_UNIX, libc::SOCK_STREAM), allow),
            (
                "pair seqpacket",
                pair(libc::AF_UNIX, libc::SOCK_SEQPACKET),
                allow,
            ),
            (
                "pair stream nonblock cloexec",
                pair(
                    libc::AF_UNIX,
                    libc::SOCK_STREAM | libc::SOCK_NONBLOCK | libc::SOCK_CLOEXEC,
                ),
                allow,
            ),
            (
                "pair seqpacket cloexec",
                pair(libc::AF_UNIX, libc::SOCK_SEQPACKET | libc::SOCK_CLOEXEC),
                allow,
            ),
            (
                "pair dgram",
                pair(libc::AF_UNIX, libc::SOCK_DGRAM),
                no_family,
            ),
            (
                "pair dgram cloexec",
                pair(libc::AF_UNIX, libc::SOCK_DGRAM | libc::SOCK_CLOEXEC),
                no_family,
            ),
            ("pair raw", pair(libc::AF_UNIX, libc::SOCK_RAW), no_family),
            (
                "pair AF_INET stream",
                pair(libc::AF_INET, libc::SOCK_STREAM),
                no_family,
            ),
            ("read", Data::native(libc::SYS_read, [0, 0]), allow),
            ("openat", Data::native(libc::SYS_openat, [0, 0]), allow),
            (
                "io_uring_setup",
                Data::native(libc::SYS_io_uring_setup, [0, 0]),
                errno(libc::ENOSYS),
            ),
            (
                "io_uring_enter",
                Data::native(libc::SYS_io_uring_enter, [0, 0]),
                errno(libc::ENOSYS),
            ),
            (
                "io_uring_register",
                Data::native(libc::SYS_io_uring_register, [0, 0]),
                errno(libc::ENOSYS),
            ),
        ];
        // `AUDIT_ARCH_I386`, the other of the pair, and one that is neither.
        for foreign in [
            0x4000_0003u32,
            0xC000_003E ^ 0xC000_00B7 ^ NATIVE_AUDIT_ARCH.unwrap(),
            0,
        ] {
            for nr in [libc::SYS_read, libc::SYS_socket, 102] {
                cases.push((
                    "foreign arch",
                    Data {
                        nr: nr as u32,
                        arch: foreign,
                        args: [a(libc::AF_INET), 0],
                    },
                    errno(libc::EPERM),
                ));
            }
        }
        if cfg!(target_arch = "x86_64") {
            cases.push((
                "x32 socket",
                Data {
                    nr: libc::SYS_socket as u32 | X32_SYSCALL_BIT,
                    arch: NATIVE_AUDIT_ARCH.unwrap(),
                    args: [a(libc::AF_INET), 0],
                },
                errno(libc::EPERM),
            ));
        }
        for (name, data, want) in &cases {
            assert_eq!(
                evaluate(&program, data),
                *want,
                "{name}: nr {} arch {:#x} args {:x?}",
                data.nr,
                data.arch,
                data.args
            );
        }
    }

    #[test]
    fn roots_mount_shallowest_first_and_read_inside_write_is_skipped() {
        let t = TempDir::new("box-order");
        let r = mkdir(t.path(), "r");
        let w_in_r = mkdir(t.path(), "r/w");
        let r_in_w = mkdir(t.path(), "r/w/sub");
        let peer_read = mkdir(t.path(), "a");
        let peer_write = mkdir(t.path(), "b");
        let p = policy(&[&r, &r_in_w, &peer_read], &[&w_in_r, &peer_write]);
        let order: Vec<(PathBuf, bool)> = mount_order(&p)
            .into_iter()
            .map(|(p, w)| (p.to_path_buf(), w))
            .collect();
        assert_eq!(
            order,
            vec![
                (r.clone(), false),
                (peer_read, false),
                (peer_write, true),
                (w_in_r, true),
            ]
        );
    }

    #[test]
    fn self_exe_is_not_bound_twice_when_a_mount_covers_it() {
        let t = TempDir::new("box-exe");
        let cwd = mkdir(t.path(), "r");
        let tools = mkdir(t.path(), "tools");
        let p = policy(&[&cwd], &[]);
        for exe in [Path::new("/usr/bin/ccc"), &tools.join("ccc")] {
            let built =
                build(&p, &settings(vec![tools.clone()]), exe, &[], Path::new("/")).unwrap();
            let args = shown(&built);
            assert!(
                !args
                    .windows(3)
                    .any(|w| w[0] == "--ro-bind" && w[1] == exe.to_str().unwrap()),
                "{args:?}"
            );
        }
    }

    #[test]
    fn a_symlink_on_a_root_or_box_ro_path_is_refused_naming_it() {
        let t = TempDir::new("box-link");
        let target = mkdir(t.path(), "target");
        let real = mkdir(t.path(), "real");
        symlink(&target, real.join("link")).unwrap();
        mkdir(&target, "inner");
        let through = real.join("link/inner");
        let p = policy(&[&real, &through], &[]);
        let err = format!(
            "{:#}",
            build(
                &p,
                &settings(vec![]),
                Path::new("/usr/bin/ccc"),
                &[],
                Path::new("/")
            )
            .err()
            .unwrap()
        );
        assert!(err.contains(&through.display().to_string()), "{err}");
        assert!(err.contains("component link is a symlink"), "{err}");

        let p = policy(&[&real], &[]);
        let err = build(
            &p,
            &settings(vec![real.join("link")]),
            Path::new("/usr/bin/ccc"),
            &[],
            Path::new("/"),
        )
        .err()
        .unwrap();
        assert!(format!("{err:#}").contains("is a symlink"), "{err:#}");

        let dotted = real.join("../real");
        assert!(format!("{:#}", open_nofollow(&dotted).unwrap_err()).contains("\"..\""));
        assert!(open_nofollow(Path::new("relative")).is_err());
        assert!(open_nofollow(&real).is_ok());
    }

    #[test]
    fn bwrap_resolves_on_path_or_as_written() {
        let t = TempDir::new("box-bwrap");
        let bin = mkdir(t.path(), "bin");
        let fake = t.write_file("bin/bwrap", "#!/bin/sh\n", 0o755);
        let env = BTreeMap::from([("PATH".to_string(), format!("rel:{}", bin.display()))]);
        assert_eq!(resolve_bwrap("bwrap", &env).unwrap(), fake);
        assert_eq!(
            resolve_bwrap("/x/bwrap", &env).unwrap(),
            Path::new("/x/bwrap")
        );
        let err = resolve_bwrap("nope", &env).unwrap_err();
        assert!(format!("{err:#}").contains("bubblewrap"), "{err:#}");
    }

    #[test]
    fn inner_command_drops_exec_roots_and_agents() {
        let t = TempDir::new("box-inner-cmd");
        let r = mkdir(t.path(), "r");
        let e = mkdir(t.path(), "e");
        let mut p = policy(&[&r], &[]);
        p.tools.insert(GooseTool::Ccc);
        p.agents = vec!["explorer".into()];
        p.exec_roots = vec![e];
        let cmd = inner_command(&p, Path::new("/usr/bin/ccc"), 7);
        let words: Vec<String> = cmd
            .iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            words[..5],
            [
                "/usr/bin/ccc",
                "goose-tools",
                "--role",
                "box-inner",
                "--policy"
            ]
        );
        assert_eq!(words[6..], ["--tool-wait-s", "7"]);
        let inner = GoosePolicy::from_hex(&words[5]).unwrap();
        assert!(inner.exec_roots.is_empty() && inner.agents.is_empty());
        assert_eq!(inner.read_roots, p.read_roots);
        assert_eq!(inner.tools, p.tools);
    }

    #[test]
    fn preflight_passes_on_a_box_host() {
        if crate::testutil::skip_without_box_host() {
            return;
        }
        let t = TempDir::new("box-preflight-pass");
        let r = mkdir(t.path(), "r");
        let p = policy(&[&r], &[]);
        let exe = std::env::current_exe().unwrap();
        preflight(&p, &settings(vec![]), &exe, Duration::from_secs(5))
            .expect("bwrap must work on this host");
    }

    #[test]
    fn preflight_fails_naming_bwrap() {
        let t = TempDir::new("box-preflight");
        let r = mkdir(t.path(), "r");
        let p = policy(&[&r], &[]);
        let exe = std::env::current_exe().unwrap();
        let grace = Duration::from_secs(5);

        let mut missing = settings(vec![]);
        missing.bwrap = t.path().join("no-bwrap");
        let err = format!("{:#}", preflight(&p, &missing, &exe, grace).unwrap_err());
        assert!(
            err.contains("bubblewrap") && err.contains("no-bwrap"),
            "{err}"
        );

        let failing = t.write_file("bwrap", "#!/bin/sh\necho 'no userns' >&2\nexit 1\n", 0o755);
        let mut bad = settings(vec![]);
        bad.bwrap = failing;
        let err = format!("{:#}", preflight(&p, &bad, &exe, grace).unwrap_err());
        assert!(err.contains("no userns"), "{err}");

        // fixture-scan: allow bounded by the 1 s preflight grace, then SIGKILL
        let slow = t.write_file("slow-bwrap", "#!/bin/sh\nexec sleep 5\n", 0o755);
        let mut slow_settings = settings(vec![]);
        slow_settings.bwrap = slow;
        let err = format!(
            "{:#}",
            preflight(&p, &slow_settings, &exe, Duration::from_secs(1)).unwrap_err()
        );
        assert!(err.contains("did not finish"), "{err}");

        // A bwrap that would succeed, so only the root walk can refuse.
        let ok = t.write_file("ok-bwrap", "#!/bin/sh\nexit 0\n", 0o755);
        let mut ok_settings = settings(vec![]);
        ok_settings.bwrap = ok;
        std::fs::remove_dir(&r).unwrap();
        let err = format!(
            "{:#}",
            preflight(&p, &ok_settings, &exe, grace).unwrap_err()
        );
        assert!(
            err.contains("the Goose box cannot mount") && err.contains(&r.display().to_string()),
            "{err}"
        );
    }

    /// A running `ccc goose-tools --role box`. It reads its stdin, whose only
    /// writer is this test, so dropping it ends the box.
    struct BoxServer {
        child: std::process::Child,
        stdin: Option<std::process::ChildStdin>,
        lines: mpsc::Receiver<String>,
        next_id: u64,
    }

    struct BoxEnv {
        t: TempDir,
        home: PathBuf,
        read: PathBuf,
        write: PathBuf,
    }

    impl BoxEnv {
        fn new(label: &str) -> BoxEnv {
            let t = TempDir::new(label);
            let home = mkdir(t.path(), "home");
            std::fs::write(home.join("secret"), "s3cret").unwrap();
            let read = mkdir(t.path(), "r");
            std::fs::write(read.join("hello.txt"), "hello\n").unwrap();
            let write = mkdir(t.path(), "w");
            std::fs::write(
                t.path().join("config.toml"),
                "[goose]\nbwrap = \"bwrap\"\ntool_wait_s = 5\n",
            )
            .unwrap();
            BoxEnv {
                t,
                home,
                read,
                write,
            }
        }

        fn policy(&self) -> GoosePolicy {
            policy(&[&self.read, &self.write], &[&self.write])
        }

        fn start(&self) -> BoxServer {
            BoxServer::serve(self.command())
        }

        /// The box role as these tests start it: the real home and an explicit
        /// `CCC_CONFIG`.
        fn command(&self) -> Command {
            let mut command = Command::new(crate::testutil::ccc_binary());
            command
                .args([
                    "goose-tools",
                    "--role",
                    "box",
                    "--policy",
                    &self.policy().to_hex(),
                ])
                .env_clear()
                .env("PATH", "/usr/bin:/bin")
                .env("HOME", &self.home)
                .env("XDG_CONFIG_HOME", self.t.path().join("xdg"))
                .env("CCC_CONFIG", self.t.path().join("config.toml"))
                .env("CCC_BOX_TEST_MARKER", "visible-on-host")
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped());
            command
        }
    }

    impl BoxServer {
        fn serve(mut command: Command) -> BoxServer {
            let mut child = command.spawn().unwrap();
            let stdin = child.stdin.take();
            let stdout = child.stdout.take().unwrap();
            let (tx, lines) = mpsc::channel();
            std::thread::spawn(move || {
                for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
            });
            BoxServer {
                child,
                stdin,
                lines,
                next_id: 1,
            }
        }

        fn request(&mut self, method: &str, params: Value) -> Value {
            let id = self.next_id;
            self.next_id += 1;
            let line = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params});
            let stdin = self.stdin.as_mut().unwrap();
            writeln!(stdin, "{line}").unwrap();
            stdin.flush().unwrap();
            let reply = self
                .lines
                .recv_timeout(Duration::from_secs(9))
                .expect("a reply from the box");
            let reply: Value = serde_json::from_str(&reply).unwrap();
            assert_eq!(reply["id"], json!(id), "{reply}");
            reply
        }

        /// Calls `tool` and returns its text and whether it is an error.
        fn call(&mut self, tool: &str, args: Value) -> (String, bool) {
            let reply = self.request("tools/call", json!({"name": tool, "arguments": args}));
            let result = &reply["result"];
            (
                result["content"][0]["text"].as_str().unwrap().to_string(),
                result["isError"].as_bool().unwrap(),
            )
        }

        fn shell(&mut self, command: &str) -> String {
            self.call("shell", json!({"command": command})).0
        }

        fn finish(mut self) -> std::process::ExitStatus {
            drop(self.stdin.take());
            self.child.wait().unwrap()
        }
    }

    #[test]
    fn the_box_confines_files_network_processes_and_environment() {
        if crate::testutil::skip_without_box_host() {
            return;
        }
        let b = BoxEnv::new("box-behaviour");
        std::fs::write(b.t.path().join("outside"), "outside").unwrap();
        symlink(b.t.path().join("outside"), b.write.join("escape")).unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let abstract_name = format!("ccc-box-test-{}", std::process::id());
        let addr = <std::os::unix::net::SocketAddr as std::os::linux::net::SocketAddrExt>::from_abstract_name(
            abstract_name.as_bytes(),
        )
        .unwrap();
        let _abstract = std::os::unix::net::UnixListener::bind_addr(&addr).unwrap();
        let host_sockets: Vec<(PathBuf, std::os::unix::net::UnixListener)> = [&b.read, &b.write]
            .into_iter()
            .map(|root| {
                let path = root.join("host.sock");
                let listener = std::os::unix::net::UnixListener::bind(&path).unwrap();
                listener.set_nonblocking(true).unwrap();
                (path, listener)
            })
            .collect();

        let mut s = b.start();
        let init = s.request("initialize", json!({"protocolVersion": "2025-03-26"}));
        assert_eq!(
            init["result"]["capabilities"],
            json!({"tools": {}}),
            "{init}"
        );
        let listed = s.request("tools/list", json!({}));
        let names: Vec<&str> = listed["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(
            names.contains(&"shell") && names.contains(&"write_file"),
            "{names:?}"
        );

        let (text, err) = s.call("read_file", json!({"path": b.read.join("hello.txt")}));
        assert!(!err && text.contains("hello"), "{text}");
        let (text, err) = s.call(
            "write_file",
            json!({"path": b.read.join("new.txt"), "content": "x"}),
        );
        assert!(err && text.contains("Read-only file system"), "{text}");
        let (text, err) = s.call(
            "write_file",
            json!({"path": b.write.join("made.txt"), "content": "from the box"}),
        );
        assert!(!err, "{text}");
        for outside in ["/var/tmp/x", "/x"] {
            let (text, err) = s.call("write_file", json!({"path": outside, "content": "x"}));
            assert!(
                err && text.contains("Read-only file system"),
                "{outside}: {text}"
            );
        }
        let (text, err) = s.call("write_file", json!({"path": "/tmp/x", "content": "x"}));
        assert!(!err, "{text}");
        let (text, err) = s.call("read_file", json!({"path": b.write.join("escape")}));
        assert!(err && text.contains("No such file or directory"), "{text}");

        let out = s.shell(&format!("ls {}; echo rc=$?", b.home.display()));
        assert!(out.contains("rc=2") && !out.contains("secret"), "{out}");
        let out = s.shell("cut -d: -f1 /proc/net/dev | tail -n +3 | tr -d ' '");
        assert_eq!(out.lines().next(), Some("lo"), "{out}");
        assert!(
            !out.contains("eth") && out.lines().filter(|l| !l.starts_with('[')).count() == 1,
            "{out}"
        );
        let out = s.shell(&format!(
            "bash -c 'exec 3<>/dev/tcp/127.0.0.1/{port}' 2>&1; echo rc=$?"
        ));
        assert!(out.contains("rc=1"), "{out}");
        // Abstract sockets live in the network namespace: the test's is not listed.
        let out = s.shell(&format!("grep -c {abstract_name} /proc/net/unix; true"));
        assert!(out.starts_with('0'), "{out}");
        // A host listener on a path under a read or write root: the filter
        // refuses the Unix-domain socket, so nothing reaches it.
        for (path, _) in &host_sockets {
            let out = s.shell(&format!(
                "logger --socket-errors=on -T -u {} x 2>&1; echo rc=$?",
                path.display()
            ));
            assert!(
                out.contains("Address family not supported") && out.contains("rc=1"),
                "{out}"
            );
        }
        let out = s.shell("env");
        assert!(
            !out.contains("CCC_BOX_TEST_MARKER") && out.contains("HOME=/tmp/home"),
            "{out}"
        );
        let out = s.shell(&format!("kill -0 {} 2>&1; echo rc=$?", std::process::id()));
        assert!(out.contains("rc=1"), "{out}");
        let out = s.shell("ls /proc/self/fd");
        let fds: Vec<&str> = out.lines().take_while(|l| !l.starts_with('[')).collect();
        assert_eq!(fds, ["0", "1", "2", "3"], "{out}");
        let out = s.shell("echo out; echo err >&2");
        assert!(out.starts_with("out\nerr\n"), "{out}");
        let (text, err) = s.call("shell", json!({"command": "sleep 3", "timeout_s": 1}));
        assert!(err && text.contains("timed out"), "{text}");
        let out = s.shell("sleep 3 & echo hi");
        assert!(
            out.starts_with("hi\n") && out.contains("[exit status 0]"),
            "{out}"
        );
        assert!(s.finish().success());

        assert_eq!(
            std::fs::read_to_string(b.write.join("made.txt")).unwrap(),
            "from the box"
        );
        assert!(!b.read.join("new.txt").exists());
        for (path, listener) in &host_sockets {
            let err = listener.accept().unwrap_err();
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::WouldBlock,
                "{}: {err}",
                path.display()
            );
        }
        drop(listener);
    }

    #[test]
    fn a_box_started_as_goose_starts_it_reads_the_host_config_and_box_ro() {
        if crate::testutil::skip_without_box_host() {
            return;
        }
        let b = BoxEnv::new("box-host-env");
        let real_home = mkdir(b.t.path(), "realhome");
        let tools = mkdir(&real_home, "tools");
        std::fs::write(tools.join("toolchain-marker"), "").unwrap();
        let conf = mkdir(&mkdir(&real_home, ".config"), "ccc");
        std::fs::write(
            conf.join("config.toml"),
            "[goose]\nbwrap = \"bwrap\"\ntool_wait_s = 5\nbox_ro = [\"~/tools\"]\n",
        )
        .unwrap();
        let goose_home = mkdir(b.t.path(), "goosehome");
        let carried =
            crate::goose_policy::hex_encode(json!({"HOME": real_home}).to_string().as_bytes());
        let mut command = b.command();
        command
            .env("HOME", &goose_home)
            .env_remove("CCC_CONFIG")
            .env_remove("XDG_CONFIG_HOME")
            .env(crate::harness::goose::HOST_ENV_VAR, carried);
        let mut s = BoxServer::serve(command);
        let out = s.shell(&format!("ls {}", tools.display()));
        assert!(out.starts_with("toolchain-marker\n"), "{out}");
        assert!(s.finish().success());
        assert_eq!(std::fs::read_dir(&goose_home).unwrap().count(), 0);
    }

    /// Whether a host process has exactly `argv` as its command line.
    fn host_process_running(argv: &[&str]) -> bool {
        let want: Vec<u8> = argv
            .iter()
            .flat_map(|a| [a.as_bytes(), b"\0"].concat())
            .collect();
        std::fs::read_dir("/proc")
            .unwrap()
            .filter_map(Result::ok)
            .any(|e| std::fs::read(e.path().join("cmdline")).is_ok_and(|c| c == want))
    }

    /// Polls `argv`'s presence on the host until it equals `running`, up to 2 s.
    fn await_host_process(argv: &[&str], running: bool) -> bool {
        for _ in 0..20 {
            if host_process_running(argv) == running {
                return true;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        false
    }

    #[test]
    fn a_background_job_ends_with_the_box() {
        if crate::testutil::skip_without_box_host() {
            return;
        }
        let b = BoxEnv::new("box-bg");
        // A fractional duration unique to this test process names the job,
        // which lives over 6 s unless the box takes it down: at most 2 s to
        // start and 2 s to go leaves it running throughout both polls.
        let duration = format!("6.{}", std::process::id());
        let job = ["sleep", duration.as_str()];
        let mut s = b.start();
        let out = s.shell(&format!("(sleep {duration} &)"));
        assert!(out.contains("[exit status 0]"), "{out}");
        assert!(
            await_host_process(&job, true),
            "the background sleep never started"
        );
        assert!(s.finish().success());
        assert!(
            await_host_process(&job, false),
            "the background sleep outlived the box"
        );
    }

    #[test]
    fn a_write_root_swapped_for_a_symlink_is_never_mounted() {
        if crate::testutil::skip_without_box_host() {
            return;
        }
        let b = BoxEnv::new("box-swap");
        let mut s = b.start();
        let (_, err) = s.call(
            "write_file",
            json!({"path": b.write.join("first.txt"), "content": "1"}),
        );
        assert!(!err);
        assert!(s.finish().success());

        let target = mkdir(b.t.path(), "target");
        std::fs::write(target.join("keep.txt"), "untouched").unwrap();
        std::fs::rename(&b.write, b.t.path().join("w-old")).unwrap();
        symlink(&target, &b.write).unwrap();
        let mut s = b.start();
        drop(s.stdin.take());
        let mut stderr = String::new();
        std::io::Read::read_to_string(s.child.stderr.as_mut().unwrap(), &mut stderr).unwrap();
        let status = s.child.wait().unwrap();
        assert_eq!(status.code(), Some(2), "{stderr}");
        // The policy decode refuses it first; the descriptor walk covers a
        // swap that lands after the decode.
        assert!(
            stderr.contains(&b.write.display().to_string()) && stderr.contains("not canonical"),
            "{stderr}"
        );
        assert!(s.lines.recv_timeout(Duration::from_secs(1)).is_err());
        assert_eq!(std::fs::read_dir(&target).unwrap().count(), 1);
        assert_eq!(
            std::fs::read_to_string(target.join("keep.txt")).unwrap(),
            "untouched"
        );
    }
}
