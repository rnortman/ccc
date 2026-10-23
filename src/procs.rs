//! Linux process identities and best-effort process-tree operations.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProcId {
    pub pid: u32,
    pub start: u64,
    pub boot_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Liveness {
    Alive,
    Dead,
    Unknown(String),
}

/// Best-effort PID presence, intentionally without process identity. This is
/// suitable only for displaying legacy audit records and never authorizes a
/// signal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Presence {
    Present,
    Absent,
    Unknown(String),
}

#[cfg(target_os = "linux")]
pub fn presence(pid: u32) -> Presence {
    match std::fs::metadata(format!("/proc/{pid}")) {
        Ok(_) => Presence::Present,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Presence::Absent,
        Err(e) => Presence::Unknown(e.to_string()),
    }
}

#[cfg(not(target_os = "linux"))]
pub fn presence(pid: u32) -> Presence {
    #[cfg(unix)]
    {
        let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
        if result == 0 {
            Presence::Present
        } else if std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
            Presence::Absent
        } else {
            Presence::Unknown("PID presence could not be determined".into())
        }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        Presence::Unknown("PID presence is unsupported".into())
    }
}

#[cfg(target_os = "linux")]
fn boot_id() -> anyhow::Result<String> {
    Ok(std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?
        .trim()
        .to_string())
}

#[cfg(target_os = "linux")]
fn read_stat(pid: u32) -> std::io::Result<(u32, u64, char)> {
    let text = std::fs::read_to_string(format!("/proc/{pid}/stat"))?;
    let close = text.rfind(')').ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "missing comm delimiter")
    })?;
    let fields: Vec<_> = text
        .get(close + 1..)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid stat suffix"))?
        .split_whitespace()
        .collect();
    let state = fields
        .first()
        .and_then(|field| field.chars().next())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing state"))?;
    let ppid = fields
        .get(1)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing parent pid"))?
        .parse()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid parent pid"))?;
    let start = fields
        .get(19)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "missing start time"))?
        .parse()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid start time"))?;
    Ok((ppid, start, state))
}

#[cfg(target_os = "linux")]
fn stat(pid: u32) -> Option<(u32, u64, char)> {
    read_stat(pid).ok()
}

#[cfg(target_os = "linux")]
impl ProcId {
    pub fn current() -> anyhow::Result<Self> {
        let boot_id = boot_id().context("reading Linux boot identity")?;
        let pid = std::process::id();
        let (_, start, _) = stat(pid).ok_or_else(|| anyhow::anyhow!("reading /proc/{pid}/stat"))?;
        Ok(Self {
            pid,
            start,
            boot_id,
        })
    }

    pub fn of(pid: u32) -> Option<Self> {
        let boot_id = boot_id().ok()?;
        let (_, start, _) = stat(pid)?;
        Some(Self {
            pid,
            start,
            boot_id,
        })
    }

    pub fn alive(&self) -> Liveness {
        let current_boot = match boot_id() {
            Ok(id) => id,
            Err(e) => return Liveness::Unknown(e.to_string()),
        };
        let (_, start, state) = match read_stat(self.pid) {
            Ok(stat) => stat,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Liveness::Dead,
            // TODO(procs-alive-esrch-read-is-unknown): a /proc/<pid>/stat read that fails with ESRCH (task released between the directory lookup and the read) is reported as Unknown, though the process is gone and the kill path treats ESRCH as "already gone"; mapping it to Dead is a liveness-rule change.
            Err(e) => return Liveness::Unknown(e.to_string()),
        };
        if self.boot_id != current_boot || self.start != start || matches!(state, 'Z' | 'X') {
            Liveness::Dead
        } else {
            Liveness::Alive
        }
    }
}

#[cfg(not(target_os = "linux"))]
impl ProcId {
    pub fn current() -> anyhow::Result<Self> {
        anyhow::bail!("process identities are unsupported on this platform")
    }

    pub fn of(_pid: u32) -> Option<Self> {
        None
    }

    pub fn alive(&self) -> Liveness {
        #[cfg(unix)]
        {
            let result = unsafe { libc::kill(self.pid as libc::pid_t, 0) };
            if result == 0 {
                return Liveness::Unknown("process identity cannot be verified".into());
            }
            if std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
                Liveness::Dead
            } else {
                Liveness::Unknown("process identity cannot be verified".into())
            }
        }
        #[cfg(not(unix))]
        {
            let _ = self;
            Liveness::Unknown("process identity cannot be verified".into())
        }
    }
}

#[cfg(target_os = "linux")]
pub fn descendants(pid: u32) -> Vec<ProcId> {
    let Ok(boot_id) = boot_id() else {
        return Vec::new();
    };
    let mut facts = std::collections::HashMap::new();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return Vec::new();
    };
    for entry in entries.flatten() {
        let Ok(pid) = entry.file_name().to_string_lossy().parse::<u32>() else {
            continue;
        };
        if let Some((ppid, start, _)) = stat(pid) {
            facts.insert(pid, (ppid, start));
        }
    }
    let mut children = Vec::new();
    let mut frontier = vec![pid];
    while let Some(parent) = frontier.pop() {
        for (&candidate, &(ppid, start)) in &facts {
            if ppid == parent && candidate != pid {
                children.push(ProcId {
                    pid: candidate,
                    start,
                    boot_id: boot_id.clone(),
                });
                frontier.push(candidate);
            }
        }
    }
    children.sort_by_key(|p| p.pid);
    children
}

#[cfg(not(target_os = "linux"))]
pub fn descendants(_pid: u32) -> Vec<ProcId> {
    Vec::new()
}

#[cfg(target_os = "linux")]
pub fn ancestors() -> Vec<ProcId> {
    let Ok(boot_id) = boot_id() else {
        return Vec::new();
    };
    let mut pid = std::process::id();
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();
    while let Some((ppid, _start, _)) = stat(pid) {
        if ppid == 0 || ppid == 1 || !seen.insert(ppid) {
            break;
        }
        let Some((_, parent_start, _)) = stat(ppid) else {
            break;
        };
        result.push(ProcId {
            pid: ppid,
            start: parent_start,
            boot_id: boot_id.clone(),
        });
        pid = ppid;
    }
    result
}

#[cfg(not(target_os = "linux"))]
pub fn ancestors() -> Vec<ProcId> {
    Vec::new()
}

#[cfg(unix)]
fn valid_signal(sig: i32) -> std::io::Result<libc::c_int> {
    if sig <= 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "signal must be positive",
        ));
    }
    Ok(sig as libc::c_int)
}

#[cfg(unix)]
pub fn signal_group(pgid: u32, sig: i32) -> std::io::Result<()> {
    let sig = valid_signal(sig)?;
    if pgid <= 1 || pgid > i32::MAX as u32 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid process group",
        ));
    }
    let result = unsafe { libc::kill(-(pgid as libc::pid_t), sig) };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(unix)]
pub fn signal(pid: u32, sig: i32) -> std::io::Result<()> {
    let sig = valid_signal(sig)?;
    if pid == 0 || pid > i32::MAX as u32 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "invalid process",
        ));
    }
    let result = unsafe { libc::kill(pid as libc::pid_t, sig) };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(unix))]
pub fn signal_group(_pgid: u32, _sig: i32) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "signals are unsupported on this platform",
    ))
}
#[cfg(not(unix))]
pub fn signal(_pid: u32, _sig: i32) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "signals are unsupported on this platform",
    ))
}

#[cfg(target_os = "linux")]
use anyhow::Context;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;
    use std::fs;
    use std::process::Command;
    use std::time::Duration;

    #[test]
    fn pid_presence_reports_live_and_missing_without_identity_authority() {
        let current = std::process::id();
        assert_eq!(presence(current), Presence::Present);
        let missing = u32::MAX;
        assert_eq!(presence(missing), Presence::Absent);
        let identity = ProcId::of(current).expect("current process identity");
        assert_eq!(identity.pid, current);
    }

    #[test]
    fn current_identity_is_live_and_changed_identity_is_dead() {
        let current = ProcId::current().unwrap();
        assert_eq!(current.alive(), Liveness::Alive);
        let mut changed = current.clone();
        changed.start += 1;
        assert_eq!(changed.alive(), Liveness::Dead);
        changed = current.clone();
        changed.boot_id.push('x');
        assert_eq!(changed.alive(), Liveness::Dead);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn reaped_child_identity_is_dead() {
        let mut child = Command::new("sh").arg("-c").arg("exit 0").spawn().unwrap();
        let identity = ProcId::of(child.id()).unwrap();
        child.wait().unwrap();
        assert_eq!(identity.alive(), Liveness::Dead);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn zombie_child_identity_is_dead_before_reap() {
        let mut child = Command::new("sh").arg("-c").arg("exit 0").spawn().unwrap();
        let identity = ProcId::of(child.id()).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            if read_stat(child.id()).is_ok_and(|(_, _, state)| state == 'Z') {
                assert_eq!(identity.alive(), Liveness::Dead);
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "child did not become a zombie"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        child.wait().unwrap();
        assert_eq!(identity.alive(), Liveness::Dead);
    }

    #[cfg(unix)]
    #[test]
    fn signal_group_rejects_broadcast_boundaries_before_kill() {
        for pgid in [0, 1, i32::MAX as u32 + 1] {
            assert_eq!(
                signal_group(pgid, libc::SIGTERM).unwrap_err().kind(),
                std::io::ErrorKind::InvalidInput
            );
        }
    }

    #[cfg(target_os = "linux")]
    struct SleepFixture {
        child: Option<std::process::Child>,
        identities: Vec<ProcId>,
    }

    #[cfg(target_os = "linux")]
    impl SleepFixture {
        fn cleanup(&mut self) {
            for identity in &self.identities {
                if identity.alive() == Liveness::Alive {
                    let _ = signal(identity.pid, libc::SIGTERM);
                }
            }
            if let Some(mut child) = self.child.take() {
                let _ = child.wait();
            }
        }
    }

    #[cfg(target_os = "linux")]
    impl Drop for SleepFixture {
        fn drop(&mut self) {
            self.cleanup();
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn descendants_finds_live_children_and_identity_checked_cleanup_reaps_them() {
        let temp = TempDir::new("proc-descendants");
        let pids = temp.path().join("pids");
        let child = Command::new("sh")
            .args([
                "-c",
                "sleep 10 & echo $! >> \"$1\"; sleep 10 & echo $! >> \"$1\"; wait",
                "sh",
                pids.to_str().unwrap(),
            ])
            .spawn()
            .unwrap();
        let root = ProcId::of(child.id()).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        let sleep_pids = loop {
            if let Ok(text) = fs::read_to_string(&pids) {
                let ids = text
                    .lines()
                    .filter_map(|line| line.parse::<u32>().ok())
                    .collect::<Vec<_>>();
                if ids.len() == 2 {
                    break ids;
                }
            }
            assert!(
                std::time::Instant::now() < deadline,
                "sleep fixture did not start"
            );
            std::thread::sleep(Duration::from_millis(10));
        };
        let identities = sleep_pids
            .iter()
            .map(|pid| ProcId::of(*pid).unwrap())
            .collect::<Vec<_>>();
        let fixture = &mut SleepFixture {
            child: Some(child),
            identities: identities.clone(),
        };
        let descendants = descendants(root.pid);
        assert!(!descendants.iter().any(|identity| identity.pid == root.pid));
        for identity in &identities {
            assert!(descendants.iter().any(|found| found == identity));
            assert_eq!(identity.alive(), Liveness::Alive);
        }
        fixture.cleanup();
        assert!(fixture.child.is_none());
        assert!(identities
            .iter()
            .all(|identity| identity.alive() == Liveness::Dead));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn spawned_grandchild_chain_contains_this_test_process() {
        let temp = TempDir::new("proc-grandchild");
        let chain = temp.path().join("chain");
        let mut child = Command::new("sh")
            .args([
                "-c",
                "sh -c 'pid=$$; while [ \"$pid\" -gt 1 ]; do echo \"$pid\"; line=$(cat /proc/$pid/stat); line=${line##*) }; set -- $line; pid=$2; done' > \"$1\"",
                "sh",
                chain.to_str().unwrap(),
            ])
            .spawn()
            .unwrap();
        child.wait().unwrap();
        let chain = fs::read_to_string(chain).unwrap();
        assert!(chain
            .lines()
            .filter_map(|pid| pid.parse::<u32>().ok())
            .any(|pid| pid == std::process::id()));
    }

    #[test]
    fn ancestors_are_live_nearest_first_without_self_or_init() {
        let self_id = std::process::id();
        let chain = ancestors();
        assert!(!chain.iter().any(|p| p.pid == self_id || p.pid == 1));
        assert!(chain.windows(2).all(|p| p[0].pid != p[1].pid));
        assert!(chain.iter().all(|p| p.alive() == Liveness::Alive));
    }
}
