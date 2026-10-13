//! Session sidecars: what a resume needs to reproduce the original invocation.
//! The Claude account (`profile`, `auth`) is recorded for reference only; each
//! resume selects its account anew.
//!
//! The id is the harness's own session id (Claude `session_id`, Codex
//! `thread_id`), so it is also usable directly with `claude --resume` and
//! `codex exec resume`.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::agents::Sandbox;
use crate::config::{ClaudeAuth, Effort, EnvSource, Harness};

/// One recorded session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionRecord {
    pub harness: Harness,
    /// `None` for a roleless session.
    #[serde(default)]
    pub agent: Option<String>,
    pub model: String,
    #[serde(default)]
    pub effort: Option<Effort>,
    /// The Claude profile the last turn ran under; it does not constrain the
    /// account a later resume selects.
    #[serde(default)]
    pub profile: Option<String>,
    /// The auth mode the last turn ran under; a later resume uses the
    /// configured mode.
    #[serde(default)]
    pub auth: Option<ClaudeAuth>,
    pub cwd: PathBuf,
    pub sandbox: Sandbox,
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    pub created_at: jiff::Timestamp,
    pub last_used_at: jiff::Timestamp,
    /// Audit trace id of the invocation that last owned this session.
    #[serde(default)]
    pub trace: Option<String>,
    /// Audit invocation id of the invocation that last owned this session.
    #[serde(default)]
    pub inv: Option<String>,
    /// The session this one was forked from, when it was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forked_from: Option<String>,
}

/// An advisory per-session exclusion lock. The file is deliberately retained
/// so its existence does not imply ownership.
#[derive(Debug)]
pub struct SessionLock {
    file: File,
}

impl SessionLock {
    fn new(file: File) -> Self {
        Self { file }
    }
}

impl Drop for SessionLock {
    /// Unlocks explicitly rather than by closing: a child forked while the
    /// lock was held shares its open file description until it execs, and
    /// closing only our descriptor would leave the lock held by that copy.
    fn drop(&mut self) {
        // `flock(LOCK_UN)` on a descriptor we own and hold the lock through
        // does not block and has no failure mode on a local filesystem. If it
        // ever failed, closing the descriptor still releases the lock once
        // any forked copy execs or exits, so there is nothing to report.
        let _ = self.file.unlock();
    }
}

/// Attempts to own the lock for `id` without waiting for another owner.
pub fn try_lock_in(dir: &Path, id: &str) -> Result<Option<SessionLock>> {
    let path = path_in(dir, id)?.with_extension("lock");
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating session directory {}", dir.display()))?;
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options
        .open(&path)
        .with_context(|| format!("opening session lock {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if file.metadata()?.mode() & 0o777 != 0o600 {
            std::fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o600))?;
        }
    }
    match file.try_lock() {
        Ok(()) => Ok(Some(SessionLock::new(file))),
        Err(std::fs::TryLockError::WouldBlock) => Ok(None),
        Err(std::fs::TryLockError::Error(e)) => {
            Err(e).with_context(|| format!("locking session {}", path.display()))
        }
    }
}

impl SessionRecord {
    /// Stamps `last_used_at` with this turn's time.
    pub fn touch(&mut self, now: jiff::Timestamp) {
        self.last_used_at = now;
    }
}

/// `$XDG_STATE_HOME/ccc/sessions`, or `~/.local/state/ccc/sessions`.
pub fn sessions_dir_in(env: &dyn EnvSource) -> PathBuf {
    crate::config::state_home_in(env).join("ccc/sessions")
}

/// Session ids come from the harnesses (UUIDs) and are used as file names.
pub(crate) fn check_id(id: &str) -> Result<()> {
    let ok = !id.is_empty()
        && id.len() <= 128
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if !ok {
        bail!("invalid session id {id:?}");
    }
    Ok(())
}

/// The file path for session `id` inside `dir`.
pub fn path_in(dir: &Path, id: &str) -> Result<PathBuf> {
    check_id(id)?;
    Ok(dir.join(format!("{id}.json")))
}

/// Persists `record` as `<dir>/<id>.json`, creating `dir` if needed. The file
/// is written whole and renamed into place, so a kill mid-write cannot leave a
/// truncated record behind.
pub fn save_in(dir: &Path, id: &str, record: &SessionRecord) -> Result<()> {
    let path = path_in(dir, id)?;
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating session directory {}", dir.display()))?;
    let text = serde_json::to_string_pretty(record)?;
    crate::fsutil::write_atomically(&path, &text)
}

/// Loads `<dir>/<id>.json` and deserializes it.
pub fn load_from(dir: &Path, id: &str) -> Result<SessionRecord> {
    let path = path_in(dir, id)?;
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            bail!(
                "unknown session id {id:?} (no record at {})",
                path.display()
            )
        }
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    serde_json::from_str(&text).with_context(|| {
        format!(
            "session record {} is corrupt; delete it and start a new session",
            path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    fn record() -> SessionRecord {
        SessionRecord {
            harness: Harness::Codex,
            agent: Some("explorer".into()),
            model: "gpt-5.6-luna".into(),
            effort: Some(Effort::Low),
            profile: None,
            auth: None,
            cwd: PathBuf::from("/srv/work"),
            sandbox: Sandbox::ReadOnly,
            tools: Some(vec!["Read".into()]),
            created_at: jiff::Timestamp::now(),
            last_used_at: jiff::Timestamp::now(),
            trace: None,
            inv: None,
            forked_from: None,
        }
    }

    #[test]
    fn round_trips() {
        let dir = TempDir::new("sessions");
        let r = record();
        save_in(dir.path(), "0199aa-bb", &r).unwrap();
        assert_eq!(load_from(dir.path(), "0199aa-bb").unwrap(), r);
    }

    #[test]
    fn claude_fields_round_trip() {
        let dir = TempDir::new("sessions-claude");
        let mut r = record();
        r.harness = Harness::Claude;
        r.profile = Some("team03".into());
        r.auth = Some(ClaudeAuth::Token);
        save_in(dir.path(), "9f22dd1b", &r).unwrap();
        let back = load_from(dir.path(), "9f22dd1b").unwrap();
        assert_eq!(back.auth, Some(ClaudeAuth::Token));
        assert_eq!(back.profile.as_deref(), Some("team03"));
    }

    #[test]
    fn a_string_agent_reads_as_some_and_a_roleless_record_round_trips_as_null() {
        let dir = TempDir::new("sessions-roleless");
        let value = serde_json::to_value(record()).unwrap();
        assert_eq!(value["agent"], "explorer");
        std::fs::write(dir.path().join("typed.json"), value.to_string()).unwrap();
        let typed = load_from(dir.path(), "typed").unwrap();
        assert_eq!(typed.agent.as_deref(), Some("explorer"));

        let mut r = record();
        r.agent = None;
        save_in(dir.path(), "roleless", &r).unwrap();
        let text = std::fs::read_to_string(dir.path().join("roleless.json")).unwrap();
        let raw: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert!(raw["agent"].is_null(), "{text}");
        assert_eq!(load_from(dir.path(), "roleless").unwrap(), r);
    }

    #[test]
    fn unknown_id_is_an_error() {
        let dir = TempDir::new("sessions-missing");
        let err = load_from(dir.path(), "nope").unwrap_err().to_string();
        assert!(err.contains("unknown session id"), "{err}");
    }

    #[test]
    fn ids_may_not_escape_the_directory() {
        let dir = TempDir::new("sessions-escape");
        for id in ["../etc/passwd", "a/b", "", "with space"] {
            assert!(path_in(dir.path(), id).is_err(), "id {id:?}");
        }
    }

    #[test]
    fn touch_advances_last_used_only() {
        let epoch: jiff::Timestamp = "2020-01-01T00:00:00Z".parse().unwrap();
        let mut r = record();
        r.created_at = epoch;
        r.last_used_at = epoch;
        r.touch(jiff::Timestamp::now());
        assert!(r.last_used_at > epoch);
        assert_eq!(r.created_at, epoch);
    }

    #[test]
    fn timestamps_are_parsed_not_just_carried() {
        let dir = TempDir::new("sessions-time");
        let r = record();
        save_in(dir.path(), "t1", &r).unwrap();
        let text = std::fs::read_to_string(dir.path().join("t1.json")).unwrap();
        let bad = text.replace(&r.created_at.to_string(), "the day before yesterday");
        assert_ne!(bad, text);
        std::fs::write(dir.path().join("t2.json"), bad).unwrap();
        assert!(load_from(dir.path(), "t2").is_err());
    }

    #[test]
    fn a_corrupt_record_says_so_and_says_what_to_do() {
        let dir = TempDir::new("sessions-corrupt");
        std::fs::write(dir.path().join("t3.json"), "{\"harness\":").unwrap();
        let err = format!("{:#}", load_from(dir.path(), "t3").unwrap_err());
        assert!(err.contains("corrupt"), "{err}");
        assert!(err.contains("t3.json"), "{err}");
    }

    #[test]
    fn saving_creates_the_directory_and_leaves_no_temporary_behind() {
        let dir = TempDir::new("sessions-nested");
        let nested = dir.path().join("a/b/c");
        let r = record();
        save_in(&nested, "n1", &r).unwrap();
        assert_eq!(load_from(&nested, "n1").unwrap(), r);
        let names: Vec<String> = std::fs::read_dir(&nested)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["n1.json".to_string()], "{names:?}");
    }

    #[test]
    fn saving_rejects_a_bad_id_before_touching_the_filesystem() {
        let dir = TempDir::new("sessions-badid");
        let target = dir.path().join("sub");
        assert!(save_in(&target, "../x", &record()).is_err());
        assert!(!target.exists());
    }

    #[test]
    fn the_sessions_directory_follows_the_state_home() {
        let env: std::collections::BTreeMap<String, String> =
            [("HOME".to_string(), "/home/tester".to_string())]
                .into_iter()
                .collect();
        assert_eq!(
            sessions_dir_in(&env),
            PathBuf::from("/home/tester/.local/state/ccc/sessions")
        );
        let env: std::collections::BTreeMap<String, String> =
            [("XDG_STATE_HOME".to_string(), "/xdg/state".to_string())]
                .into_iter()
                .collect();
        assert_eq!(
            sessions_dir_in(&env),
            PathBuf::from("/xdg/state/ccc/sessions")
        );
    }

    #[test]
    fn the_id_length_ceiling_is_where_it_says() {
        let dir = Path::new("/sessions");
        assert!(path_in(dir, &"a".repeat(128)).is_ok());
        assert!(path_in(dir, &"a".repeat(129)).is_err());
    }

    #[test]
    fn old_records_default_the_audit_backlinks() {
        let dir = TempDir::new("sessions-old");
        let mut value = serde_json::to_value(record()).unwrap();
        value.as_object_mut().unwrap().remove("trace");
        value.as_object_mut().unwrap().remove("inv");
        std::fs::create_dir_all(dir.path()).unwrap();
        std::fs::write(dir.path().join("old.json"), value.to_string()).unwrap();
        let loaded = load_from(dir.path(), "old").unwrap();
        assert_eq!(loaded.trace, None);
        assert_eq!(loaded.inv, None);
    }

    #[test]
    fn a_sidecar_without_forked_from_deserialises() {
        let dir = TempDir::new("sessions-unforked");
        let value = serde_json::to_value(record()).unwrap();
        assert!(value.get("forked_from").is_none(), "{value}");
        std::fs::create_dir_all(dir.path()).unwrap();
        std::fs::write(dir.path().join("plain.json"), value.to_string()).unwrap();
        assert_eq!(load_from(dir.path(), "plain").unwrap().forked_from, None);
    }

    #[test]
    fn forked_from_round_trips() {
        let dir = TempDir::new("sessions-forked");
        let mut r = record();
        r.forked_from = Some("parent".into());
        save_in(dir.path(), "child", &r).unwrap();
        let text = std::fs::read_to_string(dir.path().join("child.json")).unwrap();
        assert!(text.contains("\"forked_from\""), "{text}");
        assert_eq!(load_from(dir.path(), "child").unwrap(), r);

        save_in(dir.path(), "plain", &record()).unwrap();
        let text = std::fs::read_to_string(dir.path().join("plain.json")).unwrap();
        assert!(!text.contains("forked_from"), "{text}");
    }

    #[test]
    fn session_lock_is_exclusive_and_retained_after_release() {
        let dir = TempDir::new("sessions-lock");
        let first = try_lock_in(dir.path(), "s1").unwrap().unwrap();
        assert!(
            try_lock_in(dir.path(), "s1").unwrap().is_none(),
            "s1.lock was acquired a second time while the first lock was still held"
        );
        let path = dir.path().join("s1.lock");
        assert!(path.is_file());
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 0);
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            assert_eq!(std::fs::metadata(&path).unwrap().mode() & 0o777, 0o600);
        }
        drop(first);
        assert!(
            try_lock_in(dir.path(), "s1").unwrap().is_some(),
            "s1.lock still held after the holder dropped it"
        );
        assert!(try_lock_in(dir.path(), "../bad").is_err());
    }

    /// A child forked while the lock is held keeps a copy of its descriptor
    /// until it execs or exits; dropping the lock must release it anyway.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_dropped_lock_is_free_while_a_forked_child_still_holds_its_descriptor() {
        let dir = TempDir::new("sessions-lock-forked");
        let lock = try_lock_in(dir.path(), "s1").unwrap().unwrap();
        let mut gate = [0; 2];
        assert_eq!(
            unsafe { libc::pipe2(gate.as_mut_ptr(), libc::O_CLOEXEC) },
            0
        );
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0, "fork failed");
        if pid == 0 {
            // Async-signal-safe only: block until the test closes the write
            // end, or 5 s pass, then exit without running any destructors.
            unsafe {
                libc::close(gate[1]);
                let mut fds = libc::pollfd {
                    fd: gate[0],
                    events: libc::POLLIN,
                    revents: 0,
                };
                libc::poll(&mut fds, 1, 5000);
                libc::_exit(0);
            }
        }
        unsafe { libc::close(gate[0]) };
        use std::os::unix::fs::MetadataExt;
        let lock_file = std::fs::metadata(dir.path().join("s1.lock")).unwrap();
        let child_holds = std::fs::read_dir(format!("/proc/{pid}/fd"))
            .expect("reading the forked child's descriptor table")
            .flatten()
            .filter_map(|fd| std::fs::metadata(fd.path()).ok())
            .any(|m| (m.dev(), m.ino()) == (lock_file.dev(), lock_file.ino()));

        drop(lock);
        let reacquired = try_lock_in(dir.path(), "s1").unwrap();

        unsafe { libc::close(gate[1]) };
        let mut status = 0;
        assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
        assert!(
            child_holds,
            "the forked child had no copy of the lock descriptor"
        );
        assert!(
            reacquired.is_some(),
            "s1.lock stayed held by the forked child's copy after the owner dropped it"
        );
    }
}
