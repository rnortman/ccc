//! Test-only scratch directories, so tests can exercise the file-backed paths
//! without a dependency on a temp-file crate.

use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// A directory under the system temp dir, removed on drop.
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn new(label: &str) -> TempDir {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("ccc-test-{label}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&path).expect("creating temp dir");
        TempDir { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Writes `name` with the given octal mode.
    pub fn write_file(&self, name: &str, contents: &str, mode: u32) -> PathBuf {
        let path = self.path.join(name);
        std::fs::write(&path, contents).expect("writing temp file");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode))
                .expect("setting mode");
        }
        let _ = mode;
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// The `ccc` binary beside this test executable. `cargo test` builds it
/// because `tests/cli_binary.rs` is an integration-test target;
/// `cargo test --bin ccc` selects no integration test and does not.
#[cfg(target_os = "linux")]
pub fn ccc_binary() -> PathBuf {
    let binary = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("ccc");
    assert!(
        binary.is_file(),
        "{} is missing: run `cargo test` without `--bin ccc`",
        binary.display()
    );
    binary
}

/// Whether this host can start a bubblewrap box. `Err` carries why not.
///
/// Runs `/bin/true` under the `bwrap` found on `/usr/bin:/bin` with every
/// flag by which the Goose box asks the kernel for a privilege a host can
/// refuse. The argv is written out here rather than taken from
/// `boxed::build`, so a defect in `build` fails the box tests instead of
/// skipping them. Computed once per test process.
// TODO(goose-removal): this probe and the tests gated on it exist only while Goose ships; see TODO.md.
#[cfg(target_os = "linux")]
pub fn box_host() -> Result<(), String> {
    static RESULT: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
    RESULT.get_or_init(probe_box_host).clone()
}

#[cfg(target_os = "linux")]
fn probe_box_host() -> Result<(), String> {
    let bwrap = ["/usr/bin", "/bin"]
        .iter()
        .map(|dir| Path::new(dir).join("bwrap"))
        .find(|path| path.is_file())
        .ok_or_else(|| "no bwrap on /usr/bin:/bin".to_string())?;
    probe_box_host_with(&bwrap)
}

/// Runs the probe argv under `bwrap` and reports whether it exited 0.
#[cfg(target_os = "linux")]
fn probe_box_host_with(bwrap: &Path) -> Result<(), String> {
    use std::io::Read;
    use std::process::Stdio;

    let mut command = Command::new(bwrap);
    command
        .args([
            "--unshare-all",
            "--unshare-user",
            "--disable-userns",
            "--die-with-parent",
            "--ro-bind",
            "/",
            "/",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--",
            "/bin/true",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    let mut child = BoundedChild::spawn(&mut command)
        .map_err(|error| format!("could not spawn {}: {error}", bwrap.display()))?;
    let mut pipe = child.child.as_mut().and_then(|inner| inner.stderr.take());
    let status = child.finish_by(Instant::now() + Duration::from_secs(5));
    // On a timeout the drop kills and reaps the group, so the pipe reaches EOF.
    drop(child);
    let mut stderr = String::new();
    if let Some(pipe) = pipe.as_mut() {
        let _ = pipe.read_to_string(&mut stderr);
    }
    let stderr = stderr.trim();
    match status {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(format!("{} exited {status}: {stderr}", bwrap.display())),
        Err(error) if error.kind() == std::io::ErrorKind::TimedOut => Err(format!(
            "{} did not exit within 5 s: {stderr}",
            bwrap.display()
        )),
        Err(error) => Err(format!(
            "could not wait for {}: {error}: {stderr}",
            bwrap.display()
        )),
    }
}

/// Prints why and returns `true` when this host cannot start a bubblewrap
/// box; a test that needs a real box returns early on `true`.
#[cfg(target_os = "linux")]
pub fn skip_without_box_host() -> bool {
    match box_host() {
        Ok(()) => false,
        Err(why) => {
            eprintln!("skipped: this host cannot start a bubblewrap box: {why}");
            true
        }
    }
}

#[cfg(target_os = "linux")]
pub struct BoundedChild {
    child: Option<Child>,
    identity: crate::procs::ProcId,
}

#[cfg(target_os = "linux")]
impl BoundedChild {
    pub fn spawn(command: &mut Command) -> std::io::Result<Self> {
        use std::os::unix::process::CommandExt;

        command.process_group(0);
        let child = command.spawn()?;
        let identity = crate::procs::ProcId::of(child.id()).ok_or_else(|| {
            std::io::Error::other(format!("could not read identity for child {}", child.id()))
        })?;
        Ok(Self {
            child: Some(child),
            identity,
        })
    }

    pub fn pid(&self) -> u32 {
        self.identity.pid
    }

    pub fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        self.child
            .as_mut()
            .expect("bounded child already reaped")
            .try_wait()
    }

    pub fn finish_by(&mut self, deadline: Instant) -> std::io::Result<std::process::ExitStatus> {
        loop {
            if let Some(status) = self.try_wait()? {
                self.child = None;
                return Ok(status);
            }
            if Instant::now() >= deadline {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!("child {} did not finish before deadline", self.pid()),
                ));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    pub fn finish_by_boot<F>(
        &mut self,
        deadline: f64,
        boot_seconds: F,
    ) -> std::io::Result<std::process::ExitStatus>
    where
        F: Fn() -> f64,
    {
        loop {
            if let Some(status) = self.try_wait()? {
                self.child = None;
                return Ok(status);
            }
            if boot_seconds() >= deadline {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!("child {} did not finish before boot deadline", self.pid()),
                ));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for BoundedChild {
    fn drop(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        match child.try_wait() {
            Ok(Some(_)) => return,
            Ok(None) => {}
            Err(error) => eprintln!(
                "could not inspect child {} during cleanup: {error}",
                self.pid()
            ),
        }
        match self.identity.alive() {
            crate::procs::Liveness::Alive | crate::procs::Liveness::Unknown(_) => {
                if let Err(error) = crate::procs::signal_group(self.identity.pid, libc::SIGKILL) {
                    if error.raw_os_error() != Some(libc::ESRCH) {
                        eprintln!(
                            "could not kill child group {} during cleanup: {error}",
                            self.pid()
                        );
                    }
                }
            }
            crate::procs::Liveness::Dead => {}
        }
        if let Err(error) = child.wait() {
            eprintln!(
                "could not reap child {} during cleanup: {error}",
                self.pid()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_violation(line: &str) -> bool {
        if line.contains("while :") // fixture-scan: allow scanner spelling
            || line.contains("while true") // fixture-scan: allow scanner spelling
            || line.contains("for ((;;))") // fixture-scan: allow scanner spelling
            || line.contains(concat!("sleep", " infinity"))
            || line.contains(concat!("tail", " -f"))
            || line.contains("date +%s") // fixture-scan: allow scanner spelling
            || line.contains("date +'%s'") // fixture-scan: allow scanner spelling
            || line.contains("date +\"%s\"")
        // fixture-scan: allow scanner spelling
        {
            return true;
        }
        let mut rest = line;
        while let Some(index) = rest.find("sleep") {
            rest = &rest[index + "sleep".len()..];
            let trimmed = rest.trim_start_matches(|c: char| {
                c.is_ascii_whitespace() || c == '\\' || c == '"' || c == '\''
            });
            let mut number = String::new();
            for c in trimmed.chars() {
                if c.is_ascii_digit() || c == '.' {
                    number.push(c);
                } else {
                    break;
                }
            }
            if let Ok(value) = number.parse::<f64>() {
                if value > 10.0 {
                    return true;
                }
            }
        }
        false
    }

    fn assert_allowed(line: &str) {
        assert!(
            !fixture_violation(line),
            "unexpected fixture spelling: {line}"
        );
    }

    #[test]
    fn fixture_scan_helper_rejects_forbidden_spellings() {
        for line in [
            "while :",                    // fixture-scan: allow test input
            "while true",                 // fixture-scan: allow test input
            "for ((;;))",                 // fixture-scan: allow test input
            "sleep infinity",             // fixture-scan: allow test input
            "tail -f file",               // fixture-scan: allow test input
            r#"script = \"sleep 60\n\""#, // fixture-scan: allow test input
            "date +%s",                   // fixture-scan: allow test input
        ] {
            assert!(fixture_violation(line));
        }
    }

    #[test]
    fn fixture_scan_helper_accepts_bounded_sleep_forms() {
        for line in ["sleep 10", "sleep 0.5", r#"sleep \"$r\""#] {
            assert_allowed(line);
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn the_box_probe_answers_by_the_exit_status_of_bwrap() {
        let dir = TempDir::new("box-probe");
        let accepts = dir.write_file("accepts", "#!/bin/sh\nexit 0\n", 0o755);
        let refuses = dir.write_file(
            "refuses",
            "#!/bin/sh\necho 'stand-in refusal' >&2\nexit 3\n",
            0o755,
        );

        assert_eq!(probe_box_host_with(&accepts), Ok(()));

        let why = probe_box_host_with(&refuses).expect_err("a non-zero exit is a refusal");
        assert!(
            why.contains(&refuses.display().to_string()),
            "reason does not name the stand-in: {why}"
        );
        assert!(
            why.contains("stand-in refusal"),
            "reason does not carry stderr: {why}"
        );

        let not_executable = dir.write_file("not-executable", "#!/bin/sh\nexit 0\n", 0o644);
        for unspawnable in [dir.path().join("absent"), not_executable] {
            let why =
                probe_box_host_with(&unspawnable).expect_err("an unspawnable bwrap is a refusal");
            assert!(
                why.starts_with(&format!("could not spawn {}", unspawnable.display())),
                "reason does not name the unspawnable path: {why}"
            );
        }
    }

    #[test]
    fn no_unbounded_fixtures_in_test_source() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        collect_rs(&root, &mut files);
        files.sort();
        for file in files {
            let source = std::fs::read_to_string(&file).unwrap();
            for (line_number, line) in source.lines().enumerate() {
                if fixture_violation(line) && !line.contains("// fixture-scan: allow") {
                    let relative = file.strip_prefix(&root).unwrap().display();
                    panic!(
                        "fixture scan found {relative}:{}: every fixture needs an independent termination bound; see \"Test fixture hygiene\" in CLAUDE.md",
                        line_number + 1
                    );
                }
            }
        }
    }

    fn collect_rs(path: &Path, files: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                collect_rs(&path, files);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }
}
