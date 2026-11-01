//! The confinement policy of a Goose run: which ccc tool groups the model gets,
//! which agents it may run, and the read, write and exec roots those tools are
//! confined to. A computed policy has passed the protected-directory and
//! write/exec separation checks, and travels as a single hex token.

use std::collections::BTreeSet;
use std::path::{Component, Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::{
    agents_config_path_in, config_path_in, expand_tilde_in, home_dir_in, state_home_in, Config,
    EnvSource,
};

/// A ccc tool group offered to a Goose agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GooseTool {
    Read,
    Write,
    Shell,
    Ccc,
}

impl GooseTool {
    /// The group's name as written in `goose_tools`; inverse of [`GooseTool::parse`].
    pub fn as_str(self) -> &'static str {
        match self {
            GooseTool::Read => "read",
            GooseTool::Write => "write",
            GooseTool::Shell => "shell",
            GooseTool::Ccc => "ccc",
        }
    }

    /// Parses a group name, ignoring surrounding whitespace.
    pub fn parse(s: &str) -> Result<GooseTool> {
        match s.trim() {
            "read" => Ok(GooseTool::Read),
            "write" => Ok(GooseTool::Write),
            "shell" => Ok(GooseTool::Shell),
            "ccc" => Ok(GooseTool::Ccc),
            other => {
                bail!(
                    "unknown goose tool group {other:?} (expected \"read\", \"write\", \"shell\" or \"ccc\")"
                )
            }
        }
    }
}

/// The confinement a Goose run's tool server enforces. Every root is absolute
/// and canonical.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GoosePolicy {
    pub tools: BTreeSet<GooseTool>,
    /// The agents the run may launch; empty unless `tools` holds `Ccc`.
    pub agents: Vec<String>,
    pub read_roots: Vec<PathBuf>,
    pub write_roots: Vec<PathBuf>,
    pub exec_roots: Vec<PathBuf>,
}

impl GoosePolicy {
    /// Whether a run under the policy needs the bubblewrap box: any of its
    /// tool groups touches files or runs programs.
    pub fn needs_box(&self) -> bool {
        self.tools
            .iter()
            .any(|t| matches!(t, GooseTool::Read | GooseTool::Write | GooseTool::Shell))
    }

    /// The hex-encoded JSON of the policy: one argv token with no whitespace or
    /// quoting needs.
    pub fn to_hex(&self) -> String {
        hex_encode(&serde_json::to_vec(self).expect("a policy always serialises"))
    }

    /// Decodes a [`GoosePolicy::to_hex`] token and checks that every root is
    /// absolute and canonical.
    pub fn from_hex(token: &str) -> Result<GoosePolicy> {
        let bytes = hex_decode(token).map_err(|e| anyhow::anyhow!("policy token: {e}"))?;
        let policy: GoosePolicy =
            serde_json::from_slice(&bytes).context("policy token is not a policy")?;
        policy.check_canonical()?;
        Ok(policy)
    }

    fn check_canonical(&self) -> Result<()> {
        for root in self.all_roots() {
            if !root.is_absolute() {
                bail!("policy root {} is not absolute", root.display());
            }
            let canonical = std::fs::canonicalize(root)
                .with_context(|| format!("policy root {}", root.display()))?;
            if canonical != *root {
                bail!(
                    "policy root {} is not canonical (resolves to {})",
                    root.display(),
                    canonical.display()
                );
            }
        }
        Ok(())
    }

    fn all_roots(&self) -> impl Iterator<Item = &PathBuf> {
        self.read_roots
            .iter()
            .chain(&self.write_roots)
            .chain(&self.exec_roots)
    }

    /// The union of the execution surfaces of the exec roots.
    pub fn exec_surfaces(&self) -> Result<Vec<PathBuf>> {
        let mut union = Vec::new();
        for root in &self.exec_roots {
            for element in ExecSurface::of(root)?.elements() {
                push_unique(&mut union, element.to_path_buf());
            }
        }
        Ok(union)
    }
}

/// Lower-case hex of `bytes`.
pub fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{b:02x}"));
    }
    out
}

/// The bytes a [`hex_encode`] string holds; refuses odd length or a non-hex
/// character.
pub fn hex_decode(token: &str) -> Result<Vec<u8>> {
    let token = token.as_bytes();
    if !token.len().is_multiple_of(2) {
        bail!("odd length");
    }
    let mut bytes = Vec::with_capacity(token.len() / 2);
    for pair in token.chunks(2) {
        let hi = hex_digit(pair[0])?;
        let lo = hex_digit(pair[1])?;
        bytes.push(hi << 4 | lo);
    }
    Ok(bytes)
}

fn hex_digit(c: u8) -> Result<u8> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => bail!("not hex"),
    }
}

/// Whether two paths overlap: equal, or one contains the other.
pub fn overlaps(a: &Path, b: &Path) -> bool {
    a.starts_with(b) || b.starts_with(a)
}

/// Whether `path` is equal to or under one of `roots`.
pub fn under_any(path: &Path, roots: &[PathBuf]) -> bool {
    roots.iter().any(|r| path.starts_with(r))
}

fn push_unique(v: &mut Vec<PathBuf>, p: PathBuf) {
    if !v.contains(&p) {
        v.push(p);
    }
}

/// The places a harness or git invocation running in a directory reads
/// project configuration from: the project top (the nearest ancestor holding a
/// `.git` entry, else the directory itself) and, when that `.git` is not a
/// plain directory, the git directory it names and that directory's common
/// directory. Every path is canonical.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecSurface {
    pub top: PathBuf,
    /// The `.git` entry that set `top`; `None` when no ancestor holds one.
    pub marker: Option<PathBuf>,
    pub git_dir: Option<PathBuf>,
    pub common_dir: Option<PathBuf>,
}

impl ExecSurface {
    /// The surface of the canonical directory `dir`. Reads the `.git` file and
    /// `commondir` directly; runs no `git`.
    pub fn of(dir: &Path) -> Result<ExecSurface> {
        for ancestor in dir.ancestors() {
            let marker = ancestor.join(".git");
            let Ok(meta) = std::fs::symlink_metadata(&marker) else {
                continue;
            };
            let git_dir = if meta.is_file() {
                Some(gitdir_from_file(&marker, ancestor)?)
            } else if meta.file_type().is_symlink() {
                Some(std::fs::canonicalize(&marker).with_context(|| {
                    format!("resolving the git directory link {}", marker.display())
                })?)
            } else {
                None
            };
            let common_dir = match &git_dir {
                Some(g) => Some(common_dir_of(g)?),
                None => None,
            };
            return Ok(ExecSurface {
                top: ancestor.to_path_buf(),
                marker: Some(marker),
                git_dir,
                common_dir,
            });
        }
        Ok(ExecSurface {
            top: dir.to_path_buf(),
            marker: None,
            git_dir: None,
            common_dir: None,
        })
    }

    /// `top`, then the git directory and common directory where present,
    /// without duplicates.
    pub fn elements(&self) -> Vec<&Path> {
        let mut out = vec![self.top.as_path()];
        for p in [&self.git_dir, &self.common_dir].into_iter().flatten() {
            if !out.contains(&p.as_path()) {
                out.push(p);
            }
        }
        out
    }

    /// Whether every element of this surface is equal to or under some
    /// element of `union`.
    pub fn lies_inside(&self, union: &[PathBuf]) -> bool {
        self.elements().iter().all(|e| under_any(e, union))
    }

    fn describe(&self, element: &Path) -> String {
        if element == self.top {
            match &self.marker {
                Some(m) => format!("project top {} (set by {})", element.display(), m.display()),
                None => format!("directory {}", element.display()),
            }
        } else if Some(element) == self.git_dir.as_deref() {
            format!("git directory {}", element.display())
        } else {
            format!("git common directory {}", element.display())
        }
    }
}

/// The canonical git directory a `.git` file names; a relative `gitdir:`
/// resolves against `holder`, the directory holding the file.
fn gitdir_from_file(marker: &Path, holder: &Path) -> Result<PathBuf> {
    let text =
        std::fs::read_to_string(marker).with_context(|| format!("reading {}", marker.display()))?;
    let named = text
        .lines()
        .find_map(|l| l.trim().strip_prefix("gitdir:"))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .with_context(|| format!("{} has no gitdir: line", marker.display()))?;
    let path = holder.join(named);
    std::fs::canonicalize(&path).with_context(|| {
        format!(
            "resolving the git directory {} named by {}",
            path.display(),
            marker.display()
        )
    })
}

/// The canonical common directory of a git directory: its `commondir` file,
/// relative to the git directory, or the git directory itself.
fn common_dir_of(git_dir: &Path) -> Result<PathBuf> {
    let file = git_dir.join("commondir");
    match std::fs::read_to_string(&file) {
        Ok(text) => {
            let named = text.trim();
            if named.is_empty() {
                bail!("{} is empty", file.display());
            }
            let path = git_dir.join(named);
            std::fs::canonicalize(&path).with_context(|| {
                format!(
                    "resolving the common directory {} named by {}",
                    path.display(),
                    file.display()
                )
            })
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(git_dir.to_path_buf()),
        Err(e) => Err(e).with_context(|| format!("reading {}", file.display())),
    }
}

/// Refuses a write root that overlaps any element of any exec root's
/// execution surface.
pub fn check_separation(write_roots: &[PathBuf], exec_roots: &[PathBuf]) -> Result<()> {
    for exec in exec_roots {
        let surface = ExecSurface::of(exec)?;
        for element in surface.elements() {
            for write in write_roots {
                if overlaps(write, element) {
                    bail!(
                        "write root {} overlaps the {} of exec root {}: an agent may not write \
                         where the agents it runs load project configuration",
                        write.display(),
                        surface.describe(element),
                        exec.display()
                    );
                }
            }
        }
    }
    Ok(())
}

/// The directories no write root may be, contain, or lie inside, and the
/// `PATH` entries that a write-capable run must not leave relative.
#[derive(Debug, Clone, Default)]
pub struct ProtectedDirs {
    /// Each directory with what it is, for the refusal message.
    pub dirs: Vec<(PathBuf, String)>,
    /// `$HOME`, canonical where it exists.
    pub home: Option<PathBuf>,
    /// `PATH` entries that are relative or empty, verbatim.
    pub relative_path_entries: Vec<String>,
}

impl ProtectedDirs {
    /// Gathers the protected set from ccc's configuration and environment.
    /// `self_exe` is ccc's own binary.
    pub fn gather_in(
        env: &dyn EnvSource,
        cfg: &Config,
        goose_root: &Path,
        self_exe: &Path,
    ) -> ProtectedDirs {
        let mut p = ProtectedDirs::default();
        let mut add = |path: PathBuf, what: &str| {
            p.dirs
                .push((resolve_existing_prefix(&path), what.to_string()));
        };
        for (cp, what) in [
            (config_path_in(env), "ccc site-config directory"),
            (agents_config_path_in(env), "ccc agents-config directory"),
        ] {
            if let Some(dir) = cp.path.parent() {
                add(dir.to_path_buf(), what);
            }
        }
        add(cfg.agents_dir_in(env), "agents directory");
        add(state_home_in(env).join("ccc"), "ccc state directory");
        add(goose_root.to_path_buf(), "Goose root");
        add(
            expand_tilde_in(env, &cfg.claude.token_dir),
            "[claude] token_dir",
        );
        add(
            expand_tilde_in(env, &cfg.claude.profiles_dir),
            "[claude] profiles_dir",
        );
        for var in ["CLAUDE_CONFIG_DIR", "CODEX_HOME"] {
            if let Some(v) = env.get(var).filter(|v| !v.trim().is_empty()) {
                add(expand_tilde_in(env, &v), var);
            }
        }
        if let Some(dir) = self_exe.parent() {
            add(dir.to_path_buf(), "directory of the ccc binary");
        }
        let mut binaries: Vec<(String, String)> =
            ["CCC_CLAUDE_BIN", "CCC_CODEX_BIN", "CCC_GOOSE_BIN"]
                .into_iter()
                .filter_map(|var| env.get(var).map(|v| (var.to_string(), v)))
                .collect();
        binaries.push((
            "[goose] bwrap".to_string(),
            expand_tilde_in(env, &cfg.goose.bwrap)
                .to_string_lossy()
                .into_owned(),
        ));
        for (what, v) in binaries {
            // A bare name is looked up on PATH, whose entries are protected below.
            if v.trim().is_empty() || !v.contains('/') {
                continue;
            }
            let bin = std::fs::canonicalize(&v).unwrap_or_else(|_| PathBuf::from(&v));
            if let Some(dir) = bin.parent().filter(|d| d.is_absolute()) {
                add(dir.to_path_buf(), &format!("directory of {what}"));
            }
        }
        let path_var = env.get("PATH").unwrap_or_default();
        let mut relative = Vec::new();
        for entry in path_var.split(':') {
            if entry.is_empty() || !Path::new(entry).is_absolute() {
                relative.push(entry.to_string());
            } else {
                add(PathBuf::from(entry), "PATH entry");
            }
        }
        p.relative_path_entries = relative;
        p.home = Some(resolve_existing_prefix(&home_dir_in(env)));
        p
    }

    /// Refuses `root` when it overlaps a protected directory, is `$HOME` or an
    /// ancestor of it, or lies inside a dot-named entry directly under `$HOME`.
    pub fn check_write_root(&self, root: &Path) -> Result<()> {
        for (dir, what) in &self.dirs {
            if overlaps(root, dir) {
                bail!(
                    "write root {} overlaps the {} {}",
                    root.display(),
                    what,
                    dir.display()
                );
            }
        }
        if let Some(home) = &self.home {
            if home.starts_with(root) {
                bail!(
                    "write root {} is $HOME ({}) or an ancestor of it",
                    root.display(),
                    home.display()
                );
            }
            if let Ok(rest) = root.strip_prefix(home) {
                if let Some(Component::Normal(first)) = rest.components().next() {
                    if first.to_string_lossy().starts_with('.') {
                        bail!(
                            "write root {} lies inside {}, a dot-named entry of $HOME",
                            root.display(),
                            home.join(first).display()
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// Refuses a relative or empty `PATH` entry.
    pub fn check_path_entries(&self) -> Result<()> {
        if let Some(entry) = self.relative_path_entries.first() {
            bail!(
                "PATH entry {entry:?} is relative or empty; a write-capable Goose run requires \
                 every PATH entry to be absolute"
            );
        }
        Ok(())
    }
}

/// `path` with its longest existing prefix canonicalised and the rest
/// appended lexically.
fn resolve_existing_prefix(path: &Path) -> PathBuf {
    let path = normalize(path).unwrap_or_else(|| path.to_path_buf());
    let mut rest = Vec::new();
    let mut cur = path.as_path();
    loop {
        if let Ok(c) = std::fs::canonicalize(cur) {
            let mut out = c;
            for part in rest.iter().rev() {
                out.push(part);
            }
            return out;
        }
        match (cur.parent(), cur.file_name()) {
            (Some(parent), Some(name)) => {
                rest.push(name.to_os_string());
                cur = parent;
            }
            _ => return path,
        }
    }
}

/// Lexical normalisation: drops `.` and folds `..`. `None` when a `..` would
/// climb above the root of an absolute path.
pub fn normalize(path: &Path) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    return None;
                }
            }
            other => out.push(other.as_os_str()),
        }
    }
    Some(out)
}

/// The inputs a run's policy is computed from.
#[derive(Debug, Clone)]
pub struct PolicyRequest<'a> {
    pub tools: &'a BTreeSet<GooseTool>,
    pub agents: &'a [String],
    pub cwd: &'a Path,
    pub allow_read: &'a [PathBuf],
    pub allow_write: &'a [PathBuf],
    pub allow_exec: &'a [PathBuf],
    /// Set by `--goose-nested`: write and exec roots come from the flags only.
    pub nested: bool,
}

/// Computes and checks a run's policy.
pub fn compute(req: &PolicyRequest, protected: &ProtectedDirs) -> Result<GoosePolicy> {
    let write = req.tools.contains(&GooseTool::Write);
    let ccc = req.tools.contains(&GooseTool::Ccc);
    if !write {
        if let Some(d) = req.allow_write.first() {
            bail!(
                "--allow-write {} given, but the agent's goose_tools lacks \"write\"",
                d.display()
            );
        }
    }
    if !ccc {
        if let Some(d) = req.allow_exec.first() {
            bail!(
                "--allow-exec {} given, but the agent's goose_tools lacks \"ccc\"",
                d.display()
            );
        }
    }

    let cwd = canonical_dir(req.cwd, "working directory", req.nested)?;
    let allow_read = canonical_dirs(req.allow_read, "--allow-read", req.nested)?;
    let allow_write = canonical_dirs(req.allow_write, "--allow-write", req.nested)?;
    let allow_exec = canonical_dirs(req.allow_exec, "--allow-exec", req.nested)?;

    let mut write_roots = Vec::new();
    if write {
        if !req.nested {
            push_unique(&mut write_roots, cwd.clone());
        }
        for d in allow_write {
            push_unique(&mut write_roots, d);
        }
        if write_roots.is_empty() {
            bail!(
                "the agent's goose_tools contains \"write\" but this nested run has no write \
                 root: pass allow_write"
            );
        }
    }

    let mut exec_roots = Vec::new();
    if ccc {
        if !req.nested && !write {
            push_unique(&mut exec_roots, cwd.clone());
        }
        for d in allow_exec {
            push_unique(&mut exec_roots, d);
        }
        if exec_roots.is_empty() {
            bail!(
                "the agent's goose_tools contains \"ccc\" but the run has no exec root: pass \
                 --allow-exec DIR (the working directory is a write root, so it cannot also \
                 be an exec root)"
            );
        }
    }

    let mut read_roots = vec![cwd];
    for d in allow_read.into_iter().chain(write_roots.iter().cloned()) {
        push_unique(&mut read_roots, d);
    }

    if write {
        protected.check_path_entries()?;
        for root in &write_roots {
            protected.check_write_root(root)?;
        }
    }
    check_separation(&write_roots, &exec_roots)?;

    Ok(GoosePolicy {
        tools: req.tools.clone(),
        agents: if ccc { req.agents.to_vec() } else { Vec::new() },
        read_roots,
        write_roots,
        exec_roots,
    })
}

/// The canonical form of the existing directory `path`. With `exact`, a path
/// that is not already canonical is refused rather than resolved, so a symlink
/// placed on it is never followed.
fn canonical_dir(path: &Path, what: &str, exact: bool) -> Result<PathBuf> {
    let c = std::fs::canonicalize(path)
        .with_context(|| format!("{what} {} does not exist", path.display()))?;
    if exact && c != path {
        bail!(
            "{what} {} is not canonical (it contains a symlink or a relative component and \
             resolves to {}); a nested run takes only canonical paths",
            path.display(),
            c.display()
        );
    }
    if !c.is_dir() {
        bail!("{what} {} is not a directory", path.display());
    }
    Ok(c)
}

fn canonical_dirs(paths: &[PathBuf], what: &str, exact: bool) -> Result<Vec<PathBuf>> {
    paths
        .iter()
        .map(|p| canonical_dir(p, what, exact))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;
    use std::collections::BTreeMap;

    fn tools(list: &[GooseTool]) -> BTreeSet<GooseTool> {
        list.iter().copied().collect()
    }

    #[test]
    fn only_a_ccc_only_policy_runs_without_a_box() {
        for (list, boxed) in [
            (&[GooseTool::Ccc][..], false),
            (&[GooseTool::Read], true),
            (&[GooseTool::Write], true),
            (&[GooseTool::Shell], true),
            (&[GooseTool::Ccc, GooseTool::Read], true),
        ] {
            let p = GoosePolicy {
                tools: tools(list),
                agents: vec![],
                read_roots: vec![],
                write_roots: vec![],
                exec_roots: vec![],
            };
            assert_eq!(p.needs_box(), boxed, "{list:?}");
        }
    }

    fn mkdir(base: &Path, rel: &str) -> PathBuf {
        let p = base.join(rel);
        std::fs::create_dir_all(&p).unwrap();
        std::fs::canonicalize(p).unwrap()
    }

    struct Req {
        tools: BTreeSet<GooseTool>,
        agents: Vec<String>,
        cwd: PathBuf,
        read: Vec<PathBuf>,
        write: Vec<PathBuf>,
        exec: Vec<PathBuf>,
        nested: bool,
    }

    impl Req {
        fn new(t: &[GooseTool], cwd: &Path) -> Req {
            Req {
                tools: tools(t),
                agents: vec!["explorer".into()],
                cwd: cwd.to_path_buf(),
                read: vec![],
                write: vec![],
                exec: vec![],
                nested: false,
            }
        }

        fn run(&self, protected: &ProtectedDirs) -> Result<GoosePolicy> {
            compute(
                &PolicyRequest {
                    tools: &self.tools,
                    agents: &self.agents,
                    cwd: &self.cwd,
                    allow_read: &self.read,
                    allow_write: &self.write,
                    allow_exec: &self.exec,
                    nested: self.nested,
                },
                protected,
            )
        }

        fn ok(&self) -> GoosePolicy {
            self.run(&ProtectedDirs::default()).unwrap()
        }

        fn err(&self) -> String {
            format!("{:#}", self.run(&ProtectedDirs::default()).unwrap_err())
        }
    }

    use GooseTool::{Ccc, Read, Shell, Write};

    #[test]
    fn read_only_policy_reads_cwd_and_allow_read() {
        let t = TempDir::new("gp-read");
        let cwd = mkdir(t.path(), "cwd");
        let extra = mkdir(t.path(), "extra");
        let mut r = Req::new(&[Read], &cwd);
        r.read = vec![extra.clone()];
        let p = r.ok();
        assert_eq!(p.read_roots, vec![cwd, extra]);
        assert!(p.write_roots.is_empty());
        assert!(p.exec_roots.is_empty());
        assert!(p.agents.is_empty());
    }

    #[test]
    fn write_policy_writes_cwd_and_allow_write_and_reads_them() {
        let t = TempDir::new("gp-write");
        let cwd = mkdir(t.path(), "cwd");
        let w = mkdir(t.path(), "w");
        let mut r = Req::new(&[Read, Write], &cwd);
        r.write = vec![w.clone()];
        let p = r.ok();
        assert_eq!(p.write_roots, vec![cwd.clone(), w.clone()]);
        assert_eq!(p.read_roots, vec![cwd, w]);
    }

    #[test]
    fn ccc_policy_execs_cwd_and_keeps_agents() {
        let t = TempDir::new("gp-ccc");
        let cwd = mkdir(t.path(), "cwd");
        let e = mkdir(t.path(), "e");
        let mut r = Req::new(&[Read, Ccc], &cwd);
        r.exec = vec![e.clone()];
        let p = r.ok();
        assert_eq!(p.exec_roots, vec![cwd.clone(), e]);
        assert_eq!(p.read_roots, vec![cwd]);
        assert_eq!(p.agents, vec!["explorer".to_string()]);
    }

    #[test]
    fn write_and_ccc_make_cwd_a_write_root_only() {
        let t = TempDir::new("gp-wc");
        let cwd = mkdir(t.path(), "cwd");
        let e = mkdir(t.path(), "e");
        git_init(&e);
        let mut r = Req::new(&[Write, Ccc], &cwd);
        assert!(r.err().contains("--allow-exec"), "{}", r.err());
        r.exec = vec![e.clone()];
        let p = r.ok();
        assert_eq!(p.write_roots, vec![cwd]);
        assert_eq!(p.exec_roots, vec![e]);
    }

    #[test]
    fn allow_flags_without_the_matching_tool_are_refused() {
        let t = TempDir::new("gp-flags");
        let cwd = mkdir(t.path(), "cwd");
        let d = mkdir(t.path(), "d");
        let mut r = Req::new(&[Read], &cwd);
        r.write = vec![d.clone()];
        assert!(r.err().contains("lacks \"write\""), "{}", r.err());
        let mut r = Req::new(&[Read], &cwd);
        r.exec = vec![d];
        assert!(r.err().contains("lacks \"ccc\""), "{}", r.err());
    }

    #[test]
    fn missing_or_non_directory_roots_are_refused() {
        let t = TempDir::new("gp-missing");
        let cwd = mkdir(t.path(), "cwd");
        let file = t.write_file("f", "x", 0o644);
        let mut r = Req::new(&[Read], &cwd);
        r.read = vec![t.path().join("nope")];
        assert!(r.err().contains("does not exist"), "{}", r.err());
        r.read = vec![file];
        assert!(r.err().contains("not a directory"), "{}", r.err());
    }

    #[test]
    fn nested_runs_take_write_and_exec_roots_from_flags_only() {
        let t = TempDir::new("gp-nested");
        let cwd = mkdir(t.path(), "cwd");
        let w = mkdir(t.path(), "w");
        let e = mkdir(t.path(), "e");

        let mut r = Req::new(&[Read, Write], &cwd);
        r.nested = true;
        assert!(r.err().contains("allow_write"), "{}", r.err());
        r.write = vec![w.clone()];
        let p = r.ok();
        assert_eq!(p.write_roots, vec![w.clone()]);
        assert_eq!(p.read_roots, vec![cwd.clone(), w.clone()]);

        let mut r = Req::new(&[Read, Ccc], &cwd);
        r.nested = true;
        assert!(r.err().contains("--allow-exec"), "{}", r.err());
        r.exec = vec![e.clone()];
        let p = r.ok();
        assert_eq!(p.exec_roots, vec![e.clone()]);
        assert_eq!(p.read_roots, vec![cwd.clone()]);

        // Separation still applies to a hand-passed nested flag.
        let inner = mkdir(&e, "inner");
        let mut r = Req::new(&[Write, Ccc], &cwd);
        r.nested = true;
        r.write = vec![inner];
        r.exec = vec![e];
        assert!(r.err().contains("overlaps"), "{}", r.err());
    }

    #[test]
    fn nested_runs_refuse_non_canonical_paths_that_top_level_runs_resolve() {
        let t = TempDir::new("gp-nested-canon");
        let cwd = mkdir(t.path(), "cwd");
        let real = mkdir(t.path(), "real");
        let link = t.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let dotted = cwd.join("..").join("real");

        for path in [link.clone(), dotted] {
            let mut r = Req::new(&[Read, Write], &cwd);
            r.write = vec![path.clone()];
            assert_eq!(r.ok().write_roots, vec![cwd.clone(), real.clone()]);
            r.nested = true;
            let err = r.err();
            assert!(err.contains("not canonical"), "{err}");
            assert!(err.contains("--allow-write"), "{err}");
        }

        let mut r = Req::new(&[Read], &link);
        r.nested = true;
        let err = r.err();
        assert!(err.contains("working directory"), "{err}");
        assert!(err.contains("not canonical"), "{err}");
    }

    #[test]
    fn a_symlinked_root_is_recorded_canonical() {
        let t = TempDir::new("gp-link");
        let real = mkdir(t.path(), "real");
        let link = t.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let r = Req::new(&[Read], &link);
        assert_eq!(r.ok().read_roots, vec![real]);
    }

    #[test]
    fn hex_round_trips_and_rejects_bad_tokens() {
        let t = TempDir::new("gp-hex");
        let cwd = mkdir(t.path(), "cwd");
        let e = mkdir(t.path(), "e");
        let mut r = Req::new(&[Read, Ccc], &cwd);
        r.exec = vec![e];
        let p = r.ok();
        let hex = p.to_hex();
        assert!(hex.bytes().all(|b| b.is_ascii_hexdigit()));
        assert_eq!(GoosePolicy::from_hex(&hex).unwrap(), p);

        assert!(GoosePolicy::from_hex("abc").is_err());
        assert!(GoosePolicy::from_hex("zz").is_err());
        let mut relative = p.clone();
        relative.read_roots = vec![PathBuf::from("rel")];
        assert!(GoosePolicy::from_hex(&relative.to_hex()).is_err());
        let link = t.path().join("link");
        std::os::unix::fs::symlink(&cwd, &link).unwrap();
        let mut linked = p;
        linked.read_roots = vec![link];
        let err = format!("{:#}", GoosePolicy::from_hex(&linked.to_hex()).unwrap_err());
        assert!(err.contains("not canonical"), "{err}");
    }

    #[test]
    fn goose_tool_parses_its_names() {
        for t in [Read, Write, Shell, Ccc] {
            assert_eq!(GooseTool::parse(t.as_str()).unwrap(), t);
        }
        assert!(GooseTool::parse("developer").is_err());
    }

    // --- write/exec separation ---

    fn git_init(dir: &Path) {
        std::fs::create_dir_all(dir.join(".git")).unwrap();
    }

    #[test]
    fn write_root_equal_inside_or_containing_an_exec_root_is_refused() {
        let t = TempDir::new("gp-sep");
        let e = mkdir(t.path(), "e");
        git_init(&e);
        let inside = mkdir(&e, "sub");
        let w = mkdir(t.path(), "w");
        assert!(check_separation(std::slice::from_ref(&e), std::slice::from_ref(&e)).is_err());
        assert!(check_separation(&[inside], std::slice::from_ref(&e)).is_err());
        let base = std::fs::canonicalize(t.path()).unwrap();
        assert!(check_separation(&[base], std::slice::from_ref(&e)).is_err());
        assert!(check_separation(&[w], &[e]).is_ok());
    }

    #[test]
    fn write_root_at_the_git_top_above_an_exec_subdirectory_is_refused() {
        let t = TempDir::new("gp-top");
        let repo = mkdir(t.path(), "repo");
        git_init(&repo);
        let e = mkdir(&repo, "src/deep");
        let w = mkdir(&repo, "docs");
        let err = format!("{:#}", check_separation(&[w], &[e]).unwrap_err());
        assert!(err.contains("project top"), "{err}");
        assert!(err.contains(".git"), "{err}");
    }

    /// A main checkout at `main/` and a worktree at `wt/` whose `.git` file
    /// names `main/.git/worktrees/wt`, whose `commondir` is `../..`.
    fn worktree(base: &Path, relative: bool) -> (PathBuf, PathBuf, PathBuf) {
        let main = mkdir(base, "main");
        git_init(&main);
        let gd = mkdir(&main, ".git/worktrees/wt");
        std::fs::write(gd.join("commondir"), "../..\n").unwrap();
        let wt = mkdir(base, "wt");
        let named = if relative {
            "../main/.git/worktrees/wt".to_string()
        } else {
            gd.display().to_string()
        };
        std::fs::write(wt.join(".git"), format!("gitdir: {named}\n")).unwrap();
        (main, wt, gd)
    }

    #[test]
    fn worktree_surface_resolves_relative_gitdir_and_commondir() {
        let t = TempDir::new("gp-wt");
        let (main, wt, gd) = worktree(t.path(), true);
        let s = ExecSurface::of(&wt).unwrap();
        assert_eq!(s.top, wt);
        assert_eq!(s.git_dir, Some(gd));
        assert_eq!(s.common_dir, Some(main.join(".git")));
    }

    #[test]
    fn a_git_dir_without_commondir_is_its_own_common_dir() {
        let t = TempDir::new("gp-sm");
        let gd = mkdir(t.path(), "store/modules/m");
        let sub = mkdir(t.path(), "sub");
        std::fs::write(sub.join(".git"), format!("gitdir: {}\n", gd.display())).unwrap();
        let s = ExecSurface::of(&sub).unwrap();
        assert_eq!(s.git_dir, Some(gd.clone()));
        assert_eq!(s.common_dir, Some(gd));
    }

    #[test]
    fn a_symlinked_git_marker_names_its_target_as_the_git_dir() {
        let t = TempDir::new("gp-gitlink");
        let real = mkdir(t.path(), "store/real.git");
        let e = mkdir(t.path(), "e");
        std::os::unix::fs::symlink("../store/real.git", e.join(".git")).unwrap();
        let s = ExecSurface::of(&e).unwrap();
        assert_eq!(s.git_dir, Some(real.clone()));
        assert_eq!(s.common_dir, Some(real));
        let store = mkdir(t.path(), "store");
        let err = format!("{:#}", check_separation(&[store], &[e]).unwrap_err());
        assert!(err.contains("git directory"), "{err}");
    }

    #[test]
    fn malformed_git_markers_are_refused() {
        let t = TempDir::new("gp-gitbad");
        let no_line = mkdir(t.path(), "no-line");
        std::fs::write(no_line.join(".git"), "nothing here\n").unwrap();
        let missing = mkdir(t.path(), "missing");
        std::fs::write(missing.join(".git"), "gitdir: ../nowhere\n").unwrap();
        let empty_common = mkdir(t.path(), "empty-common");
        let gd = mkdir(t.path(), "gd");
        std::fs::write(gd.join("commondir"), "\n").unwrap();
        std::fs::write(
            empty_common.join(".git"),
            format!("gitdir: {}\n", gd.display()),
        )
        .unwrap();
        let dangling = mkdir(t.path(), "dangling");
        std::os::unix::fs::symlink("../absent.git", dangling.join(".git")).unwrap();
        for (dir, message) in [
            (&no_line, "has no gitdir: line"),
            (&missing, "resolving the git directory"),
            (&empty_common, "is empty"),
            (&dangling, "resolving the git directory link"),
        ] {
            let err = format!("{:#}", ExecSurface::of(dir).unwrap_err());
            assert!(err.contains(message), "{}: {err}", dir.display());
        }
    }

    #[test]
    fn write_root_holding_a_worktree_git_dir_or_common_dir_is_refused() {
        let t = TempDir::new("gp-wtsep");
        let (main, wt, gd) = worktree(t.path(), false);
        let err = format!(
            "{:#}",
            check_separation(&[gd], std::slice::from_ref(&wt)).unwrap_err()
        );
        assert!(err.contains("git directory"), "{err}");
        let err = format!(
            "{:#}",
            check_separation(&[main.join(".git")], std::slice::from_ref(&wt)).unwrap_err()
        );
        assert!(err.contains("git"), "{err}");
        let other = mkdir(t.path(), "other");
        assert!(check_separation(&[other], &[wt]).is_ok());
    }

    #[test]
    fn surface_containment() {
        let t = TempDir::new("gp-contain");
        let (main, wt, _) = worktree(t.path(), false);
        let main_surface = ExecSurface::of(&main).unwrap();
        assert!(main_surface.lies_inside(std::slice::from_ref(&main)));
        let wt_surface = ExecSurface::of(&wt).unwrap();
        assert!(!wt_surface.lies_inside(std::slice::from_ref(&wt)));
        assert!(wt_surface.lies_inside(&[wt.clone(), main.clone()]));
        let policy = GoosePolicy {
            tools: tools(&[Ccc]),
            agents: vec![],
            read_roots: vec![],
            write_roots: vec![],
            exec_roots: vec![wt.clone()],
        };
        let union = policy.exec_surfaces().unwrap();
        assert!(union.contains(&wt) && union.contains(&main.join(".git")));
    }

    #[test]
    fn disjoint_write_and_exec_trees_are_accepted_end_to_end() {
        let t = TempDir::new("gp-disjoint");
        let w = mkdir(t.path(), "w");
        git_init(&w);
        let e = mkdir(t.path(), "e");
        git_init(&e);
        let mut r = Req::new(&[Read, Write, Ccc], &w);
        r.exec = vec![e.clone()];
        let p = r.ok();
        assert_eq!(p.write_roots, vec![w]);
        assert_eq!(p.exec_roots, vec![e]);
    }

    #[test]
    fn worktree_granted_by_allow_exec_is_checked_against_write_roots() {
        let t = TempDir::new("gp-grant");
        let (main, wt, _) = worktree(t.path(), false);
        let cwd = mkdir(t.path(), "cwd");
        // The worktree's git directory lies under the write root: refused.
        let mut r = Req::new(&[Write, Ccc], &cwd);
        r.write = vec![main];
        r.exec = vec![wt.clone()];
        assert!(r.err().contains("overlaps"), "{}", r.err());
        let mut r = Req::new(&[Write, Ccc], &cwd);
        r.exec = vec![wt];
        r.ok();
    }

    // --- protected directories ---

    fn default_config() -> Config {
        let paths = crate::config::ConfigPaths::in_env(&env(&[("HOME", "/home/u")]));
        Config::from_texts(None, None, &paths).unwrap()
    }

    fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    struct Fixture {
        _t: TempDir,
        home: PathBuf,
        outside: PathBuf,
        protected: ProtectedDirs,
        goose_root: PathBuf,
    }

    fn fixture(extra: &[(&str, &str)], path: &str) -> Fixture {
        let t = TempDir::new("gp-prot");
        let home = mkdir(t.path(), "home");
        let outside = mkdir(t.path(), "outside");
        let bin = mkdir(t.path(), "bin");
        let goose_root = mkdir(t.path(), "gooseroot");
        let path = if path.is_empty() {
            bin.display().to_string()
        } else {
            path.replace("{bin}", &bin.display().to_string())
        };
        let mut pairs = vec![("HOME", home.to_str().unwrap()), ("PATH", path.as_str())];
        pairs.extend_from_slice(extra);
        let e = env(&pairs);
        let cfg = default_config();
        let self_exe = mkdir(t.path(), "cccbin").join("ccc");
        let protected = ProtectedDirs::gather_in(&e, &cfg, &goose_root, &self_exe);
        Fixture {
            _t: t,
            home,
            outside,
            protected,
            goose_root,
        }
    }

    fn refused(f: &Fixture, root: &Path) -> bool {
        f.protected.check_write_root(root).is_err()
    }

    #[test]
    fn ccc_config_agents_state_and_goose_root_are_protected() {
        let t = TempDir::new("gp-xdg");
        let xdg_config = mkdir(t.path(), "xdg-config");
        let xdg_state = mkdir(t.path(), "xdg-state");
        let f = fixture(
            &[
                ("XDG_CONFIG_HOME", xdg_config.to_str().unwrap()),
                ("XDG_STATE_HOME", xdg_state.to_str().unwrap()),
            ],
            "",
        );
        let config = mkdir(&xdg_config, "ccc");
        let agents = mkdir(&xdg_config, "ccc/agents");
        let state = mkdir(&xdg_state, "ccc");
        for dir in [&config, &agents, &state, &f.goose_root] {
            assert!(refused(&f, dir), "{}", dir.display());
            assert!(refused(&f, &mkdir(dir, "inner")), "{}", dir.display());
            assert!(refused(&f, dir.parent().unwrap()), "{}", dir.display());
        }
    }

    #[test]
    fn home_rule() {
        let f = fixture(&[], "");
        assert!(refused(&f, &f.home));
        assert!(refused(&f, f.home.parent().unwrap()));
        for dot in [".claude", ".codex", ".config"] {
            assert!(refused(&f, &mkdir(&f.home, &format!("{dot}/x"))), "{dot}");
        }
        assert!(!refused(&f, &mkdir(&f.home, "src/x")));
        assert!(!refused(&f, &f.outside));
    }

    #[test]
    fn home_rule_alone_refuses_home_and_its_ancestors() {
        let t = TempDir::new("gp-home-alone");
        let home = mkdir(t.path(), "users/u");
        let elsewhere = mkdir(t.path(), "elsewhere");
        let mut cfg = default_config();
        cfg.claude.token_dir = elsewhere.join("tokens").display().to_string();
        cfg.claude.profiles_dir = elsewhere.join("profiles").display().to_string();
        let e = env(&[
            ("HOME", home.to_str().unwrap()),
            (
                "XDG_CONFIG_HOME",
                elsewhere.join("config").to_str().unwrap(),
            ),
            ("XDG_STATE_HOME", elsewhere.join("state").to_str().unwrap()),
            ("PATH", "/usr/bin"),
        ]);
        let p = ProtectedDirs::gather_in(&e, &cfg, &elsewhere.join("goose"), Path::new("/x/ccc"));
        assert!(
            p.dirs
                .iter()
                .all(|(d, _)| !d.starts_with(&home) && !home.starts_with(d)),
            "{:?}",
            p.dirs
        );
        for root in [&home, home.parent().unwrap()] {
            let err = format!("{:#}", p.check_write_root(root).unwrap_err());
            assert!(err.contains("is $HOME"), "{err}");
        }
        assert!(p.check_write_root(&mkdir(&home, "src")).is_ok());
    }

    #[test]
    fn a_missing_protected_dir_behind_a_symlink_is_compared_canonically() {
        let t = TempDir::new("gp-prot-link");
        let real = mkdir(t.path(), "real-state");
        let link = t.path().join("link-state");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let f = fixture(&[("XDG_STATE_HOME", link.to_str().unwrap())], "");
        let state = mkdir(&real, "ccc");
        assert!(refused(&f, &state));
        assert!(refused(&f, &real));
        assert!(refused(&f, &mkdir(&state, "inner")));
    }

    #[test]
    fn harness_homes_outside_home_are_protected() {
        let t = TempDir::new("gp-hh");
        let cc = mkdir(t.path(), "claude-config");
        let cx = mkdir(t.path(), "codex-home");
        let f = fixture(
            &[
                ("CLAUDE_CONFIG_DIR", cc.to_str().unwrap()),
                ("CODEX_HOME", cx.to_str().unwrap()),
            ],
            "",
        );
        assert!(refused(&f, &cc));
        assert!(refused(&f, &mkdir(&cx, "sub")));

        let tokens = mkdir(t.path(), "tokens");
        let profiles = mkdir(t.path(), "profiles");
        let mut cfg = default_config();
        cfg.claude.token_dir = tokens.display().to_string();
        cfg.claude.profiles_dir = profiles.display().to_string();
        let e = env(&[("HOME", f.home.to_str().unwrap()), ("PATH", "/usr/bin")]);
        let p = ProtectedDirs::gather_in(&e, &cfg, &f.goose_root, Path::new("/x/ccc"));
        assert!(p.check_write_root(&tokens).is_err());
        assert!(p.check_write_root(&mkdir(&profiles, "p")).is_err());
    }

    #[test]
    fn executable_directories_are_protected() {
        let t = TempDir::new("gp-exe");
        let claude_dir = mkdir(t.path(), "claudebin");
        let claude_bin = claude_dir.join("claude");
        let f = fixture(&[("CCC_CLAUDE_BIN", claude_bin.to_str().unwrap())], "");
        let path_dir = f.protected.dirs.iter().find(|(_, w)| w == "PATH entry");
        let path_dir = path_dir.unwrap().0.clone();
        assert!(refused(&f, &path_dir));
        assert!(refused(&f, &mkdir(&path_dir, "inner")));
        assert!(refused(&f, path_dir.parent().unwrap()));
        let exe_dir = f
            .protected
            .dirs
            .iter()
            .find(|(_, w)| w == "directory of the ccc binary")
            .unwrap()
            .0
            .clone();
        assert!(refused(&f, &exe_dir));
        assert!(refused(&f, &claude_dir));
        assert!(refused(&f, &mkdir(&claude_dir, "x")));

        let bwrap_dir = mkdir(t.path(), "opt/tools/bin");
        let mut cfg = default_config();
        cfg.goose.bwrap = bwrap_dir.join("bwrap").display().to_string();
        let e = env(&[("HOME", f.home.to_str().unwrap()), ("PATH", "/usr/bin")]);
        let p = ProtectedDirs::gather_in(&e, &cfg, &f.goose_root, Path::new("/x/ccc"));
        let err = format!("{:#}", p.check_write_root(&bwrap_dir).unwrap_err());
        assert!(err.contains("directory of [goose] bwrap"), "{err}");
        assert!(p.check_write_root(&mkdir(t.path(), "opt")).is_err());
        assert!(p.check_write_root(&mkdir(&bwrap_dir, "x")).is_err());
    }

    #[test]
    fn relative_or_empty_path_entries_refuse_write_capable_runs_only() {
        for path in ["{bin}:relative/bin", "{bin}::/usr/bin"] {
            let f = fixture(&[], path);
            let cwd = mkdir(&f.outside, "cwd");
            let mut r = Req::new(&[Read], &cwd);
            r.run(&f.protected).unwrap();
            r.tools = tools(&[Read, Write]);
            let err = format!("{:#}", r.run(&f.protected).unwrap_err());
            assert!(err.contains("PATH entry"), "{path}: {err}");
        }
    }

    #[test]
    fn compute_refuses_a_protected_write_root() {
        let f = fixture(&[], "");
        let cwd = mkdir(&f.home, ".claude/project");
        let r = Req::new(&[Write], &cwd);
        let err = format!("{:#}", r.run(&f.protected).unwrap_err());
        assert!(err.contains("dot-named"), "{err}");
        let ok = mkdir(&f.home, "src/project");
        Req::new(&[Write], &ok).run(&f.protected).unwrap();
    }

    #[test]
    fn normalize_folds_dots_and_refuses_climbing_above_root() {
        assert_eq!(
            normalize(Path::new("/a/./b/../c")),
            Some(PathBuf::from("/a/c"))
        );
        assert_eq!(normalize(Path::new("/..")), None);
        assert_eq!(normalize(Path::new("/a/../..")), None);
    }
}
