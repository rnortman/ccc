//! The `box-inner` role: file and shell tools, run inside the bubblewrap box.
//! The box's mount namespace holds only the policy's roots, so these tools
//! check no paths: the kernel answers every access, and an error reply carries
//! the OS error.

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use anyhow::{bail, Result};
use regex::Regex;
use serde_json::{json, Value};

use super::{
    arg_opt_bool, arg_opt_str, arg_opt_u64, arg_str, ToolDef, ToolReply, Toolset, RESULT_CAP,
};
use crate::goose_policy::{GoosePolicy, GooseTool};

const READ_DEFAULT_LINES: u64 = 2000;
const LINE_CAP_CHARS: usize = 2000;
const GLOB_CAP: usize = 1000;
const GREP_CAP: usize = 500;
const WRITE_CAP: usize = 10 * 1024 * 1024;
/// Bytes inspected for a NUL when deciding that a file is binary.
const BINARY_PROBE: usize = 8192;
/// The longest line `grep` reads; a file holding a longer one is skipped.
const GREP_LINE_BYTES: usize = 1024 * 1024;
/// The bytes of one line `read_file` keeps: enough for `LINE_CAP_CHARS`
/// characters of any width.
const READ_LINE_BYTES: usize = LINE_CAP_CHARS * 4;

/// Closes every descriptor above 2. bwrap passes its inherited descriptors
/// through, and a bind-source directory descriptor would let a program walk
/// the host tree outside the mount namespace.
pub fn close_inherited_fds() {
    sweep_fds_from(3, Sweep::Close);
}

/// What a descriptor sweep does to each descriptor it reaches.
#[derive(Debug, Clone, Copy)]
pub(super) enum Sweep {
    Close,
    Cloexec,
}

/// Applies `op` to every descriptor numbered `first` or above.
pub(super) fn sweep_fds_from(first: u32, op: Sweep) {
    #[cfg(target_os = "linux")]
    {
        const CLOSE_RANGE_CLOEXEC: u32 = 1 << 2;
        let flags = match op {
            Sweep::Close => 0,
            Sweep::Cloexec => CLOSE_RANGE_CLOEXEC,
        };
        // SAFETY: close_range only closes descriptors or sets their
        // close-on-exec flag; the caller owns none at or above `first`.
        if unsafe { libc::syscall(libc::SYS_close_range, first, u32::MAX, flags) } == 0 {
            return;
        }
    }
    sweep_listed_fds_from(first, op);
}

/// [`sweep_fds_from`] by listing `/proc/self/fd`, where `close_range` is
/// unsupported.
fn sweep_listed_fds_from(first: u32, op: Sweep) {
    let fds: Vec<i32> = match std::fs::read_dir("/proc/self/fd") {
        Ok(entries) => entries
            .filter_map(|e| e.ok()?.file_name().to_str()?.parse().ok())
            .filter(|fd: &i32| u32::try_from(*fd).is_ok_and(|fd| fd >= first))
            .collect(),
        Err(_) => return,
    };
    for fd in fds {
        // SAFETY: closing a descriptor or setting its flag by number; the
        // directory handle used to list them is already dropped, so a stale
        // number gives EBADF.
        unsafe {
            match op {
                Sweep::Close => {
                    libc::close(fd);
                }
                Sweep::Cloexec => {
                    libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC);
                }
            }
        }
    }
}

/// Refuses to serve unless PID 1 is bwrap: these tools confine nothing
/// themselves, so they must not run on the host.
pub fn check_in_box() -> Result<()> {
    let comm = std::fs::read_to_string("/proc/1/comm").unwrap_or_default();
    let comm = comm.trim();
    if comm != "bwrap" {
        bail!(
            "goose-tools --role box-inner runs only inside the bubblewrap box (PID 1 is {comm:?})"
        );
    }
    Ok(())
}

/// The read, write and shell tools a policy grants.
pub struct BoxTools {
    policy: GoosePolicy,
    cwd: PathBuf,
    tool_wait_s: u64,
    running: Mutex<Running>,
    /// Held across each `write_file` and `edit_file`, so concurrent calls on
    /// one file neither lose an edit nor interleave their bytes.
    writes: Mutex<()>,
}

/// The `shell` calls in flight, and whether stdin has closed.
#[derive(Default)]
struct Running {
    closing: bool,
    groups: BTreeSet<u32>,
}

impl BoxTools {
    /// Relative paths resolve against `cwd`, where `shell` also runs;
    /// `tool_wait_s` caps every `shell` call's timeout.
    pub fn new(policy: GoosePolicy, cwd: PathBuf, tool_wait_s: u64) -> BoxTools {
        BoxTools {
            policy,
            cwd,
            tool_wait_s,
            running: Mutex::new(Running::default()),
            writes: Mutex::new(()),
        }
    }

    fn granted(&self, group: GooseTool) -> bool {
        self.policy.tools.contains(&group)
    }

    fn resolve(&self, path: &str) -> Result<PathBuf, ToolReply> {
        if path.is_empty() {
            return Err(ToolReply::error("path must not be empty"));
        }
        Ok(self.cwd.join(path))
    }

    fn writable_text(&self) -> String {
        if self.policy.write_roots.is_empty() {
            "nothing outside /tmp".to_string()
        } else {
            let roots: Vec<String> = self
                .policy
                .write_roots
                .iter()
                .map(|r| r.display().to_string())
                .collect();
            roots.join(", ")
        }
    }
}

impl Toolset for BoxTools {
    fn list(&self) -> Vec<ToolDef> {
        let cwd = self.cwd.display();
        let mut tools = Vec::new();
        if self.granted(GooseTool::Read) {
            tools.push(ToolDef {
                name: "read_file",
                description: format!(
                    "Read a UTF-8 text file with line numbers. `offset` is the first line (1-based), `limit` the line count (default {READ_DEFAULT_LINES}). Relative paths resolve against {cwd}."
                ),
                schema: schema(&[("path", "string", true), ("offset", "integer", false), ("limit", "integer", false)]),
            });
            tools.push(ToolDef {
                name: "list_dir",
                description: format!("List a directory's entries with type and size. Relative paths resolve against {cwd}."),
                schema: schema(&[("path", "string", true)]),
            });
            tools.push(ToolDef {
                name: "glob",
                description: format!(
                    "Find paths under `path` (default {cwd}) matching `pattern`: `**` is any number of directories, `*` and `?` stay within one name. Case-sensitive. At most {GLOB_CAP} results."
                ),
                schema: schema(&[("pattern", "string", true), ("path", "string", false)]),
            });
            tools.push(ToolDef {
                name: "grep",
                description: format!(
                    "Search files under `path` (default {cwd}) for a regex `pattern`, optionally only files matching `glob`. Prints path:line:text, at most {GREP_CAP} matches; binary files are skipped."
                ),
                schema: schema(&[("pattern", "string", true), ("path", "string", false), ("glob", "string", false)]),
            });
        }
        if self.granted(GooseTool::Write) {
            let writable = self.writable_text();
            tools.push(ToolDef {
                name: "write_file",
                description: format!(
                    "Create or overwrite a file with `content`, creating missing directories. Writes succeed only under: {writable}."
                ),
                schema: schema(&[("path", "string", true), ("content", "string", true)]),
            });
            tools.push(ToolDef {
                name: "edit_file",
                description: format!(
                    "Replace `old` with `new` in a file. `old` must occur exactly once unless `replace_all` is true. Writes succeed only under: {writable}."
                ),
                schema: schema(&[("path", "string", true), ("old", "string", true), ("new", "string", true), ("replace_all", "boolean", false)]),
            });
        }
        if self.granted(GooseTool::Shell) {
            let wait = self.tool_wait_s;
            tools.push(ToolDef {
                name: "shell",
                description: format!(
                    "Run `command` with /bin/sh in {cwd}, with no network; stdout and stderr are returned together. `timeout_s` defaults to and is capped at {wait}; on timeout the command and its background jobs are killed. Start longer work as `cmd > log 2>&1 &` and read `log` later: output of a background job that is not redirected is lost. Writes succeed only under: {}.",
                    self.writable_text()
                ),
                schema: schema(&[("command", "string", true), ("timeout_s", "integer", false)]),
            });
        }
        tools
    }

    fn call(&self, name: &str, args: &Value) -> ToolReply {
        let group = match name {
            "read_file" | "list_dir" | "glob" | "grep" => GooseTool::Read,
            "write_file" | "edit_file" => GooseTool::Write,
            "shell" => GooseTool::Shell,
            _ => return ToolReply::error(format!("unknown tool {name}")),
        };
        if !self.granted(group) {
            return ToolReply::error(format!("tool {name} is not offered to this agent"));
        }
        let reply = match name {
            "read_file" => self.read_file(args),
            "list_dir" => self.list_dir(args),
            "glob" => self.glob(args),
            "grep" => self.grep(args),
            "write_file" => self.write_file(args),
            "edit_file" => self.edit_file(args),
            _ => self.shell(args),
        };
        reply.unwrap_or_else(|e| e)
    }

    fn shutdown(&self) {
        let mut running = self.lock_running();
        running.closing = true;
        for pgid in &running.groups {
            kill_group(*pgid);
        }
    }
}

fn schema(params: &[(&str, &str, bool)]) -> Value {
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();
    for (name, ty, req) in params {
        properties.insert((*name).to_string(), json!({"type": ty}));
        if *req {
            required.push(*name);
        }
    }
    json!({"type": "object", "properties": properties, "required": required})
}

fn io_error(path: &Path, e: std::io::Error) -> ToolReply {
    ToolReply::error(format!("{}: {e}", path.display()))
}

fn cap_line(line: &str) -> String {
    match line.char_indices().nth(LINE_CAP_CHARS) {
        Some((at, _)) => format!("{}…[line truncated]", &line[..at]),
        None => line.to_string(),
    }
}

/// Reads the next line of `reader` into `line`, keeping at most `cap` bytes
/// and discarding the rest of a longer line up to and including its newline.
/// `None` at end of input; otherwise whether the line was cut.
fn read_line_capped(
    reader: &mut impl BufRead,
    line: &mut Vec<u8>,
    cap: usize,
) -> std::io::Result<Option<bool>> {
    use std::io::Read;
    line.clear();
    let n = reader.by_ref().take(cap as u64).read_until(b'\n', line)?;
    if n == 0 {
        return Ok(None);
    }
    if n < cap || line.last() == Some(&b'\n') {
        return Ok(Some(false));
    }
    let mut cut = false;
    loop {
        let buf = match reader.fill_buf() {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        if buf.is_empty() {
            return Ok(Some(cut));
        }
        match buf.iter().position(|b| *b == b'\n') {
            Some(i) => {
                cut |= i > 0;
                reader.consume(i + 1);
                return Ok(Some(cut));
            }
            None => {
                cut = true;
                let len = buf.len();
                reader.consume(len);
            }
        }
    }
}

fn looks_binary(head: &[u8]) -> bool {
    head[..head.len().min(BINARY_PROBE)].contains(&0)
}

/// Opens `path` for reading only if it is a regular file. The open does not
/// block, so a fifo swapped in after a type check cannot hang the call.
fn open_regular(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK | libc::O_NOCTTY)
        .open(path)?;
    if !file.metadata()?.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "not a regular file",
        ));
    }
    Ok(file)
}

/// Replaces the contents of `path`, creating it when missing, only if it is a
/// regular file. Neither the open nor the type check blocks on a fifo.
fn write_regular(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .custom_flags(libc::O_NONBLOCK | libc::O_NOCTTY)
        .open(path)?;
    if !file.metadata()?.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "not a regular file",
        ));
    }
    file.set_len(0)?;
    file.write_all(contents)
}

/// The `(line number, capped line)` pairs of `file` matching `regex`. A file
/// that is not regular, looks binary, holds a line that is not UTF-8 or one
/// longer than `GREP_LINE_BYTES` is skipped whole: the error says why.
fn grep_file(file: &Path, regex: &Regex) -> std::io::Result<Vec<(u64, String)>> {
    use std::io::Read;
    let skip = |why: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, why.to_string());
    let mut reader = BufReader::new(open_regular(file)?);
    if looks_binary(reader.fill_buf()?) {
        return Err(skip("binary file"));
    }
    let mut found = Vec::new();
    let mut number = 0u64;
    let mut line = Vec::new();
    loop {
        line.clear();
        let n = (&mut reader)
            .take(GREP_LINE_BYTES as u64 + 1)
            .read_until(b'\n', &mut line)?;
        if n == 0 {
            return Ok(found);
        }
        if n > GREP_LINE_BYTES {
            return Err(skip("a line is over the 1 MiB grep limit"));
        }
        number += 1;
        let text = std::str::from_utf8(&line).map_err(|_| skip("a line is not UTF-8"))?;
        let text = text.strip_suffix('\n').unwrap_or(text);
        let text = text.strip_suffix('\r').unwrap_or(text);
        if regex.is_match(text) {
            found.push((number, cap_line(text)));
            if found.len() > GREP_CAP {
                return Ok(found);
            }
        }
    }
}

/// `grep`'s reply: the match lines, or `[no matches]`, then a note for a
/// cap reached and for files the walk could not search.
fn grep_reply(out: Vec<String>, capped: bool, skipped: usize) -> ToolReply {
    let mut text = if out.is_empty() {
        "[no matches]".to_string()
    } else {
        out.join("\n")
    };
    if capped {
        text.push_str(&format!("\n[stopped at {GREP_CAP} matches]"));
    }
    if skipped > 0 {
        text.push_str(&format!(
            "\n[{skipped} file{} not searched: binary, not UTF-8, unreadable or with a line over {GREP_LINE_BYTES} bytes]",
            if skipped == 1 { "" } else { "s" }
        ));
    }
    ToolReply::ok(text)
}

impl BoxTools {
    fn read_file(&self, args: &Value) -> Result<ToolReply, ToolReply> {
        let path = self.resolve(arg_str(args, "path")?)?;
        let offset = arg_opt_u64(args, "offset")?.unwrap_or(1).max(1);
        let limit = arg_opt_u64(args, "limit")?.unwrap_or(READ_DEFAULT_LINES);
        let file = open_regular(&path).map_err(|e| io_error(&path, e))?;
        let mut reader = BufReader::new(file);
        let head = reader.fill_buf().map_err(|e| io_error(&path, e))?;
        if looks_binary(head) {
            return Err(ToolReply::error(format!(
                "{}: binary file, not shown",
                path.display()
            )));
        }
        let mut out = String::new();
        let mut number = 0u64;
        let mut line = Vec::new();
        let end = offset.saturating_add(limit);
        while let Some(cut) = read_line_capped(&mut reader, &mut line, READ_LINE_BYTES)
            .map_err(|e| io_error(&path, e))?
        {
            number += 1;
            if number < offset {
                continue;
            }
            if number >= end {
                out.push_str(&format!("[more lines follow; next offset {number}]\n"));
                break;
            }
            let text = match std::str::from_utf8(&line) {
                Ok(t) => t,
                // A cut can split the last character; keep what precedes it.
                Err(e) if cut && e.error_len().is_none() => {
                    std::str::from_utf8(&line[..e.valid_up_to()]).unwrap_or_default()
                }
                Err(_) => {
                    return Err(ToolReply::error(format!(
                        "{}: line {number} is not UTF-8 text",
                        path.display()
                    )))
                }
            };
            let text = text.strip_suffix('\n').unwrap_or(text);
            let text = text.strip_suffix('\r').unwrap_or(text);
            let mut shown = cap_line(text);
            if cut && !shown.ends_with("[line truncated]") {
                shown.push_str("…[line truncated]");
            }
            out.push_str(&format!("{number:>6}\t{shown}\n"));
            if out.len() > RESULT_CAP {
                break;
            }
        }
        if out.is_empty() {
            out = format!("[no lines at offset {offset}; the file has {number}]");
        }
        Ok(ToolReply::ok(out))
    }

    fn list_dir(&self, args: &Value) -> Result<ToolReply, ToolReply> {
        let path = self.resolve(arg_str(args, "path")?)?;
        let entries = std::fs::read_dir(&path).map_err(|e| io_error(&path, e))?;
        let mut lines = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| io_error(&path, e))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let line = match entry.metadata() {
                Ok(m) if m.file_type().is_symlink() => {
                    let target = std::fs::read_link(entry.path())
                        .map(|t| t.display().to_string())
                        .unwrap_or_default();
                    format!("link        {name} -> {target}")
                }
                Ok(m) if m.is_dir() => format!("dir         {name}/"),
                Ok(m) if m.is_file() => format!("file {:>7} {name}", human_size(m.len())),
                Ok(_) => format!("other       {name}"),
                Err(e) => format!("?           {name} ({e})"),
            };
            lines.push((name, line));
        }
        lines.sort();
        let text: Vec<String> = lines.into_iter().map(|(_, l)| l).collect();
        Ok(ToolReply::ok(if text.is_empty() {
            "[empty directory]".to_string()
        } else {
            text.join("\n")
        }))
    }

    fn glob(&self, args: &Value) -> Result<ToolReply, ToolReply> {
        let pattern = arg_str(args, "pattern")?;
        let matcher = glob_regex(pattern)?;
        let base = match arg_opt_str(args, "path")? {
            Some(p) => self.resolve(p)?,
            None => self.cwd.clone(),
        };
        let mut found = Vec::new();
        let mut capped = false;
        walk(&base, &mut |rel, _| {
            if matcher.is_match(rel) {
                if found.len() == GLOB_CAP {
                    capped = true;
                    return false;
                }
                found.push(base.join(rel).display().to_string());
            }
            true
        })
        .map_err(|e| io_error(&base, e))?;
        if found.is_empty() {
            return Ok(ToolReply::ok("[no matches]"));
        }
        let mut text = found.join("\n");
        if capped {
            text.push_str(&format!("\n[stopped at {GLOB_CAP} results]"));
        }
        Ok(ToolReply::ok(text))
    }

    fn grep(&self, args: &Value) -> Result<ToolReply, ToolReply> {
        let pattern = arg_str(args, "pattern")?;
        let regex = Regex::new(pattern).map_err(|e| ToolReply::error(format!("bad regex: {e}")))?;
        let filter = arg_opt_str(args, "glob")?.map(glob_regex).transpose()?;
        let base = match arg_opt_str(args, "path")? {
            Some(p) => self.resolve(p)?,
            None => self.cwd.clone(),
        };
        let mut out = Vec::new();
        let mut capped = false;
        let mut skipped = 0usize;
        let meta = std::fs::metadata(&base).map_err(|e| io_error(&base, e))?;
        if meta.is_file() {
            // A file named outright is reported, not skipped: a silent skip
            // would read as "no matches".
            let found = grep_file(&base, &regex).map_err(|e| io_error(&base, e))?;
            for (number, line) in found {
                if out.len() == GREP_CAP {
                    capped = true;
                    break;
                }
                out.push(format!("{}:{number}:{line}", base.display()));
            }
            return Ok(grep_reply(out, capped, 0));
        }
        let mut search = |file: &Path| -> bool {
            let Ok(found) = grep_file(file, &regex) else {
                skipped += 1;
                return true;
            };
            for (number, line) in found {
                if out.len() == GREP_CAP {
                    capped = true;
                    return false;
                }
                out.push(format!("{}:{number}:{line}", file.display()));
            }
            true
        };
        walk(&base, &mut |rel, file_type| {
            if filter.as_ref().is_some_and(|f| !f.is_match(rel)) {
                return true;
            }
            let full = base.join(rel);
            let regular = file_type.is_file()
                || (file_type.is_symlink() && std::fs::metadata(&full).is_ok_and(|m| m.is_file()));
            if !regular {
                return true;
            }
            search(&full)
        })
        .map_err(|e| io_error(&base, e))?;
        Ok(grep_reply(out, capped, skipped))
    }

    fn write_file(&self, args: &Value) -> Result<ToolReply, ToolReply> {
        let path = self.resolve(arg_str(args, "path")?)?;
        let content = arg_str(args, "content")?;
        if content.len() > WRITE_CAP {
            return Err(ToolReply::error(format!(
                "content is {} bytes; the limit is {WRITE_CAP}",
                content.len()
            )));
        }
        let _writing = self.lock_writes();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| io_error(parent, e))?;
        }
        write_regular(&path, content.as_bytes()).map_err(|e| io_error(&path, e))?;
        Ok(ToolReply::ok(format!(
            "wrote {} bytes to {}",
            content.len(),
            path.display()
        )))
    }

    fn edit_file(&self, args: &Value) -> Result<ToolReply, ToolReply> {
        let path = self.resolve(arg_str(args, "path")?)?;
        let old = arg_str(args, "old")?;
        let new = arg_str(args, "new")?;
        let replace_all = arg_opt_bool(args, "replace_all")?.unwrap_or(false);
        if old.is_empty() {
            return Err(ToolReply::error("`old` must not be empty"));
        }
        let _writing = self.lock_writes();
        let mut file = open_regular(&path).map_err(|e| io_error(&path, e))?;
        let size = file.metadata().map_err(|e| io_error(&path, e))?.len();
        if size > WRITE_CAP as u64 {
            return Err(ToolReply::error(format!(
                "{}: the file is {size} bytes; edit_file handles at most {WRITE_CAP}",
                path.display()
            )));
        }
        let mut text = String::new();
        std::io::Read::read_to_string(&mut file, &mut text).map_err(|e| io_error(&path, e))?;
        let count = text.matches(old).count();
        if count == 0 {
            return Err(ToolReply::error(format!(
                "{}: `old` does not occur in the file",
                path.display()
            )));
        }
        if count > 1 && !replace_all {
            return Err(ToolReply::error(format!(
                "{}: `old` occurs {count} times; make it unique or set replace_all",
                path.display()
            )));
        }
        let edited = text.replace(old, new);
        if edited.len() > WRITE_CAP {
            return Err(ToolReply::error(format!(
                "the edited file would be {} bytes; the limit is {WRITE_CAP}",
                edited.len()
            )));
        }
        write_regular(&path, edited.as_bytes()).map_err(|e| io_error(&path, e))?;
        Ok(ToolReply::ok(format!(
            "replaced {count} occurrence{} in {}",
            if count == 1 { "" } else { "s" },
            path.display()
        )))
    }

    fn shell(&self, args: &Value) -> Result<ToolReply, ToolReply> {
        use std::os::unix::fs::FileExt;
        use std::os::unix::process::{CommandExt, ExitStatusExt};

        let command = arg_str(args, "command")?;
        let timeout = match arg_opt_u64(args, "timeout_s")? {
            Some(0) => return Err(ToolReply::error("`timeout_s` must be at least 1")),
            Some(t) => t.min(self.tool_wait_s),
            None => self.tool_wait_s,
        };
        let capture = capture_file().map_err(|e| ToolReply::error(format!("capture file: {e}")))?;
        let dup = || {
            capture
                .try_clone()
                .map_err(|e| ToolReply::error(format!("capture file: {e}")))
        };
        let mut child = std::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(command)
            .current_dir(&self.cwd)
            .stdin(std::process::Stdio::null())
            .stdout(dup()?)
            .stderr(dup()?)
            .process_group(0)
            .spawn()
            .map_err(|e| ToolReply::error(format!("starting /bin/sh: {e}")))?;
        let pgid = child.id();
        {
            // Under the lock, so a call starting as stdin closes is killed too.
            let mut running = self.lock_running();
            running.groups.insert(pgid);
            if running.closing {
                kill_group(pgid);
            }
        }
        // `sh` is waited on, never its output: a background job that keeps the
        // capture descriptor open cannot hold the call.
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(child.wait());
        });
        let (status, timed_out) = match rx.recv_timeout(Duration::from_secs(timeout)) {
            Ok(status) => (status, false),
            Err(_) => {
                kill_group(pgid);
                (
                    rx.recv().unwrap_or_else(|e| Err(std::io::Error::other(e))),
                    true,
                )
            }
        };
        self.lock_running().groups.remove(&pgid);
        let mut read_err = None;
        let len = capture.metadata().map_or_else(
            |e| {
                read_err = Some(e);
                0
            },
            |m| m.len(),
        );
        let budget = (RESULT_CAP - 1024) as u64;
        let start = len.saturating_sub(budget);
        let mut buf = vec![0u8; (len - start) as usize];
        let mut filled = 0;
        while filled < buf.len() {
            match capture.read_at(&mut buf[filled..], start + filled as u64) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => {
                    read_err = Some(e);
                    break;
                }
            }
        }
        buf.truncate(filled);
        let mut text = String::new();
        if start > 0 {
            text.push_str(&format!(
                "[output truncated: showing the last {filled} of {len} bytes]\n"
            ));
        }
        text.push_str(&String::from_utf8_lossy(&buf));
        if !text.is_empty() && !text.ends_with('\n') {
            text.push('\n');
        }
        if let Some(e) = read_err {
            text.push_str(&format!("[reading output failed: {e}]\n"));
        }
        if timed_out {
            text.push_str(&format!(
                "[timed out after {timeout} s; the command and its process group were killed]"
            ));
            return Ok(ToolReply::error(text));
        }
        match status {
            Ok(s) => match (s.code(), s.signal()) {
                (Some(code), _) => text.push_str(&format!("[exit status {code}]")),
                (None, Some(sig)) => text.push_str(&format!("[killed by signal {sig}]")),
                _ => text.push_str("[exit status unknown]"),
            },
            Err(e) => text.push_str(&format!("[wait failed: {e}]")),
        }
        Ok(ToolReply::ok(text))
    }

    fn lock_running(&self) -> std::sync::MutexGuard<'_, Running> {
        self.running.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn lock_writes(&self) -> std::sync::MutexGuard<'_, ()> {
        self.writes.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// SIGKILLs a `shell` call's process group. The group may already be gone,
/// so a failure is not an error.
fn kill_group(pgid: u32) {
    let _ = crate::procs::signal_group(pgid, libc::SIGKILL);
}

/// An unlinked, append-only temporary file: `sh`'s stdout and stderr share
/// its one descriptor, so the two streams interleave in write order.
fn capture_file() -> std::io::Result<File> {
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir();
    loop {
        let n = N.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!("ccc-shell-{}-{n}", std::process::id()));
        match OpenOptions::new()
            .read(true)
            .append(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => {
                std::fs::remove_file(&path)?;
                return Ok(file);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
}

fn human_size(n: u64) -> String {
    match n {
        n if n < 1024 => format!("{n}B"),
        n if n < 1024 * 1024 => format!("{:.1}K", n as f64 / 1024.0),
        n => format!("{:.1}M", n as f64 / (1024.0 * 1024.0)),
    }
}

/// Walks `base` depth-first in name order, calling `visit` with each entry's
/// path relative to `base` and its own (unfollowed) type. Symlinked
/// directories are not entered. `visit` returning false stops the walk;
/// unreadable subdirectories are skipped.
fn walk(
    base: &Path,
    visit: &mut dyn FnMut(&str, std::fs::FileType) -> bool,
) -> std::io::Result<()> {
    fn inner(
        dir: &Path,
        rel: &str,
        visit: &mut dyn FnMut(&str, std::fs::FileType) -> bool,
    ) -> std::io::Result<bool> {
        let mut entries: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let name = entry.file_name().to_string_lossy().into_owned();
            let child_rel = if rel.is_empty() {
                name
            } else {
                format!("{rel}/{name}")
            };
            if !visit(&child_rel, file_type) {
                return Ok(false);
            }
            if file_type.is_dir() && !inner(&entry.path(), &child_rel, visit).unwrap_or(true) {
                return Ok(false);
            }
        }
        Ok(true)
    }
    inner(base, "", visit).map(|_| ())
}

/// Translates a glob to an anchored, case-sensitive regex over `/`-separated
/// relative paths: `**` as a whole component matches zero or more components,
/// `*` any run of characters but `/`, `?` one character but `/`.
fn glob_regex(pattern: &str) -> Result<Regex, ToolReply> {
    if pattern.is_empty() {
        return Err(ToolReply::error("glob pattern must not be empty"));
    }
    if pattern.starts_with('/') {
        return Err(ToolReply::error(
            "glob pattern must be relative; pass the directory as `path`",
        ));
    }
    let components: Vec<&str> = pattern.split('/').collect();
    let mut re = String::from("^");
    for (i, component) in components.iter().enumerate() {
        let last = i + 1 == components.len();
        if *component == "**" {
            re.push_str(if last { ".*" } else { "(?:[^/]+/)*" });
            continue;
        }
        for c in component.chars() {
            match c {
                '*' => re.push_str("[^/]*"),
                '?' => re.push_str("[^/]"),
                c => re.push_str(&regex::escape(c.encode_utf8(&mut [0; 4]))),
            }
        }
        if !last {
            re.push('/');
        }
    }
    re.push('$');
    Regex::new(&re).map_err(|e| ToolReply::error(format!("bad glob pattern: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goose_tools::tests::{exchange, reply_for};
    use crate::testutil::TempDir;
    use std::sync::Arc;
    use std::time::Instant;

    fn tools_in(dir: &Path, groups: &[GooseTool]) -> BoxTools {
        let policy = GoosePolicy {
            tools: groups.iter().copied().collect(),
            agents: vec![],
            read_roots: vec![dir.to_path_buf()],
            write_roots: vec![dir.to_path_buf()],
            exec_roots: vec![],
        };
        BoxTools::new(policy, dir.to_path_buf(), 10)
    }

    fn all(dir: &Path) -> BoxTools {
        tools_in(dir, &[GooseTool::Read, GooseTool::Write, GooseTool::Shell])
    }

    fn ok(tools: &BoxTools, name: &str, args: Value) -> String {
        let r = tools.call(name, &args);
        assert!(!r.is_error, "{name} {args}: {}", r.text);
        r.text
    }

    fn err(tools: &BoxTools, name: &str, args: Value) -> String {
        let r = tools.call(name, &args);
        assert!(r.is_error, "{name} {args} should fail: {}", r.text);
        r.text
    }

    #[test]
    fn list_offers_only_granted_groups_and_names_writable_roots() {
        let d = TempDir::new("gt-list");
        let names = |t: &BoxTools| t.list().iter().map(|d| d.name).collect::<Vec<_>>();
        assert_eq!(
            names(&tools_in(d.path(), &[GooseTool::Read])),
            ["read_file", "list_dir", "glob", "grep"]
        );
        assert_eq!(
            names(&tools_in(d.path(), &[GooseTool::Write, GooseTool::Ccc])),
            ["write_file", "edit_file"]
        );
        let shell = tools_in(d.path(), &[GooseTool::Shell]);
        assert_eq!(names(&shell), ["shell"]);
        let description = &shell.list()[0].description;
        assert!(description.contains(&d.path().display().to_string()));
        assert!(description.contains("> log 2>&1 &"), "{description}");
        let read_only = tools_in(d.path(), &[GooseTool::Read]);
        assert!(
            read_only
                .call("write_file", &json!({"path": "x", "content": ""}))
                .is_error
        );
        assert!(
            read_only
                .call("shell", &json!({"command": "true"}))
                .is_error
        );
        assert!(read_only.call("nope", &json!({})).is_error);

        let out = d.path().join("out");
        let policy = GoosePolicy {
            tools: [GooseTool::Write, GooseTool::Shell].into(),
            agents: vec![],
            read_roots: vec![d.path().to_path_buf()],
            write_roots: vec![out.clone()],
            exec_roots: vec![],
        };
        let writer = BoxTools::new(policy.clone(), d.path().to_path_buf(), 10);
        let expected = format!("Writes succeed only under: {}.", out.display());
        for def in writer.list() {
            assert!(
                def.description.contains(&expected),
                "{}: {}",
                def.name,
                def.description
            );
        }
        let none = BoxTools::new(
            GoosePolicy {
                write_roots: vec![],
                ..policy
            },
            d.path().to_path_buf(),
            10,
        );
        for def in none.list() {
            assert!(
                def.description
                    .contains("Writes succeed only under: nothing outside /tmp."),
                "{}: {}",
                def.name,
                def.description
            );
        }
    }

    #[test]
    fn read_file_numbers_lines_and_honours_offset_limit_and_caps() {
        let d = TempDir::new("gt-read");
        let body: String = (1..=5).map(|i| format!("line {i}\n")).collect();
        d.write_file("f.txt", &body, 0o644);
        let t = all(d.path());
        assert_eq!(
            ok(&t, "read_file", json!({"path": "f.txt"})),
            "     1\tline 1\n     2\tline 2\n     3\tline 3\n     4\tline 4\n     5\tline 5\n"
        );
        assert_eq!(
            ok(
                &t,
                "read_file",
                json!({"path": "f.txt", "offset": 2, "limit": 2})
            ),
            "     2\tline 2\n     3\tline 3\n[more lines follow; next offset 4]\n"
        );
        let abs = d.path().join("f.txt").display().to_string();
        assert!(ok(&t, "read_file", json!({"path": abs, "offset": 9})).contains("has 5"));
        d.write_file("long.txt", &"x".repeat(LINE_CAP_CHARS + 5), 0o644);
        let long = ok(&t, "read_file", json!({"path": "long.txt"}));
        assert!(long.ends_with("…[line truncated]\n"), "{long}");
        let huge = format!("{}é\nsecond\n", "x".repeat(READ_LINE_BYTES - 1));
        d.write_file("huge.txt", &huge, 0o644);
        assert_eq!(
            ok(&t, "read_file", json!({"path": "huge.txt", "offset": 2})),
            "     2\tsecond\n"
        );
        let first = ok(&t, "read_file", json!({"path": "huge.txt", "limit": 1}));
        assert!(
            first.ends_with("…[line truncated]\n[more lines follow; next offset 2]\n"),
            "{first}"
        );
        assert_eq!(
            ok(
                &t,
                "read_file",
                json!({"path": "f.txt", "offset": 5, "limit": u64::MAX})
            ),
            "     5\tline 5\n"
        );
        std::fs::write(d.path().join("bin"), b"ab\0cd").unwrap();
        assert!(err(&t, "read_file", json!({"path": "bin"})).contains("binary"));
        assert!(err(&t, "read_file", json!({"path": "missing"})).contains("No such file"));
        assert!(err(&t, "read_file", json!({})).contains("`path`"));
    }

    #[test]
    fn list_dir_sorts_entries_with_types() {
        let d = TempDir::new("gt-ls");
        std::fs::create_dir(d.path().join("sub")).unwrap();
        d.write_file("b.txt", "hello", 0o644);
        std::os::unix::fs::symlink("b.txt", d.path().join("a-link")).unwrap();
        let t = all(d.path());
        assert_eq!(
            ok(&t, "list_dir", json!({"path": "."})),
            "link        a-link -> b.txt\nfile      5B b.txt\ndir         sub/"
        );
        std::fs::create_dir(d.path().join("empty")).unwrap();
        assert_eq!(
            ok(&t, "list_dir", json!({"path": "empty"})),
            "[empty directory]"
        );
    }

    #[test]
    fn glob_matcher_components() {
        let m = |p: &str, s: &str| glob_regex(p).unwrap().is_match(s);
        assert!(m("**/*.rs", "a.rs"));
        assert!(m("**/*.rs", "src/x/a.rs"));
        assert!(m("src/**/a.rs", "src/a.rs"));
        assert!(m("src/**/a.rs", "src/x/y/a.rs"));
        assert!(m("**", "any/thing"));
        assert!(!m("*.rs", "src/a.rs"));
        assert!(m("?.rs", "a.rs"));
        assert!(!m("?.rs", "ab.rs"));
        assert!(!m("a?b", "a/b"));
        assert!(!m("*.RS", "a.rs"));
        assert!(m("a+b(c).txt", "a+b(c).txt"));
        assert!(!m("a.txt", "abtxt"));
        assert!(glob_regex("/abs/*").is_err());
        assert!(glob_regex("").is_err());
    }

    #[test]
    fn glob_walks_without_following_symlinked_dirs_and_caps() {
        let d = TempDir::new("gt-glob");
        let outside = TempDir::new("gt-glob-out");
        std::fs::create_dir_all(d.path().join("src/deep")).unwrap();
        d.write_file("src/a.rs", "", 0o644);
        d.write_file("src/deep/b.rs", "", 0o644);
        d.write_file("top.rs", "", 0o644);
        std::fs::write(outside.path().join("c.rs"), "").unwrap();
        std::os::unix::fs::symlink(outside.path(), d.path().join("link")).unwrap();
        let t = all(d.path());
        let base = d.path().display().to_string();
        assert_eq!(
            ok(&t, "glob", json!({"pattern": "**/*.rs"})),
            format!("{base}/src/a.rs\n{base}/src/deep/b.rs\n{base}/top.rs")
        );
        assert_eq!(
            ok(&t, "glob", json!({"pattern": "*.rs", "path": "src"})),
            format!("{base}/src/a.rs")
        );
        assert_eq!(ok(&t, "glob", json!({"pattern": "*.py"})), "[no matches]");
        let many = TempDir::new("gt-glob-many");
        for i in 0..GLOB_CAP + 3 {
            std::fs::write(many.path().join(format!("f{i}")), "").unwrap();
        }
        let capped = ok(&t, "glob", json!({"pattern": "*", "path": many.path()}));
        assert_eq!(capped.lines().count(), GLOB_CAP + 1);
        assert!(capped.ends_with(&format!("[stopped at {GLOB_CAP} results]")));
    }

    #[test]
    fn grep_matches_filters_skips_binary_and_caps() {
        let d = TempDir::new("gt-grep");
        std::fs::create_dir(d.path().join("src")).unwrap();
        d.write_file("src/a.rs", "fn main() {}\nlet x = 1;\n", 0o644);
        d.write_file("notes.md", "fn in prose\n", 0o644);
        std::fs::write(d.path().join("blob"), b"fn\0binary").unwrap();
        let t = all(d.path());
        let base = d.path().display().to_string();
        assert_eq!(
            ok(&t, "grep", json!({"pattern": "^fn"})),
            format!(
                "{base}/notes.md:1:fn in prose\n{base}/src/a.rs:1:fn main() {{}}\n\
                 [1 file not searched: binary, not UTF-8, unreadable or with a line over {GREP_LINE_BYTES} bytes]"
            )
        );
        assert!(err(&t, "grep", json!({"pattern": "fn", "path": "blob"})).contains("binary file"));
        assert_eq!(
            ok(&t, "grep", json!({"pattern": "fn", "glob": "**/*.rs"})),
            format!("{base}/src/a.rs:1:fn main() {{}}")
        );
        assert_eq!(
            ok(&t, "grep", json!({"pattern": "x =", "path": "src/a.rs"})),
            format!("{base}/src/a.rs:2:let x = 1;")
        );
        assert!(err(&t, "grep", json!({"pattern": "("})).contains("bad regex"));
        d.write_file("many.txt", &"hit\n".repeat(GREP_CAP + 10), 0o644);
        let capped = ok(&t, "grep", json!({"pattern": "hit", "path": "many.txt"}));
        assert_eq!(capped.lines().count(), GREP_CAP + 1);
        assert!(capped.ends_with(&format!("[stopped at {GREP_CAP} matches]")));
    }

    fn mkfifo(path: &Path) {
        let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
        // SAFETY: a valid NUL-terminated path.
        assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
    }

    /// Runs `f` on its own thread and fails the test if it has not returned
    /// within 10 s, so a regression that blocks on a fifo fails, not hangs.
    fn within_10s<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send(f());
        });
        rx.recv_timeout(Duration::from_secs(10))
            .expect("the tool call blocked")
    }

    #[test]
    fn fifos_are_skipped_or_refused_without_blocking() {
        let d = TempDir::new("gt-fifo");
        d.write_file("a.txt", "needle\n", 0o644);
        mkfifo(&d.path().join("pipe"));
        let base = d.path().to_path_buf();
        let (grep, read, write, edit) = within_10s(move || {
            let t = all(&base);
            (
                ok(&t, "grep", json!({"pattern": "needle"})),
                err(&t, "read_file", json!({"path": "pipe"})),
                err(&t, "write_file", json!({"path": "pipe", "content": "x"})),
                err(
                    &t,
                    "edit_file",
                    json!({"path": "pipe", "old": "a", "new": "b"}),
                ),
            )
        });
        assert_eq!(grep, format!("{}/a.txt:1:needle", d.path().display()));
        assert!(read.contains("not a regular file"), "{read}");
        // With no reader, opening a fifo for writing fails at once.
        assert!(write.contains("pipe"), "{write}");
        assert!(edit.contains("not a regular file"), "{edit}");
    }

    #[test]
    fn grep_skips_a_file_with_an_overlong_line() {
        let d = TempDir::new("gt-grep-long");
        d.write_file(
            "long.txt",
            &format!("needle{}\n", "x".repeat(GREP_LINE_BYTES)),
            0o644,
        );
        d.write_file("short.txt", "needle\n", 0o644);
        let t = all(d.path());
        let reply = ok(&t, "grep", json!({"pattern": "needle"}));
        assert!(
            reply.starts_with(&format!(
                "{}/short.txt:1:needle\n[1 file not searched",
                d.path().display()
            )),
            "{reply}"
        );
        let named = err(&t, "grep", json!({"pattern": "needle", "path": "long.txt"}));
        assert!(named.contains("1 MiB"), "{named}");
    }

    #[test]
    fn grep_searches_symlinked_files_and_reports_non_utf8_ones() {
        let d = TempDir::new("gt-grep-link");
        d.write_file("a.txt", "needle\n", 0o644);
        std::fs::write(d.path().join("latin1.txt"), b"needle \xe9\n").unwrap();
        std::os::unix::fs::symlink("a.txt", d.path().join("link.txt")).unwrap();
        mkfifo(&d.path().join("pipe"));
        std::os::unix::fs::symlink("pipe", d.path().join("pipe-link")).unwrap();
        let base = d.path().to_path_buf();
        let (walked, named) = within_10s(move || {
            let t = all(&base);
            (
                ok(&t, "grep", json!({"pattern": "needle"})),
                err(
                    &t,
                    "grep",
                    json!({"pattern": "needle", "path": "latin1.txt"}),
                ),
            )
        });
        let base = d.path().display();
        assert_eq!(
            walked,
            format!(
                "{base}/a.txt:1:needle\n{base}/link.txt:1:needle\n\
                 [1 file not searched: binary, not UTF-8, unreadable or with a line over {GREP_LINE_BYTES} bytes]"
            )
        );
        assert!(named.contains("latin1.txt: a line is not UTF-8"), "{named}");
    }

    #[test]
    fn write_file_creates_parents_and_enforces_the_cap() {
        let d = TempDir::new("gt-write");
        let t = all(d.path());
        let reply = ok(
            &t,
            "write_file",
            json!({"path": "a/b/c.txt", "content": "hi"}),
        );
        assert!(reply.starts_with("wrote 2 bytes"), "{reply}");
        assert_eq!(
            std::fs::read_to_string(d.path().join("a/b/c.txt")).unwrap(),
            "hi"
        );
        ok(
            &t,
            "write_file",
            json!({"path": "a/b/c.txt", "content": "over"}),
        );
        assert_eq!(
            std::fs::read_to_string(d.path().join("a/b/c.txt")).unwrap(),
            "over"
        );
        let big = "x".repeat(WRITE_CAP + 1);
        assert!(err(&t, "write_file", json!({"path": "big", "content": big})).contains("limit"));
        assert!(!d.path().join("big").exists());
    }

    #[test]
    fn read_line_capped_keeps_the_cap_and_skips_the_rest() {
        let mut r = std::io::BufReader::with_capacity(4, &b"abcdefgh\nab\nabcd\nabcd"[..]);
        let mut line = Vec::new();
        let mut next = || {
            let cut = read_line_capped(&mut r, &mut line, 4).unwrap();
            cut.map(|c| (String::from_utf8(line.clone()).unwrap(), c))
        };
        assert_eq!(next(), Some(("abcd".into(), true)));
        assert_eq!(next(), Some(("ab\n".into(), false)));
        assert_eq!(next(), Some(("abcd".into(), false)));
        assert_eq!(next(), Some(("abcd".into(), false)));
        assert_eq!(next(), None);
    }

    #[test]
    fn edit_file_refuses_a_file_over_the_cap_before_reading_it() {
        let d = TempDir::new("gt-edit-big");
        let f = std::fs::File::create(d.path().join("big")).unwrap();
        f.set_len(WRITE_CAP as u64 + 1).unwrap();
        let t = all(d.path());
        let e = err(
            &t,
            "edit_file",
            json!({"path": "big", "old": "a", "new": "b"}),
        );
        assert!(e.contains("at most"), "{e}");
    }

    #[test]
    fn concurrent_edits_of_one_file_all_land() {
        let d = TempDir::new("gt-edit-race");
        let body: String = (0..64).map(|i| format!("v{i}=old\n")).collect();
        let f = d.write_file("f.txt", &body, 0o644);
        let t = Arc::new(all(d.path()));
        let threads: Vec<_> = (0..64)
            .map(|i| {
                let t = Arc::clone(&t);
                std::thread::spawn(move || {
                    ok(
                        &t,
                        "edit_file",
                        json!({"path": "f.txt", "old": format!("v{i}=old\n"), "new": format!("v{i}=new\n")}),
                    )
                })
            })
            .collect();
        for th in threads {
            th.join().unwrap();
        }
        let expected: String = (0..64).map(|i| format!("v{i}=new\n")).collect();
        assert_eq!(std::fs::read_to_string(&f).unwrap(), expected);
    }

    #[test]
    fn edit_file_unique_missing_duplicate_and_replace_all() {
        let d = TempDir::new("gt-edit");
        let f = d.write_file("f.txt", "one two two\n", 0o644);
        let t = all(d.path());
        ok(
            &t,
            "edit_file",
            json!({"path": "f.txt", "old": "one", "new": "1"}),
        );
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "1 two two\n");
        assert!(err(
            &t,
            "edit_file",
            json!({"path": "f.txt", "old": "zzz", "new": ""})
        )
        .contains("does not occur"));
        assert!(err(
            &t,
            "edit_file",
            json!({"path": "f.txt", "old": "two", "new": "2"})
        )
        .contains("occurs 2 times"));
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "1 two two\n");
        let reply = ok(
            &t,
            "edit_file",
            json!({"path": "f.txt", "old": "two", "new": "2", "replace_all": true}),
        );
        assert!(reply.starts_with("replaced 2 occurrences"), "{reply}");
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "1 2 2\n");
        assert!(err(
            &t,
            "edit_file",
            json!({"path": "f.txt", "old": "", "new": "x"})
        )
        .contains("empty"));
    }

    #[test]
    fn shell_interleaves_streams_and_reports_status() {
        let d = TempDir::new("gt-sh");
        let t = all(d.path());
        assert_eq!(
            ok(
                &t,
                "shell",
                json!({"command": "echo out; echo err >&2; echo out2; exit 3"})
            ),
            "out\nerr\nout2\n[exit status 3]"
        );
        assert_eq!(
            ok(&t, "shell", json!({"command": "pwd"})),
            format!("{}\n[exit status 0]", d.path().display())
        );
        assert!(err(&t, "shell", json!({"command": "true", "timeout_s": 0})).contains("at least 1"));
    }

    #[test]
    fn shell_returns_without_waiting_for_background_output() {
        let d = TempDir::new("gt-sh-bg");
        let t = all(d.path());
        let started = Instant::now();
        assert_eq!(
            ok(&t, "shell", json!({"command": "sleep 3 & echo hi"})),
            "hi\n[exit status 0]"
        );
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "{:?}",
            started.elapsed()
        );
    }

    #[test]
    fn shell_timeout_kills_the_process_group() {
        let d = TempDir::new("gt-sh-timeout");
        let t = all(d.path());
        let started = Instant::now();
        let text = err(
            &t,
            "shell",
            json!({"command": "sleep 3 & echo $! > bg.pid; echo started; sleep 5", "timeout_s": 1}),
        );
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "{:?}",
            started.elapsed()
        );
        assert!(text.starts_with("started\n[timed out after 1 s"), "{text}");
        let pid: i32 = std::fs::read_to_string(d.path().join("bg.pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        // The background sleep was SIGKILLed with the group; once reaped it is
        // gone, and until then it is a zombie.
        let mut gone = false;
        for _ in 0..50 {
            match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
                Err(_) => gone = true,
                Ok(stat) => {
                    gone = stat
                        .rsplit(')')
                        .next()
                        .unwrap_or("")
                        .trim_start()
                        .starts_with('Z')
                }
            }
            if gone {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(gone, "background job {pid} survived the timeout");
        assert!(t.lock_running().groups.is_empty());
    }

    #[test]
    fn shell_keeps_the_tail_and_the_exit_status_of_long_output() {
        let d = TempDir::new("gt-shell-long");
        let t = all(d.path());
        let command = "head -c 400000 /dev/zero | tr '\\0' x; exit 4";
        let text = ok(&t, "shell", json!({"command": command}));
        assert!(
            text.starts_with("[output truncated: showing the last"),
            "{}",
            text.chars().take(200).collect::<String>()
        );
        assert!(
            text.ends_with("x\n[exit status 4]"),
            "{}",
            &text[text.len() - 80..]
        );
        // The reply must already fit `RESULT_CAP`: cutting it there would drop
        // the exit line.
        assert_eq!(crate::goose_tools::cap_text(text.clone(), RESULT_CAP), text);
    }

    #[test]
    fn descriptors_from_a_threshold_up_are_swept() {
        use std::os::fd::AsRawFd;
        let d = TempDir::new("gt-fds");
        let f = d.write_file("f", "", 0o644);
        let file = std::fs::File::open(&f).unwrap();
        let flags = |fd: i32| {
            // SAFETY: F_GETFD only reads the descriptor's flags.
            unsafe { libc::fcntl(fd, libc::F_GETFD) }
        };
        // SAFETY: clears the flag on a descriptor this test owns, so a sweep
        // that reached below its threshold would show.
        assert_eq!(
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_SETFD, 0) },
            0
        );
        for (fd, sweep, op) in [
            (900, sweep_fds_from as fn(u32, Sweep), Sweep::Close),
            (901, sweep_listed_fds_from, Sweep::Close),
            (902, sweep_fds_from, Sweep::Cloexec),
            (903, sweep_listed_fds_from, Sweep::Cloexec),
        ] {
            // SAFETY: dup2 onto a number this test owns; nothing else uses it.
            assert_eq!(unsafe { libc::dup2(file.as_raw_fd(), fd) }, fd);
            assert_eq!(flags(fd), 0);
            sweep(fd as u32, op);
            match op {
                Sweep::Close => assert_eq!(flags(fd), -1, "fd {fd} still open"),
                Sweep::Cloexec => {
                    assert_eq!(flags(fd), libc::FD_CLOEXEC, "fd {fd} not close-on-exec");
                    // SAFETY: closing the descriptor this iteration made.
                    unsafe { libc::close(fd) };
                }
            }
            assert_eq!(
                flags(file.as_raw_fd()),
                0,
                "a descriptor below the threshold was swept"
            );
        }
    }

    #[test]
    fn shell_timeout_is_capped_at_tool_wait() {
        let d = TempDir::new("gt-sh-cap");
        let mut t = all(d.path());
        t.tool_wait_s = 1;
        let started = Instant::now();
        let text = err(&t, "shell", json!({"command": "sleep 3", "timeout_s": 200}));
        assert!(text.contains("timed out after 1 s"), "{text}");
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn protocol_round_trip_read_and_write() {
        let d = TempDir::new("gt-proto");
        d.write_file("in.txt", "alpha\n", 0o644);
        let replies = exchange(
            Arc::new(all(d.path())),
            &[
                json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}),
                json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"read_file","arguments":{"path":"in.txt"}}}),
                json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"write_file","arguments":{"path":"out.txt","content":"beta"}}}),
            ],
        );
        assert_eq!(
            reply_for(&replies, 1)["result"]["tools"]
                .as_array()
                .unwrap()
                .len(),
            7
        );
        assert_eq!(
            reply_for(&replies, 2)["result"]["content"][0]["text"],
            "     1\talpha\n"
        );
        assert_eq!(reply_for(&replies, 3)["result"]["isError"], false);
        assert_eq!(
            std::fs::read_to_string(d.path().join("out.txt")).unwrap(),
            "beta"
        );
    }

    #[test]
    fn eof_kills_running_shell_calls() {
        let d = TempDir::new("gt-proto-eof");
        let started = Instant::now();
        let replies = exchange(
            Arc::new(all(d.path())),
            &[
                json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"shell","arguments":{"command":"sleep 5"}}}),
            ],
        );
        assert!(
            started.elapsed() < Duration::from_secs(4),
            "{:?}",
            started.elapsed()
        );
        let text = reply_for(&replies, 1)["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(text.contains("killed by signal 9"), "{text}");
    }

    #[test]
    fn box_inner_refuses_to_serve_on_the_host() {
        let err = check_in_box().unwrap_err();
        assert!(
            err.to_string().contains("only inside the bubblewrap box"),
            "{err}"
        );
    }
}
