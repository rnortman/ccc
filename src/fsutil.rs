//! Filesystem helpers shared by the modules that persist state.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Replaces `path` through a sibling temporary file and a rename, so a reader
/// never sees a torn file and a kill mid-write cannot truncate the target. A
/// new file is created 0600; an existing one keeps its mode. A `path` that is a
/// symlink is followed, so the file the link points at is what gets replaced.
pub fn write_atomically(path: &Path, contents: &str) -> Result<()> {
    // Renaming over a symlink would replace the link with a regular file and
    // orphan whatever the operator pointed it at.
    let resolved: PathBuf = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let path = resolved.as_path();

    let dir = path.parent().unwrap_or(Path::new("."));
    let temp = dir.join(format!(
        ".{}.ccc-{}",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("tmp"),
        std::process::id()
    ));

    let mut open = std::fs::OpenOptions::new();
    // `create_new` also refuses a stale temp file left by a crashed process
    // that happened to share this pid.
    open.write(true).create_new(true);
    #[cfg(unix)]
    {
        // The temp file is born with the final mode: its contents are never
        // visible through the wider mode a later chmod would have to narrow.
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        let mode = std::fs::metadata(path)
            .map(|m| m.permissions().mode() & 0o777)
            .unwrap_or(0o600);
        open.mode(mode);
    }

    let mut file = open
        .open(&temp)
        .with_context(|| format!("creating {}", temp.display()))?;
    let written = file
        .write_all(contents.as_bytes())
        .and_then(|()| file.flush());
    if let Err(e) = written {
        drop(file);
        let _ = std::fs::remove_file(&temp);
        return Err(e).with_context(|| format!("writing {}", temp.display()));
    }
    drop(file);

    if let Err(e) = std::fs::rename(&temp, path) {
        let _ = std::fs::remove_file(&temp);
        return Err(e).with_context(|| format!("replacing {}", path.display()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    #[test]
    fn a_new_file_is_created_private_and_no_temp_is_left_behind() {
        let dir = TempDir::new("fsutil-new");
        let path = dir.path().join("state.json");
        write_atomically(&path, "{}").unwrap();

        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{}");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600, "{mode:o}");
        }
        let left: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(left.len(), 1, "{left:?}");
    }

    #[test]
    fn an_existing_file_is_replaced_whole_and_keeps_its_mode() {
        let dir = TempDir::new("fsutil-existing");
        let path = dir.write_file("state.json", "old", 0o644);
        write_atomically(&path, "new").unwrap();

        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o644, "{mode:o}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_path_updates_its_target_instead_of_replacing_the_link() {
        let dir = TempDir::new("fsutil-symlink");
        let target = dir.write_file("real.json", "old", 0o600);
        let link = dir.path().join("link.json");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        write_atomically(&link, "new").unwrap();

        assert!(std::fs::symlink_metadata(&link).unwrap().is_symlink());
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "new");
    }

    #[test]
    fn a_directory_that_does_not_exist_is_an_error_not_a_panic() {
        let dir = TempDir::new("fsutil-missing");
        let path = dir.path().join("nope/state.json");
        assert!(write_atomically(&path, "{}").is_err());
    }
}
