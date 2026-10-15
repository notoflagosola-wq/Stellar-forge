// Copyright 2024 Stellar-K8s Contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Filesystem abstraction layer for the History Archive Pruning Daemon (#242).
//!
//! This module provides async-safe utilities for walking the Stellar bucket
//! tree, classifying files eligible for pruning, checking whether a file is
//! held open by another process, and deleting files safely.
//!
//! # Design Principles
//!
//! * **Read-only scan first** – all I/O during discovery is read-only.
//! * **No deletion without explicit approval** – callers must pass
//!   `DryRun::No` explicitly; the default is always dry-run.
//! * **File-lock check** – before deleting a file the pruner calls
//!   [`is_file_open`] to ensure no peer process (e.g. stellar-core or
//!   Horizon) currently holds a read descriptor on it.
//! * **Atomic progression** – errors on individual files are collected and
//!   returned in the [`ScanResult`] without aborting the whole sweep.
//!
//! # File Classification
//!
//! A file is considered *prunable* when **all** of the following are true:
//!
//! | Criterion | Detail |
//! |-----------|--------|
//! | Extension | `.xdr`, `.xdr.gz`, `.scp`, or `bucket-*` prefix |
//! | Age | `mtime < now - retention_window` |
//! | Not open | [`is_file_open`] returns `false` |

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use tokio::fs;
use tracing::{debug, trace, warn};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Controls whether deletions are actually performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DryRun {
    /// Log what would be deleted but make no filesystem changes.
    Yes,
    /// Actually remove files from disk.
    No,
}

impl Default for DryRun {
    fn default() -> Self {
        DryRun::Yes
    }
}

/// Metadata about a single file discovered during a directory walk.
#[derive(Debug, Clone)]
pub struct FileEntry {
    /// Absolute path to the file.
    pub path: PathBuf,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Last modification time.
    pub modified: SystemTime,
    /// Whether this file matches the prunable pattern.
    pub is_prunable: bool,
}

/// Aggregated result of a single directory scan pass.
#[derive(Debug, Default)]
pub struct ScanResult {
    /// All files discovered (regardless of prune eligibility).
    pub entries: Vec<FileEntry>,
    /// Total bytes across all discovered files.
    pub total_bytes: u64,
    /// Total bytes eligible for deletion.
    pub prunable_bytes: u64,
    /// Number of files that could not be stat-ed (I/O errors).
    pub stat_errors: usize,
}

// ---------------------------------------------------------------------------
// File classification
// ---------------------------------------------------------------------------

/// Returns `true` when the file name matches a known Stellar bucket / XDR
/// pattern that the pruner is allowed to remove.
///
/// Recognised patterns (case-insensitive extension check):
/// - `*.xdr`
/// - `*.xdr.gz`
/// - `*.scp`
/// - `bucket-*`  (Stellar Core bucket files, no extension)
pub fn is_prunable_name(path: &Path) -> bool {
    let name = match path.file_name().and_then(|n| n.to_str()) {
        Some(n) => n,
        None => return false,
    };

    // Bucket files: prefix-based match
    if name.starts_with("bucket-") {
        return true;
    }

    // XDR / SCP files: extension-based match
    if name.ends_with(".xdr") || name.ends_with(".xdr.gz") || name.ends_with(".scp") {
        return true;
    }

    false
}

/// Returns `true` if `modified_at` is older than `retention_window` relative
/// to `now`.
///
/// Files that cannot be compared (e.g. future timestamps or clock skew
/// producing a negative age) are conservatively treated as **not** prunable.
pub fn is_older_than(modified_at: SystemTime, now: SystemTime, retention_window: Duration) -> bool {
    match now.duration_since(modified_at) {
        Ok(age) => age > retention_window,
        Err(_) => false, // file is in the future – never prune
    }
}

// ---------------------------------------------------------------------------
// File-lock check
// ---------------------------------------------------------------------------

/// Returns `true` if any process currently has `path` open.
///
/// On Linux this is implemented by scanning `/proc/*/fd/*` symlinks.
/// On other platforms (e.g. macOS in tests) the function always returns
/// `false` so that pruning can proceed without the proc filesystem.
///
/// This is a **best-effort** check: a race between the check and the
/// subsequent `remove_file` call is unavoidable without OS-level advisory
/// locks.  The window is short and stellar-core does not hold file
/// descriptors on old XDR archives once they are written, so in practice
/// this race is negligible.
#[cfg(target_os = "linux")]
pub async fn is_file_open(path: &Path) -> bool {
    let target = match fs::canonicalize(path).await {
        Ok(p) => p,
        Err(_) => return false, // can't canonicalize → assume not open
    };
    let target_str = target.to_string_lossy().into_owned();

    let proc_fd = Path::new("/proc");
    let mut read_proc = match fs::read_dir(proc_fd).await {
        Ok(r) => r,
        Err(_) => return false,
    };

    while let Ok(Some(entry)) = read_proc.next_entry().await {
        let fd_dir = entry.path().join("fd");
        if !fd_dir.is_dir() {
            continue;
        }
        let mut read_fd = match fs::read_dir(&fd_dir).await {
            Ok(r) => r,
            Err(_) => continue,
        };
        while let Ok(Some(fd_entry)) = read_fd.next_entry().await {
            if let Ok(resolved) = fs::canonicalize(fd_entry.path()).await {
                if resolved.to_string_lossy() == target_str {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(not(target_os = "linux"))]
pub async fn is_file_open(_path: &Path) -> bool {
    // Non-Linux: conservative default – assume not open, allow deletion.
    false
}

// ---------------------------------------------------------------------------
// Directory scanner
// ---------------------------------------------------------------------------

/// Recursively walk `root_dir` and collect all files, classifying each one
/// according to [`is_prunable_name`] and the supplied `retention_window`.
///
/// `now` is accepted as a parameter (rather than calling `SystemTime::now()`
/// internally) so callers can use a fixed instant across multiple scans and
/// unit tests can inject arbitrary timestamps.
///
/// Files with I/O errors during `metadata()` are counted in
/// [`ScanResult::stat_errors`] and skipped.
pub async fn scan_directory(
    root_dir: &Path,
    retention_window: Duration,
    now: SystemTime,
) -> Result<ScanResult> {
    let mut result = ScanResult::default();
    scan_recursive(root_dir, retention_window, now, &mut result).await;
    Ok(result)
}

/// Internal recursive helper used by [`scan_directory`].
async fn scan_recursive(
    dir: &Path,
    retention_window: Duration,
    now: SystemTime,
    result: &mut ScanResult,
) {
    let mut read_dir = match fs::read_dir(dir).await {
        Ok(r) => r,
        Err(e) => {
            warn!(path = %dir.display(), error = %e, "Failed to read directory during scan");
            result.stat_errors += 1;
            return;
        }
    };

    while let Ok(Some(entry)) = read_dir.next_entry().await {
        let path = entry.path();

        if path.is_dir() {
            // Recurse – Box::pin avoids infinite-size future error.
            Box::pin(scan_recursive(&path, retention_window, now, result)).await;
            continue;
        }

        let metadata = match entry.metadata().await {
            Ok(m) => m,
            Err(e) => {
                warn!(path = %path.display(), error = %e, "Failed to stat file, skipping");
                result.stat_errors += 1;
                continue;
            }
        };

        let modified = match metadata.modified() {
            Ok(t) => t,
            Err(e) => {
                warn!(path = %path.display(), error = %e, "Failed to read mtime, skipping");
                result.stat_errors += 1;
                continue;
            }
        };

        let size_bytes = metadata.len();
        let name_prunable = is_prunable_name(&path);
        let age_prunable = is_older_than(modified, now, retention_window);
        let is_prunable = name_prunable && age_prunable;

        trace!(
            path = %path.display(),
            size_bytes,
            name_prunable,
            age_prunable,
            is_prunable,
            "Scanned file"
        );

        result.total_bytes += size_bytes;
        if is_prunable {
            result.prunable_bytes += size_bytes;
        }
        result.entries.push(FileEntry {
            path,
            size_bytes,
            modified,
            is_prunable,
        });
    }
}

// ---------------------------------------------------------------------------
// Safe deletion
// ---------------------------------------------------------------------------

/// Attempt to delete a single file, respecting the `dry_run` flag and
/// checking whether the file is currently open.
///
/// Returns the number of bytes freed (0 on dry-run or if the file is open).
///
/// # Errors
///
/// Returns an error only when `dry_run == DryRun::No` and the underlying
/// `tokio::fs::remove_file` call fails.
pub async fn delete_file(path: &Path, size_bytes: u64, dry_run: DryRun) -> Result<u64> {
    if dry_run == DryRun::Yes {
        debug!(path = %path.display(), size_bytes, "DRY-RUN: would delete file");
        return Ok(0);
    }

    // Safety check: skip files that are currently held open.
    if is_file_open(path).await {
        warn!(
            path = %path.display(),
            "Skipping deletion – file is currently open by another process"
        );
        return Ok(0);
    }

    fs::remove_file(path)
        .await
        .with_context(|| format!("Failed to delete file: {}", path.display()))?;

    debug!(path = %path.display(), size_bytes, "Deleted file");
    Ok(size_bytes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};
    use tempfile::TempDir;
    use tokio::fs as tfs;

    // -----------------------------------------------------------------------
    // is_prunable_name
    // -----------------------------------------------------------------------

    #[test]
    fn xdr_files_are_prunable() {
        assert!(is_prunable_name(Path::new("/data/ledger-001.xdr")));
        assert!(is_prunable_name(Path::new("ledger-001.xdr")));
    }

    #[test]
    fn xdr_gz_files_are_prunable() {
        assert!(is_prunable_name(Path::new("/data/ledger-001.xdr.gz")));
    }

    #[test]
    fn scp_files_are_prunable() {
        assert!(is_prunable_name(Path::new("/data/quorum.scp")));
    }

    #[test]
    fn bucket_files_are_prunable() {
        assert!(is_prunable_name(Path::new(
            "/var/lib/stellar/buckets/bucket-abc123"
        )));
    }

    #[test]
    fn config_files_are_not_prunable() {
        assert!(!is_prunable_name(Path::new("stellar-core.cfg")));
        assert!(!is_prunable_name(Path::new("stellar-history.json")));
        assert!(!is_prunable_name(Path::new("peers.dat")));
    }

    #[test]
    fn directories_are_not_prunable() {
        // A bare directory path without a file component
        assert!(!is_prunable_name(Path::new("/var/lib/stellar/buckets/")));
    }

    // -----------------------------------------------------------------------
    // is_older_than
    // -----------------------------------------------------------------------

    #[test]
    fn file_older_than_retention_is_prunable() {
        let now = SystemTime::now();
        let thirty_days = Duration::from_secs(30 * 24 * 3600);
        let modified = now - Duration::from_secs(31 * 24 * 3600); // 31 days ago
        assert!(is_older_than(modified, now, thirty_days));
    }

    #[test]
    fn file_within_retention_is_not_prunable() {
        let now = SystemTime::now();
        let thirty_days = Duration::from_secs(30 * 24 * 3600);
        let modified = now - Duration::from_secs(29 * 24 * 3600); // 29 days ago
        assert!(!is_older_than(modified, now, thirty_days));
    }

    #[test]
    fn future_file_is_not_prunable() {
        let now = SystemTime::now();
        let thirty_days = Duration::from_secs(30 * 24 * 3600);
        let modified = now + Duration::from_secs(60); // 1 min in the future
        assert!(!is_older_than(modified, now, thirty_days));
    }

    // -----------------------------------------------------------------------
    // scan_directory
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn scan_finds_prunable_xdr_files() {
        let dir = TempDir::new().unwrap();
        let old_file = dir.path().join("ledger-old.xdr");
        let new_file = dir.path().join("ledger-new.xdr");
        let cfg_file = dir.path().join("stellar-core.cfg");

        tfs::write(&old_file, b"old-data").await.unwrap();
        tfs::write(&new_file, b"new-data").await.unwrap();
        tfs::write(&cfg_file, b"config").await.unwrap();

        // Make the old file appear 40 days old by back-dating its mtime.
        let forty_days_ago = SystemTime::now() - Duration::from_secs(40 * 24 * 3600);
        filetime::set_file_mtime(
            &old_file,
            filetime::FileTime::from_system_time(forty_days_ago),
        )
        .unwrap();

        let retention = Duration::from_secs(30 * 24 * 3600);
        let result = scan_directory(dir.path(), retention, SystemTime::now())
            .await
            .unwrap();

        assert_eq!(result.entries.len(), 3, "should find 3 files total");

        let prunable: Vec<_> = result.entries.iter().filter(|e| e.is_prunable).collect();
        assert_eq!(prunable.len(), 1, "exactly one file should be prunable");
        assert_eq!(prunable[0].path, old_file);
        assert!(result.prunable_bytes > 0);
        assert_eq!(result.stat_errors, 0);
    }

    #[tokio::test]
    async fn scan_recurses_into_subdirectories() {
        let dir = TempDir::new().unwrap();
        let subdir = dir.path().join("subdir");
        tfs::create_dir_all(&subdir).await.unwrap();
        let nested = subdir.join("nested.xdr");
        tfs::write(&nested, b"nested").await.unwrap();

        // Back-date the nested file.
        let long_ago = SystemTime::now() - Duration::from_secs(365 * 24 * 3600);
        filetime::set_file_mtime(
            &nested,
            filetime::FileTime::from_system_time(long_ago),
        )
        .unwrap();

        let retention = Duration::from_secs(30 * 24 * 3600);
        let result = scan_directory(dir.path(), retention, SystemTime::now())
            .await
            .unwrap();

        let prunable: Vec<_> = result.entries.iter().filter(|e| e.is_prunable).collect();
        assert_eq!(prunable.len(), 1);
        assert_eq!(prunable[0].path, nested);
    }

    // -----------------------------------------------------------------------
    // delete_file (dry-run)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn dry_run_does_not_delete_files() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.xdr");
        tfs::write(&file, b"data").await.unwrap();

        let freed = delete_file(&file, 4, DryRun::Yes).await.unwrap();
        assert_eq!(freed, 0, "dry-run should free 0 bytes");
        assert!(file.exists(), "file must still exist after dry-run");
    }

    #[tokio::test]
    async fn real_delete_removes_file() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("test.xdr");
        tfs::write(&file, b"data").await.unwrap();

        let freed = delete_file(&file, 4, DryRun::No).await.unwrap();
        assert_eq!(freed, 4, "should report bytes freed");
        assert!(!file.exists(), "file must be gone after deletion");
    }
}
