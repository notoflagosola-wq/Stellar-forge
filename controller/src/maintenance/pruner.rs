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

//! History Archive Pruning Daemon (#242).
//!
//! Long-running Stellar validator nodes accumulate XDR history archive files
//! and bucket snapshots in `/var/lib/stellar/buckets`.  Left unmanaged, this
//! storage grows without bound and eventually exhausts the attached PVC,
//! causing OOM / disk-full crashes.
//!
//! This module implements a periodic daemon that:
//!
//! 1. **Reads** the Stellar Core TOML configuration to determine the minimum
//!    ledger retention window required by the active quorum set.
//! 2. **Scans** the configured bucket directory for XDR, SCP, and raw bucket
//!    files that are older than the computed retention window.
//! 3. **Checks** that no stellar-core or Horizon process currently holds an
//!    open file descriptor on each candidate before deletion.
//! 4. **Deletes** stale files asynchronously using bounded concurrency to
//!    avoid overwhelming the kernel I/O scheduler.
//! 5. **Reports** a structured [`PruneSummary`] after every pass so callers
//!    can emit Prometheus metrics, structured log events, or simply log the
//!    result.
//!
//! # Safety Invariants
//!
//! * The pruner **never** deletes files that are newer than
//!   [`PrunerConfig::retention_window`].
//! * Files held open by another process ([`crate::maintenance::fs::is_file_open`])
//!   are skipped unconditionally.
//! * The pruner honours the [`crate::maintenance::fs::DryRun`] flag; in
//!   dry-run mode (the default) no filesystem mutations occur.
//! * A minimum of [`PrunerConfig::min_retained_files`] files is always
//!   preserved, counting from the most recently modified down, to protect
//!   in-progress catchup operations.
//!
//! # Configuration
//!
//! All fields can be set via environment variables so that the daemon can be
//! driven from Kubernetes ConfigMaps without recompilation.
//!
//! | Environment variable              | Default                        | Description                                        |
//! |-----------------------------------|--------------------------------|----------------------------------------------------|
//! | `PRUNER_BUCKET_DIR`               | `/var/lib/stellar/buckets`     | Root directory to scan for prunable files          |
//! | `PRUNER_RETENTION_DAYS`           | `30`                           | Retain files modified within this many days        |
//! | `PRUNER_INTERVAL_SECS`            | `3600`                         | Seconds between daemon passes                      |
//! | `PRUNER_DRY_RUN`                  | `true`                         | Set to `false` to enable actual deletions          |
//! | `PRUNER_MIN_RETAINED_FILES`       | `50`                           | Minimum number of newest files to keep regardless  |
//! | `PRUNER_CONCURRENCY`              | `8`                            | Parallel delete tasks per pass                     |
//! | `PRUNER_STELLAR_CORE_CFG`         | `/etc/stellar/stellar-core.cfg`| Path to the Stellar Core TOML config               |
//!
//! # Example – spawning the daemon
//!
//! ```rust,no_run
//! use controller::maintenance::pruner::{PrunerConfig, ArchivePruner};
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = PrunerConfig::from_env();
//!     let pruner = ArchivePruner::new(config);
//!     pruner.run().await;
//! }
//! ```

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use tokio::sync::Semaphore;
use tracing::{debug, error, info, warn};

use crate::maintenance::fs::{delete_file, scan_directory, DryRun, FileEntry};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default directory containing Stellar Core bucket and XDR files.
pub const DEFAULT_BUCKET_DIR: &str = "/var/lib/stellar/buckets";

/// Default retention window: 30 days.
pub const DEFAULT_RETENTION_DAYS: u64 = 30;

/// Default daemon loop interval: 1 hour.
pub const DEFAULT_INTERVAL_SECS: u64 = 3600;

/// Default minimum number of files to retain, regardless of age.
/// This guards against deleting files that stellar-core is actively
/// using for a catchup operation.
pub const DEFAULT_MIN_RETAINED_FILES: usize = 50;

/// Default maximum parallel delete tasks.
pub const DEFAULT_CONCURRENCY: usize = 8;

/// Default Stellar Core configuration file path.
pub const DEFAULT_STELLAR_CORE_CFG: &str = "/etc/stellar/stellar-core.cfg";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`ArchivePruner`].
#[derive(Debug, Clone)]
pub struct PrunerConfig {
    /// Root directory to walk for XDR/bucket files.
    pub bucket_dir: PathBuf,
    /// Age threshold: files older than this window are candidates for deletion.
    pub retention_window: Duration,
    /// Interval between daemon passes.
    pub interval: Duration,
    /// Whether to perform actual deletions or only log what would be deleted.
    pub dry_run: DryRun,
    /// Minimum number of files to retain, measured from the most recently
    /// modified file backwards.  Provides a safety buffer for in-progress
    /// catchup operations.
    pub min_retained_files: usize,
    /// Maximum number of concurrent delete operations per pass.
    pub concurrency: usize,
    /// Path to the Stellar Core TOML configuration file.  When present and
    /// readable, the pruner reads the `MINIMUM_STORED_LEDGERS` key to
    /// optionally extend the retention window.
    pub stellar_core_cfg: PathBuf,
}

impl Default for PrunerConfig {
    fn default() -> Self {
        Self {
            bucket_dir: PathBuf::from(DEFAULT_BUCKET_DIR),
            retention_window: Duration::from_secs(DEFAULT_RETENTION_DAYS * 24 * 3600),
            interval: Duration::from_secs(DEFAULT_INTERVAL_SECS),
            dry_run: DryRun::Yes,
            min_retained_files: DEFAULT_MIN_RETAINED_FILES,
            concurrency: DEFAULT_CONCURRENCY,
            stellar_core_cfg: PathBuf::from(DEFAULT_STELLAR_CORE_CFG),
        }
    }
}

impl PrunerConfig {
    /// Build a `PrunerConfig` from environment variables, falling back to
    /// defaults for any variable that is absent or unparseable.
    pub fn from_env() -> Self {
        let bucket_dir = std::env::var("PRUNER_BUCKET_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(DEFAULT_BUCKET_DIR));

        let retention_days: u64 = std::env::var("PRUNER_RETENTION_DAYS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_RETENTION_DAYS);

        let interval_secs: u64 = std::env::var("PRUNER_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_INTERVAL_SECS);

        let dry_run_str = std::env::var("PRUNER_DRY_RUN").unwrap_or_else(|_| "true".to_string());
        let dry_run = if dry_run_str.eq_ignore_ascii_case("false") {
            DryRun::No
        } else {
            DryRun::Yes
        };

        let min_retained_files: usize = std::env::var("PRUNER_MIN_RETAINED_FILES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_MIN_RETAINED_FILES);

        let concurrency: usize = std::env::var("PRUNER_CONCURRENCY")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_CONCURRENCY);

        let stellar_core_cfg = std::env::var("PRUNER_STELLAR_CORE_CFG")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(DEFAULT_STELLAR_CORE_CFG));

        Self {
            bucket_dir,
            retention_window: Duration::from_secs(retention_days * 24 * 3600),
            interval: Duration::from_secs(interval_secs),
            dry_run,
            min_retained_files,
            concurrency,
            stellar_core_cfg,
        }
    }
}

// ---------------------------------------------------------------------------
// Summary types
// ---------------------------------------------------------------------------

/// Result of a single pruner pass.
#[derive(Debug, Default, Clone)]
pub struct PruneSummary {
    /// Total files discovered in the scan.
    pub files_scanned: usize,
    /// Total bytes across all scanned files.
    pub bytes_scanned: u64,
    /// Files that were candidates for deletion (met age + name criteria).
    pub files_eligible: usize,
    /// Bytes across eligible files.
    pub bytes_eligible: u64,
    /// Files actually deleted (0 in dry-run mode or when skipped due to lock).
    pub files_deleted: usize,
    /// Bytes actually freed.
    pub bytes_freed: u64,
    /// Files skipped because they are currently held open.
    pub files_skipped_open: usize,
    /// Files skipped because of the `min_retained_files` safety guard.
    pub files_skipped_min_guard: usize,
    /// Files that could not be stat-ed during scan.
    pub stat_errors: usize,
    /// Files for which deletion failed.
    pub delete_errors: usize,
    /// Whether this pass ran in dry-run mode.
    pub dry_run: bool,
    /// Effective retention window used in this pass.
    pub retention_window: Duration,
}

// ---------------------------------------------------------------------------
// Stellar Core config reader
// ---------------------------------------------------------------------------

/// Attempt to read `MINIMUM_STORED_LEDGERS` from a Stellar Core TOML config.
///
/// Returns `None` when the file is absent, unreadable, or does not contain
/// the key.  The caller should then fall back to the configured default.
///
/// The key is expected at the top level:
/// ```toml
/// MINIMUM_STORED_LEDGERS=2000000
/// ```
pub fn read_minimum_stored_ledgers(cfg_path: &std::path::Path) -> Option<u64> {
    let content = std::fs::read_to_string(cfg_path).ok()?;
    for line in content.lines() {
        let line = line.trim();
        // Stellar Core config uses `KEY=VALUE` (no spaces around `=` in
        // canonical form, but we tolerate spaces for robustness).
        if let Some(rest) = line
            .strip_prefix("MINIMUM_STORED_LEDGERS")
            .map(|r| r.trim())
        {
            if let Some(value_str) = rest.strip_prefix('=').map(|r| r.trim()) {
                return value_str.parse::<u64>().ok();
            }
        }
    }
    None
}

/// Convert a ledger count to an approximate retention `Duration`.
///
/// On Stellar mainnet a new ledger closes approximately every 5 seconds.
/// Using this constant gives a conservative lower bound on the age of a file
/// that stellar-core could still need for a full catchup from the oldest
/// retained ledger.
pub fn ledger_count_to_duration(ledger_count: u64) -> Duration {
    const SECONDS_PER_LEDGER: u64 = 5;
    Duration::from_secs(ledger_count * SECONDS_PER_LEDGER)
}

// ---------------------------------------------------------------------------
// Pruner
// ---------------------------------------------------------------------------

/// The History Archive Pruning Daemon.
pub struct ArchivePruner {
    config: PrunerConfig,
}

impl ArchivePruner {
    /// Create a new pruner with the supplied configuration.
    pub fn new(config: PrunerConfig) -> Self {
        Self { config }
    }

    /// Run the daemon loop indefinitely, performing one pruner pass every
    /// [`PrunerConfig::interval`].
    ///
    /// All errors encountered during a pass are logged at `ERROR` level but
    /// do **not** stop the loop – the daemon recovers and retries on the next
    /// interval.
    pub async fn run(&self) {
        info!(
            bucket_dir = %self.config.bucket_dir.display(),
            retention_days = self.config.retention_window.as_secs() / 86400,
            interval_secs = self.config.interval.as_secs(),
            dry_run = matches!(self.config.dry_run, DryRun::Yes),
            "Starting History Archive Pruning Daemon"
        );

        loop {
            match self.run_once().await {
                Ok(summary) => {
                    info!(
                        files_scanned = summary.files_scanned,
                        bytes_scanned = summary.bytes_scanned,
                        files_eligible = summary.files_eligible,
                        bytes_eligible = summary.bytes_eligible,
                        files_deleted = summary.files_deleted,
                        bytes_freed = summary.bytes_freed,
                        files_skipped_open = summary.files_skipped_open,
                        files_skipped_min_guard = summary.files_skipped_min_guard,
                        stat_errors = summary.stat_errors,
                        delete_errors = summary.delete_errors,
                        dry_run = summary.dry_run,
                        "Pruner pass complete"
                    );
                }
                Err(e) => {
                    error!(error = %e, "Pruner pass failed");
                }
            }

            tokio::time::sleep(self.config.interval).await;
        }
    }

    /// Execute a single pruner pass and return a structured summary.
    ///
    /// This method is `pub` so it can be called directly in integration tests
    /// without waiting for the interval timer.
    pub async fn run_once(&self) -> Result<PruneSummary> {
        // ------------------------------------------------------------------
        // Step 1: determine the effective retention window.
        // ------------------------------------------------------------------
        let retention_window = self.effective_retention_window();
        debug!(
            retention_secs = retention_window.as_secs(),
            "Effective retention window"
        );

        // ------------------------------------------------------------------
        // Step 2: scan the bucket directory.
        // ------------------------------------------------------------------
        let now = SystemTime::now();
        let scan = scan_directory(&self.config.bucket_dir, retention_window, now)
            .await
            .with_context(|| {
                format!(
                    "Failed to scan bucket directory: {}",
                    self.config.bucket_dir.display()
                )
            })?;

        // Save aggregate fields before consuming scan.entries.
        let bytes_scanned = scan.total_bytes;
        let scan_stat_errors = scan.stat_errors;

        // ------------------------------------------------------------------
        // Step 3: apply the min-retained-files safety guard.
        // ------------------------------------------------------------------
        // Sort all files by mtime descending (newest first) and mark the
        // top-N as protected, regardless of whether they are otherwise
        // eligible for pruning.
        let mut all_files = scan.entries;
        all_files.sort_by(|a, b| b.modified.cmp(&a.modified));

        let protected_count = self.config.min_retained_files.min(all_files.len());
        for entry in all_files.iter_mut().take(protected_count) {
            entry.is_prunable = false;
        }

        let files_skipped_min_guard = all_files
            .iter()
            .take(protected_count)
            .filter(|e| {
                // Count only those that *would* have been prunable before the guard.
                let age_ok =
                    is_older_than_retention(e.modified, now, retention_window);
                crate::maintenance::fs::is_prunable_name(&e.path) && age_ok
            })
            .count();

        // Collect candidates for deletion.
        let candidates: Vec<FileEntry> = all_files.into_iter().filter(|e| e.is_prunable).collect();

        let files_eligible = candidates.len();
        let bytes_eligible: u64 = candidates.iter().map(|e| e.size_bytes).sum();

        // ------------------------------------------------------------------
        // Step 4: delete eligible files with bounded concurrency.
        // ------------------------------------------------------------------
        let semaphore = Arc::new(Semaphore::new(self.config.concurrency));
        let dry_run = self.config.dry_run;

        let mut tasks = Vec::with_capacity(candidates.len());
        for entry in candidates {
            let sem = Arc::clone(&semaphore);
            let task = tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.expect("semaphore closed");
                let path = entry.path.clone();
                let size = entry.size_bytes;
                match delete_file(&path, size, dry_run).await {
                    Ok(freed) => (freed, false, false),   // (bytes_freed, skipped_open, error)
                    Err(e) => {
                        // Distinguish "skipped because open" from a real error.
                        // delete_file returns Ok(0) for open files, so any Err
                        // here is a genuine I/O failure.
                        warn!(path = %path.display(), error = %e, "Failed to delete file");
                        (0u64, false, true)
                    }
                }
            });
            tasks.push(task);
        }

        let mut bytes_freed: u64 = 0;
        let mut files_deleted: usize = 0;
        let mut files_skipped_open: usize = 0;
        let mut delete_errors: usize = 0;

        for task in tasks {
            match task.await {
                Ok((freed, skipped_open, had_error)) => {
                    if had_error {
                        delete_errors += 1;
                    } else if skipped_open {
                        files_skipped_open += 1;
                    } else if freed > 0 {
                        bytes_freed += freed;
                        files_deleted += 1;
                    }
                    // freed == 0 && !had_error && !skipped_open means dry-run
                }
                Err(e) => {
                    error!(error = %e, "Pruner task panicked");
                    delete_errors += 1;
                }
            }
        }

        Ok(PruneSummary {
            files_scanned: files_eligible + protected_count + scan_stat_errors,
            bytes_scanned,
            files_eligible,
            bytes_eligible,
            files_deleted,
            bytes_freed,
            files_skipped_open,
            files_skipped_min_guard,
            stat_errors: scan_stat_errors,
            delete_errors,
            dry_run: matches!(dry_run, DryRun::Yes),
            retention_window,
        })
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Compute the effective retention window, optionally extending the
    /// configured default if the Stellar Core config specifies a larger
    /// `MINIMUM_STORED_LEDGERS` value.
    fn effective_retention_window(&self) -> Duration {
        let configured = self.config.retention_window;

        if let Some(min_ledgers) =
            read_minimum_stored_ledgers(&self.config.stellar_core_cfg)
        {
            let core_window = ledger_count_to_duration(min_ledgers);
            if core_window > configured {
                info!(
                    configured_days = configured.as_secs() / 86400,
                    core_ledgers = min_ledgers,
                    core_days = core_window.as_secs() / 86400,
                    "Extending retention window to honour MINIMUM_STORED_LEDGERS"
                );
                return core_window;
            }
        }

        configured
    }
}

// ---------------------------------------------------------------------------
// Private utility
// ---------------------------------------------------------------------------

fn is_older_than_retention(modified: SystemTime, now: SystemTime, retention: Duration) -> bool {
    crate::maintenance::fs::is_older_than(modified, now, retention)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::time::Duration;
    use tempfile::TempDir;
    use tokio::fs as tfs;

    // -----------------------------------------------------------------------
    // read_minimum_stored_ledgers
    // -----------------------------------------------------------------------

    #[test]
    fn parses_minimum_stored_ledgers_from_cfg() {
        let dir = TempDir::new().unwrap();
        let cfg = dir.path().join("stellar-core.cfg");
        std::fs::write(&cfg, "MINIMUM_STORED_LEDGERS=2000000\n").unwrap();
        assert_eq!(read_minimum_stored_ledgers(&cfg), Some(2_000_000));
    }

    #[test]
    fn returns_none_when_key_absent() {
        let dir = TempDir::new().unwrap();
        let cfg = dir.path().join("stellar-core.cfg");
        std::fs::write(&cfg, "NETWORK_PASSPHRASE=\"Public Global Stellar Network ; September 2015\"\n").unwrap();
        assert_eq!(read_minimum_stored_ledgers(&cfg), None);
    }

    #[test]
    fn returns_none_for_nonexistent_file() {
        assert_eq!(
            read_minimum_stored_ledgers(Path::new("/no/such/file.cfg")),
            None
        );
    }

    #[test]
    fn tolerates_whitespace_around_equals() {
        let dir = TempDir::new().unwrap();
        let cfg = dir.path().join("stellar-core.cfg");
        std::fs::write(&cfg, "MINIMUM_STORED_LEDGERS = 1000000\n").unwrap();
        assert_eq!(read_minimum_stored_ledgers(&cfg), Some(1_000_000));
    }

    // -----------------------------------------------------------------------
    // ledger_count_to_duration
    // -----------------------------------------------------------------------

    #[test]
    fn converts_ledger_count_to_duration() {
        // 1 day ≈ 17280 ledgers (86400 / 5)
        let duration = ledger_count_to_duration(17280);
        assert_eq!(duration.as_secs(), 86400);
    }

    // -----------------------------------------------------------------------
    // PrunerConfig::from_env
    // -----------------------------------------------------------------------

    #[test]
    fn config_defaults_are_sane() {
        // Without any env vars set the defaults should be safe.
        let cfg = PrunerConfig::default();
        assert_eq!(cfg.dry_run, DryRun::Yes);
        assert_eq!(
            cfg.retention_window,
            Duration::from_secs(30 * 24 * 3600)
        );
        assert_eq!(cfg.min_retained_files, DEFAULT_MIN_RETAINED_FILES);
        assert_eq!(cfg.concurrency, DEFAULT_CONCURRENCY);
    }

    // -----------------------------------------------------------------------
    // ArchivePruner::run_once – integration-style test on temp dir
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn dry_run_does_not_delete_eligible_files() {
        let dir = TempDir::new().unwrap();
        let old_file = dir.path().join("old.xdr");
        tfs::write(&old_file, b"old-data").await.unwrap();

        // Back-date to 40 days ago.
        let forty_days_ago = SystemTime::now() - Duration::from_secs(40 * 24 * 3600);
        filetime::set_file_mtime(
            &old_file,
            filetime::FileTime::from_system_time(forty_days_ago),
        )
        .unwrap();

        let config = PrunerConfig {
            bucket_dir: dir.path().to_path_buf(),
            retention_window: Duration::from_secs(30 * 24 * 3600),
            interval: Duration::from_secs(1),
            dry_run: DryRun::Yes,
            min_retained_files: 0, // disable guard for this test
            concurrency: 1,
            stellar_core_cfg: PathBuf::from("/nonexistent/stellar-core.cfg"),
        };

        let pruner = ArchivePruner::new(config);
        let summary = pruner.run_once().await.unwrap();

        assert!(summary.dry_run);
        assert_eq!(summary.files_eligible, 1);
        assert_eq!(summary.files_deleted, 0, "dry-run must not delete files");
        assert!(old_file.exists(), "file must still exist after dry-run pass");
    }

    #[tokio::test]
    async fn real_run_deletes_eligible_files() {
        let dir = TempDir::new().unwrap();
        let old_file = dir.path().join("old.xdr");
        tfs::write(&old_file, b"old-data").await.unwrap();

        let forty_days_ago = SystemTime::now() - Duration::from_secs(40 * 24 * 3600);
        filetime::set_file_mtime(
            &old_file,
            filetime::FileTime::from_system_time(forty_days_ago),
        )
        .unwrap();

        let config = PrunerConfig {
            bucket_dir: dir.path().to_path_buf(),
            retention_window: Duration::from_secs(30 * 24 * 3600),
            interval: Duration::from_secs(1),
            dry_run: DryRun::No,
            min_retained_files: 0,
            concurrency: 1,
            stellar_core_cfg: PathBuf::from("/nonexistent/stellar-core.cfg"),
        };

        let pruner = ArchivePruner::new(config);
        let summary = pruner.run_once().await.unwrap();

        assert!(!summary.dry_run);
        assert_eq!(summary.files_eligible, 1);
        assert_eq!(summary.files_deleted, 1);
        assert!(summary.bytes_freed > 0);
        assert!(!old_file.exists(), "file should be deleted after real run");
    }

    #[tokio::test]
    async fn min_retained_files_guard_protects_newest_files() {
        let dir = TempDir::new().unwrap();

        // Create 5 old XDR files.
        for i in 0..5u8 {
            let path = dir.path().join(format!("ledger-{i:03}.xdr"));
            tfs::write(&path, b"data").await.unwrap();
            let age = SystemTime::now() - Duration::from_secs((40 + u64::from(i)) * 24 * 3600);
            filetime::set_file_mtime(
                &path,
                filetime::FileTime::from_system_time(age),
            )
            .unwrap();
        }

        let config = PrunerConfig {
            bucket_dir: dir.path().to_path_buf(),
            retention_window: Duration::from_secs(30 * 24 * 3600),
            interval: Duration::from_secs(1),
            dry_run: DryRun::No,
            min_retained_files: 3, // protect the 3 newest
            concurrency: 4,
            stellar_core_cfg: PathBuf::from("/nonexistent/stellar-core.cfg"),
        };

        let pruner = ArchivePruner::new(config);
        let summary = pruner.run_once().await.unwrap();

        // 5 files total, 3 protected → only 2 eligible → 2 deleted.
        assert_eq!(summary.files_deleted, 2);
        assert_eq!(summary.files_skipped_min_guard, 3);
    }

    #[tokio::test]
    async fn non_xdr_files_are_never_pruned() {
        let dir = TempDir::new().unwrap();
        let cfg_file = dir.path().join("stellar-core.cfg");
        tfs::write(&cfg_file, b"config data").await.unwrap();

        let forty_days_ago = SystemTime::now() - Duration::from_secs(40 * 24 * 3600);
        filetime::set_file_mtime(
            &cfg_file,
            filetime::FileTime::from_system_time(forty_days_ago),
        )
        .unwrap();

        let config = PrunerConfig {
            bucket_dir: dir.path().to_path_buf(),
            retention_window: Duration::from_secs(30 * 24 * 3600),
            interval: Duration::from_secs(1),
            dry_run: DryRun::No,
            min_retained_files: 0,
            concurrency: 1,
            stellar_core_cfg: PathBuf::from("/nonexistent/stellar-core.cfg"),
        };

        let pruner = ArchivePruner::new(config);
        let summary = pruner.run_once().await.unwrap();

        assert_eq!(summary.files_deleted, 0);
        assert!(cfg_file.exists(), "config file must not be deleted");
    }

    #[tokio::test]
    async fn stellar_core_cfg_extends_retention_window() {
        let dir = TempDir::new().unwrap();
        // Write a Stellar Core config requesting a very long retention.
        let cfg_path = dir.path().join("stellar-core.cfg");
        // 10 million ledgers * 5 s ≈ 578 days
        std::fs::write(&cfg_path, "MINIMUM_STORED_LEDGERS=10000000\n").unwrap();

        // Write a file that is 40 days old – within the extended window.
        let old_file = dir.path().join("ledger.xdr");
        tfs::write(&old_file, b"data").await.unwrap();
        let forty_days_ago = SystemTime::now() - Duration::from_secs(40 * 24 * 3600);
        filetime::set_file_mtime(
            &old_file,
            filetime::FileTime::from_system_time(forty_days_ago),
        )
        .unwrap();

        let config = PrunerConfig {
            bucket_dir: dir.path().to_path_buf(),
            retention_window: Duration::from_secs(30 * 24 * 3600), // 30 days
            interval: Duration::from_secs(1),
            dry_run: DryRun::No,
            min_retained_files: 0,
            concurrency: 1,
            stellar_core_cfg: cfg_path,
        };

        let pruner = ArchivePruner::new(config);
        let summary = pruner.run_once().await.unwrap();

        // The file is 40 days old but the effective window is ~578 days,
        // so it must NOT be deleted.
        assert_eq!(
            summary.files_deleted, 0,
            "file must be protected by extended retention from stellar-core.cfg"
        );
        assert!(old_file.exists());
    }
}
