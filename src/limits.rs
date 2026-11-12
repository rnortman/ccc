//! The exhausted flags of each subscription profile on this host: one flag
//! per bucket, a named quota of the account, raised when the provider refuses
//! a turn and expiring at the reset time the provider gave.
//!
//! One file per profile, `<state home>/ccc/limits/<profile>.json`, read and
//! rewritten under an exclusive lock on the sibling `<profile>.lock`. The file
//! is authoritative for this host; it is published as the profile's account
//! record (period [`PERIOD`], `usd` 0) so every other host reads it too.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::Result;
use claude_usage_mqtt_core::{exhausted, exhausted_buckets, AccountKind, AccountSpend, Exhausted};
use jiff::Timestamp;
use serde::{Deserialize, Serialize};

use crate::budget::{clear_point, others, Identity};
use crate::bus::Bus;
use crate::config::EnvSource;
use crate::fsutil::{ensure_private_dir, load_json, write_atomically, FileLock};

const LIMITS_VERSION: u32 = 1;

/// The fixed period of a subscription account's record.
pub const PERIOD: &str = "-";

/// The 5-hour window, every model.
pub const FIVE_HOUR: &str = "five_hour";
/// The weekly quota every model draws on.
pub const GENERAL: &str = "general";
/// The weekly Fable quota.
pub const FABLE: &str = "fable";

/// Every bucket a profile flag can name.
pub const BUCKETS: [&str; 3] = [FIVE_HOUR, GENERAL, FABLE];

/// The bucket a refusal exhausts, from the provider's name for the limit and
/// whether the refused turn's model is a Fable model. A kind that names no
/// known window is charged to the weekly quota the model draws on last.
pub fn bucket_for(kind: Option<&str>, fable_model: bool) -> &'static str {
    match kind {
        Some("five_hour") => FIVE_HOUR,
        Some("seven_day") => GENERAL,
        _ if fable_model => FABLE,
        _ => GENERAL,
    }
}

/// This host's flags on one profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LimitsFile {
    pub v: u32,
    /// The flag of each bucket this host saw refused, by bucket name.
    #[serde(default)]
    pub buckets: BTreeMap<String, Exhausted>,
    /// The last time an operator cleared the profile's flags on this host.
    pub cleared_at: Option<Timestamp>,
    pub updated_at: Timestamp,
}

impl LimitsFile {
    fn empty(now: Timestamp) -> LimitsFile {
        LimitsFile {
            v: LIMITS_VERSION,
            buckets: BTreeMap::new(),
            cleared_at: None,
            updated_at: now,
        }
    }

    /// Drops the flags that are expired at `now` or lifted by `cleared_at`.
    fn prune(&mut self, now: Timestamp) {
        let cleared = self.cleared_at;
        self.buckets
            .retain(|_, e| e.until.is_none_or(|u| u > now) && cleared.is_none_or(|c| c < e.at));
    }
}

/// The limits directory and the operations on the files in it.
#[derive(Debug, Clone)]
pub struct Limits {
    dir: PathBuf,
}

impl Limits {
    /// `$XDG_STATE_HOME/ccc/limits`, or `~/.local/state/ccc/limits`.
    pub fn in_env(env: &dyn EnvSource) -> Limits {
        Limits::at(crate::config::state_home_in(env).join("ccc/limits"))
    }

    pub fn at(dir: PathBuf) -> Limits {
        Limits { dir }
    }

    fn path(&self, profile: &str) -> PathBuf {
        self.dir.join(format!("{profile}.json"))
    }

    fn lock(&self, profile: &str) -> Result<FileLock> {
        ensure_private_dir(&self.dir)?;
        FileLock::lock(&self.dir.join(format!("{profile}.lock")), "limits lock")
    }

    /// The file's flags, or `None` when there is no file.
    fn load(path: &Path) -> Result<Option<LimitsFile>> {
        load_json(path, "limits file")
    }

    /// `profile`'s flags on this host, or why they could not be read.
    pub fn try_read(&self, profile: &str, now: Timestamp) -> Result<LimitsFile> {
        Ok(Self::load(&self.path(profile))?.unwrap_or_else(|| LimitsFile::empty(now)))
    }

    /// `profile`'s flags on this host. A file that cannot be read counts as
    /// no flags, with a warning on stderr: a flag is advice, and a damaged
    /// file must not take a profile out of use.
    pub fn read(&self, profile: &str, now: Timestamp) -> LimitsFile {
        self.try_read(profile, now).unwrap_or_else(|e| {
            crate::harness::diag(format_args!(
                "warning: the exhausted flags of profile {profile} are ignored: {e:#}"
            ));
            LimitsFile::empty(now)
        })
    }

    /// Reads `profile`'s file under its lock, replacing a damaged one,
    /// applies `change`, drops expired and lifted flags, stamps `updated_at`
    /// and writes it back.
    fn update(
        &self,
        profile: &str,
        now: Timestamp,
        change: impl FnOnce(&mut LimitsFile),
    ) -> Result<LimitsFile> {
        let _lock = self.lock(profile)?;
        let path = self.path(profile);
        let mut file = Self::load(&path)
            .ok()
            .flatten()
            .unwrap_or_else(|| LimitsFile::empty(now));
        change(&mut file);
        file.prune(now);
        file.updated_at = now;
        let mut text = serde_json::to_string_pretty(&file)?;
        text.push('\n');
        write_atomically(&path, &text)?;
        Ok(file)
    }

    /// Records `flag` as `bucket`'s, replacing any flag the bucket had.
    pub fn raise(
        &self,
        profile: &str,
        bucket: &str,
        flag: Exhausted,
        now: Timestamp,
    ) -> Result<LimitsFile> {
        self.update(profile, now, |file| {
            file.buckets.insert(bucket.to_string(), flag);
        })
    }

    /// Clears every flag of `profile` on this host: `cleared_at` is now, or
    /// one second past the newest flag in `remote` when that flag is ahead of
    /// this host's clock. Returns the file and, in that second case, a note.
    pub fn clear(
        &self,
        profile: &str,
        now: Timestamp,
        remote: &[AccountSpend],
    ) -> Result<(LimitsFile, Option<String>)> {
        let (at, note) = clear_point(
            remote
                .iter()
                .filter(|r| r.account == profile && r.period == PERIOD)
                .flat_map(|r| r.exhausted.iter().chain(r.exhausted_buckets.values()))
                .map(|e| e.at),
            now,
        );
        let file = self.update(profile, now, |file| file.cleared_at = Some(at))?;
        Ok((file, note))
    }
}

/// This host's flags on `profile` as its account record in the agent tree.
// TODO(subscription-usage-data-unpublished): the record carries the flags
// only; what else of a subscription account `ccc` sees is unpublished.
pub fn to_account_spend(profile: &str, file: &LimitsFile, me: &Identity) -> AccountSpend {
    let mut spend = AccountSpend::new(
        profile,
        AccountKind::Subscription,
        me.publisher.clone(),
        me.host.clone(),
        PERIOD,
        0.0,
        file.updated_at,
    );
    spend.exhausted_buckets = file.buckets.clone();
    spend.cleared_at = file.cleared_at;
    spend
}

/// Publishes `file` as this host's record for `profile`; nothing when the
/// bus is off.
pub fn publish(bus: &dyn Bus, profile: &str, file: &LimitsFile) {
    if let Some(me) = bus.identity() {
        bus.put_account(to_account_spend(profile, file, me));
    }
}

/// The flags in force on `profile` at `now`, by bucket, from this host's
/// `file` and, when they could be read, every other publisher's and host's
/// records (`remote`). This host's own retained record is ignored: the file
/// is authoritative for it. An account-wide flag counts for every bucket.
/// Every bucket of [`BUCKETS`] is resolved, and so is any other bucket a
/// record flags, so a flag under a name selection does not use still shows.
pub fn in_force(
    profile: &str,
    file: &LimitsFile,
    remote: Option<(&[AccountSpend], &Identity)>,
    now: Timestamp,
) -> BTreeMap<String, Exhausted> {
    let anonymous = Identity {
        publisher: String::new(),
        host: String::new(),
    };
    let me = remote.map_or(&anonymous, |(_, me)| me);
    let mut records = vec![to_account_spend(profile, file, me)];
    if let Some((remote, me)) = remote {
        records.extend(others(remote, profile, PERIOD, me).cloned());
    }
    let mut buckets: BTreeSet<&str> = BUCKETS.into_iter().collect();
    buckets.extend(exhausted_buckets(&records, profile, PERIOD, now).into_keys());
    buckets
        .into_iter()
        .filter_map(|b| {
            exhausted(&records, profile, PERIOD, Some(b), now).map(|e| (b.to_string(), e.clone()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(s: &str) -> Timestamp {
        s.parse().unwrap()
    }

    const NOW: &str = "2026-11-11T13:46:07Z";

    fn me() -> Identity {
        Identity {
            publisher: "ccc".into(),
            host: "atlas".into(),
        }
    }

    fn flag(at: &str, until: Option<&str>) -> Exhausted {
        let mut e = Exhausted::new(ts(at), Some("inv".into()), Some("refused".into()));
        e.until = until.map(ts);
        e
    }

    fn record(publisher: &str, host: &str) -> AccountSpend {
        AccountSpend::new(
            "team01",
            AccountKind::Subscription,
            publisher,
            host,
            PERIOD,
            0.0,
            ts(NOW),
        )
    }

    #[test]
    fn the_bucket_table_row_by_row() {
        assert_eq!(bucket_for(Some("five_hour"), true), FIVE_HOUR);
        assert_eq!(bucket_for(Some("five_hour"), false), FIVE_HOUR);
        assert_eq!(bucket_for(Some("seven_day"), true), GENERAL);
        assert_eq!(bucket_for(Some("seven_day"), false), GENERAL);
        assert_eq!(bucket_for(Some("seven_day_overage_included"), true), FABLE);
        assert_eq!(
            bucket_for(Some("seven_day_overage_included"), false),
            GENERAL
        );
        assert_eq!(bucket_for(None, true), FABLE);
        assert_eq!(bucket_for(None, false), GENERAL);
    }

    #[test]
    fn raise_writes_the_file_privately_and_read_returns_it() {
        let dir = crate::testutil::TempDir::new("limits");
        let l = Limits::at(dir.path().join("limits"));
        let mut e = flag(NOW, Some("2026-11-13T04:00:00Z"));
        e.kind = Some("seven_day_overage_included".into());
        let file = l.raise("team01", FABLE, e.clone(), ts(NOW)).unwrap();
        assert_eq!(file.buckets.get(FABLE), Some(&e));
        assert_eq!(file.updated_at, ts(NOW));
        assert_eq!(l.read("team01", ts(NOW)), file);
        let text = std::fs::read_to_string(dir.path().join("limits/team01.json")).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["v"], 1);
        assert_eq!(json["buckets"]["fable"]["until"], "2026-11-13T04:00:00Z");
        assert_eq!(
            json["buckets"]["fable"]["kind"],
            "seven_day_overage_included"
        );
        assert!(json["cleared_at"].is_null());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&dir.path().join("limits")), 0o700);
            assert_eq!(mode(&dir.path().join("limits/team01.json")), 0o600);
        }
    }

    #[test]
    fn expired_and_lifted_flags_are_dropped_on_write() {
        let dir = crate::testutil::TempDir::new("limits");
        let l = Limits::at(dir.path().to_path_buf());
        l.raise(
            "team01",
            FIVE_HOUR,
            flag("2026-11-11T10:00:00Z", Some("2026-11-11T12:00:00Z")),
            ts("2026-11-11T10:00:00Z"),
        )
        .unwrap();
        l.raise(
            "team01",
            GENERAL,
            flag("2026-11-11T10:00:00Z", Some("2026-11-14T00:00:00Z")),
            ts("2026-11-11T10:00:00Z"),
        )
        .unwrap();
        // The five-hour flag expired at 12:00; the general one is in force.
        let file = l
            .raise(
                "team01",
                FABLE,
                flag(NOW, Some("2026-11-13T04:00:00Z")),
                ts(NOW),
            )
            .unwrap();
        assert_eq!(
            file.buckets.keys().map(String::as_str).collect::<Vec<_>>(),
            vec![FABLE, GENERAL]
        );
        // A clear lifts both; a flag raised after it stays.
        let (file, note) = l.clear("team01", ts(NOW), &[]).unwrap();
        assert!(file.buckets.is_empty() && note.is_none());
        assert_eq!(file.cleared_at, Some(ts(NOW)));
        let later = "2026-11-11T14:00:00Z";
        let file = l
            .raise(
                "team01",
                FABLE,
                flag(later, Some("2026-11-13T04:00:00Z")),
                ts(later),
            )
            .unwrap();
        assert_eq!(file.buckets.len(), 1);
        assert_eq!(file.cleared_at, Some(ts(NOW)));
    }

    #[test]
    fn a_damaged_file_reads_as_no_flags_and_is_replaced_by_a_raise() {
        let dir = crate::testutil::TempDir::new("limits");
        let l = Limits::at(dir.path().to_path_buf());
        std::fs::write(dir.path().join("team01.json"), "{not json").unwrap();
        let err = l.try_read("team01", ts(NOW)).unwrap_err();
        assert!(format!("{err:#}").contains("damaged"), "{err:#}");
        assert!(l.read("team01", ts(NOW)).buckets.is_empty());
        let file = l
            .raise(
                "team01",
                GENERAL,
                flag(NOW, Some("2026-11-14T00:00:00Z")),
                ts(NOW),
            )
            .unwrap();
        assert_eq!(l.try_read("team01", ts(NOW)).unwrap(), file);
    }

    #[test]
    fn a_clear_steps_past_a_remote_flag_ahead_of_the_clock() {
        let dir = crate::testutil::TempDir::new("limits");
        let l = Limits::at(dir.path().to_path_buf());
        let mut ahead = record("ccc", "zeus");
        ahead
            .exhausted_buckets
            .insert(FABLE.into(), flag("2026-11-11T13:50:00Z", None));
        let (file, note) = l.clear("team01", ts(NOW), &[ahead]).unwrap();
        assert_eq!(file.cleared_at, Some(ts("2026-11-11T13:50:01Z")));
        assert!(note.unwrap().contains("ahead of this host's clock"));
    }

    #[test]
    fn the_file_maps_to_a_subscription_account_record() {
        let mut file = LimitsFile::empty(ts(NOW));
        file.buckets
            .insert(FABLE.into(), flag(NOW, Some("2026-11-13T04:00:00Z")));
        file.cleared_at = Some(ts("2026-11-10T09:00:00Z"));
        let s = to_account_spend("team01", &file, &me());
        assert_eq!(s.account, "team01");
        assert_eq!(s.account_kind, AccountKind::Subscription);
        assert_eq!((s.publisher.as_str(), s.host.as_str()), ("ccc", "atlas"));
        assert_eq!((s.period.as_str(), s.usd), (PERIOD, 0.0));
        assert_eq!(s.exhausted_buckets, file.buckets);
        assert_eq!(s.exhausted, None);
        assert_eq!((s.turns, s.budget_usd), (None, None));
        assert_eq!(s.cleared_at, file.cleared_at);
        assert_eq!(s.updated_at, file.updated_at);
    }

    #[test]
    fn in_force_merges_other_hosts_and_ignores_this_hosts_own_copy() {
        let now = ts(NOW);
        let mut local = LimitsFile::empty(now);
        local
            .buckets
            .insert(FIVE_HOUR.into(), flag(NOW, Some("2026-11-11T15:00:00Z")));
        let mut own = record("ccc", "atlas");
        own.exhausted_buckets
            .insert(GENERAL.into(), flag(NOW, Some("2026-11-14T00:00:00Z")));
        let mut zeus = record("ccc", "zeus");
        zeus.exhausted_buckets
            .insert(FABLE.into(), flag(NOW, Some("2026-11-13T04:00:00Z")));
        let mut expired = record("brenn", "x");
        expired
            .exhausted_buckets
            .insert(GENERAL.into(), flag(NOW, Some("2026-11-11T13:00:00Z")));
        let remote = vec![own, zeus, expired];

        let flags = in_force("team01", &local, Some((&remote, &me())), now);
        assert_eq!(
            flags.keys().map(String::as_str).collect::<Vec<_>>(),
            vec![FABLE, FIVE_HOUR]
        );
        let alone = in_force("team01", &local, None, now);
        assert_eq!(alone.keys().collect::<Vec<_>>(), vec![FIVE_HOUR]);

        // An account-wide flag counts for every bucket; a newer clear from
        // any host lifts every flag.
        let mut wide = record("brenn", "y");
        wide.exhausted = Some(flag(NOW, None));
        let flags = in_force(
            "team01",
            &LimitsFile::empty(now),
            Some((&[wide], &me())),
            now,
        );
        assert_eq!(flags.len(), 3);
        let mut clear = record("ccc", "zeus");
        clear.cleared_at = Some(ts("2026-11-11T13:47:00Z"));
        let flags = in_force("team01", &local, Some((&[clear], &me())), now);
        assert!(flags.is_empty(), "{flags:?}");
    }

    #[test]
    fn in_force_also_resolves_buckets_of_other_names() {
        let now = ts(NOW);
        let mut zeus = record("brenn", "zeus");
        zeus.exhausted_buckets.insert(
            "seven_day_opus".into(),
            flag(NOW, Some("2026-11-13T04:00:00Z")),
        );
        zeus.exhausted_buckets
            .insert("old".into(), flag(NOW, Some("2026-11-11T13:00:00Z")));
        let flags = in_force(
            "team01",
            &LimitsFile::empty(now),
            Some((&[zeus.clone()], &me())),
            now,
        );
        // The expired one is not in force.
        assert_eq!(flags.keys().collect::<Vec<_>>(), vec!["seven_day_opus"]);

        // An account-wide flag counts for it as for every fixed bucket.
        zeus.exhausted = Some(flag("2026-11-11T13:50:00Z", None));
        let flags = in_force(
            "team01",
            &LimitsFile::empty(now),
            Some((&[zeus], &me())),
            now,
        );
        assert_eq!(
            flags.keys().map(String::as_str).collect::<Vec<_>>(),
            vec![FABLE, FIVE_HOUR, GENERAL, "seven_day_opus"]
        );
        assert!(flags.values().all(|e| e.at == ts("2026-11-11T13:50:00Z")));
    }
}
