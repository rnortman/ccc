//! The month-to-date spend ledger of each API key account on this host, the
//! view selection takes of it, and the `ccc budget` command.
//!
//! One ledger per key, `<state home>/ccc/budget/<name>.json`, read and
//! rewritten under an exclusive lock on the sibling `<name>.lock`. A ledger
//! whose `month` is not the current month in the key's `reset_tz` reads as
//! empty for the current one: that is the monthly refill.
//!
//! `ccc budget` also shows and clears the profiles' exhausted flags, which
//! [`crate::limits`] keeps.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use claude_usage_mqtt_core::{AccountKind, AccountSpend, Exhausted};
use jiff::Timestamp;
use serde::{Deserialize, Serialize};

use crate::bus::{Bus, Remote};
use crate::config::{ApiKeyConfig, Config, EnvSource};
use crate::fsutil::{ensure_private_dir, load_json, write_atomically, FileLock};
use crate::limits::{self, Limits, LimitsFile};

const LEDGER_VERSION: u32 = 1;

/// This host's spend on one key in one month.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ledger {
    pub v: u32,
    /// `YYYY-MM` in the key's `reset_tz`.
    pub month: String,
    /// The sum of the per-result costs Claude Code stated
    /// (`model_usage_since_last_result`) for this host's key turns this month.
    pub usd: f64,
    /// Key turns charged this month, measured or not.
    pub turns: u64,
    /// Key turns that ended with no cost figure; charged nothing.
    pub unmeasured_turns: u64,
    /// The operator's correction from `ccc budget set`.
    pub adjust_usd: f64,
    /// The last time this host saw the key refused for spend.
    pub exhausted: Option<Exhausted>,
    /// The last time an operator cleared the flag on this host.
    pub cleared_at: Option<Timestamp>,
    pub updated_at: Timestamp,
}

impl Ledger {
    fn empty(month: String, now: Timestamp) -> Ledger {
        Ledger {
            v: LEDGER_VERSION,
            month,
            usd: 0.0,
            turns: 0,
            unmeasured_turns: 0,
            adjust_usd: 0.0,
            exhausted: None,
            cleared_at: None,
            updated_at: now,
        }
    }

    /// What this host publishes as its spend: the measured sum plus the
    /// operator's correction.
    pub fn figure(&self) -> f64 {
        self.usd + self.adjust_usd
    }
}

/// The publisher and host whose retained account records are this host's
/// own copy of its ledger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identity {
    pub publisher: String,
    pub host: String,
}

/// The budget period `now` falls in for `key`: the calendar month in its
/// `reset_tz`.
pub fn month_of(key: &ApiKeyConfig, now: Timestamp) -> Result<String> {
    let tz = jiff::tz::TimeZone::get(&key.reset_tz)
        .with_context(|| format!("reset_tz {:?}", key.reset_tz))?;
    Ok(now.to_zoned(tz).strftime("%Y-%m").to_string())
}

/// What to do with a ledger file that does not parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Damaged {
    Refuse,
    Replace,
}

/// The ledger directory and the operations on the ledgers in it.
#[derive(Debug, Clone)]
pub struct Ledgers {
    dir: PathBuf,
}

impl Ledgers {
    /// `$XDG_STATE_HOME/ccc/budget`, or `~/.local/state/ccc/budget`.
    pub fn in_env(env: &dyn EnvSource) -> Ledgers {
        Ledgers::at(crate::config::state_home_in(env).join("ccc/budget"))
    }

    pub fn at(dir: PathBuf) -> Ledgers {
        Ledgers { dir }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(format!("{name}.json"))
    }

    fn lock(&self, name: &str) -> Result<FileLock> {
        ensure_private_dir(&self.dir)?;
        FileLock::lock(&self.dir.join(format!("{name}.lock")), "budget lock")
    }

    /// The file's ledger, or `None` when there is none.
    fn load(path: &Path) -> Result<Option<Ledger>> {
        load_json(path, "budget ledger")
    }

    /// `name`'s ledger for the month of `now`: empty when there is no file or
    /// the file is of another month.
    pub fn read(&self, name: &str, key: &ApiKeyConfig, now: Timestamp) -> Result<Ledger> {
        let month = month_of(key, now)?;
        Ok(match Self::load(&self.path(name))? {
            Some(ledger) if ledger.month == month => ledger,
            _ => Ledger::empty(month, now),
        })
    }

    /// Reads `name`'s ledger for the month of `now` under its lock, applies
    /// `change`, stamps `updated_at` and writes it back.
    fn update(
        &self,
        name: &str,
        key: &ApiKeyConfig,
        now: Timestamp,
        damaged: Damaged,
        change: impl FnOnce(&mut Ledger) -> Result<()>,
    ) -> Result<Ledger> {
        let month = month_of(key, now)?;
        let _lock = self.lock(name)?;
        let path = self.path(name);
        let loaded = match (Self::load(&path), damaged) {
            (Ok(loaded), _) => loaded,
            (Err(_), Damaged::Replace) => None,
            (Err(e), Damaged::Refuse) => return Err(e),
        };
        let mut ledger = match loaded {
            Some(ledger) if ledger.month == month => ledger,
            _ => Ledger::empty(month, now),
        };
        change(&mut ledger)?;
        ledger.updated_at = now;
        let mut text = serde_json::to_string_pretty(&ledger)?;
        text.push('\n');
        write_atomically(&path, &text)?;
        Ok(ledger)
    }

    /// Charges one key turn: the sum of the costs its process's results
    /// stated, or nothing and one more unmeasured turn when none stated one.
    // TODO(api-key-assumptions-unconfirmed): A3 — the charge is Claude Code's
    // own `model_usage_since_last_result` cost, which has never been compared
    // with what the Platform bills for a key turn.
    pub fn charge(
        &self,
        name: &str,
        key: &ApiKeyConfig,
        now: Timestamp,
        cost: Option<f64>,
    ) -> Result<Ledger> {
        self.update(name, key, now, Damaged::Refuse, |ledger| {
            ledger.turns += 1;
            match cost.filter(|c| c.is_finite() && *c >= 0.0) {
                Some(cost) => ledger.usd += cost,
                None => ledger.unmeasured_turns += 1,
            }
            Ok(())
        })
    }

    /// Records that the provider refused a key turn for spend.
    pub fn mark_exhausted(
        &self,
        name: &str,
        key: &ApiKeyConfig,
        now: Timestamp,
        session: Option<String>,
        message: &str,
    ) -> Result<Ledger> {
        self.update(name, key, now, Damaged::Refuse, |ledger| {
            ledger.exhausted = Some(Exhausted::new(now, session, Some(message.to_string())));
            Ok(())
        })
    }

    /// Sets this host's correction so that its figure plus every other
    /// publisher's and host's figure for the month equals `spent`. `remote` is
    /// `None` when the bus is off, and the ledger alone is reconciled;
    /// `Some(None)` when the broker could not be read, which is refused. A
    /// damaged ledger is replaced.
    pub fn set_spent(
        &self,
        name: &str,
        key: &ApiKeyConfig,
        now: Timestamp,
        spent: f64,
        remote: Option<Option<(&[AccountSpend], &Identity)>>,
    ) -> Result<Ledger> {
        if !spent.is_finite() || spent < 0.0 {
            bail!("--spent must be a finite number of dollars, at least 0 (got {spent})");
        }
        let others = match remote {
            None => 0.0,
            Some(None) => bail!(
                "the broker could not be read, so the other hosts' spend on {name} is unknown; nothing was changed"
            ),
            Some(Some((records, me))) => {
                let month = month_of(key, now)?;
                others(records, name, &month, me).map(|r| r.usd).sum()
            }
        };
        self.update(name, key, now, Damaged::Replace, |ledger| {
            ledger.adjust_usd = spent - others - ledger.usd;
            Ok(())
        })
    }

    /// Clears the exhausted flag on this host: `cleared_at` is now, or one
    /// second past the newest flag in `remote` when that flag is ahead of this
    /// host's clock. Returns the ledger and, in that second case, a note.
    pub fn clear(
        &self,
        name: &str,
        key: &ApiKeyConfig,
        now: Timestamp,
        remote: &[AccountSpend],
    ) -> Result<(Ledger, Option<String>)> {
        let month = month_of(key, now)?;
        let (at, note) = clear_point(
            remote
                .iter()
                .filter(|r| r.account == name && r.period == month)
                .filter_map(|r| r.exhausted.as_ref().map(|e| e.at)),
            now,
        );
        let ledger = self.update(name, key, now, Damaged::Refuse, |ledger| {
            ledger.exhausted = None;
            ledger.cleared_at = Some(at);
            Ok(())
        })?;
        Ok((ledger, note))
    }
}

/// When a clear on this host takes effect, given the `at` of every flag
/// other publishers and hosts hold on the account: `now`, or one second past
/// the newest flag when that flag is at or ahead of this host's clock, with a
/// note saying so.
pub(crate) fn clear_point(
    remote_flags: impl Iterator<Item = Timestamp>,
    now: Timestamp,
) -> (Timestamp, Option<String>) {
    match remote_flags.max() {
        Some(flag) if flag >= now => {
            let at = flag + jiff::SignedDuration::from_secs(1);
            (
                at,
                Some(format!(
                    "the newest exhausted flag ({flag}) is ahead of this host's clock; cleared as of {at}"
                )),
            )
        }
        _ => (now, None),
    }
}

/// The records of `remote` for `account` in `month`, except `me`'s own.
pub(crate) fn others<'a>(
    remote: &'a [AccountSpend],
    account: &'a str,
    month: &'a str,
    me: &'a Identity,
) -> impl Iterator<Item = &'a AccountSpend> + 'a {
    remote.iter().filter(move |r| {
        r.account == account
            && r.period == month
            && !(r.publisher == me.publisher && r.host == me.host)
    })
}

/// This host's ledger as its account-spend record in the agent tree.
fn to_account_spend(
    name: &str,
    key: &ApiKeyConfig,
    ledger: &Ledger,
    me: &Identity,
) -> AccountSpend {
    let mut spend = AccountSpend::new(
        name,
        AccountKind::ApiKey,
        me.publisher.clone(),
        me.host.clone(),
        ledger.month.clone(),
        ledger.figure(),
        ledger.updated_at,
    );
    spend.turns = Some(ledger.turns);
    spend.unmeasured_turns = Some(ledger.unmeasured_turns);
    spend.budget_usd = Some(key.budget_usd);
    spend.exhausted = ledger.exhausted.clone();
    spend.cleared_at = ledger.cleared_at;
    spend
}

/// A record left out of a view because its period is not the key's current
/// month.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Ignored {
    pub publisher: String,
    pub host: String,
    pub period: String,
}

/// What selection, `ccc which` and `ccc budget` know about one key.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BudgetView {
    pub month: String,
    /// This host's figure plus every other publisher's and host's.
    pub spent_usd: f64,
    /// This host's figure.
    pub local_usd: f64,
    pub budget_usd: f64,
    pub reserve_usd: f64,
    pub turns: u64,
    pub unmeasured_turns: u64,
    /// The flag in force: the newest one, unless a newer clear outranks it.
    pub exhausted: Option<Exhausted>,
    /// Whether other hosts' records were read.
    pub remote: bool,
    /// Why this host's ledger could not be read; its figure is then 0.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ledger_error: Option<String>,
    /// Records for this key with another period, by publisher and host.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ignored: Vec<Ignored>,
}

impl BudgetView {
    /// This host's ledger read, no exhausted flag, and spend short of the
    /// budget less the reserve. An unreadable ledger leaves this host's spend
    /// unknown, and a key is not spent on a guess.
    pub fn in_budget(&self) -> bool {
        self.ledger_error.is_none()
            && self.exhausted.is_none()
            && self.spent_usd < self.budget_usd - self.reserve_usd
    }
}

/// The view of `name` from this host's `ledger` and, when they could be
/// read, every retained account record (`remote`). This host's own retained
/// record is ignored: the ledger is authoritative for it.
pub fn view(
    name: &str,
    key: &ApiKeyConfig,
    now: Timestamp,
    ledger: std::result::Result<Ledger, String>,
    remote: Option<(&[AccountSpend], &Identity)>,
) -> Result<BudgetView> {
    let month = month_of(key, now)?;
    let (local, ledger_error) = match ledger {
        Ok(ledger) => (ledger, None),
        Err(e) => (Ledger::empty(month.clone(), now), Some(e)),
    };
    // The combining rules read only account, period and the figures, so the
    // local record's publisher and host are immaterial here.
    let anonymous = Identity {
        publisher: String::new(),
        host: String::new(),
    };
    let me = remote.map_or(&anonymous, |(_, me)| me);
    let mut records = vec![to_account_spend(name, key, &local, me)];
    let mut ignored = Vec::new();
    if let Some((remote, me)) = remote {
        records.extend(others(remote, name, &month, me).cloned());
        ignored.extend(
            remote
                .iter()
                .filter(|r| r.account == name && r.period != month)
                .map(|r| Ignored {
                    publisher: r.publisher.clone(),
                    host: r.host.clone(),
                    period: r.period.clone(),
                }),
        );
    }
    Ok(BudgetView {
        spent_usd: claude_usage_mqtt_core::total_usd(&records, name, &month),
        local_usd: local.figure(),
        budget_usd: key.budget_usd,
        reserve_usd: key.reserve_usd,
        turns: local.turns,
        unmeasured_turns: local.unmeasured_turns,
        exhausted: claude_usage_mqtt_core::exhausted(&records, name, &month, None, now).cloned(),
        remote: remote.is_some(),
        ledger_error,
        ignored,
        month,
    })
}

/// Every configured key's view, failing on the first that cannot be built.
pub fn views(
    cfg: &Config,
    ledgers: &Ledgers,
    now: Timestamp,
    remote: Option<(&[AccountSpend], &Identity)>,
) -> Result<BTreeMap<String, BudgetView>> {
    each_view(cfg, ledgers, now, remote)
        .map(|(name, v)| Ok((name.clone(), v?)))
        .collect()
}

/// A profile's flags in force, by bucket name.
pub type ProfileFlags = BTreeMap<String, Exhausted>;

/// What one account read found, resolved at the read's clock.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Accounts {
    /// Every configured key's budget view that could be built.
    pub keys: BTreeMap<String, BudgetView>,
    /// Every configured profile's flags in force; a profile with none maps
    /// to an empty set.
    pub flags: BTreeMap<String, ProfileFlags>,
}

/// Every configured key's view and every configured profile's flags in force
/// at `now`, from this host's ledgers and limits files and one account read
/// over `bus`, for selection and `ccc which`. When the broker cannot be read
/// both are this host's alone; the bus's notes say why. A key view that
/// cannot be built is reported on stderr and left out.
pub fn bus_accounts(cfg: &Config, env: &dyn EnvSource, now: Timestamp, bus: &dyn Bus) -> Accounts {
    // With no key and no profile there is no level to publish and nothing to
    // read for.
    if cfg.claude.api_keys.is_empty() && cfg.claude.profiles.is_empty() {
        return Accounts::default();
    }
    let ledgers = Ledgers::in_env(env);
    let files = limits_files(cfg, &Limits::in_env(env), now);
    let remote = fetch_remote(cfg, &ledgers, &files, now, bus);
    let remote = remote_of(&remote, bus);
    Accounts {
        keys: lenient(each_view(cfg, &ledgers, now, remote)),
        flags: flags_in_force(&files, remote, now),
    }
}

/// Each profile's flags in force at `now`, from its limits file in `files`
/// and the records of the account read.
fn flags_in_force(
    files: &BTreeMap<String, LimitsFile>,
    remote: Option<(&[AccountSpend], &Identity)>,
    now: Timestamp,
) -> BTreeMap<String, ProfileFlags> {
    files
        .iter()
        .map(|(p, file)| (p.clone(), limits::in_force(p, file, remote, now)))
        .collect()
}

/// The views of `views` that could be built; the rest reported on stderr.
fn lenient<'a>(
    views: impl Iterator<Item = (&'a String, Result<BudgetView>)>,
) -> BTreeMap<String, BudgetView> {
    views
        .filter_map(|(name, v)| match v {
            Ok(v) => Some((name.clone(), v)),
            Err(e) => {
                crate::harness::diag(format_args!(
                    "warning: no budget view for API key {name}: {e:#}"
                ));
                None
            }
        })
        .collect()
}

/// The records of a successful read, with the identity they are read as.
fn remote_of<'a>(
    remote: &'a Remote,
    bus: &'a dyn Bus,
) -> Option<(&'a [AccountSpend], &'a Identity)> {
    match (remote, bus.identity()) {
        (Remote::Records(records), Some(me)) => Some((records.as_slice(), me)),
        _ => None,
    }
}

/// This host's level for every configured key whose ledger can be read, a
/// key never charged having a zero level, and its record of every configured
/// profile's flags.
fn levels(
    cfg: &Config,
    ledgers: &Ledgers,
    files: &BTreeMap<String, LimitsFile>,
    now: Timestamp,
    me: &Identity,
) -> Vec<AccountSpend> {
    let keys = cfg.claude.api_keys.iter().filter_map(|(name, key)| {
        let ledger = ledgers.read(name, key, now).ok()?;
        Some(to_account_spend(name, key, &ledger, me))
    });
    let profiles = files
        .iter()
        .map(|(p, file)| limits::to_account_spend(p, file, me));
    keys.chain(profiles).collect()
}

/// This host's limits file of every configured profile, read once.
fn limits_files(cfg: &Config, limits: &Limits, now: Timestamp) -> BTreeMap<String, LimitsFile> {
    cfg.claude
        .profiles
        .iter()
        .map(|p| (p.clone(), limits.read(p, now)))
        .collect()
}

/// Publishes `ledger` as this host's level for key `name`; nothing when the
/// bus is off.
pub fn publish_level(bus: &dyn Bus, name: &str, key: &ApiKeyConfig, ledger: &Ledger) {
    if let Some(me) = bus.identity() {
        bus.put_account(to_account_spend(name, key, ledger, me));
    }
}

/// The account read: this host's levels for every key and for every profile
/// in `files` published first, then every publisher's and host's records,
/// bounded by `[claude.selection]`'s broker timeout with its quiesce.
/// [`Remote::Off`] when the bus is.
fn fetch_remote(
    cfg: &Config,
    ledgers: &Ledgers,
    files: &BTreeMap<String, LimitsFile>,
    now: Timestamp,
    bus: &dyn Bus,
) -> Remote {
    let Some(me) = bus.identity() else {
        return Remote::Off;
    };
    let sel = &cfg.claude.selection;
    bus.fetch_accounts(
        levels(cfg, ledgers, files, now, me),
        std::time::Duration::from_millis(sel.mqtt_quiesce_ms),
        std::time::Duration::from_millis(sel.mqtt_timeout_ms),
    )
}

/// Each configured key, in name order, with its view or why it has none.
fn each_view<'a>(
    cfg: &'a Config,
    ledgers: &'a Ledgers,
    now: Timestamp,
    remote: Option<(&'a [AccountSpend], &'a Identity)>,
) -> impl Iterator<Item = (&'a String, Result<BudgetView>)> + 'a {
    cfg.claude.api_keys.iter().map(move |(name, key)| {
        let ledger = ledgers.read(name, key, now).map_err(|e| format!("{e:#}"));
        (name, view(name, key, now, ledger, remote))
    })
}

/// What `ccc budget` was asked to do.
#[derive(Debug, Clone, PartialEq)]
pub enum BudgetCommand {
    Show { json: bool },
    Set { name: String, spent: f64 },
    Clear { name: String },
}

/// Everything `ccc budget` reads from outside itself.
pub struct BudgetCtx<'a> {
    pub cfg: &'a Config,
    pub ledgers: &'a Ledgers,
    pub limits: &'a Limits,
    pub now: Timestamp,
    /// Opens the bus, which is read for other hosts' records and written
    /// with this host's. Called only when the command names no unknown
    /// account and a key or a profile is configured.
    pub open_bus: &'a dyn Fn() -> Box<dyn Bus>,
}

/// The marker every report of a key's spend carries when the broker was not
/// read.
pub const THIS_HOST_ONLY: &str = "spend: this host only";

/// Runs `ccc budget`. With a key or a profile configured every form reads
/// the broker first, publishing this host's levels as it does, and the two
/// that change a ledger or a limits file publish the new level. `set` is
/// refused, exit 1, when the bus is on and the broker cannot be read. The bus
/// is closed on every exit. Its
/// notes end the report, or go to `err` for `--json` and ahead of an error.
pub fn budget_with(
    ctx: &BudgetCtx<'_>,
    cmd: &BudgetCommand,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<i32> {
    match cmd {
        BudgetCommand::Set { name, .. } => {
            key_of(ctx.cfg, name)?;
        }
        BudgetCommand::Clear { name } => {
            account_of(ctx.cfg, name)?;
        }
        BudgetCommand::Show { .. } => {}
    }
    // With no key and no profile there is no level to publish and nothing to
    // read for; only `show` gets here.
    if ctx.cfg.claude.api_keys.is_empty() && ctx.cfg.claude.profiles.is_empty() {
        let files = BTreeMap::new();
        return act(ctx, &crate::bus::NoBus, cmd, &Remote::Off, &files, out, err);
    }
    let bus = (ctx.open_bus)();
    let files = limits_files(ctx.cfg, ctx.limits, ctx.now);
    let remote = fetch_remote(ctx.cfg, ctx.ledgers, &files, ctx.now, bus.as_ref());
    let code = act(ctx, bus.as_ref(), cmd, &remote, &files, out, err);
    let notes = bus.close();
    let code = match code {
        Ok(code) => code,
        Err(e) => {
            for note in &notes {
                writeln!(err, "note: {note}")?;
            }
            return Err(e);
        }
    };
    if let BudgetCommand::Show { json: true } = cmd {
        for note in &notes {
            writeln!(err, "note: {note}")?;
        }
        return Ok(code);
    }
    match &remote {
        Remote::Off => writeln!(out, "{THIS_HOST_ONLY}")?,
        // The reason is among the bus's notes, which follow.
        Remote::Unavailable(_) => writeln!(out, "{THIS_HOST_ONLY} (the broker could not be read)")?,
        Remote::Records(_) => {}
    }
    for note in &notes {
        writeln!(out, "note: {note}")?;
    }
    if let (BudgetCommand::Clear { .. }, Some(_)) = (cmd, bus.identity()) {
        if notes.is_empty() {
            writeln!(out, "published to the broker")?;
        } else {
            writeln!(
                out,
                "the publish may not have reached the broker: the flag is cleared on this host only"
            )?;
        }
    }
    Ok(code)
}

fn key_of<'c>(cfg: &'c Config, name: &str) -> Result<&'c ApiKeyConfig> {
    cfg.claude
        .api_keys
        .get(name)
        .with_context(|| format!("no API key named {name:?} in [claude.api_keys]"))
}

/// An account `ccc budget clear` can name.
enum Named<'c> {
    Key(&'c ApiKeyConfig),
    Profile,
}

fn account_of<'c>(cfg: &'c Config, name: &str) -> Result<Named<'c>> {
    if let Some(key) = cfg.claude.api_keys.get(name) {
        return Ok(Named::Key(key));
    }
    if cfg.claude.profiles.iter().any(|p| p == name) {
        return Ok(Named::Profile);
    }
    bail!("no API key named {name:?} in [claude.api_keys] and no profile of that name in [claude] profiles")
}

/// `cmd`'s own output, given the broker read's outcome and the limits
/// `files` it published.
fn act(
    ctx: &BudgetCtx<'_>,
    bus: &dyn Bus,
    cmd: &BudgetCommand,
    remote: &Remote,
    files: &BTreeMap<String, LimitsFile>,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<i32> {
    let records = remote_of(remote, bus);
    Ok(match cmd {
        BudgetCommand::Show { json } => {
            let views = views(ctx.cfg, ctx.ledgers, ctx.now, records)?;
            let profiles = flags_in_force(files, records, ctx.now);
            if *json {
                let mut doc = serde_json::json!({
                    "remote": records.is_some(),
                    "api_keys": views,
                    "profiles": profiles,
                });
                if let Remote::Unavailable(why) = remote {
                    doc["broker_error"] = why.clone().into();
                }
                serde_json::to_writer_pretty(&mut *out, &doc)?;
                writeln!(out)?;
            } else {
                if views.is_empty() {
                    writeln!(out, "no API keys configured ([claude.api_keys])")?;
                }
                for (name, view) in &views {
                    write_view(out, name, view)?;
                }
                for (name, flags) in &profiles {
                    write_profile(out, name, flags)?;
                }
            }
            0
        }
        BudgetCommand::Set { name, spent } => {
            let key = key_of(ctx.cfg, name)?;
            let remote_arg = match (remote, records) {
                (Remote::Off, _) => None,
                (_, records) => Some(records),
            };
            match ctx
                .ledgers
                .set_spent(name, key, ctx.now, *spent, remote_arg)
            {
                Ok(ledger) => {
                    publish_level(bus, name, key, &ledger);
                    writeln!(
                        out,
                        "{name}: {} spent set to ${spent:.2} (correction ${:.2})",
                        ledger.month, ledger.adjust_usd
                    )?;
                    0
                }
                Err(e) if matches!(remote, Remote::Unavailable(_)) => {
                    writeln!(err, "ccc budget set: {e:#}")?;
                    1
                }
                Err(e) => return Err(e),
            }
        }
        BudgetCommand::Clear { name } => {
            let seen = records.map_or(&[][..], |(r, _)| r);
            let (cleared_at, note, what) = match account_of(ctx.cfg, name)? {
                Named::Key(key) => {
                    let (ledger, note) = ctx.ledgers.clear(name, key, ctx.now, seen)?;
                    publish_level(bus, name, key, &ledger);
                    (ledger.cleared_at, note, "exhausted flag")
                }
                Named::Profile => {
                    let (file, note) = ctx.limits.clear(name, ctx.now, seen)?;
                    limits::publish(bus, name, &file);
                    (file.cleared_at, note, "exhausted flags")
                }
            };
            if let Some(note) = note {
                writeln!(out, "note: {note}")?;
            }
            let at = cleared_at.map(|t| t.to_string()).unwrap_or_default();
            writeln!(out, "{name}: {what} cleared as of {at}")?;
            0
        }
    })
}

/// A profile's flags in force, one line per bucket, or that it has none.
fn write_profile(
    out: &mut dyn Write,
    name: &str,
    flags: &BTreeMap<String, Exhausted>,
) -> Result<()> {
    if flags.is_empty() {
        writeln!(out, "{name}: exhausted: no")?;
    }
    for (bucket, e) in flags {
        writeln!(
            out,
            "{name}: {bucket} exhausted since {}{}{}",
            e.at,
            e.until.map(|u| format!(" until {u}")).unwrap_or_default(),
            e.message
                .as_deref()
                .map(|m| format!(": {m}"))
                .unwrap_or_default()
        )?;
    }
    Ok(())
}

fn write_view(out: &mut dyn Write, name: &str, view: &BudgetView) -> Result<()> {
    writeln!(
        out,
        "{name}: {} spent ${:.2} of ${:.2} (this host ${:.2}), reserve ${:.2}",
        view.month, view.spent_usd, view.budget_usd, view.local_usd, view.reserve_usd
    )?;
    writeln!(
        out,
        "  turns {}, unmeasured {}; in budget: {}",
        view.turns,
        view.unmeasured_turns,
        if view.in_budget() { "yes" } else { "no" }
    )?;
    match &view.exhausted {
        None => writeln!(out, "  exhausted: no")?,
        Some(e) => writeln!(
            out,
            "  exhausted: since {}{}",
            e.at,
            e.message
                .as_deref()
                .map(|m| format!(": {m}"))
                .unwrap_or_default()
        )?,
    }
    if let Some(e) = &view.ledger_error {
        writeln!(out, "  ledger unreadable: {e}")?;
    }
    for i in &view.ignored {
        writeln!(
            out,
            "  ignored: {}/{} for period {}",
            i.publisher, i.host, i.period
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::{Recorded, RecordingBus};
    use std::sync::Arc;

    fn key(tz: &str) -> ApiKeyConfig {
        let mut key = ApiKeyConfig::new("/k", 420.0, &["*fable*"]);
        key.reserve_usd = 20.0;
        key.reset_tz = tz.to_string();
        key
    }

    fn ts(s: &str) -> Timestamp {
        s.parse().unwrap()
    }

    fn me() -> Identity {
        Identity {
            publisher: "ccc".to_string(),
            host: "atlas".to_string(),
        }
    }

    fn spend(publisher: &str, host: &str, period: &str, usd: f64) -> AccountSpend {
        AccountSpend::new(
            "team",
            AccountKind::ApiKey,
            publisher,
            host,
            period,
            usd,
            ts("2026-11-10T00:00:00Z"),
        )
    }

    fn flagged(mut s: AccountSpend, at: &str) -> AccountSpend {
        s.exhausted = Some(Exhausted::new(ts(at), None, Some("refused".into())));
        s
    }

    fn cleared(mut s: AccountSpend, at: &str) -> AccountSpend {
        s.cleared_at = Some(ts(at));
        s
    }

    const NOW: &str = "2026-11-10T14:03:11Z";

    #[test]
    fn charge_accumulates_and_counts_unmeasured_turns() {
        let dir = crate::testutil::TempDir::new("budget");
        let l = Ledgers::at(dir.path().join("budget"));
        let k = key("UTC");
        l.charge("team", &k, ts(NOW), Some(1.25)).unwrap();
        l.charge("team", &k, ts(NOW), None).unwrap();
        l.charge("team", &k, ts(NOW), Some(f64::NAN)).unwrap();
        let ledger = l.charge("team", &k, ts(NOW), Some(0.5)).unwrap();
        assert_eq!(ledger.usd, 1.75);
        assert_eq!(ledger.turns, 4);
        assert_eq!(ledger.unmeasured_turns, 2);
        assert_eq!(ledger.month, "2026-11");
        assert_eq!(l.read("team", &k, ts(NOW)).unwrap(), ledger);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = |p: &Path| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode(&dir.path().join("budget")), 0o700);
            assert_eq!(mode(&dir.path().join("budget/team.json")), 0o600);
        }
    }

    #[test]
    fn month_rollover_in_reset_tz_resets_figure_flag_and_clear() {
        let dir = crate::testutil::TempDir::new("budget");
        let l = Ledgers::at(dir.path().to_path_buf());
        let k = key("America/New_York");
        // 03:00 UTC on the 1st is still October in New York.
        let before = ts("2026-11-01T03:00:00Z");
        l.charge("team", &k, before, Some(10.0)).unwrap();
        l.mark_exhausted("team", &k, before, Some("inv".into()), "refused")
            .unwrap();
        l.clear("team", &k, before, &[]).unwrap();
        l.mark_exhausted("team", &k, before, None, "again").unwrap();
        let october = l.read("team", &k, before).unwrap();
        assert_eq!(october.month, "2026-10");
        assert_eq!(october.usd, 10.0);
        assert!(october.exhausted.is_some() && october.cleared_at.is_some());

        let after = ts("2026-11-01T05:00:00Z");
        let fresh = l.read("team", &k, after).unwrap();
        assert_eq!(fresh, Ledger::empty("2026-11".into(), after));
        let charged = l.charge("team", &k, after, Some(2.0)).unwrap();
        assert_eq!(
            (charged.month.as_str(), charged.usd, charged.turns),
            ("2026-11", 2.0, 1)
        );
        assert!(charged.exhausted.is_none() && charged.cleared_at.is_none());
    }

    #[test]
    fn two_threads_charging_at_once_both_land() {
        let dir = crate::testutil::TempDir::new("budget");
        let l = Ledgers::at(dir.path().to_path_buf());
        let k = key("UTC");
        std::thread::scope(|s| {
            for _ in 0..2 {
                s.spawn(|| {
                    for _ in 0..25 {
                        l.charge("team", &k, ts(NOW), Some(1.0)).unwrap();
                    }
                });
            }
        });
        let ledger = l.read("team", &k, ts(NOW)).unwrap();
        assert_eq!((ledger.usd, ledger.turns), (50.0, 50));
    }

    #[test]
    fn a_damaged_ledger_is_reported_not_fatal_and_set_rewrites_it() {
        let dir = crate::testutil::TempDir::new("budget");
        let l = Ledgers::at(dir.path().to_path_buf());
        let k = key("UTC");
        std::fs::write(dir.path().join("team.json"), "{not json").unwrap();
        let err = l.charge("team", &k, ts(NOW), Some(1.0)).unwrap_err();
        assert!(format!("{err:#}").contains("damaged"), "{err:#}");
        assert!(l.read("team", &k, ts(NOW)).is_err());

        let cfg = config_with(&k);
        let v = views(&cfg, &l, ts(NOW), None).unwrap();
        let team = &v["team"];
        assert_eq!(team.local_usd, 0.0);
        assert!(team.ledger_error.as_deref().unwrap().contains("damaged"));
        assert!(!team.in_budget(), "unknown spend is not in budget");

        let ledger = l.set_spent("team", &k, ts(NOW), 12.5, None).unwrap();
        assert_eq!(ledger.figure(), 12.5);
        assert_eq!(l.read("team", &k, ts(NOW)).unwrap().figure(), 12.5);
        assert!(views(&cfg, &l, ts(NOW), None).unwrap()["team"].in_budget());
    }

    #[test]
    fn set_reconciles_the_all_hosts_total_and_refuses_without_the_broker() {
        let dir = crate::testutil::TempDir::new("budget");
        let l = Ledgers::at(dir.path().to_path_buf());
        let k = key("UTC");
        l.charge("team", &k, ts(NOW), Some(30.0)).unwrap();
        let remote = vec![
            spend("ccc", "atlas", "2026-11", 999.0), // this host's own copy
            spend("ccc", "borg", "2026-11", 50.0),
            spend("brenn", "atlas", "2026-11", 20.0),
            spend("ccc", "borg", "2026-10", 400.0),
        ];
        let err = l
            .set_spent("team", &k, ts(NOW), 100.0, Some(None))
            .unwrap_err();
        assert!(format!("{err:#}").contains("could not be read"));
        assert_eq!(l.read("team", &k, ts(NOW)).unwrap().adjust_usd, 0.0);

        let ledger = l
            .set_spent(
                "team",
                &k,
                ts(NOW),
                90.0,
                Some(Some((remote.as_slice(), &me()))),
            )
            .unwrap();
        assert_eq!(ledger.adjust_usd, -10.0);
        let v = view(
            "team",
            &k,
            ts(NOW),
            Ok(ledger),
            Some((remote.as_slice(), &me())),
        )
        .unwrap();
        assert_eq!(v.spent_usd, 90.0);
        assert!(l.set_spent("team", &k, ts(NOW), -1.0, None).is_err());
    }

    #[test]
    fn clear_sets_cleared_at_and_steps_past_a_flag_ahead_of_the_clock() {
        let dir = crate::testutil::TempDir::new("budget");
        let l = Ledgers::at(dir.path().to_path_buf());
        let k = key("UTC");
        l.mark_exhausted("team", &k, ts(NOW), Some("inv".into()), "refused")
            .unwrap();
        let (ledger, note) = l.clear("team", &k, ts(NOW), &[]).unwrap();
        assert!(ledger.exhausted.is_none());
        assert_eq!(ledger.cleared_at, Some(ts(NOW)));
        assert!(note.is_none());

        let ahead = vec![flagged(
            spend("ccc", "borg", "2026-11", 1.0),
            "2026-11-10T14:05:00Z",
        )];
        let (ledger, note) = l.clear("team", &k, ts(NOW), &ahead).unwrap();
        assert_eq!(ledger.cleared_at, Some(ts("2026-11-10T14:05:01Z")));
        assert!(note.unwrap().contains("ahead of this host's clock"));
        let v = view(
            "team",
            &k,
            ts(NOW),
            Ok(ledger),
            Some((ahead.as_slice(), &me())),
        )
        .unwrap();
        assert!(v.exhausted.is_none());
    }

    #[test]
    fn the_ledger_maps_to_account_spend_field_for_field() {
        let k = key("UTC");
        let ledger = Ledger {
            v: 1,
            month: "2026-11".into(),
            usd: 137.42,
            turns: 212,
            unmeasured_turns: 3,
            adjust_usd: 2.58,
            exhausted: Some(Exhausted::new(
                ts(NOW),
                Some("inv".into()),
                Some("m".into()),
            )),
            cleared_at: Some(ts("2026-11-09T00:00:00Z")),
            updated_at: ts(NOW),
        };
        let s = to_account_spend("team", &k, &ledger, &me());
        assert_eq!(s.account, "team");
        assert_eq!(s.account_kind, AccountKind::ApiKey);
        assert_eq!((s.publisher.as_str(), s.host.as_str()), ("ccc", "atlas"));
        assert_eq!(s.period, "2026-11");
        assert_eq!(s.usd, 140.0);
        assert_eq!(s.turns, Some(212));
        assert_eq!(s.unmeasured_turns, Some(3));
        assert_eq!(s.budget_usd, Some(420.0));
        assert_eq!(s.exhausted, ledger.exhausted);
        assert_eq!(s.cleared_at, ledger.cleared_at);
        assert_eq!(s.updated_at, ledger.updated_at);
    }

    fn local(usd: f64) -> Ledger {
        let mut l = Ledger::empty("2026-11".into(), ts(NOW));
        l.usd = usd;
        l
    }

    #[test]
    fn view_sums_other_hosts_for_the_current_period_only() {
        let k = key("UTC");
        let remote = vec![
            spend("ccc", "atlas", "2026-11", 500.0),
            spend("ccc", "borg", "2026-11", 40.0),
            spend("brenn", "atlas", "2026-11", 2.5),
            spend("ccc", "borg", "2026-10", 300.0),
        ];
        let v = view(
            "team",
            &k,
            ts(NOW),
            Ok(local(10.0)),
            Some((remote.as_slice(), &me())),
        )
        .unwrap();
        assert_eq!(v.spent_usd, 52.5);
        assert_eq!(v.local_usd, 10.0);
        assert!(v.remote);
        assert!(v.in_budget());
        assert_eq!(
            v.ignored,
            vec![Ignored {
                publisher: "ccc".into(),
                host: "borg".into(),
                period: "2026-10".into()
            }]
        );
        let near = vec![spend("ccc", "borg", "2026-11", 390.0)];
        let v = view(
            "team",
            &k,
            ts(NOW),
            Ok(local(10.0)),
            Some((near.as_slice(), &me())),
        )
        .unwrap();
        assert!(!v.in_budget(), "400 is not below 420 - 20");
    }

    #[test]
    fn a_flag_from_another_publisher_exhausts_and_a_newer_clear_lifts_it() {
        let k = key("UTC");
        let flag = flagged(spend("brenn", "x", "2026-11", 0.0), "2026-11-10T10:00:00Z");
        let v = view(
            "team",
            &k,
            ts(NOW),
            Ok(local(0.0)),
            Some((std::slice::from_ref(&flag), &me())),
        )
        .unwrap();
        assert!(v.exhausted.is_some());
        assert!(!v.in_budget());

        let clear = cleared(spend("ccc", "borg", "2026-11", 0.0), "2026-11-10T11:00:00Z");
        let both = vec![flag.clone(), clear];
        let v = view(
            "team",
            &k,
            ts(NOW),
            Ok(local(0.0)),
            Some((both.as_slice(), &me())),
        )
        .unwrap();
        assert!(v.exhausted.is_none());

        let last_month = vec![flagged(
            spend("brenn", "x", "2026-10", 0.0),
            "2026-10-31T10:00:00Z",
        )];
        let v = view(
            "team",
            &k,
            ts(NOW),
            Ok(local(0.0)),
            Some((last_month.as_slice(), &me())),
        )
        .unwrap();
        assert!(v.exhausted.is_none());
        assert_eq!(v.spent_usd, 0.0);
    }

    #[test]
    fn remote_unavailable_yields_the_local_figure() {
        let k = key("UTC");
        let v = view("team", &k, ts(NOW), Ok(local(7.0)), None).unwrap();
        assert_eq!((v.spent_usd, v.remote), (7.0, false));
    }

    fn config_with(k: &ApiKeyConfig) -> Config {
        let mut cfg = Config::default();
        cfg.claude.api_keys.insert("team".into(), k.clone());
        cfg
    }

    #[test]
    fn budget_command_shows_sets_and_clears_on_this_host() {
        let dir = crate::testutil::TempDir::new("budget");
        let l = Ledgers::at(dir.path().to_path_buf());
        let k = key("UTC");
        let cfg = config_with(&k);
        let open_bus = || -> Box<dyn Bus> { Box::new(crate::bus::NoBus) };
        let ctx = BudgetCtx {
            cfg: &cfg,
            ledgers: &l,
            limits: &Limits::at(l.dir.join("limits")),
            now: ts(NOW),
            open_bus: &open_bus,
        };
        let run = |cmd: BudgetCommand| {
            let mut out = Vec::new();
            let code = budget_with(&ctx, &cmd, &mut out, &mut Vec::new()).unwrap();
            assert_eq!(code, 0);
            String::from_utf8(out).unwrap()
        };
        l.charge("team", &k, ts(NOW), Some(137.42)).unwrap();
        l.charge("team", &k, ts(NOW), None).unwrap();
        l.mark_exhausted("team", &k, ts(NOW), None, "credit balance is too low")
            .unwrap();

        let text = run(BudgetCommand::Show { json: false });
        assert!(
            text.contains("team: 2026-11 spent $137.42 of $420.00"),
            "{text}"
        );
        assert!(
            text.contains("turns 2, unmeasured 1; in budget: no"),
            "{text}"
        );
        assert!(text.contains("exhausted: since"), "{text}");
        assert!(text.ends_with("spend: this host only\n"), "{text}");

        let json: serde_json::Value =
            serde_json::from_str(&run(BudgetCommand::Show { json: true })).unwrap();
        assert_eq!(json["remote"], false);
        assert_eq!(json["api_keys"]["team"]["spent_usd"], 137.42);
        assert_eq!(json["api_keys"]["team"]["unmeasured_turns"], 1);
        assert!(json["api_keys"]["team"]["exhausted"]["at"].is_string());

        let text = run(BudgetCommand::Clear {
            name: "team".into(),
        });
        assert!(text.contains("team: exhausted flag cleared"), "{text}");
        let text = run(BudgetCommand::Set {
            name: "team".into(),
            spent: 150.0,
        });
        assert!(text.contains("correction $12.58"), "{text}");
        let text = run(BudgetCommand::Show { json: false });
        assert!(text.contains("spent $150.00"), "{text}");
        assert!(text.contains("exhausted: no"), "{text}");

        // With the bus off a rejected figure is an error, not a refusal.
        let before = l.read("team", &k, ts(NOW)).unwrap();
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let result = budget_with(
            &ctx,
            &BudgetCommand::Set {
                name: "team".into(),
                spent: -1.0,
            },
            &mut out,
            &mut err,
        );
        assert!(format!("{:#}", result.unwrap_err()).contains("--spent must be"));
        assert!(out.is_empty() && err.is_empty());
        assert_eq!(l.read("team", &k, ts(NOW)).unwrap(), before);

        let mut out = Vec::new();
        let err = budget_with(
            &ctx,
            &BudgetCommand::Clear {
                name: "nope".into(),
            },
            &mut out,
            &mut Vec::new(),
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("no API key named"));
    }

    #[test]
    fn budget_command_with_no_keys_says_so_without_opening_the_bus() {
        let dir = crate::testutil::TempDir::new("budget");
        let l = Ledgers::at(dir.path().to_path_buf());
        let mut cfg = Config::default();
        cfg.claude.profiles.clear();
        let open_bus = || -> Box<dyn Bus> { panic!("the bus was opened with no key") };
        let ctx = BudgetCtx {
            cfg: &cfg,
            ledgers: &l,
            limits: &Limits::at(l.dir.join("limits")),
            now: ts(NOW),
            open_bus: &open_bus,
        };
        let run = |cmd: BudgetCommand| {
            let mut out = Vec::new();
            let code = budget_with(&ctx, &cmd, &mut out, &mut Vec::new());
            (code, String::from_utf8(out).unwrap())
        };
        let (code, text) = run(BudgetCommand::Show { json: false });
        assert_eq!(code.unwrap(), 0);
        assert_eq!(text, "no API keys configured ([claude.api_keys])\n");
        let (code, json) = run(BudgetCommand::Show { json: true });
        assert_eq!(code.unwrap(), 0);
        let json: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(
            json,
            serde_json::json!({ "remote": false, "api_keys": {}, "profiles": {} })
        );
        let (code, _) = run(BudgetCommand::Clear {
            name: "team".into(),
        });
        let err = format!("{:#}", code.unwrap_err());
        assert!(
            err.contains("no API key named") && err.contains("no profile"),
            "{err}"
        );
    }

    fn profile_flag(at: &str, until: &str) -> Exhausted {
        let mut e = Exhausted::new(ts(at), Some("inv".into()), Some("Fable limit".into()));
        e.until = Some(ts(until));
        e.kind = Some("seven_day_overage_included".into());
        e
    }

    fn subscription(publisher: &str, host: &str) -> AccountSpend {
        AccountSpend::new(
            "team01",
            AccountKind::Subscription,
            publisher,
            host,
            limits::PERIOD,
            0.0,
            ts(NOW),
        )
    }

    #[test]
    fn budget_show_lists_profiles_with_and_without_flags_when_no_key_is_configured() {
        let dir = crate::testutil::TempDir::new("budget");
        let l = Ledgers::at(dir.path().to_path_buf());
        let mut cfg = Config::default();
        cfg.claude.profiles = vec!["team01".into(), "team02".into()];
        Limits::at(l.dir.join("limits"))
            .raise(
                "team01",
                limits::FABLE,
                profile_flag(NOW, "2026-11-13T04:00:00Z"),
                ts(NOW),
            )
            .unwrap();
        let mut zeus = subscription("ccc", "zeus");
        zeus.exhausted_buckets.insert(
            limits::FIVE_HOUR.into(),
            profile_flag(NOW, "2026-11-10T18:00:00Z"),
        );
        let bus = Arc::new(RecordingBus::new(
            "ccc",
            "atlas",
            Remote::Records(vec![zeus]),
        ));

        let (code, text, _) = run_on(&cfg, &l, &bus, BudgetCommand::Show { json: false });
        assert_eq!(code, 0);
        assert!(text.starts_with("no API keys configured"), "{text}");
        assert!(
            text.contains(
                "team01: fable exhausted since 2026-11-10T14:03:11Z until 2026-11-13T04:00:00Z: Fable limit\n"
            ),
            "{text}"
        );
        assert!(text.contains("team01: five_hour exhausted since"), "{text}");
        assert!(text.contains("team02: exhausted: no\n"), "{text}");
        // This host's profile records were put first.
        match &bus.calls()[0] {
            Recorded::Fetch(levels) => {
                let names: Vec<_> = levels.iter().map(|s| s.account.as_str()).collect();
                assert_eq!(names, vec!["team01", "team02"]);
                assert!(levels[0].exhausted_buckets.contains_key("fable"));
            }
            other => panic!("{other:?}"),
        }

        let (_, json, _) = run_on(&cfg, &l, &bus, BudgetCommand::Show { json: true });
        let json: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(json["api_keys"], serde_json::json!({}));
        assert_eq!(
            json["profiles"]["team01"]["fable"]["until"],
            "2026-11-13T04:00:00Z"
        );
        assert!(json["profiles"]["team01"]["five_hour"]["at"].is_string());
        assert_eq!(json["profiles"]["team02"], serde_json::json!({}));
    }

    #[test]
    fn budget_clear_on_a_profile_lifts_every_bucket_writes_and_publishes() {
        let dir = crate::testutil::TempDir::new("budget");
        let l = Ledgers::at(dir.path().to_path_buf());
        let k = key("UTC");
        let mut cfg = config_with(&k);
        cfg.claude.profiles = vec!["team01".into()];
        let limits = Limits::at(l.dir.join("limits"));
        for bucket in [limits::FABLE, limits::GENERAL] {
            limits
                .raise(
                    "team01",
                    bucket,
                    profile_flag(NOW, "2026-11-13T04:00:00Z"),
                    ts(NOW),
                )
                .unwrap();
        }
        let mut zeus = subscription("ccc", "zeus");
        zeus.exhausted_buckets.insert(
            limits::FIVE_HOUR.into(),
            profile_flag("2026-11-10T13:00:00Z", "2026-11-10T18:00:00Z"),
        );
        let bus = Arc::new(RecordingBus::new(
            "ccc",
            "atlas",
            Remote::Records(vec![zeus]),
        ));
        let (code, out, _) = run_on(
            &cfg,
            &l,
            &bus,
            BudgetCommand::Clear {
                name: "team01".into(),
            },
        );
        assert_eq!(code, 0);
        assert!(
            out.contains("team01: exhausted flags cleared as of 2026-11-10T14:03:11Z"),
            "{out}"
        );
        assert!(out.ends_with("published to the broker\n"), "{out}");
        let file = limits.try_read("team01", ts(NOW)).unwrap();
        assert!(file.buckets.is_empty());
        assert_eq!(file.cleared_at, Some(ts(NOW)));
        let put = bus
            .calls()
            .into_iter()
            .find_map(|c| match c {
                Recorded::PutAccount(s) => Some(*s),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            (put.account.as_str(), put.cleared_at),
            ("team01", Some(ts(NOW)))
        );
        // The key's ledger is untouched.
        assert!(l.read("team", &k, ts(NOW)).unwrap().cleared_at.is_none());

        // Every bucket on every host is lifted; a flag raised after the clear
        // is in force again.
        let remote = vec![{
            let mut z = subscription("ccc", "zeus");
            z.exhausted_buckets.insert(
                limits::FIVE_HOUR.into(),
                profile_flag("2026-11-10T13:00:00Z", "2026-11-10T18:00:00Z"),
            );
            z
        }];
        let me = me();
        let flags = |file: &limits::LimitsFile| {
            limits::in_force("team01", file, Some((remote.as_slice(), &me)), ts(NOW))
        };
        assert!(flags(&file).is_empty());
        let later = "2026-11-10T14:30:00Z";
        let file = limits
            .raise(
                "team01",
                limits::FABLE,
                profile_flag(later, "2026-11-13T04:00:00Z"),
                ts(later),
            )
            .unwrap();
        assert_eq!(flags(&file).keys().collect::<Vec<_>>(), vec!["fable"]);
    }

    #[test]
    fn budget_closes_the_bus_when_the_command_fails_after_the_read() {
        let dir = crate::testutil::TempDir::new("budget");
        let l = Ledgers::at(dir.path().to_path_buf());
        let k = key("UTC");
        let cfg = config_with(&k);
        let mut bus = RecordingBus::new("ccc", "atlas", Remote::Records(Vec::new()));
        bus.close_notes = vec!["the broker refused a publish (NotAuthorized)".into()];
        let bus = Arc::new(bus);
        let open_bus = shared(&bus);
        let ctx = BudgetCtx {
            cfg: &cfg,
            ledgers: &l,
            limits: &Limits::at(l.dir.join("limits")),
            now: ts(NOW),
            open_bus: &open_bus,
        };
        // A readable broker, then a figure `set` rejects.
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let result = budget_with(
            &ctx,
            &BudgetCommand::Set {
                name: "team".into(),
                spent: -1.0,
            },
            &mut out,
            &mut err,
        );
        assert!(format!("{:#}", result.unwrap_err()).contains("--spent must be"));
        assert_eq!(bus.calls().last(), Some(&Recorded::Close));
        assert!(out.is_empty());
        assert_eq!(
            String::from_utf8(err).unwrap(),
            "note: the broker refused a publish (NotAuthorized)\n"
        );
    }

    /// Opens `bus` as the command's bus, keeping the test's handle to it.
    fn shared(bus: &Arc<RecordingBus>) -> impl Fn() -> Box<dyn Bus> {
        let bus = Arc::clone(bus);
        move || Box::new(Arc::clone(&bus))
    }

    /// Runs `cmd` against `bus`, returning the exit code, stdout and stderr.
    fn run_on(
        cfg: &Config,
        l: &Ledgers,
        bus: &Arc<RecordingBus>,
        cmd: BudgetCommand,
    ) -> (i32, String, String) {
        let open_bus = shared(bus);
        let ctx = BudgetCtx {
            cfg,
            ledgers: l,
            limits: &Limits::at(l.dir.join("limits")),
            now: ts(NOW),
            open_bus: &open_bus,
        };
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = budget_with(&ctx, &cmd, &mut out, &mut err).unwrap();
        (
            code,
            String::from_utf8(out).unwrap(),
            String::from_utf8(err).unwrap(),
        )
    }

    #[test]
    fn budget_show_puts_this_hosts_level_reads_every_host_and_ignores_its_own_copy() {
        let dir = crate::testutil::TempDir::new("budget");
        let l = Ledgers::at(dir.path().to_path_buf());
        let k = key("UTC");
        let cfg = config_with(&k);
        l.charge("team", &k, ts(NOW), Some(10.0)).unwrap();
        let bus = Arc::new(RecordingBus::new(
            "ccc",
            "atlas",
            Remote::Records(vec![
                spend("ccc", "atlas", "2026-11", 999.0),
                spend("ccc", "zeus", "2026-11", 30.0),
                spend("brenn", "atlas", "2026-10", 5.0),
            ]),
        ));
        let (code, text, _) = run_on(&cfg, &l, &bus, BudgetCommand::Show { json: false });
        assert_eq!(code, 0);
        assert!(
            text.contains("spent $40.00 of $420.00 (this host $10.00)"),
            "{text}"
        );
        assert!(
            text.contains("ignored: brenn/atlas for period 2026-10"),
            "{text}"
        );
        assert!(!text.contains(THIS_HOST_ONLY), "{text}");
        let calls = bus.calls();
        assert_eq!(calls.len(), 2, "{calls:?}");
        match &calls[0] {
            Recorded::Fetch(levels) => {
                assert_eq!(levels.len(), 1);
                assert_eq!((levels[0].host.as_str(), levels[0].usd), ("atlas", 10.0));
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(calls[1], Recorded::Close);

        let (_, json, _) = run_on(&cfg, &l, &bus, BudgetCommand::Show { json: true });
        let json: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(json["remote"], true);
        assert_eq!(json["api_keys"]["team"]["spent_usd"], 40.0);
    }

    #[test]
    fn budget_set_is_refused_on_an_unreadable_broker_and_reconciles_against_a_readable_one() {
        let dir = crate::testutil::TempDir::new("budget");
        let l = Ledgers::at(dir.path().to_path_buf());
        let k = key("UTC");
        let cfg = config_with(&k);
        l.charge("team", &k, ts(NOW), Some(10.0)).unwrap();
        let set = || BudgetCommand::Set {
            name: "team".into(),
            spent: 100.0,
        };

        let down = Arc::new(RecordingBus::new(
            "ccc",
            "atlas",
            Remote::Unavailable("timed out".into()),
        ));
        let (code, out, err) = run_on(&cfg, &l, &down, set());
        assert_eq!(code, 1);
        assert!(err.contains("the broker could not be read"), "{err}");
        assert!(out.contains("(the broker could not be read)"), "{out}");
        assert_eq!(l.read("team", &k, ts(NOW)).unwrap().adjust_usd, 0.0);
        assert!(!down
            .calls()
            .iter()
            .any(|c| matches!(c, Recorded::PutAccount(_))));

        let up = Arc::new(RecordingBus::new(
            "ccc",
            "atlas",
            Remote::Records(vec![spend("ccc", "zeus", "2026-11", 30.0)]),
        ));
        let (code, out, _) = run_on(&cfg, &l, &up, set());
        assert_eq!(code, 0);
        assert!(out.contains("correction $60.00"), "{out}");
        let put = up.calls().into_iter().find_map(|c| match c {
            Recorded::PutAccount(s) => Some(*s),
            _ => None,
        });
        assert_eq!(put.map(|s| s.usd), Some(70.0));
    }

    #[test]
    fn budget_clear_publishes_and_says_whether_the_publish_reached_the_broker() {
        let dir = crate::testutil::TempDir::new("budget");
        let l = Ledgers::at(dir.path().to_path_buf());
        let k = key("UTC");
        let cfg = config_with(&k);
        let clear = || BudgetCommand::Clear {
            name: "team".into(),
        };
        // A flag from another host, ahead of this host's clock.
        let ahead = flagged(spend("ccc", "zeus", "2026-11", 1.0), "2026-11-10T14:05:00Z");
        let bus = Arc::new(RecordingBus::new(
            "ccc",
            "atlas",
            Remote::Records(vec![ahead]),
        ));
        let (code, out, _) = run_on(&cfg, &l, &bus, clear());
        assert_eq!(code, 0);
        assert!(out.contains("is ahead of this host's clock"), "{out}");
        assert!(out.ends_with("published to the broker\n"), "{out}");
        let put = bus.calls().into_iter().find_map(|c| match c {
            Recorded::PutAccount(s) => Some(*s),
            _ => None,
        });
        assert_eq!(
            put.and_then(|s| s.cleared_at),
            Some(ts("2026-11-10T14:05:01Z"))
        );

        let mut refused = RecordingBus::new("ccc", "atlas", Remote::Records(Vec::new()));
        refused.close_notes = vec!["the broker refused a publish (NotAuthorized)".into()];
        let refused = Arc::new(refused);
        let (_, out, _) = run_on(&cfg, &l, &refused, clear());
        assert!(out.contains("note: the broker refused a publish"), "{out}");
        assert!(out.contains("cleared on this host only"), "{out}");
    }

    #[test]
    fn budget_show_json_on_an_unreadable_broker_keeps_stdout_json_and_notes_on_stderr() {
        let dir = crate::testutil::TempDir::new("budget");
        let l = Ledgers::at(dir.path().to_path_buf());
        let k = key("UTC");
        let cfg = config_with(&k);
        l.charge("team", &k, ts(NOW), Some(10.0)).unwrap();
        let mut down = RecordingBus::new("ccc", "atlas", Remote::Unavailable("timed out".into()));
        down.close_notes = vec!["account spend not read from the broker: timed out".into()];
        let down = Arc::new(down);
        let (code, out, err) = run_on(&cfg, &l, &down, BudgetCommand::Show { json: true });
        assert_eq!(code, 0);
        let json: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(json["remote"], false);
        assert_eq!(json["broker_error"], "timed out");
        assert_eq!(json["api_keys"]["team"]["spent_usd"], 10.0);
        assert_eq!(
            err,
            "note: account spend not read from the broker: timed out\n"
        );
    }

    #[test]
    fn another_hosts_spend_and_flag_reach_the_views_selection_reads() {
        let dir = crate::testutil::TempDir::new("budget");
        let env: BTreeMap<String, String> = [(
            "XDG_STATE_HOME".to_string(),
            dir.path().display().to_string(),
        )]
        .into_iter()
        .collect();
        let k = key("UTC");
        let cfg = config_with(&k);
        let l = Ledgers::in_env(&env);
        l.charge("team", &k, ts(NOW), Some(10.0)).unwrap();

        let over = RecordingBus::new(
            "ccc",
            "atlas",
            Remote::Records(vec![spend("ccc", "zeus", "2026-11", 395.0)]),
        );
        let views = bus_accounts(&cfg, &env, ts(NOW), &over).keys;
        let v = &views["team"];
        assert_eq!((v.spent_usd, v.local_usd, v.remote), (405.0, 10.0, true));
        assert!(!v.in_budget(), "{v:?}");
        match over.calls().as_slice() {
            [Recorded::Fetch(levels)] => {
                assert_eq!(levels.len(), 1);
                assert_eq!((levels[0].host.as_str(), levels[0].usd), ("atlas", 10.0));
            }
            other => panic!("{other:?}"),
        }

        let flag = RecordingBus::new(
            "ccc",
            "atlas",
            Remote::Records(vec![flagged(
                spend("ccc", "zeus", "2026-11", 1.0),
                "2026-11-10T14:00:00Z",
            )]),
        );
        let v = &bus_accounts(&cfg, &env, ts(NOW), &flag).keys["team"];
        assert!(v.exhausted.is_some() && !v.in_budget(), "{v:?}");
    }

    #[test]
    fn one_account_read_serves_the_keys_and_the_profiles_flags() {
        let dir = crate::testutil::TempDir::new("budget");
        let env: BTreeMap<String, String> = [(
            "XDG_STATE_HOME".to_string(),
            dir.path().display().to_string(),
        )]
        .into_iter()
        .collect();
        let k = key("UTC");
        let mut cfg = config_with(&k);
        cfg.claude.profiles = vec!["team01".into(), "team02".into()];
        Limits::in_env(&env)
            .raise(
                "team01",
                limits::FABLE,
                profile_flag(NOW, "2026-11-13T04:00:00Z"),
                ts(NOW),
            )
            .unwrap();

        // Another host's flag counts; this host's own retained record does
        // not, nor does an expired flag.
        let mut zeus = subscription("ccc", "zeus");
        zeus.exhausted_buckets.insert(
            limits::FIVE_HOUR.into(),
            profile_flag(NOW, "2026-11-10T18:00:00Z"),
        );
        zeus.exhausted_buckets.insert(
            limits::GENERAL.into(),
            profile_flag("2026-11-09T14:00:00Z", "2026-11-10T00:00:00Z"),
        );
        let mut own = subscription("ccc", "atlas");
        own.account = "team02".into();
        own.exhausted_buckets.insert(
            limits::GENERAL.into(),
            profile_flag(NOW, "2026-11-13T04:00:00Z"),
        );
        let bus = RecordingBus::new(
            "ccc",
            "atlas",
            Remote::Records(vec![zeus, own, spend("ccc", "zeus", "2026-11", 5.0)]),
        );
        let read = bus_accounts(&cfg, &env, ts(NOW), &bus);
        assert_eq!(read.keys["team"].spent_usd, 5.0);
        assert_eq!(
            read.flags["team01"].keys().collect::<Vec<_>>(),
            vec!["fable", "five_hour"]
        );
        assert!(read.flags["team02"].is_empty(), "{read:?}");
        match bus.calls().as_slice() {
            [Recorded::Fetch(levels)] => {
                let names: Vec<_> = levels.iter().map(|s| s.account.as_str()).collect();
                assert_eq!(names, vec!["team", "team01", "team02"]);
            }
            other => panic!("{other:?}"),
        }

        // With the bus off the flags are this host's alone.
        let read = bus_accounts(&cfg, &env, ts(NOW), &crate::bus::NoBus);
        assert_eq!(
            read.flags["team01"].keys().collect::<Vec<_>>(),
            vec!["fable"]
        );
        assert!(!read.keys["team"].remote);

        // A clear lifts every bucket at the next read.
        Limits::in_env(&env).clear("team01", ts(NOW), &[]).unwrap();
        let read = bus_accounts(&cfg, &env, ts(NOW), &crate::bus::NoBus);
        assert!(read.flags["team01"].is_empty(), "{read:?}");

        // With neither a key nor a profile nothing is read.
        let none = RecordingBus::new("ccc", "atlas", Remote::Records(Vec::new()));
        assert_eq!(
            bus_accounts(&Config::default(), &env, ts(NOW), &none),
            Accounts::default()
        );
        assert!(none.calls().is_empty());
    }
}
