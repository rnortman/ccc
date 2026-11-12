//! Claude account selection: score every configured profile from its retained
//! usage snapshot and pick the one with the most headroom per hour.
//!
//! Fable work draws on both weekly buckets at once, so a Fable model is scored
//! by whichever bucket runs out first, and non-Fable work is scored against the
//! general remainder net of what the Fable remainder will need.
//!
//! A missing 5-hour window is an unspent one: every account has that window.
//! A missing weekly window is a bucket the account does not have, and a model
//! that spends that bucket cannot use the account.
//!
//! Candidates rank by tier before score: an account above its 5-hour knee with
//! a positive score (tier 1) beats any account in its 5-hour ramp or scoring
//! zero (tier 2), whatever the weekly rates say.

use std::collections::BTreeMap;

use anyhow::{bail, Result};
use claude_usage_mqtt_core::{AuthState, FIVE_HOUR};
use jiff::{SignedDuration, Timestamp};

use crate::audit_view::sanitize;
use crate::budget::BudgetView;
use crate::config::{ApiKeyConfig, ClaudeConfig, EnvSource, SelectionConfig};
use crate::glob::any_match;
use crate::harness::claude::Account;
use crate::harness::diag;
use crate::tuning::{
    rule_suffix, weight_note, AccountPriority, NotApplied, ResetApplied, Tuning, WEEK_HOURS,
};
use crate::usage::{ProfileUsage, UsageReport, UsageSource};

/// One weekly bucket the account actually has.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bucket {
    /// Percent of the tightest matching window spent.
    pub util: f64,
    /// Hours until `resets_at`, floored by `min_time_left_h`.
    pub hours: f64,
    /// The reset `hours` counts to: the window's reset, or its fallback, after
    /// any weekly-reset override and before the floor.
    pub resets_at: Timestamp,
}

/// One profile's scoring inputs and its computed score.
#[derive(Debug, Clone, PartialEq)]
pub struct Scored {
    pub profile: String,
    /// Percent of the 5-hour window spent.
    pub util5: f64,
    /// `None`: no window matched `general_windows`; the account has no general
    /// weekly bucket, either because it is not entitled to one or because the
    /// daemon could not read its utilisation.
    pub general: Option<Bucket>,
    /// `None`: no window matched `fable_windows`; the account is not entitled
    /// to Fable, or the daemon could not read that bucket's utilisation.
    pub fable: Option<Bucket>,
    /// This profile's 5-hour floor and knee in remaining percent, after its
    /// per-profile overrides.
    pub five_hour_floor_pct: f64,
    pub five_hour_knee_pct: f64,
    /// The weight this profile's score is multiplied by for the model; 1 when
    /// no knob sets one.
    pub weight: f64,
    pub gate5: f64,
    pub score: f64,
    pub usable: bool,
    /// Why this profile is unusable; always `None` for a candidate.
    pub reason: Option<String>,
    /// `reason` as a type: set exactly when `reason` is.
    pub out: Option<Out>,
    /// Things worth knowing about a profile that is a candidate: its weight
    /// and weekly-reset override, then the scoring pass it came from, then why
    /// it is in tier 2. Always empty for an unusable profile.
    pub notes: Vec<String>,
}

impl Scored {
    /// 1: above the 5-hour knee with a positive score; 2: any other usable
    /// profile. `None` for an unusable profile.
    pub fn tier(&self) -> Option<u8> {
        self.usable.then(|| self.candidate_tier())
    }

    /// The tier this profile would have as a candidate, usable or not.
    fn candidate_tier(&self) -> u8 {
        if self.gate5 >= 1.0 && self.score > 0.0 {
            1
        } else {
            2
        }
    }

    /// Why this candidate is in tier 2, or `None` for a tier-1 one.
    fn tier_two_reason(&self) -> Option<&'static str> {
        match self.candidate_tier() {
            1 => None,
            _ if self.gate5 < 1.0 => Some("in the 5-hour ramp"),
            _ => Some("weekly score is zero"),
        }
    }

    /// The notes as one line, or `None` when there are none.
    pub fn note(&self) -> Option<String> {
        (!self.notes.is_empty()).then(|| self.notes.join("; "))
    }
}

/// Why a scored profile is unusable, as a type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Out {
    /// The login is not `Ok`.
    Auth,
    /// The daemon carried the reading forward.
    CarriedForward,
    /// The reading is older than `stale_grace_s`.
    Stale,
    /// No snapshot is published for the profile.
    NoSnapshot,
    /// The broker could not be read, or the daemon reports itself offline.
    NoData,
    /// The 5-hour window is at or below its floor.
    FiveHourFloor,
    /// No window matched `general_windows`.
    NoGeneralWindow,
    /// No window matched `fable_windows`, for a Fable model.
    NoFableWindow,
}

impl Choice {
    /// Writes everything the operator needs to understand this choice to
    /// stderr: the dropped profiles, any fallback to preference order, and a
    /// tier-2 winner.
    pub fn report(&self) {
        for warning in &self.warnings {
            diag(warning);
        }
        if let Some(reason) = &self.degraded {
            diag(format_args!("falling back to configured order: {reason}"));
        }
        if let Some(note) = self.tier_note() {
            diag(note);
        }
        for note in &self.key_notes {
            diag(format_args!("note: {note}"));
        }
    }

    /// Says why the chosen account came from tier 2; `None` when it came from
    /// tier 1 or the choice fell back to preference order.
    pub fn tier_note(&self) -> Option<String> {
        let winner = self
            .scored
            .iter()
            .find(|s| s.usable && s.profile == self.account.profile)?;
        let reason = winner.tier_two_reason()?;
        Some(format!("{} chosen from tier 2: {reason}", winner.profile))
    }
}

/// The numbers a score is computed from.
struct Features {
    util5: f64,
    /// The profile's resolved 5-hour floor and knee, remaining percent.
    floor: f64,
    knee: f64,
    gate5: f64,
    /// `None`: no window matched `general_windows`; the account has no general
    /// weekly bucket.
    general: Option<Bucket>,
    /// `None`: no window matched `fable_windows`; the account is not entitled
    /// to Fable.
    fable: Option<Bucket>,
    /// The account's priority knobs for the model being scored.
    priority: AccountPriority,
    /// The weekly-reset override against each bucket; `None` without an
    /// override or without the bucket.
    moved_g: Option<ResetApplied>,
    moved_f: Option<ResetApplied>,
}

/// The API keys a selection may overflow to.
#[derive(Clone, Copy)]
pub struct Keys<'a> {
    /// False for a turn at the terminal: no key is considered.
    pub eligible: bool,
    /// Every configured key's budget view; called at most once per selection,
    /// and only when a key passes the gates that need no budget.
    pub budget: &'a dyn Fn() -> BTreeMap<String, BudgetView>,
}

/// What selection made of the API keys.
#[derive(Debug, Clone, PartialEq)]
pub enum KeyDecision {
    /// No key is configured, or keys are not eligible for this turn.
    Ineligible,
    /// A profile with a positive score won; keys were not looked at.
    ProfileWon,
    /// Usage data is unavailable, so preference order picks a profile.
    UsageUnknown,
    /// The keys were looked at: the one taken, or why each was passed over.
    Looked {
        took: Option<String>,
        passed: Vec<(String, String)>,
    },
}

/// The selected account, with the table behind the decision.
#[derive(Debug, Clone)]
pub struct Choice {
    pub account: Account,
    pub scored: Vec<Scored>,
    /// Set when the choice fell back to preference order instead of a score.
    pub degraded: Option<String>,
    /// Profiles dropped before scoring, and why.
    pub warnings: Vec<String>,
    pub keys: KeyDecision,
    /// What the key rule has to say about this choice, one line each.
    pub key_notes: Vec<String>,
}

/// Pick the account for a Claude run: drop profiles with no local credential,
/// score the rest against the retained usage, and take the best profile with
/// a positive score. When none has one, an API key takes the run; when usage
/// data is unavailable, preference order does.
///
/// Profiles and keys named in `exclude` are not considered at all.
///
/// The only error is having nothing to run as: no credentialed profile left
/// and no key taken. Not called with empty `profiles`: that is the ambient
/// login, which has no account to choose.
#[allow(clippy::too_many_arguments)]
pub fn choose(
    claude: &ClaudeConfig,
    tuning: &Tuning,
    env: &dyn EnvSource,
    model: &str,
    now: Timestamp,
    usage: &UsageSource,
    exclude: &[String],
    keys: Keys<'_>,
) -> Result<Choice> {
    let mut candidates: Vec<Account> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut looked_for: Vec<String> = Vec::new();
    let excluded: Vec<&String> = claude
        .profiles
        .iter()
        .filter(|p| exclude.contains(p))
        .collect();
    for profile in claude.profiles.iter().filter(|p| !exclude.contains(p)) {
        let account = Account::from_config(claude, profile, env);
        match account.require_credential() {
            Ok(()) => candidates.push(account),
            Err(missing) => {
                warnings.push(format!("{missing}; not a candidate"));
                looked_for.push(account.credential_path().display().to_string());
            }
        }
    }

    let names: Vec<String> = candidates.iter().map(|a| a.profile.clone()).collect();
    let (report, mut degraded) = match usage {
        UsageSource::Unavailable(reason) => {
            (None, Some(format!("usage data unavailable: {reason}")))
        }
        UsageSource::Report(r) if !r.daemon_online() => (
            None,
            Some("the usage daemon reports itself offline".to_string()),
        ),
        UsageSource::Report(r) => (Some(r), None),
    };

    let scored = if names.is_empty() {
        Vec::new()
    } else {
        match report {
            Some(r) => score_all(&claude.selection, tuning, &names, r, model, now),
            None => {
                let reason = degraded.clone().unwrap_or_default();
                names
                    .iter()
                    .map(|p| unscored(tuning, p, model, Out::NoData, reason.clone()))
                    .collect::<Vec<_>>()
            }
        }
    };

    let best = best_of(&scored);
    let winner = |profile: &str| {
        candidates
            .iter()
            .find(|a| a.profile == profile)
            .cloned()
            .expect("the scored profiles are the candidate profiles")
    };
    let wins_outright = best.as_deref().and_then(|profile| {
        scored
            .iter()
            .find(|s| s.profile == profile && s.score > 0.0)
            .map(|_| profile)
    });
    let mut key_notes = Vec::new();
    // TODO(api-key-priority-over-quota): keys never outrank a profile with quota; see TODO.md.
    let decision = if !keys.eligible || claude.api_keys.is_empty() {
        KeyDecision::Ineligible
    } else if wins_outright.is_some() {
        KeyDecision::ProfileWon
    } else if degraded.is_some() && !candidates.is_empty() {
        KeyDecision::UsageUnknown
    } else {
        // No profile can take the work: the key does.
        let cannot: Vec<String> = excluded
            .iter()
            .map(|p| format!("{p}: set aside"))
            .chain(
                scored
                    .iter()
                    .map(|s| format!("{}: {}", s.profile, profile_state(s))),
            )
            .collect();
        let (took, passed) = pick_key(claude, tuning, env, model, exclude, keys.budget);
        if let Some((name, view)) = &took {
            key_notes.push(format!(
                "no profile can take {model} ({}); using API key {name}: ${:.2} of ${:.2} spent this month{}",
                cannot.join("; "),
                view.spent_usd,
                view.budget_usd,
                if view.remote {
                    String::new()
                } else {
                    format!(" ({})", crate::budget::THIS_HOST_ONLY)
                },
            ));
        } else {
            key_notes.extend(
                passed
                    .iter()
                    .map(|(name, why)| format!("API key {name} passed over: {why}")),
            );
        }
        KeyDecision::Looked {
            took: took.map(|(name, _)| name),
            passed,
        }
    };

    if let KeyDecision::Looked {
        took: Some(name), ..
    } = &decision
    {
        let account = Account::for_key(claude, name, env).expect("a taken key is a configured key");
        return Ok(Choice {
            account,
            scored,
            degraded: None,
            warnings,
            keys: decision,
            key_notes,
        });
    }

    if candidates.is_empty() {
        let suffix = if key_notes.is_empty() {
            String::new()
        } else {
            format!("; {}", key_notes.join("; "))
        };
        if looked_for.is_empty() {
            bail!(
                "no configured Claude profile is left once {} is set aside{suffix}",
                exclude.join(", ")
            );
        }
        bail!(
            "no configured Claude profile has a local credential; looked for {}{suffix}",
            looked_for.join(", ")
        );
    }

    let account = match best {
        Some(profile) => winner(&profile),
        None => {
            if degraded.is_none() {
                degraded = Some(format!(
                    "no candidate is usable ({})",
                    scored
                        .iter()
                        .map(|s| format!(
                            "{}: {}",
                            s.profile,
                            s.reason.as_deref().unwrap_or("unknown")
                        ))
                        .collect::<Vec<_>>()
                        .join("; ")
                ));
            }
            candidates[0].clone()
        }
    };

    Ok(Choice {
        account,
        scored,
        degraded,
        warnings,
        keys: decision,
        key_notes,
    })
}

/// A scored profile's state in one phrase: its unusable reason, or why a
/// usable one scores zero.
fn profile_state(s: &Scored) -> String {
    match &s.reason {
        Some(reason) => reason.clone(),
        None => "weekly score is zero".to_string(),
    }
}

/// Each key passed over, by name, with why.
type Passed = Vec<(String, String)>;

/// The first key that is not excluded, has a key file, matches `model` and
/// is in budget, trying keys by descending weight for `model` and then in name
/// order; and why every other key was passed over. `budget` is called once,
/// and not at all when no key passes the gates that need no budget.
fn pick_key(
    claude: &ClaudeConfig,
    tuning: &Tuning,
    env: &dyn EnvSource,
    model: &str,
    exclude: &[String],
    budget: &dyn Fn() -> BTreeMap<String, BudgetView>,
) -> (Option<(String, BudgetView)>, Passed) {
    let mut passed = Vec::new();
    let mut open = Vec::new();
    for (name, key) in &claude.api_keys {
        if let Some(why) = key_gate(claude, env, model, exclude, name, key) {
            passed.push((name.clone(), why));
        } else {
            open.push(name.clone());
        }
    }
    if open.is_empty() {
        return (None, passed);
    }
    // A stable sort keeps name order among equal weights.
    open.sort_by(|a, b| {
        let weight = |name: &str| tuning.priority(name, model).weight;
        weight(b).total_cmp(&weight(a))
    });
    let views = budget();
    let mut took = None;
    for name in open {
        match views.get(&name) {
            Some(view) if took.is_none() && view.in_budget() => {
                took = Some((name, view.clone()));
            }
            Some(_) if took.is_some() => {}
            Some(view) => passed.push((name, budget_reason(view))),
            None => passed.push((name, "no budget view".to_string())),
        }
    }
    passed.sort();
    (took, passed)
}

/// Why `name` cannot take this turn whatever its budget, or `None`.
fn key_gate(
    claude: &ClaudeConfig,
    env: &dyn EnvSource,
    model: &str,
    exclude: &[String],
    name: &str,
    key: &ApiKeyConfig,
) -> Option<String> {
    if exclude.iter().any(|e| e == name) {
        return Some("set aside".to_string());
    }
    let account = Account::for_key(claude, name, env)?;
    if !account.has_credential() {
        return Some(format!(
            "no key file at {}",
            account.credential_path().display()
        ));
    }
    if !any_match(&key.models, model) {
        return Some("models do not match".to_string());
    }
    None
}

/// Why a key with this view is not in budget.
pub fn budget_reason(view: &BudgetView) -> String {
    if let Some(e) = &view.ledger_error {
        return format!("ledger unreadable: {e}");
    }
    match &view.exhausted {
        Some(flag) => format!("exhausted since {}", flag.at),
        None => format!(
            "${:.2} of ${:.2} spent, reserve ${:.2}",
            view.spent_usd, view.budget_usd, view.reserve_usd
        ),
    }
}

/// One configured key as `ccc which` shows it.
#[derive(Debug, Clone, PartialEq)]
pub struct KeyRow {
    pub name: String,
    pub view: Option<BudgetView>,
    pub models_match: bool,
    pub used: bool,
    /// Why the key would not be used; `None` when it would.
    pub why_not: Option<String>,
    /// The weight keys are tried by for the model; 1 when no knob sets one.
    pub weight: f64,
    /// The 0-based `by_model` rule `weight` came from, if any.
    pub weight_rule: Option<usize>,
}

/// Every configured key, in name order, against `choice`.
pub fn key_rows(
    claude: &ClaudeConfig,
    tuning: &Tuning,
    model: &str,
    choice: &Choice,
    views: &BTreeMap<String, BudgetView>,
) -> Vec<KeyRow> {
    claude
        .api_keys
        .iter()
        .map(|(name, key)| {
            let used = choice.account.profile == *name && choice.account.key_file().is_some();
            let why_not = (!used).then(|| match &choice.keys {
                KeyDecision::Ineligible => "keys are not eligible for this turn".to_string(),
                KeyDecision::ProfileWon => "a profile can take the work".to_string(),
                KeyDecision::UsageUnknown => "usage data is unavailable".to_string(),
                KeyDecision::Looked { took, passed } => passed
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, why)| why.clone())
                    .unwrap_or_else(|| match took {
                        Some(took) => format!("key {took} was taken first"),
                        None => "not tried".to_string(),
                    }),
            });
            let priority = tuning.priority(name, model);
            KeyRow {
                name: name.clone(),
                view: views.get(name).cloned(),
                models_match: any_match(&key.models, model),
                used,
                why_not,
                weight: priority.weight,
                weight_rule: priority.weight_rule,
            }
        })
        .collect()
}

/// Score every candidate, applying the Fable reserve's all-zero fallback, then
/// note why each usable one below tier 1 is there.
pub fn score_all(
    sel: &SelectionConfig,
    tuning: &Tuning,
    profiles: &[String],
    report: &UsageReport,
    model: &str,
    now: Timestamp,
) -> Vec<Scored> {
    let fable = any_match(&sel.fable_models, model);
    let mut scored = score_pass(
        sel,
        tuning,
        profiles,
        report,
        model,
        now,
        sel.reserve_general_for_fable,
    );

    // A reserve that zeroes every candidate would starve non-Fable work for the
    // sake of Fable quota nobody is spending; re-score without it. With no
    // usable candidate at all there is nothing the reserve did, so no re-score.
    let mut usable = scored.iter().filter(|s| s.usable).peekable();
    let all_zero = usable.peek().is_some() && usable.all(|s| s.score <= 0.0);
    if !fable && sel.reserve_general_for_fable && all_zero {
        scored = score_pass(sel, tuning, profiles, report, model, now, false);
        for s in scored.iter_mut().filter(|s| s.usable) {
            s.notes
                .push("scored without the Fable reserve (it zeroed every candidate)".into());
        }
    }
    for s in scored.iter_mut().filter(|s| s.usable) {
        note_tier(s);
    }
    scored
}

/// Tier 1 has 5-hour headroom and something to spend; a tier-2 profile gets a
/// note saying which way it fell short.
fn note_tier(s: &mut Scored) {
    if let Some(reason) = s.tier_two_reason() {
        s.notes.push(reason.into());
    }
}

fn score_pass(
    sel: &SelectionConfig,
    tuning: &Tuning,
    profiles: &[String],
    report: &UsageReport,
    model: &str,
    now: Timestamp,
    reserve: bool,
) -> Vec<Scored> {
    profiles
        .iter()
        .map(|profile| match report.profile(profile) {
            None => unscored(
                tuning,
                profile,
                model,
                Out::NoSnapshot,
                "no usage snapshot published".to_string(),
            ),
            Some(usage) => score_one(sel, tuning, model, usage, now, reserve),
        })
        .collect()
}

/// A candidate with no numbers behind it: unusable, and the table says why.
fn unscored(tuning: &Tuning, profile: &str, model: &str, out: Out, reason: String) -> Scored {
    let (floor, knee) = tuning.ramp(profile);
    Scored {
        profile: profile.to_string(),
        util5: 0.0,
        general: None,
        fable: None,
        five_hour_floor_pct: floor,
        five_hour_knee_pct: knee,
        weight: tuning.priority(profile, model).weight,
        gate5: 0.0,
        score: 0.0,
        usable: false,
        reason: Some(reason),
        out: Some(out),
        notes: Vec::new(),
    }
}

fn score_one(
    sel: &SelectionConfig,
    tuning: &Tuning,
    model: &str,
    usage: &ProfileUsage,
    now: Timestamp,
    reserve: bool,
) -> Scored {
    let fable = any_match(&sel.fable_models, model);
    let f = features(sel, tuning, model, usage, now);
    let mut out = Scored {
        profile: usage.profile.clone(),
        util5: f.util5,
        general: f.general,
        fable: f.fable,
        five_hour_floor_pct: f.floor,
        five_hour_knee_pct: f.knee,
        weight: f.priority.weight,
        gate5: f.gate5,
        score: 0.0,
        usable: false,
        reason: None,
        out: None,
        notes: Vec::new(),
    };

    let spend = match eligible(sel, usage, now, &f, fable) {
        Ok(spend) => spend,
        Err((why, reason)) => {
            out.reason = Some(reason);
            out.out = Some(why);
            return out;
        }
    };

    let ratio = sel.fable_general_ratio;
    let unweighted = match spend {
        Spend::Fable { general, fable } => {
            let rate_g = (100.0 - general.util) / general.hours;
            let rate_f = (100.0 - fable.util) / fable.hours;
            // Fable spends both buckets, so the binding constraint is whichever
            // runs out first, expressed in Fable-bucket units.
            let general_in_fable_units = if ratio > 0.0 {
                rate_g / ratio
            } else {
                f64::INFINITY
            };
            f.gate5 * rate_f.min(general_in_fable_units)
        }
        Spend::General {
            general,
            reserve_against,
        } => {
            // An account with no Fable bucket has nothing to reserve for.
            let reserve_for_fable = match (reserve, reserve_against) {
                (true, Some(bucket_f)) => ratio * (100.0 - bucket_f.util),
                _ => 0.0,
            };
            let effective = (100.0 - general.util) - reserve_for_fable;
            f.gate5 * effective.max(0.0) / general.hours
        }
    };
    out.score = f.priority.weight * unweighted;
    out.usable = true;
    out.notes = knob_notes(&f);
    out
}

/// What the account's knobs did to its score, for a usable candidate: its
/// weight when not 1, then per bucket an override that moved its reset or was
/// not applied.
fn knob_notes(f: &Features) -> Vec<String> {
    let p = &f.priority;
    let mut notes: Vec<String> = weight_note(p.weight, p.weight_rule).into_iter().collect();
    let at = match p.weekly_reset {
        Some(crate::tuning::WeeklyReset::At(t)) => t.to_string(),
        _ => String::new(),
    };
    for (bucket, applied) in [("general", f.moved_g), ("Fable", f.moved_f)] {
        let note = match applied {
            None => continue,
            Some(ResetApplied::Moved { from, to }) if from == to => continue,
            Some(ResetApplied::Moved { from, to }) => format!(
                "{bucket} reset moved {} h earlier, to {to}",
                from.duration_since(to).as_secs_f64() / 3600.0
            ),
            Some(ResetApplied::NotApplied { reset, why }) => {
                let why = match why {
                    NotApplied::AfterReset => format!("after its reset {reset}"),
                    NotApplied::EarlierWeek => format!("its reset {reset} is in a later week"),
                };
                format!("weekly_reset_at {at} not applied to the {bucket} bucket: {why}")
            }
        };
        notes.push(format!("{note}{}", rule_suffix(p.reset_rule)));
    }
    notes
}

/// The buckets one model draws on, in the shape that model requires.
enum Spend {
    /// A model that spends the general bucket only, plus the Fable bucket the
    /// reserve holds back for when the account has one.
    General {
        general: Bucket,
        reserve_against: Option<Bucket>,
    },
    /// A Fable model: both buckets, both known present.
    Fable { general: Bucket, fable: Bucket },
}

/// What a model spending `fable` may draw on for this profile, or why the
/// profile must not be used. Returns validated buckets so callers work with
/// values, not `Option`s.
fn eligible(
    sel: &SelectionConfig,
    usage: &ProfileUsage,
    now: Timestamp,
    f: &Features,
    fable: bool,
) -> Result<Spend, (Out, String)> {
    if usage.auth != AuthState::Ok {
        return Err((Out::Auth, format!("auth is {}", usage.auth.as_str())));
    }
    if let Some(why) = &usage.stale_reason {
        // The daemon's text, which can quote an upstream error body, reaches
        // the terminal through `ccc which` and the fallback diagnostic.
        return Err((
            Out::CarriedForward,
            format!("the reading is carried forward ({})", sanitize(why)),
        ));
    }
    let age = now.duration_since(usage.updated_at).as_secs_f64();
    if age > sel.stale_grace_s as f64 {
        return Err((
            Out::Stale,
            format!(
                "the reading is {:.0} min old (grace {} s)",
                age / 60.0,
                sel.stale_grace_s
            ),
        ));
    }
    if f.gate5 <= 0.0 {
        return Err((
            Out::FiveHourFloor,
            format!(
                "the 5-hour window is at or below its floor ({:.0} % left, floor {} %)",
                100.0 - f.util5,
                f.floor
            ),
        ));
    }
    // Every model spends the general bucket.
    let Some(general) = f.general else {
        return Err((
            Out::NoGeneralWindow,
            "no general weekly window in the snapshot".to_string(),
        ));
    };
    if !fable {
        return Ok(Spend::General {
            general,
            reserve_against: f.fable,
        });
    }
    let Some(fable) = f.fable else {
        return Err((
            Out::NoFableWindow,
            "no Fable weekly window in the snapshot".to_string(),
        ));
    };
    Ok(Spend::Fable { general, fable })
}

fn features(
    sel: &SelectionConfig,
    tuning: &Tuning,
    model: &str,
    usage: &ProfileUsage,
    now: Timestamp,
) -> Features {
    // An absent 5-hour window is a window nothing has been spent in yet.
    let util5 = usage
        .window(FIVE_HOUR)
        .map(|w| w.utilization)
        .unwrap_or(0.0);

    let fresh_week = now
        .checked_add(SignedDuration::from_hours(WEEK_HOURS))
        .unwrap_or(now);

    let priority = tuning.priority(&usage.profile, model);
    // The override moves each bucket's reset before the floor applies, so a
    // moved reset already past counts as `min_time_left_h`.
    let applied = |reset: Timestamp| priority.weekly_reset.map(|r| r.apply(reset));
    let effective =
        |reset: Timestamp, applied: Option<ResetApplied>| applied.map_or(reset, |a| a.effective());

    let window_g = tightest(usage, &sel.general_windows);
    // A weekly window the daemon published without a reset time still exists;
    // the fallback clock only stands in for that gap.
    let reset_g = window_g.and_then(|w| w.resets_at).unwrap_or(fresh_week);
    let moved_g = window_g.and_then(|_| applied(reset_g));
    let general = window_g.map(|w| {
        let resets_at = effective(reset_g, moved_g);
        Bucket {
            util: w.utilization,
            hours: hours_until(now, resets_at, sel.min_time_left_h),
            resets_at,
        }
    });
    let window_f = tightest(usage, &sel.fable_windows);
    let reset_f = window_f.map(|w| w.resets_at.unwrap_or(reset_g));
    let moved_f = reset_f.and_then(applied);
    let fable = window_f.zip(reset_f).map(|(w, reset)| {
        let resets_at = effective(reset, moved_f);
        Bucket {
            util: w.utilization,
            hours: hours_until(now, resets_at, sel.min_time_left_h),
            resets_at,
        }
    });

    let (floor, knee) = tuning.ramp(&usage.profile);
    Features {
        util5,
        floor,
        knee,
        gate5: gate5(floor, knee, util5),
        general,
        fable,
        priority,
        moved_g,
        moved_f,
    }
}

/// The tightest (highest-utilisation) window matching any of `patterns`. The
/// weekly keys are scoped per model and come and go, so they are matched as
/// globs and pooled rather than named.
fn tightest<'a>(
    usage: &'a ProfileUsage,
    patterns: &[String],
) -> Option<&'a crate::usage::UsageWindow> {
    usage
        .windows
        .iter()
        .filter(|w| any_match(patterns, &w.key))
        .max_by(|a, b| a.utilization.total_cmp(&b.utilization))
}

/// A linear ramp on 5-hour headroom: 0 at or below `floor`, 1 at or above
/// `knee`, both in remaining percent. Zero makes a profile unusable outright.
fn gate5(floor: f64, knee: f64, util5: f64) -> f64 {
    let remaining = 100.0 - util5;
    let span = knee - floor;
    if span <= 0.0 {
        return if remaining > floor { 1.0 } else { 0.0 };
    }
    ((remaining - floor) / span).clamp(0.0, 1.0)
}

/// A last-ditch floor on hours-until-reset. `min_time_left_h` is validated to be
/// positive, so this only catches a caller that built a config by hand; it keeps
/// a rate from dividing by zero and poisoning every comparison with `NaN`.
const MIN_HOURS: f64 = 1e-6;

fn hours_until(now: Timestamp, at: Timestamp, floor_h: f64) -> f64 {
    let hours = at.duration_since(now).as_secs_f64() / 3600.0;
    hours.max(floor_h).max(MIN_HOURS)
}

/// The usable profile in the best tier with the highest score; ties go to the
/// earlier one, which is configured preference order. `None` when nothing is
/// usable.
fn best_of(scored: &[Scored]) -> Option<String> {
    let mut best: Option<&Scored> = None;
    for s in scored.iter().filter(|s| s.usable) {
        let better = best.is_none_or(|b| {
            let (tier, best_tier) = (s.candidate_tier(), b.candidate_tier());
            tier < best_tier || (tier == best_tier && s.score > b.score)
        });
        if better {
            best = Some(s);
        }
    }
    best.map(|s| s.profile.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ClaudeAuth;
    use crate::tuning::{AccountLayer, Priority, TuningLayer, WeeklyReset};
    use crate::usage::UsageWindow;

    fn tun() -> Tuning {
        Tuning::default()
    }

    /// The default `Tuning` with `account`'s layer over it.
    fn tuned(account: &str, layer: AccountLayer) -> Tuning {
        Tuning::default().overlay(&TuningLayer {
            accounts: BTreeMap::from([(account.to_string(), layer)]),
            ..TuningLayer::default()
        })
    }

    fn now() -> Timestamp {
        "2026-09-18T12:00:00Z".parse().unwrap()
    }

    fn at(hours: i64) -> Option<Timestamp> {
        Some(
            now()
                .checked_add(SignedDuration::from_hours(hours))
                .unwrap(),
        )
    }

    fn window(key: &str, utilization: f64, resets_in_h: i64) -> UsageWindow {
        UsageWindow {
            key: key.to_string(),
            utilization,
            resets_at: at(resets_in_h),
        }
    }

    /// A live-shaped snapshot: 5-hour, general weekly and Fable weekly.
    fn profile(name: &str, util5: f64, util_g: f64, util_f: Option<f64>) -> ProfileUsage {
        let mut windows = vec![
            window(FIVE_HOUR, util5, 3),
            window("seven_day", util_g, 100),
        ];
        if let Some(f) = util_f {
            windows.push(window("7-day_fable", f, 100));
        }
        ProfileUsage {
            profile: name.to_string(),
            auth: AuthState::Ok,
            stale_reason: None,
            updated_at: now(),
            windows,
        }
    }

    fn report(profiles: Vec<ProfileUsage>) -> UsageReport {
        UsageReport {
            profiles,
            daemon_status: Some("online".to_string()),
        }
    }

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn live() -> UsageReport {
        report(vec![
            profile("team03", 31.0, 86.0, Some(77.0)),
            profile("team04", 19.0, 90.0, Some(83.0)),
            profile("personal", 0.0, 7.0, None),
        ])
    }

    #[test]
    fn the_five_hour_gate_ramps_between_floor_and_knee() {
        let (floor, knee) = tun().ramp("any");
        assert_eq!(gate5(floor, knee, 99.0), 0.0);
        assert_eq!(gate5(floor, knee, 85.0), 1.0);
        assert_eq!(gate5(floor, knee, 0.0), 1.0);
        assert!((gate5(floor, knee, 92.0) - 0.5).abs() < 1e-9);
        assert_eq!(gate5(floor, knee, 99.5), 0.0);
    }

    #[test]
    fn a_collapsed_ramp_is_a_step_at_the_floor() {
        // floor == knee leaves no ramp to interpolate over: anything above the
        // floor scores 1, anything at or below it scores 0.
        assert_eq!(gate5(10.0, 10.0, 89.0), 1.0);
        assert_eq!(gate5(10.0, 10.0, 90.0), 0.0);
        assert_eq!(gate5(10.0, 10.0, 95.0), 0.0);
        assert!(gate5(10.0, 10.0, 0.0).is_finite());

        // An inverted ramp is refused by config validation, but the scorer
        // still answers a step rather than a negative or NaN gate.
        assert_eq!(gate5(15.0, 5.0, 80.0), 1.0);
        assert_eq!(gate5(15.0, 5.0, 90.0), 0.0);
    }

    #[test]
    fn a_zero_ratio_means_general_quota_never_binds_fable_work() {
        // With the ratio off, Fable work costs nothing from the general bucket,
        // so the Fable rate alone decides — and the score stays finite.
        let mut sel = SelectionConfig::default();
        sel.fable_general_ratio = 0.0;
        let usage = profile("team03", 0.0, 99.0, Some(50.0));
        let scored = score_one(&sel, &tun(), FABLE, &usage, now(), true);
        assert!(scored.usable);
        assert!(scored.score.is_finite(), "{}", scored.score);
        let expected = (100.0 - 50.0) / 100.0;
        assert!((scored.score - expected).abs() < 1e-9, "{}", scored.score);

        let plain = score_one(&sel, &tun(), OPUS, &usage, now(), true);
        assert!(
            (plain.score - (100.0 - 99.0) / 100.0).abs() < 1e-9,
            "{}",
            plain.score
        );
    }

    #[test]
    fn a_reset_already_past_cannot_divide_by_zero() {
        // A daemon that carries an expired reset for a few minutes, under a
        // hand-built config with no floor, must still produce a finite score.
        let mut sel = SelectionConfig::default();
        sel.min_time_left_h = 0.0;
        let usage = ProfileUsage {
            profile: "team03".into(),
            auth: AuthState::Ok,
            stale_reason: None,
            updated_at: now(),
            windows: vec![
                window(FIVE_HOUR, 0.0, 3),
                window("seven_day", 100.0, -1),
                window("7-day_fable", 100.0, -1),
            ],
        };
        let scored = score_one(&sel, &tun(), OPUS, &usage, now(), true);
        assert!(scored.score.is_finite(), "{}", scored.score);
        let fable = score_one(&sel, &tun(), FABLE, &usage, now(), true);
        assert!(fable.score.is_finite(), "{}", fable.score);
    }

    #[test]
    fn general_windows_pool_tightest_wins_and_fable_matches_its_own_globs() {
        let sel = SelectionConfig::default();
        let usage = ProfileUsage {
            profile: "team03".into(),
            auth: AuthState::Ok,
            stale_reason: None,
            updated_at: now(),
            windows: vec![
                window(FIVE_HOUR, 10.0, 3),
                window("seven_day", 40.0, 100),
                window("7-day_opus_4_8", 70.0, 50),
                window("7-day_fable", 77.0, 100),
            ],
        };
        let f = features(&sel, &tun(), "m", &usage, now());
        let g = f.general.unwrap();
        assert_eq!(g.util, 70.0);
        assert!((g.hours - 50.0).abs() < 1e-9);
        let b = f.fable.unwrap();
        assert_eq!(b.util, 77.0);
        assert!((b.hours - 100.0).abs() < 1e-9);
    }

    #[test]
    fn a_missing_weekly_window_is_a_bucket_the_account_does_not_have() {
        let sel = SelectionConfig::default();
        let usage = ProfileUsage {
            profile: "personal".into(),
            auth: AuthState::Ok,
            stale_reason: None,
            updated_at: now(),
            windows: vec![window(FIVE_HOUR, 0.0, 3), window("seven_day", 7.0, 24)],
        };
        let f = features(&sel, &tun(), "m", &usage, now());
        assert_eq!(f.fable, None);
        let g = f.general.unwrap();
        assert_eq!(g.util, 7.0);
        assert!((g.hours - 24.0).abs() < 1e-9);

        let bare = ProfileUsage {
            windows: vec![window(FIVE_HOUR, 0.0, 3)],
            ..usage
        };
        let f = features(&sel, &tun(), "m", &bare, now());
        assert_eq!(f.general, None);
        assert_eq!(f.fable, None);
    }

    #[test]
    fn a_present_weekly_window_without_a_reset_gets_a_full_week() {
        let sel = SelectionConfig::default();
        let usage = ProfileUsage {
            profile: "team03".into(),
            auth: AuthState::Ok,
            stale_reason: None,
            updated_at: now(),
            windows: vec![
                window(FIVE_HOUR, 0.0, 3),
                UsageWindow {
                    resets_at: None,
                    ..window("seven_day", 7.0, 0)
                },
                UsageWindow {
                    resets_at: None,
                    ..window("7-day_fable", 9.0, 0)
                },
            ],
        };
        let f = features(&sel, &tun(), "m", &usage, now());
        assert!((f.general.unwrap().hours - WEEK_HOURS as f64).abs() < 1e-9);
        assert!((f.fable.unwrap().hours - WEEK_HOURS as f64).abs() < 1e-9);
    }

    #[test]
    fn a_fable_window_without_a_reset_runs_on_the_general_windows_clock() {
        let sel = SelectionConfig::default();
        let usage = ProfileUsage {
            profile: "team03".into(),
            auth: AuthState::Ok,
            stale_reason: None,
            updated_at: now(),
            windows: vec![
                window(FIVE_HOUR, 0.0, 3),
                window("seven_day", 7.0, 50),
                UsageWindow {
                    resets_at: None,
                    ..window("7-day_fable", 9.0, 0)
                },
            ],
        };
        let f = features(&sel, &tun(), "m", &usage, now());
        assert!((f.general.unwrap().hours - 50.0).abs() < 1e-9);
        assert!(
            (f.fable.unwrap().hours - 50.0).abs() < 1e-9,
            "{:?}",
            f.fable
        );
    }

    #[test]
    fn an_absent_five_hour_window_is_an_unspent_one() {
        let sel = SelectionConfig::default();
        let usage = ProfileUsage {
            profile: "team03".into(),
            auth: AuthState::Ok,
            stale_reason: None,
            updated_at: now(),
            windows: vec![window("seven_day", 40.0, 100)],
        };
        let f = features(&sel, &tun(), "m", &usage, now());
        assert_eq!(f.util5, 0.0);
        assert_eq!(f.gate5, 1.0);
        assert!(score_one(&sel, &tun(), OPUS, &usage, now(), true).usable);
    }

    #[test]
    fn a_fable_model_cannot_use_an_account_without_a_fable_window() {
        let sel = SelectionConfig::default();
        let usage = profile("personal", 0.0, 7.0, None);
        let s = score_one(&sel, &tun(), FABLE, &usage, now(), true);
        assert!(!s.usable);
        assert!(s.reason.as_deref().unwrap().contains("Fable"), "{s:?}");
        assert_eq!(s.score, 0.0);

        assert!(score_one(&sel, &tun(), OPUS, &usage, now(), true).usable);
    }

    #[test]
    fn no_general_window_makes_a_profile_unusable_for_every_model() {
        let sel = SelectionConfig::default();
        let usage = ProfileUsage {
            profile: "team03".into(),
            auth: AuthState::Ok,
            stale_reason: None,
            updated_at: now(),
            windows: vec![window(FIVE_HOUR, 0.0, 3), window("7-day_fable", 10.0, 100)],
        };
        for fable in [false, true] {
            let s = score_one(
                &sel,
                &tun(),
                if fable { FABLE } else { OPUS },
                &usage,
                now(),
                true,
            );
            assert!(!s.usable, "{fable}");
            assert!(
                s.reason.as_deref().unwrap().contains("general weekly"),
                "{s:?}"
            );
        }
    }

    #[test]
    fn the_reserve_holds_nothing_back_for_an_account_with_no_fable_bucket() {
        let sel = SelectionConfig::default();
        // 60% of the general bucket left over 100h, and no Fable bucket to
        // reserve any of it for.
        let s = score_one(
            &sel,
            &tun(),
            OPUS,
            &profile("a", 0.0, 40.0, None),
            now(),
            true,
        );
        assert!((s.score - 0.6).abs() < 1e-9, "{s:?}");
    }

    #[test]
    fn scoring_knows_which_buckets_the_model_spends() {
        let sel = SelectionConfig::default();
        let usable = |model: &str| {
            score_all(&sel, &tun(), &names(&["personal"]), &live(), model, now()).remove(0)
        };
        assert!(usable("claude-opus-5[1M]").usable);
        let fable = usable("claude-fable-5-1[1m]");
        assert!(!fable.usable);
        assert!(
            fable.reason.as_deref().unwrap().contains("Fable"),
            "{fable:?}"
        );
    }

    #[test]
    fn hours_until_is_floored_so_an_imminent_reset_cannot_explode_a_rate() {
        let sel = SelectionConfig::default();
        let usage = ProfileUsage {
            windows: vec![window(FIVE_HOUR, 0.0, 1), window("seven_day", 50.0, 0)],
            ..profile("team03", 0.0, 50.0, None)
        };
        let f = features(&sel, &tun(), "m", &usage, now());
        assert_eq!(f.general.unwrap().hours, sel.min_time_left_h);
    }

    #[test]
    fn every_unusable_reason_excludes_the_profile() {
        let sel = SelectionConfig::default();
        let cases: Vec<(ProfileUsage, &str)> = vec![
            (
                ProfileUsage {
                    auth: AuthState::Stale,
                    ..profile("a", 0.0, 0.0, None)
                },
                "auth is stale",
            ),
            (
                ProfileUsage {
                    stale_reason: Some("rate limited by the API".to_string()),
                    ..profile("a", 0.0, 0.0, None)
                },
                "the reading is carried forward (rate limited by the API)",
            ),
            (
                ProfileUsage {
                    updated_at: now().checked_sub(SignedDuration::from_hours(2)).unwrap(),
                    ..profile("a", 0.0, 0.0, None)
                },
                "old",
            ),
            (
                profile("a", 99.5, 0.0, None),
                "5-hour window is at or below its floor (0 % left, floor 1 %)",
            ),
            (
                ProfileUsage {
                    windows: vec![window(FIVE_HOUR, 0.0, 3)],
                    ..profile("a", 0.0, 0.0, None)
                },
                "no general weekly window",
            ),
        ];
        for (usage, expected) in cases {
            let s = score_one(&sel, &tun(), OPUS, &usage, now(), true);
            assert!(!s.usable, "{expected}");
            assert_eq!(s.score, 0.0);
            let reason = s.reason.unwrap();
            assert!(
                reason.contains(expected),
                "{reason} does not mention {expected}"
            );
        }

        let usage = ProfileUsage {
            stale_reason: Some("rate\nlimited\r\x1b[2J".to_string()),
            ..profile("a", 0.0, 0.0, None)
        };
        let reason = score_one(&sel, &tun(), OPUS, &usage, now(), true)
            .reason
            .unwrap();
        assert_eq!(
            reason,
            "the reading is carried forward (rate\\nlimited\\r\\x1b[2J)"
        );

        // The Fable bucket only excludes a profile from Fable work.
        let s = score_one(
            &sel,
            &tun(),
            FABLE,
            &profile("a", 0.0, 0.0, None),
            now(),
            true,
        );
        assert!(!s.usable);
        assert_eq!(s.score, 0.0);
        let reason = s.reason.unwrap();
        assert!(reason.contains("no Fable weekly window"), "{reason}");
    }

    #[test]
    fn a_fable_model_is_bound_by_whichever_bucket_runs_out_first() {
        let sel = SelectionConfig::default();
        // General 60% left over 100h = 0.6%/h; in Fable units 1.2%/h. Fable has
        // 20% left over 100h = 0.2%/h, so Fable binds.
        let fable_binds = score_one(
            &sel,
            &tun(),
            FABLE,
            &profile("a", 0.0, 40.0, Some(80.0)),
            now(),
            true,
        );
        assert!((fable_binds.score - 0.2).abs() < 1e-9);

        // Fable 90% left (0.9%/h) but general only 10% left: 0.1%/h general is
        // 0.2%/h of Fable, which binds instead.
        let general_binds = score_one(
            &sel,
            &tun(),
            FABLE,
            &profile("a", 0.0, 90.0, Some(10.0)),
            now(),
            true,
        );
        assert!((general_binds.score - 0.2).abs() < 1e-9);

        // The gate damps the whole score.
        let gated = score_one(
            &sel,
            &tun(),
            FABLE,
            &profile("a", 92.0, 40.0, Some(80.0)),
            now(),
            true,
        );
        assert!((gated.score - 0.1).abs() < 1e-9);
    }

    #[test]
    fn a_fable_model_is_recognised_by_the_model_globs() {
        let sel = SelectionConfig::default();
        let r = report(vec![profile("a", 0.0, 40.0, Some(80.0))]);
        let fable = &score_all(
            &sel,
            &tun(),
            &names(&["a"]),
            &r,
            "claude-fable-5-1[1m]",
            now(),
        )[0];
        let general = &score_all(&sel, &tun(), &names(&["a"]), &r, "claude-opus-5[1M]", now())[0];
        assert!((fable.score - 0.2).abs() < 1e-9);
        // Non-Fable sees 60% general less the 0.5 × 20% Fable reserve = 50%.
        assert!((general.score - 0.5).abs() < 1e-9);
    }

    #[test]
    fn the_reserve_falls_back_to_unreserved_when_it_zeroes_everyone() {
        let sel = SelectionConfig::default();
        // 20% general left, 90% Fable left: the reserve wants 45% and there is
        // not that much, so reserved scoring gives every candidate zero.
        let r = report(vec![
            profile("a", 0.0, 80.0, Some(10.0)),
            profile("b", 0.0, 90.0, Some(10.0)),
        ]);
        let scored = score_all(
            &sel,
            &tun(),
            &names(&["a", "b"]),
            &r,
            "claude-opus-5[1M]",
            now(),
        );
        assert!((scored[0].score - 0.2).abs() < 1e-9, "{:?}", scored[0]);
        assert!((scored[1].score - 0.1).abs() < 1e-9);
        assert!(scored[0]
            .note()
            .unwrap()
            .contains("without the Fable reserve"));
        assert_eq!(scored[0].reason, None);
        // The re-score's survivors are tiered like any other candidates.
        assert_eq!(scored[0].tier(), Some(1));
        assert_eq!(scored[1].tier(), Some(1));
        assert_eq!(best_of(&scored).as_deref(), Some("a"));

        // With the reserve off in config there is no second pass and no note.
        let mut sel = sel;
        sel.reserve_general_for_fable = false;
        let scored = score_all(
            &sel,
            &tun(),
            &names(&["a", "b"]),
            &r,
            "claude-opus-5[1M]",
            now(),
        );
        assert!((scored[0].score - 0.2).abs() < 1e-9);
        assert_eq!(scored[0].note(), None);
    }

    #[test]
    fn the_live_snapshots_pick_the_least_loaded_account() {
        let sel = SelectionConfig::default();
        let scored = score_all(
            &sel,
            &tun(),
            &names(&["team03", "team04", "personal"]),
            &live(),
            "claude-opus-5[1M]",
            now(),
        );
        assert!(scored.iter().all(|s| s.usable), "{scored:?}");
        assert_eq!(best_of(&scored).as_deref(), Some("personal"));

        // On a Fable model personal is out: it has no Fable bucket to spend.
        let scored = score_all(
            &sel,
            &tun(),
            &names(&["team03", "team04", "personal"]),
            &live(),
            "claude-fable-5-1[1m]",
            now(),
        );
        let personal = scored.iter().find(|s| s.profile == "personal").unwrap();
        assert!(!personal.usable, "{personal:?}");
        assert!(
            personal.reason.as_deref().unwrap().contains("Fable"),
            "{personal:?}"
        );
        assert_eq!(best_of(&scored).as_deref(), Some("team03"));
        // team03 has more of both buckets left than team04.
        let t3 = scored.iter().find(|s| s.profile == "team03").unwrap();
        let t4 = scored.iter().find(|s| s.profile == "team04").unwrap();
        assert!(t3.score > t4.score, "{t3:?} vs {t4:?}");
    }

    #[test]
    fn ties_go_to_configured_order_and_an_all_zero_table_has_no_winner() {
        let sel = SelectionConfig::default();
        let r = report(vec![
            profile("b", 10.0, 50.0, Some(0.0)),
            profile("a", 10.0, 50.0, Some(0.0)),
        ]);
        let scored = score_all(
            &sel,
            &tun(),
            &names(&["a", "b"]),
            &r,
            "claude-opus-5[1M]",
            now(),
        );
        assert_eq!(scored[0].score, scored[1].score);
        assert_eq!(best_of(&scored).as_deref(), Some("a"));

        let none_usable = score_all(
            &sel,
            &tun(),
            &names(&["missing"]),
            &r,
            "claude-opus-5[1M]",
            now(),
        );
        assert_eq!(best_of(&none_usable), None);
        assert!(none_usable[0]
            .reason
            .as_deref()
            .unwrap()
            .contains("no usage snapshot"));
    }

    /// A usable candidate with the given gate and score, noted as scoring
    /// would note it.
    fn candidate(name: &str, gate5: f64, score: f64) -> Scored {
        let mut s = Scored {
            profile: name.to_string(),
            util5: 0.0,
            general: None,
            fable: None,
            five_hour_floor_pct: 1.0,
            five_hour_knee_pct: 15.0,
            weight: 1.0,
            gate5,
            score,
            usable: true,
            reason: None,
            out: None,
            notes: Vec::new(),
        };
        note_tier(&mut s);
        s
    }

    #[test]
    fn an_account_above_its_knee_beats_one_in_its_ramp_whatever_the_scores() {
        // The in-ramp candidate scores 40x higher and still loses.
        let ramp = candidate("ramp", 0.5, 2.0);
        let headroom = candidate("headroom", 1.0, 0.05);
        assert_eq!(ramp.tier(), Some(2));
        assert_eq!(headroom.tier(), Some(1));
        assert_eq!(
            best_of(&[ramp.clone(), headroom.clone()]).as_deref(),
            Some("headroom")
        );
        assert_eq!(best_of(&[headroom, ramp]).as_deref(), Some("headroom"));

        // Two in-ramp candidates rank by score.
        let deep = candidate("deep", 0.1, 0.3);
        let shallow = candidate("shallow", 0.9, 0.2);
        assert_eq!(
            best_of(&[shallow.clone(), deep.clone()]).as_deref(),
            Some("deep")
        );

        // An above-knee candidate that scores zero is tier 2 and ranks below a
        // positive in-ramp one.
        let zero = candidate("zero", 1.0, 0.0);
        assert_eq!(zero.tier(), Some(2));
        assert_eq!(
            best_of(&[zero.clone(), shallow.clone()]).as_deref(),
            Some("shallow")
        );

        // Equal tier and score: configured order.
        let twin = candidate("twin", 0.9, 0.2);
        assert_eq!(
            best_of(&[twin.clone(), shallow.clone()]).as_deref(),
            Some("twin")
        );
        assert_eq!(best_of(&[shallow, twin]).as_deref(), Some("shallow"));

        // Unusable profiles never win, whatever their fields say.
        let gone = Scored {
            usable: false,
            ..candidate("gone", 1.0, 9.0)
        };
        assert_eq!(best_of(&[gone, zero]).as_deref(), Some("zero"));
    }

    /// A `Tuning` whose `personal` profile sets its own floor and knee.
    fn with_override(floor: Option<f64>, knee: Option<f64>) -> Tuning {
        tuned(
            "personal",
            AccountLayer {
                five_hour_floor_pct: floor,
                five_hour_knee_pct: knee,
                ..AccountLayer::default()
            },
        )
    }

    #[test]
    fn a_per_profile_knee_hands_the_standard_seat_off_before_its_weekly_lead_runs_out() {
        // The 2026-09-18 snapshots with personal half way through its 5-hour
        // window: its weekly rate is ~30x the team seats'.
        let r = report(vec![
            profile("team03", 31.0, 86.0, Some(77.0)),
            profile("team04", 19.0, 90.0, Some(83.0)),
            profile("personal", 50.0, 7.0, None),
        ]);
        let all = names(&["team03", "team04", "personal"]);

        // Global knee only: 50% remaining is above the 15% knee, so personal is
        // tier 1 and its weekly lead wins.
        let sel = SelectionConfig::default();
        let scored = score_all(&sel, &tun(), &all, &r, "claude-opus-5[1M]", now());
        assert_eq!(best_of(&scored).as_deref(), Some("personal"));

        // Its own knee at 60% remaining puts it in its ramp, and a team seat
        // with headroom takes the work despite a far lower score.
        let tuning = with_override(None, Some(60.0));
        let scored = score_all(&sel, &tuning, &all, &r, "claude-opus-5[1M]", now());
        let personal = scored.iter().find(|s| s.profile == "personal").unwrap();
        let team03 = scored.iter().find(|s| s.profile == "team03").unwrap();
        assert_eq!(personal.tier(), Some(2), "{personal:?}");
        assert!(personal.score > team03.score, "{personal:?} vs {team03:?}");
        assert_eq!(best_of(&scored).as_deref(), Some("team03"));
    }

    #[test]
    fn per_profile_floor_and_knee_apply_to_that_profile_only() {
        let tuning = with_override(Some(20.0), Some(60.0));
        let personal = score_one(
            &SelectionConfig::default(),
            &tuning,
            OPUS,
            &profile("personal", 60.0, 7.0, None),
            now(),
            true,
        );
        assert_eq!(
            (personal.five_hour_floor_pct, personal.five_hour_knee_pct),
            (20.0, 60.0)
        );
        // 40% remaining, half way between floor 20 and knee 60.
        assert!((personal.gate5 - 0.5).abs() < 1e-9, "{personal:?}");
        let team = score_one(
            &SelectionConfig::default(),
            &tuning,
            OPUS,
            &profile("team03", 60.0, 7.0, None),
            now(),
            true,
        );
        assert_eq!(
            (team.five_hour_floor_pct, team.five_hour_knee_pct),
            (1.0, 15.0)
        );
        assert_eq!(team.gate5, 1.0);

        // At util5 85 personal is below its own floor and excluded; a team seat
        // at the same reading is still usable.
        let personal = score_one(
            &SelectionConfig::default(),
            &tuning,
            OPUS,
            &profile("personal", 85.0, 7.0, None),
            now(),
            true,
        );
        assert!(!personal.usable);
        assert!(
            personal.reason.as_deref().unwrap()
                == "the 5-hour window is at or below its floor (15 % left, floor 20 %)",
            "{personal:?}"
        );
        let team = score_one(
            &SelectionConfig::default(),
            &tuning,
            OPUS,
            &profile("team03", 85.0, 7.0, None),
            now(),
            true,
        );
        assert!(team.usable, "{team:?}");

        // Overriding the floor alone keeps the global knee.
        let tuning = with_override(Some(10.0), None);
        let personal = score_one(
            &SelectionConfig::default(),
            &tuning,
            OPUS,
            &profile("personal", 87.5, 7.0, None),
            now(),
            true,
        );
        assert!((personal.gate5 - 0.5).abs() < 1e-9, "{personal:?}");

        // A profile with no snapshot still reports its resolved knobs.
        let r = report(vec![]);
        let missing = &score_all(
            &SelectionConfig::default(),
            &tuning,
            &names(&["personal"]),
            &r,
            "m",
            now(),
        )[0];
        assert_eq!(missing.five_hour_floor_pct, 10.0);
        assert_eq!(missing.five_hour_knee_pct, 15.0);
        assert_eq!(missing.tier(), None);
    }

    #[test]
    fn tier_two_candidates_say_why_after_any_reserve_note() {
        let sel = SelectionConfig::default();
        let model = "claude-opus-5[1M]";

        // No reserve re-score: the tier note stands alone.
        let r = report(vec![
            profile("ramp", 92.0, 40.0, None),
            profile("spent", 0.0, 100.0, None),
            profile("fine", 0.0, 40.0, None),
        ]);
        let scored = score_all(
            &sel,
            &tun(),
            &names(&["ramp", "spent", "fine"]),
            &r,
            model,
            now(),
        );
        assert_eq!(scored[0].notes, vec!["in the 5-hour ramp"]);
        assert_eq!(scored[1].notes, vec!["weekly score is zero"]);
        assert_eq!(scored[2].notes, Vec::<String>::new());
        assert_eq!(scored[2].tier(), Some(1));

        // The reserve zeroes all three, so they are re-scored without it; the
        // reserve note comes first and the tier note follows.
        let r = report(vec![
            profile("ramp", 92.0, 80.0, Some(10.0)),
            profile("spent", 0.0, 100.0, Some(10.0)),
            profile("fine", 0.0, 90.0, Some(10.0)),
        ]);
        let scored = score_all(
            &sel,
            &tun(),
            &names(&["ramp", "spent", "fine"]),
            &r,
            model,
            now(),
        );
        let reserve = "scored without the Fable reserve (it zeroed every candidate)";
        assert_eq!(
            scored[0].note().unwrap(),
            format!("{reserve}; in the 5-hour ramp")
        );
        assert_eq!(
            scored[1].note().unwrap(),
            format!("{reserve}; weekly score is zero")
        );
        assert_eq!(scored[2].note().unwrap(), reserve);
        // ramp and fine both score 0.1 after the re-score; fine has headroom.
        assert!((scored[0].score - scored[2].score).abs() < 1e-9);
        assert_eq!(scored[2].tier(), Some(1));
        assert_eq!(best_of(&scored).as_deref(), Some("fine"));

        // Unusable profiles carry a reason and no tier note.
        let r = report(vec![profile("gone", 99.5, 40.0, None)]);
        let gone = &score_all(&sel, &tun(), &names(&["gone"]), &r, model, now())[0];
        assert!(gone.notes.is_empty());
        assert_eq!(gone.tier(), None);
    }

    fn priority(weight: Option<f64>, reset: Option<WeeklyReset>) -> Priority {
        Priority {
            weight,
            weekly_reset: reset,
        }
    }

    #[test]
    fn a_weight_scales_the_score_within_its_tier_only() {
        let sel = SelectionConfig::default();
        let r = report(vec![
            profile("a", 0.0, 40.0, Some(80.0)),
            profile("b", 0.0, 50.0, Some(80.0)),
            profile("ramp", 92.0, 0.0, None),
        ]);
        let all = names(&["a", "b", "ramp"]);
        let plain = score_all(&sel, &tun(), &all, &r, OPUS, now());
        assert_eq!(best_of(&plain).as_deref(), Some("a"));
        let heavy = |name: &str, w: f64| {
            let layer = AccountLayer {
                priority: priority(Some(w), None),
                ..AccountLayer::default()
            };
            tuned(name, layer)
        };
        for model in [OPUS, FABLE] {
            let base = score_all(&sel, &tun(), &all, &r, model, now());
            let scored = score_all(&sel, &heavy("b", 2.0), &all, &r, model, now());
            assert!(
                (scored[1].score - 2.0 * base[1].score).abs() < 1e-12,
                "{model}"
            );
            assert_eq!(scored[1].notes, vec!["weight 2"], "{model}");
            assert!(scored[0].notes.is_empty());
        }
        let scored = score_all(&sel, &heavy("b", 2.0), &all, &r, OPUS, now());
        assert_eq!(best_of(&scored).as_deref(), Some("b"));
        // A tier-2 profile weighted far above the rest stays in tier 2.
        let scored = score_all(&sel, &heavy("ramp", 1000.0), &all, &r, OPUS, now());
        assert_eq!(scored[2].tier(), Some(2));
        assert_eq!(scored[2].notes, vec!["weight 1000", "in the 5-hour ramp"]);
        assert_eq!(best_of(&scored).as_deref(), Some("a"));
    }

    #[test]
    fn a_weekly_reset_override_shortens_hours_and_says_so() {
        let sel = SelectionConfig::default();
        let usage = profile("p", 0.0, 40.0, Some(80.0));
        let with = |reset: WeeklyReset, rule: bool| {
            let p = priority(None, Some(reset));
            let layer = if rule {
                AccountLayer {
                    by_model: Some(vec![crate::tuning::ModelRule {
                        models: names(&["*fable*"]),
                        priority: p,
                    }]),
                    ..AccountLayer::default()
                }
            } else {
                AccountLayer {
                    priority: p,
                    ..AccountLayer::default()
                }
            };
            tuned("p", layer)
        };
        let early = WeeklyReset::EarlierBy(SignedDuration::from_hours(60));
        let s = score_one(&sel, &with(early, false), OPUS, &usage, now(), true);
        assert!((s.general.unwrap().hours - 40.0).abs() < 1e-9);
        assert!((s.fable.unwrap().hours - 40.0).abs() < 1e-9);
        // 60% general less the 10% reserve, over 40 hours instead of 100.
        assert!((s.score - 50.0 / 40.0).abs() < 1e-9, "{s:?}");
        let to = at(40).unwrap();
        assert_eq!(s.general.unwrap().resets_at, to);
        assert_eq!(s.fable.unwrap().resets_at, to);
        assert_eq!(
            s.notes,
            vec![
                format!("general reset moved 60 h earlier, to {to}"),
                format!("Fable reset moved 60 h earlier, to {to}"),
            ]
        );

        // A zero offset, or an absolute time at the reset, moves nothing and
        // says nothing.
        for none in [
            WeeklyReset::EarlierBy(SignedDuration::ZERO),
            WeeklyReset::At(at(100).unwrap()),
        ] {
            let s = score_one(&sel, &with(none, false), OPUS, &usage, now(), true);
            assert!((s.general.unwrap().hours - 100.0).abs() < 1e-9);
            assert!((s.fable.unwrap().hours - 100.0).abs() < 1e-9);
            assert!(s.notes.is_empty(), "{s:?}");
        }

        // A moved reset already past sits at the floor.
        let past = WeeklyReset::At(at(-1).unwrap());
        let s = score_one(&sel, &with(past, false), OPUS, &usage, now(), true);
        assert_eq!(s.general.unwrap().hours, sel.min_time_left_h);
        assert_eq!(s.general.unwrap().resets_at, at(-1).unwrap());

        // A window published without a reset falls back to a week from now,
        // and that fallback is what an override is tested against.
        let mut bare = usage.clone();
        for w in &mut bare.windows {
            w.resets_at = None;
        }
        let s = score_one(&sel, &with(early, false), OPUS, &bare, now(), true);
        assert!((s.general.unwrap().hours - 108.0).abs() < 1e-9);
        assert_eq!(s.general.unwrap().resets_at, at(108).unwrap());
        let s = score_one(&sel, &with(past, false), OPUS, &bare, now(), true);
        assert!((s.general.unwrap().hours - WEEK_HOURS as f64).abs() < 1e-9);
        let week = at(WEEK_HOURS).unwrap();
        assert_eq!(
            s.notes,
            ["general", "Fable"].map(|b| format!(
                "weekly_reset_at {} not applied to the {b} bucket: its reset {week} is in a later week",
                at(-1).unwrap()
            ))
        );

        // An absolute time after the reset, or a week before it.
        let reset = at(100).unwrap();
        for (t, why) in [
            (at(101).unwrap(), format!("after its reset {reset}")),
            (
                at(-68).unwrap(),
                format!("its reset {reset} is in a later week"),
            ),
        ] {
            let s = score_one(
                &sel,
                &with(WeeklyReset::At(t), true),
                FABLE,
                &usage,
                now(),
                true,
            );
            assert!((s.general.unwrap().hours - 100.0).abs() < 1e-9);
            assert_eq!(
                s.notes,
                ["general", "Fable"].map(|b| format!(
                    "weekly_reset_at {t} not applied to the {b} bucket: {why} (by_model rule 1)"
                ))
            );
            // The rule is for Fable models only.
            let s = score_one(
                &sel,
                &with(WeeklyReset::At(t), true),
                OPUS,
                &usage,
                now(),
                true,
            );
            assert!(s.notes.is_empty(), "{s:?}");
        }
    }

    #[test]
    fn each_bucket_is_moved_against_its_own_reset_and_notes_keep_their_order() {
        let sel = SelectionConfig::default();
        // General resets in 100 h, Fable in 20 h; 92% of the 5-hour window
        // spent puts the profile in the ramp.
        let mut usage = profile("p", 92.0, 40.0, Some(80.0));
        usage.windows[2] = window("7-day_fable", 80.0, 20);
        let layer = |reset: WeeklyReset, by_model| AccountLayer {
            priority: priority(None, Some(reset)),
            by_model,
            ..AccountLayer::default()
        };

        // An absolute time inside the general week but after the Fable reset.
        let t = at(50).unwrap();
        let s = score_one(
            &sel,
            &tuned("p", layer(WeeklyReset::At(t), None)),
            OPUS,
            &usage,
            now(),
            true,
        );
        let (g, f) = (s.general.unwrap(), s.fable.unwrap());
        assert!((g.hours - 50.0).abs() < 1e-9);
        assert_eq!(g.resets_at, t);
        assert!((f.hours - 20.0).abs() < 1e-9);
        assert_eq!(f.resets_at, at(20).unwrap());
        assert_eq!(
            s.notes,
            vec![
                format!("general reset moved 50 h earlier, to {t}"),
                format!(
                    "weekly_reset_at {t} not applied to the Fable bucket: after its reset {}",
                    at(20).unwrap()
                ),
            ]
        );

        // An offset moves each bucket back from its own reset; a rule's
        // weight comes first, the reset notes next, the tier note last.
        let rules = vec![crate::tuning::ModelRule {
            models: names(&["*opus*"]),
            priority: priority(Some(2.0), None),
        }];
        let early = WeeklyReset::EarlierBy(SignedDuration::from_hours(10));
        let tuning = tuned("p", layer(early, Some(rules)));
        let s = &score_all(
            &sel,
            &tuning,
            &names(&["p"]),
            &report(vec![usage]),
            OPUS,
            now(),
        )[0];
        let (g, f) = (s.general.unwrap(), s.fable.unwrap());
        assert!((g.hours - 90.0).abs() < 1e-9);
        assert!((f.hours - 10.0).abs() < 1e-9);
        assert_eq!(s.tier(), Some(2));
        assert_eq!(
            s.notes,
            vec![
                "weight 2 (by_model rule 1)".to_string(),
                format!("general reset moved 10 h earlier, to {}", at(90).unwrap()),
                format!("Fable reset moved 10 h earlier, to {}", at(10).unwrap()),
                "in the 5-hour ramp".to_string(),
            ]
        );
    }

    #[test]
    fn keys_are_tried_by_weight_then_name() {
        let dir = crate::testutil::TempDir::new("select-key-weight");
        let mut cfg = keyed(&dir, &["a"]);
        add_key(&dir, &mut cfg, "alpha", &["*"]);
        add_key(&dir, &mut cfg, "zulu", &["*"]);
        let usage = reported(vec![spent("a")]);
        let zulu = |rules: Option<Vec<crate::tuning::ModelRule>>| {
            tuned(
                "zulu",
                AccountLayer {
                    priority: priority(Some(2.0), None),
                    by_model: rules,
                    ..AccountLayer::default()
                },
            )
        };
        let views = all_at(&cfg, 10.0);
        let (choice, _) = pick_tuned(&cfg, &zulu(None), OPUS, &usage, &[], views.clone(), true);
        let choice = choice.unwrap();
        took_key(&choice, "zulu");
        let rows = key_rows(&cfg, &zulu(None), OPUS, &choice, &views);
        assert_eq!(rows[0].why_not.as_deref(), Some("key zulu was taken first"));
        let weights: Vec<(&str, f64, Option<usize>)> = rows
            .iter()
            .map(|r| (r.name.as_str(), r.weight, r.weight_rule))
            .collect();
        assert_eq!(
            weights,
            vec![
                ("alpha", 1.0, None),
                ("team", 1.0, None),
                ("zulu", 2.0, None)
            ]
        );

        // Out of budget, the heavier key is passed over for name order.
        let mut low = views.clone();
        low.get_mut("zulu").unwrap().spent_usd = 410.0;
        let (choice, _) = pick_tuned(&cfg, &zulu(None), OPUS, &usage, &[], low, true);
        took_key(&choice.unwrap(), "alpha");

        // A rule takes zulu's weight back down for Opus only.
        let rule = crate::tuning::ModelRule {
            models: names(&["*opus*"]),
            priority: priority(Some(0.5), None),
        };
        let t = zulu(Some(vec![rule]));
        let (choice, _) = pick_tuned(&cfg, &t, OPUS, &usage, &[], views.clone(), true);
        let choice = choice.unwrap();
        took_key(&choice, "alpha");
        let rows = key_rows(&cfg, &t, OPUS, &choice, &views);
        assert_eq!((rows[2].weight, rows[2].weight_rule), (0.5, Some(0)));
        let (choice, _) = pick_tuned(&cfg, &t, FABLE, &usage, &[], views.clone(), true);
        let choice = choice.unwrap();
        took_key(&choice, "zulu");
        let rows = key_rows(&cfg, &t, FABLE, &choice, &views);
        assert_eq!((rows[2].weight, rows[2].weight_rule), (2.0, None));
    }

    #[test]
    fn a_weighted_profile_with_a_zero_score_still_lets_a_key_spend() {
        let dir = crate::testutil::TempDir::new("select-key-weighted-zero");
        let cfg = keyed(&dir, &["a"]);
        let t = tuned(
            "a",
            AccountLayer {
                priority: priority(Some(1000.0), None),
                ..AccountLayer::default()
            },
        );
        let usage = reported(vec![spent("a")]);
        let (choice, calls) = pick_tuned(&cfg, &t, OPUS, &usage, &[], all_at(&cfg, 10.0), true);
        let choice = choice.unwrap();
        took_key(&choice, "team");
        assert_eq!(calls, 1);
        let a = &choice.scored[0];
        assert_eq!((a.score, a.weight, a.tier()), (0.0, 1000.0, Some(2)));
    }

    #[test]
    fn unscored_profiles_carry_their_resolved_weight() {
        let sel = SelectionConfig::default();
        let t = tuned(
            "missing",
            AccountLayer {
                priority: priority(Some(3.0), None),
                by_model: Some(vec![crate::tuning::ModelRule {
                    models: names(&["*fable*"]),
                    priority: priority(Some(0.25), None),
                }]),
                ..AccountLayer::default()
            },
        );
        let r = report(vec![profile("gone", 99.5, 40.0, None)]);
        let all = names(&["missing", "gone"]);
        let scored = score_all(&sel, &t, &all, &r, OPUS, now());
        assert_eq!(scored[0].out, Some(Out::NoSnapshot));
        assert_eq!(scored[0].weight, 3.0);
        assert!(scored[0].notes.is_empty());
        assert_eq!(scored[1].out, Some(Out::FiveHourFloor));
        assert_eq!(scored[1].weight, 1.0);
        let scored = score_all(&sel, &t, &all, &r, FABLE, now());
        assert_eq!(scored[0].weight, 0.25);
    }

    /// A config whose token dir holds credentials for `profiles`.
    fn claude_config(dir: &std::path::Path, profiles: &[&str]) -> ClaudeConfig {
        ClaudeConfig {
            auth: ClaudeAuth::Token,
            profiles: names(profiles),
            token_dir: dir.display().to_string(),
            ..ClaudeConfig::default()
        }
    }

    fn empty_env() -> std::collections::BTreeMap<String, String> {
        std::collections::BTreeMap::new()
    }

    fn no_views() -> BTreeMap<String, BudgetView> {
        BTreeMap::new()
    }

    /// `choose` with no API key configured.
    fn choose_nk(
        claude: &ClaudeConfig,
        env: &dyn EnvSource,
        model: &str,
        now: Timestamp,
        usage: &UsageSource,
        exclude: &[String],
    ) -> Result<Choice> {
        choose(
            claude,
            &tun(),
            env,
            model,
            now,
            usage,
            exclude,
            Keys {
                eligible: true,
                budget: &no_views,
            },
        )
    }

    #[test]
    fn choose_scores_the_credentialed_profiles_and_warns_about_the_rest() {
        let dir = crate::testutil::TempDir::new("select-choose");
        dir.write_file("team03.token", "t\n", 0o600);
        dir.write_file("personal.token", "t\n", 0o600);
        let cfg = claude_config(dir.path(), &["team03", "team04", "personal"]);

        let choice = choose_nk(
            &cfg,
            &empty_env(),
            "claude-opus-5[1M]",
            now(),
            &UsageSource::Report(live()),
            &[],
        )
        .unwrap();

        assert_eq!(choice.account.profile, "personal");
        assert_eq!(choice.scored.len(), 2, "team04 is not scored");
        assert!(choice.degraded.is_none());
        assert_eq!(choice.warnings.len(), 1);
        assert!(
            choice.warnings[0].contains("team04"),
            "{:?}",
            choice.warnings
        );
        assert_eq!(choice.tier_note(), None);
    }

    #[test]
    fn a_tier_two_winner_says_so_without_the_reserve_note() {
        let dir = crate::testutil::TempDir::new("select-tier-note");
        dir.write_file("ramp.token", "t\n", 0o600);
        dir.write_file("spent.token", "t\n", 0o600);
        let cfg = claude_config(dir.path(), &["ramp", "spent"]);

        let r = report(vec![
            profile("ramp", 92.0, 40.0, None),
            profile("spent", 0.0, 100.0, None),
        ]);
        let pick = |cfg: &ClaudeConfig, r: &UsageReport| {
            choose_nk(
                cfg,
                &empty_env(),
                "claude-opus-5[1M]",
                now(),
                &UsageSource::Report(r.clone()),
                &[],
            )
            .unwrap()
        };
        let choice = pick(&cfg, &r);
        assert_eq!(choice.account.profile, "ramp");
        assert_eq!(
            choice.tier_note().as_deref(),
            Some("ramp chosen from tier 2: in the 5-hour ramp")
        );

        // The default reserve zeroes both; the re-score leaves ramp in tier 2.
        let r = report(vec![
            profile("ramp", 92.0, 80.0, Some(10.0)),
            profile("spent", 0.0, 100.0, Some(10.0)),
        ]);
        let choice = pick(&cfg, &r);
        assert_eq!(choice.account.profile, "ramp");
        assert_eq!(
            choice.scored[0].note().as_deref(),
            Some(
                "scored without the Fable reserve (it zeroed every candidate); \
                 in the 5-hour ramp"
            )
        );
        assert_eq!(
            choice.tier_note().as_deref(),
            Some("ramp chosen from tier 2: in the 5-hour ramp")
        );

        // The same re-score puts an account outside its ramp in tier 1.
        dir.write_file("idle.token", "t\n", 0o600);
        let idle_cfg = claude_config(dir.path(), &["idle", "spent"]);
        let r = report(vec![
            profile("idle", 0.0, 80.0, Some(10.0)),
            profile("spent", 0.0, 100.0, Some(10.0)),
        ]);
        let choice = pick(&idle_cfg, &r);
        assert_eq!(choice.account.profile, "idle");
        assert_eq!(
            choice.scored[0].note().as_deref(),
            Some("scored without the Fable reserve (it zeroed every candidate)")
        );
        assert_eq!(choice.tier_note(), None);

        // No usable candidate: preference order, no tier.
        let choice = pick(
            &cfg,
            &report(vec![
                profile("ramp", 100.0, 40.0, None),
                profile("spent", 100.0, 40.0, None),
            ]),
        );
        assert!(choice.degraded.is_some());
        assert_eq!(choice.tier_note(), None);
    }

    #[test]
    fn choose_falls_back_to_preference_order_whenever_the_data_cannot_decide() {
        let dir = crate::testutil::TempDir::new("select-degraded");
        dir.write_file("team03.token", "t\n", 0o600);
        dir.write_file("team04.token", "t\n", 0o600);
        let cfg = claude_config(dir.path(), &["team03", "team04"]);

        let unreachable = UsageSource::Unavailable("connection refused".into());
        let choice = choose_nk(&cfg, &empty_env(), "m", now(), &unreachable, &[]).unwrap();
        assert_eq!(choice.account.profile, "team03");
        assert!(choice.degraded.unwrap().contains("connection refused"));

        let offline = UsageSource::Report(UsageReport {
            daemon_status: Some("offline".into()),
            ..live()
        });
        let choice = choose_nk(&cfg, &empty_env(), "m", now(), &offline, &[]).unwrap();
        assert_eq!(choice.account.profile, "team03");
        assert!(choice.degraded.unwrap().contains("offline"));
        assert!(choice.scored.iter().all(|s| !s.usable));

        // Data that is present but unusable for every candidate.
        let all_bad = UsageSource::Report(report(vec![
            ProfileUsage {
                auth: AuthState::Rejected,
                ..profile("team03", 0.0, 0.0, None)
            },
            profile("team04", 99.9, 0.0, None),
        ]));
        let choice = choose_nk(&cfg, &empty_env(), "m", now(), &all_bad, &[]).unwrap();
        assert_eq!(choice.account.profile, "team03");
        let degraded = choice.degraded.unwrap();
        assert!(degraded.contains("auth is rejected"), "{degraded}");
        assert!(degraded.contains("5-hour"), "{degraded}");
    }

    #[test]
    fn choose_fails_when_nothing_has_a_credential() {
        let dir = crate::testutil::TempDir::new("select-nocreds");
        let cfg = claude_config(dir.path(), &["team03", "team04"]);
        let err = choose_nk(
            &cfg,
            &empty_env(),
            "m",
            now(),
            &UsageSource::Report(live()),
            &[],
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no configured Claude profile"), "{err}");
        assert!(err.contains("team03.token"), "{err}");

        // With every profile set aside there is no credential to look for.
        dir.write_file("team03.token", "t\n", 0o600);
        let only = claude_config(dir.path(), &["team03"]);
        let err = choose_nk(
            &only,
            &empty_env(),
            "m",
            now(),
            &UsageSource::Report(live()),
            &["team03".to_string()],
        )
        .unwrap_err()
        .to_string();
        assert_eq!(
            err,
            "no configured Claude profile is left once team03 is set aside"
        );
    }

    #[test]
    fn choose_skips_excluded_profiles_and_errors_only_when_none_remains() {
        let dir = crate::testutil::TempDir::new("select-exclude");
        dir.write_file("team03.token", "t\n", 0o600);
        dir.write_file("personal.token", "t\n", 0o600);
        let cfg = claude_config(dir.path(), &["team03", "team04", "personal"]);
        let pick = |exclude: &[String]| {
            choose_nk(
                &cfg,
                &empty_env(),
                "claude-opus-5[1M]",
                now(),
                &UsageSource::Report(live()),
                exclude,
            )
        };

        let choice = pick(&["personal".to_string()]).unwrap();
        assert_eq!(choice.account.profile, "team03");
        assert!(choice.scored.iter().all(|s| s.profile != "personal"));
        assert!(choice.warnings.iter().all(|w| !w.contains("personal")));

        // team04 has no credential, so excluding both others leaves nothing.
        assert!(pick(&["personal".to_string(), "team03".to_string()]).is_err());
    }

    const OPUS: &str = "claude-opus-5[1M]";
    const FABLE: &str = "claude-fable-5-1[1m]";

    /// A token-auth config over `dir` with `profiles` credentialed and one
    /// key, `team`, whose 0600 key file exists and whose models are `*`.
    fn keyed(dir: &crate::testutil::TempDir, profiles: &[&str]) -> ClaudeConfig {
        for p in profiles {
            dir.write_file(&format!("{p}.token"), "t\n", 0o600);
        }
        let mut cfg = claude_config(dir.path(), profiles);
        add_key(dir, &mut cfg, "team", &["*"]);
        cfg
    }

    fn add_key(
        dir: &crate::testutil::TempDir,
        cfg: &mut ClaudeConfig,
        name: &str,
        models: &[&str],
    ) {
        let path = dir.write_file(&format!("{name}.key"), "sk-ant-api\n", 0o600);
        let mut key = ApiKeyConfig::new(&path.display().to_string(), 420.0, models);
        key.reserve_usd = 20.0;
        cfg.api_keys.insert(name.to_string(), key);
    }

    fn budget_view(spent: f64) -> BudgetView {
        BudgetView {
            month: "2026-09".to_string(),
            spent_usd: spent,
            local_usd: spent,
            budget_usd: 420.0,
            reserve_usd: 20.0,
            turns: 3,
            unmeasured_turns: 0,
            exhausted: None,
            remote: true,
            ledger_error: None,
            ignored: Vec::new(),
        }
    }

    /// `choose` with the default tuning, keys `eligible`, the given budget
    /// views, and how often the budget closure ran.
    fn pick_with(
        cfg: &ClaudeConfig,
        model: &str,
        usage: &UsageSource,
        exclude: &[&str],
        views: BTreeMap<String, BudgetView>,
        eligible: bool,
    ) -> (Result<Choice>, usize) {
        pick_tuned(cfg, &tun(), model, usage, exclude, views, eligible)
    }

    fn pick_tuned(
        cfg: &ClaudeConfig,
        tuning: &Tuning,
        model: &str,
        usage: &UsageSource,
        exclude: &[&str],
        views: BTreeMap<String, BudgetView>,
        eligible: bool,
    ) -> (Result<Choice>, usize) {
        let calls = std::cell::Cell::new(0);
        let budget = || {
            calls.set(calls.get() + 1);
            views.clone()
        };
        let choice = choose(
            cfg,
            tuning,
            &empty_env(),
            model,
            now(),
            usage,
            &names(exclude),
            Keys {
                eligible,
                budget: &budget,
            },
        );
        (choice, calls.get())
    }

    fn all_at(cfg: &ClaudeConfig, spent: f64) -> BTreeMap<String, BudgetView> {
        cfg.api_keys
            .keys()
            .map(|k| (k.clone(), budget_view(spent)))
            .collect()
    }

    fn pick(
        cfg: &ClaudeConfig,
        model: &str,
        usage: UsageSource,
        exclude: &[&str],
    ) -> (Result<Choice>, usize) {
        pick_with(cfg, model, &usage, exclude, all_at(cfg, 10.0), true)
    }

    fn reported(profiles: Vec<ProfileUsage>) -> UsageSource {
        UsageSource::Report(report(profiles))
    }

    /// A profile whose general bucket is spent: usable, score zero.
    fn spent(name: &str) -> ProfileUsage {
        profile(name, 0.0, 100.0, Some(0.0))
    }

    fn fresh(name: &str) -> ProfileUsage {
        profile(name, 0.0, 10.0, Some(10.0))
    }

    fn took_key(choice: &Choice, name: &str) {
        assert_eq!(choice.account.profile, name, "{choice:?}");
        assert_eq!(choice.account.auth, ClaudeAuth::ApiKey);
        assert!(choice.degraded.is_none());
    }

    #[test]
    fn a_profile_with_a_positive_score_wins_and_the_budget_is_not_read() {
        let dir = crate::testutil::TempDir::new("select-key-profile-wins");
        let cfg = keyed(&dir, &["a", "b"]);
        let (choice, calls) = pick(&cfg, OPUS, reported(vec![spent("a"), fresh("b")]), &[]);
        let choice = choice.unwrap();
        assert_eq!(choice.account.profile, "b");
        assert_eq!(choice.keys, KeyDecision::ProfileWon);
        assert!(choice.key_notes.is_empty());
        assert_eq!(calls, 0);
    }

    #[test]
    fn every_profile_out_of_quota_overflows_to_the_key_with_a_note() {
        let dir = crate::testutil::TempDir::new("select-key-overflow");
        let cfg = keyed(&dir, &["a", "b", "nocred"]);
        std::fs::remove_file(dir.path().join("nocred.token")).unwrap();
        let usage = || reported(vec![spent("a"), spent("b"), fresh("nocred")]);

        // Every usable profile zero-scored; a credential-less one is no
        // statement either way.
        let (choice, calls) = pick(&cfg, OPUS, usage(), &[]);
        let choice = choice.unwrap();
        took_key(&choice, "team");
        assert_eq!(calls, 1);
        assert_eq!(
            choice.key_notes,
            vec![format!(
                "no profile can take {OPUS} (a: weekly score is zero; b: weekly score is zero); \
                 using API key team: $10.00 of $420.00 spent this month"
            )]
        );

        // One excluded, the other zero-scored.
        let (choice, _) = pick(&cfg, OPUS, usage(), &["a"]);
        took_key(&choice.unwrap(), "team");

        // The only profile excluded: no usage report is needed.
        let one = keyed(&dir, &["a"]);
        for usage in [
            UsageSource::Unavailable("connection refused".into()),
            reported(vec![fresh("a")]),
        ] {
            let (choice, calls) = pick(&one, OPUS, usage, &["a"]);
            let choice = choice.unwrap();
            took_key(&choice, "team");
            assert_eq!(calls, 1);
            assert!(choice.key_notes[0].contains("(a: set aside)"), "{choice:?}");
        }

        // A spend read from this host alone says so.
        let mut local = all_at(&cfg, 10.0);
        local.get_mut("team").unwrap().remote = false;
        let (choice, _) = pick_with(&cfg, OPUS, &usage(), &[], local, true);
        assert!(choice.unwrap().key_notes[0].ends_with("(spend: this host only)"));

        // A model that does not draw on a missing Fable window ignores it.
        let usage = reported(vec![spent("a"), profile("b", 0.0, 100.0, None)]);
        let (choice, _) = pick(&cfg, OPUS, usage, &[]);
        took_key(&choice.unwrap(), "team");
    }

    #[test]
    fn any_profile_that_cannot_take_the_work_lets_the_key_take_it() {
        let dir = crate::testutil::TempDir::new("select-key-cannot");
        let cfg = keyed(&dir, &["a", "b"]);
        let b = fresh("b");
        let cases: Vec<(Option<ProfileUsage>, &str, Out)> = vec![
            (
                Some(ProfileUsage {
                    stale_reason: Some("rate limited".into()),
                    ..b.clone()
                }),
                FABLE,
                Out::CarriedForward,
            ),
            (
                Some(ProfileUsage {
                    updated_at: now().checked_sub(SignedDuration::from_hours(2)).unwrap(),
                    ..b.clone()
                }),
                FABLE,
                Out::Stale,
            ),
            (
                Some(ProfileUsage {
                    auth: AuthState::Rejected,
                    ..b.clone()
                }),
                FABLE,
                Out::Auth,
            ),
            (None, FABLE, Out::NoSnapshot),
            (
                Some(ProfileUsage {
                    windows: vec![window(FIVE_HOUR, 0.0, 3), window("7-day_fable", 10.0, 100)],
                    ..b.clone()
                }),
                OPUS,
                Out::NoGeneralWindow,
            ),
            (
                Some(profile("b", 0.0, 10.0, None)),
                FABLE,
                Out::NoFableWindow,
            ),
            (
                Some(profile("b", 99.5, 10.0, Some(10.0))),
                FABLE,
                Out::FiveHourFloor,
            ),
        ];
        for (b, model, out) in cases {
            let mut profiles = vec![spent("a")];
            profiles.extend(b.clone());
            let (choice, calls) = pick(&cfg, model, reported(profiles), &[]);
            let choice = choice.unwrap();
            took_key(&choice, "team");
            assert_eq!(calls, 1, "{out:?}");
            let scored_b = choice.scored.iter().find(|s| s.profile == "b").unwrap();
            assert_eq!(scored_b.out, Some(out));
            assert!(
                choice.key_notes[0]
                    .contains(&format!("b: {}", scored_b.reason.as_deref().unwrap())),
                "{choice:?}"
            );
        }

        // Every profile unusable, none of them zero-scored: still the key.
        for each in [
            |n: &str| ProfileUsage {
                stale_reason: Some("x".into()),
                ..fresh(n)
            },
            |n: &str| profile(n, 99.5, 10.0, Some(10.0)),
        ] {
            let (choice, calls) = pick(&cfg, OPUS, reported(vec![each("a"), each("b")]), &[]);
            took_key(&choice.unwrap(), "team");
            assert_eq!(calls, 1);
        }
    }

    #[test]
    fn usage_data_unavailable_keeps_preference_order_and_no_key() {
        let dir = crate::testutil::TempDir::new("select-key-nodata");
        let cfg = keyed(&dir, &["a", "b"]);
        let offline = UsageSource::Report(UsageReport {
            daemon_status: Some("offline".into()),
            ..report(vec![spent("a"), spent("b")])
        });
        for usage in [UsageSource::Unavailable("refused".into()), offline] {
            let (choice, calls) = pick(&cfg, OPUS, usage, &[]);
            let choice = choice.unwrap();
            assert_eq!(calls, 0);
            assert!(choice.scored.iter().all(|s| s.out == Some(Out::NoData)));
            assert_eq!(choice.account.profile, "a");
            assert_eq!(choice.keys, KeyDecision::UsageUnknown);
            assert!(choice.degraded.is_some());
        }
    }

    #[test]
    fn out_is_set_exactly_where_reason_is() {
        let r = report(vec![
            spent("a"),
            fresh("b"),
            profile("c", 99.5, 10.0, Some(10.0)),
            profile("d", 0.0, 10.0, None),
        ]);
        let sel = SelectionConfig::default();
        for model in [OPUS, FABLE] {
            for s in score_all(
                &sel,
                &tun(),
                &names(&["a", "b", "c", "d", "e"]),
                &r,
                model,
                now(),
            ) {
                assert_eq!(s.reason.is_some(), s.out.is_some(), "{s:?}");
            }
        }
    }

    #[test]
    fn a_setup_with_no_credential_goes_to_the_key() {
        let dir = crate::testutil::TempDir::new("select-key-nocred");
        let mut cfg = claude_config(dir.path(), &["a", "b"]);
        add_key(&dir, &mut cfg, "team", &["*"]);
        let (choice, calls) = pick(&cfg, OPUS, reported(vec![spent("a"), spent("b")]), &[]);
        took_key(&choice.unwrap(), "team");
        assert_eq!(calls, 1);

        // One credential-less beside one zero-scored: the key.
        dir.write_file("a.token", "t\n", 0o600);
        let (choice, _) = pick(&cfg, OPUS, reported(vec![spent("a"), spent("b")]), &[]);
        took_key(&choice.unwrap(), "team");
    }

    #[test]
    fn a_key_is_passed_over_for_its_own_reasons_and_todays_result_stands() {
        let dir = crate::testutil::TempDir::new("select-key-gates");
        let base = keyed(&dir, &["a"]);
        let usage = || reported(vec![spent("a")]);
        let flagged = || {
            let mut v = budget_view(10.0);
            v.exhausted = Some(claude_usage_mqtt_core::Exhausted::new(
                now(),
                Some("inv".into()),
                Some("credit balance is too low".into()),
            ));
            BTreeMap::from([("team".to_string(), v)])
        };
        let unreadable = || {
            let mut v = budget_view(0.0);
            v.ledger_error = Some("budget ledger /l/team.json is damaged".into());
            BTreeMap::from([("team".to_string(), v)])
        };
        let mut fable_only = base.clone();
        fable_only.api_keys.get_mut("team").unwrap().models = names(&["*fable*"]);
        let mut no_file = base.clone();
        no_file.api_keys.get_mut("team").unwrap().key_file =
            dir.path().join("gone.key").display().to_string();
        let gone = dir.path().join("gone.key").display().to_string();
        let cases: Vec<(&ClaudeConfig, BTreeMap<String, BudgetView>, String, usize)> = vec![
            (
                &fable_only,
                all_at(&base, 0.0),
                "models do not match".into(),
                0,
            ),
            (
                &no_file,
                all_at(&base, 0.0),
                format!("no key file at {gone}"),
                0,
            ),
            (
                &base,
                all_at(&base, 400.0),
                "$400.00 of $420.00 spent, reserve $20.00".into(),
                1,
            ),
            (&base, flagged(), format!("exhausted since {}", now()), 1),
            (
                &base,
                unreadable(),
                "ledger unreadable: budget ledger /l/team.json is damaged".into(),
                1,
            ),
        ];
        for (cfg, views, why, expected_calls) in cases {
            let (choice, calls) = pick_with(cfg, OPUS, &usage(), &[], views.clone(), true);
            let choice = choice.unwrap();
            assert_eq!(calls, expected_calls, "{why}");
            assert_eq!(choice.account.profile, "a", "{why}");
            assert_eq!(
                choice.key_notes,
                vec![format!("API key team passed over: {why}")]
            );

            // With no profile left, the reasons join the refusal.
            let (choice, _) = pick_with(cfg, OPUS, &usage(), &["a"], views, true);
            let err = choice.unwrap_err().to_string();
            assert_eq!(
                err,
                format!(
                    "no configured Claude profile is left once a is set aside; \
                     API key team passed over: {why}"
                )
            );
        }

        // Not eligible (a launch), or the key itself set aside: no key.
        let (choice, calls) = pick_with(&base, OPUS, &usage(), &[], all_at(&base, 0.0), false);
        let choice = choice.unwrap();
        assert_eq!((choice.account.profile.as_str(), calls), ("a", 0));
        assert_eq!(choice.keys, KeyDecision::Ineligible);
        assert!(choice.key_notes.is_empty());
        let (choice, calls) = pick(&base, OPUS, usage(), &["team"]);
        assert_eq!((choice.unwrap().account.profile.as_str(), calls), ("a", 0));
    }

    #[test]
    fn keys_are_tried_in_name_order_on_one_budget_read() {
        let dir = crate::testutil::TempDir::new("select-key-order");
        let mut cfg = keyed(&dir, &["a"]);
        add_key(&dir, &mut cfg, "alpha", &["*"]);
        add_key(&dir, &mut cfg, "zulu", &["*"]);
        let usage = || reported(vec![spent("a")]);
        let (choice, calls) = pick(&cfg, OPUS, usage(), &[]);
        took_key(&choice.unwrap(), "alpha");
        assert_eq!(calls, 1);

        let mut views = all_at(&cfg, 10.0);
        views.get_mut("alpha").unwrap().spent_usd = 410.0;
        let (choice, calls) = pick_with(&cfg, OPUS, &usage(), &[], views.clone(), true);
        let choice = choice.unwrap();
        took_key(&choice, "team");
        assert_eq!(calls, 1);
        let rows = key_rows(&cfg, &tun(), OPUS, &choice, &views);
        let summary: Vec<(&str, bool, Option<&str>)> = rows
            .iter()
            .map(|r| (r.name.as_str(), r.used, r.why_not.as_deref()))
            .collect();
        assert_eq!(
            summary,
            vec![
                (
                    "alpha",
                    false,
                    Some("$410.00 of $420.00 spent, reserve $20.00")
                ),
                ("team", true, None),
                ("zulu", false, Some("key team was taken first")),
            ]
        );

        // The excluded key is skipped and the next one taken.
        let (choice, _) = pick(&cfg, OPUS, usage(), &["alpha"]);
        took_key(&choice.unwrap(), "team");
    }
}
