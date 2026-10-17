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

use anyhow::{bail, Result};
use claude_usage_mqtt_core::{AuthState, FIVE_HOUR};
use jiff::{SignedDuration, Timestamp};

use crate::audit_view::sanitize;
use crate::config::{ClaudeConfig, EnvSource, SelectionConfig};
use crate::glob::any_match;
use crate::harness::claude::Account;
use crate::harness::diag;
use crate::usage::{ProfileUsage, UsageReport, UsageSource};

/// A fresh weekly bucket's assumed length, used when a weekly window the
/// profile does have carries no reset time.
const WEEK_HOURS: i64 = 168;

/// One weekly bucket the account actually has.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bucket {
    /// Percent of the tightest matching window spent.
    pub util: f64,
    /// Hours until that window resets, floored by `min_time_left_h`.
    pub hours: f64,
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
    pub gate5: f64,
    pub score: f64,
    pub usable: bool,
    /// Why this profile is unusable; always `None` for a candidate.
    pub reason: Option<String>,
    /// Things worth knowing about a profile that is a candidate, in the order
    /// they were found: the scoring pass it came from, then why it is in tier
    /// 2. Always empty for an unusable profile.
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

/// The selected account, with the table behind the decision.
#[derive(Debug, Clone)]
pub struct Choice {
    pub account: Account,
    pub scored: Vec<Scored>,
    /// Set when the choice fell back to preference order instead of a score.
    pub degraded: Option<String>,
    /// Profiles dropped before scoring, and why.
    pub warnings: Vec<String>,
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
}

/// Pick the account for a Claude run: drop profiles with no local credential,
/// score the rest against the retained usage, and fall back to preference
/// order whenever the data cannot support a decision.
///
/// Profiles named in `exclude` are not considered at all.
///
/// The only error is having no credentialed profile at all — there is nothing
/// left to run as. Not called with empty `profiles`: that is the ambient
/// login, which has no account to choose.
pub fn choose(
    claude: &ClaudeConfig,
    env: &dyn EnvSource,
    model: &str,
    now: Timestamp,
    usage: &UsageSource,
    exclude: &[String],
) -> Result<Choice> {
    let mut candidates: Vec<Account> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut looked_for: Vec<String> = Vec::new();
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
    if candidates.is_empty() {
        if looked_for.is_empty() {
            bail!(
                "no configured Claude profile is left once {} is set aside",
                exclude.join(", ")
            );
        }
        bail!(
            "no configured Claude profile has a local credential; looked for {}",
            looked_for.join(", ")
        );
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

    let scored = match report {
        Some(r) => score_all(&claude.selection, &names, r, model, now),
        None => {
            let reason = degraded.clone().unwrap_or_default();
            names
                .iter()
                .map(|p| unscored(&claude.selection, p, reason.clone()))
                .collect::<Vec<_>>()
        }
    };

    let best = best_of(&scored);
    let account = match best {
        Some(profile) => candidates
            .iter()
            .find(|a| a.profile == profile)
            .cloned()
            .expect("the scored profiles are the candidate profiles"),
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
    })
}

/// Score every candidate, applying the Fable reserve's all-zero fallback, then
/// note why each usable one below tier 1 is there.
pub fn score_all(
    sel: &SelectionConfig,
    profiles: &[String],
    report: &UsageReport,
    model: &str,
    now: Timestamp,
) -> Vec<Scored> {
    let fable = any_match(&sel.fable_models, model);
    let mut scored = score_pass(
        sel,
        profiles,
        report,
        fable,
        now,
        sel.reserve_general_for_fable,
    );

    // A reserve that zeroes every candidate would starve non-Fable work for the
    // sake of Fable quota nobody is spending; re-score without it. With no
    // usable candidate at all there is nothing the reserve did, so no re-score.
    let mut usable = scored.iter().filter(|s| s.usable).peekable();
    let all_zero = usable.peek().is_some() && usable.all(|s| s.score <= 0.0);
    if !fable && sel.reserve_general_for_fable && all_zero {
        scored = score_pass(sel, profiles, report, fable, now, false);
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
    profiles: &[String],
    report: &UsageReport,
    fable: bool,
    now: Timestamp,
    reserve: bool,
) -> Vec<Scored> {
    profiles
        .iter()
        .map(|profile| match report.profile(profile) {
            None => unscored(sel, profile, "no usage snapshot published".to_string()),
            Some(usage) => score_one(sel, usage, fable, now, reserve),
        })
        .collect()
}

/// A candidate with no numbers behind it: unusable, and the table says why.
fn unscored(sel: &SelectionConfig, profile: &str, reason: String) -> Scored {
    let (floor, knee) = sel.effective(profile);
    Scored {
        profile: profile.to_string(),
        util5: 0.0,
        general: None,
        fable: None,
        five_hour_floor_pct: floor,
        five_hour_knee_pct: knee,
        gate5: 0.0,
        score: 0.0,
        usable: false,
        reason: Some(reason),
        notes: Vec::new(),
    }
}

fn score_one(
    sel: &SelectionConfig,
    usage: &ProfileUsage,
    fable: bool,
    now: Timestamp,
    reserve: bool,
) -> Scored {
    let f = features(sel, usage, now);
    let mut out = Scored {
        profile: usage.profile.clone(),
        util5: f.util5,
        general: f.general,
        fable: f.fable,
        five_hour_floor_pct: f.floor,
        five_hour_knee_pct: f.knee,
        gate5: f.gate5,
        score: 0.0,
        usable: false,
        reason: None,
        notes: Vec::new(),
    };

    let spend = match eligible(sel, usage, now, &f, fable) {
        Ok(spend) => spend,
        Err(reason) => {
            out.reason = Some(reason);
            return out;
        }
    };

    let ratio = sel.fable_general_ratio;
    out.score = match spend {
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
    out.usable = true;
    out
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
) -> Result<Spend, String> {
    if usage.auth != AuthState::Ok {
        return Err(format!("auth is {}", usage.auth.as_str()));
    }
    if let Some(why) = &usage.stale_reason {
        // The daemon's text, which can quote an upstream error body, reaches
        // the terminal through `ccc which` and the fallback diagnostic.
        return Err(format!(
            "the reading is carried forward ({})",
            sanitize(why)
        ));
    }
    let age = now.duration_since(usage.updated_at).as_secs_f64();
    if age > sel.stale_grace_s as f64 {
        return Err(format!(
            "the reading is {:.0} min old (grace {} s)",
            age / 60.0,
            sel.stale_grace_s
        ));
    }
    if f.gate5 <= 0.0 {
        return Err(format!(
            "the 5-hour window is at or below its floor ({:.0} % left, floor {} %)",
            100.0 - f.util5,
            f.floor
        ));
    }
    // Every model spends the general bucket.
    let Some(general) = f.general else {
        return Err("no general weekly window in the snapshot".to_string());
    };
    if !fable {
        return Ok(Spend::General {
            general,
            reserve_against: f.fable,
        });
    }
    let Some(fable) = f.fable else {
        return Err("no Fable weekly window in the snapshot".to_string());
    };
    Ok(Spend::Fable { general, fable })
}

fn features(sel: &SelectionConfig, usage: &ProfileUsage, now: Timestamp) -> Features {
    // An absent 5-hour window is a window nothing has been spent in yet.
    let util5 = usage
        .window(FIVE_HOUR)
        .map(|w| w.utilization)
        .unwrap_or(0.0);

    let fresh_week = now
        .checked_add(SignedDuration::from_hours(WEEK_HOURS))
        .unwrap_or(now);

    let window_g = tightest(usage, &sel.general_windows);
    // A weekly window the daemon published without a reset time still exists;
    // the fallback clock only stands in for that gap.
    let reset_g = window_g.and_then(|w| w.resets_at).unwrap_or(fresh_week);
    let general = window_g.map(|w| Bucket {
        util: w.utilization,
        hours: hours_until(now, reset_g, sel.min_time_left_h),
    });
    let fable = tightest(usage, &sel.fable_windows).map(|w| {
        let reset = w.resets_at.unwrap_or(reset_g);
        Bucket {
            util: w.utilization,
            hours: hours_until(now, reset, sel.min_time_left_h),
        }
    });

    let (floor, knee) = sel.effective(&usage.profile);
    Features {
        util5,
        floor,
        knee,
        gate5: gate5(floor, knee, util5),
        general,
        fable,
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
    use crate::usage::UsageWindow;

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
        let (floor, knee) = SelectionConfig::default().effective("any");
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
        let sel = SelectionConfig {
            fable_general_ratio: 0.0,
            ..SelectionConfig::default()
        };
        let usage = profile("team03", 0.0, 99.0, Some(50.0));
        let scored = score_one(&sel, &usage, true, now(), true);
        assert!(scored.usable);
        assert!(scored.score.is_finite(), "{}", scored.score);
        let expected = (100.0 - 50.0) / 100.0;
        assert!((scored.score - expected).abs() < 1e-9, "{}", scored.score);

        let plain = score_one(&sel, &usage, false, now(), true);
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
        let sel = SelectionConfig {
            min_time_left_h: 0.0,
            ..SelectionConfig::default()
        };
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
        let scored = score_one(&sel, &usage, false, now(), true);
        assert!(scored.score.is_finite(), "{}", scored.score);
        let fable = score_one(&sel, &usage, true, now(), true);
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
        let f = features(&sel, &usage, now());
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
        let f = features(&sel, &usage, now());
        assert_eq!(f.fable, None);
        let g = f.general.unwrap();
        assert_eq!(g.util, 7.0);
        assert!((g.hours - 24.0).abs() < 1e-9);

        let bare = ProfileUsage {
            windows: vec![window(FIVE_HOUR, 0.0, 3)],
            ..usage
        };
        let f = features(&sel, &bare, now());
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
        let f = features(&sel, &usage, now());
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
        let f = features(&sel, &usage, now());
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
        let f = features(&sel, &usage, now());
        assert_eq!(f.util5, 0.0);
        assert_eq!(f.gate5, 1.0);
        assert!(score_one(&sel, &usage, false, now(), true).usable);
    }

    #[test]
    fn a_fable_model_cannot_use_an_account_without_a_fable_window() {
        let sel = SelectionConfig::default();
        let usage = profile("personal", 0.0, 7.0, None);
        let s = score_one(&sel, &usage, true, now(), true);
        assert!(!s.usable);
        assert!(s.reason.as_deref().unwrap().contains("Fable"), "{s:?}");
        assert_eq!(s.score, 0.0);

        assert!(score_one(&sel, &usage, false, now(), true).usable);
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
            let s = score_one(&sel, &usage, fable, now(), true);
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
        let s = score_one(&sel, &profile("a", 0.0, 40.0, None), false, now(), true);
        assert!((s.score - 0.6).abs() < 1e-9, "{s:?}");
    }

    #[test]
    fn scoring_knows_which_buckets_the_model_spends() {
        let sel = SelectionConfig::default();
        let usable =
            |model: &str| score_all(&sel, &names(&["personal"]), &live(), model, now()).remove(0);
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
        let f = features(&sel, &usage, now());
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
            let s = score_one(&sel, &usage, false, now(), true);
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
        let reason = score_one(&sel, &usage, false, now(), true).reason.unwrap();
        assert_eq!(
            reason,
            "the reading is carried forward (rate\\nlimited\\r\\x1b[2J)"
        );

        // The Fable bucket only excludes a profile from Fable work.
        let s = score_one(&sel, &profile("a", 0.0, 0.0, None), true, now(), true);
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
            &profile("a", 0.0, 40.0, Some(80.0)),
            true,
            now(),
            true,
        );
        assert!((fable_binds.score - 0.2).abs() < 1e-9);

        // Fable 90% left (0.9%/h) but general only 10% left: 0.1%/h general is
        // 0.2%/h of Fable, which binds instead.
        let general_binds = score_one(
            &sel,
            &profile("a", 0.0, 90.0, Some(10.0)),
            true,
            now(),
            true,
        );
        assert!((general_binds.score - 0.2).abs() < 1e-9);

        // The gate damps the whole score.
        let gated = score_one(
            &sel,
            &profile("a", 92.0, 40.0, Some(80.0)),
            true,
            now(),
            true,
        );
        assert!((gated.score - 0.1).abs() < 1e-9);
    }

    #[test]
    fn a_fable_model_is_recognised_by_the_model_globs() {
        let sel = SelectionConfig::default();
        let r = report(vec![profile("a", 0.0, 40.0, Some(80.0))]);
        let fable = &score_all(&sel, &names(&["a"]), &r, "claude-fable-5-1[1m]", now())[0];
        let general = &score_all(&sel, &names(&["a"]), &r, "claude-opus-5[1M]", now())[0];
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
        let scored = score_all(&sel, &names(&["a", "b"]), &r, "claude-opus-5[1M]", now());
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
        let sel = SelectionConfig {
            reserve_general_for_fable: false,
            ..sel
        };
        let scored = score_all(&sel, &names(&["a", "b"]), &r, "claude-opus-5[1M]", now());
        assert!((scored[0].score - 0.2).abs() < 1e-9);
        assert_eq!(scored[0].note(), None);
    }

    #[test]
    fn the_live_snapshots_pick_the_least_loaded_account() {
        let sel = SelectionConfig::default();
        let scored = score_all(
            &sel,
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
        let scored = score_all(&sel, &names(&["a", "b"]), &r, "claude-opus-5[1M]", now());
        assert_eq!(scored[0].score, scored[1].score);
        assert_eq!(best_of(&scored).as_deref(), Some("a"));

        let none_usable = score_all(&sel, &names(&["missing"]), &r, "claude-opus-5[1M]", now());
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
            gate5,
            score,
            usable: true,
            reason: None,
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

    fn with_override(floor: Option<f64>, knee: Option<f64>) -> SelectionConfig {
        let mut sel = SelectionConfig::default();
        sel.profile.insert(
            "personal".into(),
            crate::config::ProfileSelection {
                five_hour_floor_pct: floor,
                five_hour_knee_pct: knee,
            },
        );
        sel
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
        let scored = score_all(&sel, &all, &r, "claude-opus-5[1M]", now());
        assert_eq!(best_of(&scored).as_deref(), Some("personal"));

        // Its own knee at 60% remaining puts it in its ramp, and a team seat
        // with headroom takes the work despite a far lower score.
        let sel = with_override(None, Some(60.0));
        let scored = score_all(&sel, &all, &r, "claude-opus-5[1M]", now());
        let personal = scored.iter().find(|s| s.profile == "personal").unwrap();
        let team03 = scored.iter().find(|s| s.profile == "team03").unwrap();
        assert_eq!(personal.tier(), Some(2), "{personal:?}");
        assert!(personal.score > team03.score, "{personal:?} vs {team03:?}");
        assert_eq!(best_of(&scored).as_deref(), Some("team03"));
    }

    #[test]
    fn per_profile_floor_and_knee_apply_to_that_profile_only() {
        let sel = with_override(Some(20.0), Some(60.0));
        let personal = score_one(
            &sel,
            &profile("personal", 60.0, 7.0, None),
            false,
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
            &sel,
            &profile("team03", 60.0, 7.0, None),
            false,
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
            &sel,
            &profile("personal", 85.0, 7.0, None),
            false,
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
            &sel,
            &profile("team03", 85.0, 7.0, None),
            false,
            now(),
            true,
        );
        assert!(team.usable, "{team:?}");

        // Overriding the floor alone keeps the global knee.
        let sel = with_override(Some(10.0), None);
        let personal = score_one(
            &sel,
            &profile("personal", 87.5, 7.0, None),
            false,
            now(),
            true,
        );
        assert!((personal.gate5 - 0.5).abs() < 1e-9, "{personal:?}");

        // A profile with no snapshot still reports its resolved knobs.
        let r = report(vec![]);
        let missing = &score_all(&sel, &names(&["personal"]), &r, "m", now())[0];
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
        let scored = score_all(&sel, &names(&["ramp", "spent", "fine"]), &r, model, now());
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
        let scored = score_all(&sel, &names(&["ramp", "spent", "fine"]), &r, model, now());
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
        let gone = &score_all(&sel, &names(&["gone"]), &r, model, now())[0];
        assert!(gone.notes.is_empty());
        assert_eq!(gone.tier(), None);
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

    #[test]
    fn choose_scores_the_credentialed_profiles_and_warns_about_the_rest() {
        let dir = crate::testutil::TempDir::new("select-choose");
        dir.write_file("team03.token", "t\n", 0o600);
        dir.write_file("personal.token", "t\n", 0o600);
        let cfg = claude_config(dir.path(), &["team03", "team04", "personal"]);

        let choice = choose(
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
            choose(
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
        let choice = choose(&cfg, &empty_env(), "m", now(), &unreachable, &[]).unwrap();
        assert_eq!(choice.account.profile, "team03");
        assert!(choice.degraded.unwrap().contains("connection refused"));

        let offline = UsageSource::Report(UsageReport {
            daemon_status: Some("offline".into()),
            ..live()
        });
        let choice = choose(&cfg, &empty_env(), "m", now(), &offline, &[]).unwrap();
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
        let choice = choose(&cfg, &empty_env(), "m", now(), &all_bad, &[]).unwrap();
        assert_eq!(choice.account.profile, "team03");
        let degraded = choice.degraded.unwrap();
        assert!(degraded.contains("auth is rejected"), "{degraded}");
        assert!(degraded.contains("5-hour"), "{degraded}");
    }

    #[test]
    fn choose_fails_when_nothing_has_a_credential() {
        let dir = crate::testutil::TempDir::new("select-nocreds");
        let cfg = claude_config(dir.path(), &["team03", "team04"]);
        let err = choose(
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
        let err = choose(
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
            choose(
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
}
