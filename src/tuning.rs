//! Account-tuning knobs as data, independent of where they were read: the
//! 5-hour floor and knee, and each account's priority weight and weekly-reset
//! override with their per-model rules.
//!
//! A source of knobs produces a [`TuningLayer`], every field optional, whose
//! values it has checked with this module's value checks. Layers are applied
//! with [`Tuning::overlay`] onto a [`Tuning`], later layers winning field by
//! field, and the result is checked with [`Tuning::check_ramps`], because a
//! floor from one layer and a knee from another can cross. Selection reads only
//! the resulting `Tuning`. Config loading runs that sequence with the site
//! config's layer onto [`Tuning::default`].
//!
//! Profiles and API keys share one account namespace. A key's layer sets only
//! weights: a key has no 5-hour window and no weekly bucket.
// TODO(account-knobs-from-mqtt): the knobs come from the site config only; a
// broker-supplied layer would be checked, overlaid onto the config's `Tuning`
// and ramp-checked in the same sequence.

use std::collections::BTreeMap;

use jiff::{SignedDuration, Timestamp};

use crate::glob::any_match;

/// A weekly bucket's length, in hours.
pub const WEEK_HOURS: i64 = 168;

/// Below this much of the 5-hour window remaining, a profile is unusable.
pub const FIVE_HOUR_FLOOR_PCT: f64 = 1.0;
/// Above this much remaining, the 5-hour gate no longer damps the score.
pub const FIVE_HOUR_KNEE_PCT: f64 = 15.0;

/// A pretend weekly reset, earlier than the real one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeeklyReset {
    /// This time, when it falls in the bucket's current week.
    At(Timestamp),
    /// The real reset less this much; always applies.
    EarlierBy(SignedDuration),
}

/// Why a [`WeeklyReset::At`] was not applied to a bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NotApplied {
    /// The time is after the bucket's reset.
    AfterReset,
    /// The time is at or before the start of the bucket's week.
    EarlierWeek,
}

/// A [`WeeklyReset`] against one bucket's reset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResetApplied {
    Moved { from: Timestamp, to: Timestamp },
    NotApplied { reset: Timestamp, why: NotApplied },
}

impl ResetApplied {
    /// The reset the bucket's hours count to.
    pub fn effective(&self) -> Timestamp {
        match *self {
            ResetApplied::Moved { to, .. } => to,
            ResetApplied::NotApplied { reset, .. } => reset,
        }
    }
}

impl WeeklyReset {
    /// This override against a bucket that resets at `reset`.
    pub fn apply(&self, reset: Timestamp) -> ResetApplied {
        match *self {
            WeeklyReset::EarlierBy(d) => ResetApplied::Moved {
                from: reset,
                to: reset.checked_sub(d).unwrap_or(Timestamp::MIN),
            },
            WeeklyReset::At(t) => {
                let week_start = reset
                    .checked_sub(SignedDuration::from_hours(WEEK_HOURS))
                    .unwrap_or(Timestamp::MIN);
                if t > reset {
                    ResetApplied::NotApplied {
                        reset,
                        why: NotApplied::AfterReset,
                    }
                } else if t <= week_start {
                    ResetApplied::NotApplied {
                        reset,
                        why: NotApplied::EarlierWeek,
                    }
                } else {
                    ResetApplied::Moved { from: reset, to: t }
                }
            }
        }
    }
}

/// The priority knobs at one level. A level that sets `weekly_reset` in
/// either form replaces a lower level's override whole.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Priority {
    pub weight: Option<f64>,
    pub weekly_reset: Option<WeeklyReset>,
}

/// Priority knobs for the models any of `models` matches.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRule {
    pub models: Vec<String>,
    pub priority: Priority,
}

/// One account's knobs from one source.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AccountLayer {
    pub five_hour_floor_pct: Option<f64>,
    pub five_hour_knee_pct: Option<f64>,
    pub priority: Priority,
    /// First match wins. `None`: this source says nothing about rules;
    /// `Some(vec![])`: no rules.
    pub by_model: Option<Vec<ModelRule>>,
}

/// One source's knobs, every field optional.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TuningLayer {
    pub five_hour_floor_pct: Option<f64>,
    pub five_hour_knee_pct: Option<f64>,
    pub accounts: BTreeMap<String, AccountLayer>,
}

/// The resolved knob set selection reads.
#[derive(Debug, Clone, PartialEq)]
pub struct Tuning {
    pub five_hour_floor_pct: f64,
    pub five_hour_knee_pct: f64,
    pub accounts: BTreeMap<String, AccountLayer>,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            five_hour_floor_pct: FIVE_HOUR_FLOOR_PCT,
            five_hour_knee_pct: FIVE_HOUR_KNEE_PCT,
            accounts: BTreeMap::new(),
        }
    }
}

/// One account's priority knobs for one model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AccountPriority {
    pub weight: f64,
    pub weekly_reset: Option<WeeklyReset>,
    /// The 0-based `by_model` rule each value came from; `None` for the
    /// account level or the default.
    pub weight_rule: Option<usize>,
    pub reset_rule: Option<usize>,
}

/// `weight <w>`, with the rule it came from, when the weight is not 1.
pub fn weight_note(weight: f64, rule: Option<usize>) -> Option<String> {
    (weight != 1.0).then(|| format!("weight {weight}{}", rule_suffix(rule)))
}

/// ` (by_model rule <n>)`, 1-based like the config's error messages, for a
/// value that came from a rule; empty otherwise.
pub fn rule_suffix(rule: Option<usize>) -> String {
    rule.map_or(String::new(), |i| format!(" (by_model rule {})", i + 1))
}

/// A resolved 5-hour floor/knee pair the gate cannot use.
#[derive(Debug, Clone, PartialEq)]
pub struct RampFault {
    /// `None`: the global pair.
    pub account: Option<String>,
    pub floor: f64,
    pub knee: f64,
}

impl std::fmt::Display for RampFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&ramp_message(self.floor, self.knee))
    }
}

impl Tuning {
    /// `layer`'s values over this set: every value `layer` sets replaces the
    /// one at the same place, a `by_model` list whole; the rest is kept.
    pub fn overlay(&self, layer: &TuningLayer) -> Tuning {
        let mut out = self.clone();
        if let Some(v) = layer.five_hour_floor_pct {
            out.five_hour_floor_pct = v;
        }
        if let Some(v) = layer.five_hour_knee_pct {
            out.five_hour_knee_pct = v;
        }
        for (name, over) in &layer.accounts {
            let acc = out.accounts.entry(name.clone()).or_default();
            if over.five_hour_floor_pct.is_some() {
                acc.five_hour_floor_pct = over.five_hour_floor_pct;
            }
            if over.five_hour_knee_pct.is_some() {
                acc.five_hour_knee_pct = over.five_hour_knee_pct;
            }
            if over.priority.weight.is_some() {
                acc.priority.weight = over.priority.weight;
            }
            if over.priority.weekly_reset.is_some() {
                acc.priority.weekly_reset = over.priority.weekly_reset;
            }
            if over.by_model.is_some() {
                acc.by_model.clone_from(&over.by_model);
            }
        }
        out
    }

    /// `account`'s 5-hour `(floor, knee)` in remaining percent: its own value
    /// where it has one, else the global one, each independently.
    pub fn ramp(&self, account: &str) -> (f64, f64) {
        let acc = self.accounts.get(account);
        (
            acc.and_then(|a| a.five_hour_floor_pct)
                .unwrap_or(self.five_hour_floor_pct),
            acc.and_then(|a| a.five_hour_knee_pct)
                .unwrap_or(self.five_hour_knee_pct),
        )
    }

    /// `account`'s weight and weekly-reset override for `model`, each from
    /// the first `by_model` rule matching `model` if that rule sets it, else
    /// from the account level, else weight 1 and no override. Later rules are
    /// never consulted.
    pub fn priority(&self, account: &str, model: &str) -> AccountPriority {
        let mut out = AccountPriority {
            weight: 1.0,
            weekly_reset: None,
            weight_rule: None,
            reset_rule: None,
        };
        let Some(acc) = self.accounts.get(account) else {
            return out;
        };
        let rule = acc
            .by_model
            .iter()
            .flatten()
            .enumerate()
            .find(|(_, r)| any_match(&r.models, model));
        if let Some(w) = acc.priority.weight {
            out.weight = w;
        }
        out.weekly_reset = acc.priority.weekly_reset;
        if let Some((i, rule)) = rule {
            if let Some(w) = rule.priority.weight {
                out.weight = w;
                out.weight_rule = Some(i);
            }
            if let Some(r) = rule.priority.weekly_reset {
                out.weekly_reset = Some(r);
                out.reset_rule = Some(i);
            }
        }
        out
    }

    /// Checks the global ramp, then the resolved ramp of every account that
    /// sets a floor or a knee, in name order; the first that fails.
    pub fn check_ramps(&self) -> Result<(), RampFault> {
        let fault = |account: Option<&str>, (floor, knee): (f64, f64)| {
            check_ramp(floor, knee).map_err(|_| RampFault {
                account: account.map(str::to_string),
                floor,
                knee,
            })
        };
        fault(None, (self.five_hour_floor_pct, self.five_hour_knee_pct))?;
        for (name, acc) in &self.accounts {
            if acc.five_hour_floor_pct.is_some() || acc.five_hour_knee_pct.is_some() {
                fault(Some(name), self.ramp(name))?;
            }
        }
        Ok(())
    }
}

/// The smallest weight accepted. Within `[WEIGHT_MIN, WEIGHT_MAX]`,
/// `weight × score` is zero only when the score is, and stays finite, for any
/// score more than a factor of 1e6 inside `f64`'s limits; a weight merely
/// above 0 can underflow a positive score to zero and so change its tier.
pub const WEIGHT_MIN: f64 = 1e-6;
/// The largest weight accepted; see [`WEIGHT_MIN`].
pub const WEIGHT_MAX: f64 = 1e6;

/// Refuses a weight outside `[WEIGHT_MIN, WEIGHT_MAX]`, `NaN` included.
pub fn check_weight(v: f64) -> Result<(), String> {
    if !(WEIGHT_MIN..=WEIGHT_MAX).contains(&v) {
        return Err(format!(
            "weight must be from {WEIGHT_MIN} to {WEIGHT_MAX} (got {v})"
        ));
    }
    Ok(())
}

/// Refuses a weekly-reset offset, in hours, outside `[0, WEEK_HOURS)`.
pub fn check_earlier_h(v: f64) -> Result<(), String> {
    if !v.is_finite() || v < 0.0 || v >= WEEK_HOURS as f64 {
        return Err(format!(
            "weekly_reset_earlier_h must be a finite number, at least 0 and below {WEEK_HOURS} (got {v})"
        ));
    }
    Ok(())
}

/// Refuses a 5-hour floor/knee pair the gate cannot use: each must be a number
/// in 0-100 (which also rules out NaN and infinities) and the floor must not
/// exceed the knee.
pub fn check_ramp(floor: f64, knee: f64) -> Result<(), String> {
    if !(0.0..=100.0).contains(&floor) || !(0.0..=100.0).contains(&knee) || floor > knee {
        return Err(ramp_message(floor, knee));
    }
    Ok(())
}

fn ramp_message(floor: f64, knee: f64) -> String {
    format!(
        "needs 0 <= five_hour_floor_pct <= five_hour_knee_pct <= 100 (got floor {floor}, knee {knee})"
    )
}

/// Refuses a rule's empty `models` list.
pub fn check_rule_models(models: &[String]) -> Result<(), String> {
    if models.is_empty() {
        return Err("models must name at least one model glob".to_string());
    }
    Ok(())
}

/// Refuses a rule that sets nothing: it would still stop every later rule
/// from matching.
pub fn check_rule_sets(priority: &Priority) -> Result<(), String> {
    if priority.weight.is_none() && priority.weekly_reset.is_none() {
        return Err(
            "sets neither weight nor a weekly reset; a rule that sets nothing still stops later rules from matching"
                .to_string(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(s: &str) -> Timestamp {
        s.parse().unwrap()
    }

    fn hours(h: i64) -> SignedDuration {
        SignedDuration::from_hours(h)
    }

    fn weight(w: f64) -> Priority {
        Priority {
            weight: Some(w),
            weekly_reset: None,
        }
    }

    fn rule(models: &[&str], priority: Priority) -> ModelRule {
        ModelRule {
            models: models.iter().map(|m| m.to_string()).collect(),
            priority,
        }
    }

    fn with_account(name: &str, acc: AccountLayer) -> Tuning {
        Tuning::default().overlay(&TuningLayer {
            accounts: BTreeMap::from([(name.to_string(), acc)]),
            ..TuningLayer::default()
        })
    }

    #[test]
    fn ramp_falls_back_to_the_global_pair_field_by_field() {
        let t = Tuning::default();
        assert_eq!(t.ramp("any"), (1.0, 15.0));
        let t = with_account(
            "p",
            AccountLayer {
                five_hour_knee_pct: Some(60.0),
                ..AccountLayer::default()
            },
        );
        assert_eq!(t.ramp("p"), (1.0, 60.0));
        assert_eq!(t.ramp("q"), (1.0, 15.0));
        let t = with_account(
            "p",
            AccountLayer {
                five_hour_floor_pct: Some(10.0),
                ..AccountLayer::default()
            },
        );
        assert_eq!(t.ramp("p"), (10.0, 15.0));
    }

    #[test]
    fn priority_takes_the_first_matching_rule_then_the_account_level() {
        let early = WeeklyReset::EarlierBy(hours(36));
        let at = WeeklyReset::At(ts("2026-11-14T08:00:00Z"));
        let t = with_account(
            "p",
            AccountLayer {
                priority: Priority {
                    weight: Some(1.5),
                    weekly_reset: Some(early),
                },
                by_model: Some(vec![
                    rule(&["*opus*"], weight(0.5)),
                    rule(
                        &["*FABLE*"],
                        Priority {
                            weight: None,
                            weekly_reset: Some(at),
                        },
                    ),
                    rule(&["*fable*"], weight(9.0)),
                ]),
                ..AccountLayer::default()
            },
        );

        // No rule matches: the account level.
        assert_eq!(
            t.priority("p", "claude-sonnet-5"),
            AccountPriority {
                weight: 1.5,
                weekly_reset: Some(early),
                weight_rule: None,
                reset_rule: None,
            }
        );
        // The first rule sets only a weight; the reset falls to the account.
        assert_eq!(
            t.priority("p", "claude-opus-5[1M]"),
            AccountPriority {
                weight: 0.5,
                weekly_reset: Some(early),
                weight_rule: Some(0),
                reset_rule: None,
            }
        );
        // Case-insensitive glob; the unset weight falls to the account level,
        // not to the later matching rule.
        assert_eq!(
            t.priority("p", "claude-fable-5-1[1m]"),
            AccountPriority {
                weight: 1.5,
                weekly_reset: Some(at),
                weight_rule: None,
                reset_rule: Some(1),
            }
        );
        // An unknown account: the defaults.
        assert_eq!(
            t.priority("q", "claude-opus-5[1M]"),
            AccountPriority {
                weight: 1.0,
                weekly_reset: None,
                weight_rule: None,
                reset_rule: None,
            }
        );
    }

    #[test]
    fn check_ramps_reports_the_first_resolved_pair_that_fails() {
        assert_eq!(Tuning::default().check_ramps(), Ok(()));

        let bad_global = Tuning::default().overlay(&TuningLayer {
            five_hour_floor_pct: Some(20.0),
            ..TuningLayer::default()
        });
        let fault = bad_global.check_ramps().unwrap_err();
        assert_eq!(
            fault,
            RampFault {
                account: None,
                floor: 20.0,
                knee: 15.0
            }
        );
        assert_eq!(
            fault.to_string(),
            "needs 0 <= five_hour_floor_pct <= five_hour_knee_pct <= 100 (got floor 20, knee 15)"
        );

        let own = with_account(
            "a",
            AccountLayer {
                five_hour_floor_pct: Some(30.0),
                five_hour_knee_pct: Some(20.0),
                ..AccountLayer::default()
            },
        );
        assert_eq!(own.check_ramps().unwrap_err().account.as_deref(), Some("a"));

        // A knee alone below the global floor.
        let knee_only = Tuning {
            five_hour_floor_pct: 10.0,
            ..Tuning::default()
        }
        .overlay(&TuningLayer {
            accounts: BTreeMap::from([(
                "b".to_string(),
                AccountLayer {
                    five_hour_knee_pct: Some(5.0),
                    ..AccountLayer::default()
                },
            )]),
            ..TuningLayer::default()
        });
        assert_eq!(
            knee_only.check_ramps().unwrap_err(),
            RampFault {
                account: Some("b".to_string()),
                floor: 10.0,
                knee: 5.0
            }
        );

        // An account setting neither is not checked separately.
        let weight_only = with_account(
            "c",
            AccountLayer {
                priority: weight(2.0),
                ..AccountLayer::default()
            },
        );
        assert_eq!(weight_only.check_ramps(), Ok(()));

        // A floor from one layer and a knee from another that cross.
        let first = with_account(
            "d",
            AccountLayer {
                five_hour_floor_pct: Some(40.0),
                five_hour_knee_pct: Some(60.0),
                ..AccountLayer::default()
            },
        );
        assert_eq!(first.check_ramps(), Ok(()));
        let crossed = first.overlay(&TuningLayer {
            accounts: BTreeMap::from([(
                "d".to_string(),
                AccountLayer {
                    five_hour_knee_pct: Some(30.0),
                    ..AccountLayer::default()
                },
            )]),
            ..TuningLayer::default()
        });
        assert_eq!(
            crossed.check_ramps().unwrap_err(),
            RampFault {
                account: Some("d".to_string()),
                floor: 40.0,
                knee: 30.0
            }
        );
    }

    #[test]
    fn overlay_replaces_what_the_layer_sets_and_keeps_the_rest() {
        let base = Tuning::default().overlay(&TuningLayer {
            five_hour_floor_pct: Some(2.0),
            five_hour_knee_pct: Some(20.0),
            accounts: BTreeMap::from([
                (
                    "a".to_string(),
                    AccountLayer {
                        five_hour_floor_pct: Some(5.0),
                        five_hour_knee_pct: Some(50.0),
                        priority: Priority {
                            weight: Some(2.0),
                            weekly_reset: Some(WeeklyReset::EarlierBy(hours(10))),
                        },
                        by_model: Some(vec![rule(&["*opus*"], weight(3.0))]),
                    },
                ),
                (
                    "b".to_string(),
                    AccountLayer {
                        five_hour_floor_pct: Some(6.0),
                        priority: weight(4.0),
                        ..AccountLayer::default()
                    },
                ),
                (
                    "d".to_string(),
                    AccountLayer {
                        priority: weight(7.0),
                        ..AccountLayer::default()
                    },
                ),
            ]),
        });
        assert_eq!(
            (base.five_hour_floor_pct, base.five_hour_knee_pct),
            (2.0, 20.0)
        );

        let at = WeeklyReset::At(ts("2026-11-14T08:00:00Z"));
        let over = base.overlay(&TuningLayer {
            five_hour_knee_pct: Some(30.0),
            accounts: BTreeMap::from([
                (
                    "a".to_string(),
                    AccountLayer {
                        five_hour_knee_pct: Some(70.0),
                        priority: Priority {
                            weight: None,
                            weekly_reset: Some(at),
                        },
                        by_model: Some(vec![]),
                        ..AccountLayer::default()
                    },
                ),
                (
                    "b".to_string(),
                    AccountLayer {
                        five_hour_floor_pct: Some(8.0),
                        priority: weight(0.25),
                        ..AccountLayer::default()
                    },
                ),
                (
                    "c".to_string(),
                    AccountLayer {
                        priority: weight(0.5),
                        ..AccountLayer::default()
                    },
                ),
            ]),
            ..TuningLayer::default()
        });
        assert_eq!(
            (over.five_hour_floor_pct, over.five_hour_knee_pct),
            (2.0, 30.0)
        );
        assert_eq!(
            over.accounts["a"],
            AccountLayer {
                five_hour_floor_pct: Some(5.0),
                five_hour_knee_pct: Some(70.0),
                priority: Priority {
                    weight: Some(2.0),
                    // The other form replaces the earlier override whole.
                    weekly_reset: Some(at),
                },
                by_model: Some(vec![]),
            }
        );
        // A set weight and floor replace ones already there.
        assert_eq!(
            over.accounts["b"],
            AccountLayer {
                five_hour_floor_pct: Some(8.0),
                priority: weight(0.25),
                ..AccountLayer::default()
            }
        );
        assert_eq!(over.accounts["d"], base.accounts["d"]);
        assert_eq!(over.priority("c", "m").weight, 0.5);

        // A layer silent on rules keeps them.
        let kept = base.overlay(&TuningLayer {
            accounts: BTreeMap::from([("a".to_string(), AccountLayer::default())]),
            ..TuningLayer::default()
        });
        assert_eq!(kept, base);
    }

    #[test]
    fn a_weekly_reset_moves_the_reset_or_says_why_not() {
        let reset = ts("2026-11-15T00:00:00Z");
        assert_eq!(
            WeeklyReset::EarlierBy(hours(36)).apply(reset),
            ResetApplied::Moved {
                from: reset,
                to: ts("2026-11-13T12:00:00Z")
            }
        );
        let inside = ts("2026-11-10T00:00:00Z");
        assert_eq!(
            WeeklyReset::At(inside).apply(reset),
            ResetApplied::Moved {
                from: reset,
                to: inside
            }
        );
        assert_eq!(
            WeeklyReset::At(reset).apply(reset),
            ResetApplied::Moved {
                from: reset,
                to: reset
            }
        );
        let week_start = ts("2026-11-08T00:00:00Z");
        assert_eq!(
            WeeklyReset::At(week_start).apply(reset),
            ResetApplied::NotApplied {
                reset,
                why: NotApplied::EarlierWeek
            }
        );
        let after = ts("2026-11-15T00:00:01Z");
        let applied = WeeklyReset::At(after).apply(reset);
        assert_eq!(
            applied,
            ResetApplied::NotApplied {
                reset,
                why: NotApplied::AfterReset
            }
        );
        assert_eq!(applied.effective(), reset);
    }

    #[test]
    fn value_checks_refuse_what_scoring_cannot_use() {
        for bad in [
            0.0,
            -1.0,
            f64::NAN,
            f64::INFINITY,
            5e-324,
            9.9e-7,
            1.000001e6,
        ] {
            let err = check_weight(bad).unwrap_err();
            assert_eq!(
                err,
                format!("weight must be from 0.000001 to 1000000 (got {bad})")
            );
        }
        for good in [WEIGHT_MIN, 0.01, WEIGHT_MAX] {
            assert_eq!(check_weight(good), Ok(()));
        }

        for bad in [-1.0, 168.0, f64::NAN] {
            assert_eq!(
                check_earlier_h(bad).unwrap_err(),
                format!(
                    "weekly_reset_earlier_h must be a finite number, at least 0 and below 168 (got {bad})"
                )
            );
        }
        assert_eq!(check_earlier_h(0.0), Ok(()));
        assert_eq!(check_earlier_h(167.5), Ok(()));

        assert_eq!(
            check_rule_models(&[]).unwrap_err(),
            "models must name at least one model glob"
        );
        assert_eq!(check_rule_models(&["*".to_string()]), Ok(()));
        assert_eq!(
            check_rule_sets(&Priority::default()).unwrap_err(),
            "sets neither weight nor a weekly reset; a rule that sets nothing still stops later rules from matching"
        );
        assert_eq!(check_rule_sets(&weight(1.0)), Ok(()));

        assert!(check_ramp(1.0, 15.0).is_ok());
        assert!(check_ramp(15.0, 1.0).is_err());
        assert!(check_ramp(-1.0, 15.0).is_err());
        assert!(check_ramp(1.0, f64::NAN).is_err());
    }
}
