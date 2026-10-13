//! One retained-state read of the `claude/usage` topic tree, flattened into the
//! per-profile numbers the account scorer needs.
//!
//! The broker is never load-bearing: every failure path collapses to
//! [`UsageSource::Unavailable`] with a reason, and the caller falls back to
//! configured preference order.

use std::time::Duration;

use anyhow::{Context, Result};
use claude_usage_mqtt_client::{load_client_config, ClientConfig, Event, UsageClient};
use claude_usage_mqtt_core::{resolve_password, AuthState, ONLINE};
use jiff::Timestamp;

use crate::config::{expand_tilde_in, EnvSource, MqttConfig, SelectionConfig};

/// One usage window of one profile.
#[derive(Debug, Clone, PartialEq)]
pub struct UsageWindow {
    pub key: String,
    /// Percent of the window's allowance already spent.
    pub utilization: f64,
    pub resets_at: Option<Timestamp>,
}

/// One profile's retained snapshot, reduced to the scorer's inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileUsage {
    pub profile: String,
    /// The published auth state; only [`AuthState::Ok`] is usable.
    pub auth: AuthState,
    /// False when the numbers are last-known-good values carried forward.
    pub fresh: bool,
    pub updated_at: Timestamp,
    pub windows: Vec<UsageWindow>,
}

impl ProfileUsage {
    /// The window with the given key, if the snapshot carries one.
    pub fn window(&self, key: &str) -> Option<&UsageWindow> {
        self.windows.iter().find(|w| w.key == key)
    }
}

/// The whole retained tree, as far as account selection cares.
#[derive(Debug, Clone, Default)]
pub struct UsageReport {
    pub profiles: Vec<ProfileUsage>,
    /// The daemon's own liveness topic; `None` when it published nothing.
    pub daemon_status: Option<String>,
}

impl UsageReport {
    pub fn profile(&self, name: &str) -> Option<&ProfileUsage> {
        self.profiles.iter().find(|p| p.profile == name)
    }

    /// Whether the publishing daemon says it is up. An absent status is taken
    /// as up: the numbers are still the best available, and a missing retained
    /// liveness topic is a daemon that has never run here, not a stale one.
    pub fn daemon_online(&self) -> bool {
        match &self.daemon_status {
            Some(s) => s.trim() == ONLINE,
            None => true,
        }
    }
}

/// Usage data, or the reason there is none.
#[derive(Debug, Clone)]
pub enum UsageSource {
    Report(UsageReport),
    Unavailable(String),
}

/// Read the retained tree once. Never fails: a broker that cannot be reached,
/// authenticated to, or read within `mqtt_timeout_ms` yields `Unavailable`, so
/// a spawn is delayed by at most the timeout and never blocked. The bound
/// covers connecting, the retained read and disconnecting together.
pub fn fetch(mqtt: &MqttConfig, sel: &SelectionConfig, env: &dyn EnvSource) -> UsageSource {
    match try_fetch(mqtt, sel, env) {
        Ok(report) => UsageSource::Report(report),
        Err(e) => UsageSource::Unavailable(format!("{e:#}")),
    }
}

fn try_fetch(mqtt: &MqttConfig, sel: &SelectionConfig, env: &dyn EnvSource) -> Result<UsageReport> {
    let client_config = broker_config(mqtt, env)?;
    let quiesce = Duration::from_millis(sel.mqtt_quiesce_ms);
    let timeout = Duration::from_millis(sel.mqtt_timeout_ms);

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("starting the mqtt runtime")?;

    // The connect handshake and the disconnect have bounds of their own that
    // the caller never set, so the whole read is wrapped in the one budget the
    // caller did set.
    runtime.block_on(async move {
        let read = async {
            let mut client = UsageClient::connect(client_config)
                .await
                .context("connecting to the usage broker")?;
            let events = client
                .fetch_retained(quiesce, timeout)
                .await
                .context("reading retained usage state")?;
            client.disconnect().await;
            Ok::<UsageReport, anyhow::Error>(report_from(events))
        };
        match tokio::time::timeout(timeout, read).await {
            Ok(report) => report,
            Err(_) => anyhow::bail!(
                "the usage broker did not answer within {} ms",
                timeout.as_millis()
            ),
        }
    })
}

/// Broker connection settings: the usage client's own config discovery (or the
/// path named in `[mqtt] config`), with the password taken from the environment
/// or the file it names — never from an argument, which `ps` would show.
fn broker_config(mqtt: &MqttConfig, env: &dyn EnvSource) -> Result<ClientConfig> {
    let explicit = mqtt.config.as_deref().map(|p| expand_tilde_in(env, p));
    let mut cfg = ClientConfig::default();
    match load_client_config(explicit.as_deref())? {
        Some(file) => {
            cfg.host = file.host;
            cfg.port = file.port;
            cfg.tls = file.tls;
            cfg.username = file.username;
            cfg.topic_prefix = file.topic_prefix;
            cfg.password = resolve_password(file.password_file.as_deref())?;
        }
        None => cfg.password = resolve_password(None)?,
    }
    Ok(cfg)
}

/// Flatten the decoded retained events. A profile that published several
/// snapshots cannot happen (one retained value per topic), so the last one wins.
fn report_from(events: Vec<Event>) -> UsageReport {
    let mut report = UsageReport::default();
    for event in events {
        match event {
            Event::DaemonStatus(s) => report.daemon_status = Some(s),
            Event::Snapshot(s) => {
                let usage = ProfileUsage {
                    profile: s.profile,
                    auth: s.auth,
                    fresh: s.fresh,
                    updated_at: s.updated_at,
                    windows: s
                        .windows
                        .into_iter()
                        .map(|w| UsageWindow {
                            key: w.key,
                            utilization: w.utilization,
                            resets_at: w.resets_at,
                        })
                        .collect(),
                };
                match report
                    .profiles
                    .iter_mut()
                    .find(|p| p.profile == usage.profile)
                {
                    Some(slot) => *slot = usage,
                    None => report.profiles.push(usage),
                }
            }
            // The scalar window topics and the reset/throttle payloads restate
            // what the snapshot already carries.
            _ => {}
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use claude_usage_mqtt_core::{Snapshot, OFFLINE};

    /// The shape published on 2026-09-18, trimmed to the fields read here.
    fn snapshot(profile: &str, auth: &str, fresh: bool, windows: &str) -> Event {
        let text = format!(
            r#"{{
              "profile": "{profile}",
              "auth": "{auth}",
              "updated_at": "2026-09-18T12:00:00Z",
              "next_poll_at": "2026-09-18T12:05:00Z",
              "poll_interval_s": 300,
              "fresh": {fresh},
              "subscription_type": "max",
              "windows": [{windows}]
            }}"#
        );
        Event::Snapshot(serde_json::from_str::<Snapshot>(&text).unwrap())
    }

    const LIVE_WINDOWS: &str = r#"
        {"key":"five_hour","label":"5-hour","utilization":31.0,
         "resets_at":"2026-09-18T15:00:00Z"},
        {"key":"seven_day","label":"7-day","utilization":86.0,
         "resets_at":"2026-09-22T00:00:00Z"},
        {"key":"7-day_fable","label":"7-day (Fable)","utilization":77.0,
         "resets_at":"2026-09-22T00:00:00Z"}"#;

    #[test]
    fn snapshots_and_daemon_status_are_flattened() {
        let report = report_from(vec![
            Event::DaemonStatus(ONLINE.to_string()),
            snapshot("team03", "ok", true, LIVE_WINDOWS),
            snapshot(
                "personal",
                "stale",
                false,
                r#"{"key":"five_hour","label":"5-hour","utilization":0.0}"#,
            ),
        ]);

        assert!(report.daemon_online());
        assert_eq!(report.profiles.len(), 2);

        let team03 = report.profile("team03").unwrap();
        assert_eq!(team03.auth, AuthState::Ok);
        assert!(team03.fresh);
        assert_eq!(team03.window("five_hour").unwrap().utilization, 31.0);
        assert_eq!(team03.window("7-day_fable").unwrap().utilization, 77.0);
        assert!(team03.window("seven_day").unwrap().resets_at.is_some());

        let personal = report.profile("personal").unwrap();
        assert_eq!(personal.auth, AuthState::Stale);
        assert!(!personal.fresh);
        assert!(personal.window("7-day_fable").is_none());
        // A window with no `resets_at` is representable; the scorer supplies a
        // default rather than dropping the window.
        assert_eq!(personal.window("five_hour").unwrap().resets_at, None);
    }

    #[test]
    fn an_offline_or_absent_daemon_is_reported_honestly() {
        let offline = report_from(vec![Event::DaemonStatus(OFFLINE.to_string())]);
        assert!(!offline.daemon_online());
        assert_eq!(offline.daemon_status.as_deref(), Some("offline"));

        // Nothing retained at all: no status, no profiles, and not treated as
        // an outage — an empty tree scores as "no data for this profile".
        let empty = report_from(Vec::new());
        assert!(empty.daemon_online());
        assert!(empty.profiles.is_empty());
        assert!(empty.profile("team03").is_none());
    }

    #[test]
    fn a_later_snapshot_replaces_an_earlier_one_for_the_same_profile() {
        let report = report_from(vec![
            snapshot(
                "team04",
                "ok",
                true,
                r#"{"key":"five_hour","label":"5-hour","utilization":5.0}"#,
            ),
            snapshot(
                "team04",
                "rejected",
                true,
                r#"{"key":"five_hour","label":"5-hour","utilization":9.0}"#,
            ),
        ]);
        assert_eq!(report.profiles.len(), 1);
        assert_eq!(report.profile("team04").unwrap().auth, AuthState::Rejected);
        assert_eq!(
            report
                .profile("team04")
                .unwrap()
                .window("five_hour")
                .unwrap()
                .utilization,
            9.0
        );
    }

    #[test]
    fn a_broker_that_is_not_there_is_unavailable_not_an_error() {
        let env: std::collections::BTreeMap<String, String> =
            [("HOME".to_string(), "/nonexistent-home".to_string())]
                .into_iter()
                .collect();
        let mqtt = MqttConfig {
            config: Some("/nonexistent/client.toml".to_string()),
        };
        let sel = SelectionConfig {
            mqtt_quiesce_ms: 10,
            mqtt_timeout_ms: 50,
            ..SelectionConfig::default()
        };
        match fetch(&mqtt, &sel, &env) {
            UsageSource::Unavailable(reason) => {
                assert!(reason.contains("does not exist"), "{reason}")
            }
            UsageSource::Report(_) => panic!("a missing config file must not yield a report"),
        }
    }
}
