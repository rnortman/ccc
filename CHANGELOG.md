# Changelog

## Unreleased

- **Upgrade before the usage daemon.** `ccc` now reads usage data from
  `claude-usage-mqtt` 0.2.0, the daemon that publishes each account's quota
  usage. Deploy this `ccc` first: an older `ccc` cannot read 0.2.0's reports
  and falls back to plain preference order when choosing an account. This
  version reads both the old and the new report format.
- When an account's usage figures are out of date, `ccc which` and the
  fallback message now say why, in the daemon's own words (with control
  characters escaped so they cannot disturb the terminal).

## 0.1.0 — 2026-10-13

Initial public release.
