# Changelog

## 0.3.0 — 2026-10-27

- **Upgrade note: subagents no longer bypass permissions by default.** The
  default `[claude] permission_mode` is now `auto` instead of
  `bypassPermissions`, so Claude subagents fail closed: an action Claude's
  classifier denies fails as a tool call, and where Claude Code reports auto
  mode unavailable no subagent can do its work, instead of every
  permission check being off. The mode is read on every spawn, resumes
  included, so a session started before the upgrade resumes in `auto`. A site
  that relied on the old default keeps it with `permission_mode =
  "bypassPermissions"` under `[claude]`.

## 0.2.1 — 2026-10-22

- `ccc` is installable from crates.io: `cargo install --locked ccc-harness`
  (the crate is `ccc-harness`; the binary is still `ccc`). The usage-data
  libraries it depends on, `claude-usage-mqtt-core` and
  `claude-usage-mqtt-client`, now come from crates.io at 0.2.1 instead of a git
  revision; they are the same code as 0.2.0.
- The README now opens with installation: requirements (Linux, a Rust
  toolchain, a C compiler for the TLS stack, a logged-in `claude`), the install
  lines, and a quick start from nothing to a first `ccc run` — including the
  two things to know before it: subagents run with
  `--permission-mode bypassPermissions` by default, and every Claude run marks
  its `--cwd` trusted for that account. The reference sections follow, with the
  audit log moved after the features that produce it.
- The published crate leaves out the repository's git hooks, Claude Code
  settings, CI workflow and agent instructions; the binary is unchanged.
- `make publish-check` dry-runs `cargo publish`.
- `make setup-hooks` points git at the repo's `.githooks` and reports any
  missing hook tools; the README's Development section documents it.
- Known gaps in `ccc` that were noted only in code comments are now listed in
  `TODO.md`.

## 0.2.0 — 2026-10-18

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
