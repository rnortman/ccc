# Changelog

## Unreleased

## 0.4.0 — 2026-11-10

- **New harness: Goose, for self-hosted models.** ccc can now run agents
  through [Goose](https://github.com/aaif-goose/goose) against an
  OpenAI-compatible server you host (llama-server, vLLM, ollama). Because such
  models get less trust than Claude, Goose agents are locked down hard: they
  get only the tools ccc provides, and only the ones their definition names.
  A definition opts in with `harness: goose` and a `goose_model`, picks its
  tools with `goose_tools` (`read`, `write`, `shell`, `ccc`), and, if it may
  start subagents, lists which agent types in `goose_agents`. Definitions that
  do not name Goose in `harness` never run on it. Site settings go under
  `[goose]` in the config, model-server settings under `[goose.env]`, and
  `CCC_GOOSE_BIN` / `CCC_GOOSE_MODEL_*` work as they do for the other
  harnesses. `agents/goose-explorer.md` is a read-only example.
- **You decide which directories a Goose agent can touch.** It may read its
  working directory plus anything passed with the new `--allow-read` flag on
  `ccc run` and `ccc launch`; write with `--allow-write` (and the working
  directory, if it has `write`); and start subagents in `--allow-exec`
  directories. ccc refuses grants that would let the model tamper with things
  that run later: a write directory may not cover ccc's own config or state,
  dot-directories in `$HOME`, anything on `PATH`, harness binaries, or a
  project a subagent will run in. Any directories a Goose agent grants to the
  subagents it starts must lie within its own grants (a Claude or Codex
  subagent still has whatever tools its own definition gives it). The README's *Goose confinement* section has the full
  rules and what they guarantee.
- **Requirement: bubblewrap.** Goose agents that read, write or run commands
  need [bubblewrap](https://github.com/containers/bubblewrap) (`bwrap`) 0.8 or
  later with unprivileged user namespaces. Their file and shell tools run in a
  bubblewrap sandbox with no network, a cleared environment, a read-only root
  and only the granted directories mounted; a system-call filter also blocks
  most socket types and io_uring. A run fails at the start if the sandbox
  cannot be created. See the README's *Goose box* section, including the few
  things the sandbox does not stop.
- **API key accounts.** A Claude Platform API key with a monthly dollar
  budget — a Team plan's included API credit, say — can now sit beside the
  subscription profiles as a last resort. Configure it under
  `[claude.api_keys.<name>]`: the key file, the budget, which models it may
  serve (no default, so nothing spends money on a model you did not name), a
  reserve to stop short of the budget, and the time zone of the monthly
  refill. `ccc` uses the key only when no subscription profile can take a
  turn because its quota is spent — never because usage data is missing or
  doubtful — and `--profile <name>` on `ccc run` / `ccc resume` uses it
  directly. `ccc launch` refuses a key, because `ccc` cannot see what an
  interactive session spends. `ccc which` shows each key and whether it would
  be used. **Before relying on a key, work through the README's "Before you
  rely on this" checklist:** which credential the Platform bills and how it
  refuses a turn when the credit is gone are assumed, not tested.
- **A spend budget per key, and the new `ccc budget` command.** Each key turn
  is charged the cost Claude Code reports into a monthly ledger on each host;
  turns that report no cost are counted as unmeasured. When the Platform
  refuses a turn for spend (recognised by the key's `exhausted_patterns`),
  the key is flagged exhausted and stays out of use until `ccc budget clear`,
  and the session carries on on a subscription profile if one can take it.
  `ccc budget` shows each key's month; `ccc budget set NAME --spent USD`
  corrects the estimate to the figure in the Platform console.
- **Sessions and key spend published to MQTT (the "bus").** With
  `[bus] enabled`, every `ccc run`, `resume` and `launch` publishes a live
  record of itself — harness, model, agent, account, working directory,
  cost, and its parent session — to the same MQTT broker that carries the
  usage data, in the shared "agent tree" that claude-usage-mqtt defines so
  other tools and dashboards can read it too. Each host also publishes its
  spend on each key, so budgets count every host's spend and an exhausted
  flag reaches every host at once. The prompt is published only if you set
  `prompt_chars`. Requires claude-usage-mqtt 0.3.0 or later, an MQTT 5 broker
  that honours message expiry, and a broker user with write access to
  `<prefix>/agents/<publisher>/#` (the README's "Bus" section has the ACL).
- **Behaviour change: under `auth = "token"`, a stray `ANTHROPIC_API_KEY`,
  `ANTHROPIC_AUTH_TOKEN` or `CLAUDE_CODE_USE_*` in the environment no longer
  stops `ccc`.** It used to refuse to run, naming the variable; it now
  silently removes them from the Claude child, as `config-dir` auth always
  has. This lets a `ccc` run nested under an API key turn start on a
  profile. The cost: such a variable left in your shell is no longer
  reported. It is still never passed to a subscription turn.
- **Behaviour change: with an API key account configured, Codex and Goose
  children no longer inherit `ANTHROPIC_API_KEY`.** `ccc` removes it, with a
  note on stderr and in the spawn record, so a nested run cannot spend the
  budgeted key unseen. A Goose provider that needs an Anthropic key must get
  it from `[goose.env]`, which is kept. Without `[claude.api_keys]` nothing
  changes.
- `ccc resume`, `ccc launch` and `ccc launch --resume` work with Goose
  sessions, reusing the directory grants recorded when the session began;
  `--fork` is refused. A Goose agent cannot resume a session a human has
  opened at the terminal.
- `ccc which` on a Goose agent shows the directory grants a run would get.
- Goose telemetry is always off, and Goose ignores your own Goose settings:
  it runs with an empty home directory and cannot download its model catalog.
- **`--model fable` and `--model opus` now follow the current release.** The
  built-in aliases expand to Claude's own aliases, `fable[1m]` and `opus[1m]`,
  instead of pinned version strings, so Claude Code picks the current model
  (Fable 5.2 today; `fable` was stuck on 5.1). The bundled example agents use
  the same aliases. A `[launch.aliases]` entry still wins if you want to pin a
  version.

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
