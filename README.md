# ccc

`ccc` runs an *agent type* as a one-shot subprocess on Claude Code or Codex. The
caller names the type; `ccc` decides the harness, the model, the reasoning
effort and — for Claude — which account to spend, from live usage data.

```
ccc run    <agent-type> [--cwd DIR] [--prompt-file F] [--profile NAME] [--harness claude|codex] [--json] [PROMPT | -]
ccc resume <session-id | trace-prefix> [--fork] [--prompt-file F] [--profile NAME] [--json] [PROMPT | -]
ccc status [--trace PREFIX] [--json]
ccc kill   <TARGET>... [--all] [--trace PREFIX] [--now] [--grace SECS] [--no-wait] [--json]
ccc which  <agent-type> [--harness H] [--json]
ccc launch <agent-type> [--cwd DIR] [--profile NAME] [--harness H] [-- harness args...]
ccc launch --resume <session-id | trace-prefix> [--fork] [--profile NAME] [-- harness args...]
ccc launch [--cwd DIR] [--profile NAME] [--model M] [--effort E] [-- harness args...]
ccc log    [TRACE-PREFIX | --latest] [--flat] [--verbose] [--json]
```

## Audit log

`ccc log` reads the append-only records under
`~/.local/state/ccc/audit/<trace-id>.jsonl` and the chronological index in
`traces.jsonl`. Records describe `start`, `spawn`, `turn`, and `end` events;
prompts, results, callback payload details, and native diagnostics are retained
in the trace file. The trace file is authoritative. A start is written before
the index, so a process dying in that window is shown as unindexed; an indexed
trace whose file is missing is reported as a dangling entry. Missing,
unreadable, or malformed indexes do not hide trace files.

With no selector, `ccc log` lists roots. A unique trace-id prefix or `--latest`
selects a trace and renders its invocation tree. `--flat` renders strict
timestamp order, including interleaved sibling events, and `--verbose` prints
full ids and retained multiline details. `--json` is available only with a
selector and emits one object containing `trace`, optional `index`, and joined
`invocations`; the `t` tags and all known record fields are preserved, but JSON
byte-for-byte identity with the JSONL input is not promised. `--flat` and
`--json` require a selector; `--json` cannot be combined with `--verbose`.

### Resuming what `ccc log` lists

The TRACE column of the plain listing is also a resume argument. Both resume
commands accept a unique trace-id prefix and continue that trace's root
session:

```
ccc log 12345678
ccc launch --resume 12345678
ccc resume 12345678 "Continue the work"
```

`--fork` works with either form, and a `note:` on stderr names the full trace
id and the session it resolved to. The root is the invocation the index names for
the trace, or, for an unindexed trace, its single invocation without a parent;
its session is the one `ccc log` shows for it. A trace whose root is missing,
ambiguous or never recorded a session is refused rather than continued through
a child or another root. An ambiguous prefix is refused with the full candidate
ids; copy a full id from `ccc log --verbose`. To continue a particular
subagent, find its row in `ccc log <trace-prefix>` and pass its full
`session=`/`thread=` value (or its `ccc run` trailer session id); exact session
ids get no prefix matching.

An argument that names an existing session record is always that session, even
if it also matches a trace; trace lookup happens only when no record exists
under that name. An older trace resumes the same native session even if it has
continued since: the continuation, or fork, starts from that session's current
transcript, not a snapshot at the listed trace. The session's record must still
exist and pass the usual checks — an unknown session, one in use, or one whose
recorded owner is still running is refused as for an exact id.

The index records both config paths (site and agents, each marked
`(missing)` when the file did not exist; the listing's `CONFIG` and `AGENTS`
columns, the trace header's `config …` and `agents …`) and the recognized
`CCC_*` variables by value: `CCC_CONFIG`, `CCC_AGENTS_CONFIG`,
`CCC_CLAUDE_BIN`, `CCC_CODEX_BIN` and the overrides. Name-only unknown
variables are rendered as `<unrecognized>`. `CCC_TRACE_ID` and
`CCC_PARENT_INVOCATION_ID` are managed by `ccc`; users and agents should not
set them. Stripping the inherited environment breaks lineage. Claude session
ids may be pre-minted and are shown when available. `ccc` remains resident for
interactive launch, returns the harness exit status, injects Codex `notify` or
Claude `Stop` callbacks, and records the session so `ccc resume`/`ccc launch
--resume` can find it. A row that continues a recorded session is suffixed
`(resumes <inv>)` — `(forks <inv>)` for a fork — naming the invocation that
created it, or `(session created outside this trace)` when none in the trace
did. A user Codex notify is replaced; a later
passthrough Claude `--settings` replaces the injected hook. The hidden
`ccc hook` command is internal. No transcript file is parsed.

Audit writers hold the per-file lock through the end seek, tail check, complete
write, and flush, so concurrent appends do not interleave; a short write can
still lose that record, and the next append inserts a newline before its own
record when needed. Readers diagnose and skip damaged or non-UTF-8 lines
locally, so one damaged record does not hide its neighbors. An unterminated
damaged tail is treated as truncated (and may be the one record lost), while
turn counts include only readable turn records.

## Install

```
cargo install --locked --git https://github.com/rnortman/ccc
```

or, from a clone, `cargo install --locked --path .`. `claude` and `codex` are
found on `PATH`; `CCC_CLAUDE_BIN` and `CCC_CODEX_BIN` override them.

The `agents/` directory in this repo holds example agent definitions. They run
once copied to `~/.config/ccc/agents` (`$XDG_CONFIG_HOME/ccc/agents` when that
is set), the default agents directory:
`mkdir -p ~/.config/ccc && cp -r agents ~/.config/ccc/` from a clone. To keep
them elsewhere, point `agents_dir` in `~/.config/ccc/agents.toml` at the copy:

```toml
agents_dir = "~/src/my-agents"
```

## Development

```sh
make setup-hooks   # point git at .githooks; prints what to install if a hook's tool is missing
make check         # fmt, clippy, test, brenn-scrub init --check — what pre-commit runs
```

`make check` and the hooks need
[`brenn-scrub`](https://crates.io/crates/brenn-scrub) and the `gitleaks`
release it pins on `PATH` (`cargo install --locked brenn-scrub`).
`make rust-check` runs the Rust checks alone, as CI does.

## Using it from a parent agent

The prompt is an inline `PROMPT` argument, `--prompt-file F`, or stdin — via
`-` in either position (`ccc run explorer - <<'EOF'`), or by omitting it.

`ccc run` writes the subagent's final message to stdout, then a blank line, then
one trailer:

```
[ccc] session=9f22dd1b-… harness=claude model=claude-opus-5-5[1M] effort=low profile=personal
```

Every diagnostic — the account chosen, warnings, fallbacks — goes to stderr, so
stdout is only ever the subagent's message. `--json` replaces both with one
object: `{session, harness, model, effort, profile, agent, cwd, result, native}`.

Feed a follow-up question to the same context with `ccc resume <session-id>`
(or a trace prefix from `ccc log`, see [Resuming what `ccc log`
lists](#resuming-what-ccc-log-lists)); the session keeps its harness, model, effort, working directory and sandbox.
`ccc resume <session-id> --fork` starts a new session from the recorded
transcript instead; the original is left untouched and the new id is printed
in the trailer. `ccc launch --resume <session-id>` opens the same session
interactively instead; add `--fork` to sit down at a copy and leave the
subagent's own session untouched.

Exit codes:

| code | meaning |
|------|---------|
| 0 | the turn succeeded |
| 1 | the harness ran and reported a failed turn; the message is on stderr, and in text mode also on stdout |
| 2 | a usage, config or agent-definition error, before anything was spawned |
| 3 | the harness produced no usable result (crash, no final message, unparsable output, termination by `ccc kill` or a signal to `ccc`, or incomplete prompt delivery); the session id is retained when available |
| 4 | the Claude account hit a usage limit and no other profile could continue the session (see [Usage limits](#usage-limits)); the message and the session id are printed as for exit 1, so the caller can `ccc resume` later |

A subagent can easily outlive a parent's foreground shell-tool timeout (Claude
Code's Bash tool caps at 10 minutes), so run `ccc run` in the background and
collect its output when it finishes.

## Process management

`ccc status` reports process state in text or, with `--json`, as structured
records. A spawned agent sees only the subtree below its own invocation (not
itself), including running and finished subagents and their session ids.
Ancestry is matched against every recorded harness first; the nearest recorded
ancestor wins. Inherited `CCC_TRACE_ID` and `CCC_PARENT_INVOCATION_ID` are only
fallbacks, so removing or spoofing them does not widen a live managed agent's
scope, and agents cannot use `--trace`.

A human shell without an anchor sees unfinished invocations across the catalog,
including their ancestors. `--trace` selects one complete trace, and a
trace-only inherited environment also selects that trace. A harness that `ccc`
did not spawn is necessarily treated as a human shell because no record can
match it. Stripping lineage for a detached or new `run` breaks the audit tree,
although foreground `status` and `kill` authority is recovered through
ancestry.

The trust boundary is exact: the anchor comes from `/proc`, matched to `child`
records in the caller's own `~/.local/state/ccc`. Commands do not escape an
agent subtree, but this is not a sandbox against an agent that rewrites
user-owned state and already has `kill(2)`; audit trust is the trust in that
directory.

`idle` is a best-effort transcript-mtime signal, so a long tool call can appear
idle. Ended invocations are `done`, `failed`, or `killed`; unfinished
combinations are `running`, `stopping`, `orphaned`, `lost`, or `unknown`.
Matching zombies and dead tasks count as dead and are never presented as able
to do work.

`ccc kill` resolves invocation prefixes or full session ids and signals leaves
first. For an agent, `--all` means all targets in its local subtree; a human
shell must also provide `--trace` when using `--all`. Normal supervisor
signalling forwards to the harness and then its descendants as needed; an
orphan is signalled directly. `--now` skips the normal grace, `--grace SECS`
overrides it, `--no-wait` returns without waiting for confirmation, and
`--json` selects structured output. Targets are identity-checked before any
signal. Exit status 0 means all targets are dead or were signalled, 1 means a
target survived, and 2 means usage or scope was refused.

Session sidecars are persisted early and protected by
`sessions/<id>.lock`, and an orphan must pass the admission check before it can
be controlled. `--no-session-persistence` and `--ephemeral` are passthrough
options that intentionally make a harness session unrecoverable.

Shutdown uses the configured `[process].kill_grace_s` (10 seconds by default),
plus a fixed one-second capture drain. A held-open-stream warning means that a
process left behind by the harness still owns an output pipe. Signalling is
Linux-first; on non-Linux systems liveness can remain unknown.

## Agent definitions

One `<type>.md` per agent type in `agents_dir` (default `~/.config/ccc/agents`);
the repo ships `agents/` with three stubs. The file stem is the type and must
equal `name`:

```yaml
---
name: explorer
description: Answers a few related questions about a codebase, with citations.
model: claude-opus-5-5[1M]        # Claude model, passed to --model verbatim
effort: low                       # Claude --effort; optional
tools: Read, Grep, Glob, Bash     # Claude tool list; optional
harness: claude, codex            # allowed harnesses; first is the preference
# harness: [claude, codex]         # YAML-list form is equivalent
codex_model: gpt-5.6-terra
codex_effort: low                 # -> model_reasoning_effort; optional
codex_sandbox: read-only          # read-only | workspace-write
---
The body is the system prompt: Claude gets it as the agent's `prompt`, Codex as
`developer_instructions`.
```

`harness` accepts a case-insensitive scalar, comma-separated string, or YAML
list. Entries are trimmed, must be unique, and form an ordered allowed set;
the first is the definition preference. Without a policy, at least one of
`model` / `codex_model` is required. With one, at least one allowed harness
must have its corresponding model (an override may supply the other model).
The base body is required. Optional body-only overlays named
`<type>.claude.md` and `<type>.codex.md` are loaded when present; their whole
contents become that harness's prompt, with surrounding whitespace removed.
Overlays have no frontmatter: a leading `---` is rejected, and keys belong in
the base `<type>.md`.
Agent type names ending in `.claude` or `.codex` are reserved for these
overlays; a definition whose type name ends in one is refused.

## Configuration

ccc reads two TOML files with disjoint keys:

| | Site config | Agents config |
|---|---|---|
| Path | `CCC_CONFIG`, else `~/.config/ccc/config.toml` | `CCC_AGENTS_CONFIG`, else `~/.config/ccc/agents.toml` |
| Keys | `[claude]`, `[codex]`, `[mqtt]`, `[process]`, `[launch]` | `agents_dir`, `default_harness`, `[agents.*]` |

The site config describes the machine and the operator: accounts and how they
are scored, Codex approval, the usage broker, kill grace, roleless-launch
defaults. The agents config describes one set of agent definitions and how to
route them, so a workflow tool can switch it with `CCC_AGENTS_CONFIG` without
touching the site config. `~/.config` in these paths stands for
`$XDG_CONFIG_HOME` when it is set.

A missing file at a default path means every default for that file. A path
named by `CCC_CONFIG` or `CCC_AGENTS_CONFIG` must exist; a missing one is an
error (exit 2) naming the variable. An empty variable counts as unset.

A key that belongs in the other file is refused, naming both files:

```
parsing config /home/u/.config/ccc/config.toml (default):
  `agents` belongs in the agents config, not the site config.
  agents config: /home/u/.config/ccc/agents.toml (default)
```

A key in the wrong file is fixed by moving it to the file the message names.
Misspelled keys are refused as `unknown field`.

Agents config:

```toml
agents_dir = "~/.config/ccc/agents"
default_harness = "claude"

[agents.explorer]          # per-agent overrides; "*" is the wildcard key
harness = "codex"
model = "..."              # Claude model
effort = "..."
codex_model = "..."
codex_effort = "..."
```

Site config:

```toml
[launch]                    # a `ccc launch` with no agent type; Claude only
model = "opus"              # an alias below or a full model string; default "opus"
effort = "high"             # Claude --effort; optional, default: Claude's own
[launch.aliases]            # layered over the built-ins; an entry here wins
opus = "claude-opus-5-5[1M]"
fable = "claude-fable-5-1[1M]"

[claude]
auth = "config-dir"        # config-dir: CLAUDE_CONFIG_DIR=<profiles_dir>/<profile>
                           # token: CLAUDE_CODE_OAUTH_TOKEN from <token_dir>/<profile>.token
profiles = ["personal", "work"]   # preference order; default empty: the ambient login
permission_mode = "bypassPermissions"
token_dir = "~/.claude-tokens"
profiles_dir = "~/.claude-profiles"
extra_args = []            # appended to the claude argv verbatim

[claude.selection]
five_hour_floor_pct = 1.0   # 5-hour window percent *remaining*: util5 99
five_hour_knee_pct = 15.0   # util5 85
min_time_left_h = 1.0
stale_grace_s = 3600
fable_models = ["*fable*"]                        # globs over the resolved model
general_windows = ["seven_day", "7-day_opus*"]    # globs over snapshot window keys
fable_windows = ["7-day_fable*"]
fable_general_ratio = 0.5   # 1% of the Fable bucket costs this much of the general one
reserve_general_for_fable = true
mqtt_quiesce_ms = 1500
mqtt_timeout_ms = 8000      # bounds the whole read: connect, fetch, disconnect

[claude.selection.profile.work]   # per-profile override; either key, empty by default
five_hour_knee_pct = 60.0   # ramp from util5 40
five_hour_floor_pct = 20.0  # excluded from util5 80

[mqtt]                      # absent: the usage client's own config discovery
config = "~/.config/claude-usage-mqtt/config.toml"

[codex]
approval = "never"
extra_args = []
```

Environment overrides beat the config file. Each has a per-agent and an `_ALL`
form, with the agent type upper-cased and non-alphanumerics turned into `_`
(`comment-rewriter` → `CCC_MODEL_COMMENT_REWRITER`):

```
CCC_HARNESS_<AGENT>       CCC_HARNESS_ALL
CCC_MODEL_<AGENT>         CCC_MODEL_ALL
CCC_EFFORT_<AGENT>        CCC_EFFORT_ALL
CCC_CODEX_MODEL_<AGENT>   CCC_CODEX_MODEL_ALL
CCC_CODEX_EFFORT_<AGENT>  CCC_CODEX_EFFORT_ALL
```

`inherit`, `default`, `-` and the empty string mean "no override".

Harness routing walks these seven rungs: `--harness`,
`CCC_HARNESS_<AGENT>`, `CCC_HARNESS_ALL`, `[agents.<type>]`,
`[agents."*"]`, the definition preference, and `default_harness`. A
disallowed choice from the first, per-agent environment, or per-agent config
is rejected. A disallowed population default (`_ALL`, wildcard config, or
global default) is skipped with a diagnostic note. Other fields retain the
same per-agent, `_ALL`, specific-config, wildcard-config, definition order.
`ccc which <type>` prints the result and reports the selected prompt source.
Before its routing notes it prints the two config files on stderr, each with
`(CCC_CONFIG)` / `(CCC_AGENTS_CONFIG)` or `(default)`, and
`(missing, defaults)` when a default path does not exist:

```
config: /home/u/.config/ccc/config.toml (default)
agents config: /home/u/.config/ccc/configs/review/default.toml (CCC_AGENTS_CONFIG)
```

`--json` carries the same two paths as `config` and `agents_config`.

## Account selection

With `[claude] profiles` empty — the default — and no `--profile`, a Claude
run uses the caller's own login: `ccc` exports neither `CLAUDE_CONFIG_DIR` nor
`CLAUDE_CODE_OAUTH_TOKEN`, reads no usage data and selects nothing. It still
removes `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` and `CLAUDE_CODE_USE_*`, so
the child runs on the Claude Code login rather than an API key or cloud
provider, and says on stderr which of them it removed. The trailer says
`profile=-`, the session sidecar records no profile, a resume stays on the
ambient login, and a usage limit ends in exit 4 with nothing to continue on.
An inherited `CLAUDE_CONFIG_DIR` is left in place, and the trust marker below
goes to `$CLAUDE_CONFIG_DIR/.claude.json` when it is set (a relative value
is taken from the agent's directory, as the child takes it),
`$HOME/.claude.json` otherwise. `ccc which` prints `ambient login ([claude] profiles is
empty)` in place of the candidate table. `[mqtt]` and `[claude.selection]` are
inert in this mode, but a `[claude.selection.profile.<name>]` table is still
refused, as it is for any name not in `profiles`. `--profile NAME` works as
below.

A non-empty `profiles` list turns on everything that follows. For a Claude run
without `--profile`, `ccc` reads the retained
`claude/usage/#` snapshots once and scores every configured profile that has a
local credential:

```
gate5   = clamp(((100 - util5) - floor) / (knee - floor), 0, 1)
rate_x  = (100 - util_x) / hours_until_reset_x

fable model:  score = gate5 × min(rate_f, rate_g / ratio)
otherwise:    score = gate5 × max(0, (100 - util_g) - ratio × (100 - util_f)) / hours_g

tier    = 1 if gate5 == 1 and score > 0, else 2
winner  = lowest tier, then highest score, then earliest in profiles
```

`floor` and `knee` are percent of the 5-hour window *remaining*, so
`util5 = 100 - remaining`; each profile uses its own
`[claude.selection.profile.<name>]` values where set and the global ones
otherwise.

Fable work spends both weekly buckets, so it is scored by whichever runs out
first; other work is scored against the general remainder net of what the Fable
remainder will need, so both buckets can be spent by the end of the week. The
reserve term is zero for an account with no Fable bucket — there is nothing to
hold quota back for. A profile is excluded outright when its auth is not `ok`,
its reading is stale or carried forward, or its 5-hour window is at or below
its floor. An account above its knee with a positive score (tier 1) beats any
account in its 5-hour ramp or scoring zero (tier 2), however their weekly
rates compare: running out of 5-hour window mid-session is a hard failure, an
uneven weekly burn a soft one. Within a tier the highest score wins, ties go
to `profiles` order.

A standard seat's 5-hour window moves about five times faster than a premium
(5×) seat's for the same work, and the snapshot can be minutes behind the
sessions already started on it, so such an account wants a higher knee and
floor than the global ones; the per-profile table above is where they go. A
table naming a profile that is not in `profiles` is refused.

A missing 5-hour window is an unspent one: every account has that window. A
missing weekly window is a bucket the account does not have — an entitled
account with an unspent bucket publishes it at 0% — so a model that spends
that bucket cannot use the account, and the profile is excluded with the
missing bucket named as the reason.

Nothing here can block a spawn: an unreachable broker, an offline daemon or a
table of unusable candidates falls back to the first credentialed profile in
`profiles` order and says so on stderr. `ccc which <type>` prints the whole
table — utilisations, hours, gate, tier, score and the reason for each
exclusion, or why a candidate is in tier 2 — and `--json` adds each profile's
resolved floor and knee.

The selection knobs are validated when the config is read: a floor above the
knee, a percentage outside 0–100, a negative ratio, or a zero
`min_time_left_h`/`mqtt_*_ms` is refused by name rather than quietly producing
scores that cannot be compared.

Before every Claude spawn, `ccc` marks the working directory trusted in the
account's Claude Code config — `<profiles_dir>/<profile>/.claude.json` under
`auth = "config-dir"`, `$HOME/.claude.json` under `auth = "token"`, the
ambient login's as above — by setting
`projects[<cwd>].hasTrustDialogAccepted` to `true` and leaving everything else
in the file alone. That is what lets `ccc launch` skip the trust dialog instead
of blocking on a human, and what lets `ccc run` load the workspace's own hooks
and MCP servers rather than silently ignoring them. `--cwd` is the operator's
statement of trust, so there is no knob; the home directory is never marked,
because Claude Code keeps its trust for the session only. A config file that
cannot be read or written leaves the spawn as it was and says so on stderr.

The marker is persistent and account-wide: it outlives the run and a later
interactive `claude` in that directory under the same account does not ask
either. Pointing `ccc` at a checkout you have not read is therefore a decision
to let that checkout's hooks and MCP servers run under the account's
credentials.

`ccc resume` and `ccc launch --resume`, forks included, select the Claude
account anew on every turn, exactly as a fresh run does: the best-scoring
credentialed profile right now, under the currently configured `auth`, or the
first credentialed profile in `profiles` order when the usage data cannot
decide. The account and auth the session last ran under are recorded in its
sidecar for reference but play no part in the choice, so a session moves off a
still-usable account whenever another ranks higher. `--profile NAME` on either
form uses that account instead, bypassing the usage data; it must have a
credential under the configured `auth` and need not be listed in `profiles`.
A Codex session ignores `--profile`.

Moving a session between accounts relies on every profile reading one shared
transcript store: profiles created by `claude-profile` (a script that ships
with [claude-usage-mqtt](https://github.com/rnortman/claude-usage-mqtt)) share
it — each profile dir's `projects` is a symlink into `~/.claude` — so any
account can read the session. `ccc` assumes that layout and neither creates
nor checks it. On a profile directory outside it the resumed turn fails with
Claude's own "no conversation found"; retry with `--profile` naming an account
that can read the transcript.

### Usage limits

A Claude turn that fails with Claude Code's usage-limit message
(`You've hit your session limit · resets …`, or one naming a `usage limit`)
is continued once on another account: `ccc` sets the limited profile aside,
selects again, and when another profile wins outright it resumes the session
there with the prompt `Your previous turn was cut off by a usage limit on
another account. Continue from where you stopped.` That turn's result and
trailer — its profile — are the invocation's, and stderr names the hand-off
(`ccc: <profile> hit a usage limit: <message>; continuing session <id> on
<other>`). There is no retry when `--profile` was given, when no other profile
has a credential, or when selection could only fall back to `profiles` order;
nor for a retry that is limited in turn. Those end in exit 4, with the session
id in the trailer and `ccc: <profile> hit a usage limit: <message>; <why it was
not continued>` on stderr; a hand-off that could not start says so instead.
The same lines go into the invocation's `end` audit record as warnings,
together, after a hand-off, with the limited turn's native report. A
continuation that ran but whose result could not be written fails as that
write failure, not as exit 4. Codex limits are not recognised and stay
exit 1.

## Roleless launch

`ccc launch` with neither an agent type nor `--resume` opens a plain
interactive Claude Code session: no agent definition, no `--agent`, but ccc's
usage-driven account choice for the model, its preset session id and its
session recording. It is Claude only, so `--harness` is refused.

The model is `--model`, else `[launch] model`, else `opus`; the effort is
`--effort`, else `[launch] effort`, else Claude's own default. Environment
overrides, `[agents.*]` and `[agents."*"]` do not apply: they configure agent
types, and this launch has none. The model is looked up once in
`[launch.aliases]`, then in the built-ins (`opus` → `claude-opus-5-5[1M]`,
`fable` → `claude-fable-5-1[1M]`); any other name, such as a full model string
or Claude's own `sonnet`, is passed through verbatim. The account is scored
against the expanded model, so `--model fable` picks a profile with Fable
budget. A bad `[launch]` value fails only a roleless launch, naming the key,
and a `--effort` on the command line bypasses a bad `[launch] effort`.

The session is recorded like any other launch, with no agent, and
`ccc resume` and `ccc launch --resume` reopen it with its model, effort and
working directory.

## Interactive resume and forking

Every `ccc launch` records its session like `ccc run` does: a fresh Claude
launch under its preset id before the harness starts, a Codex launch under the
thread id of its last completed turn when it exits. A Codex launch that quits
before any turn completes records nothing and says so on stderr. Switching
sessions inside the TUI (`/resume`, `/clear`, `/new`) is recorded too: a
session that already has a record keeps it, and a brand-new one is recorded
under this launch's agent and settings.

`ccc launch --resume <session-id>` reopens a recorded session — from `run`,
`resume` or `launch` — at the terminal; a trace prefix from `ccc log` reopens
that trace's root session. The harness, agent, model, effort,
sandbox and working directory come from the record, so `--cwd` and
`--harness` are refused; the Claude account is selected as it is for
`ccc resume`, `--profile` included, and the record is rewritten to the account
the session ran under. A session whose `ccc` is still running is refused with the same
message as `ccc resume`; fork it instead.

`--fork` (on `resume` and `launch --resume`) starts a new session from the
recorded transcript. The original's sidecar and lock are never touched, so a
live subagent can be forked while it works: `ccc` says so on stderr and the
fork sees the transcript as written so far. The fork's id is learned from the
harness — the JSON result, the Stop hook or the Codex notify — and saved with
`forked_from` naming the parent; it is printed in the trailer and the end
record. A fork quit before its first completed turn leaves nothing behind and
reports no session.

Verified with codex-cli 0.155.1: both `codex resume --include-non-interactive`
and `codex fork` accept a thread created by `codex exec` and show its
transcript.

The session's recorded harness remains pinned even if a later definition
policy disallows it. The matching body-only overlay is re-read on every resume,
so edited or newly added prompt content takes effect for that recorded harness.
