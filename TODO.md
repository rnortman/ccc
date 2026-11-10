# TODO

## `ccc-detach`

A parent agent whose shell tool times out before a long subagent finishes loses the subagent's output. A `--detach` mode paired with a `ccc wait` command, writing the output to a file the parent collects later, would cover it; today the parent has to run `ccc` as a background job. Code site: the doc comment on `run` in `src/exec.rs`.

## `trust-scope-gate`

Every Claude spawn marks its working directory trusted in the account's `.claude.json`, unconditionally, and the marker outlives the run. A `--cwd` naming an untrusted checkout therefore runs that tree's hooks and MCP servers under the account's credentials and leaves the directory trusted for later interactive sessions. Narrowing this — an allowlist of roots, propagating only a directory some profile already trusts, or auto-trusting from `launch` alone — is a policy choice. Code site: `trust_cwd` in `src/exec.rs`.

## `procs-alive-esrch-read-is-unknown`

A `/proc/<pid>/stat` read that fails with ESRCH (the task was released between the directory lookup and the read) is reported as liveness `Unknown`, though the process is gone and the kill path already treats ESRCH as "already gone". Mapping it to `Dead` is a change to the liveness rules. Code site: `ProcId` liveness check in `src/procs.rs`.

## `test-cli-acceptance-silently-absent-off-linux`

The CLI-level acceptance tests (the routing-note tests and `cli_run_then_resume_admits_the_completed_session_from_a_fresh_process`) run on Linux only, because they rely on procfs and process-group identity; on other Unix systems that coverage is silently absent. Restore safe CLI-level coverage there without depending on Linux-only process identity. Code site: the test module in `src/exec.rs`.

## `test-no-acceptance-for-an-authorised-agent-scoped-kill`

No test proves, with real processes, that an agent with ancestry-scoped kill authority can kill its own genuine subordinate. Needs a bounded real-process test case. Code site: the ancestry-authority tests in `src/exec.rs`.

## `log-missing-index-is-reported-as-an-error`

Every command that loads the audit catalog — `ccc log`, `ccc status`, `ccc kill`, and `ccc resume` or `ccc launch --resume` given a trace prefix — run before any agent has run prints `reading audit file …/audit/traces.jsonl … No such file or directory` on stderr, because the catalog loader they share reports every failure to read the index file the same way, and the index does not exist until the first run writes it. The empty-log case should be silent for every caller of the loader: a `NotFound` on `traces.jsonl` is "no runs yet", not an error, and should print nothing beyond what an empty index file prints; every other read error should keep its diagnostic. The README's quick start runs `ccc run` before `ccc log`, so the steps as written do not hit this; a newcomer who tries `ccc status` or `ccc log` first does. Code site: the `Err(e)` arm of `load_catalog` in `src/audit_view.rs`.

## `agent-tree-session-coordination`

With `[bus] enabled`, every `ccc run`, `resume` and `launch` publishes its invocation into the agent tree on the broker (`<prefix>/agents/<publisher>/<host>/session/<id>`; the schema is in the `claude-usage-mqtt` README), and `claude-usage-mqtt-core` has the types to decode every session there. Nothing in `ccc` reads those sessions: the only agent-tree topics `ccc` consumes are the account-spend records behind the API-key budget. So no session acts on what the others are doing — sessions still pile onto the same account at the same moment, and nothing notices another session working in the same directory. Coordinating sessions through the tree needs a policy first: what a session looks at (account, model, cwd, the number of live sessions), what it does about it (pick another account, wait, warn), and where that sits relative to `select::choose`. Code site: the module doc of `src/bus.rs`.

## `api-key-at-launch`

An API key account (`[claude.api_keys.*]`) cannot be used by `ccc launch`: `--profile <key>` is refused with exit 2 and selection for a launch never considers keys. An interactive Claude session reports no cost to `ccc` — a launch's end record has no `native`, and the Stop hook payload carries only the session id, the last assistant message and the transcript path — so a key used at the terminal would drain the budget without the ledger seeing it. Lifting the refusal needs a cost figure for an interactive session, per turn or at least at its end, from a source still to be found (the transcript's usage records priced per model, or a hook payload that carries cost); a launch then charges the ledger as a headless turn does. Code site: the refusal of a key account for a launch in `src/exec.rs`.

## `api-key-assumptions-unconfirmed`

API key accounts rest on three assumptions about Claude Code and the Claude Platform that have never been checked against a real key, and no test covers them. A1: `claude -p` with `ANTHROPIC_API_KEY` set bills the key, not the subscription login in `~/.claude`. A2: a turn the Platform refuses for spend reports a message that one of the key's `exhausted_patterns` matches. That is two refusals, which may not share a message: A2a, the credit is gone, which the default pattern (`*credit balance is too low*`) is a guess at; A2b, a spend limit set in the Platform console is reached, for which there is no default pattern. A3: `total_cost_usd` on a key turn is within a few percent of what the Platform bills. In addition, `ccc` has never been watched publishing to a real broker: its bus is tested against a recording bus and unreachable addresses, and only the `claude-usage-mqtt` client is tested against a mosquitto. The README's "Before you rely on this" checklist, under "API key accounts", gives the command for each of the four and what it should show. Its A2 step provokes the spend-limit refusal and so settles A2b only. A2a cannot be provoked on demand: it is settled the first time the credit actually runs out, by whether that key turn exits 4 with the key shown exhausted, or exits 1 with the Platform's message in its end record. Done when the checklist has been run with a real key and broker, A2a has been settled in that way — or the user has decided that the spend limit is the only refusal they rely on — and the results are in the tree: a pattern added beside the existing ones for each refusal message that none matched (added, never put in place of one), and the README's "assumed, not tested" wording removed for whatever held. Code sites: the key branch of `Account::env_edits` in `src/harness/claude.rs` (A1); the default of `exhausted_patterns` in `src/config.rs` (A2); `charge` in `src/budget.rs` (A3).
