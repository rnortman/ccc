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
