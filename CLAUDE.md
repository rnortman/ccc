# Claude-Code-Codex (CCC) Harness

This is a small framework for creating multi-agent workflows that cut between Claude Code and Codex as the agent harness. The user can launch either app as the starting harness, and it can launch subagents (if its agent definition tells it to). But instead of using the built-in subagent facility in the harness, it shells out to a CLI app, which in turn shells out to `claude -p` or `codex exec`, which are the one-shot modes (resumable for multi-shot) of each harness respectively.

## Test fixture hygiene

Every process a test spawns must exit on its own, with no help from the test,
the code under test, or a human. Two terminations qualify, and nothing else:

- **Wall clock:** the fixture is dead within 10 seconds of starting, whatever
  any other process does.
- **EOF from the test process:** the fixture is an idle reader of its own
  stdin, whose sole writer is the test process, so the test process's exit
  ends it. This exception only prevents outliving the runner; it is not a
  wall-clock bound.

Rules:

1. No unbounded loop or read in anything a test executes: no `while :`,
   `while true`, `for ((;;))`, `yes` without `head`, `tail -f`,
   `sleep infinity`, or a read on an fd the test process's death does not
   close (a fifo the fixture creates, a tty, a file). A fixture that must stay
   alive does so with `sleep N`, or `sleep N & wait` when it must react to a
   trapped signal.
2. The 10-second bound covers the complete executed fixture: the script, its
   readiness waits and every descendant it forks.
3. Never spin. A fixture waits blocked in `sleep`, `wait`, `read` or its
   stdin; a readiness poll carries an iteration cap.
4. Ignoring `SIGTERM` is allowed only together with the 10-second bound.
5. The suite keeps exactly one kill-on-drop guard, `SleepFixture` in
   `src/procs.rs`, and adds none. Supervisor-spawned test children get no RAII
   guard: on the failure paths such a guard would cover, the supervisor never
   returns to the helper, so the guard never drops.

`testutil::tests::no_unbounded_fixtures_in_test_source` is a tripwire for the
known spellings, allowlisted per line with `// fixture-scan: allow <reason>`.
It is not proof of a fixture's lifetime; the rules are.
