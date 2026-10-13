#!/bin/bash
# Claude Code PreToolUse: scrub added text before it lands on disk.
# Exit 2 blocks the write and feeds stderr back to the agent.
set -u

if ! command -v brenn-scrub >/dev/null 2>&1; then
    echo "brenn-scrub not on PATH; cargo install --locked brenn-scrub" >&2
    exit 1
fi

exec brenn-scrub hook
