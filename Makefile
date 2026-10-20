.PHONY: check rust-check fmt clippy test init-check setup-hooks scrub-tree

check: fmt clippy test init-check

# CI's check job runs this target.
rust-check: fmt clippy test

fmt:
	cargo fmt -- --check

clippy:
	cargo clippy --locked --all-targets -- -D warnings

test:
	cargo test --locked

init-check:
	brenn-scrub init --check

setup-hooks:
	git config core.hooksPath .githooks
	@rm -f .git/hooks/pre-commit
	@command -v brenn-scrub >/dev/null 2>&1 || echo "brenn-scrub not on PATH: cargo install --locked brenn-scrub"
	@command -v gitleaks >/dev/null 2>&1 || echo "gitleaks not on PATH: install the release brenn-scrub pins (brenn-scrub reports it)"
	@echo "setup-hooks: done."

scrub-tree:
	brenn-scrub tree
