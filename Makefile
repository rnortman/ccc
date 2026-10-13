.PHONY: check rust-check fmt clippy test init-check scrub-tree

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

scrub-tree:
	brenn-scrub tree
