.PHONY: fmt lint check build test ci

fmt:
	cargo fmt --all -- --check

lint:
	cargo clippy --all-targets --all-features -- -D warnings

check:
	cargo check --all-targets --all-features

build:
	cargo build --all-features

test:
	cargo test --all-features

ci: fmt lint test build
