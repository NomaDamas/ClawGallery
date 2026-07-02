.PHONY: fmt lint check build test ci

fmt:
	cargo fmt --all -- --check

lint:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

check:
	cargo check --workspace --all-targets --all-features

build:
	cargo build --workspace --all-features

test:
	cargo test --workspace --all-features

ci: fmt lint test build
