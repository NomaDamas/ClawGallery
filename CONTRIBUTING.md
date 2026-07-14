# Contributing to ClawGallery

Thanks for helping tidy the screenshot gallery.

## Development setup

You need Rust 1.85+ (`rustup` recommended). On macOS for the MLX VDR path you also want `uv`.

```bash
git clone https://github.com/NomaDamas/ClawGallery.git
cd ClawGallery
make ci
cargo run -- --help
```

`make ci` is the gate we keep green: `fmt` → `clippy -D warnings` → `test` → `build`.

## Before you open a PR

1. Run `make ci`.
2. Prefer small, focused changes with tests for any behavioral change.
3. Do not check secrets into the tree. Use env vars (`OPENAI_API_KEY`, `GEMINI_API_KEY`, …) only in your shell.
4. `rename` stays dry-run by default; do not change that safety contract without a strong reason and an explicit design note.
5. Prefer `--json` outputs for new agent-facing surfaces.

## Local testing tips

- Point state at a temp dir so your real library is never touched:

  ```bash
  export CLAWGALLERY_CONFIG_DIR="$(mktemp -d)"
  cargo run -- init
  ```

- Bulk `caption --missing` can cost real money. Use `--dry-run` or `--file <path>` when developing.
- VDR/MLX tests that need a live embedding server should be marked or isolated; default CI is Ubuntu and does not assume MLX hardware.

## Crate layout

| Crate | Path | Role |
|---|---|---|
| `clawgallery` | workspace root | CLI binary |
| `clawgallery-vdr` | `crates/clawgallery-vdr` | Reusable VDR index + search library |

Publish order is `clawgallery-vdr` first, then `clawgallery` (versioned dependency).

## Style

- Keep CLI help and README usage examples in sync when you change flags.
- Prefer boring, explicit Rust. Avoid speculative abstraction.
- Redact secrets on every error path (`mask_api_keys` style). New error surfaces should never print raw keys.

## Issues & questions

- Bugs and feature ideas: [GitHub Issues](https://github.com/NomaDamas/ClawGallery/issues)
- Security issues: see [SECURITY.md](SECURITY.md) — do **not** open a public issue for secrets or exploitable bugs.

By contributing, you agree your contributions are licensed under the Apache-2.0 license of this repository.
