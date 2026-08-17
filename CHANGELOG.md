# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] — 2026-08-17

### Fixed

- Bootstrap is now truly incremental: unchanged files skip SHA-256, metadata-only touches keep `image_id`, and content overwrites reuse the same id
- Captions are SHA-aware, so fake changes do not recaption and overwritten bytes mark the old caption stale
- `caption --file` binds `source_sha256` to the live file bytes and pre-bootstrap captions keep working after later bootstrap
- VDR consumes bootstrap state instead of rehashing the library, streams hashes, and replaces vectors atomically

[0.1.2]: https://github.com/NomaDamas/ClawGallery/releases/tag/v0.1.2

## [0.1.0] — 2026-07-03

First public release on crates.io (`clawgallery`, `clawgallery-vdr`).

### Added

- Agent-native CLI for indexing local screenshots and photos into append-only JSONL state
- Folder registration, bootstrap, poll, and background daemon (LaunchAgent / systemd user)
- Vision captioning via OpenAI-compatible providers and Google Gemini
- Hybrid search: fzf-style keyword matching over captions/paths, fused with optional VDR embeddings
- Local Visual Document Retrieval (`vdr`) on embedded SQLite with MaxSim scoring
- Packaged MLX ColQwen2.5 auto-start path on macOS; optional external ColQwen2 / Jina Omni servers
- Safe rename workflows (dry-run by default, meaningful-filename gate, undo, no-clobber moves)
- Exact and visually-similar dedup reports (report-only; never bulk-delete)
- API-key redaction on error paths (`errors.jsonl` and stderr)
- Bundled OpenClaw / agent skill under `skills/clawgallery`

### Security notes for operators

- Managed embedding servers bind to `127.0.0.1` by default; non-loopback hosts require `--allow-remote`
- Rename never overwrites existing targets and requires `--apply`

[0.1.0]: https://github.com/NomaDamas/ClawGallery/releases/tag/v0.1.0
