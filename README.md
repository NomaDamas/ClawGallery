# ClawGallery

ClawGallery is a Rust CLI for an agent-native screenshot gallery. It registers screenshot/image folders, bootstraps and polls for new images, stores metadata in JSONL, can call a visual-understanding model for titles/captions, safely renames files, and supports keyword search.

## Install / build

```bash
make ci
cargo install --path .
```

## Quickstart

```bash
clawgallery init
clawgallery folder add ~/Desktop
clawgallery bootstrap
clawgallery search screenshot
clawgallery caption --dry-run
clawgallery rename --dry-run
```

Continuous polling:

```bash
clawgallery poll --interval 30
```

## State files

By default state is stored under `~/.config/clawgallery`:

- `config.json`
- `folders.jsonl`
- `images.jsonl`
- `captions.jsonl`
- `renames.jsonl`
- `errors.jsonl`

Set `CLAWGALLERY_CONFIG_DIR=/path/to/state` to override this location.

## Visual model auth

ClawGallery uses OpenAI-compatible `/v1/responses` requests for image understanding.

Supported environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (defaults to `https://api.openai.com/v1`)
- `CLAWGALLERY_MODEL` (defaults to `gpt-4.1-mini`)

Best-effort Codex auth reuse is supported by reading `$CODEX_HOME/auth.json` or `~/.codex/auth.json` for `OPENAI_API_KEY` or `tokens.access_token`, matching the current Codex auth-file shape observed from the OpenAI Codex Rust codebase. This is intentionally opportunistic so ClawGallery does not depend on a private/stable Codex API.

## Commands

```text
clawgallery init
clawgallery folder add <path> [--recursive]
clawgallery folder remove <id-or-path>
clawgallery folder list
clawgallery bootstrap [--folder <id>] [--path <path>]
clawgallery poll [--once] [--interval <seconds>]
clawgallery caption [--missing] [--file <path>] [--dry-run] [--model <model>]
clawgallery rename [--apply] [--file <path>] [--style title|caption|date-title]
clawgallery search <keywords...> [--limit <n>]
clawgallery status
clawgallery skill path|print
```

## Rename safety

Rename is dry-run by default. `--apply` is required to modify files. ClawGallery strips unsafe filename characters, preserves extensions, reserves suffix space for collisions, and refuses to overwrite existing files.
