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

State is split across three append-only JSONL event logs joined by `image_id`:

- `bootstrap` writes new `ImageRecord`s to `images.jsonl`. Pass `--prune` to also append `active=false` records for files that have disappeared from disk.
- `caption` writes one `CaptionRecord` per successful run to `captions.jsonl`.
- `rename --apply` writes a `RenameRecord` to `renames.jsonl` and appends a fresh `ImageRecord` with the new `path` (preserving the original `id` and `sha256`).

Each downstream command (`search`, `status`, `caption`, `rename`) treats the latest record per path as authoritative and ignores `active=false` (pruned) entries.

## Visual model auth

ClawGallery supports multiple vision providers via a unified abstraction.

### OpenAI-compatible (default)

Uses OpenAI-compatible `/v1/responses` requests for image understanding.

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (defaults to `https://api.openai.com/v1`)
- `CLAWGALLERY_MODEL` (defaults to `gpt-4.1-mini`)

Best-effort Codex auth reuse is supported by reading `$CODEX_HOME/auth.json` or `~/.codex/auth.json` for `OPENAI_API_KEY` or `tokens.access_token`.

### Google Gemini

Uses the Gemini Generative Language API.

- `GEMINI_API_KEY`
- Default model: `gemini-2.5-flash` (set via `--model` or config)

### Switching providers

Set the provider in config or override per-run:

```bash
clawgallery caption --provider gemini --model gemini-2.5-flash
clawgallery caption --provider openai-compatible --model gpt-4.1-mini
```

## Commands

```text
clawgallery init
clawgallery folder add <path> [--recursive]
clawgallery folder remove <id-or-path>
clawgallery folder list
clawgallery bootstrap [--folder <id>] [--path <path>] [--prune]
clawgallery poll [--once] [--interval <seconds>] [--prune]
clawgallery caption [--missing] [--file <path>] [--dry-run] [--model <model>] [--provider <provider>]
clawgallery rename [--apply] [--file <path>] [--style title|caption|date-title]
clawgallery search <keywords...> [--limit <n>]
clawgallery status
clawgallery skill path|print
```

## Rename safety

Rename is dry-run by default. `--apply` is required to modify files. ClawGallery strips unsafe filename characters, preserves extensions, reserves suffix space for collisions, and refuses to overwrite existing files.
