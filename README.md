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
clawgallery rename [--apply] [--dry-run] [--file <path>] [--style title|caption|date-title] [--force]
clawgallery search <keywords...> [--limit <n>]
clawgallery status
clawgallery skill path|print
```

## Rename safety

Rename is dry-run by default. `--apply` is required to modify files. ClawGallery strips unsafe filename characters, preserves extensions, reserves suffix space for collisions, and refuses to overwrite existing files.

### Meaningful-filename gate

`rename` skips files whose current name already looks human-meaningful and only renames stems that look auto-generated (`IMG_0034`, `PXL_20240316_080000123`, `Screenshot 2025-11-01 at 14.32.55`, `1696862563748`, `image (1)`, etc.). Classification runs in two tiers:

1. A pure local regex covers ~12 well-known camera, screenshot, messenger, and download families plus pure numeric stems and copy/sequence suffixes. A regex match means `Generic` and the stem is renamed without any model call.
2. Anything that does not match the regex is tagged `NeedsModel`. During `caption`, ClawGallery makes a separate text-only model call that sees only the filename stem (no image content) and asks whether the stem looks human-authored or auto-generated. The boolean is cached in `captions.jsonl` (`filename_meaningful: bool`) so future `rename` runs reuse the answer.

Pass `--force` to rename every captioned image regardless of name, or `--file <path>` to rename a single explicit target without consulting the gate.

`caption` only announces metadata writes (`captioned <path>`); the gate decision lives in `rename`'s output (`dry-run X -> Y`, `would skip ...`, `renamed X -> Y`). To audit the cached gate verdict for a specific image, read `filename_meaningful` from `captions.jsonl`.
