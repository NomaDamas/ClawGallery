---
name: clawgallery
description: Use when an agent needs to index local screenshots or photos with ClawGallery, caption visual content, run keyword or VDR embedding search, inspect results as JSON, or safely organize image filenames.
---

# ClawGallery Skill

Use ClawGallery when the user asks to search, inspect, caption, index, or organize local screenshots/photos. Prefer `--json` outputs for agent workflows.

## Safety defaults

- `clawgallery rename` is dry-run unless `--apply` is explicitly passed.
- State is JSONL under `~/.config/clawgallery` unless `CLAWGALLERY_CONFIG_DIR` is set. Use the real default state only when the user wants their library updated; otherwise set a temporary config dir.
- Basic indexing/search does not require a database or embeddings.
- Model captioning requires `OPENAI_API_KEY` or best-effort Codex auth in `$CODEX_HOME/auth.json` / `~/.codex/auth.json`; Gemini captioning uses `--provider gemini` with `GEMINI_API_KEY` and defaults to `gemini-2.5-flash`.
- Do not run paid/bulk `caption --missing` without an explicit user request or a small bounded sample.

## Install or run

From the repository:

```bash
cargo install --path .
```

For one-off use without installing:

```bash
cargo run -- <subcommand>
```

## Index local images

Initialize state, register folders, then bootstrap image records:

```bash
clawgallery init
clawgallery folder add ~/Pictures
test -d ~/Pictures/screenshots && clawgallery folder add ~/Pictures/screenshots
test -d ~/Picutres/screenshots && clawgallery folder add ~/Picutres/screenshots
clawgallery bootstrap
clawgallery status
```

Use `clawgallery bootstrap --prune` when files may have been deleted or moved outside ClawGallery.

## Caption images

Check pending caption work:

```bash
clawgallery caption --dry-run
```

Caption uncaptured images when the user approved model calls:

```bash
clawgallery caption --missing
```

## Search

Keyword/fuzzy search:

```bash
clawgallery search "login error" --limit 5 --json
clawgallery search "github actions" --limit 5
clawgallery search "'github" "actions" --json
clawgallery search "!error" "^login"
```

Search atoms follow fzf-like rules: whitespace means AND, `'foo` exact substring, `^foo` prefix, `foo$` suffix, `!foo` exclusion, and `\ ` literal space. Add `--no-fuzzy` for old exact-substring behavior.

## VDR embedding search

Default local VDR path (`vidore/colqwen2-v1.0`, dimensions `128`):

```bash
uv pip install colpali-engine torch pillow
python scripts/colqwen2_server.py --device auto
clawgallery vdr sync
clawgallery search --mode embedding "github actions" --json
```

Alternative Jina Omni path:

```bash
python scripts/jina_omni_server.py --device auto
clawgallery vdr sync --model jinaai/jina-embeddings-v5-omni-small --dimensions 1024
clawgallery search --mode embedding "github actions" --json
```

Jina search must use the same model and dimensions as the synced VDR index. The Jina server enables Hugging Face `trust_remote_code`; if Hugging Face xet downloads stall on macOS, retry the first run with `HF_HUB_DISABLE_XET=1`.

## Safe rename

Preview safe filename suggestions:

```bash
clawgallery rename --dry-run
```

Apply rename only after reviewing dry-run output:

```bash
clawgallery rename --apply
```

`rename` skips files whose current name already looks human-meaningful and only renames stems that look auto-generated (`IMG_0034`, `PXL_20240316_080000123`, `Screenshot 2025-11-01 at 14.32.55`, `1696862563748`, `image (1)`). Local regex handles known camera/screenshot/messenger families. Anything that does not match the regex triggers a separate text-only model call during `caption` that judges the filename stem on its own (no image content involved). The boolean is cached as `filename_meaningful` in `captions.jsonl`. Pass `--force` to override the gate for the whole batch, or `--file <path>` for a single explicit target.

`rename --apply` self-heals when a tracked path is missing on disk: it prints `would skip (missing source) <path>`, appends an `active=false` record so future runs stop attempting that path, and continues. Per-image failures are logged to `errors.jsonl` (with API keys redacted) and reported in the final `renamed/skipped/failed` summary; one bad image no longer aborts the batch.

## Poll and sync

```bash
clawgallery poll --once
clawgallery bootstrap --prune
clawgallery vdr sync --prune
```

`--prune` appends `active=false` records to `images.jsonl` for any tracked path that is no longer on disk. The history is preserved (JSONL is append-only); downstream commands (`search`, `status`, `caption`, `rename`) automatically ignore inactive records.

## State model

ClawGallery state is split into three append-only JSONL event logs that join via `image_id`:

| File | Owner command | Records |
|------|---------------|---------|
| `images.jsonl` | `bootstrap`, `bootstrap --prune`, `rename --apply` | `ImageRecord` per discovery, prune, or rename |
| `captions.jsonl` | `caption` | `CaptionRecord` per successful model call |
| `renames.jsonl` | `rename` | `RenameRecord` per dry-run or apply attempt |
| `vdr.sqlite3` | `vdr sync` | active image embeddings plus caption embedding rows only when captions exist |

The three steps are deliberately separated so cheap, free, idempotent indexing (`bootstrap`) is decoupled from paid network calls (`caption`) and from irreversible filesystem mutations (`rename --apply`).

## Agent guidance

1. Prefer `search --json` before asking the user to locate screenshots manually; JSONL is the stable output for agents.
2. Use `caption --dry-run` to see pending work when model credentials may be absent.
3. Use `search --mode embedding --json` after `vdr sync` when semantic image or caption similarity is more useful than keyword matching.
4. Never pass `--apply` to rename unless the user requested actual file changes or an approved workflow requires it.
5. If the user asks for an experiment, capture folder checks, `status`, selected captions, exact search commands, and top JSON results.

For a compact command reference, read `references/usage.md`.
