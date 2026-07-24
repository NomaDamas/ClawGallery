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
clawgallery bootstrap
clawgallery status
```

Use `clawgallery bootstrap --prune` when files may have been deleted or moved outside ClawGallery.

## Caption model setup

Captioning writes titles/descriptions to `captions.jsonl` and may call paid/remote model APIs. Always preview the target set first:

```bash
clawgallery caption --dry-run
```

Default provider is OpenAI-compatible (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, default model from config / `CLAWGALLERY_MODEL`). Codex auth may be reused from `$CODEX_HOME/auth.json` or `~/.codex/auth.json`. Gemini uses `GEMINI_API_KEY`:

```bash
clawgallery caption --missing --provider openai-compatible --model gpt-4.1-mini
clawgallery caption --missing --provider gemini --model gemini-2.5-flash
```

Use `--file <path>` for one image, `--concurrency <n>` for bulk throughput, and `--max-retries <n>` for transient failures. Do not run bulk `caption --missing` unless the user explicitly approved model calls.

## Search

Default search is hybrid: caption/path keyword search plus VDR embedding search when `vdr.sqlite3` has active vectors. Prefer JSONL for agent workflows:

```bash
clawgallery search "login error" --limit 5 --json
clawgallery search "github actions" --limit 5
clawgallery search "'github" "actions" --json
clawgallery search "!error" "^login"
```

Search atoms follow fzf-like rules for the keyword side: whitespace means AND, `'foo` exact substring, `^foo` prefix, `foo$` suffix, `!foo` exclusion, and `\ ` literal space. When a VDR index exists, default search also runs embedding search and fuses both rankings. Add `--mode keyword` for caption/path keyword search only, `--mode embedding` for VDR only, or `--no-fuzzy` for old exact-substring behavior.

## VDR model setup

Default managed MLX path (`qnguyen3/colqwen2.5-v0.2-mlx`, dimensions `128`):

```bash
uv tool install mlx-embeddings --with pillow --with torch --with torchvision
CLAWGALLERY_PYTHON="$(uv tool dir)/mlx-embeddings/bin/python" \
  clawgallery vdr sync
clawgallery vdr status --json
```

`clawgallery vdr sync` starts the packaged MLX `/embed` daemon automatically when no `--embedding-url` and no `CLAWGALLERY_VDR_EMBEDDING_URL` are configured, waits for it, lets the model runtime download/cache weights as needed, indexes active images, and terminates it before exit. Default search and `--mode embedding` also start a managed MLX server automatically for the query embedding when an index exists and no compatible endpoint is configured. Pass `--no-auto-start` to require an external server during sync.

The inference runtime is Rust-managed but MLX/Python-based because maintained ColQwen-family late-interaction model runtimes on macOS are not currently available as a low-risk pure Rust stack. Storage remains ClawGallery's embedded SQLite multi-vector store with Rust-side MaxSim scoring.

Managed Jina v5 Omni retrieval path for Apple Silicon (`jinaai/jina-embeddings-v5-omni-small-retrieval-mlx`, dimensions `1024`):

```bash
uv venv ~/.local/share/clawgallery/jina-mlx
uv pip install --python ~/.local/share/clawgallery/jina-mlx/bin/python \
  'mlx>=0.23' tokenizers huggingface_hub 'transformers>=4.57,<5' pillow \
  torch torchvision requests librosa av
CLAWGALLERY_PYTHON=~/.local/share/clawgallery/jina-mlx/bin/python \
  clawgallery vdr sync --backend jina-mlx
```

The packaged runtime pins Hugging Face revision `049ae923674456656be891ebb22849dd58124994`. Search infers `jina-mlx` from the exact active index model ID, but `CLAWGALLERY_PYTHON` must still point at the Jina environment. The model is CC BY-NC 4.0 and restricted to noncommercial use.

If Hugging Face xet downloads stall on macOS, retry the first sync with:

```bash
HF_HUB_DISABLE_XET=1 CLAWGALLERY_PYTHON="$(uv tool dir)/mlx-embeddings/bin/python" \
  clawgallery vdr sync --prune
```

Use `--model`, `--dimensions`, `--device`, `--python`, or `--embedding-url` only when you intentionally need a non-default compatible embedding endpoint.

## End-to-end caption + VDR + search

For a real library update, run the full pipeline in this order:

```bash
clawgallery init
clawgallery folder add ~/Pictures
clawgallery bootstrap --prune
clawgallery caption --dry-run
clawgallery caption --missing
CLAWGALLERY_PYTHON="$(uv tool dir)/mlx-embeddings/bin/python" clawgallery vdr sync --prune
clawgallery search "visual query here" --json --limit 10
```

`caption --missing` enriches title/description for keyword search and rename. `vdr sync` embeds active images and captions into `vdr.sqlite3`. Plain `search` then uses hybrid retrieval; use `--mode keyword` or `--mode embedding` only to isolate one side during debugging.

## VDR compatibility options

Legacy ColQwen2 external-server path (`vidore/colqwen2-v1.0`, dimensions `128`):

```bash
uv pip install colpali-engine torch pillow
python scripts/colqwen2_server.py --device auto
clawgallery vdr sync --no-auto-start --model vidore/colqwen2-v1.0 --dimensions 128
clawgallery search --mode embedding "github actions" --json
```

Alternative external SentenceTransformer Jina Omni path:

```bash
python scripts/jina_omni_server.py --device auto
clawgallery vdr sync --no-auto-start --model jinaai/jina-embeddings-v5-omni-small --dimensions 1024
clawgallery search --mode embedding "github actions" --json
```

This external path is separate from the managed `jina-mlx` backend. Jina search must use the same model and dimensions as the synced VDR index. The Jina server enables Hugging Face `trust_remote_code`; if Hugging Face xet downloads stall on macOS, retry the first run with `HF_HUB_DISABLE_XET=1`.

## Dedup

`dedup` reports duplicate candidates; it does not delete files.

```bash
clawgallery dedup --exact --json
clawgallery vdr sync --prune
clawgallery dedup --similar --threshold 0.95 --json
```

Use `--exact` for identical `sha256` groups. Use `--similar` after VDR sync for visually similar images. Review JSON output first; if the user explicitly wants removal, delete only chosen duplicates with `clawgallery forget --file <path> --delete` or untrack without deleting via `clawgallery forget --file <path>`. Never bulk-delete dedup candidates without confirmation.

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
3. Use default `search --json` after `vdr sync` for hybrid caption/path plus VDR retrieval; use `--mode embedding` only when you need VDR-only results.
4. Never pass `--apply` to rename unless the user requested actual file changes or an approved workflow requires it.
5. If the user asks for an experiment, capture folder checks, `status`, selected captions, exact search commands, and top JSON results.

For a compact command reference, read `references/usage.md`.
