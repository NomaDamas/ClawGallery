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
clawgallery search screenshot --json
clawgallery caption --dry-run
clawgallery rename --dry-run
```

Semantic image search through local VDR with the packaged MLX daemon on macOS:

```bash
brew install rust uv
cargo install --path .
uv tool install mlx-embeddings --with pillow --with torch --with torchvision
CLAWGALLERY_PYTHON="$(uv tool dir)/mlx-embeddings/bin/python" clawgallery vdr serve --backend mlx
clawgallery vdr sync --model qnguyen3/colqwen2.5-v0.2-mlx --dimensions 128
clawgallery search --mode embedding "login error" --json
```

The MLX path uses `mlx-embeddings` with the late-interaction ColQwen2.5 model `qnguyen3/colqwen2.5-v0.2-mlx`. `clawgallery vdr serve --backend mlx` launches ClawGallery's packaged Python `/embed` daemon, so an installed Rust binary does not need the repository's `scripts/` directory at runtime. The daemon binds to `127.0.0.1` by default and refuses non-loopback hosts unless `--allow-remote` is passed.

Legacy ColQwen2 server path (default VDR model: `vidore/colqwen2-v1.0`, dimensions `128`):

```bash
uv pip install colpali-engine torch pillow
python scripts/colqwen2_server.py --device auto
clawgallery vdr sync
clawgallery search --mode embedding "login error" --json
```

Alternative Jina Omni embedding path:

```bash
python scripts/jina_omni_server.py --device auto
clawgallery vdr sync --model jinaai/jina-embeddings-v5-omni-small --dimensions 1024
clawgallery search --mode embedding "login error" --json
```

Jina search must use the same model and dimensions as the synced VDR index. The Jina server enables Hugging Face `trust_remote_code`; if Hugging Face xet downloads stall on macOS, retry the first run with `HF_HUB_DISABLE_XET=1`.

Continuous polling:

```bash
clawgallery poll --interval 30
clawgallery poll --interval 30 --caption --sync
```

`--caption` runs missing-caption generation after each ingest pass. `--sync`
then runs `vdr sync`, so `poll --once --caption --sync` performs one
bootstrap -> caption -> VDR sync cycle. Caption or VDR failures are written to
`errors.jsonl` and reported without stopping the poll loop.

## State files

By default state is stored under `~/.config/clawgallery`:

- `config.json`
- `folders.jsonl`
- `images.jsonl`
- `captions.jsonl`
- `renames.jsonl`
- `errors.jsonl`
- `vdr.sqlite3`

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
clawgallery poll [--folder <id>] [--path <path>] [--once] [--interval <seconds>] [--prune] [--caption] [--sync] [--embedding-url <url>] [--vdr-model <model>] [--vdr-dimensions <n>] [--max-retries <n>]
clawgallery caption [--missing] [--file <path>] [--dry-run] [--model <model>] [--provider <provider>] [--concurrency <n>] [--max-retries <n>]
clawgallery rename [--apply] [--dry-run] [--file <path>] [--style title|caption|date-title] [--force]
clawgallery search [--mode keyword|embedding] <query...> [--limit <n>] [--json] [--case-sensitive] [--no-fuzzy] [--embedding-url <url>]
clawgallery vdr sync [--prune] [--embedding-url <url>] [--model <model>] [--dimensions <n>] [--max-retries <n>]
clawgallery vdr serve [--backend mlx] [--host <host>] [--port <port>] [--model <model>] [--dimensions <n>] [--device auto|mps|cpu] [--python <path>] [--allow-remote]
clawgallery vdr status [--json]
clawgallery status
clawgallery skill path|print
```

## Search syntax

`clawgallery search` scans the local JSONL state on every invocation and ranks matches by weighted fields: title matches outrank description matches, which outrank path-only matches. The default text output includes the familiar path/title/caption lines plus `score:` and `matches:` lines. Agents and brittle scripts should prefer `--json` for JSONL records, or `--no-fuzzy` to preserve the old exact substring output format.

Pass `--mode embedding` to query the VDR index instead of the keyword matcher. Embedding search sends the query to the configured local embedding server, searches both image vectors and caption vectors, then returns the best matching vector per image. JSON output uses `source: "embedding"` and `matched_field: "embedding_image"` or `matched_field: "embedding_caption"`.

Queries use nucleo/fzf-style operators:

| Syntax | Meaning | Example |
|---|---|---|
| `foo bar` | AND-match both atoms fuzzily | `clawgallery search login error` |
| `'foo` | Exact substring atom | `clawgallery search "'github"` |
| `^foo` | Prefix atom | `clawgallery search ^Login` |
| `foo$` | Suffix atom | `clawgallery search modal$` |
| `!foo` | Exclude substring | `clawgallery search login !test` |
| `!^foo`, `!foo$` | Exclude prefix/suffix | `clawgallery search !^Draft` |
| `\ ` | Literal space inside an atom | `clawgallery search github\ actions` |

Lowercase queries use smart-case matching; any uppercase atom becomes case-sensitive. Pass `--case-sensitive` to force case-sensitive matching. If the fuzzy pass returns no candidates, ClawGallery falls back to token/window typo tolerance for atoms of at least three characters. `--no-fuzzy` disables the DSL, fuzzy scoring, typo fallback, sorting, and score/matches output for compatibility with old scripts.

## Visual Document Retrieval

VDR stores image embeddings for every active image and stores caption embeddings only when an active image has caption text. The store is embedded SQLite so it needs no daemon, works well on macOS, and stays inside the same config directory as the JSONL state. `clawgallery vdr sync` is incremental: unchanged image and caption content hashes are skipped, changed files or captions are re-embedded, and `--prune` deactivates vectors for images that are no longer active after `bootstrap --prune`.

The local embedding server contract accepts `kind` values `image`, `text`, or `caption`; `caption` is caption-document text encoded like text. `role` is `document` or `query` and a compatible server may ignore it. Responses may contain either one vector per input or multi-vector embeddings per input.

```text
POST /embed
{"model":"vidore/colqwen2-v1.0","dimensions":128,"inputs":[{"kind":"image|text|caption","role":"document|query","value":"path or text"}]}
```

The packaged macOS-optimized server uses `mlx-embeddings` with `qnguyen3/colqwen2.5-v0.2-mlx` and 128-dimensional late-interaction ColQwen2.5 embeddings:

```bash
uv tool install mlx-embeddings --with pillow --with torch --with torchvision
CLAWGALLERY_PYTHON="$(uv tool dir)/mlx-embeddings/bin/python" \
  clawgallery vdr serve --backend mlx --host 127.0.0.1 --port 8765
```

The legacy local server uses `vidore/colqwen2-v1.0` with 128-dimensional ColQwen2 embeddings:

```bash
python scripts/colqwen2_server.py --host 127.0.0.1 --port 8765 --device auto
```

The alternative Jina Omni path uses `jinaai/jina-embeddings-v5-omni-small` through `sentence-transformers`, enables Hugging Face remote model code, and uses 1024 dimensions. Pass matching `--model jinaai/jina-embeddings-v5-omni-small --dimensions 1024` to `clawgallery vdr sync` when using it; embedding search should then query the same synced VDR index.

```bash
python scripts/jina_omni_server.py --host 127.0.0.1 --port 8765 --device auto
```

If Hugging Face xet downloads stall on macOS, retry the first run with `HF_HUB_DISABLE_XET=1`.
Set `CLAWGALLERY_VDR_EMBEDDING_URL` or pass `--embedding-url` to point the CLI at a different compatible local server.

## Rename safety

Rename is dry-run by default. `--apply` is required to modify files. ClawGallery strips unsafe filename characters, preserves extensions, reserves suffix space for collisions, and refuses to overwrite existing files.

When `rename --apply` encounters a tracked path that no longer exists on disk (already renamed, deleted externally, etc.) it prints `would skip (missing source) <path>`, appends an `active=false` record so the entry stops following the live set, and continues with the rest of the batch. Per-image rename failures (collision, permission, IO) are logged to `errors.jsonl` and the run prints a final `renamed N, skipped M meaningful-looking name(s), failed K` summary instead of aborting on the first failure. API keys appearing in any error message (URL `?key=`, `Authorization: Bearer …`, raw `sk-…` / `AIza…` strings) are redacted before being written to `errors.jsonl` or stderr.

### Meaningful-filename gate

`rename` skips files whose current name already looks human-meaningful and only renames stems that look auto-generated (`IMG_0034`, `PXL_20240316_080000123`, `Screenshot 2025-11-01 at 14.32.55`, `1696862563748`, `image (1)`, etc.). Classification runs in two tiers:

1. A pure local regex covers ~12 well-known camera, screenshot, messenger, and download families plus pure numeric stems and copy/sequence suffixes. A regex match means `Generic` and the stem is renamed without any model call.
2. Anything that does not match the regex is tagged `NeedsModel`. During `caption`, ClawGallery makes a separate text-only model call that sees only the filename stem (no image content) and asks whether the stem looks human-authored or auto-generated. The boolean is cached in `captions.jsonl` (`filename_meaningful: bool`) so future `rename` runs reuse the answer.

Pass `--force` to rename every captioned image regardless of name, or `--file <path>` to rename a single explicit target without consulting the gate.

`caption` only announces metadata writes (`captioned <path>`); the gate decision lives in `rename`'s output (`dry-run X -> Y`, `would skip ...`, `renamed X -> Y`). To audit the cached gate verdict for a specific image, read `filename_meaningful` from `captions.jsonl`.
