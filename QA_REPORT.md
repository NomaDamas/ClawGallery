# ClawGallery Manual QA Report

**Date:** 2026-05-05 (KST)
**Build:** `cargo build --release` from working tree (commit base `9666918` + Gemini provider work)
**Mode:** ultrawork + TDD + manual QA on a temp folder seeded from `~/Pictures`.

## Summary

| Result | Count |
|--------|-------|
| Functional behaviors verified end-to-end | 28 |
| Bugs surfaced | 2 |
| Bugs fixed (TDD: failing test → fix → green) | 2 |
| Pre-existing tests | 12 (all passing) |
| New tests added | 6 (4 unit, 2 integration) |
| Final test suite | **18 passing, 0 failing** |
| `make ci` (fmt + clippy + tests + build) | **PASS** |

## Test fixtures

Isolated state dir: `/tmp/clawgallery-qa-1777906831/state`
Image folder seeded from `~/Pictures` with 5 valid images of mixed formats:

| File | Format | Size |
|------|--------|------|
| `test-image.png` | PNG | 6,114 B |
| `wook_sign.png` | PNG | 12,032 B |
| `Eva-William.jpg` | JPG | 38,291 B |
| `dani_blonde_low_res.jpeg` | JPEG | 17,591 B |
| `baseball-die-profile.webp` | WEBP | 11,628 B |
| `notes.txt` (negative case) | non-image | 13 B |
| `extra/엘롯기.webp` | WEBP, UTF-8 name | 26,966 B |

## Feature matrix (28 cases)

All evidence written under `/tmp/clawgallery-qa-1777906831/evidence/`.

| # | Command / scenario | Expected | Actual | Result |
|---|--------------------|----------|--------|--------|
| 1 | `init` | state dir + 5 jsonl files + valid `config.json` | All present, `{"model":"gpt-4.1-mini","provider":"openai-compatible","filename_limit_bytes":240}` | PASS |
| 2 | `folder add <dir>` | folders.jsonl gets 1 record (active=true, recursive=true, UUID id, canonical path) | Recorded with macOS canonical `/private/tmp/...` | PASS |
| 3 | `folder add <dir>` (duplicate) | prints "already tracked", no new line | "folder already tracked", line count stays 1 | PASS |
| 4 | `folder list` | id, path, recursive=true (tab-separated) | Correct | PASS |
| 5 | `bootstrap` | "ingested 5 new image(s)", images.jsonl has 5 records with sha256/size/extension | All 5 records validated against on-disk sha256+size; `notes.txt` correctly excluded | PASS |
| 6 | `bootstrap` (re-run) | "ingested 0 new image(s)" (idempotent) | Confirmed, count stays 5 | PASS |
| 7 | `bootstrap --path <ext-dir>` | ingests UTF-8 named file without registering folder | `엘롯기.webp` ingested, total → 6 | PASS |
| 8 | `poll --once` | scans, prints timestamped count, exits | "ingested 0 new image(s)" with RFC3339 timestamp | PASS |
| 9 | `status` | reports config dir, provider, model, folders/images/captions counts | "folders: 1, images: 6, captions: 0" matches reality | PASS |
| 10 | `search eva` / `search baseball` / `search 엘롯기` | substring match on path (also UTF-8) | All three matched their target file | PASS |
| 11 | `search` (no args) | exit 1 with "provide at least one keyword" | Confirmed | PASS |
| 12 | `caption --dry-run` | lists "would caption …" for each image, no API call, captions.jsonl unchanged | All 6 images listed, captions.jsonl size=0, errors.jsonl size=0 | PASS |
| 13 | `caption --file <real images>` (3 OpenAI calls) | captions.jsonl gets entries with title+description, accurate captions | All 3 captions accurate (cat, doodle, book cover); recorded with model="gpt-4.1-mini", provider="openai-compatible" | PASS |
| 14 | `caption --file --provider gemini --model gemini-2.5-flash` (real Gemini) | API call routes to Gemini, captions.jsonl entry with provider="gemini", model="gemini-2.5-flash" | API call ROUTED to Gemini correctly (caption is clearly Gemini-quality, includes Korean/English mix). **`provider` field recorded as `"openai-compatible"` instead of `"gemini"`** | **FAIL → Bug #2** |
| 15 | `search cartoon` / `baseball` / `book cover` (multi-kw AND) / `--limit 1` | matches against captioned text, AND-of-keywords semantics, limit caps results | All four sub-cases pass | PASS |
| 16 | `rename --dry-run --style title` | dry-run lines, files unchanged, renames.jsonl applied=false | 4 records, `to` derives from caption title | PASS |
| 17 | `rename --dry-run --style caption` | filename derived from description | 4 records (filenames are very long but within 240-byte default limit) | PASS |
| 18 | `rename --dry-run --style date-title` (default) | filename starts with `YYYY-MM-DD` | 4 records, all prefixed `2026-05-04-…` | PASS |
| 19 | `rename --apply` (isolated sandbox) | file renamed on disk, renames.jsonl applied=true, images.jsonl gets new line with updated path + same id+sha256 | `test-image.png → my-renamed-cat.png`, all bookkeeping updated | PASS |
| 20 | `rename --apply --dry-run` | exit 1 with mutex error | "--apply and --dry-run cannot be used together" | PASS |
| 21 | `folder remove <id>` | deactivates folder, list becomes empty, folders.jsonl appended | Confirmed (2 lines: original + removal) | PASS |
| 22 | `folder remove /nonexistent` | exit 1 with "no active folder matched" | Confirmed | PASS |
| 23 | `skill path` | materializes skill file, prints absolute path | File at `<config>/skills/clawgallery/SKILL.md`, contents start with `name: clawgallery` | PASS |
| 24 | `skill print` | prints embedded SKILL.md exactly | `diff` against `skills/clawgallery/SKILL.md` shows zero differences (1673 bytes) | PASS |
| 25 | `--help` (root) | clap renders help with all 9 subcommands | Renders correctly | PASS |
| 26 | `caption --help` | renders flags | OK | PASS |
| 27 | `rename --help` | renders flags + `[default: date-title]` | OK | PASS |
| 28 | `caption --dry-run` (no creds at all) | should not require credentials (no network call) | **Errored: "missing visual model credentials"** | **FAIL → Bug #1** |

## Bugs surfaced

### Bug #1 — `caption --dry-run` requires API credentials

**Symptom**

```text
$ env -i HOME=$HOME PATH=$PATH CLAWGALLERY_CONFIG_DIR=… CODEX_HOME=/nonexistent-fake \
    clawgallery caption --dry-run
Error: missing visual model credentials: set OPENAI_API_KEY or login with Codex …
exit=1
```

**Root cause** — `cmd_caption` (src/main.rs) called `build_provider(&config, …)?` *before* the
`if args.dry_run` short-circuit. `build_provider` calls `Auth::discover()`, which fails when no
credentials exist, even though `--dry-run` does not perform any HTTP call.

**Impact** — Users wanting to preview which images would be captioned cannot use `--dry-run`
on a fresh machine until they set up credentials, defeating the purpose of dry-run.

**Fix** — Moved `build_provider(...)` past the dry-run early return. Dry-run now only resolves
the (lazy) effective provider name for record-keeping and never instantiates an HTTP client.

### Bug #2 — `--provider <name>` is not recorded in `captions.jsonl`

**Symptom** — A real Gemini call dispatched correctly (caption obviously came from Gemini), but
the record stored `"provider": "openai-compatible"`:

```json
{"…","model":"gemini-2.5-flash","provider":"openai-compatible",…}
```

**Root cause** — `CaptionRecord.provider` was built from `config.provider.clone()` rather than the
effective provider determined by CLI override → config fallback. (The `model` field was already
correct because it used `args.model.clone().unwrap_or_else(|| config.model.clone())`.)

**Impact** — Audit trails and downstream consumers of `captions.jsonl` cannot trust the
`provider` field when a CLI override is in use.

**Fix** — Extracted small pure helpers `resolve_provider` / `resolve_model` and used them for the
record-write. Added unit tests pinning the override semantics.

## TDD evidence

### Bug #1 RED (before fix)

```text
running 2 tests
test caption_dry_run_with_explicit_file_does_not_require_credentials ... FAILED
test caption_dry_run_does_not_require_credentials ... FAILED
…
caption --file --dry-run should succeed without credentials
stderr: Error: missing visual model credentials …
```

### Bug #1 GREEN (after fix)

```text
running 2 tests
test caption_dry_run_with_explicit_file_does_not_require_credentials ... ok
test caption_dry_run_does_not_require_credentials ... ok

test result: ok. 2 passed; 0 failed
```

### Bug #2 unit tests

```text
test tests::cli_provider_overrides_config_provider ... ok
test tests::config_provider_used_when_cli_absent ... ok
test tests::cli_model_overrides_config_model ... ok
test tests::config_model_used_when_cli_absent ... ok
```

### Real-world verification after fixes

```text
=== POST-FIX T1: dry-run WITHOUT credentials ===
initialized /tmp/clawgallery-qa-fix-1777907189/state
ingested 1 new image(s)
would caption /private/tmp/clawgallery-qa-fix-1777907189/imgs/test-image.png
exit=0

=== POST-FIX T2: --provider gemini records as gemini ===
captioned … -> white-cat-semester-end-thoughts
{"…","model":"gemini-2.5-flash","provider":"gemini",…}
OK: BOTH provider and model correctly reflect CLI override
```

## Final gate

```text
$ make ci
cargo fmt --all -- --check         ✓
cargo clippy --all-targets --all-features -- -D warnings   ✓
cargo test --all-features          ✓ (12 unit + 6 integration = 18 passed)
cargo build --all-features         ✓
```

## Observations (non-bugs)

- `folder add` runs `cmd_init` internally so it re-prints `"initialized …"` even on duplicate adds. Mildly noisy but matches existing semantics.
- `--style caption` produces very long filenames preserving accented Latin (`À`, `é`), apostrophes, parentheses, and commas because `sanitize_filename` only ASCII-folds for `[a-zA-Z0-9]` plus Hangul. The result is filesystem-legal on macOS/Linux but visually messy. This is a design choice, not a bug; `--style title` and `--style date-title` (the default) produce clean kebab-case names.
- `caption --file <new path> --dry-run` previously also tripped the credentials gate (same root cause as Bug #1) — this is now fixed and covered by `caption_dry_run_with_explicit_file_does_not_require_credentials`.
