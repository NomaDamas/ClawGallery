---
name: clawgallery
description: Use ClawGallery to register screenshot/image folders, caption screenshots, safely rename files, and search local visual metadata from JSONL state.
---

# ClawGallery Skill

Use this skill when an agent needs to find, inspect, caption, or organize local screenshots/images through the `clawgallery` CLI.

## Safety defaults

- `clawgallery rename` is dry-run unless `--apply` is explicitly passed.
- State is JSONL under `~/.config/clawgallery` unless `CLAWGALLERY_CONFIG_DIR` is set.
- The CLI does not require a database or embeddings.
- Model captioning requires `OPENAI_API_KEY` or best-effort Codex auth in `$CODEX_HOME/auth.json` / `~/.codex/auth.json`.

## Common workflows

Initialize and register a screenshots folder:

```bash
clawgallery init
clawgallery folder add ~/Desktop
clawgallery bootstrap
```

Search known screenshots:

```bash
clawgallery search login error
clawgallery search "github actions" --limit 5
```

Caption uncaptured images:

```bash
clawgallery caption --missing
```

Preview safe rename targets:

```bash
clawgallery rename --dry-run
```

Apply rename only after reviewing dry-run output:

```bash
clawgallery rename --apply
```

Poll once for newly added images:

```bash
clawgallery poll --once
```

## Agent guidance

1. Prefer `search` before asking the user to locate screenshots manually.
2. Use `caption --dry-run` to see pending work when model credentials may be absent.
3. Never pass `--apply` to rename unless the user requested actual file changes or an approved workflow requires it.
4. If a command needs isolated state, set `CLAWGALLERY_CONFIG_DIR` to a task-specific temporary directory.
