---
name: Bug report
about: Something went wrong in ClawGallery
title: "[bug] "
labels: bug
---

## What happened

<!-- A short summary of the unexpected behavior -->

## Expected behavior

## Steps to reproduce

```bash
# Prefer CLAWGALLERY_CONFIG_DIR pointing at a temp dir
export CLAWGALLERY_CONFIG_DIR="$(mktemp -d)"
clawgallery …
```

## Environment

- OS:
- ClawGallery version / commit:
- Install method (`cargo install --path .`, crates.io, …):
- Provider / model (if relevant):
- VDR backend (MLX / ColQwen2 / Jina / none):

## Logs / error output

```text
# redacted stderr / errors.jsonl excerpts
```

**Please redact API keys before pasting logs.** ClawGallery tries to mask them, but double-check.
