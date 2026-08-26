# ClawGallery CLI Reference

- Install: `cargo install --path .`
- Initialize state: `clawgallery init`
- Register folders: `clawgallery folder add <path>`
- Bootstrap image records: `clawgallery bootstrap` or `clawgallery bootstrap --prune`
- Check state: `clawgallery status`
- Poll: `clawgallery poll --once` or `clawgallery poll --interval 30`
- Preview caption work: `clawgallery caption --dry-run`
- Caption missing images: `clawgallery caption --missing`
- Search captions/paths plus VDR when indexed: `clawgallery search "<query>" --json --limit 5`
- Search captions/paths only: `clawgallery search --mode keyword "<query>" --json --limit 5`
- Search VDR embeddings only: `clawgallery search --mode embedding "<query>" --json --limit 5`
- Sync VDR embeddings: `clawgallery vdr sync`
- Sync V-SPLADE lexical embeddings: `CLAWGALLERY_PYTHON=/path/to/splade-mlx/.venv/bin/python clawgallery vdr sync --backend vsplade`
- Check VDR state: `clawgallery vdr status --json`
- Full enrich + retrieve: `clawgallery caption --missing && clawgallery vdr sync --prune && clawgallery search "<query>" --json`
- Report exact duplicates: `clawgallery dedup --exact --json`
- Report visual duplicates: `clawgallery dedup --similar --threshold 0.95 --json`
- Remove a reviewed duplicate: `clawgallery forget --file <path> --delete`
- Rename preview: `clawgallery rename --dry-run`
- Rename apply: `clawgallery rename --apply`

Agent default for local screenshot/photo search:

```bash
clawgallery init
clawgallery folder add ~/Pictures
test -d ~/Pictures/screenshots && clawgallery folder add ~/Pictures/screenshots
test -d ~/Picutres/screenshots && clawgallery folder add ~/Picutres/screenshots
clawgallery bootstrap
clawgallery search "<observed visual query>" --json --limit 5
```

## V-SPLADE runtime notes

V-SPLADE image indexing and lexical query embedding use the packaged managed
Python server. It is started and stopped automatically for each command; no
manual daemon step is required. Set the Python interpreter in the shell
environment before both indexing and searching:

```bash
export CLAWGALLERY_PYTHON=/path/to/splade-mlx/.venv/bin/python
clawgallery vdr sync --backend vsplade
clawgallery search --mode lexical "invoice total" --json
```

Do not set `CLAWGALLERY_PYTHON` only before `vdr sync`: a later `search`
invocation is a new process and otherwise falls back to `python3` (or
`python` on Windows). The selected interpreter must import `splade_mlx`.
Install the runtime in that environment, or use `--python <path>` for sync
and keep `CLAWGALLERY_PYTHON` set for search:

```bash
python3 -m pip install git+https://github.com/NomaDamas/SPLADE-mlx.git
export CLAWGALLERY_PYTHON=/path/to/splade-mlx/.venv/bin/python
```

An activated `VIRTUAL_ENV` is detected automatically using `bin/python` on
macOS/Linux and `Scripts/python.exe` on Windows. If no suitable environment
is configured, ClawGallery reports the selected interpreter and a remediation
command before spawning the server. It does not install Python packages
automatically. Use `--embedding-url` to connect to a compatible long-running
server or `--no-auto-start` when deliberately managing the endpoint yourself.
