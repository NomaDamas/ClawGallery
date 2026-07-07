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
