# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| `0.1.x` on the `main` branch | Yes |
| Older tags / forks | Best-effort only |

## Reporting a vulnerability

ClawGallery indexes local images, calls vision APIs, runs local embedding servers,
and can rename files on disk. Security reports are welcome and taken seriously.

**Please do not open a public GitHub issue for security bugs.**

Prefer one of:

1. **GitHub private vulnerability reporting** (preferred if enabled on the repo):
   open a private advisory at
   https://github.com/NomaDamas/ClawGallery/security/advisories/new
2. Email **vkehfdl1@gmail.com** with:
   - a short impact summary
   - reproduction steps or a proof of concept
   - affected commit / version if known
   - whether you have a suggested fix

We aim to acknowledge reports within **7 days** and to provide a status update
within **14 days**. Please give us a reasonable window to fix and publish before
any public disclosure.

## What we especially care about

- Secret leakage (API keys in logs, stderr, `errors.jsonl`, JSON output, crash reports)
- Path traversal or unexpected file overwrite during `rename` / `forget --delete`
- Local embedding server bind/auth issues (non-loopback exposure, SSRF-style abuse)
- Unsafe handling of untrusted image paths or model server responses
- Command injection via scripted / daemon paths

## Non-security issues

Bugs that do not have a security impact belong in
[GitHub Issues](https://github.com/NomaDamas/ClawGallery/issues).
