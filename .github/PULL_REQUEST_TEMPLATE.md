## Summary

<!-- What changed, and why. -->

## Checklist

- [ ] `make ci` passes locally
- [ ] Tests cover the new or fixed behavior
- [ ] README / skill usage updated if the CLI surface changed
- [ ] No secrets or personal paths in the diff
- [ ] Security-sensitive paths (rename, forget, embedding bind, error logging) still redact keys and refuse unsafe overwrites

## Test plan

```bash
make ci
# optional: targeted scenarios
```
