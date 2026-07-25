# Repository Guidelines

## crates.io Releases

- Publish crates only through `.github/workflows/publish.yml`; do not run `cargo publish` manually.
- Before a release, bump `clawgallery` and `clawgallery-vdr` to the same unpublished version, update the root `clawgallery-vdr` dependency requirement, and refresh `Cargo.lock`.
- Run the full release checks and package dry-runs before committing the version bump.
- Commit and push the release preparation to `main` before creating the release.
- Create and publish a GitHub Release tagged `vX.Y.Z` from the prepared `main` commit. The release event triggers the crates.io workflow and authenticates through GitHub OIDC trusted publishing.
- The workflow must publish `clawgallery-vdr` first, wait for that exact manifest version to appear in the registry, and then publish `clawgallery`.
- Confirm both packages are visible on crates.io after the workflow succeeds.
