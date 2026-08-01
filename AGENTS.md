# Repository agent instructions

## Releases for remote code changes

Every completed code change that an agent commits and pushes to the remote must
also produce a GitHub release. Do not leave a pushed code change unreleased.

A release is required for changes to executable or runtime behavior, including:

- Rust source under `src/`;
- runtime code in `.agents/skills/**/scripts/`;
- dependency or build changes in `Cargo.toml` or `Cargo.lock` that affect the
  shipped binary; and
- release or runtime workflow changes that affect distributed artifacts.

Documentation-only, asset-only, formatting-only, and test-only changes do not
require a release unless they also change shipped runtime behavior.

When a release is required:

1. Determine the latest `vMAJOR.MINOR.PATCH` tag and choose the next semantic
   version. Use a patch bump by default; use a minor or major bump when the
   change warrants it.
2. Update the package version in `Cargo.toml` and refresh `Cargo.lock` before
   committing.
3. Run the relevant checks, including at minimum:

   ```sh
   cargo fmt --all -- --check
   cargo clippy --all-targets -- -D warnings
   cargo test --all-targets
   ```

4. Commit the completed change and version bump together, then push the commit
   to the remote.
5. Create the matching tag at that exact commit and push it:

   ```sh
   git tag vMAJOR.MINOR.PATCH
   git push origin vMAJOR.MINOR.PATCH
   ```

6. The tag triggers `.github/workflows/release.yml`. Verify that the workflow
   succeeds and that the GitHub release contains all platform archives plus
   `SHA256SUMS`.

Do not push an interim code commit that is not ready to release. If the task
authorizes a commit but not a push, keep it local and do not create a release.
If a code change must be pushed but the version tag or release cannot be
created, stop before pushing and ask the user how to proceed.
