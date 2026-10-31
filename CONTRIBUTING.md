# Contributing

Contributions are welcome. They are accepted under the repository's [Apache-2.0](LICENSE) license, as
section 5 of the license describes; no CLA or DCO is required. AI-assisted contributions are welcome when
you understand the change, have reviewed it and take responsibility for maintaining it. Support is best
effort.

Participation in this project is governed by the [Code of Conduct](.github/CODE_OF_CONDUCT.md). Report
security problems privately, as described in [SECURITY.md](SECURITY.md), not in public issues.

rustInfraSSD ports [InferSSD](https://github.com/wici-ai/inferssd)'s SSD layer to Rust one component at
a time, keeping upstream behaviour unchanged. Before you start porting a component, open an issue so
that two people don't port the same code. The [README](README.md#status) shows which components are done.

## Setting up

You need Rust 1.85 or later ([rustup](https://rustup.rs)), git, and a C and C++ compiler (the Xcode
Command Line Tools on macOS; gcc or clang on Linux). macOS and Linux are supported; Windows is not yet.

```sh
git clone https://github.com/Fazmin/rustInfraSSD.git
cd rustInfraSSD
cargo test --workspace
```

## Checks

A pull request must pass the same checks as CI:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo run -p ssd-core --example quickstart
./scripts/check-facade.sh            # C quick start and C++ facades against the static library
```

The store and file tests use real direct I/O. On Linux, run them on a filesystem that supports
`O_DIRECT`. tmpfs does not support it, so if `/tmp` is tmpfs, point `TMPDIR` at a directory on disk.

## Porting rules

- **Match upstream behaviour exactly**, including its error statuses and messages. A deliberate
  difference goes in the README under "Deliberate differences from upstream", with the reason.
- **Port the upstream tests with the code**, and name the upstream test file in the Rust test module's
  doc comment.
- **Name the upstream source** a module was derived from in its doc comment (`//! Port of ...`).
  [NOTICE](NOTICE) relies on this.
- **Keep each component behind the C ABI** (`cpp/include/inferssd/`), with a C++ facade under
  `cpp/facade/` that keeps upstream's class name, so unported C++ code still compiles.
- **Keep floating-point results bit-identical.** Never use `mul_add` or reorder sums.
- **Document every `unsafe` block** with a `// SAFETY:` comment. No panic may cross the C ABI; wrap
  entry points in `guarded` (`crates/ssd-capi/src/lib.rs`).
- **Leave `upstream/` unmodified.** It is a pristine snapshot. [UPSTREAM.md](UPSTREAM.md) describes how
  to move it to a newer upstream revision.

## Pull requests

- Keep each pull request to one concern. Keep upstream syncs separate from other changes.
- Use a title of the form `area: summary`, for example `store: reject entries past the end of a pack`.
  The summary is imperative, starts in lowercase, has no trailing period, and the whole title is at
  most 72 characters. Areas: `store`, `platform`, `capi`, `facade`, `upstream`, `build`, `ci`, `docs`,
  `repo`, plus the name of each newly ported component.
- Pull requests are squash-merged, so the title becomes the commit title on `main`.
- Do not add automated trailers. Use `Co-authored-by` only for human co-authors.
- Add a line under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md) for every change a user of the
  library or the C ABI would notice.
- Start new files with `SPDX-License-Identifier: Apache-2.0`.

## Versions and releases

Releases follow [Semantic Versioning](https://semver.org/). The public interface is the C ABI under
`cpp/include/inferssd/` and the public Rust API of the crates. While the version is `0.x`, a minor
release may break either; the CHANGELOG says when it does.

Every release, and every build, can be traced:

| What | Where |
| --- | --- |
| Release version | `version` in `Cargo.toml`, the `INFERSSD_SSD_VERSION*` macros in `cpp/include/inferssd/version.h`, the `vX.Y.Z` git tag |
| Upstream revision ported against | [UPSTREAM.md](UPSTREAM.md), `INFERSSD_SSD_UPSTREAM_*` in `version.h`, each CHANGELOG entry |
| The linked library's version | `inferssd_ssd_version()` |
| Version, build commit and upstream revision | `inferssd_ssd_build_info()`, in Rust `inferssd_ssd::version::build_info()` |

`cargo test` fails if `Cargo.toml`, `version.h` and `UPSTREAM.md` disagree. Builds from a source archive
without git metadata can set `INFERSSD_GIT_COMMIT` to record the commit; otherwise it reads `unknown`.

To publish a release (maintainers):

1. Set `version` under `[workspace.package]` in `Cargo.toml` and the version macros in
   `cpp/include/inferssd/version.h`, then run `cargo test` so that `Cargo.lock` is updated too.
2. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` and add a new, empty
   `## [Unreleased]` section above it.
3. Merge that change, then tag the merge commit and push the tag:

   ```sh
   git tag -a vX.Y.Z -m "rustInfraSSD X.Y.Z"
   git push origin vX.Y.Z
   ```

The [release workflow](.github/workflows/release.yml) checks the tag against `Cargo.toml` and
`CHANGELOG.md` (`./scripts/check-release.sh vX.Y.Z` runs the same check locally) and runs the tests on
macOS and Linux. It then publishes a GitHub release with the CHANGELOG entry as its notes and, for each
platform, an archive holding the static library, the C headers and the C++ facades.
