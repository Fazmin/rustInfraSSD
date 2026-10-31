# Changelog

All notable changes to rustInfraSSD are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) as described in [CONTRIBUTING.md](CONTRIBUTING.md#versions-and-releases). Each release names the InferSSD revision it was ported against. Release notes are taken from this file, so keep each entry on one line.

## [Unreleased]

Ported against InferSSD v0.1.0 (`b9cc72d6ec26`).

### Added

- Expert source store (`ssd-core::store`), ported from `ggml-ssd/store`: v1 expert packs, GGUF alias TSVs and the immutable source index. Every scenario of upstream's `c0-store` oracle and `test-source-index.cpp` runs as a Rust test.
- File layer for macOS and Linux (`ssd-platform::file`), ported from `ggml-ssd/platform/file`: direct reads (`F_NOCACHE`, `O_DIRECT`), file identity and positional reads.
- C ABI in `libinferssd_ssd` (`inferssd/store.h`) and a drop-in C++ facade for upstream's `ssd::store::source_index`.
- Version reporting: `inferssd/version.h`, `inferssd_ssd_version()` and `inferssd_ssd_build_info()`, which names the release, the build commit and the upstream revision.
- Quick start examples in Rust (`cargo run -p ssd-core --example quickstart`) and C (`cpp/examples/quickstart.c`).
- CI on macOS and Linux, and a release workflow that publishes tagged versions.
