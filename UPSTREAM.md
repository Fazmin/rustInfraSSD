# Upstream tracking

| | |
| --- | --- |
| Upstream | https://github.com/wici-ai/inferssd |
| Pinned revision | `b9cc72d6ec26944e628dceb0e8bbb4e610900052` (tag `v0.1.0`, 2026-10-31) |
| Imported as | `git archive v0.1.0` extracted to `upstream/`, unmodified |
| llama.cpp base | build b8961, revision `f42e29fd` (see `upstream/NOTICE`) |

`upstream/` is a pristine snapshot. Fork changes live outside it, in `crates/`
(Rust) and `cpp/` (C ABI headers and C++ facades). A Rust component replaces its
C++ original only when the build selects it (`INFERSSD_RUST_<COMPONENT>=ON`), so
the C++ original stays buildable as the A/B oracle until the cut-over phase.

## Port status

| Upstream component | Rust | Status |
| --- | --- | --- |
| `ggml-ssd/platform/file.*` | `ssd-platform::file` | POSIX (macOS, Linux) ported; Windows pending |
| `ggml-ssd/store/*` | `ssd-core::store` | Ported, behaviour tests from `test-source-index.cpp` and the `c0-store` oracle |
| `ggml-ssd/store/index.h` C++ API | `cpp/facade/ggml-ssd/store/index.h` over `inferssd/store.h` | Facade passes `scripts/check-facade.sh`; upstream `ssd_source_index` ctest against it pending |
| everything else | — | Not started |

## Sync procedure

1. Fetch upstream and list changes to the owned SSD layer since the pinned revision:
   `git diff <pinned>..<new> -- engine/ggml/src/ggml-ssd engine/ggml/src/ggml-cpu/ssd engine/ggml/src/ggml-cuda/ssd engine/src/ssd engine/include/llama-ssd.h engine/tests/ssd`
2. For each change to a ported component, port it to the Rust crate together with
   the upstream test change, and run that component's gate (README, "Verification").
3. Replace `upstream/` with the new snapshot (`git archive <tag> | tar -x -C upstream`),
   update the pinned revision above, and record which upstream PRs were ported.
4. Update the same pin in `UPSTREAM_VERSION` and `UPSTREAM_COMMIT` (`crates/ssd-capi/src/version.rs`),
   in the `INFERSSD_SSD_UPSTREAM_*` macros (`cpp/include/inferssd/version.h`) and in the
   "Ported against" line of `CHANGELOG.md`. `cargo test` fails until the first three agree.
