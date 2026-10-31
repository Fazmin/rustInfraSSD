# rustInfraSSD

[![CI](https://github.com/Fazmin/rustInfraSSD/actions/workflows/rust.yml/badge.svg)](https://github.com/Fazmin/rustInfraSSD/actions/workflows/rust.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

rustInfraSSD is inspired by [InferSSD](https://github.com/wici-ai/inferssd), which runs
mixture-of-experts models larger than RAM or VRAM by streaming expert weights from SSD, on top of
llama.cpp. I'm rewriting its SSD layer in Rust. That layer reads, caches and schedules the expert
weights. It parses model files downloaded from the internet and coordinates I/O across many
threads, so it is where memory and thread safety matter most.

In Rust, the compiler enforces those guarantees. The aim is to match InferSSD's speed and produce
exactly the same output. llama.cpp, ggml and the GPU kernels stay in C/C++/CUDA and call the Rust
code through a C ABI.

> [!NOTE]
> **Early stage.** The expert source store and the file layer are ported and tested. The engine does
> not use them yet, so rustInfraSSD cannot run a model by itself; use
> [InferSSD](https://github.com/wici-ai/inferssd) for that. See [Status](#status).

## Quick start

You need Rust 1.85 or later ([rustup](https://rustup.rs)), git, and a C and C++ compiler (the Xcode
Command Line Tools on macOS; gcc or clang on Linux). macOS and Linux are supported.

```sh
git clone https://github.com/Fazmin/rustInfraSSD.git
cd rustInfraSSD
cargo test --workspace                       # build and run the tests
cargo run -p ssd-core --example quickstart   # index, look up and read one expert
```

The example writes a small expert index to a temporary directory, looks up one expert and reads it
with direct I/O, which bypasses the page cache:

```text
indexed 4 entries from 1 source file
blk.3.ffn_up_exps.weight expert 2: 4096 bytes at offset 8292, first byte 0x02
```

Diagnostic lines starting with `[moe_stream_batch]` go to stderr. Linux tmpfs has no direct I/O, so if
`/tmp` is tmpfs, pass a directory on disk: `cargo run -p ssd-core --example quickstart -- ~/scratch`.

### Index files

The store maps each expert of a tensor to a byte range in a file. It reads two kinds of index, which
can be mixed, and recognises each by its first bytes:

- **Expert packs (v1):** binary files starting with `GGMLMOEPACKv1` that hold an index followed by
  the expert data.
- **Alias TSVs:** tab-separated text that points into existing GGUF files, so no weights are copied:

  ```text
  tensor	expert_idx	nbytes	source_path	offset
  blk.3.ffn_up_exps.weight	0	4096	/models/model-00001.gguf	1048672
  blk.3.ffn_up_exps.weight	1	4096	/models/model-00001.gguf	1052768
  ```

  `expert` is accepted for `expert_idx`, and `model_offset` or `offset_bytes` for `offset`.

### Use it from Rust

```toml
[dependencies]
ssd-core = { git = "https://github.com/Fazmin/rustInfraSSD", branch = "main" } # or tag = "vX.Y.Z"
ssd-platform = { git = "https://github.com/Fazmin/rustInfraSSD", branch = "main" }
```

```rust
use ssd_core::store::SourceIndex;

// Packs and alias TSVs can be mixed in one index.
let index = SourceIndex::open(&["experts.tsv"])?;

// Tensor name, expert number, expected size in bytes, number of experts in the tensor.
let (fd, offset, len) = index
    .resolve(b"blk.3.ffn_up_exps.weight", 2, 4096, 4)
    .ok_or("expert not indexed")?;

// `fd` is a direct-I/O descriptor borrowed from `index`. It stays open until
// `index` is dropped; never close it yourself.
let bytes = read_direct(fd, offset, len)?;
```

On Linux, direct reads must cover whole 4 KiB blocks. `read_direct` in
[the quick start example](crates/ssd-core/examples/quickstart.rs) handles that; copy it.

### Use it from C or C++

```sh
cargo build -p ssd-capi --release            # builds target/release/libinferssd_ssd.a
```

```c
#include "inferssd/store.h"

const char *paths[] = { "experts.tsv" };
char error[512];
inferssd_store_index *index = NULL;
if (inferssd_store_open(paths, 1, &index, error, sizeof error) != INFERSSD_STATUS_OK) {
    fprintf(stderr, "cannot open index: %s\n", error);
    return 1;
}

int fd;            /* borrowed from the index: never close it */
uint64_t offset;
size_t size;
if (inferssd_store_resolve(index, "blk.3.ffn_up_exps.weight", 2, 4096, 4, &fd, &offset, &size)) {
    /* read `size` bytes at `offset` from `fd` */
}
inferssd_store_close(index);   /* also closes the descriptors */
```

Compile against `cpp/include` and link the static library:

```sh
cc -I cpp/include app.c target/release/libinferssd_ssd.a -lpthread -ldl -lm -o app
```

[cpp/examples/quickstart.c](cpp/examples/quickstart.c) is a complete program, including the aligned
read. `./scripts/check-facade.sh` builds it and runs it on a scratch index. For C++,
[cpp/facade/ggml-ssd/store/index.h](cpp/facade/ggml-ssd/store/index.h) provides upstream's
`ssd::store::source_index` class on top of the same C ABI.

## Versions

Releases are tagged `vX.Y.Z` following [Semantic Versioning](https://semver.org/), and each one is
described in [CHANGELOG.md](CHANGELOG.md). Every build reports what it is:

```c
#include "inferssd/version.h"

printf("%s\n", inferssd_ssd_build_info());
// rustInfraSSD 0.1.0 (git 1a2b3c4d5e6f, InferSSD v0.1.0 b9cc72d6ec26)
```

That line names the release, the commit the library was built from, and the InferSSD revision the
port tracks. Log it at startup to tie every report to an exact build. To catch headers that do not
match the linked library, compare `inferssd_ssd_version()` with the `INFERSSD_SSD_VERSION` macro. In
Rust, the same line is `inferssd_ssd::version::build_info()`, from the `ssd-capi` package.
[CONTRIBUTING.md](CONTRIBUTING.md#versions-and-releases) explains where each version is recorded and
how releases are made.

## Status

Phase 1 of the port has started; the store is the first component through.

| Component | State |
| --- | --- |
| Expert source store (`ggml-ssd/store`) | Ported. Rust tests cover every scenario of upstream's `c0-store` oracle and `test-source-index.cpp`. The C++ facade passes a smoke test. |
| File platform (`ggml-ssd/platform/file`) | Ported for macOS and Linux. Windows pending. |
| Engine build with the Rust store (`INFERSSD_RUST_STORE=ON`) | Not wired yet. Needs a build of the upstream engine. |
| Everything else | Not started |

Developed on macOS (Apple Silicon). CI runs the same checks on Linux (Ubuntu 24.04). No CUDA code has
been ported yet.

## Layout

| Path | Contents |
| --- | --- |
| `upstream/` | InferSSD v0.1.0, unmodified (see [UPSTREAM.md](UPSTREAM.md)) |
| `crates/ssd-platform` | Per-OS primitives. Port of `ggml-ssd/platform/` |
| `crates/ssd-core` | Result type and expert source store. Port of `ggml-ssd/settings/result.h` and `ggml-ssd/store/` |
| `crates/ssd-capi` | The C ABI and version reporting, built as `libinferssd_ssd.a` |
| `cpp/include/inferssd/` | C headers for that ABI |
| `cpp/facade/` | Drop-in C++ headers with upstream's class names over the C ABI |
| `cpp/examples/`, `cpp/tests/` | C quick start and C++ smoke tests of the facades |
| `scripts/` | `check-facade.sh` (C and C++ checks), `check-release.sh` (release tag check) |

## Development

```sh
cargo test --workspace                                   # Rust behaviour tests
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
./scripts/check-facade.sh                                # C example and C++ facade against the static library
```

[CONTRIBUTING.md](CONTRIBUTING.md) has the porting rules, pull request conventions and release steps.
Participation is governed by the [Code of Conduct](.github/CODE_OF_CONDUCT.md). Report security
problems privately as described in [SECURITY.md](SECURITY.md).

## Deliberate differences from upstream

The port keeps upstream behaviour byte for byte, including permissive legacy parsing such as
`"4097tail"` reading as 4097 and 127-byte tensor-name truncation, except in these places:

- **Alias TSV read errors.** A read error in the middle of an alias TSV fails the load. Upstream's
  `fgets` loop treats it as end of file and keeps the rows read so far.
- **Pack entry count.** A pack's entry count is not trusted for an up-front allocation; entries are
  added as they are read. Results are the same.
- **Identical keys.** Entries with identical keys from the same source are ordered stably, so the
  later record wins a replacement. Upstream's `std::sort` leaves that order unspecified.
- **Memory exhaustion.** Rust allocation failure aborts the process. Upstream reports
  `status::resource "oom"`; the C++ facade still does so for its own allocations.
- **Integer width.** Integer parsing follows LP64 (`long` is 64-bit), as upstream does on macOS and
  Linux.

## License

Apache-2.0, as upstream. Copyright 2026 Fazmin Nizam for the code this repository adds; InferSSD and
the projects it bundles keep their own copyrights. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
