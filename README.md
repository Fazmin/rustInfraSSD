# rustInfraSSD

[![CI](https://github.com/Fazmin/rustInfraSSD/actions/workflows/rust.yml/badge.svg)](https://github.com/Fazmin/rustInfraSSD/actions/workflows/rust.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Run mixture-of-experts models far larger than your RAM by streaming their experts from SSD, with
the SSD layer written in Rust.**

A mixture-of-experts (MoE) model can have hundreds of billions of parameters, but each token uses only
a few of its experts. DeepSeek-V4.1-Flash, for example, has 763B parameters and activates 8–16B of
them per token, about 1–2% of its weights. The shared weights can therefore stay in memory, while the
experts stay on the SSD and are read only when a token needs them. That is how a model of hundreds of
gigabytes runs on a laptop.

rustInfraSSD is the layer that does this work. It finds where each expert lives on the SSD, reads it
with direct I/O, keeps frequently used experts cached, and runs them. This code parses model files
downloaded from the internet and coordinates I/O across many threads. Writing it in Rust means the
compiler enforces its memory and thread safety.

> [!NOTE]
> **Early stage.** The source index and the file layer are ported and tested. The reader, the cache
> and expert compute come next. rustInfraSSD cannot run a model on its own yet; see
> [Running a model today](#running-a-model-today).

## How it works

Say you have a large MoE model and a machine with far less memory than the model's size, but with a
fast SSD. This is the whole path from model files to generated tokens. The orange box is
rustInfraSSD.

```mermaid
flowchart LR
    model[("MoE model<br/>on your SSD")] --> setup["Index experts,<br/>load dense weights"]
    setup --> router["Router picks<br/>a few experts"]
    router --> rust["rustInfraSSD<br/>find, read, cache, run"]
    rust --> token(["Next token"])
    token -. "repeat" .-> router

    classDef rustbox fill:#b7410e,stroke:#7a2b09,color:#ffffff
    class rust rustbox
```

1. **Put the model on a fast SSD.** Download the model's GGUF files to an internal NVMe drive.
   DeepSeek-V4.1-Flash at Q8_0 is about 473 GiB. The drive's random-read speed decides how quickly
   uncached experts arrive.
2. **Index the experts.** The launcher reads the GGUF headers and writes an alias TSV. The TSV maps
   each expert of each MoE tensor to its byte range in the GGUF files, so no weights are copied.
   Alternatively, the experts can be rewritten into an expert pack with 4 KiB-aligned entries.
   rustInfraSSD loads either one into its source index.
3. **Measure the machine.** Probes measure:
   - the SSD's random-read speed at several request sizes and queue depths;
   - the CPU's matrix speed;
   - memory size and bandwidth;
   - GPU copy rates.

   The caches are sized from the memory that is free, so there is no memory setting to choose.
4. **Load the dense weights.** Attention, the router and the other shared weights are a small part of
   the model. They load into RAM or VRAM and run on the GPU through llama.cpp.
5. **Generate.** In each MoE layer, the router picks the experts the token needs. rustInfraSSD serves
   each expert from its RAM cache. On a miss, it looks the expert up in the source index and reads its
   bytes from the SSD with direct I/O, which bypasses the operating system's page cache. It then runs
   the expert on the CPU, or on the GPU tier on Linux with CUDA. The expert outputs are combined and
   the next layer runs.

Everything outside the orange box stays in C/C++: llama.cpp and ggml run the dense layers, and the
GPU kernels stay in CUDA. They call the Rust code through a C ABI.

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
with direct I/O:

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
[cpp/facade/ggml-ssd/store/index.h](cpp/facade/ggml-ssd/store/index.h) provides the original C++
`ssd::store::source_index` class on top of the same C ABI.

## Built on InferSSD

This project is inspired by [InferSSD](https://github.com/wici-ai/inferssd), which built the approach
above on top of llama.cpp and runs it on macOS with Metal and on Linux with CUDA. Its benchmarks show
DeepSeek-V4.1-Flash generating about 2.9 tokens per second on a MacBook Pro with 48 GB of memory
([details](https://github.com/wici-ai/inferssd/blob/v0.1.0/README.md#performance-benchmarks)).

I'm rewriting InferSSD's SSD layer in Rust, one component at a time. The aim is to match its speed and
produce exactly the same output, token for token. llama.cpp, ggml, the GPU kernels and the launcher
stay as they are. A pinned copy of InferSSD lives in `upstream/` (see [UPSTREAM.md](UPSTREAM.md)), and
every Rust component is tested against it.

### Running a model today

Until the Rust layer is wired into the engine, run models with InferSSD itself. Its
[quick start](https://github.com/wici-ai/inferssd/blob/main/docs/quickstart.md) lists the
requirements.

```sh
curl -fsSL https://github.com/wici-ai/inferssd/releases/latest/download/install.sh | sh
inferssd cli -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF   # the first run downloads about 470 GiB
```

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
