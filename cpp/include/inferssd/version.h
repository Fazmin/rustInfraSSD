// SPDX-License-Identifier: Apache-2.0
// Version of the Rust SSD layer (crates/ssd-capi/src/version.rs).
//
// The macros describe the headers being compiled; the functions describe the
// library actually linked. Different values mean stale headers or a stale
// library:
//
//     if (strcmp(inferssd_ssd_version(), INFERSSD_SSD_VERSION) != 0) { ... }
//
// `cargo test` checks these macros against Cargo.toml and UPSTREAM.md.
#pragma once

#define INFERSSD_SSD_VERSION_MAJOR 0
#define INFERSSD_SSD_VERSION_MINOR 1
#define INFERSSD_SSD_VERSION_PATCH 0
#define INFERSSD_SSD_VERSION "0.1.0"

// The InferSSD release and commit the port tracks.
#define INFERSSD_SSD_UPSTREAM_VERSION "v0.1.0"
#define INFERSSD_SSD_UPSTREAM_COMMIT "b9cc72d6ec26944e628dceb0e8bbb4e610900052"

#ifdef __cplusplus
extern "C" {
#endif

// The linked library's version, such as "0.1.0". Static storage.
const char * inferssd_ssd_version(void);

// One line for logs naming the library version, the commit it was built from
// ("unknown" outside a git checkout) and the upstream revision, such as
// "rustInfraSSD 0.1.0 (git 1a2b3c4d5e6f, InferSSD v0.1.0 b9cc72d6ec26)".
// Static storage.
const char * inferssd_ssd_build_info(void);

#ifdef __cplusplus
}
#endif
