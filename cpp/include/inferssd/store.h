// SPDX-License-Identifier: Apache-2.0
// C ABI of the Rust expert source store (crates/ssd-capi/src/store.rs).
//
// An index owns its source descriptors. A descriptor returned by
// inferssd_store_resolve is borrowed: it stays valid until inferssd_store_close
// and the caller must never close it. Handles are immutable after open, so
// resolve and dense_blocks may run concurrently.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Status codes, equal to the C++ ssd::status values.
enum {
    INFERSSD_STATUS_OK       = 0,
    INFERSSD_STATUS_INVALID  = 1,
    INFERSSD_STATUS_IO       = 2,
    INFERSSD_STATUS_RESOURCE = 3,
};

typedef struct InferssdStoreIndex inferssd_store_index;

// Opens an index over v1 expert packs and GGUF alias TSVs, told apart by their
// leading bytes. Returns INFERSSD_STATUS_OK and stores the handle in *out, or
// returns another status and writes a NUL-terminated reason into error.
int inferssd_store_open(const char * const * paths, size_t count, inferssd_store_index ** out,
                        char * error, size_t error_size);

// Destroys an index and closes its descriptors. Null is ignored.
void inferssd_store_close(inferssd_store_index * index);

// The direct descriptor, file offset and logical size of expert `expert` of
// `experts` in tensor `name`, whose bytes the caller expects to be `bytes`.
bool inferssd_store_resolve(const inferssd_store_index * index, const char * name, int expert, size_t bytes,
                            int64_t experts, int * fd, uint64_t * offset, size_t * logical_bytes);

// The number of equal blocks 0..N-1 that make up `bytes` of dense tensor
// `name` and nothing else, or 0. Stores the block size when block_bytes is set.
size_t inferssd_store_dense_blocks(const inferssd_store_index * index, const char * name, size_t bytes,
                                   size_t * block_bytes);

#ifdef __cplusplus
}
#endif
