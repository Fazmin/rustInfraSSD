// SPDX-License-Identifier: Apache-2.0
// Rust-backed drop-in for upstream engine/ggml/src/ggml-ssd/store/index.h.
//
// Put cpp/facade before engine/ggml/src on the include path and link the
// inferssd_ssd static library; callers such as src/ssd/options.cpp and the
// executors that borrow source_view compile unchanged.
#pragma once

#include "ggml-ssd/settings/result.h"
#include "ggml-ssd/store/source.h"
#include "inferssd/store.h"

#include <memory>
#include <new>
#include <string>
#include <vector>

namespace ssd { namespace store {
class source_index {
public:
    [[nodiscard]] static result<std::unique_ptr<source_index>> open(const std::vector<std::string> & paths) noexcept {
        try {
            std::vector<const char *> raw;
            raw.reserve(paths.size());
            for (const auto & path : paths) { raw.push_back(path.c_str()); }
            char error[4096] = {};
            inferssd_store_index * index = nullptr;
            const int code = inferssd_store_open(raw.data(), raw.size(), &index, error, sizeof(error));
            if (code != INFERSSD_STATUS_OK) { return {nullptr, static_cast<status>(code), error}; }
            return {std::unique_ptr<source_index>(new source_index(index)), status::ok, {}};
        } catch (const std::bad_alloc &) {
            return {nullptr, status::resource, "oom"};
        } catch (...) {
            return {nullptr, status::invalid, "source invalid"};
        }
    }
    ~source_index() { inferssd_store_close(index_); }
    source_index(const source_index &) = delete;
    source_index & operator=(const source_index &) = delete;

    // Borrow the immutable index and its descriptors until all readers have joined.
    source_view view() noexcept {
        return {index_,
            [](void * opaque, const char * name, int expert, size_t bytes, int64_t experts,
               int * fd, uint64_t * offset, size_t * logical_bytes) noexcept {
                return inferssd_store_resolve(static_cast<const inferssd_store_index *>(opaque), name, expert, bytes,
                                              experts, fd, offset, logical_bytes);
            },
            [](void * opaque, const char * name, size_t bytes, size_t * block_bytes) noexcept -> size_t {
                return inferssd_store_dense_blocks(static_cast<const inferssd_store_index *>(opaque), name, bytes,
                                                   block_bytes);
            }};
    }

private:
    explicit source_index(inferssd_store_index * index) noexcept : index_(index) {}
    inferssd_store_index * index_;
};
} }
