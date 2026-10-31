// SPDX-License-Identifier: Apache-2.0
// The Rust store seen from C++ through the drop-in facade, the way
// src/ssd/options.cpp opens it and the executors borrow its source_view.
// Mirrors upstream engine/tests/ssd/test-source-index.cpp without linking any
// upstream implementation. Run by scripts/check-facade.sh.
#include "ggml-ssd/store/index.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

static bool is_direct(int fd) {
    const int flags = fcntl(fd, F_GETFL);
#if defined(__APPLE__)
    return flags >= 0 && (flags & 0x40000); // XNU FNOCACHE, set by F_NOCACHE
#else
    return flags >= 0 && (flags & O_DIRECT);
#endif
}

int main() {
    try {
        char pattern[] = "/tmp/rustinfrassd-facade-XXXXXX";
        require(mkdtemp(pattern) != nullptr, "temporary directory failed");
        const fs::path directory = pattern;
        const std::string header = "tensor\texpert_idx\tnbytes\tsource_path\toffset\n";
        const char * name = "blk.0.ffn_up_exps.weight";
        std::vector<std::string> aliases;
        for (int seed = 1; seed <= 2; ++seed) {
            const auto data = directory / (std::to_string(seed) + ".bin");
            const auto alias = directory / (std::to_string(seed) + ".tsv");
            std::ofstream stream(data, std::ios::binary);
            for (size_t i = 0; i < 16384; ++i) { stream.put(char((i + seed) % 251)); }
            stream.close();
            std::ofstream(alias) << header << name << "\t3\t5000\t" << data.string() << "\t4097\n";
            aliases.push_back(alias.string());
        }

        auto first = ssd::store::source_index::open({aliases[0]});
        auto second = ssd::store::source_index::open({aliases[1]});
        require(bool(first) && bool(second), "direct source creation failed");
        const ssd::store::source_view a = first.value->view(), b = second.value->view();
        int fd_a = -1, fd_b = -1;
        uint64_t offset = 0;
        size_t bytes = 0;
        require(a.resolve(a.userdata, name, 3, 5000, 4, &fd_a, &offset, &bytes), "first source lookup failed");
        require(b.resolve(b.userdata, name, 3, 5000, 4, &fd_b, &offset, &bytes), "second source lookup failed");
        require(fd_a != fd_b && offset == 4097 && bytes == 5000, "contexts shared a descriptor or changed the range");
        require(!b.resolve(b.userdata, name, 3, 5000, 3, &fd_b, &offset, &bytes), "expert-count bound ignored");
        require(!b.resolve(b.userdata, name, 3, 5001, 4, &fd_b, &offset, &bytes), "tensor byte bound ignored");
        require(!b.resolve(b.userdata, nullptr, 3, 5000, 4, &fd_b, &offset, &bytes), "null name accepted");
        require(b.dense_blocks(b.userdata, name, 5000, nullptr) == 0, "expert tensor reported as dense");

        first.value.reset();
        errno = 0;
        require(fcntl(fd_a, F_GETFD) == -1 && errno == EBADF, "source descriptor survived its owner");
        require(is_direct(fd_b), "second context lost direct I/O");
        void * buffer = nullptr;
        require(posix_memalign(&buffer, 4096, 8192) == 0, "aligned buffer failed");
        const ssize_t got = pread(fd_b, buffer, 8192, 4096);
        const auto * payload = static_cast<unsigned char *>(buffer);
        bool same = got == 8192;
        for (size_t i = 0; same && i < 5000; ++i) { same = payload[i + 1] == (4097 + i + 2) % 251; }
        std::free(buffer);
        require(same, "direct payload changed");
        second.value.reset();

        const auto bad = directory / "bad.tsv";
        std::ofstream(bad) << header << name << "\t3\t5000\t" << (directory / "1.bin").string() << "\t16000\n";
        const auto out_of_file = ssd::store::source_index::open({bad.string()});
        require(!out_of_file && out_of_file.code == ssd::status::invalid &&
                    out_of_file.error == "expert index range exceeds its source file",
                "out-of-file source published");
        const auto partial = ssd::store::source_index::open({aliases[0], (directory / "missing.tsv").string()});
        require(!partial && partial.code == ssd::status::io &&
                    partial.error == "cannot open expert index: " + (directory / "missing.tsv").string(),
                "partial source set published");
        const auto none = ssd::store::source_index::open({});
        require(!none && none.code == ssd::status::invalid, "empty source set published");

        fs::remove_all(directory);
        std::puts("SSD_CONTEXT_SOURCE_DIRECT_IO_PASS (Rust store via C++ facade)");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
