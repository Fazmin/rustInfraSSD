// SPDX-License-Identifier: Apache-2.0
// Quick start for the C ABI: open an index, resolve one expert and read it.
//
//     quickstart INDEX TENSOR EXPERT EXPERTS BYTES
//
// INDEX is a v1 expert pack or an alias TSV. Built and run against a scratch
// index by scripts/check-facade.sh.
#define _POSIX_C_SOURCE 200809L

#include "inferssd/store.h"
#include "inferssd/version.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// Covers the Linux O_DIRECT rule for buffer address, offset and length.
#define DIRECT_ALIGN 4096u

int main(int argc, char ** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s INDEX TENSOR EXPERT EXPERTS BYTES\n", argv[0]);
        return 2;
    }
    printf("%s\n", inferssd_ssd_build_info());
    if (strcmp(inferssd_ssd_version(), INFERSSD_SSD_VERSION) != 0) {
        fprintf(stderr, "headers are %s but the library is %s\n", INFERSSD_SSD_VERSION, inferssd_ssd_version());
        return 1;
    }

    const char * paths[] = { argv[1] };
    char error[512];
    inferssd_store_index * index = NULL;
    if (inferssd_store_open(paths, 1, &index, error, sizeof error) != INFERSSD_STATUS_OK) {
        fprintf(stderr, "cannot open %s: %s\n", argv[1], error);
        return 1;
    }

    const char * tensor = argv[2];
    const int expert = atoi(argv[3]);
    const int64_t experts = strtoll(argv[4], NULL, 10);
    const size_t bytes = (size_t) strtoull(argv[5], NULL, 10);
    int fd = -1; // borrowed: the index owns it, so it is never closed here
    uint64_t offset = 0;
    size_t size = 0;
    if (!inferssd_store_resolve(index, tensor, expert, bytes, experts, &fd, &offset, &size)) {
        fprintf(stderr, "%s expert %d of %" PRId64 " (%zu bytes) is not indexed\n", tensor, expert, experts, bytes);
        inferssd_store_close(index);
        return 1;
    }

    // The descriptor bypasses the page cache, so read the aligned span that
    // covers the expert into an aligned buffer.
    const uint64_t first = offset / DIRECT_ALIGN * DIRECT_ALIGN;
    const size_t span = (size_t) ((offset + size + DIRECT_ALIGN - 1) / DIRECT_ALIGN * DIRECT_ALIGN - first);
    unsigned char * buffer = NULL;
    if (posix_memalign((void **) &buffer, DIRECT_ALIGN, span) != 0) {
        fprintf(stderr, "out of memory\n");
        inferssd_store_close(index);
        return 1;
    }
    size_t done = 0;
    while (done < span) {
        const ssize_t n = pread(fd, buffer + done, span - done, (off_t) (first + done));
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            break; // error, or end of file inside the last aligned block
        }
        done += (size_t) n;
    }
    const size_t skip = (size_t) (offset - first);
    const int ok = done >= skip + size;
    if (ok) {
        printf("%s expert %d: %zu bytes at offset %" PRIu64 ", first byte 0x%02x\n", tensor, expert, size, offset,
               buffer[skip]);
    } else {
        fprintf(stderr, "short read of %s expert %d\n", tensor, expert);
    }
    free(buffer);
    inferssd_store_close(index);
    return ok ? 0 : 1;
}
