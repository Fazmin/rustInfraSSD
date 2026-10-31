#!/bin/sh
# Builds the Rust static library, then builds and runs against it:
#   - the C quick start (cpp/examples/quickstart.c) on a scratch alias TSV;
#   - the C++ facade smoke tests.
# Only the C ABI headers, the facades and two upstream declaration headers
# (ggml-ssd/settings/result.h, ggml-ssd/store/source.h) are compiled.
set -eu
export LC_ALL=C # tr must write single bytes
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
cargo build -p ssd-capi --release
lib=target/release/libinferssd_ssd.a
out=target/facade
mkdir -p "$out"
CC=${CC:-cc}
CXX=${CXX:-c++}

"$CC" -std=c11 -O1 -g -Wall -Wextra -Werror -I cpp/include \
    cpp/examples/quickstart.c "$lib" -o "$out/quickstart" -lpthread -ldl -lm
demo=$(mktemp -d)
trap 'rm -rf "$demo"' EXIT
# The same layout as the Rust quick start: a 100-byte header, so the experts
# start unaligned, then 4 experts of 4 KiB in which every byte is the expert number.
tensor=blk.3.ffn_up_exps.weight
head -c 100 /dev/zero | tr '\0' '\377' > "$demo/experts.bin"
printf 'tensor\texpert_idx\tnbytes\tsource_path\toffset\n' > "$demo/experts.tsv"
for e in 0 1 2 3; do
    head -c 4096 /dev/zero | tr '\0' "\\00$e" >> "$demo/experts.bin"
    printf '%s\t%d\t4096\t%s\t%d\n' "$tensor" "$e" "$demo/experts.bin" $((100 + e * 4096)) >> "$demo/experts.tsv"
done
for e in 2 3; do # expert 3's aligned span runs past the end of the file
    result=$("$out/quickstart" "$demo/experts.tsv" "$tensor" "$e" 4 4096)
    echo "$result"
    case $result in
    *"expert $e: 4096 bytes at offset $((100 + e * 4096)), first byte 0x0$e"*) ;;
    *) echo "quickstart.c read the wrong bytes of expert $e" >&2; exit 1 ;;
    esac
done

"$CXX" -std=c++17 -O1 -g -Wall -Wextra -Werror \
    -I cpp/facade -I cpp/include -I upstream/engine/ggml/src \
    cpp/tests/store-facade-smoke.cpp "$lib" \
    -o "$out/store-facade-smoke" -lpthread -ldl
"$out/store-facade-smoke"
