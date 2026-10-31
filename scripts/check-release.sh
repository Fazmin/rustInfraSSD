#!/bin/sh
# Checks that a release tag names the workspace version and that CHANGELOG.md
# has a dated entry for it, then prints that entry (the release notes):
#
#     ./scripts/check-release.sh v0.1.0 > notes.md
#
# `cargo test` separately checks version.h and UPSTREAM.md against Cargo.toml.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
tag=${1:?usage: check-release.sh vX.Y.Z}
version=$(sed -n '/^\[workspace\.package\]/,/^\[/s/^version = "\(.*\)"$/\1/p' Cargo.toml)
if [ "$tag" != "v$version" ]; then
    echo "tag $tag does not match the workspace version $version in Cargo.toml" >&2
    exit 1
fi
# The entry runs from "## [X.Y.Z] - YYYY-MM-DD" to the next heading or link definition.
notes=$(awk -v heading="## [$version] - " '
    /^## / || /^\[[^]]*\]: / { if (found) exit; found = index($0, heading) == 1; next }
    found' CHANGELOG.md)
if ! grep -Eq "^## \[$version\] - [0-9]{4}-[0-9]{2}-[0-9]{2}$" CHANGELOG.md || [ -z "$(printf '%s' "$notes" | tr -d '[:space:]')" ]; then
    echo "CHANGELOG.md has no dated, non-empty entry \"## [$version] - YYYY-MM-DD\"" >&2
    exit 1
fi
printf '%s\n' "$notes"
