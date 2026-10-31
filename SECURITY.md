# Security policy

## Supported versions

Fixes go to `main` and to the latest release. Older releases are not patched.

## Reporting a vulnerability

Report vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/Fazmin/rustInfraSSD/security/advisories/new).
Do not open a public issue.

A useful report includes:

- the affected version: the output of `inferssd_ssd_build_info()`, or the release tag or commit;
- the operating system and the filesystem holding the model files;
- a reproduction, ideally the smallest expert pack, alias TSV or call sequence that triggers the problem.

Handling is best effort. You will get a reply once the report has been reviewed, and fixed problems are
credited in the release notes unless you ask otherwise.

## Scope

In scope is the code this repository adds under `crates/`, `cpp/` and `scripts/`. That includes:

- parsing of untrusted expert packs, alias TSVs and file paths;
- the handling of caller-supplied pointers, buffers and descriptors in the C ABI.

Code under `upstream/` is a copy of [InferSSD](https://github.com/wici-ai/inferssd) and the llama.cpp
code it bundles. Report problems there to
[InferSSD's security policy](https://github.com/wici-ai/inferssd/security). If a problem also affects
the Rust port, report it here as well.
