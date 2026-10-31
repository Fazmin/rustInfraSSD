Describe the change, its cause and the evidence for it. For a port or an upstream sync, name the upstream
files and the upstream revision.

The title becomes the commit title on `main`; use `area: summary` as described in
[CONTRIBUTING.md](https://github.com/Fazmin/rustInfraSSD/blob/main/CONTRIBUTING.md#pull-requests).

Validation:

- [ ] `cargo fmt`, `cargo clippy -D warnings`, `cargo test`, the quick start example and `./scripts/check-facade.sh` pass.
- [ ] Ported code matches upstream behaviour and comes with upstream's tests; any deliberate difference is listed in the README.
- [ ] Checked on macOS and Linux, or the unchecked platform is named below.
- [ ] `CHANGELOG.md` is updated under `Unreleased` for user-visible changes.

State any remaining validation, compatibility limits or platforms not checked.
