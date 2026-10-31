// SPDX-License-Identifier: Apache-2.0
//! `inferssd/version.h`: which release, build commit and upstream revision a
//! library was built from.
//!
//! The header's macros describe the headers a caller compiled against; these
//! functions describe the library actually linked.

use std::ffi::{CStr, CString, c_char};
use std::sync::OnceLock;

/// This library's version, from the workspace `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// The InferSSD release the port tracks. `UPSTREAM.md` records the same pin.
pub const UPSTREAM_VERSION: &str = "v0.1.0";
/// The InferSSD commit of [`UPSTREAM_VERSION`].
pub const UPSTREAM_COMMIT: &str = "b9cc72d6ec26944e628dceb0e8bbb4e610900052";
/// The commit this library was built from, or `unknown` outside a git checkout.
pub const GIT_COMMIT: &str = env!("INFERSSD_GIT_COMMIT");

const VERSION_C: &CStr = match CStr::from_bytes_with_nul(concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes()) {
    Ok(text) => text,
    Err(_) => panic!("the package version contains a NUL byte"),
};

fn build_info_c() -> &'static CStr {
    static INFO: OnceLock<CString> = OnceLock::new();
    INFO.get_or_init(|| {
        let info = format!(
            "rustInfraSSD {VERSION} (git {GIT_COMMIT}, InferSSD {UPSTREAM_VERSION} {})",
            &UPSTREAM_COMMIT[..12]
        );
        CString::new(info).unwrap_or_default()
    })
}

/// One line naming this build, for logs:
/// `rustInfraSSD 0.1.0 (git 1a2b3c4d5e6f, InferSSD v0.1.0 b9cc72d6ec26)`.
pub fn build_info() -> &'static str {
    build_info_c().to_str().unwrap_or_default()
}

/// The linked library's version, such as `0.1.0`, in static storage.
#[unsafe(no_mangle)]
pub extern "C" fn inferssd_ssd_version() -> *const c_char {
    VERSION_C.as_ptr()
}

/// [`build_info`] as a NUL-terminated string in static storage.
#[unsafe(no_mangle)]
pub extern "C" fn inferssd_ssd_build_info() -> *const c_char {
    build_info_c().as_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The value of `#define NAME value` in `version.h`, without quotes.
    fn header_define(name: &str) -> &'static str {
        include_str!("../../../cpp/include/inferssd/version.h")
            .lines()
            .find_map(|line| line.strip_prefix("#define ")?.strip_prefix(name)?.strip_prefix(' '))
            .map(|value| value.trim().trim_matches('"'))
            .unwrap_or_else(|| panic!("version.h does not define {name}"))
    }

    #[test]
    fn header_version_matches_cargo() {
        assert_eq!(header_define("INFERSSD_SSD_VERSION"), VERSION);
        assert_eq!(header_define("INFERSSD_SSD_VERSION_MAJOR"), env!("CARGO_PKG_VERSION_MAJOR"));
        assert_eq!(header_define("INFERSSD_SSD_VERSION_MINOR"), env!("CARGO_PKG_VERSION_MINOR"));
        assert_eq!(header_define("INFERSSD_SSD_VERSION_PATCH"), env!("CARGO_PKG_VERSION_PATCH"));
    }

    #[test]
    fn upstream_pin_matches_header_and_upstream_md() {
        assert_eq!(header_define("INFERSSD_SSD_UPSTREAM_VERSION"), UPSTREAM_VERSION);
        assert_eq!(header_define("INFERSSD_SSD_UPSTREAM_COMMIT"), UPSTREAM_COMMIT);
        let pinned = include_str!("../../../UPSTREAM.md")
            .lines()
            .find(|line| line.starts_with("| Pinned revision |"))
            .expect("UPSTREAM.md has a pinned revision row");
        assert!(pinned.contains(&format!("`{UPSTREAM_COMMIT}`")), "UPSTREAM.md pins another commit: {pinned}");
        assert!(pinned.contains(&format!("tag `{UPSTREAM_VERSION}`")), "UPSTREAM.md pins another tag: {pinned}");
    }

    #[test]
    fn c_functions_report_this_build() {
        // SAFETY: both functions return NUL-terminated strings in static storage.
        let (version, info) =
            unsafe { (CStr::from_ptr(inferssd_ssd_version()), CStr::from_ptr(inferssd_ssd_build_info())) };
        assert_eq!(version.to_str(), Ok(VERSION));
        assert_eq!(info.to_str(), Ok(build_info()));
        assert!(
            build_info()
                .starts_with(&format!("rustInfraSSD {VERSION} (git {GIT_COMMIT}, InferSSD v0.1.0 b9cc72d6ec26)"))
        );
    }
}
