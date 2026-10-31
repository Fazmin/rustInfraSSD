// SPDX-License-Identifier: Apache-2.0
//! Records the commit the library is built from, for `inferssd_ssd_build_info`.
//!
//! `INFERSSD_GIT_COMMIT` overrides the lookup, for builds from a source archive.
//! Outside a git checkout and without the override, the commit reads `unknown`.

use std::path::Path;
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    let text = text.trim();
    (output.status.success() && !text.is_empty()).then(|| text.to_owned())
}

fn main() {
    println!("cargo::rerun-if-env-changed=INFERSSD_GIT_COMMIT");
    let commit = match std::env::var("INFERSSD_GIT_COMMIT") {
        Ok(commit) if !commit.is_empty() => commit,
        _ => {
            // HEAD moves on checkout (HEAD), on commit (refs/heads) and after
            // `git pack-refs` (packed-refs).
            for name in ["HEAD", "refs/heads", "packed-refs"] {
                if let Some(path) = git(&["rev-parse", "--git-path", name]).filter(|path| Path::new(path).exists()) {
                    println!("cargo::rerun-if-changed={path}");
                }
            }
            git(&["rev-parse", "--short=12", "HEAD"]).unwrap_or_else(|| "unknown".to_owned())
        }
    };
    println!("cargo::rustc-env=INFERSSD_GIT_COMMIT={commit}");
}
