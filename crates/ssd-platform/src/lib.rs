// SPDX-License-Identifier: Apache-2.0
//! The single per-OS layer of the SSD runtime. Port of
//! `engine/ggml/src/ggml-ssd/platform/`.
//!
//! Only the file module is ported so far, for macOS and Linux. The Windows
//! implementation (CRT descriptors over `CreateFileW` handles) is pending.

#[cfg(not(unix))]
compile_error!("ssd-platform is ported for macOS and Linux only; the Windows port is pending");

pub mod file;
