// SPDX-License-Identifier: Apache-2.0
//! The expert source store. Port of `engine/ggml/src/ggml-ssd/store/`.
//!
//! It parses v1 expert packs (`GGMLMOEPACKv1`) and GGUF alias TSVs into one
//! sorted, immutable index from (tensor, expert, bytes) to (direct descriptor,
//! offset). Executors only borrow that index.

pub mod ctext;
mod index;
pub mod parser;

pub use index::{ExpertIndex, Extent, SourceIdentity, SourceIndex};
pub use parser::{Description, Entry, IoBackend, OpenHooks};

#[cfg(test)]
mod tests;
