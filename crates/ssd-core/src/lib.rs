// SPDX-License-Identifier: Apache-2.0
//! Platform-independent core of the SSD runtime: the shared result type and
//! the expert source store. Ports of `engine/ggml/src/ggml-ssd/settings/result.h`
//! and `engine/ggml/src/ggml-ssd/store/`.

pub mod result;
pub mod store;

pub use result::{Error, Result, Status};
