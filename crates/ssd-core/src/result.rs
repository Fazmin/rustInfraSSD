// SPDX-License-Identifier: Apache-2.0
//! Port of `ggml-ssd/settings/result.h`. Every module reports failures with
//! one of these statuses; the C ABI passes them as the same integers as the C++
//! `ssd::status` enumeration.

use std::fmt;

/// The C++ `ssd::status` values, in declaration order.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Status {
    Ok = 0,
    Invalid = 1,
    Io = 2,
    Resource = 3,
}

/// A failed operation: its status and a human-readable reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub status: Status,
    pub message: String,
}

impl Error {
    pub fn invalid(message: impl Into<String>) -> Self {
        Self { status: Status::Invalid, message: message.into() }
    }

    pub fn io(message: impl Into<String>) -> Self {
        Self { status: Status::Io, message: message.into() }
    }

    pub fn resource(message: impl Into<String>) -> Self {
        Self { status: Status::Resource, message: message.into() }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
