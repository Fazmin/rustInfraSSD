// SPDX-License-Identifier: Apache-2.0
//! The C ABI of the Rust SSD layer. The declarations live in
//! `cpp/include/inferssd/*.h`; the C++ facades under `cpp/facade/` present
//! them under the upstream C++ names.
//!
//! No panic crosses this boundary: every entry point catches it and reports an
//! ordinary failure.

use std::ffi::{CStr, c_char, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};

pub mod store;
pub mod version;

/// Copies `message` into a caller buffer of `size` bytes as a NUL-terminated
/// string, truncating if needed. Does nothing for a null or empty buffer.
///
/// # Safety
/// `buffer` must be null or valid for writes of `size` bytes.
unsafe fn write_error(buffer: *mut c_char, size: usize, message: &str) {
    if buffer.is_null() || size == 0 {
        return;
    }
    let bytes = message.as_bytes();
    let n = bytes.len().min(size - 1);
    // SAFETY: the caller guarantees `size` writable bytes; n + 1 <= size.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.cast::<u8>(), n);
        *buffer.add(n) = 0;
    }
}

/// The bytes of a C string, or `None` for a null pointer.
///
/// # Safety
/// `text` must be null or point to a NUL-terminated string that outlives `'a`.
unsafe fn c_bytes<'a>(text: *const c_char) -> Option<&'a [u8]> {
    // SAFETY: guaranteed by the caller.
    (!text.is_null()).then(|| unsafe { CStr::from_ptr(text) }.to_bytes())
}

/// Runs `body`, turning a panic into `fallback`.
fn guarded<T>(fallback: T, body: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(body)).unwrap_or(fallback)
}

/// `ssd::status` as the C ABI passes it.
fn status_code(status: ssd_core::Status) -> c_int {
    status as c_int
}
