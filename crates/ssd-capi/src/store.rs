// SPDX-License-Identifier: Apache-2.0
//! `inferssd/store.h`: the context-owned expert source index.
//!
//! The index owns its source descriptors. A descriptor returned by
//! [`inferssd_store_resolve`] is borrowed: it stays valid until
//! [`inferssd_store_close`] and must never be closed by the caller.

use crate::{c_bytes, guarded, status_code, write_error};
use ssd_core::store::SourceIndex;
use ssd_core::{Error, Status};
use std::ffi::{OsStr, c_char, c_int};
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

/// Opaque handle to a [`SourceIndex`].
pub struct InferssdStoreIndex(SourceIndex);

/// Opens an index over `count` pack or alias TSV paths. On success stores the
/// handle in `*out` and returns 0; otherwise returns an `ssd::status` code and
/// writes the reason into `error`.
///
/// # Safety
/// `paths` must point to `count` NUL-terminated strings, `out` must be valid
/// for a write, and `error` must be null or valid for `error_size` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferssd_store_open(
    paths: *const *const c_char,
    count: usize,
    out: *mut *mut InferssdStoreIndex,
    error: *mut c_char,
    error_size: usize,
) -> c_int {
    let result = guarded(Err(Error::invalid("source invalid")), || {
        if out.is_null() || (paths.is_null() && count != 0) {
            return Err(Error::invalid("invalid inferssd_store_open arguments"));
        }
        let mut owned = Vec::with_capacity(count);
        for i in 0..count {
            // SAFETY: the caller provides `count` readable string pointers.
            let path = unsafe { c_bytes(*paths.add(i)) }.ok_or_else(|| Error::invalid("null expert source path"))?;
            owned.push(Path::new(OsStr::from_bytes(path)).to_path_buf());
        }
        SourceIndex::open(&owned)
    });
    match result {
        Ok(index) => {
            // SAFETY: `out` was checked non-null and the caller made it writable.
            unsafe { *out = Box::into_raw(Box::new(InferssdStoreIndex(index))) };
            status_code(Status::Ok)
        }
        Err(failure) => {
            // SAFETY: the caller made `error` writable for `error_size` bytes.
            unsafe { write_error(error, error_size, &failure.message) };
            status_code(failure.status)
        }
    }
}

/// Destroys an index and closes its descriptors. Null is ignored.
///
/// # Safety
/// `index` must be null or a handle from [`inferssd_store_open`] that no
/// reader still uses; it is invalid afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferssd_store_close(index: *mut InferssdStoreIndex) {
    if !index.is_null() {
        // SAFETY: the handle came from Box::into_raw and is not used again.
        drop(unsafe { Box::from_raw(index) });
    }
}

/// `source_view::resolve`: the direct descriptor, offset and logical size of
/// expert `expert` of `experts` in tensor `name`.
///
/// # Safety
/// `index` must be null or a live handle; `name` null or NUL-terminated; the
/// output pointers null or valid for writes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferssd_store_resolve(
    index: *const InferssdStoreIndex,
    name: *const c_char,
    expert: c_int,
    bytes: usize,
    experts: i64,
    fd: *mut c_int,
    offset: *mut u64,
    logical_bytes: *mut usize,
) -> bool {
    if index.is_null() || fd.is_null() || offset.is_null() || logical_bytes.is_null() {
        return false;
    }
    guarded(false, || {
        // SAFETY: checked non-null; the caller keeps the handle and name alive.
        let (index, Some(name)) = (unsafe { &(*index).0 }, unsafe { c_bytes(name) }) else {
            return false;
        };
        let Some((direct, at, logical)) = index.resolve(name, expert, bytes, experts) else {
            return false;
        };
        // SAFETY: the outputs were checked non-null and the caller made them writable.
        unsafe {
            *fd = direct.as_raw_fd();
            *offset = at;
            *logical_bytes = logical;
        }
        true
    })
}

/// `source_view::dense_blocks`: the number of equal blocks 0..N-1 that make up
/// `bytes` of tensor `name`, or 0. Stores the block size in `*block_bytes`
/// when that is non-null.
///
/// # Safety
/// `index` must be null or a live handle; `name` null or NUL-terminated;
/// `block_bytes` null or valid for a write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn inferssd_store_dense_blocks(
    index: *const InferssdStoreIndex,
    name: *const c_char,
    bytes: usize,
    block_bytes: *mut usize,
) -> usize {
    if index.is_null() {
        return 0;
    }
    guarded(0, || {
        // SAFETY: checked non-null; the caller keeps the handle and name alive.
        let (index, Some(name)) = (unsafe { &(*index).0 }, unsafe { c_bytes(name) }) else {
            return 0;
        };
        let Some((count, block)) = index.dense_blocks(name, bytes) else {
            return 0;
        };
        if !block_bytes.is_null() {
            // SAFETY: checked non-null; the caller made it writable.
            unsafe { *block_bytes = block };
        }
        count
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::ptr;

    #[test]
    fn open_resolve_and_close_through_the_c_abi() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("data.bin");
        let alias = dir.path().join("a.tsv");
        std::fs::write(&data, vec![7u8; 16384]).unwrap();
        std::fs::write(
            &alias,
            format!(
                "tensor\texpert_idx\tnbytes\tsource_path\toffset\nblk.0.ffn_up_exps.weight\t3\t5000\t{}\t4097\n",
                data.display()
            ),
        )
        .unwrap();
        let path = CString::new(alias.as_os_str().as_encoded_bytes()).unwrap();
        let paths = [path.as_ptr()];
        let mut handle = ptr::null_mut();
        let mut error = [0 as c_char; 64];
        // SAFETY: valid path array, output and error buffer.
        let code = unsafe { inferssd_store_open(paths.as_ptr(), 1, &mut handle, error.as_mut_ptr(), error.len()) };
        assert_eq!(code, 0);
        let name = CString::new("blk.0.ffn_up_exps.weight").unwrap();
        let (mut fd, mut offset, mut logical) = (-1, 0u64, 0usize);
        // SAFETY: live handle and valid outputs.
        assert!(unsafe {
            inferssd_store_resolve(handle, name.as_ptr(), 3, 5000, 4, &mut fd, &mut offset, &mut logical)
        });
        assert!(fd >= 0 && offset == 4097 && logical == 5000);
        // SAFETY: as above; null outputs and names are refused, not dereferenced.
        unsafe {
            assert!(!inferssd_store_resolve(handle, ptr::null(), 3, 5000, 4, &mut fd, &mut offset, &mut logical));
            assert!(!inferssd_store_resolve(
                handle,
                name.as_ptr(),
                3,
                5000,
                4,
                ptr::null_mut(),
                &mut offset,
                &mut logical
            ));
            assert_eq!(inferssd_store_dense_blocks(handle, name.as_ptr(), 5000, ptr::null_mut()), 0);
            inferssd_store_close(handle);
            inferssd_store_close(ptr::null_mut());
        }
    }

    #[test]
    fn failures_report_their_status_and_a_truncated_message() {
        let missing = CString::new("/nonexistent/inferssd/index.tsv").unwrap();
        let paths = [missing.as_ptr()];
        let mut handle = ptr::null_mut();
        let mut error = [0x55 as c_char; 16];
        // SAFETY: valid path array, output and error buffer.
        let code = unsafe { inferssd_store_open(paths.as_ptr(), 1, &mut handle, error.as_mut_ptr(), error.len()) };
        assert_eq!(code, Status::Io as c_int);
        assert!(handle.is_null());
        // SAFETY: write_error NUL-terminated the buffer.
        let text = unsafe { std::ffi::CStr::from_ptr(error.as_ptr()) };
        assert_eq!(text.to_bytes(), b"cannot open exp");
        // SAFETY: zero paths and a null error buffer are allowed.
        let code = unsafe { inferssd_store_open(ptr::null(), 0, &mut handle, ptr::null_mut(), 0) };
        assert_eq!(code, Status::Invalid as c_int);
    }
}
