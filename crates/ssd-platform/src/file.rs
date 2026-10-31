// SPDX-License-Identifier: Apache-2.0
//! Read-only regular files. Port of `ggml-ssd/platform/file.{h,cpp}` (POSIX).
//!
//! Descriptors stay plain file descriptors so the C ABIs that carry an `int`
//! keep their shape. Opens are non-blocking, so a FIFO path cannot stall
//! startup, and the regular-file check runs on the opened descriptor, so a
//! path swap cannot bypass it.

use std::fs::{File, OpenOptions};
use std::io;
use std::mem::MaybeUninit;
use std::os::fd::{AsRawFd, BorrowedFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

/// Largest single positional read. macOS refuses counts above `INT_MAX`;
/// 1 GiB reads short everywhere instead and callers loop.
const MAX_READ: usize = 1 << 30;

/// XNU's `FNOCACHE` file flag, which `F_NOCACHE` sets and `F_GETFL` reports.
/// The SDK keeps the name kernel-only.
#[cfg(target_os = "macos")]
const FNOCACHE: libc::c_int = 0x40000;

fn open_regular(path: &Path, flags: libc::c_int) -> io::Result<File> {
    // std adds O_CLOEXEC; O_NONBLOCK keeps a FIFO path from blocking the open.
    let file = OpenOptions::new().read(true).custom_flags(libc::O_NONBLOCK | flags).open(path)?;
    if !file.metadata()?.file_type().is_file() {
        return Err(io::Error::from_raw_os_error(libc::EINVAL));
    }
    Ok(file)
}

/// Opens `path` for ordinary cached reads, for metadata.
pub fn open_buffered(path: &Path) -> io::Result<File> {
    open_regular(path, 0)
}

/// Opens `path` bypassing the OS page cache: `F_NOCACHE` on macOS, `O_DIRECT`
/// on Linux. Linux then needs offsets, lengths and buffer addresses that are
/// multiples of [`direct_alignment`]; macOS has no such contract.
pub fn open_direct(path: &Path) -> io::Result<File> {
    #[cfg(target_os = "macos")]
    {
        let file = open_regular(path, 0)?;
        // SAFETY: fcntl on a descriptor this function owns; F_NOCACHE takes an int.
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
            return Err(io::Error::last_os_error());
        }
        Ok(file)
    }
    #[cfg(not(target_os = "macos"))]
    {
        open_regular(path, libc::O_DIRECT)
    }
}

/// True for a descriptor that bypasses the page cache. It asks the descriptor
/// itself, so every module of the process gets the same answer.
pub fn is_direct(fd: BorrowedFd<'_>) -> bool {
    // SAFETY: F_GETFL only reads the flags of a descriptor the caller keeps open.
    let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFL) };
    #[cfg(target_os = "macos")]
    let direct = FNOCACHE;
    #[cfg(not(target_os = "macos"))]
    let direct = libc::O_DIRECT;
    flags >= 0 && flags & direct != 0
}

/// The offset, length and address granularity of direct reads.
pub fn direct_alignment(_fd: BorrowedFd<'_>) -> usize {
    if cfg!(target_os = "macos") {
        1 // F_NOCACHE has no alignment contract
    } else {
        4096 // the granularity every O_DIRECT caller already pads to
    }
}

fn fstat(fd: BorrowedFd<'_>) -> io::Result<libc::stat> {
    let mut info = MaybeUninit::<libc::stat>::uninit();
    // SAFETY: fstat writes a whole `stat` into the buffer on success.
    if unsafe { libc::fstat(fd.as_raw_fd(), info.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fstat returned 0, so the buffer is initialised.
    Ok(unsafe { info.assume_init() })
}

/// The size in bytes of the file behind `fd`.
pub fn file_size(fd: BorrowedFd<'_>) -> io::Result<u64> {
    let info = fstat(fd)?;
    u64::try_from(info.st_size).map_err(|_| io::Error::from_raw_os_error(libc::EOVERFLOW))
}

/// Names one file version: the same identity means the same file and contents
/// as far as the OS reports. POSIX keeps `st_ino` in `id_lo`; `id_hi` is the
/// upper half of Windows' 128-bit file id. Times are ns since the Unix epoch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FileIdentity {
    pub device: u64,
    pub id_lo: u64,
    pub id_hi: u64,
    pub size: u64,
    pub mtime_ns: i64,
    pub ctime_ns: i64,
}

/// The identity of the file behind `fd`.
// time_t and the nanosecond fields are i64 on 64-bit targets but i32 on some others.
#[allow(clippy::useless_conversion)]
pub fn identity(fd: BorrowedFd<'_>) -> io::Result<FileIdentity> {
    let info = fstat(fd)?;
    Ok(FileIdentity {
        device: info.st_dev as u64,
        id_lo: info.st_ino as u64,
        id_hi: 0,
        size: info.st_size as u64,
        mtime_ns: i64::from(info.st_mtime) * 1_000_000_000 + i64::from(info.st_mtime_nsec),
        ctime_ns: i64::from(info.st_ctime) * 1_000_000_000 + i64::from(info.st_ctime_nsec),
    })
}

/// One positional read; safe to call concurrently on one descriptor. Returns
/// the bytes read, 0 at end of file.
pub fn pread_some(fd: BorrowedFd<'_>, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    let offset = libc::off_t::try_from(offset).map_err(|_| io::Error::from_raw_os_error(libc::EINVAL))?;
    let len = buf.len().min(MAX_READ);
    loop {
        // SAFETY: `buf` is valid for writes of `len` bytes.
        let got = unsafe { libc::pread(fd.as_raw_fd(), buf.as_mut_ptr().cast(), len, offset) };
        if got >= 0 {
            return Ok(got as usize);
        }
        let error = io::Error::last_os_error();
        if error.kind() != io::ErrorKind::Interrupted {
            return Err(error);
        }
    }
}

/// Reads exactly `buf.len()` bytes. A file that ends first is
/// [`io::ErrorKind::UnexpectedEof`].
pub fn pread_exact(fd: BorrowedFd<'_>, mut buf: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !buf.is_empty() {
        match pread_some(fd, buf, offset)? {
            0 => return Err(io::ErrorKind::UnexpectedEof.into()),
            got => {
                buf = &mut buf[got..];
                offset += got as u64;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::fd::AsFd;

    fn sample(dir: &Path, bytes: usize) -> std::path::PathBuf {
        let path = dir.join("sample.bin");
        let data: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
        File::create(&path).unwrap().write_all(&data).unwrap();
        path
    }

    #[test]
    fn direct_open_reports_direct_and_buffered_does_not() {
        let dir = tempfile::tempdir().unwrap();
        let path = sample(dir.path(), 16384);
        let direct = open_direct(&path).unwrap();
        let buffered = open_buffered(&path).unwrap();
        assert!(is_direct(direct.as_fd()));
        assert!(!is_direct(buffered.as_fd()));
        assert_eq!(file_size(direct.as_fd()).unwrap(), 16384);
    }

    #[test]
    fn both_opens_name_the_same_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = sample(dir.path(), 4096);
        let a = identity(open_direct(&path).unwrap().as_fd()).unwrap();
        let b = identity(open_buffered(&path).unwrap().as_fd()).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.size, 4096);
    }

    #[test]
    fn non_regular_paths_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let error = open_buffered(dir.path()).unwrap_err();
        assert_eq!(error.raw_os_error(), Some(libc::EINVAL));
        let fifo = dir.path().join("fifo");
        let name = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
        // SAFETY: mkfifo reads a NUL-terminated path.
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        // A FIFO without a writer must neither block nor open.
        assert_eq!(open_buffered(&fifo).unwrap_err().raw_os_error(), Some(libc::EINVAL));
        assert_eq!(open_direct(&fifo).unwrap_err().raw_os_error(), Some(libc::EINVAL));
    }

    #[test]
    fn pread_exact_reads_and_reports_short_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = sample(dir.path(), 5000);
        let file = open_buffered(&path).unwrap();
        let mut buf = [0u8; 16];
        pread_exact(file.as_fd(), &mut buf, 4097).unwrap();
        assert!(buf.iter().enumerate().all(|(i, &b)| b == ((4097 + i) % 251) as u8));
        let mut tail = [0u8; 16];
        let error = pread_exact(file.as_fd(), &mut tail, 4990).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof);
        assert_eq!(pread_some(file.as_fd(), &mut tail, 5000).unwrap(), 0);
    }
}
