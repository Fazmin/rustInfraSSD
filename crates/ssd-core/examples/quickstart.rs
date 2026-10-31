// SPDX-License-Identifier: Apache-2.0
//! Quick start: index the experts of one tensor, look one up and read it.
//!
//! ```sh
//! cargo run -p ssd-core --example quickstart [-- DIR]
//! ```
//!
//! Real indexes are v1 expert packs or alias TSVs that map each expert to a
//! byte range of a GGUF file. This example writes a small alias TSV over a
//! scratch data file in a temporary directory under DIR (default: the system
//! temporary directory), so it runs without a model.

use ssd_core::store::SourceIndex;
use ssd_platform::file;
use std::error::Error;
use std::fs;
use std::io;
use std::os::fd::BorrowedFd;
use std::path::PathBuf;

const TENSOR: &str = "blk.3.ffn_up_exps.weight";
const EXPERTS: i64 = 4;
const EXPERT_BYTES: usize = 4096;
/// Like tensor data inside a GGUF file, the experts start at an unaligned offset.
const HEADER_BYTES: usize = 100;

fn main() -> Result<(), Box<dyn Error>> {
    let parent = std::env::args_os().nth(1).map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    let dir = tempfile::tempdir_in(&parent)?;

    // A data file with a header and 4 experts of 4 KiB; every byte of expert e is e.
    let data = dir.path().join("experts.bin");
    let experts = (0..EXPERTS as u8).flat_map(|e| [e; EXPERT_BYTES]);
    fs::write(&data, [0xff; HEADER_BYTES].into_iter().chain(experts).collect::<Vec<_>>())?;

    // An alias TSV saying where each expert lives.
    let alias = dir.path().join("experts.tsv");
    let mut rows = String::from("tensor\texpert_idx\tnbytes\tsource_path\toffset\n");
    for e in 0..EXPERTS as usize {
        let offset = HEADER_BYTES + e * EXPERT_BYTES;
        rows += &format!("{TENSOR}\t{e}\t{EXPERT_BYTES}\t{}\t{offset}\n", data.display());
    }
    fs::write(&alias, rows)?;

    // Opening validates every row against its file and opens the data file
    // for direct reads, which bypass the page cache.
    let index = SourceIndex::open(&[&alias]).map_err(|e| {
        format!("{e}\n(Linux tmpfs has no direct I/O; pass a directory on disk: --example quickstart -- DIR)")
    })?;
    let sources = index.expert_index();
    println!("indexed {} entries from {} source file", sources.entries().len(), sources.source_count());

    // Resolve expert 2 to a descriptor borrowed from the index, an offset and a length.
    let expert = 2;
    let (fd, offset, len) = index.resolve(TENSOR.as_bytes(), expert, EXPERT_BYTES, EXPERTS).ok_or("not indexed")?;
    let bytes = read_direct(fd, offset, len)?;
    if bytes.iter().any(|&b| b != expert as u8) {
        return Err("read the wrong bytes".into());
    }
    println!("{TENSOR} expert {expert}: {len} bytes at offset {offset}, first byte {:#04x}", bytes[0]);
    Ok(())
}

/// Reads `len` bytes at `offset` through a direct descriptor. Direct reads
/// must start, end and land on aligned boundaries (4 KiB on Linux), so this
/// reads the aligned span around the range into an aligned buffer.
fn read_direct(fd: BorrowedFd<'_>, offset: u64, len: usize) -> io::Result<Vec<u8>> {
    let align = file::direct_alignment(fd);
    let skip = (offset % align as u64) as usize;
    let span = (skip + len).next_multiple_of(align);
    let mut storage = vec![0u8; span + align];
    let at = storage.as_ptr().addr().next_multiple_of(align) - storage.as_ptr().addr();
    let buf = &mut storage[at..at + span];
    let mut done = 0;
    // The last aligned block may extend past the end of the file; stop once the range is in.
    while done < skip + len {
        match file::pread_some(fd, &mut buf[done..], offset - skip as u64 + done as u64)? {
            0 => return Err(io::ErrorKind::UnexpectedEof.into()),
            n => done += n,
        }
    }
    Ok(buf[skip..skip + len].to_vec())
}
