// SPDX-License-Identifier: Apache-2.0
//! Descriptor ownership of the store. These checks count open descriptors and
//! look at descriptor numbers after a close, so they run as the only test in
//! this binary. Ported from upstream `c0-store/test.cpp`,
//! `test-store-parser-failures.cpp` and `test-source-index.cpp`.

use ssd_core::store::{Description, ExpertIndex, SourceIndex};
use std::fs;
use std::os::fd::AsRawFd;
use std::path::Path;

const NAME: &str = "blk.0.ffn_up_exps.weight";
const HEADER: &str = "tensor\texpert_idx\tnbytes\tsource_path\toffset\n";

fn open_descriptors() -> Vec<bool> {
    // SAFETY: F_GETFD only queries whether each descriptor number is open.
    (0..1024).map(|fd| unsafe { libc::fcntl(fd, libc::F_GETFD) } != -1).collect()
}

fn is_closed(fd: i32) -> bool {
    // SAFETY: as above.
    let result = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    result == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EBADF)
}

fn pack(path: &Path, rows: &[(i32, u64, u64)]) {
    let mut out = b"GGMLMOEPACKv1\0\0\0".to_vec();
    out.extend_from_slice(&1u32.to_ne_bytes());
    out.extend_from_slice(&40u32.to_ne_bytes());
    out.extend_from_slice(&(rows.len() as u64).to_ne_bytes());
    out.extend_from_slice(&4096u64.to_ne_bytes());
    for &(expert, offset, bytes) in rows {
        let mut name = [0u8; 128];
        name[..NAME.len()].copy_from_slice(NAME.as_bytes());
        out.extend_from_slice(&name);
        out.extend_from_slice(&expert.to_ne_bytes());
        out.extend_from_slice(&0u32.to_ne_bytes());
        out.extend_from_slice(&offset.to_ne_bytes());
        out.extend_from_slice(&bytes.to_ne_bytes());
    }
    out.resize(16384, 0);
    fs::write(path, out).unwrap();
}

#[test]
fn indexes_own_and_release_every_descriptor() {
    let dir = tempfile::tempdir().unwrap();
    let (p1, p2, data, alias) = (
        dir.path().join("one.pack"),
        dir.path().join("two.pack"),
        dir.path().join("data.bin"),
        dir.path().join("a.tsv"),
    );
    fs::write(&data, vec![0x31u8; 16384]).unwrap();
    pack(&p1, &[(3, 4096, 64)]);
    pack(&p2, &[(3, 12288, 64)]);

    // A load that fails after opening sources releases all of them.
    let before = open_descriptors();
    let failed = Description { packs: vec![p1.clone(), p2.clone()], ..Description::default() };
    assert!(ExpertIndex::open(&failed).is_none());
    let bad_alias = format!("{HEADER}{NAME}\t3\t64\t{}\t0\nshort\n", data.display());
    fs::write(&alias, bad_alias).unwrap();
    assert!(SourceIndex::open(&[&alias]).is_err());
    assert_eq!(open_descriptors(), before, "a failed load leaked a descriptor");

    // Dropping an index closes the handles it lent out, and only those.
    fs::write(&alias, format!("{HEADER}{NAME}\t3\t64\t{}\t4097\n", data.display())).unwrap();
    let first = SourceIndex::open(&[&alias]).unwrap();
    let second = SourceIndex::open(&[&alias]).unwrap();
    let fd_first = first.resolve(NAME.as_bytes(), 3, 64, 4).unwrap().0.as_raw_fd();
    let fd_second = second.resolve(NAME.as_bytes(), 3, 64, 4).unwrap().0.as_raw_fd();
    assert_ne!(fd_first, fd_second, "two indexes shared a descriptor");
    drop(first);
    assert!(is_closed(fd_first), "index destruction must close its direct handle");
    assert!(!is_closed(fd_second), "dropping one index closed another's handle");
    assert!(second.resolve(NAME.as_bytes(), 3, 64, 4).is_some());
    drop(second);
    assert_eq!(open_descriptors(), before, "dropped indexes leaked a descriptor");
}
