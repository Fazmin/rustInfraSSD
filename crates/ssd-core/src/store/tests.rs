// SPDX-License-Identifier: Apache-2.0
//! Behaviour of the store, ported from upstream
//! `engine/tests/ssd/fixtures/c0-store/test.cpp` (the frozen parser oracle) and
//! `engine/tests/ssd/test-source-index.cpp`. Checks that count or reuse
//! descriptors live in `tests/store_descriptors.rs`, which runs alone.

use super::*;
use crate::Status;
use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const NAME: &str = "blk.0.ffn_up_exps.weight";

struct Row<'a> {
    tensor: &'a [u8],
    expert: i32,
    offset: u64,
    bytes: u64,
}

fn row(tensor: &str, expert: i32, offset: u64, bytes: u64) -> Row<'_> {
    Row { tensor: tensor.as_bytes(), expert, offset, bytes }
}

fn write_file(path: &Path, bytes: usize, value: u8) {
    fs::write(path, vec![value; bytes]).unwrap();
}

/// A v1 pack padded to 16 KiB, written as the upstream fixture writes it.
fn pack(path: &Path, rows: &[Row<'_>], header_size: u32) {
    let mut out = Vec::new();
    let mut magic = [0u8; 16];
    magic[..13].copy_from_slice(b"GGMLMOEPACKv1");
    out.extend_from_slice(&magic);
    out.extend_from_slice(&1u32.to_ne_bytes());
    out.extend_from_slice(&header_size.to_ne_bytes());
    out.extend_from_slice(&(rows.len() as u64).to_ne_bytes());
    out.extend_from_slice(&4096u64.to_ne_bytes());
    for item in rows {
        let mut name = [0u8; 128];
        let n = item.tensor.len().min(128);
        name[..n].copy_from_slice(&item.tensor[..n]);
        out.extend_from_slice(&name);
        out.extend_from_slice(&item.expert.to_ne_bytes());
        out.extend_from_slice(&0u32.to_ne_bytes());
        out.extend_from_slice(&item.offset.to_ne_bytes());
        out.extend_from_slice(&item.bytes.to_ne_bytes());
    }
    if out.len() < 16384 {
        out.resize(16383, 0);
        out.push(b'x');
    }
    fs::write(path, out).unwrap();
}

fn overwrite(path: &Path, offset: u64, bytes: &[u8]) {
    let mut file = OpenOptions::new().write(true).open(path).unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(bytes).unwrap();
}

const HEADER: &str = "tensor\texpert_idx\tnbytes\tsource_path\toffset\n";

fn alias_row(tensor: &str, expert: i32, bytes: u64, source: &Path, offset: u64) -> String {
    format!("{tensor}\t{expert}\t{bytes}\t{}\t{offset}\n", source.display())
}

fn desc(packs: &[&Path], aliases: &[&Path], page_cache: &[&Path], replace: bool) -> Description {
    let owned = |paths: &[&Path]| paths.iter().map(|p| p.to_path_buf()).collect();
    Description {
        packs: owned(packs),
        aliases: owned(aliases),
        page_cache_packs: owned(page_cache),
        replace_duplicates: replace,
        ..Description::default()
    }
}

fn locate(index: &ExpertIndex, name: &str, expert: i32, bytes: u64) -> Option<Extent> {
    index.locate(name.as_bytes(), expert, bytes, true)
}

struct Fixture {
    dir: tempfile::TempDir,
}

impl Fixture {
    fn new() -> Self {
        Self { dir: tempfile::tempdir().unwrap() }
    }
    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }
}

#[test]
fn logical_bytes_distinguish_keys() {
    let fx = Fixture::new();
    let p1 = fx.path("one.pack");
    pack(&p1, &[row(NAME, 3, 4096, 64), row(NAME, 3, 8192, 128)], 40);
    let base = ExpertIndex::open(&desc(&[&p1], &[], &[], false)).unwrap();
    assert_eq!(locate(&base, NAME, 3, 128).unwrap().offset, 8192);
    assert!(locate(&base, NAME, 3, 63).is_none(), "nonmatching logical size must fail");
}

#[test]
fn duplicates_fail_unless_replaced_and_the_later_source_wins() {
    let fx = Fixture::new();
    let (p1, p2) = (fx.path("one.pack"), fx.path("two.pack"));
    pack(&p1, &[row(NAME, 3, 4096, 64), row(NAME, 3, 8192, 128)], 40);
    pack(&p2, &[row(NAME, 3, 12288, 64)], 40);
    assert!(ExpertIndex::open(&desc(&[&p1, &p2], &[], &[], false)).is_none());
    let overlaid = ExpertIndex::open(&desc(&[&p1, &p2], &[], &[], true)).unwrap();
    let found = locate(&overlaid, NAME, 3, 64).unwrap();
    assert_eq!((found.source_id, found.offset), (1, 12288), "later pack must win explicit replacement");
    let page = ExpertIndex::open(&desc(&[&p1], &[], &[&p2], false)).unwrap();
    assert_eq!(locate(&page, NAME, 3, 64).unwrap().source_id, 1, "page-cache pack must force replacement");
}

#[test]
fn aliases_merge_with_packs_first_alias_wins_and_sources_are_shared() {
    let fx = Fixture::new();
    let (p1, a1, a2, data) = (fx.path("one.pack"), fx.path("one.tsv"), fx.path("two.tsv"), fx.path("weights.gguf"));
    write_file(&data, 16384, 0x31);
    pack(&p1, &[row(NAME, 3, 4096, 64), row(NAME, 3, 8192, 128)], 40);
    fs::write(&a1, HEADER.to_owned() + &alias_row(NAME, 3, 64, &data, 4097) + &alias_row(NAME, 4, 64, &data, 8193))
        .unwrap();
    fs::write(&a2, HEADER.to_owned() + &alias_row(NAME, 4, 64, &data, 12001) + &alias_row(NAME, 5, 64, &data, 12003))
        .unwrap();
    let aliases = ExpertIndex::open(&desc(&[&p1], &[&a1, &a2], &[], true)).unwrap();
    let three = locate(&aliases, NAME, 3, 64).unwrap();
    assert!(!three.alias && three.source_id == 0, "pack entry must precede duplicate alias");
    let four = locate(&aliases, NAME, 4, 64).unwrap();
    assert!(four.alias && four.offset == 8193, "first alias must win even when replacement enabled");
    assert_eq!(aliases.source_count(), 2, "same alias path must reuse one source across TSVs");
    assert_eq!(locate(&aliases, NAME, 5, 64).unwrap().offset % 4096, 3811, "alias must retain unaligned offset");
}

#[test]
fn legacy_alias_columns_and_permissive_integers_still_parse() {
    let fx = Fixture::new();
    let (a1, data) = (fx.path("one.tsv"), fx.path("weights.gguf"));
    write_file(&data, 16384, 0x31);
    let text = format!(
        "source_path\tmodel_offset\texpert\tnbytes\ttensor\r\n{}\t4097tail\t3tail\t64tail\t{NAME}\r\n",
        data.display()
    );
    fs::write(&a1, text).unwrap();
    let permissive = ExpertIndex::open(&desc(&[], &[&a1], &[], false)).unwrap();
    assert_eq!(locate(&permissive, NAME, 3, 64).unwrap().offset, 4097, "legacy numeric suffix parsing changed");
    fs::write(
        &a1,
        "tensor\texpert\tnbytes\tsource_path\toffset_bytes\n".to_owned() + &alias_row(NAME, 3, 64, &data, 4097),
    )
    .unwrap();
    assert!(ExpertIndex::open(&desc(&[], &[&a1], &[], false)).is_some());
}

#[test]
fn the_pack_index_starts_at_byte_40_whatever_the_header_size() {
    let fx = Fixture::new();
    let p2 = fx.path("two.pack");
    pack(&p2, &[row(NAME, 3, 4096, 64)], 64);
    assert!(ExpertIndex::open(&desc(&[&p2], &[], &[], false)).is_some());
}

#[test]
fn ranges_beyond_the_file_fail_at_lookup_without_wrapping() {
    let fx = Fixture::new();
    let (a1, data) = (fx.path("one.tsv"), fx.path("weights.gguf"));
    write_file(&data, 16384, 0x31);
    fs::write(&a1, HEADER.to_owned() + &alias_row(NAME, 3, 64, &data, 16360)).unwrap();
    let tail = ExpertIndex::open(&desc(&[], &[&a1], &[], false)).unwrap();
    assert!(locate(&tail, NAME, 3, 64).is_none(), "logical range beyond EOF must fail at lookup");
    fs::write(&a1, HEADER.to_owned() + &alias_row(NAME, 3, 64, &data, u64::MAX)).unwrap();
    let overflow = ExpertIndex::open(&desc(&[], &[&a1], &[], false)).unwrap();
    assert!(locate(&overflow, NAME, 3, 64).is_none(), "offset overflow must fail without wrapping");
}

#[test]
fn malformed_alias_tsvs_are_rejected() {
    let fx = Fixture::new();
    let (a1, data) = (fx.path("one.tsv"), fx.path("weights.gguf"));
    write_file(&data, 16384, 0x31);
    let rejected = |text: String| {
        fs::write(&a1, text).unwrap();
        ExpertIndex::open(&desc(&[], &[&a1], &[], false)).is_none()
    };
    assert!(rejected("tensor\texpert\tnbytes\tsource_path\n".into()), "missing offset column accepted");
    assert!(rejected(HEADER.to_owned() + "too\tfew\tcolumns\n"), "short row accepted");
    assert!(rejected(HEADER.to_owned() + &alias_row(NAME, -1, 64, &data, 4096)), "negative expert accepted");
    assert!(rejected(String::new()), "empty TSV accepted");
    // A trailing blank line is a bad row upstream too.
    assert!(rejected(HEADER.to_owned() + &alias_row(NAME, 3, 64, &data, 4096) + "\n"));
    assert!(rejected(HEADER.to_owned() + &alias_row(NAME, 3, 0, &data, 4096)), "zero-byte entry accepted");
    assert!(rejected(HEADER.to_owned() + &alias_row(&"n".repeat(128), 3, 64, &data, 4096)), "long name accepted");
    // A bad row fails the file even after a source opened for a good one.
    assert!(rejected(HEADER.to_owned() + &alias_row(NAME, 3, 64, &data, 4096) + "x\n"));
}

#[test]
fn malformed_pack_headers_and_indexes_are_rejected() {
    let fx = Fixture::new();
    let p2 = fx.path("two.pack");
    let rejected = || ExpertIndex::open(&desc(&[&p2], &[], &[], false)).is_none();
    write_file(&p2, 24, 0);
    assert!(rejected(), "truncated header accepted");
    pack(&p2, &[row(NAME, 3, 4096, 64)], 40);
    overwrite(&p2, 0, b"X");
    assert!(rejected(), "bad magic accepted");
    pack(&p2, &[row(NAME, 3, 4096, 64)], 40);
    overwrite(&p2, 16, &2u32.to_ne_bytes());
    assert!(rejected(), "version 2 accepted");
    pack(&p2, &[row(NAME, 3, 4096, 64)], 40);
    overwrite(&p2, 24, &10_000_001u64.to_ne_bytes());
    assert!(rejected(), "entry count above the limit accepted");
    pack(&p2, &[row(NAME, 3, 4096, 64)], 40);
    overwrite(&p2, 20, &39u32.to_ne_bytes());
    assert!(rejected(), "header smaller than 40 bytes accepted");
    pack(&p2, &[row(NAME, 3, 4096, 64)], 40);
    File::options().write(true).open(&p2).unwrap().set_len(60).unwrap();
    assert!(rejected(), "short index accepted");
}

#[test]
fn pack_names_truncate_to_127_bytes_and_entries_are_not_tightened() {
    let fx = Fixture::new();
    let p2 = fx.path("two.pack");
    let long = "n".repeat(128);
    pack(&p2, &[Row { tensor: long.as_bytes(), expert: -1, offset: 4096, bytes: 0 }], 40);
    let index = ExpertIndex::open(&desc(&[&p2], &[], &[], false)).unwrap();
    assert_eq!(index.entries()[0].tensor.len(), 127, "legacy v1 tensor-name truncation changed");
    assert!(index.locate(&long.as_bytes()[..127], -1, 0, true).is_none());
}

#[test]
fn a_generated_corpus_round_trips() {
    let fx = Fixture::new();
    let (a1, data) = (fx.path("one.tsv"), fx.path("weights.gguf"));
    write_file(&data, 16384, 0x31);
    let mut corpus = Vec::new();
    let mut manifest = HEADER.to_owned();
    let mut offset = 4097;
    for layer in 0..2 {
        for expert in 0..16 {
            for role in ["up", "gate", "down"] {
                let key = format!("blk.{layer}.ffn_{role}_exps.weight");
                manifest += &alias_row(&key, expert, 64, &data, offset);
                corpus.push((key, expert, offset));
                offset += 65;
            }
        }
    }
    fs::write(&a1, manifest).unwrap();
    let generated = ExpertIndex::open(&desc(&[], &[&a1], &[], false)).unwrap();
    assert_eq!(generated.entries().len(), 96);
    for (key, expert, offset) in corpus {
        assert_eq!(locate(&generated, &key, expert, 64).unwrap().offset, offset);
    }
}

#[test]
fn an_index_keeps_the_file_it_opened_after_the_path_is_replaced() {
    let fx = Fixture::new();
    let (a1, data) = (fx.path("one.tsv"), fx.path("weights.gguf"));
    write_file(&data, 16384, 0x31);
    fs::write(&a1, HEADER.to_owned() + &alias_row(NAME, 3, 64, &data, 4097)).unwrap();
    let retained = ExpertIndex::open(&desc(&[], &[&a1], &[], false)).unwrap();
    let prior = locate(&retained, NAME, 3, 64).unwrap();
    fs::rename(&data, fx.path("renamed.gguf")).unwrap();
    write_file(&data, 16384, 0x72);
    let replacement = ExpertIndex::open(&desc(&[], &[&a1], &[], false)).unwrap();
    assert_ne!(locate(&replacement, NAME, 3, 64).unwrap().file, prior.file, "stale identity reused");
    assert_eq!(locate(&retained, NAME, 3, 64).unwrap().file, prior.file, "retained identity changed");
    let fd = retained.borrowed_fd(prior.source_id).unwrap();
    let mut value = [0u8; 1];
    ssd_platform::file::pread_exact(fd, &mut value, 4097).unwrap();
    assert_eq!(value[0], 0x31, "borrowed handle must refer to the original inode");
    drop(retained);
    assert!(locate(&replacement, NAME, 3, 64).is_some(), "dropping one index cleared another");
}

/// Two seeded 16 KiB payloads and one alias TSV each, as `test-source-index.cpp` builds them.
fn source_fixture(fx: &Fixture) -> Vec<PathBuf> {
    (1..=2u64)
        .map(|seed| {
            let data = fx.path(&format!("{seed}.bin"));
            let alias = fx.path(&format!("{seed}.tsv"));
            fs::write(&data, (0..16384u64).map(|i| ((i + seed) % 251) as u8).collect::<Vec<_>>()).unwrap();
            fs::write(&alias, HEADER.to_owned() + &alias_row(NAME, 3, 5000, &data, 4097)).unwrap();
            alias
        })
        .collect()
}

#[test]
fn source_indexes_resolve_direct_descriptors_within_bounds() {
    use std::os::fd::AsRawFd;
    let fx = Fixture::new();
    let aliases = source_fixture(&fx);
    let first = SourceIndex::open(&aliases[..1]).unwrap();
    let second = SourceIndex::open(&aliases[1..]).unwrap();
    let name = NAME.as_bytes();
    let (fd_a, offset_a, bytes_a) = first.resolve(name, 3, 5000, 4).unwrap();
    let (fd_b, offset_b, bytes_b) = second.resolve(name, 3, 5000, 4).unwrap();
    assert_ne!(fd_a.as_raw_fd(), fd_b.as_raw_fd(), "source contexts shared a descriptor");
    assert_eq!((offset_a, bytes_a, offset_b, bytes_b), (4097, 5000, 4097, 5000));
    assert!(second.resolve(name, 3, 5000, 3).is_none(), "expert-count bound ignored");
    assert!(second.resolve(name, 3, 5001, 4).is_none(), "tensor byte bound ignored");
    assert!(second.resolve(name, -1, 5000, 4).is_none());
    assert!(second.resolve(name, 3, 0, 4).is_none());
    assert!(ssd_platform::file::is_direct(fd_b));

    // An actual direct read: aligned buffer, offset and length.
    let layout = std::alloc::Layout::from_size_align(8192, 4096).unwrap();
    // SAFETY: the layout has a non-zero size.
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!raw.is_null());
    // SAFETY: `raw` is a live allocation of 8192 bytes, freed below.
    let buffer = unsafe { std::slice::from_raw_parts_mut(raw, 8192) };
    let read = ssd_platform::file::pread_exact(fd_b, buffer, 4096);
    let matches = (0..5000).all(|i| buffer[i + 1] as u64 == (4097 + i as u64 + 2) % 251);
    // SAFETY: allocated above with the same layout.
    unsafe { std::alloc::dealloc(raw, layout) };
    read.unwrap();
    assert!(matches, "direct payload changed");
}

#[test]
fn source_indexes_refuse_bad_sets() {
    let fx = Fixture::new();
    let aliases = source_fixture(&fx);
    let bad = fx.path("bad.tsv");
    fs::write(&bad, HEADER.to_owned() + &alias_row(NAME, 3, 5000, &fx.path("1.bin"), 16000)).unwrap();
    let error = SourceIndex::open(&[&bad]).unwrap_err();
    assert_eq!(error.status, Status::Invalid, "out-of-file source published");
    assert_eq!(error.message, "expert index range exceeds its source file");

    let error = SourceIndex::open(&[aliases[0].clone(), fx.path("missing.tsv")]).unwrap_err();
    assert_eq!(error.status, Status::Io, "partial source set published");

    let error = SourceIndex::open::<PathBuf>(&[]).unwrap_err();
    assert_eq!((error.status, error.message.as_str()), (Status::Invalid, "expert.source requires an index path"));

    let gguf = fx.path("model.gguf");
    fs::write(&gguf, b"GGUF\x03\0\0\0").unwrap();
    let error = SourceIndex::open(&[&gguf]).unwrap_err();
    assert_eq!(error.status, Status::Invalid);
    assert!(error.message.starts_with("expert.source takes an alias TSV for direct GGUF reads: "));

    fs::write(&bad, HEADER.to_owned() + &alias_row(NAME, 3, 5000, &fx.path("absent.bin"), 0)).unwrap();
    let error = SourceIndex::open(&[&bad]).unwrap_err();
    assert_eq!((error.status, error.message.as_str()), (Status::Invalid, "expert source index validation failed"));
}

#[test]
fn source_indexes_never_block_on_or_publish_a_fifo() {
    let fx = Fixture::new();
    let fifo = fx.path("fifo");
    let name = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
    // SAFETY: mkfifo reads a NUL-terminated path.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    // Without a writer, a blocking open would hang this test.
    assert!(SourceIndex::open(&[&fifo]).is_err(), "FIFO index published");
    let bad = fx.path("bad.tsv");
    fs::write(&bad, HEADER.to_owned() + &alias_row(NAME, 3, 5000, &fifo, 0)).unwrap();
    assert!(SourceIndex::open(&[&bad]).is_err(), "FIFO payload published");
}

#[test]
fn dense_blocks_require_exactly_blocks_zero_to_n_minus_one() {
    let fx = Fixture::new();
    let p1 = fx.path("dense.pack");
    // Pack offsets point into the pack itself, which is padded to 16 KiB.
    let dense = "blk.0.attn_q.weight";
    pack(
        &p1,
        &[
            row(dense, 0, 4096, 4096),
            row(dense, 1, 8192, 4096),
            row(dense, 2, 12288, 4096),
            row("blk.0.gap.weight", 0, 0, 512),
        ],
        40,
    );
    let index = SourceIndex::open(&[&p1]).unwrap();
    assert_eq!(index.dense_blocks(dense.as_bytes(), 3 * 4096), Some((3, 4096)));
    assert_eq!(index.dense_blocks(dense.as_bytes(), 2 * 4096), None, "extra block ignored");
    assert_eq!(index.dense_blocks(dense.as_bytes(), 4096 + 1), None, "partial block accepted");
    assert_eq!(index.dense_blocks(dense.as_bytes(), 4 * 4096), None, "missing block ignored");
    assert_eq!(index.dense_blocks(b"blk.0.absent.weight", 4096), None);
    assert_eq!(index.dense_blocks(b"", 4096), None);
    assert_eq!(index.dense_blocks(b"blk.0.gap.weight", 512), Some((1, 512)));
}
