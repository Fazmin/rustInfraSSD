// SPDX-License-Identifier: Apache-2.0
//! Expert pack and GGUF alias parsing. Port of `ggml-ssd/store/parser.{h,cpp}`
//! and `catalog.h`.
//!
//! The diagnostics keep their upstream `[moe_stream_batch]` wording.

use super::ctext;
use ssd_platform::file;
use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::os::fd::AsFd;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

/// The 13 significant bytes of a v1 pack's 16-byte magic field.
pub const PACK_MAGIC: &[u8; 13] = b"GGMLMOEPACKv1";
/// Header: magic[16], u32 version, u32 header size, u64 entries, u64 data start.
pub const PACK_HEADER_BYTES: u64 = 40;
/// Index entry: char tensor[128], i32 expert, u32 reserved, u64 offset, u64 bytes.
pub const PACK_ENTRY_BYTES: usize = 152;
/// Tensor names are stored in a `char[128]`, so at most 127 bytes survive.
pub const TENSOR_NAME_BYTES: usize = 128;
const MAX_PACK_ENTRIES: u64 = 10_000_000;

/// How payload descriptors are opened (the legacy `io_backend` integer).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum IoBackend {
    /// Buffered handles only.
    #[default]
    Buffered = 0,
    /// A page-cache-bypassing handle per source.
    Direct = 1,
    /// Direct handles, reported to an io_uring transport through the hooks.
    Uring = 2,
}

/// One (tensor, expert, logical size) key and where its bytes live.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// The C string, at most 127 bytes, compared byte-wise like `strcmp`.
    pub tensor: Box<[u8]>,
    pub expert_idx: i32,
    pub source_idx: i32,
    pub offset: u64,
    pub nbytes: u64,
    pub alias_source: bool,
}

impl Entry {
    fn key(&self) -> (&[u8], i32, u64) {
        (&self.tensor, self.expert_idx, self.nbytes)
    }
}

/// One opened data file: a buffered handle for metadata and, unless it is a
/// page-cache source or the backend is buffered, a direct handle for payloads.
#[derive(Debug)]
pub struct Source {
    pub file: File,
    pub direct: Option<File>,
    pub page_cache: bool,
    pub direct_required: bool,
    pub file_size: u64,
    pub path: PathBuf,
}

/// The sources and entries one index owns.
#[derive(Debug, Default)]
pub struct ParserState {
    pub sources: Vec<Source>,
    pub entries: Vec<Entry>,
    pub io_backend: IoBackend,
    pub page_cache_sources: u64,
}

/// What to load: v1 packs, then alias TSVs, then page-cache packs.
#[derive(Clone, Debug, Default)]
pub struct Description {
    pub packs: Vec<PathBuf>,
    pub aliases: Vec<PathBuf>,
    pub page_cache_packs: Vec<PathBuf>,
    pub replace_duplicates: bool,
    pub io_backend: IoBackend,
    pub require_direct: bool,
}

/// Called with the source label and path when a direct source opens for io_uring.
pub type ReportIouring<'a> = &'a dyn Fn(&str, &Path);

/// Transport callbacks, run synchronously at each source open.
#[derive(Clone, Copy, Default)]
pub struct OpenHooks<'a> {
    /// Set only when the transport was built with io_uring support.
    pub report_iouring_enabled: Option<ReportIouring<'a>>,
    pub page_cache_source_opened: Option<&'a dyn Fn()>,
}

fn log(message: std::fmt::Arguments<'_>) {
    eprintln!("[moe_stream_batch] {message}");
}

fn path_of(bytes: &[u8]) -> &Path {
    Path::new(OsStr::from_bytes(bytes))
}

/// The bytes of `name` truncated to the 127 a `char[128]` holds.
fn tensor_name(raw: &[u8]) -> Box<[u8]> {
    ctext::c_str(&raw[..raw.len().min(TENSOR_NAME_BYTES - 1)]).into()
}

/// `fread`-style read: as many of `buf.len()` bytes as the stream yields.
fn read_up_to(reader: &mut impl Read, buf: &mut [u8]) -> usize {
    let mut done = 0;
    while done < buf.len() {
        match reader.read(&mut buf[done..]) {
            Ok(0) => break,
            Ok(got) => done += got,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
    done
}

/// The leading magic bytes of `path`, for choosing between pack and alias.
pub fn probe_magic(path: &Path) -> io::Result<([u8; 16], usize)> {
    let mut magic = [0u8; 16];
    let mut file = file::open_buffered(path)?;
    let count = read_up_to(&mut file, &mut magic);
    Ok((magic, count))
}

struct Parser<'s, 'h> {
    state: &'s mut ParserState,
    hooks: OpenHooks<'h>,
    force_direct: bool,
}

impl Parser<'_, '_> {
    fn open_data_source(&mut self, path: &Path, label: &str, page_cache: bool, direct_required: bool) -> bool {
        if path.as_os_str().is_empty() {
            return false;
        }
        let direct_required = direct_required || self.force_direct;
        let Ok(file) = file::open_buffered(path) else {
            log(format_args!("{label}: open failed: {}", path.display()));
            return false;
        };
        let Ok(file_size) = file::file_size(file.as_fd()) else {
            log(format_args!("{label}: stat failed: {}", path.display()));
            return false;
        };
        let mut direct = None;
        let backend = self.state.io_backend;
        if !page_cache && matches!(backend, IoBackend::Direct | IoBackend::Uring) {
            match file::open_direct(path) {
                Err(_) if direct_required => {
                    log(format_args!("{label}: required direct open failed: {}", path.display()));
                    return false;
                }
                Err(_) => {
                    log(format_args!("{label}: direct open failed; using buffered reads: {}", path.display()));
                    self.state.io_backend = IoBackend::Buffered;
                }
                Ok(handle) => {
                    direct = Some(handle);
                    if backend == IoBackend::Uring {
                        if let Some(report) = self.hooks.report_iouring_enabled {
                            report(label, path);
                        } else {
                            let fallback =
                                if direct_required { "using direct fallback anyway" } else { "using direct fallback" };
                            log(format_args!(
                                "{label}: io_uring requested but liburing headers are unavailable; {fallback}"
                            ));
                            self.state.io_backend = IoBackend::Direct;
                        }
                    } else {
                        log(format_args!("{label}: direct reads enabled: {}", path.display()));
                    }
                }
            }
        } else if page_cache {
            self.state.page_cache_sources += 1;
            if let Some(opened) = self.hooks.page_cache_source_opened {
                opened();
            }
            log(format_args!("{label}: page-cache buffered source enabled: {}", path.display()));
        }
        self.state.sources.push(Source {
            file,
            direct,
            page_cache,
            direct_required,
            file_size,
            path: path.to_path_buf(),
        });
        true
    }

    fn load_pack(&mut self, path: &Path, entries: &mut Vec<Entry>, page_cache: bool) -> bool {
        if path.as_os_str().is_empty() {
            return false;
        }
        let Ok(mut probe) = file::open_buffered(path) else {
            log(format_args!("expert pack: open failed: {}", path.display()));
            return false;
        };
        let mut header = [0u8; PACK_HEADER_BYTES as usize];
        let complete = read_up_to(&mut probe, &mut header) == header.len();
        drop(probe);
        let u32_at = |at: usize| u32::from_ne_bytes(header[at..at + 4].try_into().unwrap());
        let u64_at = |at: usize| u64::from_ne_bytes(header[at..at + 8].try_into().unwrap());
        let (version, header_size, n_entries, data_start) = (u32_at(16), u32_at(20), u64_at(24), u64_at(32));
        if !complete
            || &header[..13] != PACK_MAGIC
            || version != 1
            || header_size < 40
            || data_start < u64::from(header_size)
            || n_entries > MAX_PACK_ENTRIES
        {
            log(format_args!("expert pack: invalid header: {}", path.display()));
            return false;
        }

        let source_idx = self.state.sources.len() as i32;
        let label = if page_cache { "page-cache expert pack" } else { "expert pack" };
        if !self.open_data_source(path, label, page_cache, false) {
            return false;
        }
        let source = self.state.sources.last().expect("source was just opened");
        // The index always starts right after the 40-byte header.
        let mut index = BufReader::new(&source.file);
        if index.seek(SeekFrom::Start(PACK_HEADER_BYTES)).is_err() {
            log(format_args!("expert pack: seek index failed: {}", path.display()));
            return false;
        }
        // Grow per record rather than trusting the header's count up front.
        let mut record = [0u8; PACK_ENTRY_BYTES];
        for _ in 0..n_entries {
            if index.read_exact(&mut record).is_err() {
                log(format_args!("expert pack: short index: {}", path.display()));
                return false;
            }
            entries.push(Entry {
                tensor: tensor_name(&record[..TENSOR_NAME_BYTES]),
                expert_idx: i32::from_ne_bytes(record[128..132].try_into().unwrap()),
                source_idx,
                offset: u64::from_ne_bytes(record[136..144].try_into().unwrap()),
                nbytes: u64::from_ne_bytes(record[144..152].try_into().unwrap()),
                alias_source: false,
            });
        }
        log(format_args!("expert pack: loaded {n_entries} entries from {}", path.display()));
        true
    }

    fn load_alias_tsv(
        &mut self,
        path: &Path,
        entries: &mut Vec<Entry>,
        alias_sources: &mut HashMap<Vec<u8>, i32>,
    ) -> bool {
        if path.as_os_str().is_empty() {
            return true;
        }
        let Ok(file) = file::open_buffered(path) else {
            log(format_args!("GGUF alias: open failed: {}", path.display()));
            return false;
        };
        let mut reader = BufReader::new(file);
        let mut existing: HashSet<(Box<[u8]>, i32, u64)> =
            entries.iter().map(|e| (e.tensor.clone(), e.expert_idx, e.nbytes)).collect();

        let header_line = match ctext::fgets(&mut reader) {
            Ok(Some(line)) => line,
            Ok(None) => {
                log(format_args!("GGUF alias: empty TSV: {}", path.display()));
                return false;
            }
            Err(error) => {
                log(format_args!("GGUF alias: read failed: {}: {error}", path.display()));
                return false;
            }
        };
        let header = ctext::split_tsv(&header_line);
        let column = |name: &[u8]| header.iter().position(|field| *field == name);
        let columns = (
            column(b"tensor"),
            column(b"expert_idx").or_else(|| column(b"expert")),
            column(b"nbytes"),
            column(b"source_path"),
            column(b"offset").or_else(|| column(b"model_offset")).or_else(|| column(b"offset_bytes")),
        );
        let (Some(col_tensor), Some(col_expert), Some(col_nbytes), Some(col_source), Some(col_offset)) = columns else {
            log(format_args!("GGUF alias: required TSV columns missing: {}", path.display()));
            return false;
        };
        let need_cols = [col_tensor, col_expert, col_nbytes, col_source, col_offset].into_iter().max().unwrap() + 1;

        let (mut rows, mut loaded, mut skipped, mut bad_rows) = (0u64, 0u64, 0u64, 0u64);
        loop {
            let line = match ctext::fgets(&mut reader) {
                Ok(Some(line)) => line,
                Ok(None) => break,
                Err(error) => {
                    log(format_args!("GGUF alias: read failed: {}: {error}", path.display()));
                    return false;
                }
            };
            let fields = ctext::split_tsv(&line);
            rows += 1;
            if fields.len() < need_cols {
                bad_rows += 1;
                continue;
            }
            let tensor = fields[col_tensor];
            let source_path = fields[col_source];
            // (int)strtol: the low 32 bits, as the C conversion keeps them.
            let expert_idx = ctext::strtol(fields[col_expert]) as i32;
            let nbytes = ctext::strtoull(fields[col_nbytes]);
            let offset = ctext::strtoull(fields[col_offset]);
            if tensor.is_empty()
                || source_path.is_empty()
                || expert_idx < 0
                || nbytes == 0
                || tensor.len() >= TENSOR_NAME_BYTES
            {
                bad_rows += 1;
                continue;
            }
            let key = (Box::<[u8]>::from(tensor), expert_idx, nbytes);
            if existing.contains(&key) {
                skipped += 1;
                continue;
            }
            let source_idx = match alias_sources.get(source_path) {
                Some(&index) => index,
                None => {
                    let index = self.state.sources.len() as i32;
                    let force = self.force_direct;
                    if !self.open_data_source(path_of(source_path), "GGUF alias", false, force) {
                        return false;
                    }
                    alias_sources.insert(source_path.to_vec(), index);
                    index
                }
            };
            entries.push(Entry { tensor: key.0.clone(), expert_idx, source_idx, offset, nbytes, alias_source: true });
            existing.insert(key);
            loaded += 1;
        }
        log(format_args!(
            "GGUF alias: loaded {loaded} entries from {} sources={} skipped_duplicates={skipped} bad_rows={bad_rows} rows={rows}",
            path.display(),
            alias_sources.len()
        ));
        bad_rows == 0
    }
}

/// Opens one data source into `state` (the legacy per-source open point).
pub fn open_data_source(
    state: &mut ParserState,
    path: &Path,
    label: &str,
    page_cache: bool,
    direct_required: bool,
    hooks: OpenHooks<'_>,
) -> bool {
    Parser { state, hooks, force_direct: false }.open_data_source(path, label, page_cache, direct_required)
}

/// Appends the sources of `desc` to `state` and their entries to `entries`.
pub fn append_sources(
    state: &mut ParserState,
    desc: &Description,
    entries: &mut Vec<Entry>,
    hooks: OpenHooks<'_>,
) -> bool {
    let mut parser = Parser { state, hooks, force_direct: desc.require_direct };
    for path in &desc.packs {
        if !parser.load_pack(path, entries, false) {
            return false;
        }
    }
    let mut aliases = HashMap::new();
    for path in &desc.aliases {
        if !parser.load_alias_tsv(path, entries, &mut aliases) {
            return false;
        }
    }
    for path in &desc.page_cache_packs {
        if !parser.load_pack(path, entries, true) {
            return false;
        }
    }
    true
}

/// Sorts entries by (tensor, expert, size, source). With `replace_duplicates`
/// the entry from the later source wins each key; any duplicate key left is
/// an error.
///
/// The sort is stable, so among identical keys from one source the later
/// record wins a replacement. Upstream's `std::sort` leaves that order unspecified.
pub fn finalize_entries(entries: &mut Vec<Entry>, replace_duplicates: bool) -> bool {
    entries.sort_by(|a, b| a.key().cmp(&b.key()).then(a.source_idx.cmp(&b.source_idx)));
    if replace_duplicates && !entries.is_empty() {
        let mut replaced = 0u64;
        let mut deduped: Vec<Entry> = Vec::with_capacity(entries.len());
        for entry in entries.drain(..) {
            match deduped.last_mut() {
                Some(last) if last.key() == entry.key() => {
                    if entry.source_idx >= last.source_idx {
                        *last = entry;
                    }
                    replaced += 1;
                }
                _ => deduped.push(entry),
            }
        }
        *entries = deduped;
        if replaced > 0 {
            log(format_args!("expert pack: replaced {replaced} duplicate keys with later pack sources"));
        }
    }
    if let Some(pair) = entries.windows(2).find(|pair| pair[0].key() == pair[1].key()) {
        log(format_args!(
            "expert pack: duplicate key across packs: {} expert={} bytes={}",
            String::from_utf8_lossy(&pair[1].tensor),
            pair[1].expert_idx,
            pair[1].nbytes
        ));
        return false;
    }
    true
}
