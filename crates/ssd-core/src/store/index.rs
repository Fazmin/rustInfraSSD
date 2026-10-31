// SPDX-License-Identifier: Apache-2.0
//! Immutable expert source indexes. Port of `ggml-ssd/store/legacy-index.h`
//! and `index.{h,cpp}`.

use super::parser::{self, Description, Entry, IoBackend, OpenHooks, ParserState};
use crate::{Error, Result};
use ssd_platform::file;
use std::fs::File;
use std::os::fd::{AsFd, BorrowedFd};
use std::path::Path;

/// The parts of a file identity that pin one source: device, file id, size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SourceIdentity {
    pub device: u64,
    pub inode: u64,
    /// The upper half of a 128-bit file id (Windows).
    pub inode_hi: u64,
    pub size: u64,
}

fn identity_of(file: &File) -> SourceIdentity {
    file::identity(file.as_fd())
        .map(|id| SourceIdentity { device: id.device, inode: id.id_lo, inode_hi: id.id_hi, size: id.size })
        .unwrap_or_default()
}

/// Where one entry's logical bytes live.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Extent {
    pub source_id: i32,
    pub offset: u64,
    pub logical_bytes: u64,
    pub alias: bool,
    /// Default unless the lookup asked for the source's identity.
    pub file: SourceIdentity,
}

/// The parsed, sorted, immutable index (`c0::expert_index`). It owns every
/// source handle; borrowed descriptors are valid until it is dropped.
#[derive(Debug)]
pub struct ExpertIndex {
    state: ParserState,
}

impl ExpertIndex {
    /// Parses `desc`; `None` when any source fails, no entry loads, or a
    /// duplicate key remains.
    pub fn open(desc: &Description) -> Option<Self> {
        let mut state = ParserState { io_backend: desc.io_backend, ..ParserState::default() };
        let mut entries = Vec::new();
        if !parser::append_sources(&mut state, desc, &mut entries, OpenHooks::default()) || entries.is_empty() {
            return None;
        }
        if !parser::finalize_entries(&mut entries, desc.replace_duplicates || !desc.page_cache_packs.is_empty()) {
            return None;
        }
        state.entries = entries;
        Some(Self { state })
    }

    /// The extent of one key. `None` for an absent key or a logical range
    /// beyond the opened file. Pack v1 stores no tensor type; the caller's
    /// tensor context supplies it.
    pub fn locate(&self, name: &[u8], expert: i32, bytes: u64, include_identity: bool) -> Option<Extent> {
        if name.is_empty() || expert < 0 || bytes == 0 {
            return None;
        }
        let entries = &self.state.entries;
        let lo = entries.partition_point(|e| (&*e.tensor, e.expert_idx, e.nbytes) < (name, expert, bytes));
        let item = entries.get(lo)?;
        if &*item.tensor != name || item.expert_idx != expert || item.nbytes != bytes {
            return None;
        }
        let source = self.state.sources.get(usize::try_from(item.source_idx).ok()?)?;
        if item.offset > source.file_size || bytes > source.file_size - item.offset {
            return None;
        }
        Some(Extent {
            source_id: item.source_idx,
            offset: item.offset,
            logical_bytes: item.nbytes,
            alias: item.alias_source,
            file: if include_identity { identity_of(&source.file) } else { SourceIdentity::default() },
        })
    }

    /// The buffered descriptor of a source.
    pub fn borrowed_fd(&self, source_id: i32) -> Option<BorrowedFd<'_>> {
        let source = self.state.sources.get(usize::try_from(source_id).ok()?)?;
        Some(source.file.as_fd())
    }

    /// The page-cache-bypassing descriptor of a source, if it has one.
    pub fn borrowed_direct_fd(&self, source_id: i32) -> Option<BorrowedFd<'_>> {
        let source = self.state.sources.get(usize::try_from(source_id).ok()?)?;
        source.direct.as_ref().map(AsFd::as_fd)
    }

    pub fn source_count(&self) -> usize {
        self.state.sources.len()
    }

    pub fn entries(&self) -> &[Entry] {
        &self.state.entries
    }

    pub fn source_identity(&self, index: usize) -> SourceIdentity {
        identity_of(&self.state.sources[index].file)
    }
}

/// A context-owned index whose every source has a direct descriptor
/// (`ssd::store::source_index`). Executors borrow it through [`SourceIndex::resolve`]
/// and [`SourceIndex::dense_blocks`].
#[derive(Debug)]
pub struct SourceIndex {
    index: ExpertIndex,
}

impl SourceIndex {
    /// Opens v1 packs and alias TSVs (chosen by their leading bytes) with
    /// required direct reads, and checks every published range against its file.
    pub fn open<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        if paths.is_empty() {
            return Err(Error::invalid("expert.source requires an index path"));
        }
        // The index opens direct handles; the CPU executor owns its read ring.
        let mut desc = Description { io_backend: IoBackend::Direct, require_direct: true, ..Description::default() };
        for path in paths {
            let path = path.as_ref();
            let (magic, count) = parser::probe_magic(path)
                .map_err(|_| Error::io(format!("cannot open expert index: {}", path.display())))?;
            if count >= 13 && magic[..13] == parser::PACK_MAGIC[..] {
                desc.packs.push(path.to_path_buf());
            } else if count >= 4 && &magic[..4] == b"GGUF" {
                return Err(Error::invalid(format!(
                    "expert.source takes an alias TSV for direct GGUF reads: {}",
                    path.display()
                )));
            } else {
                desc.aliases.push(path.to_path_buf());
            }
        }
        let index = ExpertIndex::open(&desc).ok_or_else(|| Error::invalid("expert source index validation failed"))?;
        // Buffered metadata and direct payload handles must identify the same
        // file even if a path was replaced between the two opens.
        let mut identities = Vec::with_capacity(index.source_count());
        for (i, source) in index.state.sources.iter().enumerate() {
            let metadata = index.source_identity(i);
            let same =
                source.direct.as_ref().and_then(|direct| file::identity(direct.as_fd()).ok()).is_some_and(|direct| {
                    metadata.device == direct.device
                        && metadata.inode == direct.id_lo
                        && metadata.inode_hi == direct.id_hi
                        && metadata.size == direct.size
                });
            if !same {
                return Err(Error::io("direct expert source identity differs from metadata handle"));
            }
            identities.push(metadata);
        }
        for entry in index.entries() {
            let in_range = usize::try_from(entry.source_idx)
                .ok()
                .and_then(|i| identities.get(i))
                .is_some_and(|id| entry.offset <= id.size && entry.nbytes <= id.size - entry.offset);
            if !in_range {
                return Err(Error::invalid("expert index range exceeds its source file"));
            }
        }
        Ok(Self { index })
    }

    /// The direct descriptor, offset and logical size of expert `expert` of
    /// `experts` in tensor `name`, whose bytes the caller expects to be `bytes`.
    pub fn resolve(
        &self,
        name: &[u8],
        expert: i32,
        bytes: usize,
        experts: i64,
    ) -> Option<(BorrowedFd<'_>, u64, usize)> {
        if expert < 0 || experts <= 0 || i64::from(expert) >= experts || bytes == 0 {
            return None;
        }
        let extent = self.index.locate(name, expert, bytes as u64, false)?;
        let direct = self.index.borrowed_direct_fd(extent.source_id)?;
        if !file::is_direct(direct) {
            return None;
        }
        Some((direct, extent.offset, usize::try_from(extent.logical_bytes).ok()?))
    }

    /// For a dense tensor stored as blocks: the block count and block size when
    /// `name` has exactly blocks 0..N-1 of equal size summing to `bytes`, and no
    /// other entries.
    pub fn dense_blocks(&self, name: &[u8], bytes: usize) -> Option<(usize, usize)> {
        if name.is_empty() || bytes == 0 {
            return None;
        }
        let entries = self.index.entries();
        let first_at = entries.partition_point(|e| &*e.tensor < name);
        let first = entries.get(first_at)?;
        if &*first.tensor != name || first.expert_idx != 0 || first.nbytes == 0 {
            return None;
        }
        let block = usize::try_from(first.nbytes).ok()?;
        if bytes % block != 0 {
            return None;
        }
        let count = bytes / block;
        let rest = &entries[first_at..];
        if count == 0 || count > i32::MAX as usize || count > rest.len() {
            return None;
        }
        let exact = rest[..count]
            .iter()
            .enumerate()
            .all(|(i, e)| &*e.tensor == name && e.expert_idx == i as i32 && e.nbytes == first.nbytes);
        if !exact || rest.get(count).is_some_and(|next| &*next.tensor == name) {
            return None;
        }
        Some((count, block))
    }

    /// The underlying parsed index.
    pub fn expert_index(&self) -> &ExpertIndex {
        &self.index
    }
}
