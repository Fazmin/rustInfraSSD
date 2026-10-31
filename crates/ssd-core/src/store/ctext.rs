// SPDX-License-Identifier: Apache-2.0
//! The C library text handling that `store/parser.cpp` builds on, reproduced
//! exactly so alias TSVs parse the same way: `fgets` into a 4096-byte buffer,
//! C-string truncation at NUL, the tab splitter, and the permissive
//! `strtol`/`strtoull` that ignore trailing text ("4097tail" is 4097).
//!
//! Integer conversions follow LP64 (macOS, Linux), where `long` is 64 bits.

use std::io::{self, BufRead};

/// The `char line[4096]` buffer the parser hands to `fgets`.
pub const LINE_BUFFER: usize = 4096;

/// One `fgets(line, LINE_BUFFER, file)` call: bytes up to and including the
/// next newline, or `LINE_BUFFER - 1` bytes, whichever comes first. `None` at
/// end of file. A longer line arrives as several chunks, as it does in C.
///
/// Unlike C, a read error is returned rather than treated as end of file.
pub fn fgets(reader: &mut impl BufRead) -> io::Result<Option<Vec<u8>>> {
    let cap = LINE_BUFFER - 1;
    let mut line = Vec::new();
    while line.len() < cap {
        let available = match reader.fill_buf() {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        if available.is_empty() {
            break;
        }
        let room = cap - line.len();
        let take = available.iter().take(room).position(|&b| b == b'\n').map_or(room.min(available.len()), |i| i + 1);
        line.extend_from_slice(&available[..take]);
        let ended = line.last() == Some(&b'\n');
        reader.consume(take);
        if ended {
            break;
        }
    }
    Ok(if line.is_empty() { None } else { Some(line) })
}

/// The C string a byte buffer holds: everything before the first NUL.
pub fn c_str(bytes: &[u8]) -> &[u8] {
    bytes.iter().position(|&b| b == 0).map_or(bytes, |end| &bytes[..end])
}

/// `expert_pack_split_tsv_line`: tab-separated fields of one C string. A tab
/// at the very end yields no trailing empty field. The last field loses one
/// trailing `\n` and then one trailing `\r`.
pub fn split_tsv(line: &[u8]) -> Vec<&[u8]> {
    let line = c_str(line);
    let mut fields = Vec::new();
    let mut start = 0;
    while start < line.len() {
        match line[start..].iter().position(|&b| b == b'\t') {
            Some(tab) => {
                fields.push(&line[start..start + tab]);
                start += tab + 1;
            }
            None => {
                fields.push(&line[start..]);
                break;
            }
        }
    }
    if let Some(last) = fields.last_mut() {
        if let Some(rest) = last.strip_suffix(b"\n") {
            *last = rest;
        }
        if let Some(rest) = last.strip_suffix(b"\r") {
            *last = rest;
        }
    }
    fields
}

fn is_c_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

/// Leading whitespace, an optional sign, and the decimal digits that follow.
fn decimal(text: &[u8]) -> (bool, &[u8]) {
    let text = c_str(text);
    let start = text.iter().position(|&b| !is_c_space(b)).unwrap_or(text.len());
    let mut rest = &text[start..];
    let negative = rest.first() == Some(&b'-');
    if matches!(rest.first(), Some(b'+' | b'-')) {
        rest = &rest[1..];
    }
    let digits = rest.iter().position(|b| !b.is_ascii_digit()).unwrap_or(rest.len());
    (negative, &rest[..digits])
}

/// Magnitude of a digit run, or `None` when it exceeds `u64`.
fn magnitude(digits: &[u8]) -> Option<u64> {
    digits.iter().try_fold(0u64, |value, &d| value.checked_mul(10)?.checked_add(u64::from(d - b'0')))
}

/// `strtol(text, nullptr, 10)` with a 64-bit `long`: saturates at
/// `i64::MAX`/`i64::MIN`, and is 0 when no digits follow.
pub fn strtol(text: &[u8]) -> i64 {
    let (negative, digits) = decimal(text);
    match magnitude(digits) {
        Some(m) if negative => {
            if m > i64::MAX as u64 + 1 {
                i64::MIN
            } else {
                (m as i64).wrapping_neg()
            }
        }
        Some(m) => i64::try_from(m).unwrap_or(i64::MAX),
        None if negative => i64::MIN,
        None => i64::MAX,
    }
}

/// `strtoull(text, nullptr, 10)`: `u64::MAX` on overflow, and a leading minus
/// negates the value modulo 2^64.
pub fn strtoull(text: &[u8]) -> u64 {
    let (negative, digits) = decimal(text);
    match magnitude(digits) {
        Some(m) if negative => m.wrapping_neg(),
        Some(m) => m,
        None => u64::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(input: &[u8]) -> Vec<Vec<u8>> {
        let mut reader = io::BufReader::with_capacity(7, input);
        std::iter::from_fn(|| fgets(&mut reader).unwrap()).collect()
    }

    #[test]
    fn fgets_splits_at_newlines_and_at_the_buffer_size() {
        assert_eq!(lines(b"a\nbc\r\nd"), vec![b"a\n".to_vec(), b"bc\r\n".to_vec(), b"d".to_vec()]);
        assert!(lines(b"").is_empty());
        let long = vec![b'x'; LINE_BUFFER + 10];
        let chunks = lines(&[long.as_slice(), b"\n"].concat());
        assert_eq!(chunks.iter().map(Vec::len).collect::<Vec<_>>(), vec![LINE_BUFFER - 1, 11 + 1]);
    }

    #[test]
    fn split_matches_the_c_splitter() {
        assert_eq!(split_tsv(b"a\tb\tc\n"), vec![&b"a"[..], b"b", b"c"]);
        assert_eq!(split_tsv(b"a\t\tb\r\n"), vec![&b"a"[..], b"", b"b"]);
        assert_eq!(split_tsv(b"a\t"), vec![&b"a"[..]]);
        assert_eq!(split_tsv(b"a\t\n"), vec![&b"a"[..], b""]);
        assert_eq!(split_tsv(b"\n"), vec![&b""[..]]);
        assert_eq!(split_tsv(b"a\0\tb"), vec![&b"a"[..]]);
        assert!(split_tsv(b"").is_empty());
    }

    #[test]
    fn integers_parse_like_the_c_library() {
        assert_eq!(strtol(b"4097tail"), 4097);
        assert_eq!(strtol(b"  -12x"), -12);
        assert_eq!(strtol(b"+7"), 7);
        assert_eq!(strtol(b"tail"), 0);
        assert_eq!(strtol(b"99999999999999999999"), i64::MAX);
        assert_eq!(strtol(b"-99999999999999999999"), i64::MIN);
        assert_eq!(strtol(b"-9223372036854775808"), i64::MIN);
        assert_eq!(strtoull(b"18446744073709551615"), u64::MAX);
        assert_eq!(strtoull(b"18446744073709551616"), u64::MAX);
        assert_eq!(strtoull(b"-1"), u64::MAX);
        assert_eq!(strtoull(b"64tail"), 64);
        assert_eq!(strtoull(b""), 0);
    }
}
