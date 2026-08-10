//! Keys and node paths as nibble sequences, and the hex-prefix form a node encodes a path in.

use alloy_primitives::B256;

use crate::guest::path::{error::Error, node::DIGEST_LEN};

/// Nibbles in a key, which is always a 32-byte hash here.
pub(super) const MAX_NIBBLES: usize = 2 * DIGEST_LEN;

/// Longest hex-prefix encoding of a path, which is a flag byte plus a whole key.
pub(super) const MAX_ENCODED_PATH: usize = 1 + DIGEST_LEN;

/// Path holds an odd number of nibbles, so the low nibble of the flag byte is data.
pub(super) const HEX_PREFIX_FLAG_ODD: u8 = 0x10;
/// Node is a leaf rather than an extension.
pub(super) const HEX_PREFIX_FLAG_LEAF: u8 = 0x20;

/// Expands a key into its nibbles, most significant first.
pub(super) fn key_nibbles(key: &B256) -> [u8; MAX_NIBBLES] {
    let mut nibbles = [0; MAX_NIBBLES];
    let (pairs, _) = nibbles.as_chunks_mut::<2>();
    for (byte, pair) in key.iter().zip(pairs) {
        pair[0] = byte >> 4;
        pair[1] = byte & 0x0f;
    }
    nibbles
}

/// The nibble a key holds at `depth`.
pub(super) fn nibble_at(key: &B256, depth: usize) -> u8 {
    let byte = key[depth / 2];
    if depth.is_multiple_of(2) {
        byte >> 4
    } else {
        byte & 0x0f
    }
}

/// The nibbles `encoded` holds where they are the ones `key` carries from `depth`, or `None` where
/// the two part.
///
/// A key stays packed. Where what is left of the path starts on a whole byte of it, which is every
/// leaf and every extension of even length, the two are compared whole rather than expanded into
/// nibbles a walk of a level or two would never look at.
pub(super) fn path_from(key: &B256, depth: usize, encoded: &[u8]) -> Option<usize> {
    let (&flags, payload) = encoded.split_first()?;
    let is_odd = flags & HEX_PREFIX_FLAG_ODD != 0;
    let len = 2 * payload.len() + usize::from(is_odd);
    // A path running past the last nibble of a key is one no key reaches the end of, and is what
    // holds every index below in range.
    if depth + len > MAX_NIBBLES {
        return None;
    }

    let mut taken = 0;
    if is_odd {
        if flags & 0x0f != nibble_at(key, depth) {
            return None;
        }
        taken = 1;
    }
    if (depth + taken).is_multiple_of(2) {
        let start = (depth + taken) / 2;
        return (payload == &key[start..start + payload.len()]).then_some(len);
    }
    payload
        .iter()
        .enumerate()
        .all(|(step, &byte)| {
            let at = depth + taken + 2 * step;
            byte >> 4 == nibble_at(key, at) && byte & 0x0f == nibble_at(key, at + 1)
        })
        .then_some(len)
}

/// The hex-prefix encoding of the nibbles `key` carries from `from` up to `to`.
///
/// A key stays packed. Where what is left to encode starts on a whole byte of it, which is every
/// leaf written at an even depth, the bytes are copied rather than expanded into nibbles this would
/// only pack again.
pub(super) fn hex_prefix_from_key<'o>(
    key: &B256,
    from: usize,
    to: usize,
    is_leaf: bool,
    out: &'o mut [u8; MAX_ENCODED_PATH],
) -> &'o [u8] {
    let len = to - from;
    let out = &mut out[..1 + len / 2];

    let mut flags = if is_leaf { HEX_PREFIX_FLAG_LEAF } else { 0 };
    let start = if len.is_multiple_of(2) {
        from
    } else {
        flags |= HEX_PREFIX_FLAG_ODD | nibble_at(key, from);
        from + 1
    };
    out[0] = flags;
    if start.is_multiple_of(2) {
        out[1..].copy_from_slice(&key[start / 2..to / 2]);
    } else {
        for (step, byte) in out[1..].iter_mut().enumerate() {
            let at = start + 2 * step;
            *byte = (nibble_at(key, at) << 4) | nibble_at(key, at + 1);
        }
    }
    out
}

/// Length of the common prefix of two nibble sequences.
pub(super) fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

/// Expands a hex-prefix encoded path into `out` and returns its nibbles.
pub(super) fn hex_prefix_decode<'o>(
    encoded: &[u8],
    out: &'o mut [u8; MAX_NIBBLES],
) -> Result<&'o [u8], Error> {
    let (&flags, rest) = encoded.split_first().ok_or(Error::MalformedPath)?;
    let is_odd = flags & HEX_PREFIX_FLAG_ODD != 0;
    let len = 2 * rest.len() + usize::from(is_odd);
    if len > MAX_NIBBLES {
        return Err(Error::MalformedPath);
    }

    let start = usize::from(is_odd);
    if is_odd {
        out[0] = flags & 0x0f;
    }
    let (pairs, _) = out[start..len].as_chunks_mut::<2>();
    for (byte, pair) in rest.iter().zip(pairs) {
        pair[0] = byte >> 4;
        pair[1] = byte & 0x0f;
    }
    Ok(&out[..len])
}

/// Encodes `nibbles` into hex-prefix form in `out` and returns the encoding.
pub(super) fn hex_prefix_encode<'o>(
    nibbles: &[u8],
    is_leaf: bool,
    out: &'o mut [u8; MAX_ENCODED_PATH],
) -> &'o [u8] {
    let is_odd = !nibbles.len().is_multiple_of(2);
    let out = &mut out[..1 + nibbles.len() / 2];

    let mut flags = if is_leaf { HEX_PREFIX_FLAG_LEAF } else { 0 };
    let rest = if is_odd {
        flags |= HEX_PREFIX_FLAG_ODD | nibbles[0];
        &nibbles[1..]
    } else {
        nibbles
    };
    out[0] = flags;
    let (pairs, _) = rest.as_chunks::<2>();
    for (byte, pair) in out[1..].iter_mut().zip(pairs) {
        *byte = (pair[0] << 4) | pair[1];
    }
    out
}

/// Joins a parent path and its child's into `out`.
pub(super) fn merge_paths<'o>(
    parent: &[u8],
    child: &[u8],
    out: &'o mut [u8; MAX_NIBBLES],
) -> Result<&'o [u8], Error> {
    let len = parent.len() + child.len();
    if len > MAX_NIBBLES {
        return Err(Error::MalformedPath);
    }
    out[..parent.len()].copy_from_slice(parent);
    out[parent.len()..len].copy_from_slice(child);
    Ok(&out[..len])
}
