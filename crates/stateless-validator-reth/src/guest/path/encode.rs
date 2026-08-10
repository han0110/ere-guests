//! RLP-encoding the nodes an update builds, straight into the scratch they live in.

use alloy_primitives::{B256, keccak256};
use alloy_rlp::Encodable;

use crate::guest::path::{
    error::Error,
    nibbles::{MAX_ENCODED_PATH, MAX_NIBBLES, hex_prefix_encode, hex_prefix_from_key},
    node::{DIGEST_LEN, MAX_BRANCH_RLP, MAX_LEAF_VALUE},
    stack::DynStack,
};

/// RLP-encodes a leaf holding `value` at `path`.
pub(super) fn encode_leaf<'s>(path: &[u8], value: &[u8], stack: &mut DynStack<'s>) -> &'s [u8] {
    let mut buf = [0; MAX_ENCODED_PATH];
    encode_leaf_encoded(hex_prefix_encode(path, true, &mut buf), value, stack)
}

/// RLP-encodes a leaf holding `value` at the nibbles `key` carries from `depth`, which is where a
/// leaf a descent writes takes its path from.
pub(super) fn encode_leaf_from_key<'s>(
    key: &B256,
    depth: usize,
    value: &[u8],
    stack: &mut DynStack<'s>,
) -> &'s [u8] {
    let mut buf = [0; MAX_ENCODED_PATH];
    let path = hex_prefix_from_key(key, depth, MAX_NIBBLES, true, &mut buf);
    encode_leaf_encoded(path, value, stack)
}

/// RLP-encodes a leaf whose path is already hex-prefix encoded.
pub(super) fn encode_leaf_encoded<'s>(
    path: &[u8],
    value: &[u8],
    stack: &mut DynStack<'s>,
) -> &'s [u8] {
    let (out, pos) = alloc_list(path.length() + value.length(), stack);
    let pos = write_string(out, pos, path);
    write_string(out, pos, value);
    out
}

/// RLP-encodes an extension at `path` referencing its child through the whole RLP item `item`.
pub(super) fn encode_extension<'s>(path: &[u8], item: &[u8], stack: &mut DynStack<'s>) -> &'s [u8] {
    let mut buf = [0; MAX_ENCODED_PATH];
    encode_extension_encoded(hex_prefix_encode(path, false, &mut buf), item, stack)
}

/// RLP-encodes an extension whose path is already hex-prefix encoded, which is every extension a
/// descent leaves the path of alone.
pub(super) fn encode_extension_encoded<'s>(
    path: &[u8],
    item: &[u8],
    stack: &mut DynStack<'s>,
) -> &'s [u8] {
    let (out, pos) = alloc_list(path.length() + item.len(), stack);
    let pos = write_string(out, pos, path);
    out[pos..pos + item.len()].copy_from_slice(item);
    out
}

/// RLP-encodes a branch from its child items, leaving the value slot a trie of fixed-length keys
/// never fills empty.
pub(super) fn encode_branch<'s>(children: &[&[u8]; 16], stack: &mut DynStack<'s>) -> &'s [u8] {
    let payload_length = children.iter().map(|child| child.len()).sum::<usize>() + 1;
    let (out, mut pos) = alloc_list(payload_length, stack);
    for child in children {
        out[pos..pos + child.len()].copy_from_slice(child);
        pos += child.len();
    }
    out[pos] = alloy_rlp::EMPTY_STRING_CODE;
    out
}

/// Re-encodes a branch with the `start..end` bytes of its payload, one child, replaced by `item`.
///
/// The children on either side keep the bytes they arrived as, so replacing one costs two copies
/// rather than reassembling all sixteen.
///
/// A branch of the canonical trie holds seventeen items no wider than a reference, so what this
/// comes to is a branch of that trie too. Checking the length it comes to is what the frame this
/// builds in is sized against, and costs one comparison rather than a walk over the items on either
/// side.
pub(super) fn splice_branch<'s>(
    payload: &[u8],
    start: usize,
    end: usize,
    item: &[u8],
    stack: &mut DynStack<'s>,
) -> Result<&'s [u8], Error> {
    let payload_length = payload.len() - (end - start) + item.len();
    if alloy_rlp::length_of_length(payload_length) + payload_length > MAX_BRANCH_RLP {
        return Err(Error::OversizedNode);
    }
    let (out, pos) = alloc_list(payload_length, stack);
    out[pos..pos + start].copy_from_slice(&payload[..start]);
    let pos = pos + start;
    out[pos..pos + item.len()].copy_from_slice(item);
    let pos = pos + item.len();
    out[pos..].copy_from_slice(&payload[end..]);
    Ok(out)
}

/// RLP-encodes `digest` as the 33-byte item a parent references a hashed node through.
pub(super) fn encode_digest<'s>(digest: B256, stack: &mut DynStack<'s>) -> &'s [u8] {
    let out = stack.alloc(1 + DIGEST_LEN);
    out[0] = alloy_rlp::EMPTY_STRING_CODE + DIGEST_LEN as u8;
    out[1..].copy_from_slice(digest.as_slice());
    out
}

/// RLP-encodes a value straight into the scratch.
pub(super) fn encode_value<'s>(value: impl Encodable, stack: &mut DynStack<'s>) -> &'s [u8] {
    debug_assert!(
        value.length() <= MAX_LEAF_VALUE,
        "a leaf carries no more than an account"
    );
    let out = stack.alloc(value.length());
    let mut cursor = &mut *out;
    value.encode(&mut cursor);
    debug_assert!(cursor.is_empty(), "an encoding fills the length it reports");
    out
}

/// The item a parent references `node_rlp` through, which is the node itself when short enough to
/// sit in place and the RLP of its hash otherwise.
pub(super) fn reference<'s>(node_rlp: &'s [u8], stack: &mut DynStack<'s>) -> &'s [u8] {
    if node_rlp.len() < DIGEST_LEN {
        return node_rlp;
    }
    encode_digest(keccak256(node_rlp), stack)
}

/// A copy of the item a parent references a node through, taken because a frame that lends its
/// scratch to a child takes those bytes back when the loan ends, so the item has to leave the loan
/// before it does. [`reference`] is that same item for a node the frame built in its own scratch,
/// which needs no copy.
///
/// Carrying it out is what the borrow checker asks for, and it asks at every such site rather than
/// wherever anyone remembered to look.
pub(super) enum Carried {
    /// The node is shorter than a digest, so it sits in its parent's place and the item is the node
    /// itself.
    Inline([u8; DIGEST_LEN], usize),
    /// The node is referenced by hash.
    Hashed(B256),
}

impl Carried {
    /// The item `node_rlp` is referenced through.
    pub(super) fn of(node_rlp: &[u8]) -> Self {
        if node_rlp.len() < DIGEST_LEN {
            let mut inline = [0; DIGEST_LEN];
            inline[..node_rlp.len()].copy_from_slice(node_rlp);
            return Self::Inline(inline, node_rlp.len());
        }
        Self::Hashed(keccak256(node_rlp))
    }

    /// Writes the item into the scratch.
    pub(super) fn write<'s>(self, stack: &mut DynStack<'s>) -> &'s [u8] {
        match self {
            Self::Inline(inline, len) => stack.alloc_copy(&inline[..len]),
            Self::Hashed(digest) => encode_digest(digest, stack),
        }
    }
}

/// Allocates a list of `payload_length` bytes in the scratch and returns it with the position its
/// payload starts at.
fn alloc_list<'s>(payload_length: usize, stack: &mut DynStack<'s>) -> (&'s mut [u8], usize) {
    let len = alloy_rlp::length_of_length(payload_length) + payload_length;
    let out = stack.alloc(len);
    let pos = write_header(out, 0, alloy_rlp::EMPTY_LIST_CODE, payload_length);
    (out, pos)
}

/// Writes an RLP header with the given base code at `pos` and returns the position after it.
fn write_header(out: &mut [u8], pos: usize, base_code: u8, payload_length: usize) -> usize {
    if payload_length < 56 {
        out[pos] = base_code + payload_length as u8;
        return pos + 1;
    }
    let len_be = payload_length.to_be_bytes();
    let num_len_bytes = alloy_rlp::length_of_length(payload_length) - 1;
    out[pos] = base_code + 55 + num_len_bytes as u8;
    out[pos + 1..pos + 1 + num_len_bytes].copy_from_slice(&len_be[len_be.len() - num_len_bytes..]);
    pos + 1 + num_len_bytes
}

/// Writes `bytes` as an RLP string at `pos` and returns the position after it.
fn write_string(out: &mut [u8], pos: usize, bytes: &[u8]) -> usize {
    if let [byte] = bytes
        && *byte < alloy_rlp::EMPTY_STRING_CODE
    {
        out[pos] = *byte;
        return pos + 1;
    }
    let pos = write_header(out, pos, alloy_rlp::EMPTY_STRING_CODE, bytes.len());
    out[pos..pos + bytes.len()].copy_from_slice(bytes);
    pos + bytes.len()
}
