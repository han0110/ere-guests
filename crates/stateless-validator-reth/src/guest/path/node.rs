//! Decoding a trie node no further than a walk or an update needs it.
//!
//! Paths stay hex-prefix encoded and children stay whole RLP items, both borrowing the node's own
//! encoding, so re-encoding a node copies the parts it does not change.

use crate::guest::path::{
    error::Error,
    nibbles::{HEX_PREFIX_FLAG_LEAF, HEX_PREFIX_FLAG_ODD},
};

/// Length of a node reference that is a Keccak digest rather than the node's own encoding.
pub(super) const DIGEST_LEN: usize = 32;

/// Longest item a parent references a child through, which is a digest behind its header. A node
/// short enough to sit in its parent's place stays under it, so one bound covers both forms.
pub(super) const MAX_ITEM: usize = 1 + DIGEST_LEN;

/// Longest RLP encoding of a branch, which is sixteen items and an empty value slot behind a
/// three-byte header.
pub(super) const MAX_BRANCH_RLP: usize = 3 + 16 * MAX_ITEM + 1;

/// Longest value a leaf carries, which is the RLP of an account holding a 9-byte nonce, a 33-byte
/// balance and two 33-byte hashes. A storage slot is a 32-byte integer and stays well under it, so
/// one bound covers both tries.
pub(super) const MAX_LEAF_VALUE: usize = 2 + 9 + 33 + 2 * MAX_ITEM;

/// RLP of the empty node, which is also how a parent references an absent child.
pub(super) const EMPTY_NODE: &[u8] = &[alloy_rlp::EMPTY_STRING_CODE];

/// A decoded trie node. Paths stay hex-prefix encoded and children stay whole RLP items, both
/// borrowing the node's own encoding, so a re-encoding can copy the parts it does not change.
#[derive(Clone, Copy)]
pub(super) enum Node<'a> {
    Leaf { path: &'a [u8], value: &'a [u8] },
    Extension { path: &'a [u8], child: &'a [u8] },
    Branch(Branch<'a>),
}

/// A branch and the two children [`decode_node`] split off to recognize it, so a lookup resumes
/// past them rather than walking the payload from the front again.
#[derive(Clone, Copy)]
pub(super) struct Branch<'a> {
    /// Every child item, unparsed.
    pub(super) payload: &'a [u8],
    /// The two children a third item told apart from the path and value of a short node.
    head: [&'a [u8]; 2],
}

impl<'a> Branch<'a> {
    /// The range the child at `nibble` occupies in the payload.
    ///
    /// The items on either side are not inspected, since what re-encoding the branch around this
    /// one costs is settled by the length that re-encoding comes to rather than by the items it
    /// copies.
    pub(super) fn child_span(&self, nibble: u8) -> Result<(usize, usize), Error> {
        let [first, second] = self.head;
        if nibble == 0 {
            return Ok((0, first.len()));
        }
        if nibble == 1 {
            return Ok((first.len(), first.len() + second.len()));
        }
        let mut start = first.len() + second.len();
        let mut rest = &self.payload[start..];
        for _ in 2..nibble {
            start += split_item(&mut rest)?.len();
        }
        Ok((start, start + split_item(&mut rest)?.len()))
    }

    /// The item the branch holds at `nibble`.
    pub(super) fn child(&self, nibble: u8) -> Result<&'a [u8], Error> {
        let (start, end) = self.child_span(nibble)?;
        Ok(&self.payload[start..end])
    }
}

/// Decodes a node far enough to tell which of the three it is, which for a branch is two item
/// headers. Its children stay in the payload, so an update that replaces one never pays to
/// materialize the fifteen it leaves alone.
///
/// What a short node holds is checked against the widest the canonical trie has, since re-encoding
/// one is sized against those widths and a node reached by the hash its parent commits to is a node
/// only a chain the guest is not validating could hold.
pub(super) fn decode_node(node_rlp: &[u8]) -> Result<Node<'_>, Error> {
    let payload = list_payload(node_rlp)?;
    let mut rest = payload;
    let first = split_item(&mut rest)?;
    let second = split_item(&mut rest)?;

    if !rest.is_empty() {
        return Ok(Node::Branch(Branch {
            payload,
            head: [first, second],
        }));
    }

    let mut path = first;
    let path = alloy_rlp::Header::decode_bytes(&mut path, false)?;
    let &flags = path.first().ok_or(Error::MalformedPath)?;
    if flags & HEX_PREFIX_FLAG_LEAF == 0 {
        // An extension holding no nibble at all would leave a descent at the depth it arrived on,
        // which the trie has no such node for and which the scratch is not sized for.
        if path.len() == 1 && flags & HEX_PREFIX_FLAG_ODD == 0 {
            return Err(Error::MalformedPath);
        }
        if second.len() > MAX_ITEM {
            return Err(Error::OversizedItem);
        }
        return Ok(Node::Extension {
            path,
            child: second,
        });
    }
    let mut value = second;
    let value = alloy_rlp::Header::decode_bytes(&mut value, false)?;
    if value.len() > MAX_LEAF_VALUE {
        return Err(Error::OversizedValue);
    }
    Ok(Node::Leaf { path, value })
}

/// The child items of a branch, given its payload.
pub(super) fn branch_children(payload: &[u8]) -> Result<[&[u8]; 16], Error> {
    let mut rest = payload;
    let mut children = [EMPTY_NODE; 16];
    for child in &mut children {
        *child = split_item(&mut rest)?;
        if child.len() > MAX_ITEM {
            return Err(Error::OversizedItem);
        }
    }
    if !is_empty_node(split_item(&mut rest)?) {
        return Err(Error::ValueInBranch);
    }
    if !rest.is_empty() {
        return Err(Error::Rlp(alloy_rlp::Error::UnexpectedLength));
    }
    Ok(children)
}

/// How many children a branch still holds, which only a deletion can bring below two.
pub(super) enum Filled {
    None,
    One(u8),
    Many,
}

/// Which children of a branch are still filled.
pub(super) fn filled_children(children: &[&[u8]; 16]) -> Filled {
    let mut filled = Filled::None;
    for (nibble, child) in children.iter().enumerate() {
        if is_empty_node(child) {
            continue;
        }
        match filled {
            Filled::None => filled = Filled::One(nibble as u8),
            _ => return Filled::Many,
        }
    }
    filled
}

/// Whether an item is the RLP of an empty node, which is how a parent references an absent child.
///
/// Written as a pattern match because comparing against [`EMPTY_NODE`] with `==` compiles to a
/// `memcmp` call whose overhead dwarfs this one-byte check, and every branch child pays it.
pub(super) fn is_empty_node(item: &[u8]) -> bool {
    matches!(item, [byte] if *byte == alloy_rlp::EMPTY_STRING_CODE)
}

/// Splits `len` bytes off the front of `buf`, leaving `buf` on what follows them.
pub(super) fn split_payload<'a>(buf: &mut &'a [u8], len: usize) -> Result<&'a [u8], Error> {
    if buf.len() < len {
        return Err(Error::Rlp(alloy_rlp::Error::InputTooShort));
    }
    let (payload, rest) = buf.split_at(len);
    *buf = rest;
    Ok(payload)
}

/// Splits the RLP item at the front of `buf` off whole, header included.
///
/// Almost every item a node holds is an empty child or a digest, both of which their first byte
/// tells apart and gives the length of, so the walk over a branch reads a byte rather than decoding
/// a header sixteen times over.
fn split_item<'a>(buf: &mut &'a [u8]) -> Result<&'a [u8], Error> {
    let len = match buf.first() {
        Some(&alloy_rlp::EMPTY_STRING_CODE) => 1,
        Some(&byte) if byte == alloy_rlp::EMPTY_STRING_CODE + DIGEST_LEN as u8 => 1 + DIGEST_LEN,
        _ => {
            let start = *buf;
            let header = alloy_rlp::Header::decode(buf)?;
            split_payload(buf, header.payload_length)?;
            return Ok(&start[..start.len() - buf.len()]);
        }
    };
    split_payload(buf, len)
}

/// The payload of the list `node_rlp` encodes.
fn list_payload(node_rlp: &[u8]) -> Result<&[u8], Error> {
    let mut buf = node_rlp;
    let header = alloy_rlp::Header::decode(&mut buf)?;
    if !header.list {
        return Err(Error::Rlp(alloy_rlp::Error::UnexpectedString));
    }
    split_payload(&mut buf, header.payload_length)
}
