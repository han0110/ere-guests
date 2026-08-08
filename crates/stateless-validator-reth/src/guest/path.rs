//! [`PathState`], a [`StatelessTrie`] that walks the witness where it lies.
//!
//! Nothing is materialized. The witness is indexed once by the Keccak hash each node is referenced
//! through, reads walk that index from the root decoding nodes as they go, and recomputing the root
//! descends once over the whole change set so every node it touches is decoded and re-encoded
//! exactly once.
//!
//! Looking a node up by the hash its parent commits to is what verifies it, so a node is trusted
//! precisely when a walk reaches it, and a reference the witness has no node for is a witness too
//! incomplete to validate against. That also bounds the shapes this has to handle to the ones the
//! chain holds, with paths of at most [`MAX_NIBBLES`] nibbles, empty branch value slots because
//! every key is a hash of one length, and no empty extension path. Every malformed encoding is an
//! error rather than a panic, so a bad witness can cost a block its validation but never its
//! correctness.
//!
//! Modelled on the trie zesu proves its stateless guest with, `src/stateless/mpt` of
//! <https://github.com/eth-act/zesu>, whose `verifyProofIndexed`, `batchUpdateIndexed` and
//! `updNodeExImpl` this mirrors.

use alloc::{boxed::Box, vec::Vec};
use core::cell::RefCell;

use alloy_primitives::{
    Address, B256, Bytes, KECCAK256_EMPTY, U256, keccak256,
    map::{B256IndexMap, B256Map},
};
use alloy_rlp::{Decodable, Encodable};
use alloy_rpc_types_debug::ExecutionWitness;
use alloy_trie::{EMPTY_ROOT_HASH, TrieAccount};
use bumpalo::Bump;
use reth_trie_common::{HashedPostState, HashedStorageSorted};
use reth_tries::{StatelessTrie, StatelessTrieError, WitnessDbError};

/// Length of a node reference that is a Keccak digest rather than the node's own encoding.
const DIGEST_LEN: usize = 32;

/// Nibbles in a key, which is always a 32-byte hash here.
const MAX_NIBBLES: usize = 2 * DIGEST_LEN;

/// Longest hex-prefix encoding of a path, which is a flag byte plus a whole key.
const MAX_ENCODED_PATH: usize = 1 + DIGEST_LEN;

/// Longest RLP encoding of a [`TrieAccount`], which a 9-byte nonce, a 33-byte balance and two
/// 33-byte hashes hold to 110.
const MAX_ACCOUNT_RLP: usize = 128;

/// Longest RLP encoding of a storage value, which is a 32-byte integer behind its header.
const MAX_STORAGE_RLP: usize = 1 + DIGEST_LEN;

/// RLP of the empty node, which is also how a parent references an absent child.
const EMPTY_NODE: &[u8] = &[alloy_rlp::EMPTY_STRING_CODE];

/// One key's new value, or `None` to remove it.
type Change = (B256, Option<&'static [u8]>);

/// The Ethereum world state over the witness bytes, walked rather than built.
#[derive(Debug)]
pub(crate) struct PathState {
    /// Trie nodes by the Keccak hash they are referenced through, from the witness and from
    /// whatever recomputing the root builds on top of it.
    nodes: B256Map<&'static [u8]>,
    /// Arena the nodes built while recomputing the root live in.
    bump: &'static Bump,
    /// Root every read is anchored at, which the parent block header commits to.
    root: B256,
    /// Accounts the state trie has already been walked for, holding the value they had before
    /// execution. Execution reads an account and then its storage, which would otherwise walk the
    /// state trie twice for every account whose storage a block touches.
    accounts: RefCell<B256Map<Option<TrieAccount>>>,
}

impl StatelessTrie for PathState {
    fn new(
        witness: ExecutionWitness,
        pre_state_root: B256,
    ) -> Result<(Self, B256IndexMap<Bytes>), StatelessTrieError> {
        // The nodes outlive the index so it holds borrows of them rather than a second handle on
        // each, halving what a lookup has to load. Nothing is resolved here, since a walk verifies
        // the nodes it reaches by the hashes their parents commit to, so a node the block never
        // reads is a node it never has to trust.
        let nodes: &'static [Bytes] = Box::leak(witness.state.into_boxed_slice());
        let nodes = nodes
            .iter()
            .map(|rlp| (keccak256(rlp), rlp.as_ref()))
            .collect();
        let bump: &'static Bump = Box::leak(Box::new(Bump::new()));

        let bytecode = witness
            .codes
            .into_iter()
            .map(|code| (keccak256(&code), code))
            .collect();

        Ok((
            Self {
                nodes,
                bump,
                root: pre_state_root,
                accounts: RefCell::new(B256Map::default()),
            },
            bytecode,
        ))
    }

    fn account(&self, address: Address) -> Result<Option<TrieAccount>, WitnessDbError> {
        Ok(self.account_at(keccak256(address))?)
    }

    fn storage(&self, address: Address, slot: U256) -> Result<U256, WitnessDbError> {
        let Some(account) = self.account_at(keccak256(address))? else {
            return Ok(U256::ZERO);
        };
        Ok(
            match self.get(account.storage_root, &keccak256(B256::from(slot)))? {
                Some(mut value) => U256::decode(&mut value)?,
                None => U256::ZERO,
            },
        )
    }

    fn calculate_state_root(&mut self, state: HashedPostState) -> Result<B256, StatelessTrieError> {
        let state = state.into_sorted();

        // Every account whose storage changed also has its leaf rewritten, since the leaf commits
        // to the storage root, so one pass over the accounts covers both tries.
        let mut changes = Vec::with_capacity(state.accounts.len());
        for (hashed_address, account) in &state.accounts {
            let Some(account) = account else {
                changes.push((*hashed_address, None));
                continue;
            };
            let storage_root = self
                .storage_root(*hashed_address, state.storages.get(hashed_address))
                .map_err(|_| StatelessTrieError::StatelessStateRootCalculationFailed)?;
            let value = encode_value::<MAX_ACCOUNT_RLP>(
                self.bump,
                TrieAccount {
                    nonce: account.nonce,
                    balance: account.balance,
                    storage_root,
                    code_hash: account.bytecode_hash.unwrap_or(KECCAK256_EMPTY),
                },
            );
            changes.push((*hashed_address, Some(value)));
        }

        let root = self.root;
        self.batch_update(root, &changes)
            .map_err(|_| StatelessTrieError::StatelessStateRootCalculationFailed)
    }
}

impl PathState {
    /// The node `digest` commits to. A digest the witness has no node for means the block read
    /// below a subtree the witness never revealed.
    fn resolve(&self, digest: B256) -> Result<&'static [u8], Error> {
        self.nodes
            .get(&digest)
            .copied()
            .ok_or(Error::NodeNotResolved(digest))
    }

    /// The node a parent references through `item`, or the empty node when the slot is absent.
    fn child(&self, item: &'static [u8]) -> Result<&'static [u8], Error> {
        let mut buf = item;
        let header = alloy_rlp::Header::decode(&mut buf)?;
        if header.list {
            return Ok(item);
        }
        let payload = split_payload(&mut buf, header.payload_length)?;
        match payload.len() {
            0 => Ok(EMPTY_NODE),
            DIGEST_LEN => self.resolve(B256::from_slice(payload)),
            _ => Err(Error::Rlp(alloy_rlp::Error::UnexpectedLength)),
        }
    }

    /// Walks the trie rooted at `root` for `key` and returns the value it holds, or `None` when the
    /// walk proves the key absent.
    fn get(&self, root: B256, key: &B256) -> Result<Option<&'static [u8]>, Error> {
        if root == EMPTY_ROOT_HASH {
            return Ok(None);
        }
        let nibbles = key_nibbles(key);
        let mut remaining = nibbles.as_slice();
        let mut node_rlp = self.resolve(root)?;
        let mut buf = [0; MAX_NIBBLES];

        loop {
            if let Some((&nibble, rest)) = remaining.split_first()
                && let Some(item) = decode_branch_child(node_rlp, nibble)?
            {
                let child = self.child(item)?;
                if is_empty_node(child) {
                    return Ok(None);
                }
                node_rlp = child;
                remaining = rest;
                continue;
            }

            match decode_node(node_rlp)? {
                Node::Leaf { path, value } => {
                    let path = hex_prefix_nibbles(path, &mut buf)?;
                    return Ok((path == remaining).then_some(value));
                }
                Node::Extension { path, child } => {
                    let path = hex_prefix_nibbles(path, &mut buf)?;
                    let Some(rest) = remaining.strip_prefix(path) else {
                        return Ok(None);
                    };
                    let child = self.child(child)?;
                    if is_empty_node(child) {
                        return Ok(None);
                    }
                    node_rlp = child;
                    remaining = rest;
                }
                // Every key is one length, so no key ends where a branch sits.
                Node::Branch(_) => return Err(Error::ValueInBranch),
            }
        }
    }

    /// The account the state trie records under `hashed_address`, walking the trie for it only the
    /// first time it is asked for.
    ///
    /// Recomputing the state root reads each account before writing it back, so the values held
    /// here stay the ones the trie was read with.
    fn account_at(&self, hashed_address: B256) -> Result<Option<TrieAccount>, Error> {
        if let Some(account) = self.accounts.borrow().get(&hashed_address) {
            return Ok(*account);
        }
        let account = match self.get(self.root, &hashed_address)? {
            Some(mut value) => Some(TrieAccount::decode(&mut value)?),
            None => None,
        };
        self.accounts.borrow_mut().insert(hashed_address, account);
        Ok(account)
    }

    /// The storage root an account ends the block with, applying `storage` to the trie it held
    /// before execution.
    fn storage_root(
        &mut self,
        hashed_address: B256,
        storage: Option<&HashedStorageSorted>,
    ) -> Result<B256, Error> {
        let root = self
            .account_at(hashed_address)?
            .map_or(EMPTY_ROOT_HASH, |account| account.storage_root);
        let Some(storage) = storage else {
            return Ok(root);
        };

        let bump = self.bump;
        let changes: Vec<Change> = storage
            .storage_slots
            .iter()
            .map(|(slot, value)| {
                let value =
                    (!value.is_zero()).then(|| encode_value::<MAX_STORAGE_RLP>(bump, *value));
                (*slot, value)
            })
            .collect();
        // Wiping drops the account's storage, so what is left is only what execution wrote back.
        let root = if storage.wiped { EMPTY_ROOT_HASH } else { root };
        self.batch_update(root, &changes)
    }

    /// Applies `changes`, ordered by key, to the trie rooted at `root` and returns the new root.
    fn batch_update(&mut self, root: B256, changes: &[Change]) -> Result<B256, Error> {
        if changes.is_empty() {
            return Ok(root);
        }
        let node_rlp = if root == EMPTY_ROOT_HASH {
            EMPTY_NODE
        } else {
            self.resolve(root)?
        };
        let root_rlp = self.update(node_rlp, changes, 0)?;
        Ok(if is_empty_node(root_rlp) {
            EMPTY_ROOT_HASH
        } else {
            keccak256(root_rlp)
        })
    }

    /// Applies every change in `changes`, all of whose keys reach this node, and returns the RLP of
    /// the node that replaces it.
    ///
    /// Ordered changes make each branch's children contiguous runs, so one descent covers the whole
    /// change set and no node is decoded or re-encoded twice.
    fn update(
        &mut self,
        node_rlp: &'static [u8],
        changes: &[Change],
        depth: usize,
    ) -> Result<&'static [u8], Error> {
        if let [(key, value)] = changes {
            // A deletion can collapse a branch below, which resolves the node that replaces it by
            // hash, so only a deletion needs its new nodes in the index.
            let nibbles = key_nibbles(key);
            return self.update_one(node_rlp, &nibbles[depth..], *value, value.is_none());
        }

        let branch = if is_empty_node(node_rlp) {
            None
        } else {
            match decode_node(node_rlp)? {
                Node::Branch(payload) => Some(branch_children(payload)?),
                Node::Leaf { .. } | Node::Extension { .. } => None,
            }
        };

        // An empty subtree, or a leaf or extension several keys share, is shaped by each change in
        // turn, since what one leaves behind decides what the next descends into.
        let Some(mut children) = branch else {
            let mut current = node_rlp;
            for (key, value) in changes {
                let nibbles = key_nibbles(key);
                current = self.update_one(current, &nibbles[depth..], *value, true)?;
            }
            return Ok(current);
        };

        let has_deletion = changes.iter().any(|(_, value)| value.is_none());
        let mut start = 0;
        while start < changes.len() {
            let nibble = nibble_at(&changes[start].0, depth);
            let mut end = start + 1;
            while end < changes.len() && nibble_at(&changes[end].0, depth) == nibble {
                end += 1;
            }
            let child = self.child(children[nibble as usize])?;
            let updated = self.update(child, &changes[start..end], depth + 1)?;
            children[nibble as usize] = self.reference(updated, has_deletion);
            start = end;
        }

        if has_deletion {
            match filled_children(&children) {
                Filled::None => return Ok(EMPTY_NODE),
                Filled::One(nibble) => return self.collapse(nibble, children[nibble as usize]),
                Filled::Many => {}
            }
        }
        Ok(encode_branch(self.bump, &children))
    }

    /// Applies one change to this node, whose subtree its key reaches with `remaining` nibbles
    /// left.
    fn update_one(
        &mut self,
        node_rlp: &'static [u8],
        remaining: &[u8],
        value: Option<&'static [u8]>,
        index: bool,
    ) -> Result<&'static [u8], Error> {
        if is_empty_node(node_rlp) {
            return Ok(match value {
                Some(value) => encode_leaf(self.bump, remaining, value),
                None => EMPTY_NODE,
            });
        }

        match decode_node(node_rlp)? {
            Node::Branch(payload) => {
                let Some((&nibble, rest)) = remaining.split_first() else {
                    return Err(Error::ValueInBranch);
                };
                let (start, end) = child_span(payload, nibble)?;
                let child = self.child(&payload[start..end])?;
                let updated = self.update_one(child, rest, value, index)?;
                let item = self.reference(updated, index);

                // A deletion can leave the branch with a single child or none at all, which only
                // inspecting every one of them tells, so it takes the path that materializes them.
                if value.is_none() {
                    let mut children = branch_children(payload)?;
                    children[nibble as usize] = item;
                    return match filled_children(&children) {
                        Filled::None => Ok(EMPTY_NODE),
                        Filled::One(nibble) => self.collapse(nibble, children[nibble as usize]),
                        Filled::Many => Ok(encode_branch(self.bump, &children)),
                    };
                }
                Ok(splice_branch(self.bump, payload, start, end, item))
            }

            Node::Extension { path, child } => {
                let mut buf = [0; MAX_NIBBLES];
                let path = hex_prefix_nibbles(path, &mut buf)?;
                let shared = common_prefix_len(path, remaining);

                if shared == path.len() {
                    let resolved = self.child(child)?;
                    let updated = self.update_one(resolved, &remaining[shared..], value, index)?;
                    if is_empty_node(updated) {
                        return Ok(EMPTY_NODE);
                    }
                    // A deletion can collapse the child into a short node, whose path this
                    // extension's must absorb for the trie to stay canonical.
                    if value.is_none() {
                        let mut child_buf = [0; MAX_NIBBLES];
                        let mut merged = [0; MAX_NIBBLES];
                        match decode_node(updated)? {
                            Node::Leaf {
                                path: child_path,
                                value,
                            } => {
                                let merged = merge_paths(
                                    path,
                                    hex_prefix_nibbles(child_path, &mut child_buf)?,
                                    &mut merged,
                                )?;
                                return Ok(encode_leaf(self.bump, merged, value));
                            }
                            Node::Extension {
                                path: child_path,
                                child,
                            } => {
                                let merged = merge_paths(
                                    path,
                                    hex_prefix_nibbles(child_path, &mut child_buf)?,
                                    &mut merged,
                                )?;
                                return Ok(encode_extension(self.bump, merged, child));
                            }
                            Node::Branch(_) => {}
                        }
                    }
                    let reference = self.reference(updated, index);
                    return Ok(encode_extension(self.bump, path, reference));
                }

                // The key leaves the path, so it was never in this subtree.
                let Some(value) = value else {
                    return Ok(node_rlp);
                };

                // Split the extension where the key leaves it, putting what is left of each side
                // under a branch.
                let mut items = [EMPTY_NODE; 16];
                items[path[shared] as usize] = if shared + 1 < path.len() {
                    let extension = encode_extension(self.bump, &path[shared + 1..], child);
                    self.reference(extension, index)
                } else {
                    child
                };
                if shared >= remaining.len() {
                    return Err(Error::ValueInBranch);
                }
                let leaf = encode_leaf(self.bump, &remaining[shared + 1..], value);
                items[remaining[shared] as usize] = self.reference(leaf, index);

                let branch = encode_branch(self.bump, &items);
                if shared == 0 {
                    return Ok(branch);
                }
                let reference = self.reference(branch, index);
                Ok(encode_extension(self.bump, &path[..shared], reference))
            }

            Node::Leaf {
                path,
                value: present,
            } => {
                let mut buf = [0; MAX_NIBBLES];
                let path = hex_prefix_nibbles(path, &mut buf)?;
                if path == remaining {
                    return Ok(match value {
                        Some(value) => encode_leaf(self.bump, path, value),
                        None => EMPTY_NODE,
                    });
                }

                // The key is not this leaf's, so it was never in this subtree.
                let Some(value) = value else {
                    return Ok(node_rlp);
                };

                // Both keys move under a branch at the nibble where they part.
                let shared = common_prefix_len(path, remaining);
                if shared >= path.len() || shared >= remaining.len() {
                    return Err(Error::ValueInBranch);
                }
                let mut items = [EMPTY_NODE; 16];
                let leaf = encode_leaf(self.bump, &path[shared + 1..], present);
                items[path[shared] as usize] = self.reference(leaf, index);
                let leaf = encode_leaf(self.bump, &remaining[shared + 1..], value);
                items[remaining[shared] as usize] = self.reference(leaf, index);

                let branch = encode_branch(self.bump, &items);
                if shared == 0 {
                    return Ok(branch);
                }
                let reference = self.reference(branch, index);
                Ok(encode_extension(self.bump, &path[..shared], reference))
            }
        }
    }

    /// Replaces a branch a deletion left with one child by the short node the canonical trie asks
    /// for, which carries the child's nibble at the front of its path.
    fn collapse(&mut self, nibble: u8, item: &'static [u8]) -> Result<&'static [u8], Error> {
        let child_rlp = self.child(item)?;
        match decode_node(child_rlp)? {
            Node::Leaf { path, value } => {
                let (mut buf, mut merged) = ([0; MAX_NIBBLES], [0; MAX_NIBBLES]);
                let path =
                    merge_paths(&[nibble], hex_prefix_nibbles(path, &mut buf)?, &mut merged)?;
                Ok(encode_leaf(self.bump, path, value))
            }
            Node::Extension { path, child } => {
                let (mut buf, mut merged) = ([0; MAX_NIBBLES], [0; MAX_NIBBLES]);
                let path =
                    merge_paths(&[nibble], hex_prefix_nibbles(path, &mut buf)?, &mut merged)?;
                Ok(encode_extension(self.bump, path, child))
            }
            // A branch cannot absorb the nibble, so an extension of it carries it instead.
            Node::Branch(_) => Ok(encode_extension(self.bump, &[nibble], item)),
        }
    }

    /// The item a parent references `node_rlp` through, which is the node itself when short enough
    /// to sit in place and the RLP of its hash otherwise.
    ///
    /// A hashed node joins the index only when `index` is set, which the callers that can go on to
    /// resolve it by hash ask for. Every other caller receives the node up the call stack instead
    /// and would only be paying for an entry nothing reads.
    fn reference(&mut self, node_rlp: &'static [u8], index: bool) -> &'static [u8] {
        if node_rlp.len() < DIGEST_LEN {
            return node_rlp;
        }
        let digest = keccak256(node_rlp);
        if index {
            self.nodes.insert(digest, node_rlp);
        }
        encode_digest(self.bump, digest)
    }
}

/// A decoded trie node. Paths stay hex-prefix encoded and children stay whole RLP items, both
/// borrowing the node's own encoding, so a re-encoding can copy the parts it does not change.
enum Node<'a> {
    Leaf { path: &'a [u8], value: &'a [u8] },
    Extension { path: &'a [u8], child: &'a [u8] },
    Branch(&'a [u8]),
}

/// Decodes a node far enough to tell which of the three it is, which for a branch is two item
/// headers. Its children stay in the payload, so an update that replaces one never pays to
/// materialize the fifteen it leaves alone.
fn decode_node(node_rlp: &[u8]) -> Result<Node<'_>, Error> {
    let payload = list_payload(node_rlp)?;
    let mut rest = payload;
    let first = split_item(&mut rest)?;
    let second = split_item(&mut rest)?;

    if !rest.is_empty() {
        return Ok(Node::Branch(payload));
    }

    let mut path = first;
    let path = alloy_rlp::Header::decode_bytes(&mut path, false)?;
    let &flags = path.first().ok_or(Error::MalformedPath)?;
    if flags & HEX_PREFIX_FLAG_LEAF == 0 {
        return Ok(Node::Extension {
            path,
            child: second,
        });
    }
    let mut value = second;
    let value = alloy_rlp::Header::decode_bytes(&mut value, false)?;
    Ok(Node::Leaf { path, value })
}

/// Decodes only the branch child at `nibble`, leaving the fifteen a walk will not follow unparsed.
/// Returns `None` when the node turns out to be a leaf or an extension.
fn decode_branch_child(node_rlp: &[u8], nibble: u8) -> Result<Option<&[u8]>, Error> {
    let mut payload = list_payload(node_rlp)?;
    let mut child = None;
    for index in 0..=nibble {
        if payload.is_empty() {
            return Ok(None);
        }
        let item = split_item(&mut payload)?;
        if index == nibble {
            child = Some(item);
        }
    }

    // A leaf or an extension holds two items, so what proves this is a branch is a third. Slot 2
    // and beyond cleared that bar by being reached at all, and the two below it look ahead.
    if nibble == 0 {
        if payload.is_empty() {
            return Ok(None);
        }
        split_item(&mut payload)?;
    }
    if nibble < 2 && payload.is_empty() {
        return Ok(None);
    }
    Ok(child)
}

/// The range the child at `nibble` occupies in a branch's payload.
fn child_span(payload: &[u8], nibble: u8) -> Result<(usize, usize), Error> {
    let mut rest = payload;
    let mut start = 0;
    for _ in 0..nibble {
        start += split_item(&mut rest)?.len();
    }
    Ok((start, start + split_item(&mut rest)?.len()))
}

/// The child items of a branch, given its payload.
fn branch_children(payload: &[u8]) -> Result<[&[u8]; 16], Error> {
    let mut rest = payload;
    let mut children = [EMPTY_NODE; 16];
    for child in &mut children {
        *child = split_item(&mut rest)?;
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
enum Filled {
    None,
    One(u8),
    Many,
}

/// Which children of a branch are still filled.
fn filled_children(children: &[&[u8]; 16]) -> Filled {
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
fn is_empty_node(item: &[u8]) -> bool {
    matches!(item, [byte] if *byte == alloy_rlp::EMPTY_STRING_CODE)
}

/// Splits `len` bytes off the front of `buf`, leaving `buf` on what follows them.
fn split_payload<'a>(buf: &mut &'a [u8], len: usize) -> Result<&'a [u8], Error> {
    if buf.len() < len {
        return Err(Error::Rlp(alloy_rlp::Error::InputTooShort));
    }
    let (payload, rest) = buf.split_at(len);
    *buf = rest;
    Ok(payload)
}

/// Splits the RLP item at the front of `buf` off whole, header included.
fn split_item<'a>(buf: &mut &'a [u8]) -> Result<&'a [u8], Error> {
    let start = *buf;
    let header = alloy_rlp::Header::decode(buf)?;
    split_payload(buf, header.payload_length)?;
    Ok(&start[..start.len() - buf.len()])
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

/// RLP-encodes a leaf holding `value` at `path`.
fn encode_leaf(bump: &'static Bump, path: &[u8], value: &[u8]) -> &'static [u8] {
    let mut buf = [0; MAX_ENCODED_PATH];
    let path = hex_prefix_encode(path, true, &mut buf);
    let (out, pos) = alloc_list(bump, path.length() + value.length());
    let pos = write_string(out, pos, path);
    write_string(out, pos, value);
    out
}

/// RLP-encodes an extension at `path` referencing `child`, which is already a whole RLP item.
fn encode_extension(bump: &'static Bump, path: &[u8], child: &[u8]) -> &'static [u8] {
    let mut buf = [0; MAX_ENCODED_PATH];
    let path = hex_prefix_encode(path, false, &mut buf);
    let (out, pos) = alloc_list(bump, path.length() + child.len());
    let pos = write_string(out, pos, path);
    out[pos..pos + child.len()].copy_from_slice(child);
    out
}

/// RLP-encodes a branch from its child items, leaving the value slot a trie of fixed-length keys
/// never fills empty.
fn encode_branch(bump: &'static Bump, children: &[&[u8]; 16]) -> &'static [u8] {
    let payload_length = children.iter().map(|child| child.len()).sum::<usize>() + 1;
    let (out, mut pos) = alloc_list(bump, payload_length);
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
fn splice_branch(
    bump: &'static Bump,
    payload: &[u8],
    start: usize,
    end: usize,
    item: &[u8],
) -> &'static [u8] {
    let payload_length = payload.len() - (end - start) + item.len();
    let (out, pos) = alloc_list(bump, payload_length);
    out[pos..pos + start].copy_from_slice(&payload[..start]);
    let pos = pos + start;
    out[pos..pos + item.len()].copy_from_slice(item);
    let pos = pos + item.len();
    out[pos..].copy_from_slice(&payload[end..]);
    out
}

/// RLP-encodes `digest` as the 33-byte item a parent references a hashed node through.
fn encode_digest(bump: &'static Bump, digest: B256) -> &'static [u8] {
    let out = bump.alloc_slice_fill_copy(1 + DIGEST_LEN, 0);
    out[0] = alloy_rlp::EMPTY_STRING_CODE + DIGEST_LEN as u8;
    out[1..].copy_from_slice(digest.as_slice());
    out
}

/// RLP-encodes a value into the arena through an `N`-byte buffer, which the caller sizes from the
/// value's widest encoding.
fn encode_value<const N: usize>(bump: &'static Bump, value: impl Encodable) -> &'static [u8] {
    debug_assert!(
        value.length() <= N,
        "the buffer is sized for the widest encoding of the value"
    );
    let mut buf = [0; N];
    let mut out = buf.as_mut_slice();
    value.encode(&mut out);
    let written = N - out.len();
    bump.alloc_slice_copy(&buf[..written])
}

/// Allocates a list of `payload_length` bytes in the arena and returns it with the position its
/// payload starts at.
fn alloc_list(bump: &'static Bump, payload_length: usize) -> (&'static mut [u8], usize) {
    let len = alloy_rlp::length_of_length(payload_length) + payload_length;
    let out = bump.alloc_slice_fill_copy(len, 0);
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

/// Path holds an odd number of nibbles, so the low nibble of the flag byte is data.
const HEX_PREFIX_FLAG_ODD: u8 = 0x10;
/// Node is a leaf rather than an extension.
const HEX_PREFIX_FLAG_LEAF: u8 = 0x20;

/// Expands a key into its nibbles, most significant first.
fn key_nibbles(key: &B256) -> [u8; MAX_NIBBLES] {
    let mut nibbles = [0; MAX_NIBBLES];
    let (pairs, _) = nibbles.as_chunks_mut::<2>();
    for (byte, pair) in key.iter().zip(pairs) {
        pair[0] = byte >> 4;
        pair[1] = byte & 0x0f;
    }
    nibbles
}

/// The nibble a key holds at `depth`.
fn nibble_at(key: &B256, depth: usize) -> u8 {
    let byte = key[depth / 2];
    if depth.is_multiple_of(2) {
        byte >> 4
    } else {
        byte & 0x0f
    }
}

/// Length of the common prefix of two nibble sequences.
fn common_prefix_len(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

/// Expands a hex-prefix encoded path into `out` and returns its nibbles.
fn hex_prefix_nibbles<'o>(
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
fn hex_prefix_encode<'o>(
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
fn merge_paths<'o>(
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

#[derive(Debug, thiserror::Error)]
enum Error {
    /// The witness holds no node for a hash a walk had to follow.
    #[error("reached an unresolved node: {0:#}")]
    NodeNotResolved(B256),
    /// Errors related to RLP encoding and decoding.
    #[error("rlp decode error: {0}")]
    Rlp(#[from] alloy_rlp::Error),
    /// A branch carried a value, which a trie whose keys are all one length never has.
    #[error("branch node with value")]
    ValueInBranch,
    /// A hex-prefix path was empty or longer than a key.
    #[error("malformed hex-prefix path")]
    MalformedPath,
}

impl From<Error> for WitnessDbError {
    fn from(error: Error) -> Self {
        match error {
            Error::Rlp(error) => Self::Rlp(error),
            error => Self::TrieWitness(alloc::format!("{error}")),
        }
    }
}
