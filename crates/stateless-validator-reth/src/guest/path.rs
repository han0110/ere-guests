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
//! chain holds, and every one of those bounds is checked rather than assumed. Paths hold at most
//! [`MAX_NIBBLES`] nibbles and an extension holds at least one, a node is referenced through no
//! more than a digest, a leaf carries no more than an account, and a branch value slot is empty
//! because every key is a hash of one length. Every malformed encoding is an error rather than a
//! panic, so a bad witness can cost a block its validation but never its correctness.
//!
//! Those checks are also what bounds the scratch the descent builds in, which `stack.rs` derives
//! from the arms below.
//!
//! Modelled on the trie zesu proves its stateless guest with, `src/stateless/mpt` of
//! <https://github.com/eth-act/zesu>, whose `verifyProofIndexed`, `batchUpdateIndexed` and
//! `updNodeExImpl` this mirrors.

mod encode;
mod error;
mod nibbles;
mod node;
mod stack;
#[cfg(test)]
mod tests;

use alloc::{boxed::Box, vec, vec::Vec};
use core::cell::RefCell;

use alloy_primitives::{
    Address, B256, Bytes, KECCAK256_EMPTY, U256, keccak256,
    map::{B256IndexMap, B256Map},
};
use alloy_rlp::Decodable;
use alloy_rpc_types_debug::ExecutionWitness;
use alloy_trie::{EMPTY_ROOT_HASH, TrieAccount};
use reth_trie_common::{HashedPostState, HashedStorageSorted};
use reth_tries::{StatelessTrie, StatelessTrieError, WitnessDbError};

use crate::guest::path::{
    encode::{
        Carried, encode_branch, encode_extension, encode_extension_encoded, encode_leaf,
        encode_leaf_from_key, encode_value, reference, splice_branch,
    },
    error::Error,
    nibbles::{
        MAX_NIBBLES, common_prefix_len, hex_prefix_decode, key_nibbles, merge_paths, nibble_at,
        path_from,
    },
    node::{
        DIGEST_LEN, EMPTY_NODE, Filled, MAX_BRANCH_RLP, Node, branch_children, decode_node,
        filled_children, is_empty_node, split_payload,
    },
    stack::{DynStack, STACK_LEN},
};

/// One key's new value as an index into the change source, or `None` to remove it.
type Change = (B256, Option<u32>);

/// The values a change set carries, encoded only once the descent reaches the leaf that holds them
/// so nothing a change set contributes outlives the node it is copied into.
///
/// A change names its value by the index it took in the source the caller drew it from, so an
/// implementation is whatever slice that source is.
trait ChangedValues {
    /// The RLP of the value at `index`.
    fn encode<'s>(&self, index: u32, stack: &mut DynStack<'s>) -> &'s [u8];
}

/// Accounts, each already carrying the storage root recomputing it produced.
impl ChangedValues for [TrieAccount] {
    fn encode<'s>(&self, index: u32, stack: &mut DynStack<'s>) -> &'s [u8] {
        encode_value(self[index as usize], stack)
    }
}

/// Storage slots, indexed as the change set that named them.
impl ChangedValues for [(B256, U256)] {
    fn encode<'s>(&self, index: u32, stack: &mut DynStack<'s>) -> &'s [u8] {
        encode_value(self[index as usize].1, stack)
    }
}

/// A child whose encoding this frame holds rather than the witness, named by the nibble it sits
/// under, since looking a node the descent built up by hash is the one lookup the witness cannot
/// answer.
///
/// A collapse happens only when a single child is left, so at most one child a frame builds can
/// survive it and every other child a collapse could reach is still an item the witness supplied.
type Prebuilt<'s> = Option<(u8, &'s [u8])>;

/// The Ethereum world state over the witness bytes, walked rather than built.
#[derive(Debug)]
pub(crate) struct PathState {
    /// Trie nodes by the Keccak hash they are referenced through, which the witness supplies and
    /// nothing the descent builds ever joins.
    nodes: B256Map<&'static [u8]>,
    /// Root every read is anchored at, which the parent block header commits to.
    root: B256,
    /// Accounts the state trie holds and a read has already walked it for, as they stood before
    /// execution. Execution reads an account and then its storage, which would otherwise walk the
    /// state trie twice for every account whose storage a block touches. An account the trie has
    /// none of is left out, since a block reading a great many of those would fill this with
    /// entries answering only the reads that never come.
    accounts: RefCell<B256Map<TrieAccount>>,
    /// Where a read stands once it has taken the first nibbles of its key, held per trie and
    /// prefix.
    ///
    /// Every read of one trie crosses the same nodes near its root, so a block reading a thousand
    /// accounts, or a thousand slots of one account, would otherwise split those nodes and look
    /// their children up by digest a thousand times over. Deeper down a witness holds far more
    /// nodes than any block reads twice, so a read stops recording there. Each entry carries
    /// the root it was taken from, so the tries read alongside one another share the room
    /// rather than evicting one another.
    prefix_nodes: RefCell<[Option<PrefixNode>; 1 << (4 * PREFIX_NIBBLES)]>,
}

/// The trie a read was walking, the nibbles of its key it had taken and the node it had reached.
type PrefixNode = (B256, usize, &'static [u8]);

/// Nibbles of a key a prefix covers, which are the ones its first byte holds.
const PREFIX_NIBBLES: usize = 2;

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

        let bytecode = witness
            .codes
            .into_iter()
            .map(|code| (keccak256(&code), code))
            .collect();

        Ok((
            Self {
                nodes,
                root: pre_state_root,
                accounts: RefCell::new(B256Map::default()),
                prefix_nodes: RefCell::new([None; 1 << (4 * PREFIX_NIBBLES)]),
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
        // An account holding no storage answers every slot with zero, so the slot is hashed only
        // where there is a trie to look it up in. A block reading cold slots of accounts that hold
        // none would otherwise pay a hash for every one of them.
        if account.storage_root == EMPTY_ROOT_HASH {
            return Ok(U256::ZERO);
        }
        Ok(
            match self.get(account.storage_root, &keccak256(B256::from(slot)))? {
                Some(mut value) => U256::decode(&mut value)?,
                None => U256::ZERO,
            },
        )
    }

    fn calculate_state_root(&mut self, state: HashedPostState) -> Result<B256, StatelessTrieError> {
        // Every trie recomputed here builds in this one buffer, which a descent hands back whole
        // because it returns a hash rather than a borrow. Lending it out rather than holding it in
        // the state is what leaves no descent able to run inside another.
        let mut scratch = vec![0; STACK_LEN];

        let state = state.into_sorted();

        // Every account whose storage changed also has its leaf rewritten, since the leaf commits
        // to the storage root, so one pass over the accounts covers both tries.
        let mut changes = Vec::with_capacity(state.accounts.len());
        let mut accounts = Vec::with_capacity(state.accounts.len());
        for (hashed_address, account) in &state.accounts {
            let Some(account) = account else {
                changes.push((*hashed_address, None));
                continue;
            };
            let storage = state.storages.get(hashed_address);
            let storage_root = self
                .storage_root(*hashed_address, storage, &mut scratch)
                .map_err(|_| StatelessTrieError::StatelessStateRootCalculationFailed)?;
            changes.push((*hashed_address, Some(accounts.len() as u32)));
            accounts.push(TrieAccount {
                nonce: account.nonce,
                balance: account.balance,
                storage_root,
                code_hash: account.bytecode_hash.unwrap_or(KECCAK256_EMPTY),
            });
        }

        let root = self.root;
        self.batch_update(root, accounts.as_slice(), &changes, &mut scratch)
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
    ///
    /// A child is an empty slot or a digest everywhere but the shortest of nodes, and its first
    /// byte tells which, so resolving one reads a byte rather than decoding a header.
    fn resolve_child<'s>(&self, item: &'s [u8]) -> Result<&'s [u8], Error> {
        let mut buf = item;
        let header = alloy_rlp::Header::decode(&mut buf)?;
        if header.list {
            return Ok(item);
        }
        let payload = split_payload(&mut buf, header.payload_length)?;
        match payload.len() {
            0 => Ok(EMPTY_NODE),
            DIGEST_LEN => Ok(self.resolve(B256::from_slice(payload))?),
            _ => Err(Error::Rlp(alloy_rlp::Error::UnexpectedLength)),
        }
    }

    /// Walks the trie rooted at `root` for `key` and returns the value it holds, or `None` when the
    /// walk proves the key absent.
    fn get(&self, root: B256, key: &B256) -> Result<Option<&'static [u8]>, Error> {
        if root == EMPTY_ROOT_HASH {
            return Ok(None);
        }
        // The key stays packed, since a node's path is compared against the bytes it already sits
        // in and a walk of a level or two would never look at most of the nibbles in it.
        let prefix = usize::from(key[0]);
        let (mut depth, mut node_rlp) = match self.prefix_nodes.borrow()[prefix] {
            Some((held, depth, node_rlp)) if held == root => (depth, node_rlp),
            _ => (0, self.resolve(root)?),
        };

        loop {
            // A read stops recording where it leaves the nibbles an entry is held per, so what
            // stands is the deepest node every key sharing them reaches.
            if depth <= PREFIX_NIBBLES {
                self.prefix_nodes.borrow_mut()[prefix] = Some((root, depth, node_rlp));
            }
            // A slot the parent left empty ends the walk, since the key is under it or nowhere.
            if is_empty_node(node_rlp) {
                return Ok(None);
            }

            match decode_node(node_rlp)? {
                Node::Leaf { path, value } => {
                    let Some(taken) = path_from(key, depth, path) else {
                        return Ok(None);
                    };
                    // A leaf ends exactly where a key does, so one reaching any other depth holds
                    // the key of no trie this walks.
                    return Ok((depth + taken == MAX_NIBBLES).then_some(value));
                }
                Node::Extension { path, child } => {
                    let Some(taken) = path_from(key, depth, path) else {
                        return Ok(None);
                    };
                    node_rlp = self.resolve_child(child)?;
                    depth += taken;
                }
                // Only a branch's value slot answers a key that has run out, and it stays empty
                // because every key is a hash of one length.
                Node::Branch(branch) => {
                    if depth == MAX_NIBBLES {
                        return Err(Error::ValueInBranch);
                    }
                    node_rlp = self.resolve_child(branch.child(nibble_at(key, depth))?)?;
                    depth += 1;
                }
            }
        }
    }

    /// The account the state trie records under `hashed_address`, walking the trie for it only the
    /// first time it is asked for and every time for an account the trie holds none of.
    ///
    /// Recomputing the state root reads each account before writing it back, so the values held
    /// here stay the ones the trie was read with.
    fn account_at(&self, hashed_address: B256) -> Result<Option<TrieAccount>, Error> {
        if let Some(account) = self.accounts.borrow().get(&hashed_address) {
            return Ok(Some(*account));
        }
        let Some(mut value) = self.get(self.root, &hashed_address)? else {
            return Ok(None);
        };
        let account = TrieAccount::decode(&mut value)?;
        self.accounts.borrow_mut().insert(hashed_address, account);
        Ok(Some(account))
    }

    /// The storage root an account ends the block with, applying `storage` to the trie it held
    /// before execution.
    fn storage_root(
        &self,
        hashed_address: B256,
        storage: Option<&HashedStorageSorted>,
        scratch: &mut [u8],
    ) -> Result<B256, Error> {
        let root = self
            .account_at(hashed_address)?
            .map_or(EMPTY_ROOT_HASH, |account| account.storage_root);
        let Some(storage) = storage else {
            return Ok(root);
        };

        let changes: Vec<Change> = storage
            .storage_slots
            .iter()
            .enumerate()
            .map(|(index, (slot, value))| (*slot, (!value.is_zero()).then_some(index as u32)))
            .collect();
        // Wiping drops the account's storage, so what is left is only what execution wrote back.
        let root = if storage.wiped { EMPTY_ROOT_HASH } else { root };
        self.batch_update(root, storage.storage_slots.as_slice(), &changes, scratch)
    }

    /// Applies `changes`, ordered by key, to the trie rooted at `root` and returns the new root.
    fn batch_update<V: ChangedValues + ?Sized>(
        &self,
        root: B256,
        values: &V,
        changes: &[Change],
        scratch: &mut [u8],
    ) -> Result<B256, Error> {
        if changes.is_empty() {
            return Ok(root);
        }
        // Ordered, distinct keys are what leave each branch's children a contiguous sub-slice, so
        // descent covers the whole change set, no key reaches a node twice, and both the depth a
        // descent runs to and the scratch it holds stay bounded by the trie. Every change set comes
        // from a map the caller sorted, so this holds by construction.
        debug_assert!(
            changes.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "a change set is ordered by key"
        );
        let node_rlp = if root == EMPTY_ROOT_HASH {
            EMPTY_NODE
        } else {
            self.resolve(root)?
        };
        let mut stack = DynStack::new(scratch);
        let descent = Descent {
            state: self,
            values,
        };
        descent
            .update(node_rlp, changes, 0, &mut stack)
            .map(|root_rlp| {
                if is_empty_node(root_rlp) {
                    EMPTY_ROOT_HASH
                } else {
                    keccak256(root_rlp)
                }
            })
    }
}

/// What every frame of one descent reads, which is fixed for the whole change set and so stays
/// clear of the arguments the recursion carries.
struct Descent<'a, V: ChangedValues + ?Sized> {
    /// The trie the descent walks.
    state: &'a PathState,
    /// The values its change set draws on.
    values: &'a V,
}

impl<V: ChangedValues + ?Sized> Descent<'_, V> {
    /// Applies every change in `changes`, all of whose keys reach this node, and returns the RLP of
    /// the node that replaces it.
    ///
    /// Ordered changes leave each branch's children a contiguous sub-slice of the change set, so
    /// one descent covers the whole of it and no node is decoded or re-encoded twice.
    fn update<'s>(
        &self,
        node_rlp: &'s [u8],
        changes: &[Change],
        depth: usize,
        stack: &mut DynStack<'s>,
    ) -> Result<&'s [u8], Error> {
        if let [(key, value)] = changes {
            return self.update_one(node_rlp, key, depth, *value, stack);
        }

        // An empty subtree is shaped by the change set alone, so it is built in one top-down pass
        // rather than by inserting one key at a time.
        if is_empty_node(node_rlp) {
            return self.build_subtree(changes, depth, None, stack);
        }

        match decode_node(node_rlp)? {
            Node::Leaf { path, value } => self.update_leaf(path, value, changes, depth, stack),
            Node::Extension { path, child } => {
                self.update_extension(path, child, changes, depth, stack)
            }
            Node::Branch(branch) => self.update_branch(
                branch_children(branch.payload)?,
                changes,
                depth,
                None,
                stack,
            ),
        }
    }

    /// The RLP of the subtree that replaces this leaf, which keeps the leaf itself only when no
    /// change names its key.
    fn update_leaf<'s>(
        &self,
        path: &[u8],
        value: &'s [u8],
        changes: &[Change],
        depth: usize,
        stack: &mut DynStack<'s>,
    ) -> Result<&'s [u8], Error> {
        let mut buf = [0; MAX_NIBBLES];
        let leaf_path = hex_prefix_decode(path, &mut buf)?;
        // Every key is a hash of one length, so a leaf ends exactly where a key does. A path any
        // longer would carry the split past the last nibble a key has.
        if depth + leaf_path.len() != MAX_NIBBLES {
            return Err(Error::MalformedPath);
        }
        // A change naming the leaf's own key replaces or removes it, so the leaf survives as an
        // entry of its own only when no change reaches it.
        let overridden = changes
            .binary_search_by(|(key, _)| key_nibbles(key)[depth..].cmp(leaf_path))
            .is_ok();
        let existing = (!overridden).then_some((leaf_path, value));
        self.build_subtree(changes, depth, existing, stack)
    }

    /// The RLP of the node that replaces this extension, which is a branch when the change set
    /// leaves its path part way along.
    fn update_extension<'s>(
        &self,
        encoded: &[u8],
        child: &'s [u8],
        changes: &[Change],
        depth: usize,
        stack: &mut DynStack<'s>,
    ) -> Result<&'s [u8], Error> {
        // Sorted keys put every change between the first and the last, so a path those two both
        // follow is one they all follow and the whole change set stays inside this extension. Their
        // keys stay packed for it, since an extension the set stays inside keeps its own encoding.
        if let Some(taken) = path_from(&changes[0].0, depth, encoded)
            && path_from(&changes[changes.len() - 1].0, depth, encoded).is_some()
        {
            let has_deletion = changes.iter().any(|(_, value)| value.is_none());
            let resolved = self.state.resolve_child(child)?;
            // A deletion can collapse the child into a short node, whose path this extension's must
            // absorb for the trie to stay canonical, so the child builds in this frame's own
            // scratch rather than in a loan it gives back.
            if has_deletion {
                let updated = self.update(resolved, changes, depth + taken, stack)?;
                if is_empty_node(updated) {
                    return Ok(EMPTY_NODE);
                }
                let mut buf = [0; MAX_NIBBLES];
                return node_under(hex_prefix_decode(encoded, &mut buf)?, updated, None, stack);
            }
            let carried = {
                let mut lent = stack.lend();
                let updated = self.update(resolved, changes, depth + taken, &mut lent)?;
                if is_empty_node(updated) {
                    return Ok(EMPTY_NODE);
                }
                Carried::of(updated)
            };
            let item = carried.write(stack);
            return Ok(encode_extension_encoded(encoded, item, stack));
        }

        let mut buf = [0; MAX_NIBBLES];
        let path = hex_prefix_decode(encoded, &mut buf)?;
        // A path that would carry the descent past the last nibble of a key belongs to no node the
        // trie holds, and is what keeps every slice below in range.
        if depth + path.len() > MAX_NIBBLES {
            return Err(Error::MalformedPath);
        }
        debug_assert!(changes.len() >= 2, "a single change takes update_one");
        // The change set leaves this path part way along, so the extension becomes the branch the
        // trie would hold there and the ordinary descent takes it from that point.
        let low = key_nibbles(&changes[0].0);
        let high = key_nibbles(&changes[changes.len() - 1].0);
        let shared = prefix_shared_by_all(&low[depth..], &high[depth..], Some(path));
        let mut items = [EMPTY_NODE; 16];
        // The branch arm runs on the children below rather than on a branch anyone encodes, so what
        // is left of this extension is a child this frame holds rather than one any later trie
        // could find.
        let mut prebuilt = None;
        items[path[shared] as usize] = if shared + 1 < path.len() {
            let extension = encode_extension(&path[shared + 1..], child, stack);
            prebuilt = Some((path[shared], extension));
            reference(extension, stack)
        } else {
            child
        };
        let updated = self.update_branch(items, changes, depth + shared, prebuilt, stack)?;
        if is_empty_node(updated) {
            return Ok(EMPTY_NODE);
        }
        node_under(&path[..shared], updated, None, stack)
    }

    /// The RLP of the node that replaces this branch, which a deletion can leave collapsed into a
    /// short node or empty.
    fn update_branch<'s>(
        &self,
        mut children: [&'s [u8]; 16],
        changes: &[Change],
        depth: usize,
        prebuilt: Prebuilt<'s>,
        stack: &mut DynStack<'s>,
    ) -> Result<&'s [u8], Error> {
        let has_deletion = changes.iter().any(|(_, value)| value.is_none());

        // A collapse can pick a child this frame builds only when every child it leaves alone is
        // already empty, since one surviving witness child is enough to keep the branch a branch.
        let collapse_reaches_built = has_deletion && {
            // Ordered, distinct keys put at least one nibble below this node, so `depth` is short
            // of a whole key and every change still has a nibble here.
            let touched = changes
                .iter()
                .fold(0u16, |mask, (key, _)| mask | 1 << nibble_at(key, depth));
            children
                .iter()
                .enumerate()
                .all(|(nibble, child)| touched & (1 << nibble) != 0 || is_empty_node(child))
        };
        // Room for the one child a collapse could still read, taken from this frame's own scratch
        // rather than from what it lends, so the loans a child returns leave it standing.
        let mut spare = collapse_reaches_built.then(|| stack.alloc(MAX_BRANCH_RLP));
        let mut built: Prebuilt<'s> = None;
        for (nibble, child_changes) in changes_by_nibble(changes, depth) {
            // Sorted keys leave this child a contiguous sub-slice, so its subtree is finished here
            // and only the item the parent references it through has to outlive it.
            let child = match prebuilt {
                Some((prebuilt, node_rlp)) if prebuilt == nibble => node_rlp,
                _ => self.state.resolve_child(children[nibble as usize])?,
            };
            let carried = {
                let mut lent = stack.lend();
                let updated = self.update(child, child_changes, depth + 1, &mut lent)?;
                if !is_empty_node(updated)
                    && let Some(spare) = spare.take()
                {
                    let len = updated.len();
                    if len > MAX_BRANCH_RLP {
                        return Err(Error::OversizedNode);
                    }
                    spare[..len].copy_from_slice(updated);
                    let spare: &'s [u8] = spare;
                    built = Some((nibble, &spare[..len]));
                }
                Carried::of(updated)
            };
            children[nibble as usize] = carried.write(stack);
        }

        if has_deletion {
            // A prebuilt child no change reaches keeps its slot and its encoding. A change that
            // reaches it overwrites that slot, so the entry a collapse could read is the one this
            // frame rebuilt, and where it rebuilt nothing the slot is empty and no collapse
            // reaches it.
            debug_assert!(
                built.is_some()
                    || prebuilt.is_none_or(|(nibble, _)| {
                        !changes
                            .iter()
                            .any(|(key, _)| nibble_at(key, depth) == nibble)
                            || is_empty_node(children[nibble as usize])
                    }),
                "a collapse could read a prebuilt child the branch loop overwrote"
            );
            return self.close_branch(&children, built.or(prebuilt), stack);
        }
        Ok(encode_branch(&children, stack))
    }

    /// The subtree a change set alone shapes, built top down.
    ///
    /// Applying the changes one at a time would re-encode the node under construction once per
    /// change and reach back into what it had just built by hash, holding every intermediate alive
    /// until the last change lands. Taking the whole change set at once instead makes what this
    /// holds a function of the depth of the trie rather than of the size of the change set.
    ///
    /// Removals are dropped rather than applied, since a subtree with no node in it holds nothing
    /// to remove, and they are the reason a subtree is delimited by the writes it carries rather
    /// than by every key in it.
    fn build_subtree<'s>(
        &self,
        changes: &[Change],
        depth: usize,
        existing: Option<(&[u8], &'s [u8])>,
        stack: &mut DynStack<'s>,
    ) -> Result<&'s [u8], Error> {
        let mut writes = changes
            .iter()
            .filter_map(|(key, value)| value.map(|value| (key, value)));
        let Some((first, first_value)) = writes.next() else {
            // Nothing is written here, so only what the subtree already held can remain.
            return Ok(match existing {
                Some((path, value)) => encode_leaf(path, value, stack),
                None => EMPTY_NODE,
            });
        };
        let first_nibbles = key_nibbles(first);
        let last = writes.next_back();
        if last.is_none() && existing.is_none() {
            let value = self.values.encode(first_value, stack);
            return Ok(encode_leaf(&first_nibbles[depth..], value, stack));
        }

        let last_nibbles = last.map(|(key, _)| key_nibbles(key));
        let low = &first_nibbles[depth..];
        let high = last_nibbles.as_ref().map_or(low, |last| &last[depth..]);
        let shared = prefix_shared_by_all(low, high, existing.map(|(path, _)| path));
        let split = depth + shared;
        // An entry that survived the writes parts from them before its path ends, and that path
        // ends where a key does, so both it and the writes still have a nibble at the split.
        let existing = existing.map(|(path, value)| (path[shared], &path[shared + 1..], value));

        let mut items = [EMPTY_NODE; 16];
        for (nibble, child_changes) in changes_by_nibble(changes, split)
            .filter(|(_, child_changes)| child_changes.iter().any(|(_, value)| value.is_some()))
        {
            // The entry the subtree already held joins the changes that share its nibble, and
            // stands alone below only when no write reaches it.
            let existing_below = existing
                .filter(|(existing_nibble, _, _)| *existing_nibble == nibble)
                .map(|(_, path, value)| (path, value));
            let carried = {
                let mut lent = stack.lend();
                Carried::of(self.build_subtree(
                    child_changes,
                    split + 1,
                    existing_below,
                    &mut lent,
                )?)
            };
            items[nibble as usize] = carried.write(stack);
        }

        if let Some((nibble, path, value)) = existing
            && is_empty_node(items[nibble as usize])
        {
            let carried = {
                let mut lent = stack.lend();
                Carried::of(encode_leaf(path, value, &mut lent))
            };
            items[nibble as usize] = carried.write(stack);
        }

        // The lowest and the highest entry part at the split, so the branch always keeps two
        // children and never needs collapsing.
        Ok(branch_under(&low[..shared], &items, stack))
    }

    /// Applies one change to this node, whose subtree its key reaches with the nibbles it carries
    /// from `depth` left.
    ///
    /// The key stays packed, since a node whose path the key follows is one whose path is compared
    /// against the bytes it already sits in and whose encoding the change leaves alone. Only a key
    /// that parts from a node is expanded, and only in the frame that splits it.
    fn update_one<'s>(
        &self,
        node_rlp: &'s [u8],
        key: &B256,
        depth: usize,
        value: Option<u32>,
        stack: &mut DynStack<'s>,
    ) -> Result<&'s [u8], Error> {
        if is_empty_node(node_rlp) {
            return Ok(self.leaf_or_empty(key, depth, value, stack));
        }

        match decode_node(node_rlp)? {
            Node::Leaf {
                path: encoded,
                value: present,
            } => {
                // Every key is a hash of one length, so a leaf ends exactly where a key does. A
                // leaf reaching any other depth belongs to no node the trie holds,
                // and answering a removal with it would leave the branch above
                // fuller than the scratch is sized for.
                if let Some(taken) = path_from(key, depth, encoded) {
                    if depth + taken != MAX_NIBBLES {
                        return Err(Error::MalformedPath);
                    }
                    return Ok(self.leaf_or_empty(key, depth, value, stack));
                }
                let mut buf = [0; MAX_NIBBLES];
                let path = hex_prefix_decode(encoded, &mut buf)?;
                if depth + path.len() != MAX_NIBBLES {
                    return Err(Error::MalformedPath);
                }

                // The key is not this leaf's, so it was never in this subtree.
                let Some(value) = value else {
                    return Ok(node_rlp);
                };

                // Both keys move under a branch at the nibble where they part, which is short of
                // where either path ends because the two are the same length and differ.
                let written = key_nibbles(key);
                let remaining = &written[depth..];
                let shared = common_prefix_len(path, remaining);
                let leaf = encode_leaf(&path[shared + 1..], present, stack);
                let existing_item = reference(leaf, stack);
                Ok(self.parting_branch(path, existing_item, remaining, shared, value, stack))
            }

            Node::Extension {
                path: encoded,
                child,
            } => {
                if let Some(taken) = path_from(key, depth, encoded) {
                    let resolved = self.state.resolve_child(child)?;
                    // A deletion can collapse the child into a short node, whose path this
                    // extension's must absorb for the trie to stay canonical.
                    if value.is_none() {
                        let updated =
                            self.update_one(resolved, key, depth + taken, value, stack)?;
                        if is_empty_node(updated) {
                            return Ok(EMPTY_NODE);
                        }
                        let mut buf = [0; MAX_NIBBLES];
                        let path = hex_prefix_decode(encoded, &mut buf)?;
                        return node_under(path, updated, None, stack);
                    }
                    let carried = {
                        let mut lent = stack.lend();
                        let updated =
                            self.update_one(resolved, key, depth + taken, value, &mut lent)?;
                        if is_empty_node(updated) {
                            return Ok(EMPTY_NODE);
                        }
                        Carried::of(updated)
                    };
                    let item = carried.write(stack);
                    return Ok(encode_extension_encoded(encoded, item, stack));
                }

                let mut buf = [0; MAX_NIBBLES];
                let path = hex_prefix_decode(encoded, &mut buf)?;
                // A path that would carry the descent past the last nibble of a key belongs to no
                // node the trie holds. Refusing it before the split builds anything is what keeps a
                // witness the descent turns down from costing it scratch.
                if depth + path.len() > MAX_NIBBLES {
                    return Err(Error::MalformedPath);
                }

                // The key leaves the path, so it was never in this subtree.
                let Some(value) = value else {
                    return Ok(node_rlp);
                };

                // Split the extension where the key leaves it, putting what is left of each side
                // under a branch.
                let written = key_nibbles(key);
                let remaining = &written[depth..];
                let shared = common_prefix_len(path, remaining);
                let existing_item = if shared + 1 < path.len() {
                    let extension = encode_extension(&path[shared + 1..], child, stack);
                    reference(extension, stack)
                } else {
                    child
                };
                Ok(self.parting_branch(path, existing_item, remaining, shared, value, stack))
            }

            Node::Branch(branch) => {
                if depth == MAX_NIBBLES {
                    return Err(Error::ValueInBranch);
                }
                let nibble = nibble_at(key, depth);
                let (start, end) = branch.child_span(nibble)?;
                let child = self.state.resolve_child(&branch.payload[start..end])?;

                // A deletion can leave the branch with a single child or none at all, which only
                // inspecting every one of them tells, and a collapse may still read the child's own
                // encoding, so it builds in this frame's own scratch rather than in a loan.
                if value.is_none() {
                    let mut children = branch_children(branch.payload)?;
                    let updated = self.update_one(child, key, depth + 1, value, stack)?;
                    children[nibble as usize] = reference(updated, stack);
                    return self.close_branch(&children, Some((nibble, updated)), stack);
                }

                let carried = {
                    let mut lent = stack.lend();
                    Carried::of(self.update_one(child, key, depth + 1, value, &mut lent)?)
                };
                let item = carried.write(stack);
                splice_branch(branch.payload, start, end, item, stack)
            }
        }
    }

    /// The branch that parts the subtree `existing_item` references from the key being written,
    /// each under the nibble its own path takes at the split, with the nibbles they share above
    /// it.
    ///
    /// Both still have a nibble there, since a key parts from a path in the trie before either
    /// ends.
    fn parting_branch<'s>(
        &self,
        existing_path: &[u8],
        existing_item: &'s [u8],
        written: &[u8],
        shared: usize,
        value: u32,
        stack: &mut DynStack<'s>,
    ) -> &'s [u8] {
        let mut items = [EMPTY_NODE; 16];
        items[existing_path[shared] as usize] = existing_item;
        let value = self.values.encode(value, stack);
        let leaf = encode_leaf(&written[shared + 1..], value, stack);
        items[written[shared] as usize] = reference(leaf, stack);
        branch_under(&written[..shared], &items, stack)
    }

    /// The leaf `value` writes at the nibbles `key` carries from `depth`, or the empty node when it
    /// removes one instead.
    fn leaf_or_empty<'s>(
        &self,
        key: &B256,
        depth: usize,
        value: Option<u32>,
        stack: &mut DynStack<'s>,
    ) -> &'s [u8] {
        let Some(value) = value else {
            return EMPTY_NODE;
        };
        let value = self.values.encode(value, stack);
        encode_leaf_from_key(key, depth, value, stack)
    }

    /// The node a branch closes as once a deletion has been applied to its children, which is that
    /// branch itself unless the deletion left it with one child or none at all.
    ///
    /// One child left collapses into the short node the canonical trie asks for, which carries that
    /// child's nibble at the front of its path.
    fn close_branch<'s>(
        &self,
        children: &[&'s [u8]; 16],
        prebuilt: Prebuilt<'s>,
        stack: &mut DynStack<'s>,
    ) -> Result<&'s [u8], Error> {
        match filled_children(children) {
            Filled::None => Ok(EMPTY_NODE),
            Filled::One(nibble) => {
                let item = children[nibble as usize];
                let child_rlp = match prebuilt.filter(|(built, _)| *built == nibble) {
                    Some((_, node_rlp)) => node_rlp,
                    None => self.state.resolve_child(item)?,
                };
                node_under(&[nibble], child_rlp, Some(item), stack)
            }
            Filled::Many => Ok(encode_branch(children, stack)),
        }
    }
}

/// The sub-slices of `changes` whose keys share a nibble at `depth`, each with that nibble. Ordered
/// keys leave the nibble non-decreasing, so the changes sharing one are contiguous.
fn changes_by_nibble(changes: &[Change], depth: usize) -> impl Iterator<Item = (u8, &[Change])> {
    changes
        .chunk_by(move |left, right| nibble_at(&left.0, depth) == nibble_at(&right.0, depth))
        .map(move |child_changes| (nibble_at(&child_changes[0].0, depth), child_changes))
}

/// What a set of sorted keys and a path already in the subtree all share below the depth they are
/// taken from, which is the least any pair of them shares, so the lowest key, the highest and that
/// path settle it. A subtree spans every key below its own path, which is why it can never lengthen
/// what they share.
fn prefix_shared_by_all(low: &[u8], high: &[u8], existing_path: Option<&[u8]>) -> usize {
    let shared = common_prefix_len(low, high);
    match existing_path {
        Some(path) => shared
            .min(common_prefix_len(low, path))
            .min(common_prefix_len(high, path)),
        None => shared,
    }
}

/// The node `node_rlp` becomes with `path` nibbles above it, which a short node absorbs into its
/// own path for the trie to stay canonical and a branch takes an extension for.
///
/// `item` is what a parent already references the subtree through, which only the branch case needs
/// and which sparing lets the other two avoid hashing a node they re-encode anyway.
fn node_under<'s>(
    path: &[u8],
    node_rlp: &'s [u8],
    item: Option<&'s [u8]>,
    stack: &mut DynStack<'s>,
) -> Result<&'s [u8], Error> {
    if path.is_empty() {
        return Ok(node_rlp);
    }
    let (child_path, second, is_leaf) = match decode_node(node_rlp)? {
        Node::Leaf { path, value } => (path, value, true),
        Node::Extension { path, child } => (path, child, false),
        Node::Branch(_) => {
            let item = item.unwrap_or_else(|| reference(node_rlp, stack));
            return Ok(encode_extension(path, item, stack));
        }
    };
    let (mut buf, mut merged) = ([0; MAX_NIBBLES], [0; MAX_NIBBLES]);
    let merged = merge_paths(path, hex_prefix_decode(child_path, &mut buf)?, &mut merged)?;
    Ok(if is_leaf {
        encode_leaf(merged, second, stack)
    } else {
        encode_extension(merged, second, stack)
    })
}

/// The node the branch `items` encode to becomes with `path` nibbles above it, which is that branch
/// under an extension unless `path`, the nibbles those entries share, is empty.
fn branch_under<'s>(path: &[u8], items: &[&[u8]; 16], stack: &mut DynStack<'s>) -> &'s [u8] {
    let branch = encode_branch(items, stack);
    if path.is_empty() {
        return branch;
    }
    let item = reference(branch, stack);
    encode_extension(path, item, stack)
}
