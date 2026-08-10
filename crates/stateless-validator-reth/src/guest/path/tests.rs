//! Differential tests for [`PathState`] against `alloy_trie::HashBuilder`.
//!
//! Retaining proofs for every key yields every node on every path, which is a complete witness, so
//! the hash builder serves as both the witness source and the expected-root oracle.

use alloc::{collections::BTreeMap, format, vec::Vec};
use core::ops::Range;

use alloy_primitives::{B256, Bytes, keccak256};
use alloy_rpc_types_debug::ExecutionWitness;
use alloy_trie::{EMPTY_ROOT_HASH, HashBuilder, Nibbles, proof::ProofRetainer};
use reth_tries::StatelessTrie;

use crate::guest::path::{
    ChangedValues, Error, PathState,
    nibbles::{MAX_ENCODED_PATH, MAX_NIBBLES, hex_prefix_encode, hex_prefix_from_key, key_nibbles},
    node::MAX_LEAF_VALUE,
    stack::{DynStack, STACK_LEN, take_peak},
};

/// Raw bytes, indexed as the change set that named them. The differential suite writes values of
/// widths neither an account nor a storage slot can take.
impl ChangedValues for [&[u8]] {
    fn encode<'s>(&self, index: u32, stack: &mut DynStack<'s>) -> &'s [u8] {
        stack.alloc_copy(self[index as usize])
    }
}

/// A key and the value it takes, or its removal.
type Change = (B256, Option<Vec<u8>>);

/// The trie a case starts from and the changes it applies to it.
type Update = (BTreeMap<B256, Vec<u8>>, Vec<Change>);

/// The case a row carries, which builds the update it is walked over.
type Case = fn() -> Update;

/// The trie a set of key-value pairs hashes to, together with every node in it.
fn build(entries: &BTreeMap<B256, Vec<u8>>) -> (B256, Vec<Bytes>) {
    if entries.is_empty() {
        return (EMPTY_ROOT_HASH, Vec::new());
    }
    let targets = entries.keys().map(Nibbles::unpack).collect::<Vec<_>>();
    let mut builder = HashBuilder::default().with_proof_retainer(ProofRetainer::new(targets));
    for (key, value) in entries {
        builder.add_leaf(Nibbles::unpack(key), value);
    }
    let root = builder.root();
    let nodes = builder
        .take_proof_nodes()
        .into_nodes_sorted()
        .into_iter()
        .map(|(_, node)| node);
    (root, nodes.collect())
}

/// A deterministic 32-byte key for `index`, so a failure is always reproducible.
fn key(index: u64) -> B256 {
    keccak256(index.to_be_bytes())
}

/// A 32-byte key whose leading nibbles are `nibbles` and whose tail is zero, which is how the
/// cases below reach shapes hashed keys never produce.
fn from_nibbles(nibbles: &[u8]) -> B256 {
    let mut bytes = [0; 32];
    for (index, nibble) in nibbles.iter().enumerate() {
        bytes[index / 2] |= nibble << if index.is_multiple_of(2) { 4 } else { 0 };
    }
    B256::from(bytes)
}

/// A 32-byte key of `depth` copies of `nibble` ending in `last`.
fn spined(nibble: u8, depth: usize, last: u8) -> B256 {
    let mut nibbles = alloc::vec![nibble; depth];
    nibbles.push(last);
    from_nibbles(&nibbles)
}

/// A 32-byte key of `fill` bytes whose last two carry `tail`.
fn tailed(fill: u8, tail: u64) -> B256 {
    let mut bytes = [fill; 32];
    bytes[30] = (tail >> 8) as u8;
    bytes[31] = tail as u8;
    B256::from(bytes)
}

/// A deterministic value of `len` bytes for `index`.
fn value(index: u64, len: usize) -> Vec<u8> {
    keccak256(index.to_le_bytes())
        .0
        .iter()
        .copied()
        .cycle()
        .take(len)
        .collect()
}

/// A trie of `count` hashed keys.
fn populated(count: u64) -> BTreeMap<B256, Vec<u8>> {
    (0..count)
        .map(|index| (key(index), value(index, 40)))
        .collect()
}

/// A trie of the keys `indexes` names under [`tailed`] with `fill`.
fn tailed_trie(fill: u8, indexes: Range<u64>) -> BTreeMap<B256, Vec<u8>> {
    indexes
        .map(|index| (tailed(fill, index), value(index, 40)))
        .collect()
}

/// Writes of the keys `indexes` names, each taking the value `value_offset` past its own, so an
/// offset of anything but zero rewrites a populated trie rather than restating it.
fn writes(indexes: Range<u64>, value_offset: u64) -> Vec<Change> {
    indexes
        .map(|index| (key(index), Some(value(index + value_offset, 40))))
        .collect()
}

/// Removals of the keys `indexes` names.
fn removals(indexes: Range<u64>) -> Vec<Change> {
    indexes.map(|index| (key(index), None)).collect()
}

/// Changes over the keys `indexes` names, removing every `period`th of them and writing the rest
/// the value `value_offset` past their own.
fn writes_around_removals(indexes: Range<u64>, period: u64, value_offset: u64) -> Vec<Change> {
    indexes
        .map(|index| {
            let written = !index.is_multiple_of(period);
            (key(index), written.then(|| value(index + value_offset, 40)))
        })
        .collect()
}

/// Writes of the keys `indexes` names under [`tailed`] with `fill`.
fn tailed_writes(fill: u8, indexes: Range<u64>) -> Vec<Change> {
    indexes
        .map(|index| (tailed(fill, index), Some(value(index, 40))))
        .collect()
}

/// Removals of the keys `indexes` names under [`tailed`] with `fill`.
fn tailed_removals(fill: u8, indexes: Range<u64>) -> Vec<Change> {
    indexes.map(|index| (tailed(fill, index), None)).collect()
}

/// Applies `changes`, which have to be ordered by key, to the trie `nodes` holds rooted at `root`,
/// and returns what the descent came to along with the most bytes it held at once.
fn update(nodes: Vec<Bytes>, root: B256, changes: &[Change]) -> (Result<B256, Error>, usize) {
    let witness = ExecutionWitness {
        state: nodes,
        ..Default::default()
    };
    let (state, _) = PathState::new(witness, root).expect("witness is complete");
    let indexed = changes
        .iter()
        .enumerate()
        .map(|(index, (key, value))| (*key, value.is_some().then_some(index as u32)))
        .collect::<Vec<_>>();
    let raw = changes
        .iter()
        .map(|(_, value)| value.as_deref().unwrap_or_default())
        .collect::<Vec<_>>();
    let mut scratch = alloc::vec![0; STACK_LEN];
    let root = state.batch_update(root, raw.as_slice(), &indexed, &mut scratch);
    (root, take_peak())
}

/// Applies `changes` to `initial` through [`PathState`] and asserts the root matches a trie rebuilt
/// from scratch over the same final contents, reporting `name` when it does not.
///
/// Returns the most bytes the descent held at once.
fn assert_update(name: &str, initial: BTreeMap<B256, Vec<u8>>, changes: Vec<Change>) -> usize {
    let (root, nodes) = build(&initial);

    let mut expected_entries = initial;
    for (key, value) in &changes {
        match value {
            Some(value) => expected_entries.insert(*key, value.clone()),
            None => expected_entries.remove(key),
        };
    }
    let (expected_root, _) = build(&expected_entries);

    let mut changes = changes;
    changes.sort_by_key(|(key, _)| *key);
    let (actual_root, peak) = update(nodes, root, &changes);
    let actual_root =
        actual_root.unwrap_or_else(|error| panic!("{name} failed to update, {error}"));
    assert_eq!(actual_root, expected_root, "{name}");
    peak
}

/// The shapes an update is walked over, one row per shape.
#[rustfmt::skip]
const SHAPES: &[(&str, Case)] = &[
    // Sixteen keys land in a trie of sixty-four.
    ("insert_into_populated_trie", || (populated(64), writes(64..80, 0))),
    // Half the leaves of a populated trie take new values.
    ("update_existing_values", || (populated(64), writes(0..32, 1000))),
    // Half a populated trie goes, so branches thin without emptying.
    ("delete_some_keys", || (populated(64), removals(0..32))),
    // All but one key goes, so the trie collapses to a single leaf.
    ("delete_all_but_one_key", || (populated(64), removals(1..64))),
    // Every key goes, so the root becomes the empty node.
    ("delete_every_key", || (populated(64), removals(0..64))),
    // Removals, rewrites and inserts interleave over one descent.
    ("mixed_insert_update_delete", || {
        let changes = [writes_around_removals(0..128, 3, 1000), writes(128..170, 0)].concat();
        (populated(128), changes)
    }),
    // One and two byte values, whose nodes sit in their parent rather than behind a hash.
    ("short_values_stay_inline", || {
        let initial = (0..64).map(|index| (key(index), value(index, 1))).collect();
        let changes = (0..32).map(|index| (key(index), Some(value(index, 2)))).collect();
        (initial, changes)
    }),
    // A one-key trie gains a second, so the leaf splits under a branch.
    ("insert_beside_a_single_key", || (populated(1), writes(1..2, 0))),
    // A one-key trie loses the key it holds.
    ("delete_the_only_key", || (populated(1), removals(0..1))),
    // Five hundred keys build an empty trie top down rather than one key at a time.
    ("many_keys_into_empty_trie", || (BTreeMap::new(), writes(0..512, 0))),
    // Removals of keys an empty subtree never held, interleaved with the writes.
    ("absent_deletions_interleaved_with_writes_into_empty_trie", || {
        (BTreeMap::new(), writes_around_removals(0..512, 4, 0))
    }),
    // An empty trie and nothing but removals, so nothing is built.
    ("deletions_only_into_empty_trie", || (BTreeMap::new(), removals(0..16))),
    // Five hundred keys build under the one leaf a trie already holds.
    ("many_keys_under_a_single_leaf", || (populated(1), writes(1..512, 0))),
    // Keys sharing every byte but their last, with an unrelated removal among them.
    ("shared_prefix_keys_into_empty_trie", || {
        let mut changes = tailed_writes(0, 0..64);
        changes.push((key(999), None));
        (BTreeMap::new(), changes)
    }),
    // A long shared prefix, which puts an extension above the branch a build produces.
    ("shared_prefix_keys_over_a_populated_trie", || (populated(32), tailed_writes(0xab, 0..64))),
    // A run that leaves an extension part way along, so the extension splits.
    ("changes_diverge_inside_an_extension", || {
        let mut changes = [tailed_writes(0x11, 100..108), tailed_writes(0x12, 0..4)].concat();
        changes.push((tailed(0x11, 0), None));
        (tailed_trie(0x11, 0..8), changes)
    }),
    // The same divergence with every change a removal, which drives the collapse arms.
    ("deletions_diverge_inside_an_extension", || {
        let mut initial = tailed_trie(0x11, 0..8);
        initial.extend(tailed_trie(0x12, 0..2));
        let changes = [tailed_removals(0x11, 0..8), tailed_removals(0x12, 0..1)].concat();
        (initial, changes)
    }),
];

#[test]
fn every_shape_updates_to_a_rebuilt_root() {
    for (name, case) in SHAPES {
        let (initial, changes) = case();
        assert_update(name, initial, changes);
    }
}

/// Every deletion pattern over a small trie, which reaches collapse shapes a hand-picked case set
/// would miss.
#[test]
fn exhaustive_deletion_subsets() {
    const KEYS: u64 = 10;
    let initial = populated(KEYS);
    for mask in 0u64..(1 << KEYS) {
        let changes = (0..KEYS)
            .filter(|index| mask & (1 << index) != 0)
            .map(|index| (key(index), None))
            .collect();
        assert_update(&format!("mask {mask:#x}"), initial.clone(), changes);
    }
}

/// Depth the widest descent runs to, branching sixteen ways at every level of it.
const WIDE_DEPTH: usize = 60;

/// Nibble the spine of that descent runs on, which is the last one so every frame finishes all of
/// its other runs before it descends.
const SPINE: u8 = 15;

/// A trie whose keys share every nibble but their last four, and a change set whose inserts part
/// from that shared path at nibble `parting`, so the extension the trie holds splits there.
///
/// `parting` is the low nibble of a byte and so is odd.
fn extension_split(parting: usize) -> Update {
    let deep = |tail: u64| {
        let mut bytes = [0x5a; 32];
        bytes[parting / 2] = 0x10 | ((tail >> 8) & 0x0f) as u8;
        bytes[30] = (tail >> 4) as u8;
        bytes[31] = tail as u8;
        B256::from(bytes)
    };
    let initial = (0..48)
        .map(|index| (deep(index), value(index, 40)))
        .collect();
    let changes = (0..48)
        .map(|index| match index % 3 {
            0 => (deep(index), None),
            1 => (deep(index + 900), Some(value(index, 40))),
            _ => (deep(index), Some(value(index + 7, 40))),
        })
        .collect();
    (initial, changes)
}

/// One row per frame shape the buffer is sized against, each asserting a floor on what that shape
/// holds.
///
/// Hashed keys scatter, so a real trie is eight or so levels deep with branches of two or three
/// children and reaches none of these. Every row forges its keys nibble by nibble to reach the
/// shape instead. Floors rather than exact figures leave an encoding change room while still
/// failing a row that stopped reaching its worst case.
#[rustfmt::skip]
const FORGED: &[(&str, usize, Case)] = &[
    // The widest a plain branch frame gets. A child is a nibble, so sixteen of them is all a branch
    // can have, and a removal among them is what buys this row its spare. That spare, sixteen items
    // and the branch being closed is `BRANCH_CARRY` plus the branch itself.
    ("a_branch_frame_at_its_widest", 1500, || {
        let initial = (0..32u8)
            .map(|index| (from_nibbles(&[index / 2, index % 2]), value(index.into(), 40)))
            .collect();
        let changes = (0..16u8)
            .map(|nibble| (from_nibbles(&[nibble, 0]), (nibble > 0).then(|| value(700, 40))))
            .collect();
        (initial, changes)
    }),
    // The same branch reached through an extension the change set leaves at its first nibble. The
    // frame re-encodes the rest of its own path, references it and then runs the branch loop on the
    // children it built without going a level deeper, so one nibble pays for both. This is
    // `frame_carry`, the widest a single frame gets.
    ("a_frame_at_its_widest", 1600, || {
        let initial = (0..8u8).map(|last| (spined(7, 62, last), value(last.into(), 40))).collect();
        let changes = (0..16u8)
            .map(|nibble| (from_nibbles(&[nibble, 1]), (nibble != 3).then(|| value(500, 40))))
            .collect();
        (initial, changes)
    }),
    // That frame at nearly every depth the sum covers, spined on the last nibble so each frame
    // finishes its other fifteen runs before descending and holds their items the whole time the
    // deeper frame is live. Stacking the per-frame worst case is what makes the descent worst, and
    // the floor is the fifteen items and the child encoding charged at each depth.
    ("a_descent_that_holds_a_wide_frame_at_every_depth", WIDE_DEPTH * (15 * 33 + 532), || {
        let deep = from_nibbles(&[SPINE; WIDE_DEPTH + 2]);
        let initial: BTreeMap<_, _> = (0..WIDE_DEPTH)
            .flat_map(|depth| (0..SPINE).map(move |nibble| spined(SPINE, depth, nibble)))
            .chain([deep])
            .map(|key| (key, value(0, MAX_LEAF_VALUE)))
            .collect();
        let changes = initial
            .keys()
            .map(|key| (*key, (*key != deep).then(|| value(500, MAX_LEAF_VALUE))))
            .collect();
        (initial, changes)
    }),
    // Keys parting only at their last nibble, as deep as a branch can sit, since two keys agreeing
    // further would be one key. `FRAME_TAIL` covers this frame, which is cheaper than the ones
    // above it because its extension has a single nibble left and the deletion that buys its spare
    // empties one of the sixteen slots.
    ("a_descent_to_the_last_nibble", 1250, || {
        let initial = (0..8u8).map(|last| (spined(5, 63, last), value(last.into(), 40))).collect();
        let changes = (0..16u8)
            .map(|last| (spined(5, 63, last), (!last.is_multiple_of(3)).then(|| value(300, 40))))
            .collect();
        (initial, changes)
    }),
    // A run parting from a sixty-nibble extension partway along, the widest split because the frame
    // re-encodes the whole rest of the path. Splits further down re-encode less, so this one stands
    // for every split above it.
    ("an_extension_split_at_nibble_49", 1800, || extension_split(49)),
    // The same split one nibble above the branch the extension ends in, where nothing is left to
    // re-encode and the frame hands the child down as it stands. That is the other arm of the same
    // decision, so the pair brackets every split between them.
    ("an_extension_split_at_nibble_59", 1800, || extension_split(59)),
];

#[test]
fn every_forged_shape_holds_the_bytes_it_is_forged_for() {
    for (name, floor, case) in FORGED {
        let (initial, changes) = case();
        let peak = assert_update(name, initial, changes);
        assert!(
            peak > *floor,
            "{name} held {peak} bytes, under its floor of {floor}"
        );
        std::eprintln!("{name} held {peak} bytes of {STACK_LEN}");
    }
}

/// Frames are sized against re-encoding a leaf that carries at most an account, so a wider one is
/// refused rather than built into scratch sized for less.
#[test]
fn a_leaf_value_wider_than_an_account_is_refused() {
    for len in [MAX_LEAF_VALUE, MAX_LEAF_VALUE + 1] {
        let initial: BTreeMap<_, _> = (0..4)
            .map(|index| (key(index), value(index, len)))
            .collect();
        let (root, nodes) = build(&initial);
        let (result, _) = update(nodes, root, &[(key(0), Some(value(0, len)))]);
        if len > MAX_LEAF_VALUE {
            assert!(
                matches!(result, Err(Error::OversizedValue)),
                "a {len}-byte value"
            );
        } else {
            assert!(result.is_ok(), "a {len}-byte value");
        }
    }
}

/// Applies `changes` to the hand-written nodes `witness` holds, rooted at the first of them.
fn forged_update(witness: Vec<Vec<u8>>, changes: &[Change]) -> (Result<B256, Error>, usize) {
    let root = keccak256(&witness[0]);
    let nodes = witness.into_iter().map(Bytes::from).collect();
    update(nodes, root, changes)
}

/// Walks the hand-written nodes `witness` holds, rooted at the first of them, for `key`.
fn forged_get(witness: Vec<Vec<u8>>, key: B256) -> Result<Option<&'static [u8]>, Error> {
    let root = keccak256(&witness[0]);
    let witness = ExecutionWitness {
        state: witness.into_iter().map(Bytes::from).collect(),
        ..Default::default()
    };
    let (state, _) = PathState::new(witness, root).expect("witness is complete");
    state.get(root, &key)
}

/// Fifteen writes leaving the spine at every depth down to `depth`.
fn spine_writes(depth: usize) -> Vec<Change> {
    (0..depth)
        .flat_map(|depth| (0..SPINE).map(move |nibble| spined(SPINE, depth, nibble)))
        .map(|key| (key, Some(value(500, MAX_LEAF_VALUE))))
        .collect()
}

/// Fifteen writes leaving the spine at every depth, and the removal of the key the spine itself
/// holds. Every frame therefore runs all sixteen of its runs and finishes the fifteen that stop
/// before the one it descends on, and the removal is what buys each of them the spare a collapse
/// could read.
fn spine_change_set() -> Vec<Change> {
    let mut changes = spine_writes(64);
    changes.push((from_nibbles(&[SPINE; 64]), None));
    changes.sort_by_key(|(key, _)| *key);
    changes
}

/// An extension spending a whole key, referencing `bottom` by hash.
fn spending_extension(bottom: &[u8]) -> Vec<u8> {
    let mut extension = alloc::vec![0xf8, 67, 0xa1, 0x00];
    extension.extend([0xff; 32]);
    extension.push(0xa0);
    extension.extend(keccak256(bottom).0);
    extension
}

/// The descent `STACK_LEN` is derived from, which holds every byte of it.
///
/// The costly arm is an extension the change set leaves at its first nibble, since that frame pays
/// for a split and a whole branch loop while consuming a single nibble. A key is 64 nibbles, so a
/// witness that puts one extension over the entire key makes it happen 63 times over.
///
/// [`build`] cannot write that trie, because a canonical writer folds an extension above a pathless
/// leaf into a single leaf. The two nodes are written out by hand instead, which is all a witness
/// ever is, and the root is the hash of the one the walk starts on.
#[test]
fn the_shape_the_bound_is_derived_from_reaches_it() {
    // A leaf holding an account's worth of value at no path of its own, which sits at the last
    // nibble because the extension above it spends every one.
    let mut leaf = alloc::vec![0xf8, 113, 0x20, 0xb8, MAX_LEAF_VALUE as u8];
    leaf.extend(value(0, MAX_LEAF_VALUE));

    let witness = alloc::vec![spending_extension(&leaf), leaf];
    let (root, peak) = forged_update(witness, &spine_change_set());
    root.expect("the descent runs to a root");

    std::eprintln!("the derived shape held {peak} bytes of {STACK_LEN}");
    assert_eq!(peak, STACK_LEN);
}

/// A leaf holding nibbles where a key has none left, which is refused rather than answered.
///
/// That leaf sits under the extension of the descent above, so the same change set reaches it, and
/// the removal it does not match is one it would otherwise hand back whole. The branch at the last
/// nibble would then close with all sixteen slots filled while still holding the spare that removal
/// bought it, which is 64 bytes past what the buffer charges the frame there.
#[test]
fn a_leaf_holding_nibbles_past_the_last_is_refused() {
    let mut leaf = alloc::vec![0xf8, 113, 0x33, 0xb8, MAX_LEAF_VALUE as u8];
    leaf.extend(value(0, MAX_LEAF_VALUE));

    let witness = alloc::vec![spending_extension(&leaf), leaf];
    let (result, _) = forged_update(witness, &spine_change_set());
    assert!(matches!(result, Err(Error::MalformedPath)));
}

/// An extension where a key has no nibble left for it, refused for the same reason.
///
/// An extension a key leaves is one no removal reaches below, so it too hands itself back whole and
/// fills the slot the branch above is sized for losing. It is the only other node that can, since a
/// leaf that matches and a branch at that depth are both already accounted for.
#[test]
fn an_extension_holding_nibbles_past_the_last_is_refused() {
    let mut leaf = alloc::vec![0xf8, 113, 0x20, 0xb8, MAX_LEAF_VALUE as u8];
    leaf.extend(value(0, MAX_LEAF_VALUE));

    let mut bottom = alloc::vec![0xe2, 0x13, 0xa0];
    bottom.extend(keccak256(&leaf).0);

    let witness = alloc::vec![spending_extension(&bottom), bottom, leaf];
    let (result, _) = forged_update(witness, &spine_change_set());
    assert!(matches!(result, Err(Error::MalformedPath)));
}

/// A key that runs out on a branch, which only that branch's value slot could answer.
///
/// The extension above spends every nibble the key has, so the walk arrives with nothing left to
/// steer by and the `split_first` that opens the branch shortcut yields nothing. Every key is a
/// 32-byte hash and ends where every other one does, so the slot stays empty and no walk needs it.
#[test]
fn a_key_ending_on_a_branch_is_refused() {
    // Seventeen empty items, which is a branch holding nothing.
    let mut branch = alloc::vec![0xd1];
    branch.extend([0x80; 17]);

    let witness = alloc::vec![spending_extension(&branch), branch];
    let result = forged_get(witness, B256::repeat_byte(0xff));
    assert!(matches!(result, Err(Error::ValueInBranch)));
}

/// A branch too short to hold the nibble the walk asked for, which a third item makes a branch
/// without giving it the seventeen a branch of the trie holds, so only a witness can present one.
/// The walk runs off the end of the payload looking for the slot, which is where it is refused.
#[test]
fn a_branch_too_short_for_the_nibble_is_refused() {
    let witness = alloc::vec![alloc::vec![0xc3, 0x80, 0x80, 0x80]];
    let result = forged_get(witness, B256::repeat_byte(0x50));
    assert!(matches!(
        result,
        Err(Error::Rlp(alloy_rlp::Error::InputTooShort))
    ));
}

/// A change set naming one key twice, which no caller can produce because each sorts one from a
/// map. Where assertions are checked it is raised, since the pair would otherwise reach the
/// top-down build sharing all 64 of its nibbles and index a key past its last.
#[test]
#[should_panic(expected = "a change set is ordered by key")]
#[cfg(debug_assertions)]
fn a_duplicated_key_is_raised_where_assertions_are_checked() {
    let changes = [(key(0), Some(value(0, 40))), (key(0), Some(value(1, 40)))];
    let _ = update(Vec::new(), EMPTY_ROOT_HASH, &changes);
}

/// The worst a trie [`build`] can actually write does, which lands [`CANONICAL_SHORTFALL`] bytes
/// short.
///
/// A canonical extension always sits above a branch, and a branch needs two children a nibble
/// lower, so the deepest branch is at nibble 63 and the extension above it gets 63 nibbles where
/// the hand-written witness gets 64. That trades one split frame for an ordinary branch frame, and
/// the rest of the descent is identical. Sixteen keys sharing 63 nibbles and parting at the last
/// are what make [`build`] write it.
///
/// The buffer still covers the wider case, since nothing in the guest checks canonicity and a
/// witness node is trusted for hashing to what its parent commits to rather than for its shape.
#[test]
fn the_worst_a_canonical_trie_can_do() {
    let initial: BTreeMap<_, _> = (0..16u8)
        .map(|last| (spined(SPINE, 63, last), value(last.into(), MAX_LEAF_VALUE)))
        .collect();
    let mut changes = spine_writes(63);
    // One of the sixteen at the bottom goes, which is what buys every frame above it the spare a
    // collapse could read.
    for last in 0..16u8 {
        changes.push((
            spined(SPINE, 63, last),
            (last != SPINE).then(|| value(600, MAX_LEAF_VALUE)),
        ));
    }
    let peak = assert_update("the_worst_a_canonical_trie_can_do", initial, changes);
    std::eprintln!("a canonical trie held {peak} bytes of {STACK_LEN}");
    assert_eq!(peak, STACK_LEN - CANONICAL_SHORTFALL);
}

/// Bytes a canonical trie falls short of the buffer, the split frame it cannot have less the branch
/// frame it gets instead, `1128 - 1027`. A search over canonical shapes and depths finds no wider
/// one.
const CANONICAL_SHORTFALL: usize = 101;

/// A read comes to what the trie the witness was built from holds, whether or not an earlier read
/// left an entry behind, at every parity a path can take.
///
/// Two tries are read one key at a time in turn, so every entry one leaves behind is one the
/// other reaches for next. Keys sharing every byte but their last give the long extensions a path
/// is compared against where it starts part way into a byte, and absent keys are read alongside
/// them, since a walk ending on a slot a branch leaves empty parts from a path the same way.
#[test]
fn a_read_comes_to_what_the_trie_holds() {
    let mut entries = populated(512);
    entries.extend(tailed_trie(0x11, 0..64));
    entries.extend(tailed_trie(0x12, 0..3));
    for depth in 1..8 {
        entries.insert(spined(SPINE, depth, 0), value(depth as u64, 40));
    }
    // A second trie over the same keys, whose every value differs, so an entry followed across
    // the two answers with the other's value rather than with nothing.
    let other = entries
        .iter()
        .map(|(key, value)| (*key, value.iter().map(|byte| !byte).collect()))
        .collect::<BTreeMap<_, Vec<u8>>>();
    let (root, mut nodes) = build(&entries);
    let (other_root, other_nodes) = build(&other);
    nodes.extend(other_nodes);
    let witness = ExecutionWitness {
        state: nodes,
        ..Default::default()
    };
    let (state, _) = PathState::new(witness, root).expect("witness is complete");
    let mut absent = (512..640).map(key).collect::<Vec<_>>();
    absent.extend((64..80).map(|index| tailed(0x11, index)));
    absent.extend((0..8).map(|depth| spined(SPINE, depth, 1)));

    for pass in 0..2 {
        for ((key, expected), (_, other_expected)) in entries.iter().zip(&other) {
            for (root, expected) in [(root, expected), (other_root, other_expected)] {
                let value = state
                    .get(root, key)
                    .expect("the witness holds every node")
                    .expect("the key is in the trie");
                assert_eq!(value, expected.as_slice(), "pass {pass}");
            }
        }
        for key in &absent {
            for root in [root, other_root] {
                let value = state.get(root, key).expect("the witness holds every node");
                assert_eq!(value, None, "pass {pass}");
            }
        }
        // A pass that recorded nothing would leave the second reading exactly as the first did,
        // which would let an entry that is never followed pass this.
        let prefix_nodes = state.prefix_nodes.borrow();
        assert!(prefix_nodes.iter().any(Option::is_some), "pass {pass}");
    }
}

/// A path taken from a packed key is the one taken from its nibbles, at every depth either can
/// start on and for both the node kinds a path belongs to.
///
/// A key is packed two nibbles to the byte, so a path starting part way into one is assembled a
/// nibble at a time where a path starting on a whole byte is copied. Every depth is walked because
/// the two differ only in parity, and the leaf flag is carried because it shares the flag byte with
/// the nibble an odd path holds there.
#[test]
fn a_path_from_a_key_is_the_path_from_its_nibbles() {
    for index in 0..8 {
        let key = key(index);
        let nibbles = key_nibbles(&key);
        for depth in 0..=MAX_NIBBLES {
            for is_leaf in [false, true] {
                let (mut from_key, mut from_nibbles) =
                    ([0; MAX_ENCODED_PATH], [0; MAX_ENCODED_PATH]);
                assert_eq!(
                    hex_prefix_from_key(&key, depth, MAX_NIBBLES, is_leaf, &mut from_key),
                    hex_prefix_encode(&nibbles[depth..], is_leaf, &mut from_nibbles),
                    "key {index} at depth {depth}, leaf {is_leaf}"
                );
            }
        }
    }
}
