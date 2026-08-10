//! [`DynStack`], the scratch the nodes built while recomputing the root live in.
//!
//! An allocation splits the buffer and leaves the stack on what follows it, so what a frame lends a
//! child comes back when that loan ends rather than when a caller remembers to rewind. A node the
//! child built therefore cannot outlive the loan it was built in, and the borrow checker is what
//! says so, which is why nothing here is unsafe.
//!
//! A deleting frame passes its stack down by value and takes back what is left instead of lending,
//! since it must keep a child's own encoding. The two forms are what the arms of `path.rs`
//! distinguish, and the type of each call says which it is.
//!
//! Modelled on <https://github.com/sarah-quinones/dynstack>.

use alloc::{boxed::Box, vec};
#[cfg(test)]
use core::cell::Cell;

use crate::guest::path::{
    nibbles::MAX_NIBBLES,
    node::{MAX_BRANCH_RLP, MAX_ITEM},
};

/// `alloy_rlp::length_of_length` as a constant, so the buffer derives at compile time.
const fn length_of_length(payload: usize) -> usize {
    if payload < 56 {
        1
    } else if payload < 256 {
        2
    } else {
        3
    }
}

/// Bytes an RLP list of `payload` bytes occupies.
const fn list_rlp(payload: usize) -> usize {
    length_of_length(payload) + payload
}

/// Bytes `nibbles` occupy as a hex-prefix path, which is a flag byte and half a nibble each behind
/// a string header. One nibble or none is that flag byte alone, whose value is at most `0x3f`, so
/// `write_string` writes it bare and it carries no header.
const fn path_rlp(nibbles: usize) -> usize {
    if nibbles <= 1 {
        return 1;
    }
    let bytes = nibbles / 2 + 1;
    length_of_length(bytes) + bytes
}

/// Bytes the extension a diverging frame at `depth` re-encodes occupies. Its path is what a whole
/// key leaves below the nibble that frame splits on, so it narrows as the descent goes deeper.
const fn residual_rlp(depth: usize) -> usize {
    list_rlp(path_rlp(MAX_NIBBLES - 1 - depth) + MAX_ITEM)
}

/// Bytes a branch frame holds while one of its children runs, which is the child encoding a
/// collapse could still read plus an item per finished child. A child's item is taken when it
/// returns, so the one the descent continues on leaves fifteen of the sixteen outstanding.
const BRANCH_CARRY: usize = (16 - 1) * MAX_ITEM + MAX_BRANCH_RLP;

/// Bytes one frame holds while a deeper frame is live.
///
/// The widest is an extension the change set leaves at its first nibble. That frame re-encodes the
/// rest of its own path, references it and then runs the whole branch loop on the children it built
/// without going a level deeper, so one nibble pays for a split and a branch frame together.
const fn frame_carry(depth: usize) -> usize {
    residual_rlp(depth) + MAX_ITEM + BRANCH_CARRY
}

/// Bytes the frame at the last nibble holds. Its extension has a single nibble left, so it runs its
/// branch loop without re-encoding a residual, and the deletion that buys the collapse spare
/// empties one of the sixteen slots rather than filling it, because a key ends where those slots
/// sit and `update_one` refuses a node whose path reaches past the end of one. That gap narrows
/// both the items the frame holds and the branch it closes.
const FRAME_TAIL: usize =
    MAX_BRANCH_RLP + ((16 - 1) * MAX_ITEM + 1) + list_rlp((16 - 1) * MAX_ITEM + 2);

/// Bytes the scratch reserves, which is the most a descent can hold at once.
///
/// Keys are 32-byte hashes, so a path is [`MAX_NIBBLES`] nibbles and every frame consumes at least
/// one. A descent is therefore at most that many frames deep, and the sum is the worst each frame
/// holds while the one under it runs. Every term comes from the trie and from RLP, so the figure
/// does not move with the gas limit or with anything else a block chooses.
///
/// The tests forge a witness that holds every byte of this, so none of it is slack.
pub(super) const STACK_LEN: usize = {
    let mut total = FRAME_TAIL;
    let mut depth = 0;
    while depth < MAX_NIBBLES - 1 {
        total += frame_carry(depth);
        depth += 1;
    }
    total
};

#[cfg(test)]
std::thread_local! {
    /// Most bytes a descent has handed out at once, which only a build measuring itself records,
    /// so nothing the guest runs carries either the figure or the code that keeps it. One test
    /// runs on a thread of its own, which is what keeps the tests asserting on this from reading
    /// each other's descents.
    static PEAK: Cell<usize> = const { Cell::new(0) };
}

/// The peak since it was last taken, which leaves the descent that follows measuring from zero.
#[cfg(test)]
pub(super) fn take_peak() -> usize {
    PEAK.with(|peak| peak.replace(0))
}

/// A bump allocator that hands out the front of what is left and keeps the rest.
#[derive(Debug)]
pub(super) struct DynStack<'a> {
    /// Bytes not yet handed out.
    free: &'a mut [u8],
}

impl<'a> DynStack<'a> {
    /// A stack over `buffer`.
    pub(super) fn new(buffer: &'a mut [u8]) -> Self {
        Self { free: buffer }
    }

    /// Lends what is left, which is how a frame gives a child scratch it takes back.
    pub(super) fn lend(&mut self) -> DynStack<'_> {
        DynStack { free: self.free }
    }

    /// Scratch of `len` bytes, which holds whatever the last frame to be handed these bytes wrote.
    ///
    /// Every encoding fills what it asks for, so nothing reads scratch it has not written.
    pub(super) fn alloc(&mut self, len: usize) -> &'a mut [u8] {
        // The buffer is sized from the trie, so no descent reaches this. Builds that check their
        // assertions raise it where the derivation can still be corrected, and builds that do not
        // serve the request from the heap, which keeps a bound that turned out to be wrong a cost
        // in memory rather than a block rejected.
        debug_assert!(len <= self.free.len(), "a descent outgrew the scratch");
        // What the heap serves is recorded like any other request, so a descent that outgrew the
        // buffer shows as a peak above it rather than as one pinned to its size.
        #[cfg(test)]
        {
            let handed = STACK_LEN - self.free.len() + len;
            PEAK.with(|peak| peak.set(peak.get().max(handed)));
        }
        if len > self.free.len() {
            return Box::leak(vec![0; len].into_boxed_slice());
        }
        let (out, free) = core::mem::take(&mut self.free).split_at_mut(len);
        self.free = free;
        out
    }

    /// Scratch holding a copy of `src`.
    pub(super) fn alloc_copy(&mut self, src: &[u8]) -> &'a [u8] {
        let out = self.alloc(src.len());
        out.copy_from_slice(src);
        out
    }
}

#[cfg(test)]
mod tests {
    use crate::guest::path::stack::{DynStack, STACK_LEN, take_peak};

    /// The figures the buffer is sized from, pinned so a change to the derivation has to be
    /// deliberate. Each part is checked and not only the total, since two compensating errors would
    /// leave the total alone. A frame narrows with depth because the extension it re-encodes covers
    /// only what a key leaves below it.
    #[test]
    fn the_scratch_is_the_size_the_bound_derives() {
        use crate::guest::path::stack::{BRANCH_CARRY, FRAME_TAIL, frame_carry};
        assert_eq!((BRANCH_CARRY, FRAME_TAIL), (1027, 1528));
        assert_eq!((frame_carry(0), frame_carry(62)), (1128, 1095));
        assert_eq!(STACK_LEN, 71589);
    }

    #[test]
    fn allocations_are_disjoint_and_writable() {
        let mut buffer = [0; 64];
        let mut stack = DynStack::new(&mut buffer);
        let first = stack.alloc(8);
        first.fill(1);
        let second = stack.alloc(8);
        second.fill(2);
        assert_eq!(first, &[1; 8]);
        assert_eq!(second, &[2; 8]);
        assert_eq!(take_peak(), STACK_LEN - 48);
    }

    #[test]
    fn alloc_copy_returns_the_source_bytes() {
        let mut buffer = [0; 64];
        assert_eq!(
            DynStack::new(&mut buffer).alloc_copy(&[7, 8, 9]),
            &[7, 8, 9]
        );
    }

    /// What a frame lends comes back when the loan ends, so the same addresses serve every child in
    /// turn without the frame having to ask for them back.
    #[test]
    fn a_loan_is_returned_when_it_ends() {
        let mut buffer = [0; 64];
        let mut stack = DynStack::new(&mut buffer);
        let first = stack.lend().alloc(8).as_ptr();
        assert_eq!(stack.lend().alloc(8).as_ptr(), first);
        // What the frame takes for itself is not part of any loan, so the next loan starts past it.
        stack.alloc(8);
        assert_ne!(stack.lend().alloc(8).as_ptr(), first);
    }

    /// A request the buffer cannot hold is one the derivation says cannot arise, so where
    /// assertions are checked it is raised rather than absorbed.
    #[test]
    #[should_panic(expected = "a descent outgrew the scratch")]
    #[cfg(debug_assertions)]
    fn a_request_past_the_buffer_is_raised_where_assertions_are_checked() {
        let mut buffer = [0; 64];
        DynStack::new(&mut buffer).alloc(65);
    }

    /// Where they are not, it is served rather than refused, so a descent that outgrew the bound
    /// would cost memory rather than cost a block its validation, and the buffer is left whole for
    /// what follows.
    #[test]
    #[cfg(not(debug_assertions))]
    fn a_request_past_the_buffer_is_served_from_the_heap() {
        let mut buffer = [0; 64];
        let mut stack = DynStack::new(&mut buffer);
        assert_eq!(stack.alloc(65), &[0; 65]);
        assert_eq!(stack.alloc(64).len(), 64);
    }
}
