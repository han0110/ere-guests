//! [`Error`], what a malformed encoding or a reference the witness cannot supply becomes.

use alloy_primitives::B256;
use reth_tries::WitnessDbError;

#[derive(Debug, thiserror::Error)]
pub(super) enum Error {
    /// The witness holds no node for a hash a walk had to follow.
    #[error("reached an unresolved node: {0:#}")]
    NodeNotResolved(B256),
    /// Errors related to RLP encoding and decoding.
    #[error("rlp decode error: {0}")]
    Rlp(#[from] alloy_rlp::Error),
    /// The value slot a branch holds beside its sixteen children was filled or needed. It answers a
    /// key that stops where others carry on, which 32-byte hashes never do.
    #[error("branch node with value")]
    ValueInBranch,
    /// A hex-prefix path was longer than a key, or an extension held none at all.
    #[error("malformed hex-prefix path")]
    MalformedPath,
    /// A node referenced a child through more bytes than a Keccak digest takes, which the canonical
    /// trie has no node for and which re-encoding the parent is sized against.
    #[error("node reference longer than a digest")]
    OversizedItem,
    /// A leaf carried more than an account's worth of value, which neither trie the guest walks
    /// holds and which re-encoding the leaf is sized against.
    #[error("leaf value longer than an account")]
    OversizedValue,
    /// A node came out longer than the widest the canonical trie holds, which the scratch a frame
    /// keeps one in is sized against.
    #[error("node longer than a branch")]
    OversizedNode,
}

impl From<Error> for WitnessDbError {
    fn from(error: Error) -> Self {
        match error {
            Error::Rlp(error) => Self::Rlp(error),
            error => Self::TrieWitness(alloc::format!("{error}")),
        }
    }
}
