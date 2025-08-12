import numpy as np
from numba import njit, prange
from numba.extending import register_jitable

@register_jitable(inline='always')
def popcount_u64(x):
    """
    SWAR popcount on a uint64 value:
    Performs pairwise, 2-bit, then 4-bit summations and folds bytes into the top byte.
    """
    # Stage 1: subtract pairwise bits
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    # Stage 2: sum per 2 bits
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    # Stage 3: sum per 4 bits
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    # Stage 4: accumulate into top byte
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)

@njit(parallel=True, nogil=True)
def batch_hamming(docs, queries, out):
    """
    Compute pairwise Hamming distances between
    docs and queries, both uint64-packed arrays.
    docs: shape (n_docs, n_words)
    queries: shape (n_queries, n_words)
    out: pre-allocated array (n_docs, n_queries)
    """
    D, Q, W = docs.shape[0], queries.shape[0], docs.shape[1]
    for i in prange(D):
        for j in range(Q):
            s = np.uint64(0)
            for w in range(W):
                # XOR then popcount, inlined via register_jitable
                s += popcount_u64(docs[i, w] ^ queries[j, w])
            out[i, j] = s

    return out