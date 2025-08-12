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

@njit(parallel=True, nogil=True, cache=True)
def binary_search_kernel(docs, queries, k):
    """
    A unified Numba kernel that computes distances and finds top-k
    results simultaneously, avoiding a large intermediate distance matrix.
    """
    Q, D, W = queries.shape[0], docs.shape[0], docs.shape[1]

    # Pre-allocate final output arrays for indices and distances
    # Shape: (num_queries, k)
    top_k_indices = np.full((Q, k), -1, dtype=np.int64)
    top_k_distances = np.full((Q, k), np.iinfo(np.uint64).max, dtype=np.uint64)

    # Parallelize over queries
    for j in prange(Q):
        q_vec = queries[j]
        
        # These are temporary buffers for the current query's top-k
        current_top_indices = top_k_indices[j]
        current_top_distances = top_k_distances[j]
        
        # Iterate over all documents for the current query
        for i in range(D):
            # 1. Calculate Hamming Distance
            dist = np.uint64(0)
            for w in range(W):
                dist += popcount_u64(docs[i, w] ^ q_vec[w])

            # 2. Perform Top-K selection (like a manual argpartition)
            # Find the largest distance currently in our top-k list
            if dist < current_top_distances[-1]:
                # If the new distance is smaller, we need to insert it
                # and keep the list sorted.
                
                # Find insertion point
                insertion_point = np.searchsorted(current_top_distances, dist)
                
                # Shift elements to the right to make space
                for idx in range(k - 1, insertion_point, -1):
                    current_top_distances[idx] = current_top_distances[idx-1]
                    current_top_indices[idx] = current_top_indices[idx-1]
                
                # Insert the new distance and index
                current_top_distances[insertion_point] = dist
                current_top_indices[insertion_point] = i

    return top_k_indices # We only need the indices as the final result