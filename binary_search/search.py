import numpy as np
from numba import njit, prange
from numba.extending import register_jitable
from heap import heap_push, heap_replace

@njit(nogil=True,cache=True)
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
        
        # --- Top-K selection using a Max-Heap ---
        
        # 1. Fill the heap with the first k documents
        for i in range(k):
            dist = np.uint64(0)
            for w in range(W):
                dist += popcount_u64(docs[i, w] ^ q_vec[w])
            heap_push(current_top_distances, current_top_indices, dist, i)

        # 2. For the rest of the documents, if a distance is smaller than the
        #    largest distance in the heap, replace it.
        for i in range(k, D):
            dist = np.uint64(0)
            for w in range(W):
                dist += popcount_u64(docs[i, w] ^ q_vec[w])
            
            # If the new distance is smaller than the largest in the heap
            if dist < current_top_distances[0]:
                heap_replace(current_top_distances, current_top_indices, dist, i)

    # Sort the results by distance for each query before returning
    for j in range(Q):
        sorted_indices = np.argsort(top_k_distances[j])
        top_k_distances[j] = top_k_distances[j][sorted_indices]
        top_k_indices[j] = top_k_indices[j][sorted_indices]

    return top_k_indices