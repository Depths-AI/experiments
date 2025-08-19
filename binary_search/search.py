from numba import njit, prange
import numpy as np

@njit(nogil=True)
def heap_push(heap_distances, heap_indices, dist, index, pos):
    '''
    Utility function to push a new item into a max-heap.
    This function assumes that the heap is already a valid max-heap and that the new item
    is larger than the current item at position `pos`.
    Args:
        heap_distances: (K,) float64, distances in the heap
        heap_indices: (K,) int64, indices corresponding to the distances
        dist: float64, new distance to be added
        index: int64, index corresponding to the new distance
        pos: int, position in the heap where the new item should be inserted
    Returns:
        None, modifies the heap in place
    '''
    heap_distances[pos] = dist
    heap_indices[pos] = index
    
    while pos > 0:
        parent_pos = (pos - 1) // 2
        if heap_distances[pos] > heap_distances[parent_pos]:
            heap_distances[pos], heap_distances[parent_pos] = heap_distances[parent_pos], heap_distances[pos]
            heap_indices[pos], heap_indices[parent_pos] = heap_indices[parent_pos], heap_indices[pos]
            pos = parent_pos
        else:
            break

@njit(nogil=True)
def heap_replace(heap_distances, heap_indices, dist, index):
    '''
    Utility function to replace the root of a max-heap with a new item.
    This function assumes that the new item is larger than the current root.
    Args:
        heap_distances: (K,) float64, distances in the heap
        heap_indices: (K,) int64, indices corresponding to the distances
        dist: float64, new distance to be added
        index: int64, index corresponding to the new distance
    Returns:
        None, modifies the heap in place
    '''
    # Replace the root of the heap
    heap_distances[0] = dist
    heap_indices[0] = index
    
    # Sift down to maintain the heap property
    pos = 0
    k = heap_distances.shape[0]
    while True:
        left_child = 2 * pos + 1
        right_child = 2 * pos + 2
        largest = pos
        
        if left_child < k and heap_distances[left_child] > heap_distances[largest]:
            largest = left_child
        
        if right_child < k and heap_distances[right_child] > heap_distances[largest]:
            largest = right_child
        
        if largest != pos:
            heap_distances[pos], heap_distances[largest] = heap_distances[largest], heap_distances[pos]
            heap_indices[pos], heap_indices[largest] = heap_indices[largest], heap_indices[pos]
            pos = largest
        else:
            break

@njit(nogil=True,cache=True)
def popcount_u64(x):
    '''
    Utility function to count the number of set bits in a 64-bit unsigned integer.
    This function uses a bitwise algorithm to efficiently count bits.
    Args:
        x: np.uint64, the input number
    Returns:
        np.uint64, the count of set bits in x
    '''
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
    '''
    Utility to perform efficient top-k search for a batch of queries against a set of documents
    by computing Hamming distances.
    Args:
        docs: (D, W) np.ndarray, binary document vectors 
        queries: (Q, W) np.ndarray, binary query vectors
        k: int, number of top results to return for each query
    Returns:
        top_k_indices: (Q, k) np.ndarray, indices of the top-k closest documents for each query
    '''
    Q, D, W = queries.shape[0], docs.shape[0], docs.shape[1]
    
    top_k_indices = np.full((Q, k), -1, dtype=np.int32)
    top_k_distances = np.full((Q, k), np.iinfo(np.int16).max, dtype=np.int16)

    for j in prange(Q):
        q_vec = queries[j]
        
        current_top_indices = top_k_indices[j]
        current_top_distances = top_k_distances[j]
        
        for i in range(D):
            dist = np.int16(0)
            for w in range(W):
                dist += popcount_u64(docs[i, w] ^ q_vec[w])

            if i < k:
                heap_push(current_top_distances, current_top_indices, dist, i, i)
            elif dist < current_top_distances[0]:
                heap_replace(current_top_distances, current_top_indices, dist, i)

    for j in range(Q):
        sorted_indices = np.argsort(top_k_distances[j])
        top_k_distances[j] = top_k_distances[j][sorted_indices]
        top_k_indices[j] = top_k_indices[j][sorted_indices]

    return top_k_indices, top_k_distances

@njit(parallel=True, nogil=True)
def pack_signs_to_uint64(proj):
    n, d = proj.shape
    nwords = (d + 63) // 64
    out = np.zeros((n, nwords), dtype=np.uint64)
    for i in prange(n):
        for j in range(d):
            if proj[i, j] >= 0.0:
                word = j >> 6
                bitpos = j & 63
                out[i, word] |= (np.uint64(1) << np.uint64(bitpos))
    return out
