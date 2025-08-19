# hamming_kcenter.py
import numpy as np
from numba import njit, prange
# Reuse your optimized popcount
from search import popcount_u64  # :contentReference[oaicite:2]{index=2}

# -----------------------
# Low-level Hamming utils
# -----------------------

@njit(inline='always', nogil=True)
def _hamming_row(a: np.ndarray, b: np.ndarray) -> np.int32:
    """
    Hamming distance between two bit-packed codes (W uint64 words each).
    a, b: shape (W,), dtype=uint64
    returns int32 in [0, 64*W]
    """
    s = np.int32(0)
    W = a.shape[0]
    for w in range(W):
        s += np.int32(popcount_u64(a[w] ^ b[w]))
    return s

@njit(nogil=True)
def _maxpos_i32(arr: np.ndarray) -> int:
    """
    Argmax for 1D int32 array.
    """
    m = arr[0]
    p = 0
    for i in range(1, arr.shape[0]):
        v = arr[i]
        if v > m:
            m = v
            p = i
    return p

@njit(nogil=True)
def _insertion_sort_by_key_i32(keys: np.ndarray, vals: np.ndarray):
    """
    Inplace insertion sort by int32 keys (ascending); vals are indices aligned to keys.
    """
    L = keys.shape[0]
    for i in range(1, L):
        key = keys[i]
        val = vals[i]
        j = i - 1
        while j >= 0 and keys[j] > key:
            keys[j + 1] = keys[j]
            vals[j + 1] = vals[j]
            j -= 1
        keys[j + 1] = key
        vals[j + 1] = val

# ------------------------------------------
# Greedy farthest-first k-center (Hamming)
# ------------------------------------------

@njit(nogil=True)
def hamming_greedy_k_center_indices(codes: np.ndarray,
                                    K: int,
                                    start_index: int = 0) -> np.ndarray:
    """
    Farthest-first k-center on bit-packed binary vectors (Hamming metric).
    - codes: (N, W) uint64   (W = ceil(num_bits/64); e.g., 1024 bits -> W=16)
    - K:     number of centers to pick (clamped to [1..N])
    - start_index: deterministic first center (e.g., 0 or an arbitrary seed)

    Returns:
      centers_idx: (K,) int32 indices into codes.
    """
    N = codes.shape[0]
    if K <= 0 or N == 0:
        return np.empty(0, dtype=np.int32)
    if K > N:
        K = N

    centers_idx = np.empty(K, dtype=np.int32)

    # Track min Hamming distance from each point to the current center set
    # Use int32 sentinel; Hamming max at 1024 bits is 1024 << 2^31
    min_d = np.empty(N, dtype=np.int32)
    for i in range(N):
        min_d[i] = 1 << 30  # "infinity"

    # First center
    c0 = np.int32(start_index)
    centers_idx[0] = c0
    # Initialize min_d with distances to c0
    for i in range(N):
        min_d[i] = _hamming_row(codes[i], codes[c0])

    # Iteratively pick the farthest point from the current center set
    for t in range(1, K):
        far_idx = _maxpos_i32(min_d)
        centers_idx[t] = far_idx

        # Update min_d with this new center
        for i in range(N):
            d = _hamming_row(codes[i], codes[far_idx])
            if d < min_d[i]:
                min_d[i] = d

    return centers_idx

# ---------------------------------------------------
# Assign each point to the top-L nearest Hamming centers
# ---------------------------------------------------

@njit(parallel=True, nogil=True)
def hamming_assign_labels_topL(codes: np.ndarray,
                               centers_idx: np.ndarray,
                               L: int = 1) -> np.ndarray:
    """
    For each point, return indices (into centers_idx) of the top-L nearest centers.
    - codes: (N, W) uint64
    - centers_idx: (K,) int32
    - L: number of centers to keep per point; clamped to K

    Returns:
      labels_topL: (N, L) int32  -- indices in [0..K-1], not absolute dataset indices
                                   (i.e., positions in centers_idx)
    """
    N = codes.shape[0]
    K = centers_idx.shape[0]
    if K == 0:
        return np.empty((N, 0), dtype=np.int32)
    if L > K:
        L = K

    labels_topL = np.empty((N, L), dtype=np.int32)

    for i in prange(N):
        best_d = np.empty(L, dtype=np.int32)
        best_k = np.empty(L, dtype=np.int32)
        # init with +inf
        for z in range(L):
            best_d[z] = 1 << 30
            best_k[z] = -1

        filled = 0
        worst_pos = 0

        # scan all centers
        for t in range(K):
            cidx = centers_idx[t]
            d = _hamming_row(codes[i], codes[cidx])

            if filled < L:
                best_d[filled] = d
                best_k[filled] = t
                filled += 1
                if filled == L:
                    worst_pos = _maxpos_i32(best_d)
            else:
                if d < best_d[worst_pos]:
                    best_d[worst_pos] = d
                    best_k[worst_pos] = t
                    worst_pos = _maxpos_i32(best_d)

        # sort the L slots by distance ascending (stable enough for small L)
        _insertion_sort_by_key_i32(best_d, best_k)
        labels_topL[i, :] = best_k

    return labels_topL

# ---------------------------------------------------
# Optional: majority-code center refinement (1–2 iters)
# ---------------------------------------------------

@njit(nogil=True)
def _bitwise_majority_code_span(codes: np.ndarray,
                                members: np.ndarray,
                                start: int,
                                end: int) -> np.ndarray:
    """
    Majority representative over members[start:end].
    codes:   (N, W) uint64
    members: (N,)   int32, cluster member indices laid out contiguously
    """
    W = codes.shape[1]
    M = end - start
    maj = np.zeros(W, dtype=np.uint64)

    if M <= 0:
        return maj

    # Fast path: single member → copy its code
    if M == 1:
        idx = members[start]
        for w in range(W):
            maj[w] = codes[idx, w]
        return maj

    # Bitwise majority per 64-bit word
    for w in range(W):
        word_majority = np.uint64(0)
        for bit in range(64):
            mask = np.uint64(1) << np.uint64(bit)
            cnt = 0
            for p in range(start, end):
                idx = members[p]
                if (codes[idx, w] & mask) != 0:
                    cnt += 1
            if cnt * 2 >= M:
                word_majority |= mask
        maj[w] = word_majority

    return maj


@njit(parallel=True, nogil=True)
def hamming_refine_centers_majority(codes: np.ndarray,
                                    centers_idx: np.ndarray,
                                    labels: np.ndarray) -> np.ndarray:
    """
    Replace each center with the cluster's bitwise-majority code (not necessarily a dataset point).
    - codes:       (N, W) uint64 (bit-packed)
    - centers_idx: (K,)   int32  (used for fallback on empty clusters)
    - labels:      (N,)   int32  single-assignment labels in [0..K-1]

    Returns:
      centers_codes: (K, W) uint64
    """
    N, W = codes.shape
    K = centers_idx.shape[0]
    centers_codes = np.zeros((K, W), dtype=np.uint64)

    # 1) Count sizes per cluster
    sizes = np.zeros(K, dtype=np.int32)
    for i in range(N):
        k = labels[i]
        # (Optional safety: skip invalid labels)
        if 0 <= k < K:
            sizes[k] += 1

    # 2) Build CSR-style offsets
    offsets = np.empty(K + 1, dtype=np.int32)
    offsets[0] = 0
    for k in range(K):
        offsets[k + 1] = offsets[k] + sizes[k]

    # 3) Fill contiguous members array
    members = np.empty(N, dtype=np.int32)
    cur = offsets.copy()
    for i in range(N):
        k = labels[i]
        if 0 <= k < K:
            pos = cur[k]
            members[pos] = i
            cur[k] = pos + 1

    # 4) Majority per cluster (parallelizable across k)
    for k in prange(K):
        start = offsets[k]
        end = offsets[k + 1]
        if start == end:
            # empty cluster → fallback to original chosen center's code
            base = centers_idx[k]
            for w in range(W):
                centers_codes[k, w] = codes[base, w]
        else:
            centers_codes[k, :] = _bitwise_majority_code_span(codes, members, start, end)

    return centers_codes

# ---------------------------------------------------
# Convenience: top-p nearest centers to a query code
# ---------------------------------------------------

@njit(nogil=True)
def hamming_top_p_centers(query_code: np.ndarray,
                          centers_codes: np.ndarray,
                          p: int) -> np.ndarray:
    """
    Return indices [0..K-1] of the p nearest centers to 'query_code' in Hamming metric.
    - query_code:   (W,) uint64
    - centers_codes:(K, W) uint64
    - p:            int  (clamped to K)
    """
    K = centers_codes.shape[0]
    if p > K:
        p = K
    d = np.empty(K, dtype=np.int32)
    idx = np.empty(K, dtype=np.int32)
    for k in range(K):
        d[k] = _hamming_row(query_code, centers_codes[k])
        idx[k] = k
    _insertion_sort_by_key_i32(d, idx)
    out = np.empty(p, dtype=np.int32)
    for i in range(p):
        out[i] = idx[i]
    return out
