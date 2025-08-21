# binary_utils.py
import numpy as np
from numba import njit, prange

# ---------- bit pack & popcount ----------

@njit(parallel=True, nogil=True, cache=True)
def pack_signs_to_uint64(proj: np.ndarray) -> np.ndarray:
    """
    proj: (N, D) float32 -> bit-packed (N, ceil(D/64)) uint64
    """
    n, d = proj.shape
    w = (d + 63) // 64
    out = np.zeros((n, w), dtype=np.uint64)
    for i in prange(n):
        for j in range(d):
            if proj[i, j] >= 0.0:
                out[i, j >> 6] |= (np.uint64(1) << np.uint64(j & 63))
    return out

@njit(nogil=True, cache=True)
def popcount_u64(x: np.uint64) -> np.uint64:
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)

# ---------- Hamming core ----------

@njit(inline='always', nogil=True)
def _ham_row(a: np.ndarray, b: np.ndarray) -> np.int32:
    s = 0
    W = a.shape[0]
    for w in range(W):
        s += int(popcount_u64(a[w] ^ b[w]))
    return np.int32(s)

# ---------- top-p by size-p max-heap (O(K log p)) ----------

@njit(nogil=True)
def _sift_down(vals: np.ndarray, ids: np.ndarray, i: int, size: int):
    # max-heap by vals
    while True:
        l = 2 * i + 1
        r = l + 1
        m = i
        if l < size and vals[l] > vals[m]:
            m = l
        if r < size and vals[r] > vals[m]:
            m = r
        if m == i:
            break
        v = vals[i]; vals[i] = vals[m]; vals[m] = v
        t = ids[i];  ids[i]  = ids[m];  ids[m]  = t
        i = m

@njit(nogil=True)
def hamming_top_p_centers(q_code: np.ndarray, centers_codes: np.ndarray, p: int) -> np.ndarray:
    """
    Return indices of the p nearest centers to q_code (by Hamming).
    Uses a size-p max-heap: O(K log p). Output sorted by distance (asc).
    """
    K = centers_codes.shape[0]
    if K == 0 or p <= 0:
        return np.empty(0, dtype=np.int32)
    if p > K: p = K

    heap_vals = np.empty(p, dtype=np.int32)
    heap_ids  = np.empty(p, dtype=np.int32)

    # seed heap with first p
    for i in range(p):
        heap_vals[i] = _ham_row(q_code, centers_codes[i])
        heap_ids[i]  = i
    # heapify
    for i in range(p // 2 - 1, -1, -1):
        _sift_down(heap_vals, heap_ids, i, p)

    # scan remainder
    for k in range(p, K):
        d = _ham_row(q_code, centers_codes[k])
        if d < heap_vals[0]:
            heap_vals[0] = d
            heap_ids[0]  = k
            _sift_down(heap_vals, heap_ids, 0, p)

    # order heap by distance asc
    order = np.argsort(heap_vals)
    out = np.empty(p, dtype=np.int32)
    for i in range(p):
        out[i] = heap_ids[order[i]]
    return out

@njit(nogil=True)
def hamming_top_p_subset(q_code: np.ndarray, centers_codes: np.ndarray, subset_ids: np.ndarray, p: int) -> np.ndarray:
    """
    Return p nearest indices from the provided subset_ids.
    Also heap-based O(m log p) where m=len(subset_ids).
    """
    m = subset_ids.shape[0]
    if m == 0 or p <= 0:
        return np.empty(0, dtype=np.int32)
    if p > m: p = m

    heap_vals = np.empty(p, dtype=np.int32)
    heap_ids  = np.empty(p, dtype=np.int32)

    # seed from first p subset entries
    for i in range(p):
        cid = int(subset_ids[i])
        heap_vals[i] = _ham_row(q_code, centers_codes[cid])
        heap_ids[i]  = cid
    for i in range(p // 2 - 1, -1, -1):
        _sift_down(heap_vals, heap_ids, i, p)

    # scan the rest
    for t in range(p, m):
        cid = int(subset_ids[t])
        d = _ham_row(q_code, centers_codes[cid])
        if d < heap_vals[0]:
            heap_vals[0] = d
            heap_ids[0]  = cid
            _sift_down(heap_vals, heap_ids, 0, p)

    order = np.argsort(heap_vals)
    out = np.empty(p, dtype=np.int32)
    for i in range(p):
        out[i] = heap_ids[order[i]]
    return out

# ---------- float -> binary (random orthonormal) ----------

def binary_quantize_batch(x: np.ndarray, Q: np.ndarray | None = None) -> np.ndarray:
    """
    x: (N,D) float32; Q: (D,D) orthonormal (if None, random via QR)
    return: bit-packed uint64 codes
    """
    _, D = x.shape
    if Q is None:
        rng = np.random.default_rng(0)
        A = rng.standard_normal((D, D), dtype=np.float32)
        Q, _ = np.linalg.qr(A)
    proj = (x @ Q).astype(np.float32, copy=False)
    return pack_signs_to_uint64(proj)
