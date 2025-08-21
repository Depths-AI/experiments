# binary_utils.py
import numpy as np
from numba import njit, prange

# ---------- bit pack & popcount ----------

@njit(parallel=True, nogil=True, cache=True)
def pack_signs_to_uint64(proj: np.ndarray) -> np.ndarray:
    """
    proj: (N, D) float32
    return: bit-packed (N, ceil(D/64)) uint64
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

@njit(nogil=True)
def _insort_i32(keys: np.ndarray, vals: np.ndarray):
    L = keys.shape[0]
    for i in range(1, L):
        key = keys[i]; val = vals[i]
        j = i - 1
        while j >= 0 and keys[j] > key:
            keys[j + 1] = keys[j]; vals[j + 1] = vals[j]
            j -= 1
        keys[j + 1] = key; vals[j + 1] = val

@njit(nogil=True)
def hamming_top_p_centers(q_code: np.ndarray, centers_codes: np.ndarray, p: int) -> np.ndarray:
    K = centers_codes.shape[0]
    if p > K: p = K
    d = np.empty(K, dtype=np.int32)
    idx = np.empty(K, dtype=np.int32)
    for k in range(K):
        d[k] = _ham_row(q_code, centers_codes[k])
        idx[k] = k
    _insort_i32(d, idx)
    out = np.empty(p, dtype=np.int32)
    for i in range(p): out[i] = idx[i]
    return out

@njit(nogil=True)
def hamming_top_p_subset(q_code: np.ndarray, centers_codes: np.ndarray, subset_ids: np.ndarray, p: int) -> np.ndarray:
    m = subset_ids.shape[0]
    if m == 0: return subset_ids
    if p > m: p = m
    d = np.empty(m, dtype=np.int32)
    idx = np.empty(m, dtype=np.int32)
    for t in range(m):
        cid = int(subset_ids[t])
        d[t] = _ham_row(q_code, centers_codes[cid]); idx[t] = t
    _insort_i32(d, idx)
    out = np.empty(p, dtype=np.int32)
    for i in range(p): out[i] = subset_ids[idx[i]]
    return out

# ---------- float -> binary (random orthonormal) ----------

def binary_quantize_batch(x: np.ndarray, Q: np.ndarray | None = None) -> np.ndarray:
    """
    x: (N,D) float32; Q: (D,D) orthonormal (if None, random via QR)
    return: bit-packed uint64 codes
    """
    N, D = x.shape
    if Q is None:
        rng = np.random.default_rng(0)
        A = rng.standard_normal((D, D), dtype=np.float32)
        Q, _ = np.linalg.qr(A)
    proj = (x @ Q).astype(np.float32, copy=False)
    return pack_signs_to_uint64(proj)
