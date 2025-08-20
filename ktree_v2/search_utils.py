# search_utils.py
import numpy as np
from numba import njit, prange

# ===== bit packing & popcount =====

@njit(parallel=True, nogil=True)
def pack_signs_to_uint64(proj: np.ndarray) -> np.ndarray:
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

@njit(inline='always', nogil=True)
def _ham_row(a: np.ndarray, b: np.ndarray) -> np.int16:
    s = 0
    W = a.shape[0]
    for w in range(W):
        s += int(popcount_u64(a[w] ^ b[w]))
    # shard distance fits easily in int16 for codes <= 4096 bits
    return np.int16(s)

# ===== tiny utils =====

@njit(nogil=True)
def _maxpos_i16(arr: np.ndarray) -> int:
    m = arr[0]
    p = 0
    for i in range(1, arr.shape[0]):
        v = arr[i]
        if v > m:
            m = v
            p = i
    return p

@njit(nogil=True)
def _insort_i16(keys: np.ndarray, vals: np.ndarray):
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

# ===== Hamming center selection =====

@njit(nogil=True)
def ham_top_p_centers(q: np.ndarray, centers_codes: np.ndarray, p: int) -> np.ndarray:
    K = centers_codes.shape[0]
    if p > K: p = K
    d = np.empty(K, dtype=np.int16)
    idx = np.empty(K, dtype=np.int16)
    for k in range(K):
        d[k] = _ham_row(q, centers_codes[k])
        idx[k] = np.int16(k)
    _insort_i16(d, idx)
    out = np.empty(p, dtype=np.int16)
    for i in range(p):
        out[i] = idx[i]
    return out

@njit(nogil=True)
def ham_top_p_subset(q: np.ndarray, centers_codes: np.ndarray, subset_ids: np.ndarray, p: int) -> np.ndarray:
    m = subset_ids.shape[0]
    if m == 0:
        return np.empty(0, dtype=np.int16)
    if p > m: p = m
    d = np.empty(m, dtype=np.int16)
    idx = np.empty(m, dtype=np.int16)
    for t in range(m):
        cid = int(subset_ids[t])
        d[t] = _ham_row(q, centers_codes[cid])
        idx[t] = np.int16(t)
    _insort_i16(d, idx)
    out = np.empty(p, dtype=np.int16)
    for i in range(p):
        out[i] = subset_ids[idx[i]]
    return out

# ===== Candidate materialization (CSR postings) =====

def materialize_candidates_or(
    S1: np.ndarray,
    S2: np.ndarray,
    N: int,
    l1_offsets: np.ndarray,
    l1_members: np.ndarray,
    l2_post_offsets: np.ndarray,
    l2_members: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(N, dtype=bool)
    # L1 union
    for c1 in S1.astype(np.int32):
        s, e = int(l1_offsets[c1]), int(l1_offsets[c1+1])
        if e > s:
            mask[l1_members[s:e]] = True
    # L2 union
    for c2 in S2.astype(np.int32):
        s, e = int(l2_post_offsets[c2]), int(l2_post_offsets[c2+1])
        if e > s:
            mask[l2_members[s:e]] = True
    return np.nonzero(mask)[0].astype(np.int32)

def materialize_candidates_and(
    S1: np.ndarray,
    S2: np.ndarray,
    N: int,
    l1_offsets: np.ndarray,
    l1_members: np.ndarray,
    A2_pos_global: np.ndarray,
) -> np.ndarray:
    # compute S1 mask; then filter by A2 in S2
    mask1 = np.zeros(N, dtype=bool)
    s2mask = np.zeros(int(A2_pos_global.max()) + 1 if S2.size else 1, dtype=bool)
    for c1 in S1.astype(np.int32):
        s, e = int(l1_offsets[c1]), int(l1_offsets[c1+1])
        if e > s:
            mask1[l1_members[s:e]] = True
    for c2 in S2.astype(np.int32):
        if c2 < s2mask.shape[0]:
            s2mask[c2] = True
    if not mask1.any() or not s2mask.any():
        return np.empty(0, dtype=np.int32)
    idx = np.nonzero(mask1)[0]
    idx = idx[s2mask[A2_pos_global[idx]]]
    return idx.astype(np.int32)

# ===== Cascade probing (OR/AND) =====

def cascade_candidates(
    q_code: np.ndarray,
    index: dict,
    p1: int,
    p2: int,
    use_and: bool = False,
) -> np.ndarray:
    C1 = index["C1_codes"]             # (k1, W) uint64
    C2 = index["C2_codes"]             # (K2tot, W) uint64
    N  = int(index["N"])
    l1_offsets = index["l1_offsets"]   # (k1+1,) int32
    l1_members = index["l1_members"]   # (N,)    int16
    l2c_offsets = index["l2c_offsets"] # (k1+1,) int32  slice of C2 per c1
    l2post_offs = index["l2_post_offsets"]  # (K2tot+1,) int32
    l2_members  = index["l2_members"]       # (N,) int16
    A2_pos_g    = index["A2_pos_global"]    # (N,) int16

    # L1 probe
    S1 = ham_top_p_centers(q_code, C1, p1)   # int16 ids in [0..k1)
    if S1.size == 0:
        return np.empty(0, dtype=np.int32)

    # gather all local L2 ids under selected coarse centers
    total = 0
    for c1 in S1.astype(np.int32):
        total += int(l2c_offsets[c1+1] - l2c_offsets[c1])
    S2_subset = np.empty(total, dtype=np.int16)
    pos = 0
    for c1 in S1.astype(np.int32):
        s, e = int(l2c_offsets[c1]), int(l2c_offsets[c1+1])
        if e > s:
            L = e - s
            S2_subset[pos:pos+L] = np.arange(s, e, dtype=np.int16)
            pos += L
    if pos == 0:
        # no fine centers under chosen L1 (degenerate)
        return materialize_candidates_or(S1, np.empty(0, np.int16), N, l1_offsets, l1_members, l2post_offs, l2_members)

    S2 = ham_top_p_subset(q_code, C2, S2_subset[:pos], p2)

    if use_and:
        return materialize_candidates_and(S1, S2, N, l1_offsets, l1_members, A2_pos_g)
    else:
        return materialize_candidates_or(S1, S2, N, l1_offsets, l1_members, l2post_offs, l2_members)

# ===== Float -> binary helper (optional) =====
def binary_quantize_batch(x: np.ndarray, Q: np.ndarray) -> np.ndarray:
    # x: (N,D) float32; Q: (D,D) orthonormal
    proj = (x @ Q).astype(np.float32, copy=False)
    return pack_signs_to_uint64(proj)
