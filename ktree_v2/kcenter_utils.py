# kcenter_utils.py
import numpy as np
from numba import njit, prange
from typing import Tuple
from search_utils import _ham_row

IDX_DTYPE = np.int16  # shard < 2^15-1 → int16 everywhere
OFF_DTYPE = np.int32  # safe for prefix sums

# ---------- Greedy k-center (farthest-first, Hamming) ----------
# (2-approx primitive widely used for k-center)  [Gonzalez 1985]   :contentReference[oaicite:1]{index=1}

@njit(nogil=True)
def ham_greedy_kcenter_indices(codes: np.ndarray, K: int, start_index: int = 0) -> np.ndarray:
    N = codes.shape[0]
    if K <= 0 or N == 0:
        return np.empty(0, dtype=IDX_DTYPE)
    if K > N: K = N
    centers = np.empty(K, dtype=IDX_DTYPE)
    mind = np.empty(N, dtype=np.int16)
    for i in range(N): mind[i] = np.int16(0x3FFF)  # big sentinel

    c0 = np.int32(start_index)
    centers[0] = np.int16(c0)
    for i in range(N):
        mind[i] = _ham_row(codes[i], codes[c0])

    for t in range(1, K):
        far_idx = 0
        far_val = np.int16(-1)
        for i in range(N):
            v = mind[i]
            if v > far_val:
                far_val = v; far_idx = i
        centers[t] = np.int16(far_idx)
        for i in range(N):
            d = _ham_row(codes[i], codes[far_idx])
            if d < mind[i]: mind[i] = d
    return centers

@njit(parallel=True, nogil=True)
def ham_assign_top1_to_centers(codes: np.ndarray, centers_idx: np.ndarray) -> np.ndarray:
    N = codes.shape[0]; K = centers_idx.shape[0]
    if K == 0:
        return np.empty(N, dtype=IDX_DTYPE)
    out = np.empty(N, dtype=IDX_DTYPE)
    for i in prange(N):
        best_d = np.int16(0x3FFF); best_k = np.int16(0)
        for t in range(K):
            cidx = int(centers_idx[t])
            d = _ham_row(codes[i], codes[cidx])
            if d < best_d:
                best_d = d; best_k = np.int16(t)
        out[i] = best_k
    return out

@njit(parallel=True, nogil=True)
def ham_assign_top1_by_codes(local_codes: np.ndarray, centers_codes: np.ndarray) -> np.ndarray:
    # Assign each local_codes[i] to nearest row in centers_codes
    N = local_codes.shape[0]; K = centers_codes.shape[0]
    out = np.empty(N, dtype=IDX_DTYPE)
    for i in prange(N):
        best_d = np.int16(0x3FFF); best_k = np.int16(0)
        for t in range(K):
            d = _ham_row(local_codes[i], centers_codes[t])
            if d < best_d:
                best_d = d; best_k = np.int16(t)
        out[i] = best_k
    return out

# ---------- CSR helpers ----------

@njit(nogil=True)
def _count_hist(labels: np.ndarray, K: int) -> np.ndarray:
    sizes = np.zeros(K, dtype=OFF_DTYPE)
    for i in range(labels.shape[0]):
        k = int(labels[i])
        sizes[k] += 1
    return sizes

@njit(nogil=True)
def _prefix_from_sizes(sizes: np.ndarray) -> np.ndarray:
    K = sizes.shape[0]
    offs = np.empty(K + 1, dtype=OFF_DTYPE)
    offs[0] = 0
    for k in range(K):
        offs[k+1] = offs[k] + sizes[k]
    return offs

@njit(nogil=True)
def _scatter_members(labels: np.ndarray, offs: np.ndarray) -> np.ndarray:
    N = labels.shape[0]
    K = offs.shape[0] - 1
    cur = offs.copy()
    mem = np.empty(N, dtype=IDX_DTYPE)
    for i in range(N):
        k = int(labels[i])
        p = cur[k]
        mem[p] = np.int16(i)
        cur[k] = p + 1
    return mem

# ---------- Build: L1 global + L2 per L1 ----------

def build_index_localL2(
    codes_u64: np.ndarray,
    k1: int,
    k2_per_l1: int,
    start_l1: int = 0,
    start_l2: int = 0,  # kept for API parity
) -> dict:
    """
    Build 2-layer index where L2 is trained *inside each L1* cluster.

    Returns dict with:
      C1_codes (k1,W), C2_codes (K2tot,W)
      A1_pos (N,), A2_pos_global (N,)
      l1_offsets (k1+1,), l1_members (N,)
      l2c_offsets (k1+1,)  # slice of C2 per c1
      l2_post_offsets (K2tot+1,), l2_members (N,)
    All indices are int16 (IDs) and int32 (offsets). Assumes N < 32767.
    """
    N = codes_u64.shape[0]
    assert N < 32767, "Shard too big for int16 IDs"

    # L1
    centers1_idx = ham_greedy_kcenter_indices(codes_u64, int(k1), start_l1)
    A1_pos_local = ham_assign_top1_to_centers(codes_u64, centers1_idx)  # positions 0..k1-1
    C1_codes = codes_u64[centers1_idx.astype(np.int32)].copy()

    # L1 postings CSR
    l1_sizes = _count_hist(A1_pos_local.astype(np.int32), k1)
    l1_offsets = _prefix_from_sizes(l1_sizes)
    l1_members = _scatter_members(A1_pos_local.astype(np.int32), l1_offsets)

    # First pass: decide how many L2 centers total
    K2tot = 0
    for c1 in range(k1):
        m = int(l1_sizes[c1])
        if m > 0:
            K2tot += min(k2_per_l1, m)
    if K2tot == 0:
        # degenerate: no docs
        return {
            "C1_codes": C1_codes, "C2_codes": np.empty((0, codes_u64.shape[1]), dtype=np.uint64),
            "A1_pos": A1_pos_local.astype(IDX_DTYPE), "A2_pos_global": np.empty(N, dtype=IDX_DTYPE),
            "l1_offsets": l1_offsets, "l1_members": l1_members,
            "l2c_offsets": np.zeros(k1+1, dtype=OFF_DTYPE),
            "l2_post_offsets": np.zeros(1, dtype=OFF_DTYPE), "l2_members": np.empty(0, dtype=IDX_DTYPE),
            "N": np.int32(N)
        }

    # Allocate C2 and per-L1 slices
    W = codes_u64.shape[1]
    C2_codes = np.zeros((K2tot, W), dtype=np.uint64)
    l2c_offsets = np.empty(k1 + 1, dtype=OFF_DTYPE)
    l2c_offsets[0] = 0
    for c1 in range(k1):
        m = int(l1_sizes[c1])
        k2c = min(k2_per_l1, m) if m > 0 else 0
        l2c_offsets[c1+1] = l2c_offsets[c1] + k2c

    # Assign L2 centers per L1, and collect A2 labels
    A2_pos_global = np.empty(N, dtype=IDX_DTYPE)
    # temporary sizes for L2 postings
    l2_sizes = np.zeros(K2tot, dtype=OFF_DTYPE)

    for c1 in range(k1):
        s, e = int(l1_offsets[c1]), int(l1_offsets[c1+1])
        m = e - s
        if m == 0:
            continue
        ids = l1_members[s:e].astype(np.int32)  # global doc ids in this L1
        codes_local = codes_u64[ids]
        k2c = int(l2c_offsets[c1+1] - l2c_offsets[c1])
        if k2c <= 0:
            continue

        # greedy k-center on local slice
        # we need indices relative to local slice
        centers_local_idx = ham_greedy_kcenter_indices(codes_local, k2c, start_l2)
        centers_local_codes = codes_local[centers_local_idx.astype(np.int32)]
        # materialize into global C2
        g0 = int(l2c_offsets[c1])
        for t in range(k2c):
            C2_codes[g0 + t, :] = centers_local_codes[t, :]

        # local assignment -> labels in [0..k2c-1]
        lab2_local = ham_assign_top1_by_codes(codes_local, centers_local_codes)
        # write global A2 ids and sizes
        for i in range(m):
            g_c2 = g0 + int(lab2_local[i])
            A2_pos_global[ids[i]] = np.int16(g_c2)
            l2_sizes[g_c2] += 1

    # L2 postings CSR across all local L2s (concatenated)
    l2_post_offsets = _prefix_from_sizes(l2_sizes)
    l2_members = np.empty(N, dtype=IDX_DTYPE)
    cur = l2_post_offsets.copy()
    for c1 in range(k1):
        s, e = int(l1_offsets[c1]), int(l1_offsets[c1+1])
        m = e - s
        if m == 0:
            continue
        ids = l1_members[s:e].astype(np.int32)
        g0 = int(l2c_offsets[c1])
        # recompute local assignments quickly (small k2c) to avoid storing them twice
        k2c = int(l2c_offsets[c1+1] - l2c_offsets[c1])
        if k2c == 0:
            continue
        centers_local_codes = C2_codes[g0:g0+k2c]
        lab2_local = ham_assign_top1_by_codes(codes_u64[ids], centers_local_codes)
        for i in range(m):
            g_c2 = g0 + int(lab2_local[i])
            p = cur[g_c2]
            l2_members[p] = np.int16(ids[i])
            cur[g_c2] = p + 1

    return {
        "C1_codes": C1_codes,
        "C2_codes": C2_codes,
        "A1_pos": A1_pos_local.astype(IDX_DTYPE),
        "A2_pos_global": A2_pos_global,
        "l1_offsets": l1_offsets,
        "l1_members": l1_members,
        "l2c_offsets": l2c_offsets,
        "l2_post_offsets": l2_post_offsets,
        "l2_members": l2_members,
        "N": np.int32(N),
    }
