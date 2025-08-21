# kcenter_utils.py
import numpy as np
from numba import njit, prange
from typing import Dict

from binary_utils import _ham_row, hamming_top_p_centers, hamming_top_p_subset

IDX_DTYPE = np.int16   # shard < 2^15-1
OFF_DTYPE = np.int32

# --------- Greedy farthest-first k-center (Hamming) ---------

@njit(nogil=True, cache=True)
def ham_greedy_kcenter_indices(codes: np.ndarray, K: int, start_index: int = 0) -> np.ndarray:
    N = codes.shape[0]
    if K <= 0: return np.empty(0, dtype=np.int32)
    if K > N:  K = N
    centers = np.empty(K, dtype=np.int32)
    mind = np.empty(N, dtype=np.int32)
    big = 1 << 30
    for i in range(N): mind[i] = big

    c0 = np.int32(start_index)
    centers[0] = c0
    for i in range(N): mind[i] = _ham_row(codes[i], codes[c0])

    for t in range(1, K):
        far_idx = 0; far_val = -1
        for i in range(N):
            v = mind[i]
            if v > far_val: far_val = v; far_idx = i
        centers[t] = far_idx
        for i in range(N):
            d = _ham_row(codes[i], codes[far_idx])
            if d < mind[i]: mind[i] = d
    return centers

@njit(parallel=True, nogil=True, cache=True)
def ham_assign_top1_to_centers(codes: np.ndarray, centers_idx: np.ndarray) -> np.ndarray:
    N = codes.shape[0]; K = centers_idx.shape[0]
    out = np.empty(N, dtype=IDX_DTYPE)
    for i in prange(N):
        best_d = 1 << 30; best_k = 0
        for t in range(K):
            cidx = int(centers_idx[t])
            d = _ham_row(codes[i], codes[cidx])
            if d < best_d: best_d = d; best_k = t
        out[i] = np.int16(best_k)
    return out

# --------- CSR helpers ---------

def _csr_from_labels(labels: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.zeros(K, dtype=OFF_DTYPE)
    for i in range(labels.shape[0]):
        sizes[int(labels[i])] += 1
    offs = np.empty(K + 1, dtype=OFF_DTYPE); offs[0] = 0
    for k in range(K): offs[k+1] = offs[k] + sizes[k]
    mem = np.empty(labels.shape[0], dtype=IDX_DTYPE)
    cur = offs.copy()
    for i in range(labels.shape[0]):
        k = int(labels[i]); p = cur[k]
        mem[p] = np.int16(i); cur[k] = p + 1
    return offs, mem

@njit(nogil=True, cache=True)
def _adj12_csr_from_postings_ts(A2_pos: np.ndarray,
                                l1_offsets: np.ndarray,
                                l1_members: np.ndarray,
                                k1: int, k2: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build adjacency with a timestamped visited array (no np.unique).
    For each coarse c1, collect unique A2 labels seen under its members.
    """
    sizes = np.zeros(k1, dtype=OFF_DTYPE)
    visited = np.zeros(k2, dtype=np.int32)
    stamp = 1

    # pass 1: count uniques per c1
    for c1 in range(k1):
        s = int(l1_offsets[c1]); e = int(l1_offsets[c1+1])
        cnt = 0
        for i in range(s, e):
            c2 = int(A2_pos[int(l1_members[i])])
            if visited[c2] != stamp:
                visited[c2] = stamp
                cnt += 1
        sizes[c1] = cnt
        stamp += 1

    # prefix
    offs = np.empty(k1 + 1, dtype=OFF_DTYPE)
    offs[0] = 0
    for c1 in range(k1):
        offs[c1+1] = offs[c1] + sizes[c1]

    # pass 2: fill indices (unsorted; order not required)
    idx = np.empty(int(offs[-1]), dtype=IDX_DTYPE)
    visited[:] = 0; stamp = 1
    for c1 in range(k1):
        s = int(l1_offsets[c1]); e = int(l1_offsets[c1+1])
        pos = offs[c1]
        for i in range(s, e):
            c2 = int(A2_pos[int(l1_members[i])])
            if visited[c2] != stamp:
                visited[c2] = stamp
                idx[pos] = np.int16(c2)
                pos += 1
        stamp += 1

    return offs, idx

# --------- Build: global L1, global L2, adjacency ---------

def build_two_layer_index(
    codes_u64: np.ndarray,
    k1: int,
    k2: int,
    start_index_l1: int = 0,
    start_index_l2: int = 0,
) -> Dict[str, object]:
    """
    Global L1 + Global L2 (both k-center on bit codes). Adjacency maps each coarse center
    to the set of fine centers seen under its posting list (unique A2 labels of its members).
    """
    N = codes_u64.shape[0]
    # L1
    c1_idx = ham_greedy_kcenter_indices(codes_u64, int(k1), start_index_l1)
    A1_pos = ham_assign_top1_to_centers(codes_u64, c1_idx)          # in [0..k1-1]
    C1_codes = codes_u64[c1_idx.astype(np.int32)].copy()
    l1_offsets, l1_members = _csr_from_labels(A1_pos.astype(np.int32), k1)

    # L2
    k2_eff = int(min(k2, N))
    c2_idx = ham_greedy_kcenter_indices(codes_u64, k2_eff, start_index_l2)
    A2_pos = ham_assign_top1_to_centers(codes_u64, c2_idx)          # in [0..k2_eff-1]
    C2_codes = codes_u64[c2_idx.astype(np.int32)].copy()
    l2_offsets, l2_members = _csr_from_labels(A2_pos.astype(np.int32), k2_eff)

    # adjacency (CSR) via timestamps
    adj12_offsets, adj12_indices = _adj12_csr_from_postings_ts(
        A2_pos.astype(np.int32), l1_offsets, l1_members, int(k1), k2_eff
    )

    return {
        "C1_codes": C1_codes, "C2_codes": C2_codes,
        "A1_pos": A1_pos.astype(IDX_DTYPE), "A2_pos": A2_pos.astype(IDX_DTYPE),
        "l1_offsets": l1_offsets, "l1_members": l1_members,
        "l2_offsets": l2_offsets, "l2_members": l2_members,
        "adj12_offsets": adj12_offsets, "adj12_indices": adj12_indices,
        "k1": int(k1), "k2": k2_eff, "N": np.int32(N),
    }

# --------- Query-time helpers (timestamps; no uniques/masks clears) ---------

@njit(nogil=True, cache=True)
def _gather_adj_subset_ts(S1: np.ndarray, adj_offs: np.ndarray, adj_idx: np.ndarray, k2: int) -> np.ndarray:
    visited = np.zeros(k2, dtype=np.int32)
    out = np.empty(k2, dtype=np.int32)  # worst case
    stamp = 1; pos = 0
    for t in range(S1.shape[0]):
        c1 = int(S1[t])
        s = int(adj_offs[c1]); e = int(adj_offs[c1+1])
        for j in range(s, e):
            c2 = int(adj_idx[j])
            if visited[c2] != stamp:
                visited[c2] = stamp
                out[pos] = c2; pos += 1
    return out[:pos]

@njit(nogil=True, cache=True)
def _materialize_or_ts(S1: np.ndarray, S2: np.ndarray,
                       l1_off: np.ndarray, l1_mem: np.ndarray,
                       l2_off: np.ndarray, l2_mem: np.ndarray,
                       N: int) -> np.ndarray:
    visited = np.zeros(N, dtype=np.int32)
    out = np.empty(N, dtype=np.int32)
    stamp = 1; pos = 0
    # L1 union
    for t in range(S1.shape[0]):
        c1 = int(S1[t]); s = int(l1_off[c1]); e = int(l1_off[c1+1])
        for i in range(s, e):
            did = int(l1_mem[i])
            if visited[did] != stamp:
                visited[did] = stamp; out[pos] = did; pos += 1
    # L2 union
    for t in range(S2.shape[0]):
        c2 = int(S2[t]); s = int(l2_off[c2]); e = int(l2_off[c2+1])
        for i in range(s, e):
            did = int(l2_mem[i])
            if visited[did] != stamp:
                visited[did] = stamp; out[pos] = did; pos += 1
    return out[:pos]

@njit(nogil=True, cache=True)
def _materialize_and_ts(S1: np.ndarray, S2: np.ndarray,
                        l1_off: np.ndarray, l1_mem: np.ndarray,
                        A2_pos: np.ndarray, k2: int, N: int) -> np.ndarray:
    if S2.shape[0] == 0 or S1.shape[0] == 0:
        return np.empty(0, dtype=np.int32)
    s2mark = np.zeros(k2, dtype=np.int32)
    for t in range(S2.shape[0]):
        s2mark[int(S2[t])] = 1
    visited = np.zeros(N, dtype=np.int32)
    out = np.empty(N, dtype=np.int32)
    stamp = 1; pos = 0
    for t in range(S1.shape[0]):
        c1 = int(S1[t]); s = int(l1_off[c1]); e = int(l1_off[c1+1])
        for i in range(s, e):
            did = int(l1_mem[i])
            if visited[did] == stamp:  # already added
                continue
            if s2mark[int(A2_pos[did])] == 1:
                visited[did] = stamp
                out[pos] = did; pos += 1
    return out[:pos]

# --------- Candidate generation (public API unchanged) ---------

def _adj_slice(adj_offs: np.ndarray, adj_idx: np.ndarray, c1: int) -> np.ndarray:
    s, e = int(adj_offs[c1]), int(adj_offs[c1+1])
    return adj_idx[s:e]

def two_layer_candidates_for_query(
    q_code: np.ndarray,
    index: Dict[str, object],
    p1: int,
    p2: int,
    enforce_and: bool = False,
) -> np.ndarray:
    C1 = index["C1_codes"]; C2 = index["C2_codes"]
    N  = int(index["N"]); k2 = int(index["k2"])
    l1_off = index["l1_offsets"]; l1_mem = index["l1_members"]
    l2_off = index["l2_offsets"]; l2_mem = index["l2_members"]
    A2_pos = index["A2_pos"]
    a_off  = index["adj12_offsets"]; a_idx = index["adj12_indices"]

    # L1 probe (heap)
    S1 = hamming_top_p_centers(q_code, C1, p1)

    # adjacency-restricted L2 subset (timestamps, no unique)
    S2_subset = _gather_adj_subset_ts(S1, a_off, a_idx, k2) if S1.size else np.empty(0, dtype=np.int32)

    # L2 probe (heap)
    S2 = hamming_top_p_subset(q_code, C2, S2_subset, p2) if S2_subset.size else np.empty(0, dtype=np.int32)

    # Materialize
    if not enforce_and:
        return _materialize_or_ts(S1, S2, l1_off, l1_mem, l2_off, l2_mem, N).astype(np.int32, copy=False)
    else:
        return _materialize_and_ts(S1, S2, l1_off, l1_mem, A2_pos, k2, N).astype(np.int32, copy=False)

def two_layer_candidates_batch(
    queries_codes: np.ndarray,
    index: Dict[str, object],
    p1: int,
    p2: int,
    enforce_and: bool = False,
) -> list[np.ndarray]:
    return [two_layer_candidates_for_query(q, index, p1, p2, enforce_and) for q in queries_codes]
