# kcenter_utils.py
import numpy as np
from numba import njit, prange
from typing import Dict

from binary_utils import _ham_row, hamming_top_p_centers, hamming_top_p_subset

IDX_DTYPE = np.int16   # shard < 2^15-1
OFF_DTYPE = np.int32

# --------- Greedy farthest-first k-center (Hamming) ---------
# Gonzalez farthest-first is a 2-approx for k-center in metrics; great practical primitive. :contentReference[oaicite:2]{index=2}

@njit(nogil=True)
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

@njit(parallel=True, nogil=True)
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

def _adj12_csr_from_postings(A2_pos: np.ndarray, l1_offsets: np.ndarray, l1_members: np.ndarray, k1: int, k2: int) -> tuple[np.ndarray, np.ndarray]:
    # first pass sizes
    sizes = np.zeros(k1, dtype=OFF_DTYPE)
    for c1 in range(k1):
        s, e = int(l1_offsets[c1]), int(l1_offsets[c1+1])
        if e > s:
            a2 = A2_pos[l1_members[s:e]].astype(np.int32, copy=False)
            sizes[c1] = np.int32(np.unique(a2).shape[0])
    offs = np.empty(k1 + 1, dtype=OFF_DTYPE); offs[0] = 0
    for c1 in range(k1): offs[c1+1] = offs[c1] + sizes[c1]
    idx = np.empty(int(offs[-1]), dtype=IDX_DTYPE)
    # fill
    for c1 in range(k1):
        s, e = int(l1_offsets[c1]), int(l1_offsets[c1+1])
        if e <= s: continue
        a2 = A2_pos[l1_members[s:e]].astype(np.int32, copy=False)
        u = np.unique(a2)
        pos = offs[c1]
        idx[pos:pos+u.shape[0]] = u.astype(IDX_DTYPE)
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
    c2_idx = ham_greedy_kcenter_indices(codes_u64, int(min(k2, N)), start_index_l2)
    A2_pos = ham_assign_top1_to_centers(codes_u64, c2_idx)          # in [0..k2-1]
    C2_codes = codes_u64[c2_idx.astype(np.int32)].copy()
    l2_offsets, l2_members = _csr_from_labels(A2_pos.astype(np.int32), int(min(k2, N)))
    k2_eff = int(C2_codes.shape[0])

    # adjacency (CSR)
    adj12_offsets, adj12_indices = _adj12_csr_from_postings(
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

# --------- Candidate generation (OR/AND) ---------

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

    # L1 probe
    S1 = hamming_top_p_centers(q_code, C1, p1)

    # adjacency-restricted L2 subset
    if S1.size:
        counts = 0
        for c1 in S1: counts += int(a_off[int(c1)+1] - a_off[int(c1)])
        buf = np.empty(counts, dtype=np.int32); pos = 0
        for c1 in S1:
            sl = _adj_slice(a_off, a_idx, int(c1))
            L = sl.shape[0]
            if L: buf[pos:pos+L] = sl.astype(np.int32); pos += L
        S2_subset = np.unique(buf[:pos]) if pos else np.empty(0, dtype=np.int32)
    else:
        S2_subset = np.empty(0, dtype=np.int32)

    # L2 probe
    S2 = hamming_top_p_subset(q_code, C2, S2_subset, p2) if S2_subset.size else np.empty(0, dtype=np.int32)

    # Materialize
    if not enforce_and:
        mask = np.zeros(N, dtype=bool)
        # OR over L1 postings
        for c1 in S1:
            s, e = int(l1_off[int(c1)]), int(l1_off[int(c1)+1])
            if e > s: mask[l1_mem[s:e]] = True
        # OR over L2 postings
        for c2 in S2:
            s, e = int(l2_off[int(c2)]), int(l2_off[int(c2)+1])
            if e > s: mask[l2_mem[s:e]] = True
        return np.nonzero(mask)[0].astype(np.int32)
    else:
        # AND: for docs in selected L1 postings, keep only those whose A2 in S2
        if S2.size == 0: return np.empty(0, dtype=np.int32)
        s2mask = np.zeros(k2, dtype=bool); s2mask[S2] = True
        out = []
        for c1 in S1:
            s, e = int(l1_off[int(c1)]), int(l1_off[int(c1)+1])
            if e <= s: continue
            ids = l1_mem[s:e].astype(np.int32, copy=False)
            sel = ids[s2mask[A2_pos[ids]]]
            if sel.size: out.append(sel)
        return (np.unique(np.concatenate(out)) if out else np.empty(0, dtype=np.int32))

def two_layer_candidates_batch(
    queries_codes: np.ndarray,
    index: Dict[str, object],
    p1: int,
    p2: int,
    enforce_and: bool = False,
) -> list[np.ndarray]:
    return [two_layer_candidates_for_query(q, index, p1, p2, enforce_and) for q in queries_codes]
