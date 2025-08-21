# kcenter_utils.py
import numpy as np
from numba import njit, prange
from typing import Dict

from binary_utils import _ham_row, hamming_top_p_centers, hamming_top_p_subset

IDX_DTYPE = np.int16   # shard < 2^15-1
OFF_DTYPE = np.int32

# ---- Tunables for diversification (kept internal; API unchanged) ----
_DIVERSIFY_ALPHA = 0.8          # fraction of p2 for core picks
_C2_NEIGHBORS_T = 4             # neighbors per L2 center
_MAX_EXPAND_PER_CORE = 2        # cap explores per seed core center

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
    """Adjacency L1→(unique) L2 via timestamped visited array (no np.unique)."""
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

    # pass 2: fill indices
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

# --------- C2→C2 neighbor list for diversified probing ---------

@njit(nogil=True, cache=True)
def _build_c2_neighbors(C2_codes: np.ndarray, T: int) -> np.ndarray:
    """
    For each L2 center i, keep its T nearest L2 centers by Hamming
    (excluding self). Uses existing heap-based hamming_top_p_centers.
    """
    k2 = C2_codes.shape[0]
    out = np.empty((k2, T), dtype=IDX_DTYPE)
    for i in range(k2):
        # nearest T+1 will always include 'i' at distance 0
        top = hamming_top_p_centers(C2_codes[i], C2_codes, T + 1)
        pos = 0
        for j in range(top.shape[0]):
            cid = int(top[j])
            if cid == i: 
                continue
            out[i, pos] = np.int16(cid)
            pos += 1
            if pos == T:
                break
    return out

# --------- Build: global L1, global L2, adjacency (+ C2_nn) ---------

def build_two_layer_index(
    codes_u64: np.ndarray,
    k1: int,
    k2: int,
    start_index_l1: int = 0,
    start_index_l2: int = 0,
) -> Dict[str, object]:
    """
    Global L1 + Global L2 (both k-center on bit codes).
    Adjacency maps each coarse center to the set of fine centers seen under its members.
    Also builds a small C2→C2 neighbor list for multi-probe diversification.
    """
    N = codes_u64.shape[0]

    # L1
    c1_idx = ham_greedy_kcenter_indices(codes_u64, int(k1), start_index_l1)
    A1_pos = ham_assign_top1_to_centers(codes_u64, c1_idx)
    C1_codes = codes_u64[c1_idx.astype(np.int32)].copy()
    l1_offsets, l1_members = _csr_from_labels(A1_pos.astype(np.int32), k1)

    # L2
    k2_eff = int(min(k2, N))
    c2_idx = ham_greedy_kcenter_indices(codes_u64, k2_eff, start_index_l2)
    A2_pos = ham_assign_top1_to_centers(codes_u64, c2_idx)
    C2_codes = codes_u64[c2_idx.astype(np.int32)].copy()
    l2_offsets, l2_members = _csr_from_labels(A2_pos.astype(np.int32), k2_eff)

    # adjacency (CSR)
    adj12_offsets, adj12_indices = _adj12_csr_from_postings_ts(
        A2_pos.astype(np.int32), l1_offsets, l1_members, int(k1), k2_eff
    )

    # L2→L2 neighbors (tiny; enables diversified probing at same p2)
    C2_nn = _build_c2_neighbors(C2_codes, _C2_NEIGHBORS_T)

    return {
        "C1_codes": C1_codes, "C2_codes": C2_codes,
        "A1_pos": A1_pos.astype(IDX_DTYPE), "A2_pos": A2_pos.astype(IDX_DTYPE),
        "l1_offsets": l1_offsets, "l1_members": l1_members,
        "l2_offsets": l2_offsets, "l2_members": l2_members,
        "adj12_offsets": adj12_offsets, "adj12_indices": adj12_indices,
        "C2_nn": C2_nn,                        # << new
        "k1": int(k1), "k2": k2_eff, "N": np.int32(N),
    }

# --------- Query-time helpers (timestamps; no uniques/mask clears) ---------

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

# --------- Diversified L2 selection at fixed p2 (Core + Explore) ---------

@njit(nogil=True, cache=True, boundscheck=True)
def _diversified_L2(q_code: np.ndarray, C2: np.ndarray, S2_subset: np.ndarray, p2: int,
                    C2_nn: np.ndarray, alpha: float, max_expand_per_core: int) -> np.ndarray:
    if S2_subset.shape[0] == 0 or p2 <= 0:
        return np.empty(0, dtype=np.int32)

    # 1) Core picks
    p_core = int(alpha * p2)
    if p_core < 1: p_core = 1
    if p_core > p2: p_core = p2
    core = hamming_top_p_subset(q_code, C2, S2_subset, p_core)

    # visited & subset marks
    k2 = C2.shape[0]
    visited = np.zeros(k2, dtype=np.int32)
    for i in range(core.shape[0]): visited[int(core[i])] = 1
    in_subset = np.zeros(k2, dtype=np.int32)
    for i in range(S2_subset.shape[0]): in_subset[int(S2_subset[i])] = 1

    # 2) Explore from a few core seeds
    budget = p2 - p_core
    explore = np.empty(p2, dtype=np.int32)
    pos = 0
    if budget > 0:
        seeds_to_use = core.shape[0]
        if max_expand_per_core > 0:
            # ceil(budget / max_expand_per_core)
            need = (budget + max_expand_per_core - 1) // max_expand_per_core
            if need < seeds_to_use: seeds_to_use = need
        for s in range(seeds_to_use):
            seed = int(core[s]); taken = 0
            neigh = C2_nn[seed]
            for u in range(neigh.shape[0]):
                if budget == 0: break
                nb = int(neigh[u])
                if in_subset[nb] == 0: continue
                if visited[nb] == 1: continue
                visited[nb] = 1
                explore[pos] = nb; pos += 1
                budget -= 1; taken += 1
                if max_expand_per_core > 0 and taken >= max_expand_per_core:
                    break
            if budget == 0: break

    # 3) Fill any remainder with next best from subset (ignore already picked)
    if budget > 0:
        extra = hamming_top_p_subset(q_code, C2, S2_subset, min(S2_subset.shape[0], p2 * 2))
        for i in range(extra.shape[0]):
            if budget == 0: break
            nb = int(extra[i])
            if visited[nb] == 1: continue
            visited[nb] = 1
            explore[pos] = nb; pos += 1
            budget -= 1

    # concatenate core + explore[:pos] (size <= p2)
    base = core.shape[0]
    total = base + pos
    if total > p2:
        total = p2
    out = np.empty(total, dtype=np.int32)

    # copy core
    for t in range(base):
        out[t] = int(core[t])

    # copy as many explore as fit
    fill = total - base
    for j in range(fill):
        out[base + j] = int(explore[j])

    return out

# --------- Candidate generation (public API unchanged) ---------

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
    C2_nn  = index.get("C2_nn", None)

    # L1 probe (heap)
    S1 = hamming_top_p_centers(q_code, C1, p1)

    # adjacency-restricted L2 subset
    S2_subset = _gather_adj_subset_ts(S1, a_off, a_idx, k2) if S1.size else np.empty(0, dtype=np.int32)

    # L2 probe (Core+Explore under fixed p2)
    if C2_nn is not None and S2_subset.size:
        S2 = _diversified_L2(q_code, C2, S2_subset, p2, C2_nn,
                             float(_DIVERSIFY_ALPHA), int(_MAX_EXPAND_PER_CORE))
    else:
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
