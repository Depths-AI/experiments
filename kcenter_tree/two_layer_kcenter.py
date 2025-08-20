# two_layer_kcenter.py
import numpy as np
from typing import Dict, List, Tuple

# Reuse your Hamming k-center primitives
# - hamming_greedy_k_center returns (centers_codes, labels_topL, centers_idx)
#   where labels_topL is (N,L) with indices in [0..K-1] (positions within centers_idx)
from utils import hamming_greedy_k_center, binary_vector_search, proportion_in_filtered  # noqa: F401 (kept for symmetry)  # :contentReference[oaicite:4]{index=4}
from binary_kcenter import hamming_top_p_centers  # :contentReference[oaicite:5]{index=5}

# -----------------------------
# CSR helpers for postings/adj
# -----------------------------

def _build_postings(labels_pos: np.ndarray, K: int) -> List[np.ndarray]:
    """
    Build postings as a Python list of np.int32 arrays.
    labels_pos: (N,) int32 in [0..K-1], single-assignment positions
    """
    N = labels_pos.shape[0]
    sizes = np.zeros(K, dtype=np.int32)
    for i in range(N):
        k = labels_pos[i]
        if 0 <= k < K:
            sizes[k] += 1

    offsets = np.empty(K + 1, dtype=np.int32)
    offsets[0] = 0
    for k in range(K):
        offsets[k + 1] = offsets[k] + sizes[k]

    members = np.empty(N, dtype=np.int32)
    cur = offsets.copy()
    for i in range(N):
        k = labels_pos[i]
        if 0 <= k < K:
            p = cur[k]
            members[p] = i
            cur[k] = p + 1

    postings: List[np.ndarray] = []
    for k in range(K):
        start, end = offsets[k], offsets[k + 1]
        postings.append(members[start:end].copy())
    return postings


def _build_adj12(postings1: List[np.ndarray], labels2_pos: np.ndarray, K1: int) -> List[np.ndarray]:
    """
    For each coarse center c1, collect distinct fine centers observed among its members.
    Returns a list length K1 with sorted unique np.int32 arrays of fine IDs.
    """
    adj12: List[np.ndarray] = [None] * K1  # type: ignore
    for c1 in range(K1):
        idxs = postings1[c1]
        if idxs.size == 0:
            adj12[c1] = np.empty(0, dtype=np.int32)
            continue
        c2_vals = labels2_pos[idxs]
        # Using np.unique for dedup/sort; could be replaced by counting if needed.
        adj12[c1] = np.unique(c2_vals.astype(np.int32))
    return adj12


# -----------------------------
# Index builder (2-layer)
# -----------------------------

def build_two_layer_index(
    codes_u64: np.ndarray,
    k1: int = 1024,
    k2: int = 8192,
    start_index_l1: int = 0,
    start_index_l2: int = 0,
) -> Dict[str, object]:
    """
    Build 2-layer k-center index over bit-packed binary codes (uint64).
    Returns a dict with centers, labels (positions), postings, and adjacency.

    codes_u64: (N, W) uint64
    """
    # Layer 1 (coarse)
    centers1_codes, labels1_top1, centers1_idx = hamming_greedy_k_center(
        codes_u64, K=k1, num_centers=1, start_index=start_index_l1
    )  # labels_top1: (N,1) positions in [0..k1-1]  # :contentReference[oaicite:6]{index=6}
    A1_pos = labels1_top1[:, 0].astype(np.int32)

    # Layer 2 (fine)
    centers2_codes, labels2_top1, centers2_idx = hamming_greedy_k_center(
        codes_u64, K=k2, num_centers=1, start_index=start_index_l2
    )  # labels_top1: (N,1) positions in [0..k2-1]  # :contentReference[oaicite:7]{index=7}
    A2_pos = labels2_top1[:, 0].astype(np.int32)

    postings1 = _build_postings(A1_pos, k1)
    postings2 = _build_postings(A2_pos, k2)
    adj12 = _build_adj12(postings1, A2_pos, k1)

    return {
        "centers1_codes": centers1_codes,  # (k1, W) uint64
        "centers2_codes": centers2_codes,  # (k2, W) uint64
        "A1_pos": A1_pos,                  # (N,) int32 positions in [0..k1-1]
        "A2_pos": A2_pos,                  # (N,) int32 positions in [0..k2-1]
        "postings1": postings1,            # List[np.ndarray]
        "postings2": postings2,            # List[np.ndarray]
        "adj12": adj12,                    # List[np.ndarray], c1 -> distinct fine centers
        "k1": k1,
        "k2": k2,
        "centers1_idx": centers1_idx,      # absolute dataset indices if needed
        "centers2_idx": centers2_idx,
    }


# -----------------------------
# Conditional 2-layer probing
# -----------------------------

def _hamming_top_p_subset(
    query_code: np.ndarray, centers_codes: np.ndarray, subset_ids: np.ndarray, p: int
) -> np.ndarray:
    """
    Compute top-p over a subset of centers by evaluating only those rows,
    then map local ranks back to global center IDs.
    """
    if subset_ids.size == 0:
        return subset_ids  # empty
    view = centers_codes[subset_ids]  # contiguous gather
    local = hamming_top_p_centers(query_code, view, min(p, view.shape[0]))  # positions 0..|subset|-1  # :contentReference[oaicite:8]{index=8}
    return subset_ids[local]


def two_layer_candidates_for_query(
    query_code: np.ndarray,
    index: Dict[str, object],
    p1: int = 256,
    p2: int = 512,
    enforce_and: bool = True,
) -> np.ndarray:
    """
    Return candidate doc ids after 2-layer conditional probing.
    - Probe L1 top-p1
    - Restrict L2 to ⋃ Adj12[c1] for selected c1
    - Probe L2 top-p2 on restricted set
    - Candidates = docs with (A1 in S1) AND (A2 in S2)  [or OR if enforce_and=False]
    """
    centers1 = index["centers1_codes"]
    centers2 = index["centers2_codes"]
    adj12 = index["adj12"]
    postings1: List[np.ndarray] = index["postings1"]  # type: ignore
    A2_pos: np.ndarray = index["A2_pos"]  # type: ignore
    k2: int = int(index["k2"])  # type: ignore

    # L1 probe: full 1,024 center set (fast)
    S1 = hamming_top_p_centers(query_code, centers1, p1)  # indices in [0..k1-1]  # :contentReference[oaicite:9]{index=9}

    # Restrict L2 by adjacency
    if S1.size == 0:
        return np.empty(0, dtype=np.int32)
    cand2_lists = [adj12[int(c1)] for c1 in S1]
    if len(cand2_lists) == 1:
        S2_raw = cand2_lists[0]
    else:
        S2_raw = np.unique(np.concatenate(cand2_lists))  # ~1–3k typical

    # L2 probe over restricted set only
    S2 = _hamming_top_p_subset(query_code, centers2, S2_raw, p2)

    # Materialize candidates
    if enforce_and:
        # AND: (A1 in S1) ∧ (A2 in S2)
        S2mask = np.zeros(k2, dtype=np.bool_)
        S2mask[S2] = True
        out = []
        for c1 in S1:
            ids = postings1[int(c1)]
            if ids.size:
                sel = ids[S2mask[A2_pos[ids]]]
                if sel.size:
                    out.append(sel)
        return np.unique(np.concatenate(out)) if out else np.empty(0, dtype=np.int32)
    else:
        # OR: (A1 in S1) ∪ (A2 in S2)
        # still collect via postings for S1, and add postings2 for S2
        postings2: List[np.ndarray] = index["postings2"]  # type: ignore
        out = []
        for c1 in S1:
            ids = postings1[int(c1)]
            if ids.size:
                out.append(ids)
        for c2 in S2:
            ids = postings2[int(c2)]
            if ids.size:
                out.append(ids)
        return np.unique(np.concatenate(out)) if out else np.empty(0, dtype=np.int32)


def two_layer_candidates_batch(
    queries_codes: np.ndarray,
    index: Dict[str, object],
    p1: int = 256,
    p2: int = 512,
    enforce_and: bool = True,
) -> List[np.ndarray]:
    """
    Batch wrapper; returns a list of candidate index arrays per query.
    """
    Q = queries_codes.shape[0]
    out: List[np.ndarray] = []
    for q in range(Q):
        out.append(
            two_layer_candidates_for_query(
                queries_codes[q], index, p1=p1, p2=p2, enforce_and=enforce_and
            )
        )
    return out
