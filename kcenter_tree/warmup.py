# warmup.py
import numpy as np
from typing import Dict, List

# ---- primitives from your repo ----
from search import binary_search_kernel, pack_signs_to_uint64  # JIT targets  :contentReference[oaicite:4]{index=4}
from utils import binary_quantize_batch, hamming_greedy_k_center  # wrappers   :contentReference[oaicite:5]{index=5}
from binary_kcenter import hamming_top_p_centers                 # L1/L2 probe :contentReference[oaicite:6]{index=6}

# Try to use the 2-layer module we added earlier; if missing, we fall back to a tiny local impl.
try:
    from two_layer_kcenter import build_two_layer_index, two_layer_candidates_batch
    HAVE_TLK = True
except Exception:
    HAVE_TLK = False

    def _build_postings(labels_pos: np.ndarray, K: int) -> List[np.ndarray]:
        sizes = np.zeros(K, dtype=np.int32)
        for i in range(labels_pos.shape[0]):
            k = labels_pos[i]
            if 0 <= k < K:
                sizes[k] += 1
        offsets = np.empty(K + 1, dtype=np.int32)
        offsets[0] = 0
        for k in range(K):
            offsets[k + 1] = offsets[k] + sizes[k]
        members = np.empty(labels_pos.shape[0], dtype=np.int32)
        cur = offsets.copy()
        for i in range(labels_pos.shape[0]):
            k = labels_pos[i]
            if 0 <= k < K:
                p = cur[k]
                members[p] = i
                cur[k] = p + 1
        posts = []
        for k in range(K):
            s, e = offsets[k], offsets[k + 1]
            posts.append(members[s:e].copy())
        return posts

    def _build_adj12(postings1: List[np.ndarray], labels2_pos: np.ndarray, K1: int) -> List[np.ndarray]:
        adj: List[np.ndarray] = [None] * K1  # type: ignore
        for c1 in range(K1):
            ids = postings1[c1]
            if ids.size == 0:
                adj[c1] = np.empty(0, dtype=np.int32)
            else:
                adj[c1] = np.unique(labels2_pos[ids].astype(np.int32))
        return adj

    def build_two_layer_index(
        codes_u64: np.ndarray, k1: int = 1024, k2: int = 8192, start_index_l1: int = 0, start_index_l2: int = 0
    ) -> Dict[str, object]:
        # Reuse your hamming k-center (single-assignment)
        centers1_codes, labels1, _ = hamming_greedy_k_center(codes_u64, K=k1, num_centers=1, start_index=start_index_l1)  # :contentReference[oaicite:7]{index=7}
        A1 = labels1[:, 0].astype(np.int32)
        centers2_codes, labels2, _ = hamming_greedy_k_center(codes_u64, K=k2, num_centers=1, start_index=start_index_l2)  # :contentReference[oaicite:8]{index=8}
        A2 = labels2[:, 0].astype(np.int32)
        postings1 = _build_postings(A1, k1)
        postings2 = _build_postings(A2, k2)
        adj12 = _build_adj12(postings1, A2, k1)
        return {
            "centers1_codes": centers1_codes,
            "centers2_codes": centers2_codes,
            "A1_pos": A1,
            "A2_pos": A2,
            "postings1": postings1,
            "postings2": postings2,
            "adj12": adj12,
            "k1": int(k1),
            "k2": int(k2),
        }

    def _hamming_top_p_subset(q: np.ndarray, C: np.ndarray, subset: np.ndarray, p: int) -> np.ndarray:
        if subset.size == 0:
            return subset
        local = hamming_top_p_centers(q, C[subset], min(p, subset.shape[0]))  # :contentReference[oaicite:9]{index=9}
        return subset[local]

    def two_layer_candidates_batch(
        queries_codes: np.ndarray, index: Dict[str, object], p1: int = 256, p2: int = 512, enforce_and: bool = True
    ) -> List[np.ndarray]:
        C1 = index["centers1_codes"]; C2 = index["centers2_codes"]
        A2 = index["A2_pos"]; posts1 = index["postings1"]; adj12 = index["adj12"]; k2 = int(index["k2"])
        out: List[np.ndarray] = []
        for q in queries_codes:
            S1 = hamming_top_p_centers(q, C1, p1)  # coarse probe                           :contentReference[oaicite:10]{index=10}
            cand2 = np.unique(np.concatenate([adj12[int(c1)] for c1 in S1])) if S1.size else np.empty(0, np.int32)
            S2 = _hamming_top_p_subset(q, C2, cand2, p2)
            S2mask = np.zeros(k2, dtype=np.bool_)
            S2mask[S2] = True
            cand = []
            for c1 in S1:
                ids = posts1[int(c1)]
                if ids.size:
                    sel = ids[S2mask[A2[ids]]] if enforce_and else ids
                    if sel.size:
                        cand.append(sel)
            out.append(np.unique(np.concatenate(cand)) if cand else np.empty(0, dtype=np.int32))
        return out


def warm_run_all(
    dim: int = 1024,
    n_docs: int = 2048,
    n_queries: int = 8,
    k1: int = 1024,
    k2: int = 8192,
    p1: int = 256,
    p2: int = 512,
    top_k: int = 10,
    seed: int = 0,
):
    """
    Unified warm-up:
      (1) JIT-compile Hamming batch search, binarization & bitpacking paths.
      (2) JIT-compile 2-layer k-center build + conditional probing.

    Uses tiny synthetic batches to keep compile time low.
    """
    rng = np.random.default_rng(seed)

    # ---- synthetic float docs/queries ----
    docs = rng.standard_normal((n_docs, dim), dtype=np.float32)
    queries = rng.standard_normal((n_queries, dim), dtype=np.float32)

    # Random orthonormal projection (same path used in your code)
    A = rng.standard_normal((dim, dim), dtype=np.float32)
    Qm, _ = np.linalg.qr(A)

    # ---- (1) warm binarization & bitpacking ----
    docs_b = binary_quantize_batch(docs, Qm)      # calls pack_signs_to_uint64 under the hood  :contentReference[oaicite:11]{index=11}
    queries_b = binary_quantize_batch(queries, Qm)

    # Force pack_signs JIT independently as well (optional)
    _ = pack_signs_to_uint64((docs @ Qm).astype(np.float32))                                  # :contentReference[oaicite:12]{index=12}

    # ---- (1) warm Hamming batch search kernel ----
    k = min(top_k, max(1, n_docs // 64))
    _ = binary_search_kernel(docs_b, queries_b, k)                                            # :contentReference[oaicite:13]{index=13}

    # ---- (2) warm 2-layer k-center index build + conditional probe ----
    k1_eff = min(k1, n_docs); k2_eff = min(k2, n_docs)
    idx = build_two_layer_index(docs_b, k1=k1_eff, k2=k2_eff)  # builds both layers           

    # small probe set to trigger JIT on hamming_top_p_centers & cascade
    p1_eff = min(p1, k1_eff); p2_eff = min(p2, k2_eff)
    _ = two_layer_candidates_batch(queries_b, idx, p1=p1_eff, p2=p2_eff, enforce_and=True)

    # Return tiny summary for logging
    return {
        "dims": dim,
        "n_docs": n_docs,
        "n_queries": n_queries,
        "k1": k1_eff,
        "k2": k2_eff,
        "p1": p1_eff,
        "p2": p2_eff,
        "module_two_layer": "two_layer_kcenter" if HAVE_TLK else "inline_fallback",
    }


if __name__ == "__main__":
    info = warm_run_all()
    print("[warmup] done:", info)
