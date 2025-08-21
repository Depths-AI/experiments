# auxil.py
import numpy as np
import polars as pl

from typing import Optional, Dict, Any

from binary_utils import binary_quantize_batch, pack_signs_to_uint64
from kcenter_utils import build_two_layer_index, two_layer_candidates_batch

# ---------- data loading ----------

def load_embeddings(path: str, col: str, dim: int, offset: int, take: int) -> np.ndarray:
    df = (
        pl.scan_parquet(path)
        .slice(offset, take)
        .collect()
        .with_columns(pl.col(col).cast(pl.Array(pl.Float32, dim)).alias("vec"))
    )
    arr = df["vec"].to_numpy()  # (take, dim) float32
    # keep original convention (global norm)
    arr = arr / (np.linalg.norm(arr))
    return arr

# ---------- brute force float top-K (cosine) ----------

def vector_search(queries: np.ndarray, docs: np.ndarray, top_k: int) -> np.ndarray:
    qn = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)
    dn = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-12)
    sims = qn @ dn.T
    k = min(top_k, sims.shape[1])
    part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(sims, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    topk_idx = np.take_along_axis(part, order, axis=1).astype(np.int32)
    return topk_idx

# ---------- recall wrt brute set ----------

def proportion_in_filtered(brute_topk: np.ndarray, filtered_lists: list[np.ndarray]) -> np.ndarray:
    Q, K = brute_topk.shape
    out = np.empty(Q, dtype=np.float32)
    for i in range(Q):
        cand = filtered_lists[i]
        if cand.size == 0: out[i] = 0.0; continue
        out[i] = np.isin(brute_topk[i], cand, assume_unique=False).sum(dtype=np.int32) / float(K)
    return out

# ---------- plotting ----------

def plot_search_hists(
    cand_percent: np.ndarray,
    recall: np.ndarray,
    provider_name: str = "",
    show: bool = True,
    save_prefix: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    # Matplotlib's hist matches numpy.histogram; PercentFormatter formats axes in %. :contentReference[oaicite:3]{index=3}
    assert cand_percent.ndim == 1 and recall.ndim == 1
    # candidate %
    fig1, ax1 = plt.subplots()
    bins_pct = np.linspace(0.0, 100.0, 21)
    ax1.hist(np.clip(cand_percent.astype(float), 0.0, 100.0), bins=bins_pct, edgecolor="black")
    ax1.set_title(f"Candidate Share (% DB){' - ' + provider_name if provider_name else ''}")
    ax1.set_xlabel("% of DB"); ax1.set_ylabel("Queries"); ax1.set_xlim(0.0, 100.0)
    ax1.xaxis.set_major_formatter(PercentFormatter(xmax=100.0))
    fig1.tight_layout()
    if save_prefix: fig1.savefig(f"{save_prefix}_cand_share_hist.png", dpi=140)
    # recall [0,1]
    fig2, ax2 = plt.subplots()
    bins_rec = np.linspace(0.0, 1.0, 21)
    ax2.hist(np.clip(recall.astype(float), 0.0, 1.0), bins=bins_rec, edgecolor="black")
    ax2.set_title(f"Recall per Query{' - ' + provider_name if provider_name else ''}")
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Queries"); ax2.set_xlim(0.0, 1.0)
    ax2.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    fig2.tight_layout()
    if save_prefix: fig2.savefig(f"{save_prefix}_recall_hist.png", dpi=140)
    if show: plt.show()
    else: plt.close(fig1); plt.close(fig2)

# ---------- inline warmup (JIT) ----------

def warmup(dim=1024, n_docs=4096, n_queries=16, k1=256, k2=1024, p1=16, p2=64, seed=0):
    """
    JIT-compile the hot Numba kernels (bit-pack, popcnt, k-center assign, top-p center scans)
    so main timing reflects a warmed state. Use small synthetic batches for speed.
    Numba parallelization with prange is triggered only where profitable. :contentReference[oaicite:4]{index=4}
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_docs + n_queries, dim), dtype=np.float32)
    A = rng.standard_normal((dim, dim), dtype=np.float32)
    Qm, _ = np.linalg.qr(A)
    codes = binary_quantize_batch(X, Qm)
    docs, queries = codes[:n_docs], codes[n_docs:]
    idx = build_two_layer_index(docs, k1=k1, k2=k2)
    _ = two_layer_candidates_batch(queries[:min(n_queries, 4)], idx, p1=p1, p2=p2, enforce_and=False)
    return idx  # not used; side-effect is JIT
