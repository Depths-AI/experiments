# main.py
import time
from typing import Dict, Any, Optional

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# new lean modules
from search_utils import binary_quantize_batch, cascade_candidates
from kcenter_utils import build_index_localL2

# -------------------
# Config
# -------------------
NUM_VECS = 32760         # shard size < 2^15-1  → int16-safe
CHUNK_SIZE = 32760
NUM_QUERIES = 1000
TOP_K = 1

# index params (per-L1 local L2)
K1 = 512                 # coarse centers
K2_PER_L1 = 8           # fine centers per coarse list
P1 = 64*int(1+np.log10(TOP_K))                 # L1 probes
P2 = 256*int(1+np.log10(TOP_K))                  # L2 probes (within selected L1 lists)
USE_AND = True          # OR mode default

# providers to benchmark (same two as before)
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": {
        "path": "openai.parquet",
        "col": "text-embedding-3-large-1536-embedding",
        "dim": 1536,
    },
    "cohere": {
        "path": "cohere_v3.parquet",
        "col": "emb",
        "dim": 1024,
    },
}

# -------------------
# Helpers (keep local to avoid extra files)
# -------------------
def load_embeddings(path: str, col: str, dim: int, offset: int, take: int) -> np.ndarray:
    df = (
        pl.scan_parquet(path)
        .slice(offset, take)
        .collect()
        .with_columns(pl.col(col).cast(pl.Array(pl.Float32, dim)).alias("vec"))
    )
    arr = df["vec"].to_numpy()  # (take, dim) float32
    # preserve legacy normalization behavior (global)
    arr = arr / (np.linalg.norm(arr))
    return arr

def vector_search_bruteforce(queries: np.ndarray, docs: np.ndarray, top_k: int) -> np.ndarray:
    # cosine (row-wise normalization) → top-k doc indices per query
    qn = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)
    dn = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-12)
    sims = qn @ dn.T
    # partial top-k, then sort within top-k
    part = np.argpartition(-sims, kth=top_k - 1, axis=1)[:, :top_k]
    part_scores = np.take_along_axis(sims, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    topk_idx = np.take_along_axis(part, order, axis=1).astype(np.int32)
    return topk_idx  # shape: (Q, top_k)

def proportion_in_filtered(brute_topk: np.ndarray, filtered_lists: list[np.ndarray]) -> np.ndarray:
    # recall@K = |{topK} ∩ candidates| / K  (per query)
    Q, K = brute_topk.shape
    out = np.empty(Q, dtype=np.float32)
    for i in range(Q):
        cand = filtered_lists[i]
        if cand.size == 0:
            out[i] = 0.0
            continue
        # np.isin vectorizes well on int32 arrays
        out[i] = np.isin(brute_topk[i], cand, assume_unique=False).sum(dtype=np.int32) / float(K)
    return out


def plot_search_hists(
    cand_percent: np.ndarray,
    recall: np.ndarray,
    provider_name: str = "",
    show: bool = True,
    save_prefix: Optional[str] = None,
) -> None:
    """
    Plot histograms for:
      (1) cand_percent: percentage of DB scanned per query (shape: [NUM_QUERIES], in %)
      (2) recall: recall per query in [0, 1] (shape: [NUM_QUERIES])

    If save_prefix is provided, saves PNGs as:
      f"{save_prefix}_cand_share_hist.png" and f"{save_prefix}_recall_hist.png"
    """
    
    assert cand_percent.ndim == 1 and recall.ndim == 1, "Inputs must be 1-D arrays"

    # --- Histogram 1: candidate share (% of DB) ---
    fig1, ax1 = plt.subplots()
    bins_pct = np.linspace(0.0, 100.0, 21)  # 5% bins
    ax1.hist(np.clip(cand_percent.astype(float), 0.0, 100.0),
             bins=bins_pct, edgecolor="black")
    ax1.set_title(f"Candidate Share (% of DB){' - ' + provider_name if provider_name else ''}")
    ax1.set_xlabel("% of DB scanned")
    ax1.set_ylabel("Queries")
    ax1.set_xticks(np.arange(0, 101, 10))  # ticks at 0, 10, ..., 100
    ax1.set_xlim(0.0, 100.0)
    ax1.xaxis.set_major_formatter(PercentFormatter(xmax=100.0))
    fig1.tight_layout()
    if save_prefix:
        fig1.savefig(f"{save_prefix}_cand_share_hist.png", dpi=140)

    # --- Histogram 2: recall (0..1) ---
    fig2, ax2 = plt.subplots()
    bins_rec = np.linspace(0.0, 1.0, 21)  # 0.05-wide bins
    ax2.hist(np.clip(recall.astype(float), 0.0, 1.0),
             bins=bins_rec, edgecolor="black")
    ax2.set_title(f"Recall per Query{ ' - ' + provider_name if provider_name else ''}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Queries")
    ax2.set_xticks(np.arange(0.0, 1.1, 0.1))  # ticks at 0.0, 0.1, ..., 1.0
    ax2.set_xlim(0.0, 1.0)
    fig2.tight_layout()
    if save_prefix:
        fig2.savefig(f"{save_prefix}_recall_hist.png", dpi=140)

    if show:
        plt.show()
    else:
        plt.close(fig1); plt.close(fig2)
# -------------------
# Inline warmup: JIT-compile hot kernels before real runs
# -------------------
def warmup(dim=1024, n_docs=4096, n_queries=16, k1=128, k2_per_l1=8, p1=8, p2=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_docs + n_queries, dim), dtype=np.float32)
    A = rng.standard_normal((dim, dim), dtype=np.float32)
    Qm, _ = np.linalg.qr(A)  # random orthonormal
    codes = binary_quantize_batch(X, Qm)
    docs_b, queries_b = codes[:n_docs], codes[n_docs:]
    index = build_index_localL2(docs_b, k1=k1, k2_per_l1=k2_per_l1)
    # fire a couple of search calls to compile numba paths
    for i in range(min(n_queries, 4)):
        _ = cascade_candidates(queries_b[i], index, p1=p1, p2=p2, use_and=False)
    return index  # not used further; JIT side-effects are the goal

# -------------------
# Single provider run
# -------------------
def run_provider(name: str, meta: Dict[str, Any]):
    dim = meta["dim"]
    N = NUM_VECS
    print(f"\n===== Provider: {name} (dim={dim}, N={N}) =====")

    # 1) Load docs + queries (contiguous chunking as before)
    docs = load_embeddings(meta["path"], meta["col"], dim, offset=0, take=N)
    queries = load_embeddings(meta["path"], meta["col"], dim, offset=N, take=NUM_QUERIES)

    # 2) Random orthonormal Q and binarize
    rng = np.random.default_rng(0)
    A = rng.standard_normal((dim, dim), dtype=np.float32)
    Qm, _ = np.linalg.qr(A)
    docs_b = binary_quantize_batch(docs, Qm)
    queries_b = binary_quantize_batch(queries, Qm)

    # 3) Float brute-force baseline (top-K)
    t0 = time.time()
    brute_topk = vector_search_bruteforce(queries, docs, TOP_K)
    t1 = time.time()
    print(f"Brute-force (float) time: {t1 - t0:.3f}s")

    # 4) Build per-L1 local L2 index
    t2 = time.time()
    index = build_index_localL2(docs_b, k1=K1, k2_per_l1=K2_PER_L1)
    t3 = time.time()
    print(f"Build time (k1={K1}, k2_per_l1={K2_PER_L1}): {t3 - t2:.3f}s")

    # 5) Candidate generation via cascade (OR or AND)
    t4 = time.time()
    filtered_lists = [cascade_candidates(queries_b[i], index, p1=P1, p2=P2, use_and=USE_AND)
                      for i in range(NUM_QUERIES)]
    t5 = time.time()
    print(f"Search time (Q={NUM_QUERIES}, P1={P1}, P2={P2}, AND={USE_AND}): {t5 - t4:.3f}s")

    # 6) Metrics
    cand_props = np.array([len(ids) * 100.0 / N for ids in filtered_lists], dtype=np.float32)
    print(
        "Candidate share (% of DB): "
        f"mean={cand_props.mean():.2f} | p50={np.quantile(cand_props,0.5):.2f} | "
        f"p10={np.quantile(cand_props,0.10):.2f} | p95={np.quantile(cand_props,0.95):.2f}"
    )
    recall = proportion_in_filtered(brute_topk, filtered_lists)
    print(
        f"Recall@{TOP_K}: mean={recall.mean():.3f} | "
        f"p50={np.quantile(recall,0.5):.3f} | p10={np.quantile(recall,0.10):.3f} | "
        f"p95={np.quantile(recall,0.95):.3f}"
    )

    plot_search_hists(cand_props, recall, provider_name=name, show=False, save_prefix=f"{name}_recall_at_{TOP_K}_hist")
# -------------------
# Entry
# -------------------
def main():
    print("Warmup: JIT-compiling kernels ...")
    warmup(dim=1024, n_docs=4096, n_queries=16, k1=128, k2_per_l1=8, p1=8, p2=32, seed=0)
    for prov, meta in PROVIDERS.items():
        run_provider(prov, meta)

if __name__ == "__main__":
    main()
