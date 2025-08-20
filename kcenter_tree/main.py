# main_two_layer.py
import time
import numpy as np
import polars as pl

from utils import (
    binary_quantize_batch,
    vector_search,
    proportion_in_filtered
)
from warmup import warm_run_all
from two_layer_kcenter import build_two_layer_index, two_layer_candidates_batch  # new
from typing import Dict, Any

# ---- Match your existing constants & provider map ----
NUM_VECS = 32760
CHUNK_SIZE = 32760
NUM_QUERIES = 1000          # feel free to change to 1 to mirror your current run
TOP_K = 10
K1 = 256                   # coarse
K2 = 2048                   # fine
P1 = 12                    # top-L1
P2 = 96                    # top-L2

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

def load_embeddings(path: str, col: str, dim: int, offset: int, chunk_size: int = CHUNK_SIZE):
    df = (
        pl.scan_parquet(path)
        .slice(offset, CHUNK_SIZE + offset)
        .collect()
        .with_columns(pl.col(col).cast(pl.Array(pl.Float32, dim)).alias("vec"))
    )
    docs = df.head(chunk_size)["vec"].to_numpy()
    # retain your original normalization convention from main.py
    docs = docs / (np.linalg.norm(docs))   # NOTE: global norm as in your script  # :contentReference[oaicite:10]{index=10}
    return docs

def run_provider(prov: str, meta: Dict[str, Any]):
    print(f"\n===== Provider: {prov}  (dim={meta['dim']}, N={NUM_VECS}) =====")
    # 1) Load docs & queries (same pattern as your main.py)
    docs = load_embeddings(meta["path"], meta["col"], meta["dim"], offset=0, chunk_size=NUM_VECS)
    queries = load_embeddings(meta["path"], meta["col"], meta["dim"], offset=NUM_VECS, chunk_size=NUM_QUERIES)

    # 2) Random orthonormal Q and binarize (reuse your util)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((meta["dim"], meta["dim"]))
    Q, _ = np.linalg.qr(A, mode="reduced")

    docs_b = binary_quantize_batch(docs, Q)      # (N, W) uint64  # :contentReference[oaicite:11]{index=11}
    queries_b = binary_quantize_batch(queries, Q)

    # 3) Brute-force float baseline (reuse your vector_search)
    brute_idx = vector_search(queries, docs, TOP_K)  # (Q, TOP_K)  # :contentReference[oaicite:12]{index=12}

    # 4) Build 2-layer k-center index
    t0 = time.time()
    index = build_two_layer_index(docs_b, k1=K1, k2=K2)
    t1 = time.time()
    print(f"Build time (k1={K1}, k2={K2}): {t1 - t0:.3f}s")

    # 5) Search via conditional probing
    t2 = time.time()
    filtered_lists = two_layer_candidates_batch(queries_b, index, p1=P1, p2=P2, enforce_and=False)
    t3 = time.time()
    print(f"Search time (Q={NUM_QUERIES}, P1={P1}, P2={P2}): {t3 - t2:.3f}s")

    # 6) Metrics: candidate % and recall@K wrt brute force
    cand_props = np.array([len(ids) * 100.0 / NUM_VECS for ids in filtered_lists])
    print(f"Candidate share: {cand_props.mean():.2f}% , p50={np.quantile(cand_props, 0.5):.2f}%, p10={np.quantile(cand_props, 0.10):.2f}%, p95={np.quantile(cand_props, 0.95):.2f}%")

    recall = proportion_in_filtered(brute_idx, filtered_lists)  # (Q,) ∈ [0,1]  # :contentReference[oaicite:14]{index=14}
    print(f"Recall@{TOP_K}: mean={recall.mean():.3f}, p50={np.quantile(recall,0.5):.3f}, p10={np.quantile(recall,0.10):.3f}, p95={np.quantile(recall,0.95):.3f}")
    
    # print(np.sort(recall)[:NUM_QUERIES//2])  # sorted recall for debugging
def main():
    # keep parity with your script’s warm-run structure
    print("Warming up JIT-compiled paths...")
    warm_run_all(dim=1536, n_docs=4096, n_queries=16, k1=64, k2=256, p1=16, p2=64)  # :contentReference[oaicite:15]{index=15}
    for prov, meta in PROVIDERS.items():
        run_provider(prov, meta)

if __name__ == "__main__":
    main()
