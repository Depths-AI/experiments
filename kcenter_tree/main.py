# main.py
import time
import numpy as np
from typing import Dict, Any

from binary_utils import binary_quantize_batch
from kcenter_utils import build_two_layer_index, two_layer_candidates_batch
from auxil import load_embeddings, vector_search, proportion_in_filtered, plot_search_hists, warmup

# -------- config --------
NUM_VECS = 32760
NUM_QUERIES = 100
TOP_K = 5

# global L1/L2
K1 = 1024
K2 = 8192
P1 = 32
P2 = 64
ENFORCE_AND = False  # OR by default

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openai": {"path": "openai.parquet",  "col": "text-embedding-3-large-1536-embedding", "dim": 1536},
    "cohere": {"path": "cohere_v3.parquet","col": "emb",                                   "dim": 1024},
}

def run_provider(name: str, meta: Dict[str, Any]):
    dim = meta["dim"]; N = NUM_VECS
    print(f"\n===== Provider: {name} (dim={dim}, N={N}) =====")

    # 1) Load docs + queries
    docs = load_embeddings(meta["path"], meta["col"], dim, offset=0, take=N)
    queries = load_embeddings(meta["path"], meta["col"], dim, offset=N, take=NUM_QUERIES)

    # 2) Random orthonormal Q and binarize
    rng = np.random.default_rng(0)
    A = rng.standard_normal((dim, dim), dtype=np.float32)
    Qm, _ = np.linalg.qr(A)
    docs_b = binary_quantize_batch(docs, Qm)
    queries_b = binary_quantize_batch(queries, Qm)

    # 3) Float brute force baseline
    t0 = time.time()
    brute_topk = vector_search(queries, docs, TOP_K)
    t1 = time.time()
    print(f"Brute-force (float) time: {t1 - t0:.3f}s")

    # 4) Build index (global L1/L2 + adjacency)
    t2 = time.time()
    index = build_two_layer_index(docs_b, k1=K1, k2=K2)
    t3 = time.time()
    print(f"Build time (k1={K1}, k2={K2}): {t3 - t2:.3f}s")

    # 5) Cascade search
    t4 = time.time()
    cand_lists = two_layer_candidates_batch(queries_b, index, p1=P1, p2=P2, enforce_and=ENFORCE_AND)
    t5 = time.time()
    print(f"Search time (Q={NUM_QUERIES}, P1={P1}, P2={P2}, AND={ENFORCE_AND}): {t5 - t4:.3f}s")

    # 6) metrics
    cand_props = np.array([len(ids) * 100.0 / N for ids in cand_lists], dtype=np.float32)
    print(f"Candidate share (%): mean={cand_props.mean():.2f} | p50={np.quantile(cand_props,0.5):.2f} | "
          f"p10={np.quantile(cand_props,0.10):.2f} | p95={np.quantile(cand_props,0.95):.2f}")
    recall = proportion_in_filtered(brute_topk, cand_lists)
    print(f"Recall@{TOP_K}: mean={recall.mean():.3f} | p50={np.quantile(recall,0.5):.3f} | "
          f"p10={np.quantile(recall,0.10):.3f} | p95={np.quantile(recall,0.95):.3f}")

    # 7) plots
    plot_search_hists(cand_props, recall, provider_name=name, show=False, save_prefix=f"{name}_hist")

def main():
    print("Warming up kernels ...")
    warmup(dim=1024, n_docs=4096, n_queries=16, k1=128, k2=256, p1=8, p2=32, seed=0)
    for prov, meta in PROVIDERS.items():
        run_provider(prov, meta)

if __name__ == "__main__":
    main()
