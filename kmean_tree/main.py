import polars as pl
import numpy as np
from typing import Optional
import time
from ktree import KMeansTree3

NUM_VECS = 10000
NUM_QUERIES = 100
TOP_K = 10
OUT_CSV = f"recall_at_{TOP_K}.csv"
TIME_CSV = f"query_times_{TOP_K}.csv"

PROVIDERS: dict[str, dict[str, object]] = {
    "openai": {
        "path": "openai.parquet", # Download and rename the first file from huggingface link in the article/ README
        "col": "text-embedding-3-large-1536-embedding",
        "dim": 1536,
    }
}

def load_embeddings(path: str, col: str, dim: int):
    df = (
        pl.scan_parquet(path)
        .slice(0, NUM_VECS + NUM_QUERIES)
        .collect()
        .with_columns(pl.col(col).cast(pl.Array(pl.Float32, dim)).alias("vec"))
    )
    docs = df.head(NUM_VECS)["vec"].to_numpy()
    queries = df.tail(NUM_QUERIES)["vec"].to_numpy()
    return docs, queries

def vector_search(queries: np.ndarray, docs: np.ndarray, top_k: int=10):
    '''
    Optimal NumPy routine to perform search for a batch of queries

    Note that, for compute, we are still relying on projecting our stored vector
    to float32. Ofcourse, translating float16 or int8 back to float32 does not
    carry the same precision as the original vector.
    '''

    sims = docs @ queries.T
    k = min(top_k, sims.shape[0])
    top = np.argpartition(-sims, k - 1, axis=0)[:k]
    top_sims = np.take_along_axis(sims, top, axis=0)
    order = np.argsort(-top_sims, axis=0)
    idxs = np.take_along_axis(top, order, axis=0).T
    return idxs, None  # Not returning similarity scores, change this if you wanna see those as well

def recall_at_k(ref: np.ndarray, test: np.ndarray) -> float:
    '''
    Simple NumPy routine to compute recall@k for a given result compared to float32 brute force search as the reference.
    '''
    return float((ref[:, :, None] == test[:, None, :]).any(axis=2).mean())

def benchmark():
    for prov, meta in PROVIDERS.items():
        docs, queries= load_embeddings(meta["path"], meta["col"], meta["dim"])

        brute_ids, _ = vector_search(queries, docs, TOP_K)

        index = KMeansTree3(
            n_l1=1_000, n_l2=100, n_l3=10,
            k1=30, k2=20, k3=10,
            beam=60,          # traversal budget
            m_assign=2,       # postings per doc
            metric="cosine",
            random_state=0
        ).fit(docs)


        ids_list, dists_list, meta = index.batch_search(queries, s=TOP_K)
        ktree_ids=np.array(ids_list)
        print(meta["avg_candidates"])
        r=recall_at_k(brute_ids, ktree_ids)
        print(r)

if __name__ == "__main__":
    benchmark()
