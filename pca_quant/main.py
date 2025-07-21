import polars as pl
import numpy as np
from typing import Optional

NUM_VECS = 10_000
NUM_QUERIES = 1000
TOP_K = 100
PCA_FACTORS = [4]
DTYPES = ["fp32", "fp16"]
OUT_CSV = f"benchmark_results_k_{TOP_K}.csv"

PROVIDERS: dict[str, dict[str, object]] = {
    "openai": {
        "path": "openai.parquet", # Download and rename the first file from huggingface link in the article/ README
        "col": "text-embedding-3-large-1536-embedding",
        "dim": 1536,
    },
    "cohere": {
        "path": "cohere.parquet", # Download and rename the first file from huggingface link in the article/ README
        "col": "emb",
        "dim": 768,
    },
}

def vector_search(queries: np.ndarray, docs: np.ndarray, top_k: int):
    '''
    Optimal NumPy routine to perform search for a batch of queries

    Note that, for compute, we are still relying on projecting our stored vector
    to float32. Ofcourse, translating float16 or int8 back to float32 does not
    carry the same precision as the original vector.
    '''
    if docs.dtype==np.float16:
        docs = docs.astype(np.float32, copy=True)
    
    if docs.dtype==np.int8:
        docs = docs.astype(np.float32, copy=True)
        docs=docs/(np.linalg.norm(docs, axis=1, keepdims=True))

    if queries.dtype==np.float16:
        queries = queries.astype(np.float32, copy=True)
        queries=queries/(np.linalg.norm(queries, axis=1, keepdims=True))

    
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

def pca_reduce(docs: np.ndarray, queries: np.ndarray, factor: int):
    '''
    Simple NumPy routine to perform PCA on a batch of docs, and then the same transformation on the batch of queries
    '''
    if factor <= 0:
        raise ValueError("factor must be a positive integer")

    original_dim = docs.shape[1]
    new_dim = original_dim // factor
    if new_dim < 1:
        raise ValueError("factor is too large; resulting dimension is < 1")

    mean = docs.mean(axis=0, keepdims=True)
    docs_c = docs - mean
    queries_c = queries - mean

    docs_c = docs_c.astype(np.float32, copy=True)
    queries_c = queries_c.astype(np.float32, copy=True)

    _, _, Vt = np.linalg.svd(docs_c, full_matrices=False)

    components = Vt[:new_dim]
    docs_red = docs_c @ components.T
    queries_red = queries_c @ components.T

    return docs_red, queries_red

def int8_quantize(
    vectors: np.ndarray, integer_bits: int = 8, seed: int = 0, Q: Optional[np.ndarray] = None
) -> np.ndarray:
    '''
    A not-so simple NumPy routine to quantize a batch of vectors to int8.

    The idea is derived from LSH hashing concept, where we project the vectors to a random space and then quantize them to int8.
    '''
    rng = np.random.default_rng(seed)

    _, dim = vectors.shape
    if Q is None:
        random_matrix = rng.standard_normal((dim, dim))
        Q, _ = np.linalg.qr(random_matrix, mode="reduced")

    projections = vectors @ Q

    width = 2.0 / (2**integer_bits)

    quantized = np.clip(
        np.round(projections / width),
        -(2 ** (integer_bits - 1)),
        2 ** (integer_bits - 1) - 1,
    ).astype(np.int8)
    return quantized

def apply_variant(docs: np.ndarray, queries: np.ndarray, pca: int|None, dtype: str, q8: bool):
    if dtype == "fp16":
        docs, queries = docs.astype(np.float16, copy=True), queries.astype(np.float16, copy=True)
    if q8:
        rng = np.random.default_rng(0)
        _, dim = docs.shape
        random_matrix = rng.standard_normal((dim, dim))
        Q, _ = np.linalg.qr(random_matrix, mode="reduced")
        docs, queries = int8_quantize(docs, Q=Q), int8_quantize(queries, Q=Q)
    if pca is not None:
        docs, queries = pca_reduce(docs, queries, pca)
    return docs, queries

VARIANTS: list[tuple[str, dict]] = []
# baselines
for dt in DTYPES:
    VARIANTS.append((f"baseline_{dt}", {"pca": None, "dtype": dt, "q8": False}))
# PCA only
for dt in DTYPES:
    for pf in PCA_FACTORS:
        VARIANTS.append((f"pca{pf}_{dt}", {"pca": pf, "dtype": dt, "q8": False}))
# int8 variants
for dt in DTYPES:
    VARIANTS.append((f"int8_{dt}", {"pca": None, "dtype": dt, "q8": True}))
    for pf in PCA_FACTORS:
        VARIANTS.append((f"int8_pca{pf}_{dt}", {"pca": pf, "dtype": dt, "q8": True}))

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

def benchmark():
    scores: dict[str, dict[str, float]] = {
        name: {"variant": name, **{p: None for p in PROVIDERS}} for name, _ in VARIANTS
    }

    for prov, meta in PROVIDERS.items():
        docs_fp32, queries_fp32 = load_embeddings(meta["path"], meta["col"], meta["dim"])
        ref_idxs, _ = vector_search(queries_fp32, docs_fp32, TOP_K)

        for name, cfg in VARIANTS:
            d, q = apply_variant(docs_fp32, queries_fp32, cfg["pca"], cfg["dtype"], cfg["q8"])
            idxs, _ = vector_search(q, d, TOP_K)
            scores[name][prov] = recall_at_k(ref_idxs, idxs)

    rows = list(scores.values())
    return pl.DataFrame(rows).sort("variant")

def main():
    df = benchmark()
    df.write_csv(OUT_CSV)
    print(df)

if __name__ == "__main__":
    main()
