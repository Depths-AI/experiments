import numpy as np
import polars as pl
import time
from search import *

NUM_VECS = 10000
NUM_QUERIES = 100
NUM_DIMS=1536
TOP_K=10
PCA_FACTOR=4
OVER_SAMPLE_FACTOR=[i for i in range(1,11,1)]
CSV_PATH=f"search_speed_{TOP_K}_{NUM_DIMS}_PCA_{PCA_FACTOR}.csv"

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


def binary_quantize_batch(vectors: np.ndarray, seed: int = 0):
    _, dims = vectors.shape

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dims, dims))
    Q, _ = np.linalg.qr(A, mode="reduced")

    projections = vectors @ Q
    bits_bool=projections>=0

    bin_signs=np.where(bits_bool,  1.0, -1.0)
    bit_norms= np.linalg.norm(bin_signs, axis=1)

    errors=np.divide((np.linalg.norm(projections - bin_signs, axis=1)),bit_norms)

    bits = bits_bool.astype(np.bool)
    packed= np.packbits(bits, axis=-1)
    packed= packed.view(np.uint64)
    return packed

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
    return idxs

def binary_vector_search(queries: np.ndarray, docs: np.ndarray, top_k: int = 10):
    '''
    Optimal NumPy+Numba routine using a single unified kernel.
    '''
    k = min(top_k, docs.shape[0])
    # The entire search logic is now inside this one call
    idxs = binary_search_kernel(docs, queries, k)
    return idxs

def recall_at_k(ref: np.ndarray, test: np.ndarray):
    '''
    Finding recall at k, even under oversampling
    '''
    recalls=[]
    for r,t in zip(ref, test):
        matches=np.intersect1d(r,t)
        recalls.append(matches.size/r.size)

    recalls=np.array(recalls)
    return round(recalls.mean(),3)

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

def hamming_warm_run():
    d=np.random.random(size=(10,NUM_DIMS))
    bits=(d >= 0).astype(np.bool)
    d=np.packbits(bits, axis=-1)
    d=d.view(np.uint64)
    
    q=np.random.random(size=(10,NUM_DIMS))
    bits=(q >= 0).astype(np.bool)
    q=np.packbits(bits, axis=-1)
    q=q.view(np.uint64)

    ids=binary_search_kernel(d,q,1)

def main():
    providers=[]
    num_vecs=[]
    times=[]
    b_times=[]
    r_times=[]

    recall=[]
    r_recall=[]

    o_sample=[]
    hamming_warm_run()
    for prov, meta in PROVIDERS.items():
        docs, queries = load_embeddings(meta["path"], meta["col"], meta["dim"])

        if PCA_FACTOR>0:
            docs_r, queries_r = pca_reduce(docs,queries,PCA_FACTOR)
            docs_b=binary_quantize_batch(docs_r)
            queries_b=binary_quantize_batch(queries_r)
        else:
            docs_b=binary_quantize_batch(docs)
            queries_b=binary_quantize_batch(queries)

        start_time=time.time_ns()
        idxs = vector_search(queries, docs, TOP_K)
        end_time=time.time_ns()
        bfs_time=(end_time-start_time)*1.0/1e6
        times.extend([bfs_time]*len(OVER_SAMPLE_FACTOR))

        for o in OVER_SAMPLE_FACTOR:
            num_vecs.append(NUM_VECS)
            providers.append(prov)
            o_sample.append(o)

            start_time=time.time_ns()
            b_idxs = binary_vector_search(queries_b, docs_b, TOP_K*o)
            end_time=time.time_ns()
            b_times.append((end_time-start_time)*1.0/1e6)

            if PCA_FACTOR>0:
                start_time=time.time_ns()
                r_idxs=vector_search(queries_r, docs_r,TOP_K*o)
                end_time=time.time_ns()
                r_times.append((end_time-start_time)*1.0/1e6)
                r=recall_at_k(idxs, r_idxs)
                r_recall.append(r)
            else:
                r_times.append(0)
                r_recall.append(0)
        
            r=recall_at_k(idxs, b_idxs)
            recall.append(r)

            
        print("Done with",prov)

    pl.DataFrame({
        "provider": providers,
        "num_vecs": num_vecs,
        "oversample": o_sample,
        "brute time (ms)": times,
        "reduced time (ms)":r_times,
        "binary time (ms)": b_times,
        "reduced vec. recall":r_recall,
        "binary recall": recall}).write_csv(CSV_PATH)

if __name__ == "__main__":
    main()