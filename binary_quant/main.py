import polars as pl
import numpy as np
import time
from utils import kmeans, vector_search
import os
from shutil import rmtree

NUM_VECS = 10000
NUM_QUERIES = 1000
TOP_K = 10
NUM_SHARDS=2
CLUSTER_COUNT=100

CLUSTER_READ=(NUM_SHARDS*CLUSTER_COUNT)//100
PCA_FACTORS = [4]
DTYPES = ["fp32", "fp16"]
OUT_CSV = f"benchmark_results_k_{TOP_K}.csv"

if os.path.exists("data/test"):
    rmtree("data/test")
    os.mkdir("data/test")
else:
    os.mkdir("data/test")

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

def prep(X):
    X = X.astype(np.float32, copy=True)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    X /= np.where(n == 0, 1, n)
    return X

def batch_vector_search(queries: np.ndarray, docs: np.ndarray, top_k: int):
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
    return idxs, None 

def recall_at_k(ref: np.ndarray, test: np.ndarray) -> float:
    '''
    Simple NumPy routine to compute recall@k for a given result compared to float32 brute force search as the reference.
    '''
    return float((ref[:, :, None] == test[:, None, :]).any(axis=2).mean())

docs_openai, queries_openai = load_embeddings(**PROVIDERS["openai"])
docs_openai=prep(docs_openai)
queries_openai=prep(queries_openai)

sharded_docs=np.split(docs_openai, NUM_SHARDS)
all_centroids=np.empty((NUM_SHARDS, CLUSTER_COUNT, docs_openai.shape[1]), dtype=np.float16)

for s_num,s in enumerate(sharded_docs):
    start_time = time.time()
    c,l,i=kmeans(s, CLUSTER_COUNT)
    end_time = time.time()
    print(f"Time taken for k-means on {s.shape[0]} vecs with dimensions {docs_openai.shape[1]} into {CLUSTER_COUNT} clusters: {end_time - start_time}")
    all_centroids[s_num]=c.astype(np.float16)

    for j,cluster in enumerate(l):
        cluster_store_path=f"data/test/cluster_{s_num}_{j}.npy"
        np.save(cluster_store_path, cluster.astype(np.float16))

centroid_store_path=f"data/test/centroids.npy"
np.save(centroid_store_path, all_centroids)

all_centroids=np.load("data/test/centroids.npy").reshape((NUM_SHARDS*CLUSTER_COUNT, docs_openai.shape[1]))
print(all_centroids.shape, all_centroids.dtype)

# start_time=time.time()
# brute_results, brute_sims = batch_vector_search(queries_openai, docs_openai, TOP_K)
# end_time = time.time()
# print(f"Time taken for brute force vector search for {queries_openai.shape[0]} queries with dimensions {docs_openai.shape[1]} among {docs_openai.shape[0]} docs: {end_time - start_time}")

# start_time = time.time()
# idxs, sims = batch_vector_search(queries_openai, all_centroids, CLUSTER_READ)
# end_time = time.time()
# print(f"Time taken for vector search for {queries_openai.shape[0]} queries with dimensions {docs_openai.shape[1]} among {CLUSTER_COUNT} cluster centroids: {end_time - start_time}")

# for i in idxs:
#     shard_id=i//CLUSTER_COUNT
#     cluster_id=i%CLUSTER_COUNT

# coarse_search_results=np.empty((queries_openai.shape[0], TOP_K), dtype=np.int64)
# for i in range(queries_openai.shape[0]):
#     relevant_docs=[]
#     for j in range(CLUSTER_READ):
#         shard_id=idxs[i][j]//CLUSTER_COUNT
#         cluster_id=idxs[i][j]%CLUSTER_COUNT
#         read_docs=np.load(f"data/test/cluster_{shard_id}_{cluster_id}.npy")
#         relevant_docs.append(read_docs)

#     relevant_docs=np.concatenate(relevant_docs, axis=0)
#     preprocessed_results, _ =batch_vector_search(queries_openai[i], relevant_docs, TOP_K)
    

# print(coarse_search_results[0])
# print(brute_results[0])

# recall_score=recall_at_k(brute_results, coarse_search_results)
# print(f"Recall@{TOP_K}: {recall_score}")

rmtree("data/test")
