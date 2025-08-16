import polars as pl
import numpy as np
from utils import *
import time

NUM_VECS=100_000
CHUNK_SIZE=10_000
NUM_QUERIES=1
K=100
TOP_C=20
TOP_K=10
NUM_DIMS=1536
PROVIDERS: dict[str, dict[str, object]] = {
    # "openai": {
    #     "path": "openai.parquet", # Download and rename the first file from huggingface link in the article/ README
    #     "col": "text-embedding-3-large-1536-embedding",
    #     "dim": 1536,
    # },
    "cohere": {
        "path": "cohere_v3.parquet", # Download and rename the first file from huggingface link in the article/ README
        "col": "emb",
        "dim": 1024,
    },
}

def load_embeddings(path: str, col: str, dim: int,offset:int, chunk_size: int=CHUNK_SIZE):
    df = (
        pl.scan_parquet(path)
        .slice(offset, CHUNK_SIZE + offset)
        .collect()
        .with_columns(pl.col(col).cast(pl.Array(pl.Float32, dim)).alias("vec"))
    )
    docs = df.head(chunk_size)["vec"].to_numpy()
    docs = docs/(np.linalg.norm(docs))

    return docs

def main():
    hamming_warm_run()
    for prov, meta in PROVIDERS.items():
        num_iterations=NUM_VECS//CHUNK_SIZE

        docs=np.empty(shape=(NUM_VECS,meta["dim"]),dtype=np.float32)
        queries=np.empty(shape=(NUM_VECS,meta["dim"]),dtype=np.float32)

        docs_b=np.empty(shape=(NUM_VECS,meta["dim"]//64),dtype=np.uint64)
        queries_b=np.empty(shape=(NUM_VECS,meta["dim"]//64),dtype=np.uint64)

        centroids=np.empty(shape=(K*num_iterations,meta["dim"]),dtype=np.float32)

        labels=np.empty(shape=(NUM_VECS,),dtype=np.int32)
        
        rng = np.random.default_rng(0)
        A=rng.standard_normal((meta["dim"], meta["dim"]))
        Q, _ = np.linalg.qr(A, mode="reduced")

        for i in range(num_iterations):
            docs[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]=load_embeddings(
                path=meta["path"], 
                col=meta["col"], 
                dim=meta["dim"],
                offset=i*CHUNK_SIZE,
                chunk_size=CHUNK_SIZE)
        
            docs_b[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]=binary_quantize_batch(
                docs[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE],Q)
            
            centroids[i*K:(i+1)*K], labels[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] = compute_cluster(
                docs[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE], K=K)

            labels[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE] += i*K
            print(f"Iteration {i+1}/{num_iterations} done")
        
        queries=load_embeddings(
            path=meta["path"], 
            col=meta["col"], 
            dim=meta["dim"],
            offset=NUM_VECS,
            chunk_size=NUM_QUERIES)
        
        queries_b=binary_quantize_batch(queries,Q)
        centroids_b=binary_quantize_batch(centroids,Q)

        brute_f_results=vector_search(queries,docs,TOP_K)
        
        for c in range(TOP_C, num_iterations*K+1, TOP_C):
            closest_centroids=vector_search(queries,centroids,c)
            _, top_vecs_id=filter_docs_by_query(docs,labels,closest_centroids)

            recall_centroid=proportion_in_filtered(brute_f_results,top_vecs_id)
            print(f"((Float centroids) Recall@10 at centroid filter level for TOP C={c}:",recall_centroid.mean())

        for c in range(TOP_C, num_iterations*K+1, TOP_C):
            closest_centroids=binary_vector_search(queries_b,centroids_b,c)
            _, top_vecs_id=filter_docs_by_query(docs,labels,closest_centroids)

            recall_centroid=proportion_in_filtered(brute_f_results,top_vecs_id)
            print(f"((Binarized centroids) Recall@10 at centroid filter level for TOP C={c}:",recall_centroid.mean())

        print("All calculations done")
        

if __name__ == "__main__":
    main()     