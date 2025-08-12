import polars as pl
import numpy as np
from utils import *
import time

NUM_VECS=10_000
NUM_QUERIES=100
K=100
TOP_C=30
TOP_K=10
NUM_DIMS=1536
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

def main():

    for prov, meta in PROVIDERS.items():
        docs, queries = load_embeddings(meta["path"], meta["col"], meta["dim"])

        start_time=time.time_ns()
        centroids, labels = compute_kmeans(docs, K=K, max_iter=1000)
        end_time=time.time_ns()
        print("KMeans time (ms):",(end_time-start_time)*1.0/1e6)

        brute_f_results=vector_search(queries,docs,TOP_K)
        
        closest_centroids=search_centroids(queries,centroids,TOP_C)
        top_vecs, top_vecs_id=filter_docs_by_query(docs,labels,closest_centroids)

        recall_centroid=proportion_in_filtered(brute_f_results,top_vecs_id)
        print("Recall@10 at centroid filter level:",recall_centroid.mean())



if __name__ == "__main__":
    main()     