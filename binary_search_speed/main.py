import numpy as np
import polars as pl
import time

NUM_VECS = [10_000*i for i in range(1,2,1)]
NUM_QUERIES = 100
NUM_DIMS=1536
TOP_K=100
CSV_PATH=f"search_speed_{NUM_QUERIES}_{NUM_DIMS}.csv"

def binary_quantize_batch(vectors: np.ndarray, seed: int = 0) -> np.ndarray[np.uint8]:
    bits = (vectors >= 0).astype(np.bool)
    return np.packbits(bits, axis=-1,bitorder="big")

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

def binary_vector_search(queries: np.ndarray, docs: np.ndarray, top_k: int = 10):
    '''
    Optimal NumPy routine to perform binary search for a batch of queries using popcount hamming distance.
    '''
    # The arrays are 2D, so we need to add a new axis to one of them to broadcast correctly.
    # queries is (n_queries, n_bytes), docs is (n_docs, n_bytes)
    # We want a result of shape (n_docs, n_queries)
    xor_result = np.bitwise_xor(docs[:, np.newaxis, :], queries)
    
    # popcount calculates the number of set bits (1s)
    distances = np.unpackbits(xor_result, axis=-1, bitorder='big').sum(axis=-1)
    
    k = min(top_k, distances.shape[0])
    
    # We want the smallest distances, so we don't negate `distances`
    top = np.argpartition(distances, k - 1, axis=0)[:k]
    
    top_distances = np.take_along_axis(distances, top, axis=0)
    order = np.argsort(top_distances, axis=0)
    
    idxs = np.take_along_axis(top, order, axis=0).T
    return idxs, None # Not returning similarity scores for now

def recall_at_k(ref: np.ndarray, test: np.ndarray) -> float:
    '''
    Simple NumPy routine to compute recall@k for a given result compared to float32 brute force search as the reference.
    '''
    return float((ref[:, :, None] == test[:, None, :]).any(axis=2).mean())


def main():
    times=[]
    b_times=[]
    recall=[]
    for n_vecs in NUM_VECS:
        docs=np.random.random((n_vecs, NUM_DIMS))

        docs_b=binary_quantize_batch(docs)

        queries=np.random.random((NUM_QUERIES, NUM_DIMS))

        queries_b=binary_quantize_batch(queries)

        start_time=time.time_ns()
        idxs, _ = vector_search(queries, docs, TOP_K)
        end_time=time.time_ns()
        times.append((end_time-start_time)*1.0/1e6)
        print(f"Time taken (ms) for search for {n_vecs} vecs with dimensions {NUM_DIMS} among {n_vecs} docs: {(end_time-start_time)*1.0/1e6}")

        start_time=time.time_ns()
        b_idxs, _ = binary_vector_search(queries_b, docs_b, TOP_K)
        end_time=time.time_ns()
        b_times.append((end_time-start_time)*1.0/1e6)
        print(f"Time taken (ms) for binary search for {n_vecs} vecs with dimensions {NUM_DIMS} among {n_vecs} docs: {(end_time-start_time)*1.0/1e6}")
        
        r=recall_at_k(idxs, b_idxs)
        print(f"Recall @ {TOP_K} for binary search: {r}")
        recall.append(r)

    pl.DataFrame({
        "n_vecs": NUM_VECS,
        "brute time (ms)": times,
        "binary time (ms)": b_times,
        "binary recall": recall}).write_csv(CSV_PATH)

if __name__ == "__main__":
    main()