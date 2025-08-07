import numpy as np
import polars as pl
import time

NUM_VECS = [10_000*i for i in range(1,11,1)]
NUM_QUERIES = 1
NUM_DIMS=1536
CSV_PATH=f"search_speed_{NUM_QUERIES}_{NUM_DIMS}.csv"

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

def main():
    times=[]
    for n_vecs in NUM_VECS:
        docs=np.random.random((n_vecs, NUM_DIMS))
        queries=np.random.random((NUM_QUERIES, NUM_DIMS))
        start_time=time.time_ns()
        idxs, _ = vector_search(queries, docs)
        end_time=time.time_ns()
        times.append((end_time-start_time)*1.0/1e6)
        print(f"Time taken (ms) for search for {n_vecs} vecs with dimensions {NUM_DIMS} among {n_vecs} docs: {(end_time-start_time)*1.0/1e6}")
    pl.DataFrame({"n_vecs": NUM_VECS, "time (ms)": times}).write_csv(CSV_PATH)

if __name__ == "__main__":
    main()