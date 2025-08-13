from sklearn.cluster import KMeans
import numpy as np
from search import *

def compute_cluster(X: np.ndarray, K: int, **kwargs):
    """
    Perform KMeans clustering on X and return:
    - centroids: (K, D) ndarray
    - labels: (N,) ndarray of centroid indices for each sample
    """
    #X=X-X.mean(axis=0)
    kmeans = KMeans(n_clusters=K, **kwargs)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    centroids=centroids/(np.linalg.norm(centroids))
    
    return centroids, labels

def search_centroids(queries: np.ndarray, centroids: np.ndarray, top_c: int):
    dists = np.sum((queries[:, None, :] - centroids[None, :, :]) ** 2, axis=2)

    return np.argsort(dists, axis=1)[:, :top_c]

def filter_docs_by_query(doc_vectors: np.ndarray,
                         labels: np.ndarray,
                         top_c_indices: np.ndarray) -> list[np.ndarray]:

    Q = top_c_indices.shape[0]
    filtered_docs = []
    filtered_indices = []

    for q in range(Q):
        valid_labels = top_c_indices[q]                     # shape (top_c,)
        mask = np.isin(labels, valid_labels)               # shape (N,)
        idx = np.nonzero(mask)[0]                          # indices of matching docs
        filtered_indices.append(idx)
        filtered_docs.append(doc_vectors[idx])

    return filtered_docs, filtered_indices

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

def proportion_in_filtered(brute_indices: np.ndarray,
                           filtered_indices: list[np.ndarray]) -> np.ndarray:
    """
    Returns an array of shape (Q,) with the proportion of brute_indices present in filtered_indices for each query.

    Params:
        brute_indices: np.ndarray of shape (Q, top_k) — indices from brute-force search.
        filtered_indices: list of length Q — each a 1D array of indices kept post-filter.

    Returns:
        np.ndarray of shape (Q,) with float proportions between 0 and 1.
    """
    Q, top_k = brute_indices.shape
    proportions = np.empty(Q, dtype=float)

    for q in range(Q):
        brute = brute_indices[q]
        filt = filtered_indices[q]
        # Count how many brute indices are in the filtered set
        count = np.isin(brute, filt).sum()
        proportions[q] = count / top_k

    return proportions

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

def binary_vector_search(queries: np.ndarray, docs: np.ndarray, top_k: int = 10):
    '''
    Optimal NumPy+Numba routine using a single unified kernel.
    '''
    k = min(top_k, docs.shape[0])
    # The entire search logic is now inside this one call
    idxs = binary_search_kernel(docs, queries, k)
    return idxs

def hamming_warm_run():
    d=np.random.random(size=(10,1024))
    bits=(d >= 0).astype(np.bool)
    d=np.packbits(bits, axis=-1)
    d=d.view(np.uint64)
    
    q=np.random.random(size=(10,1024))
    bits=(q >= 0).astype(np.bool)
    q=np.packbits(bits, axis=-1)
    q=q.view(np.uint64)

    ds=binary_search_kernel(d,q,1)