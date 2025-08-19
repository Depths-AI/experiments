import numpy as np
from search import *
from typing import Optional
from kcent_utils import greedy_k_center_indices, assign_labels_topL
from binary_kcenter import hamming_greedy_k_center_indices,hamming_assign_labels_topL, hamming_refine_centers_majority

def greedy_k_center(
    X: np.ndarray,
    K: int,
    num_centers: int=3,
    normalized: bool = True,
    start_index: int = 0,
    dtype = np.float32
):
    """
    Greedy k-center clustering (Gonzalez).
    Parameters
    ----------
    X : (N, D) array-like
        Document/embedding matrix.
    K : int
        Number of clusters (clamped to N).
    normalized : bool, default False
        If True, uses the unit-norm shortcut d^2 = 2 - 2 * dot(x, c).
        Only set True if rows of X are L2-normalized.
    start_index : int, default 0
        Index of the initial center. Any index is valid with the 2-approx guarantee.
    dtype : numpy dtype, default float32
        X is cast to this dtype before clustering.

    Returns
    -------
    centers : (K, D) ndarray
        Selected centers (subset of X).
    labels : (N,) ndarray of int64
        Index into 0..K-1 for each row of X.
    centers_idx : (K,) ndarray of int64
        Indices into X for the chosen centers.
    """
    X = np.asarray(X, dtype=dtype)
    centers_idx = greedy_k_center_indices(X, int(K), bool(normalized), int(start_index))
    labels = assign_labels_topL(X, centers_idx,num_centers, bool(normalized))
    centers = X[centers_idx].copy()  # materialize to decouple from X
    return centers, labels, centers_idx

def hamming_greedy_k_center(
        X: np.ndarray,
    K: int,
    num_centers: int=1,
    start_index: int = 0
):
    """
    Greedy k-center clustering (Gonzalez).
    Parameters
    ----------
    X : (N, D//64) array-like
        Binarized bitpacked Document/embedding matrix.
    K : int
        Number of clusters (clamped to N).
    start_index : int, default 0
        Index of the initial center. Any index is valid with the 2-approx guarantee.
    dtype : numpy dtype, default float32
        X is cast to this dtype before clustering.

    Returns
    -------
    centers : (K, D) ndarray
        Selected centers (subset of X).
    labels : (N,) ndarray of int64
        Index into 0..K-1 for each row of X.
    centers_idx : (K,) ndarray of int64
        Indices into X for the chosen centers.
    """
    centers_idx = hamming_greedy_k_center_indices(X, int(K), int(start_index))
    labels = hamming_assign_labels_topL(X, centers_idx,num_centers)
    centers = X[centers_idx].copy()  # materialize to decouple from X
    #centers= hamming_refine_centers_majority(X, centers_idx, labels)
    return centers, labels, centers_idx

def search_centroids(queries: np.ndarray, centroids: np.ndarray, top_c: int):
    dists = np.sum((queries[:, None, :] - centroids[None, :, :]) ** 2, axis=2)

    return np.argsort(dists, axis=1)[:, :top_c]

def filter_docs_by_query(doc_vectors: np.ndarray,
                         labels: np.ndarray,
                         top_c_indices: np.ndarray) -> list[np.ndarray]:

    Q = top_c_indices.shape[0]
    filtered_docs: list[np.ndarray] = []
    filtered_indices: list[np.ndarray] = []

    for q in range(Q):
        valid = top_c_indices[q]                                     # (C,)
        # isin broadcasts over 'labels' and checks membership in 'valid'
        hit = np.isin(labels, valid)                                 # (N, L) bool
        mask = np.any(hit, axis=1)                                   # (N,)  bool
        idx = np.nonzero(mask)[0]                                    # (M,)
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

def binary_quantize_batch(vectors: np.ndarray, Q:Optional[np.ndarray]=None):
    _, dims = vectors.shape

    if Q is None:
        rng = np.random.default_rng(0)
        A=rng.standard_normal((dims, dims))
        Q, _ = np.linalg.qr(A, mode="reduced")
    Q=np.ascontiguousarray(Q, dtype=np.float32)

    projections = vectors @ Q

    packed = pack_signs_to_uint64(projections)
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

def pca(docs: np.ndarray):
    '''
    Simple NumPy routine to perform PCA on a batch of docs
    '''
    original_dim = docs.shape[1]
    mean = docs.mean(axis=0, keepdims=True)
    docs_c = docs - mean

    docs_c = docs_c.astype(np.float32, copy=True)

    # To align the data with its principal components without reducing dimensionality,
    # we need the eigenvectors of the covariance matrix.
    # These eigenvectors form the rotation matrix.
    covariance_matrix = np.cov(docs_c, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # The rotated data is the product of the centered data and the eigenvectors.
    # This aligns the data with the principal axes.
    docs_red = docs_c @ eigenvectors

    return docs_red

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