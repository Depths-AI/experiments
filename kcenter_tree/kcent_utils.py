import numpy as np
from numba import njit, prange

@njit(inline="always")
def _sqeuclidean(a: np.ndarray, b: np.ndarray) -> float:
    '''
    Evaluate squared Euclidean distance between two vectors a and b.
    Args:
        a: (D,) float32/float64
        b: (D,) float32/float64
    Returns:
        s: Squared distance float64
    '''
    s = 0.0
    for j in range(a.shape[0]):
        d = a[j] - b[j]
        s += d * d
    return s

@njit(inline="always")
def _dot(a: np.ndarray, b: np.ndarray) -> float:
    '''
    Evaluate dot product between two vectors a and b.

    Equivalent to cosine for normalized vectors.

    Args:
        a: (D,) float32/float64
        b: (D,) float32/float64
    Returns:
        s: Dot product float64
    '''
    s = 0.0
    for j in range(a.shape[0]):
        s += a[j] * b[j]
    return s

@njit(inline="always")
def _d2_unitnorm(a: np.ndarray, b: np.ndarray) -> float:
    '''
    Squared L2 distance for unit-normalized vectors:
      ||a - b||^2 = 2 - 2 * <a, b>
    Assumes ||a|| = ||b|| = 1.
    Args:
        a: (D,) float32/float64, unit-normalized vector
        b: (D,) float32/float64, unit-normalized vector
    Returns:
        d2: Squared distance float64
    Note: This is equivalent to 2.0 - 2.0 * np.dot(a, b) for unit-normalized vectors.
    It is faster than computing the full squared distance, as it avoids squaring the differences.
    '''
    return 2.0 - 2.0 * _dot(a, b)

@njit
def greedy_k_center_indices(X: np.ndarray, K: int, normalized: bool, start_index: int) -> np.ndarray:
    '''
    Greedily select K centers from X, maximizing the minimum distance to any point in X.
    Args:
        X: (N, D) float32, input data points
        K: int, number of centers to select
        normalized: bool, whether to use unit-normalized vectors (for cosine similarity)
        start_index: int, index of the first center to select (deterministic)
    Returns:
        centers_idx: (K,) int16, indices of selected centers in X
    Note: This is a greedy algorithm that provides a 2-approximation for the k-center problem.
    It iteratively selects the point that is farthest from the already selected centers.
    '''
    N, D = X.shape
    if K > N:
        K = N
    if K <= 0:
        return np.empty(0, dtype=np.int16)
    if K>=32767:
        return np.empty(0, dtype=np.int16)

    centers_idx = np.empty(K, dtype=np.int16)
    min_d2 = np.empty(N, dtype=np.float32)

    for i in range(N):
        min_d2[i] = np.inf

    c0 = np.int64(start_index)  
    centers_idx[0] = c0

    if normalized:
        for i in range(N):
            min_d2[i] = _d2_unitnorm(X[i], X[c0])
    else:
        for i in range(N):
            min_d2[i] = _sqeuclidean(X[i], X[c0])

    for t in range(1, K):
        far_idx = 0
        far_val = -1.0
        for i in range(N):
            v = min_d2[i]
            if v > far_val:
                far_val = v
                far_idx = i

        centers_idx[t] = far_idx

        if normalized:
            for i in range(N):
                d2 = _d2_unitnorm(X[i], X[far_idx])
                if d2 < min_d2[i]:
                    min_d2[i] = d2
        else:
            for i in range(N):
                d2 = _sqeuclidean(X[i], X[far_idx])
                if d2 < min_d2[i]:
                    min_d2[i] = d2

    return centers_idx

@njit
def _maxpos(arr: np.ndarray) -> int:
    '''
    Small utility to find the index of the maximum value in a 1D array.
    Args:
        arr: (N,) float32, input array
    Returns:
        p: int, index of the maximum value in arr
    Note: This is a simple linear search, suitable for small arrays.
    '''
    m = arr[0]
    p = 0
    for i in range(1, arr.shape[0]):
        v = arr[i]
        if v > m:
            m = v
            p = i
    return p

@njit
def _insertion_sort_by_key(keys: np.ndarray, vals: np.ndarray):
    '''
    Basic insertion sort that sorts keys in ascending order
    and reorders vals accordingly, in-place.
    Args:
        keys: (N,) float32, array of keys to sort
        vals: (N,) float32, array of values to reorder
    Since it is an in-place sort, it modifies keys and vals directly.
    Note: This is a simple sorting algorithm, suitable for small arrays.
    '''
    L = keys.shape[0]
    for i in range(1, L):
        key = keys[i]
        val = vals[i]
        j = i - 1
        while j >= 0 and keys[j] > key:
            keys[j + 1] = keys[j]
            vals[j + 1] = vals[j]
            j -= 1
        keys[j + 1] = key
        vals[j + 1] = val

@njit(parallel=True)
def assign_labels_topL(
    X: np.ndarray,         
    centers_idx: np.ndarray,  
    L: int = 1,
    normalized: bool = False
) -> np.ndarray:
    '''
    In k-center clustering, assign each point in X to the indices of the top L nearest centers.

    This ensures sufficient coverage of the dataset by the selected centers.
    Args:
        X: (N, D) float32, input data points
        centers_idx: (K,) int32, indices of selected centers in X
        L: int, number of nearest centers to return for each point (default: 3)
        normalized: bool, whether to use unit-normalized vectors (for cosine similarity)
    Returns:
        labels_topL: (N, L) int32, indices of the top L nearest centers for each point in X.
        Each row corresponds to a point in X, and contains the indices of the L nearest centers
    '''
    N, D = X.shape
    K = centers_idx.shape[0]
    if L > K:
        L = K

    labels_topL = np.empty((N, L), dtype=np.int32)

    for i in prange(N):
        best_d = np.empty(L, dtype=np.float32)
        best_k = np.empty(L, dtype=np.int32)
        for z in range(L):
            best_d[z] = np.inf
            best_k[z] = -1

        worst_pos = 0 

        filled = 0
        for t in range(K):
            cidx = centers_idx[t]
            if normalized:
                d2 = _d2_unitnorm(X[i], X[cidx])
            else:
                d2 = _sqeuclidean(X[i], X[cidx])

            if filled < L:
                best_d[filled] = d2
                best_k[filled] = t
                filled += 1
                if filled == L:
                    worst_pos = _maxpos(best_d)
            else:
                if d2 < best_d[worst_pos]:
                    best_d[worst_pos] = d2
                    best_k[worst_pos] = t
                    worst_pos = _maxpos(best_d)

        _insertion_sort_by_key(best_d, best_k)

        labels_topL[i, :] = best_k

    return labels_topL