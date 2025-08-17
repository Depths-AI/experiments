import numpy as np
from numba import njit, prange

# ------------------------
# Low-level distance helpers
# ------------------------

@njit(inline="always")
def _sqeuclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Squared L2 distance."""
    s = 0.0
    for j in range(a.shape[0]):
        d = a[j] - b[j]
        s += d * d
    return s

@njit(inline="always")
def _dot(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product."""
    s = 0.0
    for j in range(a.shape[0]):
        s += a[j] * b[j]
    return s

@njit(inline="always")
def _d2_unitnorm(a: np.ndarray, b: np.ndarray) -> float:
    """
    Squared L2 distance for unit-normalized vectors:
      ||a - b||^2 = 2 - 2 * <a, b>
    Assumes ||a|| = ||b|| = 1.
    """
    return 2.0 - 2.0 * _dot(a, b)

# ------------------------
# Greedy k-center (indices)
# ------------------------

@njit
def greedy_k_center_indices(X: np.ndarray, K: int, normalized: bool, start_index: int) -> np.ndarray:
    """
    Select K center indices by farthest-first traversal (Gonzalez).
    X: (N, D) float32/float64
    Returns: centers_idx (K,)
    """
    N, D = X.shape
    if K > N:
        K = N

    centers_idx = np.empty(K, dtype=np.int64)
    min_d2 = np.empty(N, dtype=np.float64)

    # Initialize min distances to +inf
    for i in range(N):
        min_d2[i] = np.inf

    # Pick initial center
    c0 = np.int64(start_index)  # deterministic; any point works for the 2-approx
    centers_idx[0] = c0

    # First update: distances to c0
    if normalized:
        for i in range(N):
            min_d2[i] = _d2_unitnorm(X[i], X[c0])
    else:
        for i in range(N):
            min_d2[i] = _sqeuclidean(X[i], X[c0])

    # Iteratively add farthest point
    for t in range(1, K):
        # argmax over current min_d2
        far_idx = 0
        far_val = -1.0
        for i in range(N):
            v = min_d2[i]
            if v > far_val:
                far_val = v
                far_idx = i

        centers_idx[t] = far_idx

        # Update min_d2 with the new center
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

# ------------------------
# Label assignment (parallel)
# ------------------------

@njit(parallel=True)
def assign_labels(X: np.ndarray, centers_idx: np.ndarray, normalized: bool) -> np.ndarray:
    """
    Assign each point to nearest selected center (returns labels of shape (N,)).
    """
    N, D = X.shape
    K = centers_idx.shape[0]
    labels = np.empty(N, dtype=np.int64)

    for i in prange(N):
        best_k = 0
        # initialize with first center
        if normalized:
            best = _d2_unitnorm(X[i], X[centers_idx[0]])
        else:
            best = _sqeuclidean(X[i], X[centers_idx[0]])

        for t in range(1, K):
            if normalized:
                d2 = _d2_unitnorm(X[i], X[centers_idx[t]])
            else:
                d2 = _sqeuclidean(X[i], X[centers_idx[t]])
            if d2 < best:
                best = d2
                best_k = t
        labels[i] = best_k

    return labels

@njit
def _maxpos(arr: np.ndarray) -> int:
    """Index of maximum element in a 1D float64 array (length L, L is small)."""
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
    """Sort (keys, vals) by ascending keys in-place. L is small, O(L^2) is fine."""
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
    X: np.ndarray,         # (N, D)
    centers_idx: np.ndarray,  # (K,)
    L: int = 3,
    normalized: bool = False
) -> np.ndarray:
    """
    For each row in X, return indices of the top-L nearest centers (shape (N, L)),
    ordered by increasing distance (nearest first).
    """
    N, D = X.shape
    K = centers_idx.shape[0]
    if L > K:
        L = K

    labels_topL = np.empty((N, L), dtype=np.int64)

    for i in prange(N):
        # small fixed-size buffers for best-L
        best_d = np.empty(L, dtype=np.float64)
        best_k = np.empty(L, dtype=np.int64)
        for z in range(L):
            best_d[z] = np.inf
            best_k[z] = -1

        # Track position of current worst among the best-L
        worst_pos = 0  # arbitrary init; valid once buffer fills

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
                # replace current worst if new distance is better
                if d2 < best_d[worst_pos]:
                    best_d[worst_pos] = d2
                    best_k[worst_pos] = t
                    # recompute worst_pos (L is small)
                    worst_pos = _maxpos(best_d)

        # order (ascending) so column 0 is the nearest
        _insertion_sort_by_key(best_d, best_k)

        # store center indices (not positions t) -> map to actual indices if needed outside
        # Here we store the *position in centers_idx* (consistent with previous _assign_labels).
        labels_topL[i, :] = best_k

    return labels_topL