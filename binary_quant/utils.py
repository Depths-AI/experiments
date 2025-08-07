import numpy as np
from numba import njit, prange

# ---------- helpers ----------

@njit(cache=True)
def squared_euclidean(A, B):
    """
    A: (n, d), B: (k, d)
    returns (n, k) of ||A_i - B_j||^2 without python loops.
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    An = np.sum(A*A, axis=1)          # (n,)
    Bn = np.sum(B*B, axis=1)          # (k,)
    # reshape for broadcasting inside numba
    return (An[:, None] + Bn[None, :] - 2.0 * (A @ B.T))


@njit(cache=True)
def assign_points(X, centroids):
    # return cluster indices and distances
    dist = squared_euclidean(X, centroids)
    labels = np.argmin(dist, axis=1)
    return labels


@njit(cache=True, parallel=True)
def update_centroids(X, labels, k):
    n_features = X.shape[1]
    newc = np.zeros((k, n_features), dtype=X.dtype)
    counts = np.zeros(k, dtype=np.int64)

    # accumulate sums in parallel-friendly way
    for i in prange(X.shape[0]):
        c = labels[i]
        counts[c] += 1
        newc[c] += X[i]

    # divide (avoid zero-div by leaving unchanged if empty)
    for c in range(k):
        if counts[c] > 0:
            newc[c] /= counts[c]
    return newc, counts


@njit(cache=True)
def inertia(X, centroids, labels):
    dist = squared_euclidean(X, centroids)
    total = 0.0
    for i in range(X.shape[0]):
        total += dist[i, labels[i]]
    return total


def kmeans_plus_plus_init(X, k, rng_seed=1234):
    """
    Simple k-means++ initializer in numba.
    """
    n = X.shape[0]
    rng = np.random.RandomState(rng_seed)
    centroids = np.empty((k, X.shape[1]), dtype=X.dtype)

    # pick first randomly
    idx0 = rng.randint(0, n)
    centroids[0] = X[idx0]

    # distances to nearest chosen centroid
    d2 = squared_euclidean(X, centroids[0:1]).ravel()

    for c in range(1, k):
        # choose next weighted by distance^2
        probs = d2 / d2.sum()
        # cumulative sampling
        r = rng.rand()
        cum = 0.0
        pick = 0
        for i in range(n):
            cum += probs[i]
            if cum >= r:
                pick = i
                break
        centroids[c] = X[pick]
        # update d2
        d2 = np.minimum(d2, squared_euclidean(X, centroids[c:c+1]).ravel())
    return centroids


# ---------- main API ----------

def kmeans(X, k, max_iter=100, tol=1e-4, init="k-means++", rng_seed=1234, verbose=False):
    """
    X: (N, D) float32/64
    returns centroids, labels, inertia_
    """
    X = np.ascontiguousarray(X)  # numba-friendly

    if init == "random":
        rng = np.random.default_rng(rng_seed)
        init_idx = rng.choice(X.shape[0], k, replace=False)
        centroids = X[init_idx].copy()
    elif init == "k-means++":
        centroids = kmeans_plus_plus_init(X, k, rng_seed)
    else:
        raise ValueError("init must be 'random' or 'k-means++'")

    prev_inertia = np.inf

    for it in range(max_iter):
        labels = assign_points(X, centroids)
        centroids, counts = update_centroids(X, labels, k)

        # handle empty clusters by re-seeding them randomly
        empties = np.where(counts == 0)[0]
        if empties.size:
            rng = np.random.default_rng(rng_seed + it)
            repl = rng.choice(X.shape[0], empties.size, replace=False)
            centroids[empties] = X[repl]

        curr_inertia = inertia(X, centroids, labels)
        if verbose:
            print(f"iter {it}: inertia={curr_inertia:.4f}")

        if abs(prev_inertia - curr_inertia) <= tol * prev_inertia:
            break
        prev_inertia = curr_inertia
    
    # counts per cluster
    counts = np.bincount(labels, minlength=k)

    # sort by label once, then split
    order = np.argsort(labels)
    X_sorted = X[order]
    splits = np.cumsum(counts)[:-1]
    clusters = np.split(X_sorted, splits)   # list of length k

    # indices per cluster (if needed)
    idx_sorted = order
    idx_clusters = np.split(idx_sorted, splits)

    return centroids, clusters, curr_inertia

@njit(cache=True)
def vector_search(queries: np.ndarray, docs: np.ndarray, top_k: int):
    '''
    Optimal NumPy routine to perform search for a batch of queries

    Note that, for compute, we are still relying on projecting our stored vector
    to float32. Ofcourse, translating float16 or int8 back to float32 does not
    carry the same precision as the original vector.
    '''
    sims_full = docs @ queries.T

    N, Q = sims_full.shape
    k = top_k if top_k < N else N

    idxs  = np.empty((Q, k), dtype=np.int64)
    sims  = np.empty((Q, k), dtype=sims_full.dtype)

    for j in range(Q):  # loop is fine in njit
        col = sims_full[:, j]

        # 1-D argpartition
        top = np.argpartition(-col, k - 1)[:k]
        vals = col[top]

        # order those k
        ordk = np.argsort(-vals)

        idxs[j, :] = top[ordk]
        sims[j, :] = vals[ordk]

    return idxs, sims