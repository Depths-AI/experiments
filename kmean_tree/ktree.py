from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, List, Dict
import numpy as np
from heapq import heappush, heappop
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

Metric = Literal["cosine", "euclidean"]

# --------------------------- helpers -----------------------------------------
def _partial_topk(x: np.ndarray, k: int, smallest=True) -> np.ndarray:
    if k >= x.shape[0]:
        return np.arange(x.shape[0])
    if smallest:
        return np.argpartition(x, k)[:k]
    n = x.shape[0]
    return np.argpartition(x, n - k)[-k:]

# --------------------------- main class --------------------------------------
@dataclass
class KMeansTree3:
    # tree size
    n_l1: int = 1000; n_l2: int = 100; n_l3: int = 10
    # traversal limits
    k1: int = 30; k2: int = 20; k3: int = 10
    beam: int = 60               # <<< NEW best-bin-first budget
    m_assign: int = 2            # <<< NEW multi-assignment
    # misc
    metric: Metric = "cosine"
    random_state: int = 42
    n_init: int = 20             # stronger k-means
    max_iter: int = 100

    # learned state (identical names ⇢ compatible)
    docs: np.ndarray|None = None
    c1: np.ndarray|None = None; c2: np.ndarray|None = None; c3: np.ndarray|None = None
    l1_children_of_l2: List[np.ndarray]|None = None
    l2_children_of_l3: List[np.ndarray]|None = None
    doc_ids_of_l1: List[np.ndarray]|None = None

    # ---------------- private -------------------------------------------------
    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float32, order="C")
        if self.metric == "cosine":
            X = normalize(X, axis=1, copy=True).astype(np.float32)
        return X

    def _fit(self, X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        km = KMeans(k, n_init=self.n_init, max_iter=self.max_iter,
                    random_state=self.random_state, algorithm="lloyd")
        labels = km.fit_predict(X)
        C = km.cluster_centers_.astype(np.float32)
        if self.metric == "cosine":                     # spherical centroids
            C = normalize(C, axis=1, copy=False).astype(np.float32)
        return C, labels.astype(np.int32)

    def _dist(self, q: np.ndarray, C: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            return 1.0 - C @ q
        diff = C - q
        return np.einsum("ij,ij->i", diff, diff)

    # -------------------------- build ----------------------------------------
    def fit(self, docs: np.ndarray) -> "KMeansTree3":
        X = self._prep(docs); self.docs = X
        # L1
        self.c1, l1_lbl = self._fit(X, self.n_l1)
        # multi-assignment postings
        sim = X @ self.c1.T if self.metric == "cosine" else -(
              np.linalg.norm(X[:, None] - self.c1, axis=2)**2)
        topm = np.argpartition(-sim, self.m_assign-1, axis=1)[:, :self.m_assign]
        postings = [ [] for _ in range(self.n_l1) ]
        for doc_id, leaves in enumerate(topm):
            for l in leaves:
                postings[l].append(doc_id)
        self.doc_ids_of_l1 = [np.asarray(p, np.int32) for p in postings]
        # upper layers
        self.c2, l2_lbl = self._fit(self.c1, self.n_l2)
        self.c3, l3_lbl = self._fit(self.c2, self.n_l3)
        self.l1_children_of_l2 = [np.nonzero(l2_lbl == i)[0].astype(np.int32)
                                  for i in range(self.n_l2)]
        self.l2_children_of_l3 = [np.nonzero(l3_lbl == i)[0].astype(np.int32)
                                  for i in range(self.n_l3)]
        return self

    # -------------------------- search ---------------------------------------
    def search(self, q: np.ndarray, s: int = 10) -> Tuple[np.ndarray, np.ndarray, Dict]:
        q = self._prep(q.reshape(1, -1))[0] if self.metric == "cosine" else q.astype(np.float32)

        # priority queue: (dist , ("L3"| "L2"| "L1", idx))
        pq: List[Tuple[float, Tuple[str, int]]] = []
        for i, d in enumerate(self._dist(q, self.c3)):
            heappush(pq, (d, ("L3", i)))

        cand_ids: List[int] = []
        visited_l1 = set(); visited_l2 = set(); visited_l3 = set()

        while pq and len(cand_ids) < self.beam:
            d, (lvl, idx) = heappop(pq)
            if lvl == "L1":
                visited_l1.add(idx)
                cand_ids.extend(self.doc_ids_of_l1[idx])
            elif lvl == "L2":
                visited_l2.add(idx)
                for child in self.l1_children_of_l2[idx]:
                    heappush(pq, (self._dist(q, self.c1[child:child+1])[0],
                                  ("L1", child)))
            else:  # L3
                visited_l3.add(idx)
                for child in self.l2_children_of_l3[idx]:
                    heappush(pq, (self._dist(q, self.c2[child:child+1])[0],
                                  ("L2", child)))

        cand = np.unique(np.fromiter(cand_ids, np.int32))
        if cand.size == 0:
            return np.empty(0, np.int32), np.empty(0, np.float32), {"candidates": 0}

        d = self._dist(q, self.docs[cand])
        k_final = min(s, d.size)
        local = _partial_topk(d, k_final, smallest=True)
        order = np.argsort(d[local])
        top_ids = cand[local][order]
        top_d = d[local][order]

        stats = {
            "l3_selected": sorted(visited_l3),
            "l2_pool": int(sum(len(self.l2_children_of_l3[x]) for x in visited_l3)),
            "l2_selected": sorted(visited_l2),
            "l1_pool": int(sum(len(self.l1_children_of_l2[x]) for x in visited_l2)),
            "l1_selected": sorted(visited_l1),
            "candidates": int(cand.size),
            "beam_popped": len(visited_l1) + len(visited_l2) + len(visited_l3)
        }
        return top_ids.astype(np.int32), top_d.astype(np.float32), stats

    # identical signature
    def batch_search(self, Q: np.ndarray, s: int = 10) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
        Q = np.asarray(Q, np.float32)
        if self.metric == "cosine":
            Q = normalize(Q, axis=1, copy=True).astype(np.float32)
        ids_list, dists_list, cand = [], [], []
        for q in Q:
            ids, d, st = self.search(q, s)
            ids_list.append(ids); dists_list.append(d); cand.append(st["candidates"])
        return ids_list, dists_list, {
            "avg_candidates": float(np.mean(cand)) if cand else 0.0,
            "queries": int(Q.shape[0]),
        }
