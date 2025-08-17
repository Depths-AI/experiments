"""
End-to-end: generate random floats -> binarize -> packbits -> build Faiss IndexBinaryHNSW
Requirements:
  pip install faiss-cpu numpy
(Or install via conda per Faiss docs: conda install -c pytorch faiss-cpu)
See Faiss binary index docs: https://github.com/facebookresearch/faiss/wiki/Binary-indexes
"""

import time
import numpy as np
import faiss

# --- PARAMETERS (edit as needed) ---
NUM_VECS = 10000      # e.g., 1_000, 10_000, 100_000, ...
D_BITS = 1000           # number of bits per vector (must be multiple of 8; we'll pad if not)
M = 32                  # HNSW connectivity
ef_construction = 400    # exploration during construction
NUM_QUERIES = 1          # number of queries to test after building index
# --- END PARAMETERS ---
# -------------------------------

# 1) ensure D_BITS multiple of 8 (Faiss requires vectors size multiple of 8)
pad_bits = (8 - (D_BITS % 8)) % 8
D_bits_padded = D_BITS + pad_bits
bytes_per_vector = D_bits_padded // 8

print("NUM_VECS =", NUM_VECS)

# 2) generate random floats and binarize with element > 0
#    We use a normal distribution; change RNG/threshold if you want other sparsity.
rng = np.random.default_rng(12345)
# create boolean matrix shape (NUM_VECS, D_bits_padded)
# first generate only D_BITS then pad zeros if needed
Xf = rng.standard_normal(size=(NUM_VECS, D_BITS)).astype(np.float32)
# binarize: True where > 0
Xb_bool = (Xf > 0)

# if padding required, append zeros (False)
if pad_bits:
    pad = np.zeros((NUM_VECS, pad_bits), dtype=bool)
    Xb_bool = np.concatenate([Xb_bool, pad], axis=1)

assert Xb_bool.shape[1] == D_bits_padded

# 3) pack bits into uint8 per row (shape -> (NUM_VECS, bytes_per_vector))
#    Use bitorder='big' so the first eight booleans in each row map to the first byte
X_packed = np.packbits(Xb_bool, axis=1, bitorder='big').astype(np.uint8)
assert X_packed.shape == (NUM_VECS, bytes_per_vector)

# Faiss expects a contiguous uint8 array flattened when calling add()
X_packed_c = np.ascontiguousarray(X_packed)

# 4) create IndexBinaryHNSW
#    In Python Faiss API the constructor mirrors the C++ IndexBinaryHNSW(d, M)
index = faiss.IndexBinaryHNSW(D_bits_padded, M)

# set efConstruction before adding the vectors
# (the .hnsw object and its fields are present for HNSW indices in Faiss)
index.hnsw.efConstruction = ef_construction

# 5) time the build (index.add)
t0 = time.time()
index.add(X_packed_c)    # add expects a uint8 matrix of shape (n, d/8)
t1 = time.time()
build_seconds = t1 - t0

print(f"Build time: {build_seconds:.6f} seconds")

# optional: save index to disk
# faiss.write_index(index, "binary_hnsw.index")   # works for binary indices too

# small verification: query the index with a few packed vectors
q = X_packed_c[:NUM_QUERIES]  # 5 queries
k = 3
t0 = time.time_ns()
D, I = index.search(q, k)
t1 = time.time_ns()
search_seconds = t1 - t0
print(f"Search time for {k} nearest neighbors for {NUM_QUERIES} queries: {(search_seconds*1.0/1e3):.6f} us")