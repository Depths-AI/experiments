# Inverted File (IVF) with Binary Quantization for Approximate Nearest Neighbor Search

This project implements an Approximate Nearest Neighbor (ANN) search algorithm using an Inverted File (IVF) structure combined with binary quantization for efficient and fast vector search. The core idea is to reduce the search space by first identifying promising candidates (via clustering) and then performing a very fast distance calculation (Hamming distance) on their binary representations.

## Algorithm Overview

The process can be broken down into two main phases: **Indexing** and **Searching**.

### 1. Indexing Phase

The goal of the indexing phase is to pre-process the document vectors into a structure that can be searched efficiently.

#### a. K-Means Clustering
First, all document vectors are clustered using the K-Means algorithm. This partitions the entire dataset into `K` distinct clusters.
- **Function**: [`utils.compute_cluster()`](ivf_quant/utils.py:5)
- Each cluster is represented by its centroid (the mean of all vectors in that cluster).
- The output of this step is a set of `K` centroids and a `labels` array, where `labels[i]` is the cluster ID for the `i`-th document vector.

#### b. Binary Quantization
To enable extremely fast distance calculations, all document vectors and centroids are converted from floating-point representations to a binary format.
- **Function**: [`utils.binary_quantize_batch()`](ivf_quant/utils.py:60)
- This function uses a random projection matrix to project the vectors and then assigns a binary `1` or `0` based on the sign of the projected values.
- These bits are then packed into `uint64` integers for efficient storage and computation.

The combination of the cluster assignments (the "inverted file") and the quantized vectors forms our index.

### 2. Searching Phase

When a new query vector arrives, the following steps are performed to find its nearest neighbors:

#### a. Identify Candidate Clusters
Instead of searching through all documents, we first find the clusters that are most likely to contain the nearest neighbors.
1.  The incoming query vector is also quantized into its binary representation using the same process as the document vectors.
2.  The quantized query is compared against all the quantized centroids to find the `TOP_C` closest centroids.
- **Function**: [`utils.binary_vector_search()`](ivf_quant/utils.py:131) which calls [`search.binary_search_kernel()`](ivf_quant/search.py:21)
- The "distance" used here is the **Hamming distance**, which is the number of differing bits between two binary vectors. This is calculated efficiently using bitwise XOR and a `popcount` operation.

#### b. Filter Documents
Once the `TOP_C` most promising clusters are identified, we create a candidate set of documents to search. This set consists of all the documents belonging to any of these `TOP_C` clusters.
- **Function**: [`utils.filter_docs_by_query()`](ivf_quant/utils.py:26)

#### c. Perform Exact Search on Candidate Set
A final, exhaustive search is performed, but only on the much smaller, filtered set of candidate documents.
1.  The Hamming distance is calculated between the quantized query vector and each of the quantized candidate document vectors.
2.  A **Max-Heap** data structure of size `TOP_K` is used to keep track of the `TOP_K` nearest neighbors found so far.
- **Heap Implementation**: [`heap.py`](ivf_quant/heap.py)
- For each candidate document, its Hamming distance to the query is computed. If the distance is smaller than the largest distance currently in the heap (the root of the max-heap), the root is replaced with the new, closer document.
- This avoids sorting the entire list of candidate distances and is very efficient for finding a small number of top results.

#### d. Return Results
After checking all candidate documents, the heap contains the `TOP_K` documents with the smallest Hamming distance to the query. These are the final results of the search.

## Core Components

- **[`main.py`](ivf_quant/main.py)**: The main script that orchestrates the loading of data, indexing, and searching processes. It also contains logic for evaluating the recall of the IVF-based search against a brute-force search.
- **[`search.py`](ivf_quant/search.py)**: Contains the high-performance Numba-jitted kernels for the search.
  - `popcount_u64`: An efficient, bit-twiddling algorithm to count the number of set bits in a 64-bit integer.
  - `binary_search_kernel`: The core search function that computes Hamming distances and uses a heap to find the top-k results in parallel over multiple queries.
- **[`heap.py`](ivf_quant/heap.py)**: A Numba-jitted implementation of a Max-Heap, used for efficient top-k selection during the search.
- **[`utils.py`](ivf_quant/utils.py)**: Contains helper functions for clustering, quantization, and filtering.