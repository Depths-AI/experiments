# Compression is all you need: 2x, 4x, 8x and 16x cheaper vector database

While building Depths AI, I have been scratching my head on how we can make vector databases at least 10x cheaper. Just like you, I also hate paying bank breaking amounts for storing vector embeddings

In this article, I cover 4 possible ways to reduce your vector database storage bills 2x, 4x, 8x and up to 16x lower:

1. **2x:** Just store the embeddings as Float16. Simplest trick, so lame that it is surprising that it is not the default
2. **4x:** Slim down even further, approximate the embedding as an array of 8-bit integers: 4x lesser bits per vector embedding.
3. **8x**: Don’t just stop at Float16, make your vector embedding 4x smaller: so, for example, have 384 instead of 1536 dimensions.
4. **16x**: What if we quantize our embeddings down to int8 and also make them 4x smaller?

## *What* do we compress?

Current **dense** vector embeddings possibly use too many bytes to convey the information? Can we try to squeeze in the same semantic information in lesser number of bytes? We have 2 ways to do this:

1. Either make the vector smaller: cut down on the number of dimensions
2. Keep the dimensions intact, but approximate with a lower precision number to represent each dimension: int8 or float16 instead of float32: using 8 and 16 bits per number respectively, instead of 32.

And of course, merge the two techniques.

Larger the vector embedding, more possible scope to cut down on less relevant info

## Shortening the vector: Principal Component Analysis

OG ML gang knows the power of PCA: an algorithm that basically:

1. Takes in a batch of vector and finds the “components” that define the information stored in the vectors (eigenvalues to be precise)
2. Then, we simply take, say top 25%, top 50% of those components, sorted by how much of info (variance to be precise) do they contribute to, within the vector. It turns out, for most vector embeddings, ~90% information gets captured within just top 50% components

Obviously, a smaller vector embedding would have a lesser room for irrelevance, so compressions would show better impact on larger embeddings.

<aside>
💡

Note that the only drawback of PCA is that you would have to separately store the “matrix” that is used for “compressing” the document vectors, and the mean of all your document vectors as well. These two metrics would then be needed to first “compress” incoming query to same size as document vectors, and then perform search.

</aside>

## Show me the numbers homeboy

I chose two different embedding sizes, to drive the point home.

Tested on Cohere vector embeddings (768 dimensional originally): https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings

and OpenAI embeddings (1536 dimensional originally): https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M

### Testing setup

1. 10k rows from each dataset as the reference documents for relevance search
2. Additional 1k rows from each dataset as our queries
3. Recall numbers averaged over 1000 queries, for top 1, top 10 and top 100 docs. 
4. Cosine similarity as the metric for relevance.

<aside>
💡

Here, the recall is defined relative to brute force float32 search: If brute force search for a query gives us certain top 10 relevant docs, and our optimized version’s top 10 list has 9 of brute force’s top 10, we say we have a relative recall of 0.9: only 1 top doc was missed. Similarly, a recall of 0.8 would mean, on average we missed 20% of top 10 docs. So, **higher is better.**

</aside>

### Recall @ 1: How often do find the top document?

| **Relative Recall @ 1** | **Cohere (768 dimensions originally)** | **OpenAI (1536 dimensions originally)** | **Storage cost reduction** |
| --- | --- | --- | --- |
| Float16  | **0.998** | **1.0** | **2x**  |
| Int8 | 0.192 | **0.973** | **4x**  |
| 4x shorter Float16 | 0.6 | **0.782** | **8x**  |
| 4x shorter Int8 | **0.585** | **0.772** | **16x** |

### Recall @ 10:  What proportion of the top 10 docs do we find?

| **Relative Recall @ 10** | **Cohere (768 dimensions originally)** | **OpenAI (1536 dimensions originally)** | **Storage cost reduction** |
| --- | --- | --- | --- |
| Float16  | **0.998** | **1.0** | **2x**  |
| Int8 | 0.278 | **0.956** | **4x**  |
| 4x shorter Float16 | 0.711 | **0.828** | **8x**  |
| 4x shorter Int8 | **0.703** | **0.822** | **16x** |

<aside>
💡

**Note how 4x length reduction via PCA before quantization lead to dramatic recall jump on int8 quantized version. This is possibly due to lesser noise in quantization of “most relevant components” of the vector.**

</aside>

### Recall @ 100: What proportion of the top 100 docs do we find?

| **Relative Recall @ 10** | **Cohere (768 dimensions originally)** | **OpenAI (1536 dimensions originally)** | **Storage cost reduction** |
| --- | --- | --- | --- |
| Float16  | **0.998** | **1.0** | **2x**  |
| Int8 | 0.338 | **0.967** | **4x**  |
| 4x shorter Float16 | 0.767 | **0.828** | **8x**  |
| 4x shorter Int8 | **0.760** | **0.839** | **16x** |

So, it turns out, you just lose on 2 out 10 relevant docs even if you cut down your embeddings by 16x

<aside>
💡

The cosine similarity search and the PCA still happen after converting float16 back to float32 due to overflow concerns during linear algebra in NumPy, **the emphasis is on** **storage compression, you do not lose info on storing float16 instead of float32 in semantic search context**

</aside>
