# Micro experiments at Depths AI

## PCA and Integer Quantization of Vector Embeddings:

Lowering the vector database storage costs by up to 16x by using PCA and integer quantization.

Results:
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


### Recall @ 100: What proportion of the top 100 docs do we find?

| **Relative Recall @ 10** | **Cohere (768 dimensions originally)** | **OpenAI (1536 dimensions originally)** | **Storage cost reduction** |
| --- | --- | --- | --- |
| Float16  | **0.998** | **1.0** | **2x**  |
| Int8 | 0.338 | **0.967** | **4x**  |
| 4x shorter Float16 | 0.767 | **0.828** | **8x**  |
| 4x shorter Int8 | **0.760** | **0.839** | **16x** |

So, it turns out, you just lose on 2 out 10 relevant docs even if you cut down your embeddings by 16x