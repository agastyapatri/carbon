```text
                   ___.                  
  ____ _____ ______\_ |__   ____   ____  
_/ ___\\__  \\_  __ \ __ \ /  _ \ /    \ 
\  \___ / __ \|  | \/ \_\ (  <_> )   |  \
 \___  >____  /__|  |___  /\____/|___|  /
     \/     \/          \/            \/ 
```
carbon is a tiny C++20 tensor library





### TODOs 
GPT2 forward pass workflow is: 
-   Embedding lookup `wte[tokens] + wpe[tokens]` 
-   12 blocks of `LayerNorm -> attention -> residual -> LayerNorm -> MLP with GELU -> residual` 
-   Final `LayerNorm`
-   Compute Logits as `x @ wte_transpose`

Things that `carbon` needs in order to do this: 
1.  **Views and Shape Ops**: `reshape`, `permuete / transpose`, `narrow / slice`, `contiguous()`, `clone()`. Attention is mostly shape manipulation
2.  **Axis reductions with keepdim:** `sum, mean, max` using `reduced_shape()` and `flat_idx_to_coord()`. Numerically stable softmax and layernorm were headed there. 
3.  **NN ops**: `softmax()`, `layernorm()`, `gelu()`, causal mask, out of place versions of unary ops. 
4.  **Matmul generalization**: `(...., M, K) x (...., K, N)`. Expose transpose flags, add batched matmul as a loop of `gemm` over the leading dimensions and add `linear(x, W, b)`. 
5.  **Indexing:** `embedding`/ gather by row and `cat` along a dimension. 
6.  **I/O and Assembly:**
    -   Safetensors loader 
    -   Tokenization: python??
    -   Write out the forward pass and validate against HF `transformers`
7.  **Performance:** KV cache for incremental loading and fused kernels for threading. 
