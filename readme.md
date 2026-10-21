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



##  Building a testing and performance benchmarking harness for `carbon`
1. **Step 0: Restructuring the Build**: Make the library a single CMake target that `main`, the tests ahnd the benchmarks all link to. The concept is that everything must measure the same compiled code. Otherwise, a flag difference between two executables invalidates any comparisons. 
    *Milesone: It can be explained why `add_library` + `target_link_libraries` beats compiling `carbon.cpp` into each executable*


2.  **Step 2: Benchmarking with `std::chrono`, by hand, on purpose**: `gemm` should be timed manually and issues with the process should be unconvered. The compiler may delete work whose result is never used. The first run is slow. Small ops finish faster than the timer can resolve. Run-to-run variance is large. Only after these issues are hit must Google Benchmark be used. 
    *Milestone: `DoNotOptimize` can be explained* 

3.  **Step 3: Rates and Ceilings:** the CPU theoretical peak will be calculated: `cores x clock speed x SIMD x 2`. the memory bandwidth will also be measured with `memcpy`, then every op will be expressed as a percentage of the theoretical hardware maximum. This truly quantifies the speed of the operation being measured. 
    *Milestone: When it can be determined that `gemv` at 4096x4096 is limited by compute or by memory, and prove it empirically*

4.  **Noise and Statistics:** Identical code will be profiled twice and the per-run variance in the numbers will be measured. This is called an A/A test. Then repititions, coefficient of variation will be learned about and how to use they outperform eyeballing the difference in performance. 

5.  **Profiling:** `perf stat` answers "what kind of slow is slow": IPC, cache misses. `perf record` and `perf annotate` answer "which line is the slowdown occuring in?". `cachegrind` gives exact cache miss counts. The exact understanding will be learned. 
    *Milestone: The exact line in the code which causes most of the cache misses in the generic `blas<T>::gemm` can be pointed out and improved.* 

6.  **Step 6: the performance improvement loop:** Pick one bottleneck, form a hypothesis, make one change, measure it, explain the result. GEMM loop order is the first natural target. 


### Roadmap for building the test harness. 
**Steps 0 and 1** 
    1.  Change `CMakeLists.txt` so the library is built as a library target, `carbon_core`. The existing carbon executable should link to it.
    2.  Add GoogleTest using CMake's `FetchContent`, plus a a `carbon_tests` executable.
    3.  Write one test: `gemm` on small-nonsquare shapes (for example, a 3x5 times 5x7) compared against a triple loop reference implementation that is custom written. Do it for `f32` and `i32`













