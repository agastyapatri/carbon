```text
                   ___.                  
  ____ _____ ______\_ |__   ____   ____  
_/ ___\\__  \\_  __ \ __ \ /  _ \ /    \ 
\  \___ / __ \|  | \/ \_\ (  <_> )   |  \
 \___  >____  /__|  |___  /\____/|___|  /
     \/     \/          \/            \/ 
```

# carbon

Carbon is a C++20 tensor library, intended as the foundation for building neural networks. It is a from-scratch rewrite of the earlier [benzene](https://github.com/agastyapatri/benzene) project, adding two things: **templating** over element type, and (planned) **automatic differentiation**.

> **Status: early development (v0.0.1).** The templated tensor, BLAS dispatch layer, and a first set of ops exist. Autograd is designed (see [Roadmap](#roadmap)) but not yet implemented.

## What exists today

### `carbon::tensor<T>` (`src/carbon.hpp`, `src/carbon.cpp`)

A dense, row-major, N-dimensional array backed by `std::vector<T>`, with shape and stride bookkeeping. It is explicitly instantiated for `f32`, `f64`, `i32` and `i64`.

Type aliases: `u32 u64 u08 i32 i64 f32 f64`.

| Area | API |
|---|---|
| Construction | `tensor<T>(shape)` (zero-initialised), `fill(v)` |
| Factories | `zeros`, `ones`, `eye`, `linspace`, `arange` |
| Random | `randn`, `randu`, `rand_normal(shape, mean, std)`, He init (`randn_he`, `randu_he`), Xavier init (`randn_xavier`, `randu_xavier`), `manual_seed` |
| Accessors | `data()`, `shape()`, `strides()`, `numel()`, `ndim()`, `at({i, j, ...})`, `begin()/end()` |
| In-place math | `log_ exp_ sin_ cos_ tanh_ relu_ sigmoid_ square_ cube_ pow_(e)` |
| Arithmetic | `operator+`, `operator-` with NumPy-style broadcasting (via BLAS `axpy`); `==`, `!=` |
| Linear algebra | `dot`, `gemv` (matrix x vector), `gemm` (matrix x matrix) |
| I/O | `operator<<` pretty-prints tensors of any rank |

In-place elementwise ops use `std::transform` with `std::execution::par_unseq` once a tensor has at least 2^16 elements, and `unseq` below that.

### `carbon::blas<T>` (`src/blas.hpp`)

A traits struct that maps an element type to the right BLAS routines:

- `blas<f32>` calls `cblas_saxpy / sdot / sgemm / sgemv`
- `blas<f64>` calls `cblas_daxpy / ddot / dgemm / dgemv`
- the primary template `blas<T>` is a plain-loop fallback (used for `i32`, `i64`), so integer tensors work with the same API.

Each provides `axpy`, `dot`, `gemm` and `gemv`.

### Example (`main.cpp`)

```cpp
#include "carbon.hpp"
#include <iostream>

int main(){
    using T = carbon::f32;
    auto a = carbon::tensor<T>::randn({10, 10});
    auto b = carbon::tensor<T>::randn({10});
    auto c = carbon::gemv(a, b);   // 10x10 matrix times length-10 vector
    std::cout << c << std::endl;
}
```

## Building

Requirements: a C++20 compiler, CMake >= 4.0, and OpenBLAS (headers are included as `<openblas/cblas.h>`).

```sh
cmake -S . -B build
cmake --build build
./build/carbon
```

`compile_flags.txt` holds the clangd flags (`-std=c++20 -Wall -Wextra -Wpedantic -Isrc/`). The `Makefile` and the checked-in `main` binary are leftovers from an earlier layout and are stale (the Makefile references a `tensor.cpp` that no longer exists); use CMake.

## Project layout

```text
main.cpp          example / scratch driver
CMakeLists.txt    build (links OpenBLAS via find_package(BLAS))
src/carbon.hpp    tensor<T> class declaration, type aliases
src/carbon.cpp    tensor<T> implementation + explicit instantiations
src/blas.hpp      blas<T> traits (BLAS dispatch / generic fallback)
```

(`src/carbon_copy.*` are untracked-by-design backups and git-ignored.)

## Known gaps

- `matmul` is declared but not defined.
- Only `+` and `-` are implemented as operators; `*`, `/`, scalar ops and `pow` are commented out.
- The generic (non-BLAS) `blas<T>::gemm` indexes `B_at(k, k)` where it should be `B_at(k, j)`, so integer `gemm` results are wrong.
- `blas<f64>::gemv` and the generic `gemv` take `alpha`/`beta` as `f32` rather than `T`.
- `tensor` currently has value semantics; no autograd, no views/slicing, no reductions yet.
- No test suite.

## Roadmap

The plan is deep and structural: templating first, then a storage/graph split, then autograd on top.

1. **Templating (mostly done).** `tensor` -> `tensor<T>`, `blas<T>` traits routing to `cblas_*` for `f32`/`f64` with a plain-loop fallback, explicit instantiation for the supported types. Mixed-type ops will not be allowed implicitly; they will need an explicit `.cast<T>()`.
2. **Storage / graph split.** `tensor<T>` stays pure data (data, shape, strides, no grad awareness). A new `variable<T>` wraps `shared_ptr<tensor<T>> data`, `shared_ptr<tensor<T>> grad`, `requires_grad`, and `shared_ptr<node> grad_fn`, so many variables can share storage without deep copies.
3. **Autograd engine.** A `node` base class holding `weak_ptr`s to its input variables and a virtual `backward(const tensor<T>& grad_output)`. `variable<T>::backward()` does a post-order DFS topological sort and walks it in reverse. Every op needs a forward/backward pair, and broadcasting backward must sum gradients back down to the pre-broadcast shape.
4. **Utilities and validation.** `no_grad()` scope guard, `.detach()`, `zero_grad()`, leaf vs. non-leaf tracking, and gradient checking against finite differences.
