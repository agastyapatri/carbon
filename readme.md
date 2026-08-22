```text
                   ___.                  
  ____ _____ ______\_ |__   ____   ____  
_/ ___\\__  \\_  __ \ __ \ /  _ \ /    \ 
\  \___ / __ \|  | \/ \_\ (  <_> )   |  \
 \___  >____  /__|  |___  /\____/|___|  /
     \/     \/          \/            \/ 
```


Carbon is a C++20 tensor library meant to be the base on which neural nets are built. Carbon is built upon BLAS and provides automatic differentiation out of the box.

Carbon builds upon earlier work done in [https://github.com/agastyapatri/benzene](benzene). This project adds two important features: templating + autodiff. 



##  Roadmap 
The changes I want to make are deep structural ones. The way i'm going to tackle both are: implement templating first, storage / graph split, autograd on top of that split. 


###  1. **Templating:** `carbon::tensor` -> `carbon::tensor<T>`
Things to take a note of: `cblas` is used whenever possible, but `cblas` only provides functions for single- and double-precision floating points; `cblas_sgemm` and `cblas_dgemm` being two. In the current structure of the code, internal helpers call `cblas_saxpy` directly. Order of business: 

1.  Make `tensor` a class template. `std::vector<float> _data` -> `std::vector<T> _data`
1.  Introduce a small `blas_traits<T>` that routes operations to `cblas_saxpy / cblas_daxpy / cblas_sgemm / cblas_dgemm` for `f32` and `f64`. In all other cases, the function must fall back to a plain loop. 
3.  Move implementations to the header file, the implementation of a templated class must be visible at instantiation. Since the implementation is going to be many lines, it would also make sense for `f32 / f64` to have explicit instantiation in one .cpp file. 
4.  All functions of the form `friend tensor function_name(type arg)` need to be changed to `template<typename T> friend tensor<T> function_name(type args)`
5.  Mixed type ops are not going to be allowed: `(carbon::tensor<f32> + carbon::tensor<f64>)` type ops will requie explicit casting with `.cast<T>`. 


###  2. Splitting storage form the graph node. 
`carbon::tensor<T>` stays as is, pure data, shape, strides, no grad awareness. This just handles the data and becomes the leaf storage type.

1.  Create a wrapper `carbon::variable<T>` around:
-   `std::shared_ptr<carbon::tensor<T> data` 
-   `std::shared_ptr<carbon::tensor<T> grad` 
-   `bool requires_grad`
-   `std::shared_ptr<Node> grad_fn`

2.  Switching from value semantics everywhere to shared storage. Autograd needs many `Variable`s to be able to reference the same underlying data without triggering deep copies. `std::shared_ptr<tensor<T>>` achieves this cheaplywithout rewriting the whole allocation model.

###  3. The autograd engine itself.
1.  `node` base class: holds `weak_ptrs` to input `variable`s and a `backward(const tensor<T>& grad_output)` virtual method that computes and accumulates gradients into each inputs `.grad` 
2.  `variable<T>::backward()`: topological sorting from the output node (post order DFS), then walk it in reverse, calling each nodes' `backward()`
3. **Every op in the library needs a forward/backward pair.**
4.  Broadcasting needs to be handled specially. Backward through a broadcast requires summing the gradient back down to the original, pre-broadcast shape. 


###  4. Utilities + Validation
1.  `no_grad()` scope guard, `.detach()`, `zero_grad()`, leaf  vs non-leaf tracking. 
2.  Validating gradient comparisons by comparing them with finite differences.



##  Notes 
Inherited from an earlier design, `tensor::arithmetic(const tensor& inp2, f32 op)` is a private helper function used in `tensor::operator+` and `tensor::operator-`. `arithmetic` uses `cblas_saxpy` to perform elementwise addition / subtraction on two `tensors`. Changes need to be made to this function and the way it is structured in order to handle both single- and double-precision as well as integer tensors. 

There are two real ways to do this: 
1.  Conditional branching based on the type of the tensors: 
    ```cpp
    if constexpr(std::is_same_v<T, float>){
        //cblas_saxpy
    }else if constexpr(std::is_same_v<T, double>){
        //cblas_daxpy
    }else{
        //plain loop
    }
    ```
2. A small traits type `blas<T>` with `blas<T>::axpy()`. `blas<T>::gemm()`, `blas<T>::dot()`, etc.
This traits struct isn't a regular struct that can be called and have methods. Its a template whose job is purely to map "a type" to a "set of associated behaviour". 


```cpp 

template<typename T>
struct blas {
    static void axpy(int n, T alpha,
                     const T* x, int incx,
                     T* y, int incy)
    {
        for (int i = 0; i < n; ++i) {
            y[i * incy] += alpha * x[i * incx];
        }
    }

    static T dot(int n,
                 const T* x, int incx,
                 const T* y, int incy)
    {
        T result{};

        for (int i = 0; i < n; ++i) {
            result += x[i * incx] * y[i * incy];
        }

        return result;
    }
};

template<>
struct blas<float> {
    static void axpy(int n, float alpha,
                     const float* x, int incx,
                     float* y, int incy)
    {
        cblas_saxpy(n, alpha, x, incx, y, incy);
    }

    static float dot(int n,
                     const float* x, int incx,
                     const float* y, int incy)
    {
        return cblas_sdot(n, x, incx, y, incy);
    }
};


template<>
struct blas<double> {
    static void axpy(int n, double alpha,
                     const double* x, int incx,
                     double* y, int incy)
    {
        cblas_daxpy(n, alpha, x, incx, y, incy);
    }

    static double dot(int n,
                      const double* x, int incx,
                      const double* y, int incy)
    {
        return cblas_ddot(n, x, incx, y, incy);
    }
};



```


