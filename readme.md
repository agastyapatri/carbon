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







