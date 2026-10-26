#ifndef CARBON_BLAS_HPP
#define CARBON_BLAS_HPP
#ifdef CARBON_USE_BLAS
#include <cblas.h> 
#endif
#include "carbon.hpp"

namespace carbon{
#ifndef CARBON_USE_BLAS 
enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} ;
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} ;
enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} ;
enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132} ;
enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142} ;
using CBLAS_LAYOUT = CBLAS_ORDER;
#endif

static constexpr bool is_trans(CBLAS_TRANSPOSE t){
	return ((t == CblasTrans) || (t == CblasConjTrans));
}


template<typename T>
struct blas {
    static void axpy(i32 n, T alpha, const T* x, i32 incx,T* y, i32 incy){
        for (int i = 0; i < n; ++i) {
            y[i * incy] += alpha * x[i * incx];
        }
	}
    static T dot(i32 n, const T* x, i32 incx,const T* y, i32 incy){
		T res = 0;
		for(int i = 0; i < n; i++){
			res += x[i*incx] * y[i*incy];
		}
		return res;
	}
    static void gemm(
		const CBLAS_LAYOUT layout,
		const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
		const int M, const int N, const int K,
		const T alpha,
		const T *A, const int lda,
		const T *B, const int ldb,
		const T beta,
		T *C, const int ldc)
	{
		auto A_at = [&](i32 r, i32 c) -> T{
			bool row_major = (layout == CblasRowMajor);
			bool trans = is_trans(TransA);
			if(row_major != trans) return A[r * lda + c];
			else				   return A[c * lda + r];
		};
		auto B_at = [&](i32 r, i32 c) -> T{
			bool row_major = (layout == CblasRowMajor);
			bool trans = is_trans(TransB);
			if(row_major != trans) return B[r * ldb + c];
			else				   return B[c * ldb + r];
		};
		auto C_idx = [&](T _alpha, T _acc, T _beta, i32 _c_idx, T* _C) -> T{
			if(_beta == 0)	return _alpha*_acc; 
			if(_alpha == 0)	return _beta*_C[_c_idx];
			else{return _alpha*_acc + _beta*_C[_c_idx];}
		};
		for(i32 i = 0; i < M; i++){
			for(i32 j = 0; j < N; j++){
				T acc = 0; 
				for(i32 k = 0; k < K; k++){
					acc += A_at(i, k) * B_at(k, j);
				}
				i32 c_idx = (layout == CblasRowMajor) ? (i * ldc + j) : (j * ldc + i);
				C[c_idx] = C_idx(alpha, acc, beta, c_idx, C);
			} 
		}
	}

    static void gemv(
		const enum CBLAS_ORDER layout,
		const enum CBLAS_TRANSPOSE trans,
		const int M,
		const int N,
		const T alpha,
		const T *a,
		const int lda,
		const T *x,
		const int incx,
		const T beta,
		T *y,
		const int incy){

		auto A_at = [&](int r, int c) -> T {
			return (layout == CblasRowMajor) ? a[r * lda + c] : a[c * lda + r];
		};
		bool _is_trans = is_trans(trans);
		int out_len = _is_trans ? N : M;
		int sum_len = _is_trans ? M : N;
		if(alpha == 0){
			for(int i = 0; i < out_len; ++i){
				y[i * incy] = (beta == 0) ? T(0) : beta*y[i*incy];
			}
			return; 
		}
		for (int i = 0; i < out_len; ++i) {
			T acc = 0;
			for (int k = 0; k < sum_len; ++k) {
				T a_val = _is_trans ? A_at(k, i) : A_at(i, k);
				acc += a_val * x[k * incx];
			}
			if(beta == 0){
				y[i * incy] = alpha * acc;
				continue; 
			}
			y[i * incy] = beta*y[i*incy] + alpha*acc;
		}
	}
};


#ifdef CARBON_USE_BLAS
template<>
struct blas<f32>{
    static void axpy(i32 n, f32 alpha, const f32* x, i32 incx, f32* y, i32 incy){
		cblas_saxpy(n,alpha,x, incx, y, incy);
	}
    static f32 dot(i32 n, const f32* x, i32 incx,const f32* y, i32 incy){
		return cblas_sdot(n,x, incx, y, incy);
	}
    static void gemm(
		const CBLAS_LAYOUT layout,
		const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
		const int M, const int N, const int K,
		const float alpha,
		const float *A, const int lda,
		const float *B, const int ldb,
		const float beta,
		float *C, const int ldc){
		cblas_sgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
	}
    static void gemv(
		const enum CBLAS_ORDER order,
		const enum CBLAS_TRANSPOSE trans,
		const int M,
		const int N,
		const float alpha,
		const float *a,
		const int lda,
		const float *x,
		const int incx,
		const float beta,
		float *y,
		const int incy){
		cblas_sgemv(order, trans, M, N, alpha, a, lda, x, incx, beta, y, incy);
	}
};

template<>
struct blas<f64>{
    static void axpy(i32 n, f64 alpha, const f64* x, i32 incx, f64* y, i32 incy){
		cblas_daxpy(n,alpha,x, incx, y, incy);
	}
    static f64 dot(i32 n, const f64* x, i32 incx,const f64* y, i32 incy){
		return cblas_ddot(n,x, incx, y, incy);
	}
    static void gemm(
		const CBLAS_LAYOUT layout,
		const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
		const int M, const int N, const int K,
		const double alpha,
		const double *A, const int lda,
		const double *B, const int ldb,
		const double beta,
		double *C, const int ldc){
		cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

	}
    static void gemv(
		const enum CBLAS_ORDER order,
		const enum CBLAS_TRANSPOSE trans,
		const int M,
		const int N,
		const f64 alpha,
		const f64 *a,
		const int lda,
		const f64 *x,
		const int incx,
		const f64 beta,
		f64 *y,
		const int incy){
		cblas_dgemv(order, trans, M, N, alpha, a, lda, x, incx, beta, y, incy);

	}
};
#endif




}
#endif
