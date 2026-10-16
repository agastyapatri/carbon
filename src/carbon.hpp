#ifndef CARBON_TENSOR_HPP
#define CARBON_TENSOR_HPP

// #include "blas.hpp"
#include <cstdint>
#include <optional>
#include <random>
#include <vector>
#include <concepts>
#include <random>

namespace carbon{

typedef uint32_t u32; 
typedef uint64_t u64; 
typedef uint8_t  u08; 
typedef int32_t  i32; 
typedef int64_t  i64; 
typedef float  	 f32; 
typedef double   f64; 
static constexpr int PARALLEL_THRESHOLD = 65536; //2^16

template<typename U> 
class tensor;

template<typename U> 
tensor<U> dot(const tensor<U>& inp1, const tensor<U>& inp2);

template<typename U>
tensor<U> matmul(const tensor<U>& inp1, const tensor<U>& inp2);

template<typename U>
tensor<U> gemm(const tensor<U>& inp1, const tensor<U>& inp2);

template<typename U>
tensor<U> gemv(const tensor<U>& inp1, const tensor<U>& inp2);



template<typename T>
class tensor{
private: 
	std::vector<T> _data;
	std::vector<i32> _shape;
	std::vector<i64> _strides;
	i64  _numel = 0;
	i32  _ndim = 0;

	void compute_strides();
	static std::mt19937 rand_engine;
	i64 offset(std::vector<i32> idxs) const;

	bool is_contiguous() const;
	// tensor contiguous() const;

	std::vector<i32> flat_idx_to_coord(i64 idx) const;

	std::vector<i32> reduced_shape(i32 axis) const;

	std::optional<std::vector<i32>> broadcast_shapes(const std::vector<i32>& s1, const std::vector<i32>& s2) const;

	tensor<T> arithmetic(const tensor<T>& inp2, T op) const ; 

	// template<typename F> 
	// void apply_(F f){
	// 	if(_numel >= PARALLEL_THRESHOLD){
	// 		std::transform(std::execution::par_unseq, _data.begin(), _data.end(), _data.begin(), f);
	// 	}else{
	// 		std::transform(std::execution::unseq, _data.begin(), _data.end(), _data.begin(), f);
	// 	}
	// }


public:
	tensor() = default;
	tensor(std::vector<i32> shape);
	static void manual_seed(u32 seed){rand_engine.seed(seed);}

	const std::vector<T>& data()    const	{return _data;}
	const std::vector<i32>& shape()   const {return _shape;}
	const std::vector<i64>& strides() const {return _strides;}
	i64  numel()   const {return _numel; }
	i32  ndim()    const {return _ndim;}

	//	accessors
	T  at(std::vector<i32> idxs) const;
	T& at(std::vector<i32> idxs);
	auto begin() {return _data.begin();}
	auto end()   {return _data.end();}
	// tensor operator[](const std::vector<i32> idxs) const ;

	//	carbon tensor factory
	void   fill(T value);
	static tensor zeros(std::vector<i32> shape);
	static tensor ones(std::vector<i32> shape);
	static tensor rand_normal (std::vector<i32> shape, f64 mean, f64 std);
	static tensor randn(std::vector<i32> shape);
	static tensor randu(std::vector<i32> shape, f32 low = 0.0f, f32 high = 1.0f);
	static tensor randn_he(std::vector<i32> shape, i32 fan_in);
	static tensor randu_he(std::vector<i32> shape, i32 fan_in);
	static tensor randn_xavier (std::vector<i32> shape, u32 fan_in, u32 fan_out);
	static tensor randu_xavier (std::vector<i32> shape, u32 fan_in, u32 fan_out);
	static tensor eye(const i32 size);
	static tensor linspace(T start, T end, i32 num);
	static tensor arange(T start, T end, T step);

	//	inplace math
	void log_();
	void exp_();
	void sin_();
	void cos_();
	void tanh_();
	void relu_(); 
	void sigmoid_();
	void square_();
	void cube_();
	void pow_(const T exponent);
	
	template <typename U>
	friend std::ostream& operator<<(std::ostream& os, const tensor<U>& t);
	bool   operator==(const tensor<T>& other) const;
	bool   operator!=(const tensor<T>& other) const;
	tensor operator+(const tensor& other) const;
	tensor operator-(const tensor& other) const;
	// tensor operator*(const tensor& other) const;
	// tensor operator/(const tensor& other) const;
	// tensor operator*(f32 scalar) const;
	// tensor operator/(f32 scalar) const;
	// tensor operator+(f32 scalar) const;

	// tensor operator-(f32 scalar) const;
	// tensor pow(const f32 exponent) const; 
	template<typename U>
	friend tensor<U> dot(const tensor<U>& inp1, const tensor<U>& inp2);

	template<typename U>
	friend tensor<U> matmul(const tensor<U>& inp1, const tensor<U>& inp2);

	template<typename U>
	friend tensor<U> gemm(const tensor<U>& inp1, const tensor<U>& inp2);

	template<typename U>
	friend tensor<U> gemv(const tensor<U>& inp1, const tensor<U>& inp2);


};



} 





#endif

