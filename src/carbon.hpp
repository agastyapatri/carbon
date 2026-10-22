#ifndef CARBON_TENSOR_HPP
#define CARBON_TENSOR_HPP

// #include "blas.hpp"
#include <concepts>
#include <cstdint>
#include <optional>
#include <random>
#include <span>
#include <vector>


namespace carbon {
typedef uint32_t u32;
typedef uint64_t u64;
typedef uint8_t u08;
typedef int32_t i32;
typedef int64_t i64;
typedef float f32;
typedef double f64;
static constexpr int PARALLEL_THRESHOLD = 65536; // 2^16

namespace detail{
	std::mt19937_64& rand_engine();
}
void manual_seed(u32 seed); 


template <typename U> class tensor;

template <typename U>
tensor<U> dot(const tensor<U> &inp1, const tensor<U> &inp2);

template <typename U>
tensor<U> gemm(const tensor<U> &inp1, const tensor<U> &inp2);

template <typename U>
tensor<U> gemv(const tensor<U> &inp1, const tensor<U> &inp2);

template <typename U>
tensor<U> matmul(const tensor<U> &inp1, const tensor<U> &inp2);

template <typename U> tensor<U> sin(const tensor<U>& inp)  ;
template <typename U> tensor<U> cos(const tensor<U>& inp)  ;
template <typename U> tensor<U> log(const tensor<U>& inp)  ;
template <typename U> tensor<U> tanh(const tensor<U>& inp) ;
template <typename U> tensor<U> exp(const tensor<U>& inp)  ;
template <typename U> tensor<U> sigmoid(const tensor<U>& inp); 
template <typename U> tensor<U> relu(const tensor<U>& inp);
template <typename U> tensor<U> cube(const tensor<U>& inp);
template <typename U> tensor<U> square(const tensor<U>& inp);


template <typename T> class tensor {
private:
	std::vector<T> _data;
	std::vector<i32> _shape;
	std::vector<i64> _strides;
	i64 _numel = 0;
	i32 _ndim = 0;

	void compute_strides();
	i64 offset(std::span<const i32> idxs) const;

	bool is_contiguous() const;
	// tensor contiguous() const;

	std::vector<i32> flat_idx_to_coord(i64 idx) const;

	std::vector<i32> reduced_shape(i32 axis) const;

	std::optional<std::vector<i32>>
	broadcast_shapes(const std::vector<i32> &s1,
				  const std::vector<i32> &s2) const;

	tensor<T> arithmetic(const tensor<T> &inp2, T op) const;

	template <typename F> void apply_(F f);

public:
	tensor() = default;
	explicit tensor(std::vector<i32> shape);
	explicit tensor(std::vector<i32> shape, std::vector<T> data); 

	const std::vector<T> &data() const { return _data; }
	const std::vector<i32> &shape() const { return _shape; }
	const std::vector<i64> &strides() const { return _strides; }
	i64 numel() const { return _numel; }
	i32 ndim() const { return _ndim; }

	//	accessors
	T at(std::initializer_list<i32> idxs) const;
	T &at(std::initializer_list<i32> idxs);
	auto begin() const{return _data.cbegin();} 
	auto end() const {return _data.cend();}



	//	carbon tensor factory
	void fill(T value);
	static tensor zeros(std::vector<i32> shape);
	static tensor ones(std::vector<i32> shape);
	static tensor rand_normal(std::vector<i32> shape, T mean, T std)
	requires std::floating_point<T>;
	static tensor randn(std::vector<i32> shape)
	requires std::floating_point<T>;
	static tensor randu(std::vector<i32> shape, T low = T(0), T high = T(1))
	requires std::floating_point<T>;
	static tensor randint(std::vector<i32> shape, T low = T(0), T high = T(1))
	requires std::integral<T>;
	static tensor randn_he(std::vector<i32> shape, i32 fan_in)
	requires std::floating_point<T>;
	static tensor randu_he(std::vector<i32> shape, i32 fan_in)
	requires std::floating_point<T>;
	static tensor randn_xavier(std::vector<i32> shape, u32 fan_in, u32 fan_out)
	requires std::floating_point<T>;
	static tensor randu_xavier(std::vector<i32> shape, u32 fan_in, u32 fan_out)
	requires std::floating_point<T>;
	static tensor eye(const i32 size);
	static tensor linspace(T start, T end, i32 num);
	static tensor arange(T start, T end, T step);

	//	inplace math
	void log_()
	requires std::floating_point<T>;
	void exp_()
	requires std::floating_point<T>;
	void sigmoid_()
	requires std::floating_point<T>;
	void sin_()
	requires std::floating_point<T>;
	void cos_()
	requires std::floating_point<T>;
	void tanh_()
	requires std::floating_point<T>;
	void relu_();
	void square_();
	void cube_();
	void pow_(const T exponent)
	requires std::floating_point<T>;

	template <typename U>
	friend std::ostream &operator<<(std::ostream &os, const tensor<U> &t);

	//	TENSOR-TENSOR MATH
	bool operator==(const tensor<T> &other) const;
	bool operator!=(const tensor<T> &other) const;
	tensor operator+(const tensor &other) const;
	tensor operator-(const tensor &other) const;
	tensor operator*(const tensor& other) const;


	//	TODO: iron out all the edge cases
	tensor operator/(const tensor& other) const;

	// TENSOR - SCALAR MATH
	tensor operator*(T scalar) const;
	tensor operator/(T scalar) const;
	tensor operator+(T scalar) const;
	tensor operator-(T scalar) const;

	template <typename U>
	friend tensor<U> dot(const tensor<U> &inp1, const tensor<U> &inp2);

	template <typename U>
	friend tensor<U> gemm(const tensor<U> &inp1, const tensor<U> &inp2);

	template <typename U>
	friend tensor<U> gemv(const tensor<U> &inp1, const tensor<U> &inp2);

	template <typename U>
	friend tensor<U> matmul(const tensor<U> &inp1, const tensor<U> &inp2);

	template <typename U> friend tensor<U> sin(const tensor<U>& inp) ;
	template <typename U> friend tensor<U> cos(const tensor<U>& inp) ;
	template <typename U> friend tensor<U> log(const tensor<U>& inp) ;
	template <typename U> friend tensor<U> tanh(const tensor<U>& inp) ;
	template <typename U> friend tensor<U> exp(const tensor<U>& inp) ;
	template <typename U> friend tensor<U> sigmoid(const tensor<U>& inp) ;
	template <typename U> friend tensor<U> relu(const tensor<U>& inp);
	template <typename U> friend tensor<U> cube(const tensor<U>& inp);
	template <typename U> friend tensor<U> square(const tensor<U>& inp);


	//	TODO: views and shape ops; the function signatures are not solid yet.
	void reshape();
	void transpose();
	void slice();
	void contiguous();
	void clone();


	//	TODO: Axis reductions with keepdim
	void sum(); 
	void mean();
	void max(); 


	//	TODO: NN ops 
	void softmax(); 
	void layernorm();








};

typedef tensor<f32> tf32;
typedef tensor<f64> tf64;
typedef tensor<i32> ti32;
typedef tensor<i64> ti64;

} // namespace carbon
//


#endif

