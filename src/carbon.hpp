#ifndef CARBON_TENSOR_HPP
#define CARBON_TENSOR_HPP

#include <cstdint>
#include <random>
#include <vector>
#include <random>

namespace carbon{

typedef uint32_t u32; 
typedef uint64_t u64; 
typedef uint8_t  u08; 
typedef int32_t  i32; 
typedef int64_t  i64; 
typedef float  	 f32; 
typedef double   f64; 


template<typename T>
class tensor{
private: 
	std::vector<T> _data;
	std::vector<i32> _shape;
	std::vector<i32> _strides;
	i64  _numel;
	i32  _ndim;

	void compute_strides();
	static std::mt19937 rand_engine;
	i64 offset(std::vector<i32> idxs) const;

	bool is_contiguous() const;
	// tensor contiguous() const;


public:
	tensor() = default; 
	tensor(std::vector<i32> shape);
	static void manual_seed(u32 seed){rand_engine.seed(seed);}

	const std::vector<T>& data()    const	{return _data;}
	const std::vector<i32>& shape()   const {return _shape;}
	const std::vector<i32>& strides() const {return _strides;}
	i64  numel()   const {return _numel; }
	i32  ndim()    const {return _ndim;}

	T  at(std::vector<i32> idxs) const;
	T& at(std::vector<i32> idxs);

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
	static tensor linspace(f32 start, f32 end, i32 num);
	static tensor arange(f32 start, f32 end, f32 step);

	//	transcendentals; inplace
	void log_();
	void exp_();
	void sin_();
	void cos_();
	void tanh_();
	void relu_(); 
	void sigmoid_();
	void square_();
	void cube_();

};



} 





#endif

