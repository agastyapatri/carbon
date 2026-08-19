#include "carbon.hpp"
#include <algorithm>
#include <ranges>
#include <execution>
#include <random>
#include <assert.h>
static constexpr int PARALLEL_THRESHOLD = 65536; //2^16

namespace carbon{

template<typename T> 
std::mt19937 tensor<T>::rand_engine(std::random_device{}());


template<typename T>
tensor<T>::tensor(std::vector<i32> shape){
	_shape = shape;
	_ndim  = shape.size(); 
	_numel = 1; 
	for(int i : shape) _numel *= i;
	compute_strides();
	_data.resize(_numel, 0.0f);
}

template<typename T>
bool tensor<T>::is_contiguous() const{
	i32 expected_stride = 1;
	for(i32 i = _ndim-1; i >= 0; i--){
		if(expected_stride != _strides[i])	return false; 
		expected_stride *= _shape[i];
	}
	return true;
}

template<typename T>
i64 tensor<T>::offset(std::vector<i32> idxs) const{
	i64 offset = 0 ;
	for(i32 i = 0; i < _ndim; i++){
		assert(idxs[i] >= 0 && idxs[i] < _shape[i]);
		offset += idxs[i] * _strides[i];
	}
	return offset;
}


template<typename T>
T  tensor<T>::at(std::vector<i32> idxs) const{
	assert((i32)idxs.size() == _ndim);
	i64 _offset = offset(idxs);
	return _data[_offset];

}

template<typename T>
T& tensor<T>::at(std::vector<i32> idxs){
	assert((i32)idxs.size() == _ndim);
	i64 _offset = offset(idxs);
	return _data[_offset];
}

template <typename T>
void tensor<T>::compute_strides(){
	_strides.resize(_ndim, 1);
	for(int i = _ndim - 2; i >= 0; i--){
		_strides[i] = _strides[i + 1] * _shape[i + 1];
	}
}

template<typename T>
void tensor<T>::fill(T value){
	if(_numel >= PARALLEL_THRESHOLD){
		std::fill(std::execution::par_unseq, _data.begin(), _data.end(), value);
	}else{
		std::fill(std::execution::unseq, _data.begin(), _data.end(), value);

	}
}

template<typename T>
tensor<T> tensor<T>::zeros(std::vector<i32> shape){
	tensor<T> out(shape);
	return out;
}


template<typename T>
tensor<T> tensor<T>::ones(std::vector<i32> shape){
	tensor<T> out(shape);
	out.fill(1);
	return out;
}

template<typename T> 
tensor<T> tensor<T>::rand_normal (std::vector<i32> shape, f64 mean, f64 std){
	tensor<T> out(shape);
	std::normal_distribution<f32> dist(mean, std);
	std::generate(out._data.begin(), out._data.end(), [&]()->f32{
			return dist(rand_engine);
	});
	return out;
}

template<typename T> 
tensor<T> tensor<T>::randn(std::vector<i32> shape){
	return tensor<T>::rand_normal(shape, 0, 1);
}

template<typename T> 
tensor<T> tensor<T>::randu(std::vector<i32> shape, f32 low, f32 high){
	tensor out(shape);
	std::uniform_real_distribution<f32> dist(low, high);
	std::generate(out._data.begin(), out._data.end(), [&]()->f32{
			return dist(rand_engine);
	});
	return out;
}

template<typename T> 
tensor<T> tensor<T>::randn_he(std::vector<i32> shape, i32 fan_in){
	return tensor::rand_normal(shape, 0, std::sqrt(2.0f / fan_in));
}

template<typename T> 
tensor<T> tensor<T>::randu_he(std::vector<i32> shape, i32 fan_in){
	return tensor::randu(shape, -std::sqrt(2.0f / fan_in), std::sqrt(2.0f / fan_in));
}



template<typename T> 
tensor<T> tensor<T>::randn_xavier(std::vector<i32> shape, u32 fan_in, u32 fan_out){
	return tensor::rand_normal(shape, 0, std::sqrt(2.0f / (fan_in + fan_out)));
}

template<typename T> 
tensor<T> tensor<T>::randu_xavier(std::vector<i32> shape, u32 fan_in, u32 fan_out){
	return tensor::randu(shape, -std::sqrt(2.0f / (fan_in + fan_out)), std::sqrt(2.0f / (fan_in + fan_out)));
}




template<typename T> 
tensor<T> tensor<T>::eye(const i32 size){
	tensor out({size, size});
	for(i32 i = 0; i < size; i++){
		out.at({i, i}) = 1;
	}
	return out;
}



template<typename T> 
tensor<T> tensor<T>::linspace(f32 start, f32 end, i32 num){
	tensor<T> out({num});
	f32 res = (end - start) / (f32)num;
	auto indices = std::views::iota(0, num);
	auto generator = [start, res](i32 i){
		return static_cast<T>(start + i*res);
	};

	if(num >= PARALLEL_THRESHOLD){
		std::transform(
			std::execution::par_unseq,
			indices.begin(),
			indices.end(), 
			out._data.begin(),
			generator
		);
	}else{
		std::transform(
			std::execution::unseq,
			indices.begin(),
			indices.end(), 
			out._data.begin(),
			generator
		);

	}
	return out;
}

template<typename T> 
tensor<T> tensor<T>::arange(f32 start, f32 end, f32 step){
	i32 num = static_cast<i32>(std::ceil((end - start) / step));
	if(num <= 0)	return tensor<T>({0});
	tensor<T> out({num});
	auto indices = std::views::iota(0, num);
	auto generator = [start, step](i32 i){
		return static_cast<T>(start + i*step);
	};
	if(num >= PARALLEL_THRESHOLD){
		std::transform(
			std::execution::par_unseq, 
			indices.begin(), 
			indices.end(), 
			out._data.begin(), 
			generator
		);
	}else{
		std::transform(
			std::execution::unseq, 
			indices.begin(), 
			indices.end(), 
			out._data.begin(), 
			generator
		);
	}
	return out;

}


template<typename T>
void tensor<T>::log_(){
	if(_numel >= PARALLEL_THRESHOLD){
		std::transform(
			std::execution::par_unseq, 
			_data.begin(), 
			_data.end(), 
			_data.begin(),
			[](T val){return static_cast<T>(std::log(val));}
		); 
	}else{
		std::transform(
			std::execution::unseq, 
			_data.begin(), 
			_data.end(), 
			_data.begin(),
			[](T val){return static_cast<T>(std::log(val));}
		); 

	}

}

template<typename T>
void tensor<T>::exp_(){
	if(_numel >= PARALLEL_THRESHOLD){
		std::transform(
			std::execution::par_unseq, 
			_data.begin(), 
			_data.end(), 
			_data.begin(),
			[](T val){return static_cast<T>(std::exp(val));}
		); 
	}else{
		std::transform(
			std::execution::unseq, 
			_data.begin(), 
			_data.end(), 
			_data.begin(),
			[](T val){return static_cast<T>(std::exp(val));}
		); 

	}


}

template<typename T>
void tensor<T>::sin_(){
	if(_numel >= PARALLEL_THRESHOLD){
		std::transform(
			std::execution::par_unseq, 
			_data.begin(), 
			_data.end(), 
			_data.begin(),
			[](T val){return static_cast<T>(std::sin(val));}
		); 
	}else{
		std::transform(
			std::execution::unseq, 
			_data.begin(), 
			_data.end(), 
			_data.begin(),
			[](T val){return static_cast<T>(std::sin(val));}
		); 

	}
}

template<typename T>
void tensor<T>::cos_(){
	if(_numel >= PARALLEL_THRESHOLD){
		std::transform(
			std::execution::par_unseq, 
			_data.begin(), 
			_data.end(), 
			_data.begin(),
			[](T val){return static_cast<T>(std::cos(val));}
		); 
	}else{
		std::transform(
			std::execution::unseq, 
			_data.begin(), 
			_data.end(), 
			_data.begin(),
			[](T val){return static_cast<T>(std::cos(val));}
		); 

	}
}

template<typename T>
void tensor<T>::tanh_(){
	auto generator = [](T val){return static_cast<T>(std::tanh(val));};
	if(_numel >= PARALLEL_THRESHOLD){
		std::transform(
			std::execution::par_unseq, 
			_data.begin(), 
			_data.end(), 
			_data.begin(),
			generator
		); 
	}else{
		std::transform(
			std::execution::unseq, 
			_data.begin(), 
			_data.end(), 
			_data.begin(),
			generator
		); 

	}
}



template<typename T>
void tensor<T>::relu_(){
	auto generator = [](T val){return (val > 0 ? val : 0);};
	if(_numel >= PARALLEL_THRESHOLD){
		std::transform(
			std::execution::par_unseq, 
			_data.begin(), 
			_data.end(), 
			_data.begin(),
			generator
		); 
	}else{
		std::transform(
			std::execution::unseq, 
			_data.begin(), 
			_data.end(), 
			_data.begin(),
			generator
		); 

	}
}


template<typename T>
void tensor<T>::sigmoid_(){
	auto generator = [](T val){return (1 / (1 + std::exp(val)));};
	if(_numel >= PARALLEL_THRESHOLD){
		std::transform(
			std::execution::par_unseq, 
			_data.begin(), 
			_data.end(), 
			_data.begin(),
			generator
		); 
	}else{
		std::transform(
			std::execution::unseq, 
			_data.begin(), 
			_data.end(), 
			_data.begin(),
			generator
		); 

	}
}

template<typename T>
void tensor<T>::square_(){
	auto generator = [](T val){return val*val;};
	if(_numel >= PARALLEL_THRESHOLD){
		std::transform(
			std::execution::par_unseq, 
			_data.begin(), 
			_data.end(), 
			_data.begin(),
			generator
		); 
	}else{
		std::transform(
			std::execution::unseq, 
			_data.begin(), 
			_data.end(), 
			_data.begin(),
			generator
		); 

	}
}
//
template<typename T>
void tensor<T>::cube_(){
	auto generator = [](T val){return val*val*val;};
	if(_numel >= PARALLEL_THRESHOLD){
		std::transform(
			std::execution::par_unseq, 
			_data.begin(), 
			_data.end(), 
			_data.begin(),
			generator
		); 
	}else{
		std::transform(
			std::execution::unseq, 
			_data.begin(), 
			_data.end(), 
			_data.begin(),
			generator
		); 

	}
}






template class tensor<f32>;
template class tensor<f64>;
template class tensor<i32>;
template class tensor<i64>;
}








