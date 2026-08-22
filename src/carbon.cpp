#include "carbon.hpp"
#include "blas.hpp"
#include <algorithm>
#include <iomanip>
#include <ranges>
#include <execution>
#include <random>
#include <assert.h>

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
std::optional<std::vector<i32>> tensor<T>::broadcast_shapes(const std::vector<i32>& shape1, const std::vector<i32>& shape2) const{
	//	the smaller tensor is stretched over the larger; out takes the shape of the larger. 
	i32 ndim1 = shape1.size(); 
	i32 ndim2 = shape2.size();
	i32 out_ndim = std::max(ndim1, ndim2);
	std::vector<i32> out_shape(out_ndim);
	for(i32 i = 0; i < out_ndim; i++){
		i32 d1 = (i < out_ndim - ndim1) ? 1 : shape1[i - (out_ndim - ndim1)];
		i32 d2 = (i < out_ndim - ndim2) ? 1 : shape2[i - (out_ndim - ndim2)];
		if(d1 == d2){
			out_shape[i] = d1; 
		} else if(d1 == 1){
			out_shape[i] = d2;
		} else if(d2 == 1){
			out_shape[i] = d1;
		} else{
			return std::nullopt;
		}
	}
	return out_shape;
}

template<typename T>
std::vector<i32> tensor<T>::reduced_shape(i32 axis) const{
	std::vector<i32> outshape;
	for(i32 i = 0; i < this->_ndim; i++){
		if(i != (i32)axis)
			outshape.push_back(this->_shape[i]); 
	}
	return outshape;
}

template<typename T>
std::vector<i32> tensor<T>::flat_idx_to_coord(i64 idx) const {
	std::vector<i32> coords(this->_ndim, 0);
	for(i32 j = 0; j < this->_ndim; j++){
		coords[j] = idx / this->_strides[j];
		idx  = idx % this->_strides[j];
	}
	return coords;
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


template<typename T>
void tensor<T>::pow_(const T exponent){
	auto generator = [exponent](T val){return std::pow(val, exponent);};
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




template<typename U>
static inline void print_tensor_recursive(std::ostream& os, const tensor<U>& t, int dim, int offset, int depth){
	if(dim == t.ndim() - 1){
		os << "[";
		for(int i = 0; i < t.shape()[dim]; i++){
			os << std::setw(10) << std::fixed << std::setprecision(7) << t.data()[offset + i * t.strides()[dim]];
			if(i < t.shape()[dim] - 1) os << ", ";
		}
		os << "]";
	} else {
		os << "[";
		for(int i = 0; i < t.shape()[dim]; i++){
			print_tensor_recursive(os, t, dim+1, offset + i*t.strides()[dim], depth + 1);
		if(i < t.shape()[dim] - 1){
				os << ",";
				int newlines = t.ndim() - 1 - dim;
				for(int n = 0; n < newlines; n++) os << "\n";
				for(int d = 0; d < depth + 1; d++) os << " ";
			}
		}
		os << "]";
	}
}
template<typename U> 
std::ostream& operator<<(std::ostream& os, const tensor<U>& t){
	print_tensor_recursive(os, t, 0, 0, 0);
	os << "\n";
	return os;
}

template<typename T>
bool tensor<T>::operator==(const tensor<T>& other) const{
	if(_shape != other._shape) return false; 
	for(i64 i = 0; i < _numel; i++){
		if(this->_data[i] != other._data[i])
			return false;
	}
	return true; 

}

template<typename T> 
bool tensor<T>::operator!=(const tensor<T>& other) const{
	return !(*(this) == other);
}



template<typename T>
tensor<T> tensor<T>::arithmetic(const tensor<T>& inp2, f32 op) const {
	// assert(this->_shape == other._shape);
	auto outshape = broadcast_shapes(this->_shape, inp2._shape);
	if(!outshape){
		throw std::runtime_error("Tensors are not broadcast compatible.");
	}
	tensor<T> out(outshape.value());

	//	if both shapes are equal, simply add elementwise.
	if(this->_shape == inp2._shape){
		std::copy(this->_data.begin(), this->_data.end(), out._data.begin());
		blas<T>::axpy(out._numel, op, inp2._data.data(), 1, out._data.data(), 1);
		return out;
	}

	// Generalized (..., N) + (N) case
	if (inp2._ndim == 1 && this->_shape.back() == inp2._shape[0]) {
		i32 N = inp2._shape[0];
		i32 total_rows = this->_numel / N; 
		std::copy(this->_data.begin(), this->_data.end(), out._data.begin());
		for (i32 i = 0; i < total_rows; i++) {
			blas<T>::axpy(N, op, inp2._data.data(), 1, out._data.data() + (i*N), 1);
		}
		return out;
	}

	//	(....,  K, N) + (K, N)
	if(this->_ndim > 2 && inp2._ndim == 2){
		i32 N = inp2._shape[0]*inp2._shape[1];
		i32 num_iters = std::accumulate(this->_shape.begin(), this->_shape.end() - 2, 1, std::multiplies<i32>()); 
		std::copy(this->_data.begin(), this->_data.end(), out._data.begin());
		for(i32 i = 0; i < num_iters; i++){
			blas<T>::axpy(N, op, inp2._data.data(), 1, out._data.data() + (i*N), 1);

		}
		return out;
	}

	if(inp2._numel == 1){
		float scalar = op * inp2._data[0];
		for(i64 i = 0; i < _numel; i++)
			out._data[i] = this->_data[i] + scalar;
		return out;
	}

	throw std::runtime_error("Broacasting pattern not currently handled by carbon");

}


template<typename T>
tensor<T> tensor<T>::operator+(const tensor<T>& other) const{
	return this->arithmetic(other, 1.0f);
}

template<typename T>
tensor<T> tensor<T>::operator-(const tensor<T>& other) const{
	return this->arithmetic(other, -1.0f);
}


template<typename U>
tensor<U> dot(const tensor<U>& inp1, const tensor<U>& inp2){
	assert(inp1._shape[0] == inp2._shape[0]);
	i32 N = inp1._shape[0];
	tensor<U> out({1});
	U res = blas<U>::dot(N, inp1._data.data(), 1, inp2._data.data(), 1);
	out._data[0] = res;
	return out;

}





































//	explicit instantation, apparently
template class tensor<f32>;
template class tensor<f64>;
template class tensor<i32>;
template class tensor<i64>;
template std::ostream& operator<<(std::ostream& os, const tensor<f32>& t);
template std::ostream& operator<<(std::ostream& os, const tensor<f64>& t);
template std::ostream& operator<<(std::ostream& os, const tensor<i32>& t);
template std::ostream& operator<<(std::ostream& os, const tensor<i64>& t);
}








