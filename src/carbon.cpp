#include "carbon.hpp"
#include "blas.hpp"
#include <algorithm>
#include <ostream>
#include <cassert>
#include <concepts>
#include <execution>
#include <iomanip>
#include <random>
#include <stdexcept>

namespace carbon {

namespace detail{
std::mt19937_64& rand_engine(){
	static std::mt19937_64 engine(std::random_device{}()); 
	return engine;
}
}

void manual_seed(u32 seed) {detail::rand_engine().seed(seed);}

template <typename T> tensor<T>::tensor(std::vector<i32> shape) {
	_shape = shape;
	_ndim = shape.size();
	_numel = 1;
	for (int i : shape)
	_numel *= i;
	compute_strides();
	_data.resize(_numel, T(0.0));
}

template <typename T> tensor<T>::tensor(std::vector<i32> shape, std::vector<T> data) {
	_shape = shape;
	_ndim = shape.size();
	_numel = 1;
	for (int i : shape)
	_numel *= i;
	compute_strides();
	_data = data;
}


template <typename T> bool tensor<T>::is_contiguous() const {
	i64 expected_stride = 1;
	for (i32 i = _ndim - 1; i >= 0; i--) {
		if (expected_stride != _strides[i])
			return false;
		expected_stride *= _shape[i];
	}
	return true;
}

template <typename T> template <typename F> void tensor<T>::apply_(F f) {
	if (_numel >= PARALLEL_THRESHOLD) {
		std::transform(std::execution::par_unseq, _data.begin(), _data.end(),
				 _data.begin(), f);
	} else {
		std::transform(std::execution::unseq, _data.begin(), _data.end(),
				 _data.begin(), f);
	}
}

template <typename T>
std::optional<std::vector<i32>>
tensor<T>::broadcast_shapes(const std::vector<i32> &shape1,
							const std::vector<i32> &shape2) const {
	//	the smaller tensor is stretched over the larger; out takes the shape of
	//the larger.
	i32 ndim1 = shape1.size();
	i32 ndim2 = shape2.size();
	i32 out_ndim = std::max(ndim1, ndim2);
	std::vector<i32> out_shape(out_ndim);
	for (i32 i = 0; i < out_ndim; i++) {
		i32 d1 = (i < out_ndim - ndim1) ? 1 : shape1[i - (out_ndim - ndim1)];
		i32 d2 = (i < out_ndim - ndim2) ? 1 : shape2[i - (out_ndim - ndim2)];
		if (d1 == d2) {
			out_shape[i] = d1;
		} else if (d1 == 1) {
			out_shape[i] = d2;
		} else if (d2 == 1) {
			out_shape[i] = d1;
		} else {
			return std::nullopt;
		}
	}
	return out_shape;
}

template <typename T>
std::vector<i32> tensor<T>::reduced_shape(i32 axis) const {
	std::vector<i32> outshape;
	for (i32 i = 0; i < this->_ndim; i++) {
		if (i != (i32)axis)
			outshape.push_back(this->_shape[i]);
	}
	return outshape;
}

template <typename T>
std::vector<i32> tensor<T>::flat_idx_to_coord(i64 idx) const {
	std::vector<i32> coords(this->_ndim, 0);
	for (i32 j = 0; j < this->_ndim; j++) {
		coords[j] = idx / this->_strides[j];
		idx = idx % this->_strides[j];
	}
	return coords;
}

template <typename T> i64 tensor<T>::offset(std::span<const i32> idxs) const {
	i64 offset = 0;
	for (i32 i = 0; i < _ndim; i++) {
		assert(idxs[i] >= 0 && idxs[i] < _shape[i]);
		offset += idxs[i] * _strides[i];
	}
	return offset;
}

template <typename T> T tensor<T>::at(std::initializer_list<i32> idxs) const {
	assert(i32(idxs.size()) == _ndim);
	i64 _offset = offset(idxs);
	return _data[_offset];
}

template <typename T> T &tensor<T>::at(std::initializer_list<i32> idxs) {
	assert((i32)idxs.size() == _ndim);
	i64 _offset = offset(idxs);
	return _data[_offset];
}

template <typename T> void tensor<T>::compute_strides() {
	_strides.assign(_ndim, 1);
	for (int i = _ndim - 2; i >= 0; i--) {
		_strides[i] = _strides[i + 1] * _shape[i + 1];
	}
}

template <typename T> void tensor<T>::fill(T value) {
	if (_numel >= PARALLEL_THRESHOLD) {
		std::fill(std::execution::par_unseq, _data.begin(), _data.end(), value);
	} else {
		std::fill(std::execution::unseq, _data.begin(), _data.end(), value);
	}
}

template <typename T> tensor<T> tensor<T>::zeros(std::vector<i32> shape) {
	tensor<T> out(shape);
	return out;
}

template <typename T> tensor<T> tensor<T>::ones(std::vector<i32> shape) {
	tensor<T> out(shape);
	out.fill(1);
	return out;
}

template <typename T>
tensor<T> tensor<T>::rand_normal(std::vector<i32> shape, T mean, T std)
requires std::floating_point<T>
{
	tensor<T> out(shape);
	std::normal_distribution<T> dist(mean, std);
	auto &gen = detail::rand_engine();
	std::generate(out._data.begin(), out._data.end(),
			   [&]() -> T { return dist(gen); });
	return out;
}

template <typename T>
tensor<T> tensor<T>::randn(std::vector<i32> shape)
requires std::floating_point<T>
{
	return tensor<T>::rand_normal(shape, 0, 1);
}

template<typename T>
tensor<T> tensor<T>::randint(std::vector<i32> shape, T low , T high)
requires std::integral<T>
{
	tensor<T> out(shape);
	std::uniform_int_distribution<T> dist(low, high);
	auto &gen = detail::rand_engine();
	std::generate(out._data.begin(), out._data.end(), [&]() -> T {return dist(gen);});
	return out;
}

template <typename T>
tensor<T> tensor<T>::randu(std::vector<i32> shape, T low, T high)
requires std::floating_point<T>
{
	tensor out(shape);
	std::uniform_real_distribution<T> dist(low, high);
	auto &gen = detail::rand_engine();
	std::generate(out._data.begin(), out._data.end(),
			   [&]() -> T { return dist(gen); });
	return out;
}

template <typename T>
tensor<T> tensor<T>::randn_he(std::vector<i32> shape, i32 fan_in)
requires std::floating_point<T>
{
	return tensor::rand_normal(shape, 0, std::sqrt(T(2) / fan_in));
}

template <typename T>
tensor<T> tensor<T>::randu_he(std::vector<i32> shape, i32 fan_in)
requires std::floating_point<T>
{
	return tensor::randu(shape, -std::sqrt(T(6) / fan_in),
					  std::sqrt(T(6) / fan_in));
}

template <typename T>
tensor<T> tensor<T>::randn_xavier(std::vector<i32> shape, u32 fan_in,
								  u32 fan_out)
requires std::floating_point<T>
{
	return tensor::rand_normal(shape, 0, std::sqrt(T(2) / (fan_in + fan_out)));
}

template <typename T>
tensor<T> tensor<T>::randu_xavier(std::vector<i32> shape, u32 fan_in,
								  u32 fan_out)
requires std::floating_point<T>
{
	return tensor::randu(shape, -std::sqrt(T(6) / (fan_in + fan_out)),
					  std::sqrt(T(6) / (fan_in + fan_out)));
}

template <typename T> tensor<T> tensor<T>::eye(const i32 size) {
	tensor out({size, size});
	for (i32 i = 0; i < size; i++) {
		out.at({i, i}) = 1;
	}
	return out;
}

template <typename T> tensor<T> tensor<T>::linspace(T start, T end, i32 num) {
	assert(num >= 0);
	tensor<T> out({num});
	if (num == 0)
		return out;
	if (num == 1) {
		out._data[0] = start;
		return out;
	}
	f64 res = (f64(end) - f64(start)) / f64((num - 1));
	for (i64 i = 0; i < out._numel; i++) {
		out._data[i] = static_cast<T>(f64(start) + i * res);
	}
	out._data[num - 1] = end;
	return out;
}

template <typename T> tensor<T> tensor<T>::arange(T start, T end, T step) {
	assert(step > 0);
	i64 num = static_cast<i64>(std::ceil((end - start) / f64(step)));
	if (num <= 0)
		return tensor<T>({0});
	tensor<T> out({static_cast<i32>(num)});
	for (i64 i = 0; i < out._numel; i++) {
		out._data[i] = static_cast<T>(f64(start) + i * step);
	}
	return out;
}

template <typename T>
void tensor<T>::log_()
requires std::floating_point<T>
{
	apply_([](T val) { return static_cast<T>(std::log(val)); });
}

template <typename T>
void tensor<T>::exp_()
requires std::floating_point<T>
{
	apply_([](T val) { return static_cast<T>(std::exp(val)); });
}

template <typename T>
void tensor<T>::sin_()
requires std::floating_point<T>
{
	apply_([](T val) { return static_cast<T>(std::sin(val)); });
}

template <typename T>
void tensor<T>::cos_()
requires std::floating_point<T>
{
	apply_([](T val) { return static_cast<T>(std::cos(val)); });
}

template <typename T>
void tensor<T>::tanh_()
requires std::floating_point<T>
{
	auto generator = [](T val) { return static_cast<T>(std::tanh(val)); };
	apply_(generator);
}

template <typename T> void tensor<T>::relu_() {
	auto generator = [](T val) { return (val > 0 ? val : 0); };
	apply_(generator);
}

template <typename T>
void tensor<T>::sigmoid_()
requires std::floating_point<T>
{
	auto generator = [](T val) { return (1 / (1 + std::exp(-val))); };
	apply_(generator);
}

template <typename T> void tensor<T>::square_() {
	auto generator = [](T val) { return val * val; };
	apply_(generator);
}
//
template <typename T> void tensor<T>::cube_() {
	auto generator = [](T val) { return val * val * val; };
	apply_(generator);
}

template <typename T>
void tensor<T>::pow_(const T exponent)
requires std::floating_point<T>
{
	auto generator = [exponent](T val) { return std::pow(val, exponent); };
	apply_(generator);
}

template <typename U>
static void print_tensor_recursive(std::ostream &os, const tensor<U> &t,
                                   i32 dim, i64 offset, int width) {
	os << "[";
	const i32 n = t.shape()[dim];
	const i64 stride = t.strides()[dim];
	if (dim == t.ndim() - 1) {
		for (i32 i = 0; i < n; i++) {
			os << std::setw(width) << t.data()[offset + i * stride];
			if (i < n - 1) os << ", ";
		}
	} else {
		for (i32 i = 0; i < n; i++) {
			print_tensor_recursive(os, t, dim + 1, offset + i * stride, width);
			if (i < n - 1) {
				os << ",";
				for (i32 k = 0; k < t.ndim() - 1 - dim; k++) os << "\n";
				for (i32 k = 0; k <= dim; k++) os << " ";
			}
		}
	}
	os << "]";
}

template <typename U>
std::ostream &operator<<(std::ostream &os, const tensor<U> &t) {
	const std::ios_base::fmtflags old_flags = os.flags();
	const std::streamsize old_prec = os.precision();
	if constexpr (std::floating_point<U>) {
		os << std::fixed << std::setprecision(7);
	}

	if (t.data().empty() && t.ndim() == 0) {          // default-constructed
		os << "[]";
	} else if (t.ndim() == 0) {                        // 0-d scalar
		os << "[" << t.data()[0] << "]";
	} else {
		int width = 0;                                 // widest formatted element
		std::ostringstream probe;
		probe.copyfmt(os);
		for (U v : t) {
			probe.str("");
			probe << v;
			width = std::max(width, static_cast<int>(probe.str().size()));
		}
		print_tensor_recursive(os, t, 0, 0, width);
	}

	os.flags(old_flags);
	os.precision(old_prec);
	return os;
}

template <typename T> bool tensor<T>::operator==(const tensor<T> &other) const {
	return ((_shape == other._shape) && (std::equal(begin(), end(), other.begin())));
}

template <typename T> bool tensor<T>::operator!=(const tensor<T> &other) const {
	return !(*(this) == other);
}

template <typename T>
tensor<T> tensor<T>::arithmetic(const tensor<T> &inp2, T op) const {
	auto outshape = broadcast_shapes(this->_shape, inp2._shape);
	if (!outshape) {
		throw std::runtime_error("Tensors are not broadcast compatible.");
	}
	tensor<T> out(outshape.value());

	//	(S) + (S)
	if (this->_shape == inp2._shape) {
		std::copy(this->_data.begin(), this->_data.end(), out._data.begin());
		blas<T>::axpy(out._numel, op, inp2._data.data(), 1, out._data.data(), 1);
		return out;
	}

	//	(..., N) + (N)
	if (inp2._ndim == 1 && this->_ndim >= 1 &&
		this->_shape.back() == inp2._shape[0]) {
		i32 N = inp2._shape[0];
		i64 total_rows = this->_numel / N;
		std::copy(this->_data.begin(), this->_data.end(), out._data.begin());
		for (i64 i = 0; i < total_rows; i++) {
			blas<T>::axpy(N, op, inp2._data.data(), 1, out._data.data() + i * N, 1);
		}
		return out;
	}

	//	(..., K, N) + (K, N)  -- inp2 must match this's trailing two dims
	//exactly
	if (this->_ndim > 2 && inp2._ndim == 2 &&
		inp2._shape[0] == this->_shape[this->_ndim - 2] &&
		inp2._shape[1] == this->_shape[this->_ndim - 1]) {
		i32 N = static_cast<i32>(inp2._numel);
		i64 num_iters = this->_numel / N;
		std::copy(this->_data.begin(), this->_data.end(), out._data.begin());
		for (i64 i = 0; i < num_iters; i++) {
			blas<T>::axpy(N, op, inp2._data.data(), 1, out._data.data() + i * N, 1);
		}
		return out;
	}

	//	scalar
	if (inp2._numel == 1) {
		T scalar = op * inp2._data[0];
		for (i64 i = 0; i < out._numel; i++)
			out._data[i] = this->_data[i] + scalar;
		return out;
	}

	//	general broadcast fallback
	const i32 nd = out._ndim;

	//	broadcast strides: real stride where the dim matches, 0 where it's 1 or
	//absent
	auto broadcast_strides = [nd](const tensor<T> &t) {
		std::vector<i64> s(nd, 0);
		i32 lead = nd - t._ndim;
		for (i32 i = 0; i < t._ndim; i++) {
			if (t._shape[i] != 1)
				s[lead + i] = t._strides[i];
		}
		return s;
	};
	std::vector<i64> sa = broadcast_strides(*this);
	std::vector<i64> sb = broadcast_strides(inp2);

	std::vector<i32> coord(nd, 0);
	i64 oa = 0, ob = 0;
	for (i64 i = 0; i < out._numel; i++) {
		out._data[i] = this->_data[oa] + op * inp2._data[ob];

		//	odometer increment, carrying from the last dim
		for (i32 d = nd - 1; d >= 0; d--) {
			coord[d]++;
			oa += sa[d];
			ob += sb[d];
			if (coord[d] < out._shape[d])
				break;
			oa -= sa[d] * out._shape[d];
			ob -= sb[d] * out._shape[d];
			coord[d] = 0;
		}
	}
	return out;
}

template <typename T>
tensor<T> tensor<T>::operator+(const tensor<T> &other) const {
	return this->arithmetic(other, T(1));
}

template <typename T>
tensor<T> tensor<T>::operator-(const tensor<T> &other) const {
	return this->arithmetic(other, T(-1));
}

template<typename T>
tensor<T> tensor<T>::operator*(const tensor& other) const{
	auto outshape = broadcast_shapes(this->_shape, other._shape);
	if (!outshape) {
		throw std::runtime_error("Tensors are not broadcast compatible.");
	}
	tensor<T> out(outshape.value());
	if(out._numel == 0)	return out;
	auto gen = [](T x, T y){return x * y;};

	//	(S) * (S)
	if (this->_shape == other._shape) {
		if (_numel >= PARALLEL_THRESHOLD) {
			std::transform(std::execution::par_unseq, _data.begin(), _data.end(), other._data.begin(), out._data.begin(), gen);
		} else {
			std::transform(std::execution::unseq, _data.begin(), _data.end(), other._data.begin(), out._data.begin(), gen);
		}

		return out;
	}

	//	(..., N) * (N)
	if (other._ndim == 1 && this->_ndim >= 1 &&
		this->_shape.back() == other._shape[0]) {
		i32 N = other._shape[0];
		i64 total_rows = this->_numel / N;
		for (i64 i = 0; i < total_rows; i++) {
			auto a_row = _data.begin() + i*N;
			std::transform(std::execution::unseq, a_row, a_row + N, other._data.begin(), out._data.begin() + i*N, gen);
		}
		return out;
	}

	//	(..., K, N) * (K, N)  -- other must match this's trailing two dims
	//exactly
	if (this->_ndim > 2 && other._ndim == 2 &&
		other._shape[0] == this->_shape[this->_ndim - 2] &&
		other._shape[1] == this->_shape[this->_ndim - 1]) {
		i32 N = static_cast<i32>(other._numel);
		i64 num_iters = this->_numel / N;
		for (i64 i = 0; i < num_iters; i++) {
			auto a_block = _data.begin() + i*N;
			std::transform(std::execution::unseq, a_block, a_block + N, other._data.begin(), out._data.begin() + i*N, gen);
		}
		return out;
	}

	//	scalar
	if (other._numel == 1) {
		T scalar = other._data[0];
		for (i64 i = 0; i < out._numel; i++)
			out._data[i] = this->_data[i] * scalar;
		return out;
	}

	//	general broadcast fallback
	const i32 nd = out._ndim;

	//	broadcast strides: real stride where the dim matches, 0 where it's 1 or
	//absent
	auto broadcast_strides = [nd](const tensor<T> &t) {
		std::vector<i64> s(nd, 0);
		i32 lead = nd - t._ndim;
		for (i32 i = 0; i < t._ndim; i++) {
			if (t._shape[i] != 1)
				s[lead + i] = t._strides[i];
		}
		return s;
	};
	std::vector<i64> sa = broadcast_strides(*this);
	std::vector<i64> sb = broadcast_strides(other);

	std::vector<i32> coord(nd, 0);
	i64 oa = 0, ob = 0;
	for (i64 i = 0; i < out._numel; i++) {
		out._data[i] = this->_data[oa] * other._data[ob];

		//	odometer increment, carrying from the last dim
		for (i32 d = nd - 1; d >= 0; d--) {
			coord[d]++;
			oa += sa[d];
			ob += sb[d];
			if (coord[d] < out._shape[d])
				break;
			oa -= sa[d] * out._shape[d];
			ob -= sb[d] * out._shape[d];
			coord[d] = 0;
		}
	}
	return out;
}

template<typename T>
tensor<T> tensor<T>::operator/(const tensor& other) const{
	auto outshape = broadcast_shapes(this->_shape, other._shape);
	if (!outshape) {
		throw std::runtime_error("Tensors are not broadcast compatible.");
	}
	tensor<T> out(outshape.value());
	if(out._numel == 0)	return out;

	auto gen = [](T x, T y){return x / y;};
	//	(S) * (S)
	if (this->_shape == other._shape) {
		if (_numel >= PARALLEL_THRESHOLD) {
			std::transform(std::execution::par_unseq, _data.begin(), _data.end(), other._data.begin(), out._data.begin(), gen);
		} else {
			std::transform(std::execution::unseq, _data.begin(), _data.end(), other._data.begin(), out._data.begin(), gen);
		}

		return out;
	}

	//	(..., N) * (N)
	if (other._ndim == 1 && this->_ndim >= 1 &&
		this->_shape.back() == other._shape[0]) {
		i32 N = other._shape[0];
		i64 total_rows = this->_numel / N;
		for (i64 i = 0; i < total_rows; i++) {
			auto a_row = _data.begin() + i*N;
			std::transform(std::execution::unseq, a_row, a_row + N, other._data.begin(), out._data.begin() + i*N, gen);
		}
		return out;
	}

	//	(..., K, N) * (K, N)  -- other must match this's trailing two dims
	//exactly
	if (this->_ndim > 2 && other._ndim == 2 &&
		other._shape[0] == this->_shape[this->_ndim - 2] &&
		other._shape[1] == this->_shape[this->_ndim - 1]) {
		i32 N = static_cast<i32>(other._numel);
		i64 num_iters = this->_numel / N;
		for (i64 i = 0; i < num_iters; i++) {
			auto a_block = _data.begin() + i*N;
			std::transform(std::execution::unseq, a_block, a_block + N, other._data.begin(), out._data.begin() + i*N, gen);
		}
		return out;
	}

	//	scalar
	if (other._numel == 1) {
		T scalar = other._data[0];
		for (i64 i = 0; i < out._numel; i++)
			out._data[i] = this->_data[i] / scalar;
		return out;
	}

	//	general broadcast fallback
	const i32 nd = out._ndim;

	//	broadcast strides: real stride where the dim matches, 0 where it's 1 or
	//absent
	auto broadcast_strides = [nd](const tensor<T> &t) {
		std::vector<i64> s(nd, 0);
		i32 lead = nd - t._ndim;
		for (i32 i = 0; i < t._ndim; i++) {
			if (t._shape[i] != 1)
				s[lead + i] = t._strides[i];
		}
		return s;
	};
	std::vector<i64> sa = broadcast_strides(*this);
	std::vector<i64> sb = broadcast_strides(other);

	std::vector<i32> coord(nd, 0);
	i64 oa = 0, ob = 0;
	for (i64 i = 0; i < out._numel; i++) {
		out._data[i] = this->_data[oa] / other._data[ob];

		//	odometer increment, carrying from the last dim
		for (i32 d = nd - 1; d >= 0; d--) {
			coord[d]++;
			oa += sa[d];
			ob += sb[d];
			if (coord[d] < out._shape[d])
				break;
			oa -= sa[d] * out._shape[d];
			ob -= sb[d] * out._shape[d];
			coord[d] = 0;
		}
	}
	return out;
}


template<typename T> tensor<T> tensor<T>::operator+(T scalar) const{
	tensor<T> out(_shape);
	for (i64 i = 0; i < out._numel; i++)
		out._data[i] = this->_data[i] + scalar;
	return out;
}
template<typename T> tensor<T> tensor<T>::operator-(T scalar) const{
	tensor<T> out(_shape);
	for (i64 i = 0; i < out._numel; i++)
		out._data[i] = this->_data[i] - scalar;
	return out;

}
template<typename T> tensor<T> tensor<T>::operator*(T scalar) const{
	tensor<T> out(_shape);
	for (i64 i = 0; i < out._numel; i++)
		out._data[i] = this->_data[i] * scalar;
	return out;

}
template<typename T> tensor<T> tensor<T>::operator/(T scalar) const{
	tensor<T> out(_shape);
	for (i64 i = 0; i < out._numel; i++)
		out._data[i] = this->_data[i] / scalar;
	return out;

}


template <typename U>
tensor<U> dot(const tensor<U> &inp1, const tensor<U> &inp2) {
	assert((inp1._ndim == 1) && (inp2._ndim == 1));
	assert(inp1._shape[inp1._ndim - 1] == inp2._shape[inp2._ndim - 1]);
	i32 N = inp1._shape[0];
	tensor<U> out({1});
	U res = blas<U>::dot(N, inp1._data.data(), 1, inp2._data.data(), 1);
	out._data[0] = res;
	return out;
}

template <typename U>
tensor<U> gemm(const tensor<U> &inp1, const tensor<U> &inp2) {
	assert((inp1._ndim == 2) && (inp2._ndim == 2));
	assert(inp1._shape[1] == inp2._shape[0]);
	i32 M = inp1._shape[0]; // inp1.rows
	i32 N = inp2._shape[1]; // inp2.col
	i32 K = inp1._shape[1]; // inp1.cols
	tensor<U> out({M, N});
	blas<U>::gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, U(1),
			   inp1._data.data(), K, inp2._data.data(), N, U(0),
			   out._data.data(), N);
	return out;
}

template <typename U>
tensor<U> gemv(const tensor<U> &inp1, const tensor<U> &inp2) {
	assert(inp1._ndim == 2);
	assert(inp2._ndim == 1);
	assert(inp1._shape[1] == inp2._shape[0]);
	i32 M = inp1._shape[0];
	i32 N = inp1._shape[1];
	tensor<U> out({M});
	blas<U>::gemv(CblasRowMajor, CblasNoTrans, M, N, U(1), inp1._data.data(), N,
			   inp2._data.data(), 1, U(0), out._data.data(), 1);

	return out;
}

template <typename U> tensor<U> 
sin(const tensor<U>& inp) 
{
	tensor<U> out(inp._shape);
	auto gen = [](U x){return static_cast<U>(std::sin(x));};
	if (out._numel >= PARALLEL_THRESHOLD) {
		std::transform(std::execution::par_unseq, inp._data.begin(), inp._data.end(),
				 out._data.begin(), gen);
	} else {
		std::transform(std::execution::unseq, inp._data.begin(), inp._data.end(),
				 out._data.begin(), gen);
	}
	return out;
}

template <typename U> tensor<U> cos(const tensor<U>& inp){
	tensor<U> out(inp._shape);
	auto gen = [](U x){return static_cast<U>(std::cos(x));};
	if (out._numel >= PARALLEL_THRESHOLD) {
		std::transform(std::execution::par_unseq, inp._data.begin(), inp._data.end(),
				 out._data.begin(), gen);
	} else {
		std::transform(std::execution::unseq, inp._data.begin(), inp._data.end(),
				 out._data.begin(), gen);
	}
	return out;
}

template <typename U> tensor<U> log(const tensor<U>& inp){
	tensor<U> out(inp._shape);
	auto gen = [](U x){return static_cast<U>(std::log(x));};
	if (out._numel >= PARALLEL_THRESHOLD) {
		std::transform(std::execution::par_unseq, inp._data.begin(), inp._data.end(),
				 out._data.begin(), gen);
	} else {
		std::transform(std::execution::unseq, inp._data.begin(), inp._data.end(),
				 out._data.begin(), gen);
	}
	return out;
}
//
template <typename U> tensor<U> tanh(const tensor<U>& inp){
	tensor<U> out(inp._shape);
	auto gen = [](U x){return static_cast<U>(std::tanh(x));};
	if (out._numel >= PARALLEL_THRESHOLD) {
		std::transform(std::execution::par_unseq, inp._data.begin(), inp._data.end(),
				 out._data.begin(), gen);
	} else {
		std::transform(std::execution::unseq, inp._data.begin(), inp._data.end(),
				 out._data.begin(), gen);
	}
	return out;
}
//
template <typename U> tensor<U> exp(const tensor<U>& inp){
	tensor<U> out(inp._shape);
	auto gen = [](U x){return static_cast<U>(std::exp(x));};
	if (out._numel >= PARALLEL_THRESHOLD) {
		std::transform(std::execution::par_unseq, inp._data.begin(), inp._data.end(),
				 out._data.begin(), gen);
	} else {
		std::transform(std::execution::unseq, inp._data.begin(), inp._data.end(),
				 out._data.begin(), gen);
	}
	return out;
}
//
template <typename U> tensor<U> sigmoid(const tensor<U>& inp){
	tensor<U> out(inp._shape);
	auto gen = [](U x){return static_cast<U>(1 / (1 + std::exp(-x)));};
	if (out._numel >= PARALLEL_THRESHOLD) {
		std::transform(std::execution::par_unseq, inp._data.begin(), inp._data.end(),
				 out._data.begin(), gen);
	} else {
		std::transform(std::execution::unseq, inp._data.begin(), inp._data.end(),
				 out._data.begin(), gen);
	}
	return out;
}
//
template <typename U> tensor<U> relu(const tensor<U>& inp){
	tensor<U> out(inp._shape);
	auto gen = [](U x){return static_cast<U>(x > 0 ? x : 0);};
	if (out._numel >= PARALLEL_THRESHOLD) {
		std::transform(std::execution::par_unseq, inp._data.begin(), inp._data.end(),
				 out._data.begin(), gen);
	} else {
		std::transform(std::execution::unseq, inp._data.begin(), inp._data.end(),
				 out._data.begin(), gen);
	}
	return out;
}
//
template <typename U> tensor<U> cube(const tensor<U>& inp){
	tensor<U> out(inp._shape);
	auto gen = [](U x){return x*x*x;};
	if (out._numel >= PARALLEL_THRESHOLD) {
		std::transform(std::execution::par_unseq, inp._data.begin(), inp._data.end(),
				 out._data.begin(), gen);
	} else {
		std::transform(std::execution::unseq, inp._data.begin(), inp._data.end(),
				 out._data.begin(), gen);
	}
	return out;
}
//
template <typename U> tensor<U> square(const tensor<U>& inp){
	tensor<U> out(inp._shape);
	auto gen = [](U x){return x*x;};
	if (out._numel >= PARALLEL_THRESHOLD) {
		std::transform(std::execution::par_unseq, inp._data.begin(), inp._data.end(),
				 out._data.begin(), gen);
	} else {
		std::transform(std::execution::unseq, inp._data.begin(), inp._data.end(),
				 out._data.begin(), gen);
	}
	return out;
}
//

//	explicit instantation, apparently
template class tensor<f32>;
template class tensor<f64>;
template class tensor<i32>;
template class tensor<i64>;
template std::ostream &operator<<(std::ostream &os, const tensor<f32> &t);
template std::ostream &operator<<(std::ostream &os, const tensor<f64> &t);
template std::ostream &operator<<(std::ostream &os, const tensor<i32> &t);
template std::ostream &operator<<(std::ostream &os, const tensor<i64> &t);


template tensor<f32> gemm(const tensor<f32> &inp1, const tensor<f32> &inp2);
template tensor<f64> gemm(const tensor<f64> &inp1, const tensor<f64> &inp2);
template tensor<i32> gemm(const tensor<i32> &inp1, const tensor<i32> &inp2);
template tensor<i64> gemm(const tensor<i64> &inp1, const tensor<i64> &inp2);

template tensor<f32> gemv(const tensor<f32> &inp1, const tensor<f32> &inp2);
template tensor<f64> gemv(const tensor<f64> &inp1, const tensor<f64> &inp2);
template tensor<i32> gemv(const tensor<i32> &inp1, const tensor<i32> &inp2);
template tensor<i64> gemv(const tensor<i64> &inp1, const tensor<i64> &inp2);


template tensor<f32> dot(const tensor<f32> &inp1, const tensor<f32> &inp2);
template tensor<f32> sin(const tensor<f32>& inp);
template tensor<f32> cos(const tensor<f32>& inp) ;
template tensor<f32> log(const tensor<f32>& inp) ;
template tensor<f32> tanh(const tensor<f32>& inp);
template tensor<f32> exp(const tensor<f32>& inp) ;
template tensor<f32> sigmoid(const tensor<f32>& inp);
template tensor<f32> relu(const tensor<f32>& inp);
template tensor<f32> cube(const tensor<f32>& inp);
template tensor<f32> square(const tensor<f32>& inp);
//
//
template tensor<f64> dot(const tensor<f64> &inp1, const tensor<f64> &inp2);
template tensor<f64> sin(const tensor<f64>& inp) ;
template tensor<f64> cos(const tensor<f64>& inp) ;
template tensor<f64> log(const tensor<f64>& inp) ;
template tensor<f64> tanh(const tensor<f64>& inp);
template tensor<f64> exp(const tensor<f64>& inp) ;
template tensor<f64> sigmoid(const tensor<f64>& inp); 
template tensor<f64> relu(const tensor<f64>& inp);
template tensor<f64> cube(const tensor<f64>& inp);
template tensor<f64> square(const tensor<f64>& inp);
//
template tensor<i32> dot(const tensor<i32> &inp1, const tensor<i32> &inp2);
template tensor<i32> cube(const tensor<i32>& inp);
template tensor<i32> square(const tensor<i32>& inp);
template tensor<i32> relu(const tensor<i32>& inp);

//
template tensor<i64> dot(const tensor<i64> &inp1, const tensor<i64> &inp2);
template tensor<i64> cube(const tensor<i64>& inp);
template tensor<i64> square(const tensor<i64>& inp);
template tensor<i64> relu(const tensor<i64>& inp);
} // namespace carbon
