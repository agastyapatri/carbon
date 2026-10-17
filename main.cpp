#include "carbon.hpp"
#include <cstdlib>
#include <iostream> 
#include <numbers>
#define type carbon::f64
#define LEN 1000
using tensor = carbon::tensor<type>;


int main(){
	tensor::manual_seed(0);
	tensor a = tensor::randn({1000, 1000});
	tensor b = tensor::randn({1000, 1000});
	std::cout << (carbon::gemm(a, b)).shape()[0] << ", " <<(carbon::gemm(a, b)).shape()[1] << std::endl;
	return 0;
}

