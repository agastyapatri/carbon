#include "carbon.hpp"
#include <cstdlib>
#include <iostream> 
#include <numbers>
#define type carbon::f64
#define LEN 1000
using tensor = carbon::tensor<type>;


int main(){
	// tensor::manual_seed(0);
	tensor a = tensor::randn({100, 100, 100});
	tensor b = tensor::randn({100, 100, 100});
	std::cout << (a == b) << std::endl;

}

