#include <cstdlib>
#include "carbon.hpp"
#include <iostream> 
#include <numbers>
#include <chrono> 
#define type carbon::f64
#define LEN 100
using tensor = carbon::tensor<type>;


int main(){
	carbon::manual_seed(0);
	carbon::tensor<carbon::f32> a = carbon::tensor<carbon::f32>::randn({10});
	carbon::tensor<carbon::f64> b = carbon::tensor<carbon::f64>::randn({10});
	std::cout << a << "\n\n";
	std::cout << b << "\n\n";

	return 0;
}

