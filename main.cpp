#include <cstdlib>
#include "carbon.hpp"
#include <iostream> 
#include <numbers>
#include <chrono> 
#define type carbon::f64
#define LEN 1000
using tensor = carbon::tensor<type>;


int main(){
	carbon::manual_seed(0);
	carbon::tensor<double> a = carbon::tensor<double>::randn({LEN, LEN});
	carbon::tensor<double> b = carbon::tensor<double>::randn({LEN, LEN});
	auto start = std::chrono::high_resolution_clock::now();
	carbon::tensor<double> c = a / b;
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << c << std::endl;


	return 0;
}

