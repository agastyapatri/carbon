#include <cstdlib>
#include "carbon.hpp"
#include <iostream> 
#include <numbers>
#include <chrono> 
#define type carbon::f64
#define LEN 1000
using tensor = carbon::tensor<type>;


int main(){
	tensor::manual_seed(0);
	tensor a = tensor::randn({LEN, LEN});
	tensor b = tensor::randn({LEN, LEN});
	auto start = std::chrono::high_resolution_clock::now(); 
	tensor c = carbon::gemm(a, b);
	auto end = std::chrono::high_resolution_clock::now(); 
	std::chrono::duration<carbon::f64, std::milli> elapsed = end - start; 
	std::cout << "time taken for matmul of " << LEN << "x" << LEN << " matrices: " << elapsed.count() << '\n';


	return 0;
}

