#include <cstdlib>
#include "carbon.hpp"
#include <iostream> 
#include <numbers>
#include <chrono> 
#define LEN 5

int main(){
	carbon::manual_seed(0);
	carbon::tensor<carbon::f32> a({10}, {1,2,3,4,5,6,7,8,9,10});
	carbon::tensor<carbon::f32> b = carbon::softmax(a, 1);
	std::cout << b << std::endl;

	



	return 0;
}

