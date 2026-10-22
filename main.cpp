#include <cstdlib>
#include "carbon.hpp"
#include <iostream> 
#include <numbers>
#include <chrono> 
#define LEN 5

int main(){
	carbon::manual_seed(0);
	carbon::tensor<int> a({10}, {1,2,3,4,5,6,7,8,9,10});
	std::cout << a << std::endl;
	



	return 0;
}

