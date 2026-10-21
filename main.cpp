#include <cstdlib>
#include "carbon.hpp"
#include <iostream> 
#include <numbers>
#include <chrono> 
#define LEN 5

int main(){
	carbon::manual_seed(0);
	tf32 a = tf32::randn({10, 10});
	std::cout << a << std::endl;
	



	return 0;
}

