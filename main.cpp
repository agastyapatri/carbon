#include <cstdlib>
#include "carbon.hpp"
#include <iostream> 
#include <numbers>
#include <chrono> 
#define type carbon::f32
#define LEN 5
using tensor = carbon::tensor<type>;


int main(){
	carbon::manual_seed(0);
	carbon::tensor<type> a = carbon::tensor<type>::randn({LEN, LEN, LEN});
	std::cout << a << "\n\n\n";
	carbon::tensor<type> b = carbon::exp(a);
	std::cout << b << std::endl;



	return 0;
}

