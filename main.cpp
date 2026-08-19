#include "carbon.hpp"
#include <iostream> 
#define type float



int main(){
	carbon::tensor<type>::manual_seed(0);
	carbon::tensor<float> a = carbon::tensor<float>::randn({3,3,3});
	std::cout << a << std::endl;
	std::cout << std::endl;
	a.pow_(3);
	std::cout << a << std::endl;
	return 0;

}

