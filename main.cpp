#include "carbon.hpp"
#include <iostream> 
#define type carbon::i32


int main(){
	// carbon::tensor<type>::manual_seed(0);
	carbon::tensor<type> a = carbon::tensor<type>::randn({10, 10});
	carbon::tensor<type> b = carbon::tensor<type>::randn({10});
	carbon::tensor<type> c = carbon::gemv(a, b);
	std::cout << c << std::endl;
	return 0;

}

