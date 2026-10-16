#include "carbon.hpp"
#include <iostream> 
#define type carbon::f32
#define LEN 1000


int main(){
	// carbon::tensor<type>::manual_seed(0);
	carbon::tensor<type> a = carbon::tensor<type>::randn({LEN, LEN});
	carbon::tensor<type> b = carbon::tensor<type>::randn({LEN});
	carbon::tensor<type> c = a + b;
	std::cout << c.shape()[0] << ", " << c.shape()[1] << std::endl;
	return 0;

}

