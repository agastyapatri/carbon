#include "carbon.hpp"
#include <iostream> 
#define type double



int main(){
	carbon::tensor<type>::manual_seed(0);
	carbon::tensor<type> a = carbon::tensor<type>::randn({5,5});
	a.cube_();
	for(auto i : a.data())
		std::cout << i << " ";
	std::cout << std::endl;
	return 0;
}

