#include "carbon.hpp"
#include <iostream> 
#define type float


int main(){
	carbon::tensor<type>::manual_seed(0);
	carbon::tensor<float> a = carbon::tensor<float>::randn({1,10});
	carbon::tensor<float> b = carbon::tensor<float>::randn({1,10});
	return 0;

}

