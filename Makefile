main: main.cpp tensor.cpp 
	clang++ -std=c++20 main.cpp tensor.cpp -o main -lopenblas

