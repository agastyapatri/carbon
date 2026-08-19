cmake_minimum_required(VERSION 4.0)
project(carbon VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)
set(SOURCES 
	src/carbon.cpp
)

add_executable(carbon main.cpp ${SOURCES})

target_include_directories(carbon PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(carbon PRIVATE ${BLAS_LIBRARIES})



