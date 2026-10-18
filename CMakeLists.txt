cmake_minimum_required(VERSION 4.0)
set(CMAKE_CXX_COMPILER_NAMES clang++ g++ CXX)
project(carbon VERSION 0.0.1 LANGUAGES CXX)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build Type" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release
		RelWithDebInfo MinSizeRel)
endif()
option(CARBON_NATIVE "Optimize for the host CPU (-march=native)" OFF)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(CARBON_USE_BLAS "Option for using BLAS for f32 / f64 kernels" ON)

set(SOURCES src/carbon.cpp)
add_executable(carbon main.cpp ${SOURCES})
if(CARBON_NATIVE)
	target_compile_options(carbon PRIVATE -march=native)
endif()
target_include_directories(carbon PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)


if(CARBON_USE_BLAS)
	if(BLA_VENDOR STREQUAL "FLAME")
		set(CBLAS_SUFFIXES blis)
	elseif(BLA_VENDOR STREQUAL "OpenBLAS")
		set(CBLAS_SUFFIXES openblas)
	else()
		set(CBLAS_SUFFIXES openblas blis)
	endif()
	find_package(BLAS)
	find_path(CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES ${CBLAS_SUFFIXES})
	if(BLAS_FOUND AND CBLAS_INCLUDE_DIR)
		include(CheckCXXSymbolExists)
		set(CMAKE_REQUIRED_INCLUDES ${CBLAS_INCLUDE_DIR})
		set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
		check_cxx_symbol_exists(cblas_sgemm cblas.h CARBON_HAVE_BLAS)
		unset(CMAKE_REQUIRED_INCLUDES)
		unset(CMAKE_REQUIRED_LIBRARIES)
	endif()
	if(CARBON_HAVE_BLAS)
		message(STATUS "carbon: using CBLAS from ${BLAS_LIBRARIES}")
		target_compile_definitions(carbon PRIVATE CARBON_USE_BLAS)
		target_include_directories(carbon PRIVATE ${CBLAS_INCLUDE_DIR})
		target_link_libraries(carbon PRIVATE BLAS::BLAS)
	else() 
		message(WARNING: "carbon: no usable CBLAS found; using generic kernels")
	endif()
else()
	message(STATUS "carbon: BLAS disabled; using generic kernels")
endif()






