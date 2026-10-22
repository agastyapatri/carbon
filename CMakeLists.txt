cmake_minimum_required(VERSION 4.0)
set(CMAKE_CXX_COMPILER_NAMES clang++ g++ CXX)
project(carbon VERSION 0.0.1 LANGUAGES CXX)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build Type" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release
		RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


#	CARBON BUILD OPTIONS
option(CARBON_USE_BLAS         "Use CBLAS for f32 / f64 kernels" ON)
option(CARBON_NATIVE           "optimize for the host CPU (-march=native)" OFF)
option(CARBON_USE_TBB          "Link TBB to enable parallel execution of std::execution::par_unseq" ON)
option(CARBON_BUILD_TESTS      "Build the correctness test suite" OFF)
option(CARBON_BUILD_BENCHMARKS "Build the benchmarking suite" OFF)
option(CARBON_PROFILING        "Keep the frame pointers + debug info so profilers can unwind" OFF)
option(CARBON_SANITIZE         "Build with ASan + UBSan for tests only. Never to be used while benchmarking" OFF)

#	Build the library. Everything (main, tests, benchmarking) links against this one lib. 
add_library(carbon STATIC src/carbon.cpp)
target_include_directories(carbon PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(CARBON_NATIVE)
	message(STATUS "carbon: Optimizing for the host CPU; building with -march=native")
	target_compile_options(carbon PRIVATE -march=native)
endif()

if(CARBON_PROFILING)
	message(STATUS "carbon: Profiling")
	target_compile_options(carbon PUBLIC -g -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer)
endif()

if(CARBON_SANITIZE)
	message(STATUS "carbon: Building with -fsanitize=address,undefined")
	target_compile_options(carbon PUBLIC -g -fsanitize=address,undefined -fno-omit-frame-pointer)
	target_link_options(carbon PUBLIC -fsanitize=address,undefined)
endif()

#	FINDING AND LINKING CBLAS
if(CARBON_USE_BLAS)
	if(BLA_VENDOR STREQUAL "FLAME")
		set(CBLAS_SUFFIXES blis)
	elseif(BLA_VENDOR STREQUAL "OpenBLAS")
		set(CBLAS_SUFFIXES openblas openblas-pthread)
	else()
		set(CBLAS_SUFFIXES openblas openblas-pthread blis)
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
		message(WARNING "carbon: no usable CBLAS found; using generic kernels")
	endif()
else()
	message(STATUS "carbon: BLAS disabled; using generic kernels")
endif()

#	ENABLING THE PARALLEL STL BACKEND BY LINKING AGAINST INTEL TBB 
set(CARBON_PSTL_BACKEND "serial")
if(CARBON_USE_TBB)
	find_package(TBB CONFIG QUIET)
	if(TBB_FOUND)
		target_link_libraries(carbon PUBLIC TBB::tbb)
		target_compile_definitions(carbon PUBLIC _GLIBCXX_USE_TBB_PAR_BACKEND=1)
		set(CARBON_PSTL_BACKEND "tbb")
	else()
		message(WARNING "carbon: TBB not found; std::execution::par_unseq will run serially")
	endif()
endif()

#	LINKING INTEL TBB FOR PARALLEL EXECUTION POLICIES
if(CARBON_PSTL_BACKEND STREQUAL "serial")
	target_compile_definitions(carbon PUBLIC _GLIBCXX_USE_TBB_PAR_BACKEND=0)
endif()
message(STATUS "carbon: STL execution backend: ${CARBON_PSTL_BACKEND}")

#	BUILDING MAIN
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
	add_executable(main main.cpp)
	target_link_libraries(main PRIVATE carbon)
endif()


#	ADDING: GOOGLETEST

if(CARBON_BUILD_TESTS)
	include(FetchContent)
	FetchContent_Declare(
		googletest
		GIT_REPOSITORY https://github.com/google/googletest.git 
		GIT_TAG		   v1.18.0 
		GIT_SHALLOW	   TRUE
	)
	set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
	FetchContent_MakeAvailable(googletest)
	enable_testing() 
	# add_executable(carbon_tests tests/*.cpp)
	file(GLOB CARBON_TEST_SRCS CONFIGURE_DEPENDS tests/*.cpp)
	add_executable(carbon_tests ${CARBON_TEST_SRCS})
	target_link_libraries(carbon_tests PRIVATE carbon GTest::gtest_main)
	include(GoogleTest)
	gtest_discover_tests(carbon_tests)
endif()

# if(CARBON_BUILD_BENCHMARKS)
# endif()
#













