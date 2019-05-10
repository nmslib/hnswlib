cmake_minimum_required (VERSION 2.6)
project (hnsw_lib)

include_directories("${PROJECT_BINARY_DIR}")



set(SOURCE_EXE main.cpp)            

set(SOURCE_LIB sift_1b.cpp delete_test.cpp)

add_library(sift_test STATIC ${SOURCE_LIB})


add_executable(main ${SOURCE_EXE})
SET( CMAKE_CXX_FLAGS  "-Ofast -DNDEBUG -std=c++11 -DHAVE_CXX0X -openmp -march=native -fpic -w -ftree-vectorize" ) # -lrt -fopenmp -ftree-vectorizer-verbose=0
target_link_libraries(main sift_test) 
