# CMakeLists.txt
cmake_minimum_required(VERSION 3.11)

include(FetchContent)


# Google Benchmark
# close benchmark-test
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG        v1.9.4
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(benchmark)


# Use master branch as standard
FetchContent_Declare(
  hnswlib_std
  GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
  GIT_TAG        develop
  GIT_SHALLOW    TRUE
)
# avoid library name conflict 
FetchContent_GetProperties(hnswlib_std)
if(NOT hnswlib_std_POPULATED)
  FetchContent_Populate(hnswlib_std)
  # rename master branch library
  add_library(hnswlib_std INTERFACE)
  add_library(hnswlib_std::hnswlib ALIAS hnswlib_std)
  target_include_directories(hnswlib_std INTERFACE 
    $<BUILD_INTERFACE:${hnswlib_std_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>)
endif()


# create benchmark binaries with different versions of hnswlib
# for standard library
add_executable(benchmark_standard benchmarks_main.cpp
  bm_basic.cpp
)
target_link_libraries(benchmark_standard benchmark::benchmark)
target_link_libraries(benchmark_standard hnswlib_std)
set_target_properties(benchmark_standard PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# for current library
add_executable(benchmark_current benchmarks_main.cpp
  bm_basic.cpp
)
target_link_libraries(benchmark_current benchmark::benchmark)
target_link_libraries(benchmark_current hnswlib)
set_target_properties(benchmark_current PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)