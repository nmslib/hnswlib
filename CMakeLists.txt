cmake_minimum_required(VERSION 3.0...3.26)

project(hnswlib
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

add_library(hnswlib INTERFACE)
add_library(hnswlib::hnswlib ALIAS hnswlib)

target_include_directories(hnswlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Install
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS hnswlib
    EXPORT hnswlibTargets)

install(EXPORT hnswlibTargets
    FILE hnswlibConfig.cmake
    NAMESPACE hnswlib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hnswlib)

# Examples and tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(HNSWLIB_EXAMPLES "Build examples and tests." ON)
else()
    option(HNSWLIB_EXAMPLES "Build examples and tests." OFF)
endif()

# Option to treat warnings as errors (useful for CI)
option(HNSWLIB_WERROR "Treat warnings as errors" OFF)

if(HNSWLIB_EXAMPLES)
    set(CMAKE_CXX_STANDARD 14)

    # Common warning flags for GCC/Clang
    set(HNSWLIB_WARNING_FLAGS
        -Wall
        -Wextra
        -Wno-unknown-pragmas
    )

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      set(CMAKE_CXX_FLAGS "-O3 -DHAVE_CXX0X -fpic -ftree-vectorize")
      check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
      if(COMPILER_SUPPORT_NATIVE_FLAG)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
        message(STATUS "set -march=native flag")
      else()
        check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
        if(COMPILER_SUPPORT_M1_FLAG)
          set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1")
          message(STATUS "set -mcpu=apple-m1 flag")
        endif()
      endif()
      # Add OpenMP if available
      find_package(OpenMP QUIET)
      if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
      endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set(CMAKE_CXX_FLAGS "-O3 -lrt -DHAVE_CXX0X -march=native -fpic -fopenmp -ftree-vectorize")
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      set(CMAKE_CXX_FLAGS "/O2 -DHAVE_CXX0X /W3 /openmp /EHsc")
      set(HNSWLIB_WARNING_FLAGS /W3)
    endif()

    # Add warning-as-error flag if requested
    if(HNSWLIB_WERROR)
      if(MSVC)
        list(APPEND HNSWLIB_WARNING_FLAGS /WX)
      else()
        list(APPEND HNSWLIB_WARNING_FLAGS -Werror)
      endif()
    endif()

    # Helper function to add warning flags to targets
    function(hnswlib_add_executable target_name)
        add_executable(${target_name} ${ARGN})
        target_link_libraries(${target_name} hnswlib)
        target_compile_options(${target_name} PRIVATE ${HNSWLIB_WARNING_FLAGS})
    endfunction()

    # examples
    hnswlib_add_executable(example_search examples/cpp/example_search.cpp)
    hnswlib_add_executable(example_epsilon_search examples/cpp/example_epsilon_search.cpp)
    hnswlib_add_executable(example_multivector_search examples/cpp/example_multivector_search.cpp)
    hnswlib_add_executable(example_filter examples/cpp/example_filter.cpp)
    hnswlib_add_executable(example_replace_deleted examples/cpp/example_replace_deleted.cpp)
    hnswlib_add_executable(example_mt_search examples/cpp/example_mt_search.cpp)
    hnswlib_add_executable(example_mt_filter examples/cpp/example_mt_filter.cpp)
    hnswlib_add_executable(example_mt_replace_deleted examples/cpp/example_mt_replace_deleted.cpp)

    # tests
    hnswlib_add_executable(multivector_search_test tests/cpp/multivector_search_test.cpp)
    hnswlib_add_executable(epsilon_search_test tests/cpp/epsilon_search_test.cpp)
    hnswlib_add_executable(test_updates tests/cpp/updates_test.cpp)
    hnswlib_add_executable(searchKnnCloserFirst_test tests/cpp/searchKnnCloserFirst_test.cpp)
    hnswlib_add_executable(searchKnnWithFilter_test tests/cpp/searchKnnWithFilter_test.cpp)
    hnswlib_add_executable(multiThreadLoad_test tests/cpp/multiThreadLoad_test.cpp)
    hnswlib_add_executable(multiThread_replace_test tests/cpp/multiThread_replace_test.cpp)

    add_executable(main tests/cpp/main.cpp tests/cpp/sift_1b.cpp)
    target_link_libraries(main hnswlib)
    target_compile_options(main PRIVATE ${HNSWLIB_WARNING_FLAGS})
endif()
