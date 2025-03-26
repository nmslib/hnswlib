cmake_minimum_required(VERSION 3.0...3.26)

project(hnswlib
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

# These example/test targets catch exceptions, so exceptions should always be
# enabled building these files even if they are disabled in other targets.
# We check that each target included in this list is a real target.
set(HNSWLIB_TARGETS_REQUIRING_EXCEPTIONS
    example_mt_filter
    example_mt_replace_deleted
    example_mt_search
    multiThread_replace_test
    test_updates)

set(EXAMPLE_NAMES
    example_epsilon_search
    example_filter
    example_mt_filter
    example_mt_replace_deleted
    example_mt_search
    example_multivector_search
    example_replace_deleted
    example_search
    )

set(TEST_NAMES
    epsilon_search_test
    multiThread_replace_test
    multiThreadLoad_test
    multivector_search_test
    resize_test
    searchKnnCloserFirst_test
    searchKnnWithFilter_test
    )

function(add_cxx_flags)
    foreach(flag IN LISTS ARGN)
        string(APPEND CMAKE_CXX_FLAGS " ${flag}")
    endforeach()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" PARENT_SCOPE)
endfunction()

# Adds an example or test target. The target name parameter is followed by
# the list of source files. Automatically links with the hnswlib library.
# Also decides whether to enable exceptions when building the target.
# If HNSWLIB_ENABLE_EXCEPTIONS is ON, exceptions are always enabled.
# If HNSWLIB_ENABLE_EXCEPTIONS is OFF, exceptions are only enabled for the
# specific targets listed in HNSWLIB_TARGETS_REQUIRING_EXCEPTIONS.
function(add_example_or_test TARGET_NAME ...)
    add_executable(${ARGV})
    target_link_libraries(${TARGET_NAME} hnswlib)
    list(FIND HNSWLIB_TARGETS_REQUIRING_EXCEPTIONS "${TARGET_NAME}" found_at_index)
    if(found_at_index GREATER -1)
        if(NOT HNSWLIB_ENABLE_EXCEPTIONS)
            message("Enabling exceptions for target ${TARGET_NAME} as a special case")
        endif()
        set(should_enable_exceptions ON)
    else()
        set(should_enable_exceptions "${HNSWLIB_ENABLE_EXCEPTIONS}")
    endif()
    if(should_enable_exceptions)
        target_compile_options("${TARGET_NAME}" PUBLIC ${ENABLE_EXCEPTIONS_FLAGS})
    else()
        target_compile_options("${TARGET_NAME}" PUBLIC ${DISABLE_EXCEPTIONS_FLAGS})
    endif()
    if(NOT ${TARGET_NAME} MATCHES "^(main|test_updates)$")
        # test_updates is not included here as a ctest-enabled test because it
        # requires generating test data using update_gen_data.py, which requires
        # installing numpy, which should probably be done in a virtual
        # environment. Also test_updates needs to be invoked twice: without
        # arguments, and with one "update" argument. This is currently handled
        # in the GitHub Actions build.yml file.
        add_test(
            NAME ${TARGET_NAME}
            COMMAND ${TARGET_NAME}
        )
    endif()
endfunction()


option(HNSWLIB_ENABLE_EXCEPTIONS "Whether to enable exceptions in hnswlib" ON)
if(HNSWLIB_ENABLE_EXCEPTIONS)
    message("Exceptions are enabled using HNSWLIB_ENABLE_EXCEPTIONS=ON (default)")
else()
    message("Exceptions are disabled using HNSWLIB_ENABLE_EXCEPTIONS=OFF")
endif()
option(ENABLE_ASAN "Whether to enable AddressSanitizer" OFF)
option(ENABLE_UBSAN "Whether to enable UndefinedBehaviorSanitizer" OFF)
option(ENABLE_TSAN "Whether to enable ThreadSanitizer" OFF)
option(ENABLE_MSAN "Whether to enable MemorySanitizer" OFF)

set(CMAKE_CXX_STANDARD 11)

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

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(ENABLE_EXCEPTIONS_FLAGS /EHsc)
    set(DISABLE_EXCEPTIONS_FLAGS /GR- /D_HAS_EXCEPTIONS=0)
else()
    set(ENABLE_EXCEPTIONS_FLAGS -fexceptions)
    set(DISABLE_EXCEPTIONS_FLAGS -fno-exceptions)
endif()

# Turn on assertions in the RelWithDebInfo build type.
foreach(NDEBUG_FLAG_STR IN ITEMS "/DNDEBUG" "/D NDEBUG" "-DNDEBUG")
    string(REPLACE "${NDEBUG_FLAG_STR}" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO
           "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endforeach()
string(STRIP "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}" CMAKE_CXX_FLAGS_RELWITHDEBINFO)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}"
    CACHE STRING "Flags for RelWithDebInfo configuration." FORCE)

# Start with an empty value of CMAKE_CXX_FLAGS, not the value from the cache.
# It will not override any "default" flags -- those will come from
# per-build-type variables (CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}).
set(CMAKE_CXX_FLAGS "")

if(HNSWLIB_EXAMPLES)
    message("Building examples and tests")
    message("System architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
    enable_testing()
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        # We enable optimizations in all build types, even in Debug.
        add_cxx_flags(-O3 -ffast-math -openmp -ftree-vectorize)
        check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
        if(COMPILER_SUPPORT_NATIVE_FLAG)
            add_cxx_flags(-march=native)
            message("set -march=native flag")
        else()
            check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
            if(COMPILER_SUPPORT_M1_FLAG)
                add_cxx_flags(-mcpu=apple-m1)
                message("set -mcpu=apple-m1 flag")
            endif()
        endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # We enable optimizations in all build types, even in Debug.
        add_cxx_flags(
            -Ofast -lrt -march=native -w -fopenmp -ftree-vectorize
            -ftree-vectorizer-verbose=0)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        if (NOT HNSWLIB_ENABLE_EXCEPTIONS)
            # Do not enable exceptions by default. We will enable them on a
            # case by case basis when needed.
            foreach(config IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
                string(TOUPPER ${config} config_upper)
                set(FLAGS_VAR "CMAKE_CXX_FLAGS_${config_upper}")
                string(REPLACE "/EHsc" "" ${FLAGS_VAR} "${${FLAGS_VAR}}")
                set(${FLAGS_VAR} "${${FLAGS_VAR}}" CACHE STRING
                    "Flags for ${config} configuration." FORCE)
            endforeach()
        endif()
        add_cxx_flags(/O2 /W1 /openmp)
    endif()
    add_cxx_flags(-DHAVE_CXX0X)

    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # Clang/GCC specific flags.
        add_cxx_flags(-fpic)
        if(ENABLE_ASAN)
            add_cxx_flags(-fsanitize=address)
        endif()
        if(ENABLE_UBSAN)
            add_cxx_flags(-fsanitize=undefined)
        endif()
        if(ENABLE_TSAN)
            add_cxx_flags(-fsanitize=thread)
        endif()
        if(ENABLE_MSAN)
            add_cxx_flags(-fsanitize=memory)
        endif()
        if(ENABLE_ASAN OR ENABLE_UBSAN)
            add_cxx_flags(-DHNSWLIB_USE_PREFETCH=0)
        endif()

        add_cxx_flags(-Wall -Wextra -Wpedantic -Werror)

        # Unused functions in header files might still be used by other code
        # including those header files.
        add_cxx_flags(-Wno-unused-function)

        # Unused parameters are OK.
        add_cxx_flags(-Wno-unused-parameter)

        # TODO: re-enable and fix comparisons of integers of different
        # signedness. Not using -Wno-error=sign-compare here, because that will
        # produce a lot of warnings.
        add_cxx_flags(-Wno-sign-compare)

        if(CMAKE_BUILD_TYPE MATCHES "^(Release|MinSizeRel)$")
            # For build types that disable assertions some variables might look
            # like they are not being used. RelWithDebugInfo is not included in
            # this list because we specifically enable assertions for that
            # build type.
            add_cxx_flags(-Wno-unused-variable)
            add_cxx_flags(-Wno-unused-but-set-variable)
        endif()
    endif()

    foreach(example_name IN LISTS EXAMPLE_NAMES)
        add_example_or_test("${example_name}" "examples/cpp/${example_name}.cpp")
    endforeach()

    foreach(test_name IN LISTS TEST_NAMES)
        add_example_or_test("${test_name}" "tests/cpp/${test_name}.cpp")
    endforeach()

    # This test deviates from the above pattern of naming test executables.
    add_example_or_test(test_updates tests/cpp/updates_test.cpp)

    # For historical reasons, the "main" program links with sift_1b.cpp.
    add_example_or_test(main tests/cpp/main.cpp tests/cpp/sift_1b.cpp)

    foreach(target_name IN LISTS HNSWLIB_TARGETS_REQUIRING_EXCEPTIONS)
        if(NOT TARGET ${target_name})
            message(FATAL_ERROR
                    "Target '${target_name}' included in "
                    "HNSWLIB_TARGETS_REQUIRING_EXCEPTIONS does not exist. "
                    "Please check if this is a typo.")
        endif()
    endforeach()
endif()

# Persist CMAKE_CXX_FLAGS in the cache for debuggability.
string(STRIP "${CMAKE_CXX_FLAGS}" CMAKE_CXX_FLAGS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}"
    CACHE STRING "Flags used by the CXX compiler during all build types." FORCE)
