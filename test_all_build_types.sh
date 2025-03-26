#!/bin/bash

set -euo pipefail

source_path="${PWD}"
for c_compiler in clang gcc; do
  for build_type in Debug Release RelWithDebInfo MinSizeRel; do
    # "tsan" is currently not present here because it has some unresolved
    # issues.
    for sanitizers in "no_sanitizers" "asan_ubsan"; do
      if [[ "${c_compiler}" == "gcc" &&
            "${sanitizers}" == "tsan" ]]; then
        continue
      fi
      echo -------------------------------------------------------------------
      echo "Starting ${c_compiler} ${build_type} ${sanitizers}"
      echo -------------------------------------------------------------------
      echo
      "${c_compiler}" --version
      echo

      build_path="${source_path}/build/${build_type}_${c_compiler}_${sanitizers}"
      rm -rf "${build_path}"
      mkdir -p "${build_path}"
      cd "${build_path}"
      if [[ "${c_compiler}" == "gcc" ]]; then
        cxx_compiler=g++
      else
        cxx_compiler=clang++
      fi
      cmake_cmd=(
        cmake -G Ninja 
        "-DCMAKE_BUILD_TYPE=${build_type}"
        "-DCMAKE_C_COMPILER=${c_compiler}"
        "-DCMAKE_CXX_COMPILER=${cxx_compiler}"
        -S "${source_path}"
        -B "${build_path}"
      )
      if [[ "${sanitizers}" == "asan_ubsan" ]]; then
        cmake_cmd+=( -DENABLE_ASAN=ON -DENABLE_UBSAN=ON )
      fi
      if [[ "${sanitizers}" == "tsan" ]]; then
        cmake_cmd+=( -DENABLE_TSAN=ON )
      fi
      ( set -x; "${cmake_cmd[@]}" )
      time ( set -x; ninja -j8 )
      time ( set -x; ctest )
      cd "${source_path}"
      echo
      echo -------------------------------------------------------------------
      echo "Finished ${c_compiler} ${build_type} ${sanitizers}"
      echo -------------------------------------------------------------------
      echo
      echo
    done
  done
done
