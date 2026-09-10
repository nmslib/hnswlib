#pragma once

// https://github.com/nmslib/hnswlib/pull/508
// This allows others to provide their own error stream (e.g. RcppHNSW)
#ifndef HNSWLIB_ERR_OVERRIDE
  #define HNSWERR std::cerr
#else
  #define HNSWERR HNSWLIB_ERR_OVERRIDE
#endif

#ifndef NO_MANUAL_VECTORIZATION
#if (defined(__SSE__) || _M_IX86_FP > 0 || defined(_M_AMD64) || defined(_M_X64))
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#ifdef __AVX512F__
#define USE_AVX512
#endif
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
static void cpuid(int32_t out[4], int32_t eax, int32_t ecx) {
    __cpuidex(out, eax, ecx);
}
static __int64 xgetbv(unsigned int x) {
    return _xgetbv(x);
}
#else
#include <x86intrin.h>
#include <cpuid.h>
#include <stdint.h>
static void cpuid(int32_t cpuInfo[4], int32_t eax, int32_t ecx) {
    __cpuid_count(eax, ecx, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
}
static uint64_t xgetbv(unsigned int index) {
    uint32_t eax, edx;
    __asm__ __volatile__("xgetbv" : "=a"(eax), "=d"(edx) : "c"(index));
    return ((uint64_t)edx << 32) | eax;
}
#endif

#if defined(USE_AVX512)
#include <immintrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#define PORTABLE_ALIGN64 __declspec(align(64))
#endif

// Adapted from https://github.com/Mysticial/FeatureDetector
#define _XCR_XFEATURE_ENABLED_MASK  0
static bool AVXCapable() {
    int cpuInfo[4];
    cpuid(cpuInfo, 0, 0);
    if (cpuInfo[0] < 1) return false;

    cpuid(cpuInfo, 1, 0);
    bool hasAVX = (cpuInfo[2] & (1 << 28)) != 0;
    bool hasOSXSAVE = (cpuInfo[2] & (1 << 27)) != 0;
    if (!hasAVX || !hasOSXSAVE) return false;
  
    return (xgetbv(_XCR_XFEATURE_ENABLED_MASK) & 0x6) == 0x6;
}

static bool AVX512Capable() {
    if (!AVXCapable()) return false;

    int cpuInfo[4];
    cpuid(cpuInfo, 0, 0);
    if (cpuInfo[0] < 7) return false;

    cpuid(cpuInfo, 7, 0);
    if ((cpuInfo[1] & (1 << 16)) == 0) return false;

    return (xgetbv(_XCR_XFEATURE_ENABLED_MASK) & 0xe6) == 0xe6;
}
#endif

#include <queue>
#include <vector>
#include <iostream>
#include <string.h>

namespace hnswlib {
typedef size_t labeltype;

// This can be extended to store state for filtering (e.g. from a std::set)
class BaseFilterFunctor {
 public:
    virtual bool operator()(hnswlib::labeltype id) { return true; }
    virtual ~BaseFilterFunctor() {};
};

template<typename dist_t>
class BaseSearchStopCondition {
 public:
    virtual void add_point_to_result(labeltype label, const void *datapoint, dist_t dist) = 0;

    virtual void remove_point_from_result(labeltype label, const void *datapoint, dist_t dist) = 0;

    virtual bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) = 0;

    virtual bool should_consider_candidate(dist_t candidate_dist, dist_t lowerBound) = 0;

    virtual bool should_remove_extra() = 0;

    virtual void filter_results(std::vector<std::pair<dist_t, labeltype >> &candidates) = 0;

    virtual ~BaseSearchStopCondition() {}
};

template <typename T>
class pairGreater {
 public:
    bool operator()(const T& p1, const T& p2) {
        return p1.first > p2.first;
    }
};

template<typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
    out.write((char *) &podRef, sizeof(T));
}

template<typename T>
static void readBinaryPOD(std::istream &in, T &podRef) {
    in.read((char *) &podRef, sizeof(T));
}

template<typename MTYPE>
using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);

template<typename MTYPE>
class SpaceInterface {
 public:
    // virtual void search(void *);
    virtual size_t get_data_size() = 0;

    virtual DISTFUNC<MTYPE> get_dist_func() = 0;

    virtual void *get_dist_func_param() = 0;

    virtual ~SpaceInterface() {}
};

template<typename dist_t>
class AlgorithmInterface {
 public:
    virtual void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false) = 0;

    virtual std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(const void*, size_t, BaseFilterFunctor* isIdAllowed = nullptr) const = 0;

    // Return k nearest neighbor in the order of closer fist
    virtual std::vector<std::pair<dist_t, labeltype>>
        searchKnnCloserFirst(const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const;

    virtual void saveIndex(const std::string &location) = 0;
    virtual ~AlgorithmInterface(){
    }
};

template<typename dist_t>
std::vector<std::pair<dist_t, labeltype>>
AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void* query_data, size_t k,
                                                 BaseFilterFunctor* isIdAllowed) const {
    std::vector<std::pair<dist_t, labeltype>> result;

    // here searchKnn returns the result in the order of further first
    auto ret = searchKnn(query_data, k, isIdAllowed);
    {
        size_t sz = ret.size();
        result.resize(sz);
        while (!ret.empty()) {
            result[--sz] = ret.top();
            ret.pop();
        }
    }

    return result;
}
}  // namespace hnswlib

#include "space_l2.h"
#include "space_ip.h"
#include "stop_condition.h"
#include "bruteforce.h"
#include "hnswalg.h"
