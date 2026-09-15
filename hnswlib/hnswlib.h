#pragma once

// We use prefetch instructions by default, but we allow their use to be
// disabled by setting HNSWLIB_USE_PREFETCH to 0.
#ifndef HNSWLIB_USE_PREFETCH
#define HNSWLIB_USE_PREFETCH 1
#endif

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

    // CPU support
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX = false;
    if (nIds >= 0x00000001) {
        cpuid(cpuInfo, 0x00000001, 0);
        HW_AVX = (cpuInfo[2] & ((int)1 << 28)) != 0;
    }

    // OS support
    cpuid(cpuInfo, 1, 0);

    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    bool avxSupported = false;
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avxSupported = (xcrFeatureMask & 0x6) == 0x6;
    }
    return HW_AVX && avxSupported;
}

static bool AVX512Capable() {
    if (!AVXCapable()) return false;

    int cpuInfo[4];

    // CPU support
    cpuid(cpuInfo, 0, 0);
    int nIds = cpuInfo[0];

    bool HW_AVX512F = false;
    if (nIds >= 0x00000007) {  //  AVX512 Foundation
        cpuid(cpuInfo, 0x00000007, 0);
        HW_AVX512F = (cpuInfo[1] & ((int)1 << 16)) != 0;
    }

    // OS support
    cpuid(cpuInfo, 1, 0);

    bool osUsesXSAVE_XRSTORE = (cpuInfo[2] & (1 << 27)) != 0;
    bool cpuAVXSuport = (cpuInfo[2] & (1 << 28)) != 0;

    bool avx512Supported = false;
    if (osUsesXSAVE_XRSTORE && cpuAVXSuport) {
        uint64_t xcrFeatureMask = xgetbv(_XCR_XFEATURE_ENABLED_MASK);
        avx512Supported = (xcrFeatureMask & 0xe6) == 0xe6;
    }
    return HW_AVX512F && avx512Supported;
}
#endif

#include <queue>
#include <vector>
#include <iostream>
#include <new>
#include <string>
#include <utility>
#include <string.h>
#include <stdlib.h>

#if defined(__EXCEPTIONS) || _HAS_EXCEPTIONS == 1
#define HNSWLIB_THROW_RUNTIME_ERROR(message) throw std::runtime_error(message)
#else
#define HNSWLIB_THROW_RUNTIME_ERROR(message) do { \
    fprintf(stderr, \
        "FATAL: hnswlib compiled without exception support. " \
        "Use ...NoExceptions functions. " \
        "Exception message: %s", \
        (message)); \
    abort(); \
} while (false)
#endif

#if __cplusplus >= 201703L
#define HNSWLIB_NODISCARD [[nodiscard]]
#else
#define HNSWLIB_NODISCARD
#endif

namespace hnswlib {

// Lightweight Status. Empty message is OK. The error text is copied, so
// callers may pass a stack buffer or a temporary std::string.c_str().
class HNSWLIB_NODISCARD Status {
public:
    Status() {}

    // nullptr is interpreted as OK status.
    Status(const char* message) {
        if (message != nullptr) {
            message_.assign(message);
        }
    }

    Status(std::string message) : message_(std::move(message)) {}

    bool ok() const { return message_.empty(); }

    // nullptr if OK. Pointer is valid for the lifetime of *this.
    const char* message() const { return ok() ? nullptr : message_.c_str(); }

private:
    std::string message_;
};

inline Status OkStatus() { return Status(); }

// Result-or-error container. T is stored inline (no extra heap allocation
// for the wrapper itself) and is move-constructed from the success path so
// search / getDataByLabel do not copy the full result. T is not constructed
// on error paths.
template <typename T>
class StatusOr {
public:
    StatusOr() : status_("StatusOr has no value"), has_value_(false) {}

    StatusOr(T value) : status_(), has_value_(false) {
        new (storage()) T(std::move(value));
        has_value_ = true;
    }

    StatusOr(const char* error)
        : status_(error ? error : "StatusOr has no value"), has_value_(false) {}

    StatusOr(Status status)
        : status_(status.ok() ? Status("StatusOr has no value") : status),
          has_value_(false) {}

    StatusOr(const StatusOr& other)
        : status_(other.status_), has_value_(false) {
        if (other.has_value_) {
            new (storage()) T(*other.storage());
            has_value_ = true;
        }
    }

    StatusOr(StatusOr&& other)
        : status_(other.status_), has_value_(false) {
        if (other.has_value_) {
            new (storage()) T(std::move(*other.storage()));
            has_value_ = true;
        }
    }

    StatusOr& operator=(const StatusOr& other) {
        if (this == &other) {
            return *this;
        }
        destroy();
        status_ = other.status_;
        if (other.has_value_) {
            new (storage()) T(*other.storage());
            has_value_ = true;
        }
        return *this;
    }

    StatusOr& operator=(StatusOr&& other) {
        if (this == &other) {
            return *this;
        }
        destroy();
        status_ = other.status_;
        if (other.has_value_) {
            new (storage()) T(std::move(*other.storage()));
            has_value_ = true;
        }
        return *this;
    }

    ~StatusOr() { destroy(); }

    bool ok() const { return status_.ok() && has_value_; }

    T& value() & { return *storage(); }
    const T& value() const & { return *storage(); }
    T&& value() && { return std::move(*storage()); }

    const T& operator*() const { return value(); }

    Status status() const {
        return ok() ? OkStatus() : (status_.ok() ? Status("StatusOr has no value") : status_);
    }

private:
    T* storage() { return reinterpret_cast<T*>(storage_); }
    const T* storage() const { return reinterpret_cast<const T*>(storage_); }

    void destroy() {
        if (has_value_) {
            storage()->~T();
            has_value_ = false;
        }
    }

    Status status_;
    alignas(T) unsigned char storage_[sizeof(T)];
    bool has_value_;
};

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
    virtual Status addPointNoExceptions(
        const void *datapoint, labeltype label, bool replace_deleted = false) = 0;

    virtual void addPoint(
        const void *datapoint, labeltype label, bool replace_deleted = false) {
        auto status = addPointNoExceptions(datapoint, label, replace_deleted);
        if (!status.ok()) {
            HNSWLIB_THROW_RUNTIME_ERROR(status.message());
        }
    }

    using DistanceLabelPair = std::pair<dist_t, labeltype>;

    // A priority queue of (distance, label) pairs. The largest element at the
    // top corresponds to the element furthest from the query.
    using DistanceLabelPriorityQueue = std::priority_queue<DistanceLabelPair>;

    // A vector of (distance, label) pairs.
    using DistanceLabelVector = std::vector<DistanceLabelPair>;

    virtual DistanceLabelPriorityQueue searchKnn(
            const void* query_data,
            size_t k,
            BaseFilterFunctor* isIdAllowed = nullptr) const {
        auto result = searchKnnNoExceptions(query_data, k, isIdAllowed);
        if (!result.ok()) {
            HNSWLIB_THROW_RUNTIME_ERROR(result.status().message());
        }
        return std::move(result).value();
    }

    virtual StatusOr<DistanceLabelPriorityQueue> searchKnnNoExceptions(
        const void* query_data,
        size_t k,
        BaseFilterFunctor* isIdAllowed = nullptr) const = 0;

    // Return k nearest neighbor in the order of closest neighbor first.
    virtual DistanceLabelVector searchKnnCloserFirst(
            const void* query_data,
            size_t k,
            BaseFilterFunctor* isIdAllowed = nullptr) const {
        auto result =
            searchKnnCloserFirstNoExceptions(query_data, k, isIdAllowed);
        if (!result.ok()) {
            HNSWLIB_THROW_RUNTIME_ERROR(result.status().message());
        }
        return std::move(result).value();
    }

    virtual StatusOr<DistanceLabelVector> searchKnnCloserFirstNoExceptions(
            const void* query_data,
            size_t k,
            BaseFilterFunctor* isIdAllowed = nullptr) const {

        // Here searchKnn returns the result in the order of further first.
        auto status_or_result = searchKnnNoExceptions(query_data, k, isIdAllowed);
        if (!status_or_result.ok()) {
            return status_or_result.status();
        }
        auto ret = std::move(status_or_result.value());

        DistanceLabelVector final_vector;
        size_t sz = ret.size();
        final_vector.resize(sz);
        while (!ret.empty()) {
            final_vector[--sz] = ret.top();
            ret.pop();
        }

        return final_vector;
    }

    virtual void saveIndex(const std::string &location) {
        Status status = saveIndexNoExceptions(location);
        if (!status.ok()) {
            HNSWLIB_THROW_RUNTIME_ERROR(status.message());
        }
    }

    virtual Status saveIndexNoExceptions(const std::string &location) = 0;

    virtual ~AlgorithmInterface(){
    }
};

}  // namespace hnswlib

#include "space_l2.h"
#include "space_ip.h"
#include "stop_condition.h"
#include "bruteforce.h"
#include "hnswalg.h"
