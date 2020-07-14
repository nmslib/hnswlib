#pragma once
#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#else
#include <x86intrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif
#endif

#include <queue>
#include <vector>
#include <iostream>
#include <string.h>

namespace hnswlib {
    template <typename T>
    class pairGreater {
    public:
        bool operator()(const T& p1, const T& p2) {
            return p1.first > p2.first;
        }
    };

    static void checkIOError(std::istream &in) {
        if (in.eof())
            throw std::runtime_error("Error loading index: eof reached before expectation.");
        if (in.fail())
            throw std::runtime_error("Error loading index: failed to read file.");
    }

    static void checkIOError(std::ostream &out) {
        if (!out.good())
            throw std::runtime_error("Error saving index.");
    }

    template<typename T>
    static void writeBinaryPOD(std::ostream &out, const T &podRef) {
        out.write((char *) &podRef, sizeof(T));
        checkIOError(out);
    }

    template<typename T>
    static void readBinaryPOD(std::istream &in, T &podRef) {
        in.read((char *) &podRef, sizeof(T));
        checkIOError(in);
    }

    template<typename dist_t>
    class SpaceInterface {
    protected:
        const size_t dim_;

    public:
        SpaceInterface(size_t dim) : dim_(dim) {
        }

        size_t get_dimension() {
            return dim_;
        }

        virtual dist_t calculate_distance(const dist_t *pVect1, const dist_t *pVect2) = 0;

        virtual ~SpaceInterface() {}
    };

    template<typename src_t, typename dist_t>
    class NormalizerInterface {
        protected:
            const size_t dim_;

        public:
            NormalizerInterface(size_t dim) : dim_(dim) {
            }

            virtual void normalize_vector(const src_t *data, dist_t *norm_array) = 0;

            virtual ~NormalizerInterface() {}
    };

    template<typename dist_t, typename labeltype>
    class AlgorithmInterface {
    public:
        struct Neighbour {
            dist_t distance;
            labeltype label;
        };

        virtual void addPoint(const dist_t *datapoint, const labeltype& label)=0;
        virtual std::vector<Neighbour> searchKnn(const dist_t *, size_t) const = 0;
        virtual void saveIndex(const std::string &location)=0;
        virtual ~AlgorithmInterface(){
        }
    };
}

#include "space_l2.h"
#include "space_ip.h"
#include "bruteforce.h"
#include "hnswalg.h"
