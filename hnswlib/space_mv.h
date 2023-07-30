#pragma once
#include "space_l2.h"
#include "space_ip.h"
#include <unordered_map>

namespace hnswlib {

template<typename DOCIDTYPE>
class BaseMultiVectorSpace : public SpaceInterface<float> {
 public:
    virtual DOCIDTYPE get_doc_id(const void *datapoint) = 0;

    virtual void set_doc_id(void *datapoint, DOCIDTYPE doc_id) = 0;
};

template<typename DOCIDTYPE>
class MultiVectorL2Space : public BaseMultiVectorSpace<DOCIDTYPE> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t vector_size_;
    size_t dim_;

 public:
    MultiVectorL2Space(size_t dim) {
        fstdistfunc_ = L2Sqr;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
        else if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #elif defined(USE_AVX)
        if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #endif

        if (dim % 16 == 0)
            fstdistfunc_ = L2SqrSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = L2SqrSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = L2SqrSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = L2SqrSIMD4ExtResiduals;
#endif
        dim_ = dim;
        vector_size_ = dim * sizeof(float);
        data_size_ = vector_size_ + sizeof(DOCIDTYPE);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    DOCIDTYPE get_doc_id(const void *datapoint) {
        return *(DOCIDTYPE *)((char *)datapoint + vector_size_);
    }

    void set_doc_id(void *datapoint, DOCIDTYPE doc_id) {
        *(DOCIDTYPE*)((char *)datapoint + vector_size_) = doc_id;
    }

    ~MultiVectorL2Space() {}
};

template<typename DOCIDTYPE>
class MultiVectorInnerProductSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t vector_size_;
    size_t dim_;

 public:
    MultiVectorInnerProductSpace(size_t dim) {
        fstdistfunc_ = InnerProductDistance;
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX512;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX512;
        } else if (AVXCapable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        }
    #elif defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        }
    #endif
    #if defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductSIMD4Ext = InnerProductSIMD4ExtAVX;
            InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtAVX;
        }
    #endif

        if (dim % 16 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = InnerProductDistanceSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = InnerProductDistanceSIMD4ExtResiduals;
#endif
        vector_size_ = dim * sizeof(float);
        data_size_ = vector_size_ + sizeof(DOCIDTYPE);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    DOCIDTYPE get_doc_id(const void *datapoint) {
        return *(DOCIDTYPE *)((char *)datapoint + vector_size_);
    }

    void set_doc_id(void *datapoint, DOCIDTYPE doc_id) {
        *(DOCIDTYPE*)((char *)datapoint + vector_size_) = doc_id;
    }

    ~MultiVectorInnerProductSpace() {}
};

template<typename DOCIDTYPE, typename dist_t>
class MultiVectorSearchStopCondition : public BaseSearchStopCondition<dist_t> {
    size_t num_docs;
    std::unordered_map<DOCIDTYPE, size_t> doc_counter;
    BaseMultiVectorSpace<DOCIDTYPE>& space;

 public:
    MultiVectorSearchStopCondition(BaseMultiVectorSpace<DOCIDTYPE>& space, size_t dim)
        : space(space) {
            num_docs = 0;
        }

    void add_point(labeltype label, const void *datapoint, dist_t dist) {
        DOCIDTYPE doc_id = space.get_doc_id(datapoint);
        if (doc_counter[doc_id] == 0) {
            num_docs += 1;
        }
        doc_counter[doc_id] += 1;
    }

    void remove_point(labeltype label, const void *datapoint, dist_t dist) {
        DOCIDTYPE doc_id = space.get_doc_id(datapoint);
        doc_counter[doc_id] -= 1;
        if (doc_counter[doc_id] == 0) {
            num_docs -= 1;
        }
    }

    bool stop_search(dist_t candidate_dist, dist_t lowerBound, size_t ef) {
        bool stop_search = candidate_dist > lowerBound && num_docs == ef;
        return stop_search;
    }

    bool consider_candidate(dist_t candidate_dist, dist_t lowerBound, size_t ef) {
        bool consider_candidate = num_docs < ef || lowerBound > candidate_dist;
        return consider_candidate;
    }

    bool remove_extra(size_t ef) {
        bool remove_extra = num_docs > ef;
        return remove_extra;
    }

    ~MultiVectorSearchStopCondition() {}
};

}  // namespace hnswlib
