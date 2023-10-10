#pragma once
#include "space_l2.h"
#include "space_ip.h"
#include <assert.h>
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
    size_t curr_num_docs;
    size_t num_docs_to_search;
    size_t ef_collection;
    std::unordered_map<DOCIDTYPE, size_t> doc_counter;
    std::priority_queue<std::pair<dist_t, DOCIDTYPE>> search_results;
    BaseMultiVectorSpace<DOCIDTYPE>& space;

 public:
    MultiVectorSearchStopCondition(
        BaseMultiVectorSpace<DOCIDTYPE>& space,
        size_t num_docs_to_search,
        size_t ef_collection = 10)
        : space(space) {
            curr_num_docs = 0;
            this->num_docs_to_search = num_docs_to_search;
            this->ef_collection = std::max(ef_collection, num_docs_to_search);
        }

    void add_point(labeltype label, const void *datapoint, dist_t dist) {
        DOCIDTYPE doc_id = space.get_doc_id(datapoint);
        if (doc_counter[doc_id] == 0) {
            curr_num_docs += 1;
        }
        search_results.emplace(dist, doc_id);
        doc_counter[doc_id] += 1;
    }

    void remove_point(labeltype label, const void *datapoint, dist_t dist) {
        DOCIDTYPE doc_id = space.get_doc_id(datapoint);
        doc_counter[doc_id] -= 1;
        if (doc_counter[doc_id] == 0) {
            curr_num_docs -= 1;
        }
        search_results.pop();
    }

    bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) {
        bool stop_search = candidate_dist > lowerBound && curr_num_docs == ef_collection;
        return stop_search;
    }

    bool consider_candidate(dist_t candidate_dist, dist_t lowerBound) {
        bool consider_candidate = curr_num_docs < ef_collection || lowerBound > candidate_dist;
        return consider_candidate;
    }

    bool remove_extra() {
        bool remove_extra = curr_num_docs > ef_collection;
        return remove_extra;
    }

    void filter_results(std::vector<std::pair<dist_t, labeltype >> &candidates) {
        while (curr_num_docs > num_docs_to_search) {
            dist_t dist_cand = candidates.back().first;
            dist_t dist_res = search_results.top().first;
            assert(dist_cand == dist_res);
            DOCIDTYPE doc_id = search_results.top().second;
            doc_counter[doc_id] -= 1;
            if (doc_counter[doc_id] == 0) {
                curr_num_docs -= 1;
            }
            search_results.pop();
            candidates.pop_back();
        }
    }

    ~MultiVectorSearchStopCondition() {}
};


template<typename dist_t>
class EpsilonSearchStopCondition : public BaseSearchStopCondition<dist_t> {
    float epsilon;
    size_t min_num_candidates;
    size_t max_num_candidates;
    size_t curr_num_items;
 public:
    EpsilonSearchStopCondition(float epsilon, size_t min_num_candidates, size_t max_num_candidates) {
        assert(min_num_candidates <= max_num_candidates);
        this->epsilon = epsilon;
        this->min_num_candidates = min_num_candidates;
        this->max_num_candidates = max_num_candidates;
        curr_num_items = 0;
    }

    void add_point(labeltype label, const void *datapoint, dist_t dist) {
        curr_num_items += 1;
    }

    void remove_point(labeltype label, const void *datapoint, dist_t dist) {
        curr_num_items -= 1;
    }

    bool should_stop_search(dist_t candidate_dist, dist_t lowerBound) {
        if (candidate_dist > lowerBound && curr_num_items == max_num_candidates) {
            // new candidate can't improve found results
            return true;
        }
        if (candidate_dist > epsilon && curr_num_items >= min_num_candidates) {
            // new candidate is out of epsilon region and
            // minimum number of candidates is checked
            return true;
        }
        return false;
    }

    bool consider_candidate(dist_t candidate_dist, dist_t lowerBound) {
        bool consider_candidate = curr_num_items < max_num_candidates || lowerBound > candidate_dist;
        return consider_candidate;
    }

    bool remove_extra() {
        bool remove_extra = curr_num_items > max_num_candidates;
        return remove_extra;
    }

    void filter_results(std::vector<std::pair<dist_t, labeltype >> &candidates) {
        while (!candidates.empty() && candidates.back().first > epsilon) {
            candidates.pop_back();
        }
        while (candidates.size() > max_num_candidates) {
            candidates.pop_back();
        }
    }

    ~EpsilonSearchStopCondition() {}
};
}  // namespace hnswlib
