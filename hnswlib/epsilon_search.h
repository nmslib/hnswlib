#pragma once

namespace hnswlib {

template<typename dist_t>
class EpsilonSearchStopCondition : public BaseSearchStopCondition<dist_t> {
    float epsilon;
    size_t num_items;
 public:
    EpsilonSearchStopCondition(float epsilon) {
        this->epsilon = epsilon;
        num_items = 0;
    }

    void add_point(labeltype label, const void *datapoint, dist_t dist) {
        num_items += 1;
    }

    void remove_point(labeltype label, const void *datapoint, dist_t dist) {
        num_items -= 1;
    }

    bool stop_search(dist_t candidate_dist, dist_t lowerBound, size_t ef) {
        bool stop_search = (candidate_dist > epsilon) || (candidate_dist > lowerBound && num_items == ef);
        return stop_search;
    }

    bool consider_candidate(dist_t candidate_dist, dist_t lowerBound, size_t ef) {
        bool consider_candidate = (candidate_dist < epsilon) && (num_items < ef || lowerBound > candidate_dist);
        return consider_candidate;
    }

    bool remove_extra(size_t ef) {
        bool remove_extra = num_items > ef;
        return remove_extra;
    }

    ~EpsilonSearchStopCondition() {}
};

}  // namespace hnswlib