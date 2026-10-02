#pragma once

#include "hnswlib.h"

#include <unordered_map>
#include <fstream>
#include <mutex>
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <limits>

namespace hnswlib {
template<typename dist_t>
class BruteforceSearch : public AlgorithmInterface<dist_t> {
 public:
    char *data_;
    size_t maxelements_;
    size_t cur_element_count;
    size_t size_per_element_;

    size_t data_size_;
    DISTFUNC <dist_t> fstdistfunc_;
    void *dist_func_param_;
    std::mutex index_lock;

    std::unordered_map<labeltype, size_t > dict_external_to_internal;


    BruteforceSearch(SpaceInterface <dist_t> *s)
        : data_(nullptr),
            maxelements_(0),
            cur_element_count(0),
            size_per_element_(0),
            data_size_(0),
            dist_func_param_(nullptr) {
    }


    BruteforceSearch(SpaceInterface<dist_t> *s, const std::string &location)
        : data_(nullptr),
            maxelements_(0),
            cur_element_count(0),
            size_per_element_(0),
            data_size_(0),
            dist_func_param_(nullptr) {
        loadIndex(location, s);
    }


    BruteforceSearch(SpaceInterface <dist_t> *s, size_t maxElements) {
        maxelements_ = maxElements;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        size_per_element_ = data_size_ + sizeof(labeltype);
        data_ = (char *) malloc(maxElements * size_per_element_);
        if (data_ == nullptr)
            HNSWLIB_THROW_RUNTIME_ERROR("Not enough memory: BruteforceSearch failed to allocate data");
        cur_element_count = 0;
    }


    ~BruteforceSearch() {
        free(data_);
    }

    // Labels sit at a packed offset; load via memcpy (nmslib/hnswlib#665).
    inline labeltype getExternalLabel(size_t internal_id) const {
        labeltype return_label;
        memcpy(&return_label, data_ + internal_id * size_per_element_ + data_size_, sizeof(labeltype));
        return return_label;
    }


    Status addPointNoExceptions(const void *datapoint, labeltype label, bool replace_deleted = false) override {
        int idx;
        {
            std::unique_lock<std::mutex> lock(index_lock);

            auto search = dict_external_to_internal.find(label);
            if (search != dict_external_to_internal.end()) {
                idx = search->second;
            } else {
                if (cur_element_count >= maxelements_) {
                    return Status("The number of elements exceeds the specified limit");
                }
                idx = cur_element_count;
                dict_external_to_internal[label] = idx;
                cur_element_count++;
            }
        }
        memcpy(data_ + size_per_element_ * idx + data_size_, &label, sizeof(labeltype));
        memcpy(data_ + size_per_element_ * idx, datapoint, data_size_);
        return OkStatus();
    }


    void removePoint(labeltype cur_external) {
        std::unique_lock<std::mutex> lock(index_lock);

        auto found = dict_external_to_internal.find(cur_external);
        if (found == dict_external_to_internal.end()) {
            return;
        }

        dict_external_to_internal.erase(found);

        size_t cur_c = found->second;
        labeltype label = getExternalLabel(cur_element_count - 1);
        dict_external_to_internal[label] = cur_c;
        memcpy(data_ + size_per_element_ * cur_c,
                data_ + size_per_element_ * (cur_element_count-1),
                data_size_+sizeof(labeltype));
        cur_element_count--;
    }


    using DistanceLabelPriorityQueue = typename AlgorithmInterface<dist_t>::DistanceLabelPriorityQueue;
    StatusOr<DistanceLabelPriorityQueue>
    searchKnnNoExceptions(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const override {
        assert(k <= cur_element_count);
        std::priority_queue<std::pair<dist_t, labeltype >> topResults;
        if (k == 0)
            return topResults;
        dist_t lastdist = std::numeric_limits<dist_t>::max();
        for (int i = 0; i < cur_element_count; i++) {
            dist_t dist = fstdistfunc_(query_data, data_ + size_per_element_ * i, dist_func_param_);
            if (!distanceIsFinite(dist)) {
                labeltype label = getExternalLabel(i);
                if (isIdAllowed && !(*isIdAllowed)(label))
                    continue;
                return Status("BruteforceSearch encountered a non-finite distance; "
                              "check input values or rescale to avoid overflow");
            }
            if (dist <= lastdist || topResults.size() < k) {
                labeltype label = getExternalLabel(i);
                if ((!isIdAllowed) || (*isIdAllowed)(label)) {
                    topResults.emplace(dist, label);
                    if (topResults.size() > k)
                        topResults.pop();
                    if (!topResults.empty())
                        lastdist = topResults.top().first;
                }
            }
        }
        return topResults;
    }


    Status saveIndexNoExceptions(std::ostream &output) {
        StreamExceptionsOff guard(output);
        return invokeWithoutStreamThrow([&]() -> Status {
            if (!output) {
                return Status("Cannot save index: output stream is not open or in a failed state");
            }
            writeBinaryPOD(output, maxelements_);
            writeBinaryPOD(output, size_per_element_);
            writeBinaryPOD(output, cur_element_count);
            if (!output.good()) {
              return Status("Failed writing index metadata");
            }

            output.write(data_, maxelements_ * size_per_element_);
            if (!output.good()) {
              return Status("Failed writing vector data");
            }
            return OkStatus();
        });
    }


    Status saveIndexNoExceptions(const std::string &location) override {
        std::ofstream output(location, std::ios::binary);
        if (!output.is_open()) {
            return Status("Cannot save index: failed to open output file");
        }
        return saveIndexNoExceptions(output);
    }


    void loadIndex(std::istream &input, SpaceInterface<dist_t> *s) {
        if (!input) {
            HNSWLIB_THROW_RUNTIME_ERROR("Cannot load index: input stream is not open or not readable");
        }
        readBinaryPOD(input, maxelements_);
        readBinaryPOD(input, size_per_element_);
        readBinaryPOD(input, cur_element_count);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        size_per_element_ = data_size_ + sizeof(labeltype);
        data_ = (char *) malloc(maxelements_ * size_per_element_);
        if (data_ == nullptr)
            HNSWLIB_THROW_RUNTIME_ERROR("Not enough memory: loadIndex failed to allocate data");

        input.read(data_, maxelements_ * size_per_element_);
    }


    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s) {
        std::ifstream input(location, std::ios::binary);

        loadIndex(input, s);

        input.close();
    }

 private:
    template<typename T>
    static bool distanceIsFinite(const T&) {
        return true;
    }

    // Inspect the exponent bits: std::isfinite may be optimized away by fast-math.
    static bool distanceIsFinite(float distance) {
        static_assert(sizeof(float) == sizeof(std::uint32_t) &&
                      std::numeric_limits<float>::is_iec559,
                      "BruteforceSearch requires IEEE 754 binary32 floats");
        std::uint32_t bits;
        std::memcpy(&bits, &distance, sizeof(bits));
        return (bits & 0x7f800000u) != 0x7f800000u;
    }

    static bool distanceIsFinite(double distance) {
        static_assert(sizeof(double) == sizeof(std::uint64_t) &&
                      std::numeric_limits<double>::is_iec559,
                      "BruteforceSearch requires IEEE 754 binary64 doubles");
        std::uint64_t bits;
        std::memcpy(&bits, &distance, sizeof(bits));
        return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
    }
};
}  // namespace hnswlib
