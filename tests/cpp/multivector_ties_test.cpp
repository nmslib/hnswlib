#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include "../../hnswlib/hnswlib.h"

namespace {
struct Point {
    float coordinates[2];
    unsigned int document;
};

class LabelFilter : public hnswlib::BaseFilterFunctor {
    unsigned int mode_;

 public:
    explicit LabelFilter(unsigned int mode) : mode_(mode) {}

    bool operator()(hnswlib::labeltype label) override {
        return mode_ == 0 || (mode_ == 1 && label % 2 == 0);
    }
};

class CountingStopCondition : public hnswlib::MultiVectorSearchStopCondition<unsigned int, float> {
 public:
    size_t removals = 0;
    size_t stopped_searches = 0;

    CountingStopCondition(hnswlib::BaseMultiVectorSpace<unsigned int>& space, size_t k, size_t ef)
        : hnswlib::MultiVectorSearchStopCondition<unsigned int, float>(space, k, ef) {}

    void remove_point_from_result(hnswlib::labeltype label, const void* point, float distance) override {
        ++removals;
        hnswlib::MultiVectorSearchStopCondition<unsigned int, float>::remove_point_from_result(
            label, point, distance);
    }

    bool should_stop_search(float candidate_distance, float lower_bound) override {
        const bool stop = hnswlib::MultiVectorSearchStopCondition<unsigned int, float>::should_stop_search(
            candidate_distance, lower_bound);
        if (stop) ++stopped_searches;
        return stop;
    }
};

float distance(const Point& point, const float* query) {
    const float x = point.coordinates[0] - query[0];
    const float y = point.coordinates[1] - query[1];
    return x * x + y * y;
}

void check_results(const std::vector<Point>& points, const std::vector<bool>& deleted,
                   LabelFilter& filter, const float* query, size_t k,
                   const std::vector<std::pair<float, hnswlib::labeltype>>& results) {
    std::map<unsigned int, float> expected_documents;
    for (size_t label = 0; label < points.size(); ++label) {
        if (deleted[label] || !filter(label)) continue;
        const unsigned int doc = points[label].document;
        const float d = distance(points[label], query);
        auto found = expected_documents.find(doc);
        if (found == expected_documents.end() || d < found->second) expected_documents[doc] = d;
    }
    std::vector<float> expected_distances;
    for (const auto& doc : expected_documents) expected_distances.push_back(doc.second);
    std::sort(expected_distances.begin(), expected_distances.end());
    expected_distances.resize(std::min(k, expected_distances.size()));

    std::map<unsigned int, float> actual_documents;
    float previous_distance = -1.0f;
    std::set<hnswlib::labeltype> labels;
    for (const auto& result : results) {
        const size_t label = result.second;
        assert(label < points.size());
        assert(!deleted[label] && filter(label));
        assert(labels.insert(label).second);
        assert(result.first == distance(points[label], query));
        assert(previous_distance <= result.first);
        previous_distance = result.first;
        const unsigned int doc = points[label].document;
        if (actual_documents.count(doc) == 0) actual_documents[doc] = result.first;
    }
    std::vector<float> actual_distances;
    for (const auto& doc : actual_documents) actual_distances.push_back(doc.second);
    std::sort(actual_distances.begin(), actual_distances.end());
    // Tied documents can be returned in any order, but the number of distinct
    // documents and their nearest-vector distances must match the exact oracle.
    assert(actual_distances == expected_distances);
}

void test_three_tied_vectors() {
    const std::vector<Point> points = {{{0, 1}, 1}, {{0, 1}, 1}, {{1, 0}, 2}};
    hnswlib::MultiVectorL2Space<unsigned int> space(2);
    hnswlib::HierarchicalNSW<float> index(&space, points.size(), 16, 100, 8);
    for (size_t i = 0; i < points.size(); ++i) index.addPoint(&points[i], i);
    const float query[2] = {0, 0};
    CountingStopCondition stop(space, 1, 3);
    const auto result = index.searchStopConditionClosest(query, stop);
    LabelFilter filter(0);
    check_results(points, std::vector<bool>(points.size(), false), filter, query, 1, result);
}

void test_ties_against_exact_oracle() {
    size_t removals = 0;
    size_t stopped_searches = 0;
    const float queries[][2] = {{0, 0}, {1, 1}, {-1, 2}, {3, 3}};
    for (unsigned int seed = 0; seed < 24; ++seed) {
        std::mt19937 rng(seed);
        std::vector<Point> points(24);
        hnswlib::MultiVectorL2Space<unsigned int> space(2);
        // More neighbors than points avoids approximate recall being the
        // explanation for a mismatch with the exhaustive document oracle.
        hnswlib::HierarchicalNSW<float> index(&space, points.size(), 32, 100, seed);
        for (size_t i = 0; i < points.size(); ++i) {
            points[i] = {{static_cast<float>(rng() % 3), static_cast<float>(rng() % 3)},
                         static_cast<unsigned int>(rng() % 7)};
            index.addPoint(&points[i], i);
        }
        std::vector<bool> deleted(points.size(), false);
        for (unsigned int deletion_mode = 0; deletion_mode < 2; ++deletion_mode) {
            if (deletion_mode != 0) {
                for (size_t i = 0; i < points.size(); i += 3) {
                    index.markDelete(i);
                    deleted[i] = true;
                }
            }
            for (unsigned int filter_mode = 0; filter_mode < 3; ++filter_mode) {
                LabelFilter filter(filter_mode);
                for (const auto& query : queries) {
                    for (size_t k : {size_t(0), size_t(1), size_t(2), size_t(7), size_t(9)}) {
                        for (size_t ef : {size_t(0), k, k + 1, size_t(8)}) {
                            CountingStopCondition stop(space, k, ef);
                            auto result = index.searchStopConditionClosest(query, stop, &filter);
                            check_results(points, deleted, filter, query, k, result);
                            removals += stop.removals;
                            stopped_searches += stop.stopped_searches;
                        }
                    }
                }
            }
        }
    }
    assert(removals > 0);
    assert(stopped_searches > 0);
    std::cout << "tied-document queries: 11520; heap removals: " << removals
              << "; distance-based stops: " << stopped_searches << '\n';
}
}  // namespace

int main() {
    test_three_tied_vectors();
    test_ties_against_exact_oracle();
    return 0;
}
