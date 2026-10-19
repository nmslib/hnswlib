// Tests setAllowEfBelowK(): with ef >= k the results do not change, and with ef < k the nearest ef of the k
// results are exactly the results of the default search with k = ef (the same walk, whose ef candidates are the
// ef nearest elements it scored).

#include "../../hnswlib/hnswlib.h"

#include <assert.h>

#include <iostream>
#include <random>
#include <vector>

namespace {

using idx_t = hnswlib::labeltype;
using Result = std::priority_queue<std::pair<float, idx_t>>;

class PickDivisibleIds : public hnswlib::BaseFilterFunctor {
    unsigned int divisor;

 public:
    explicit PickDivisibleIds(unsigned int divisor) : divisor(divisor) {}
    bool operator()(idx_t label_id) {
        return label_id % divisor == 0;
    }
};

const int d = 16;
const size_t n = 10000;
const size_t nq = 100;

std::vector<std::pair<float, idx_t>> nearest_first(Result r) {
    std::vector<std::pair<float, idx_t>> v(r.size());
    for (size_t i = v.size(); i-- > 0;) {
        v[i] = r.top();
        r.pop();
    }
    return v;
}

Result search(hnswlib::HierarchicalNSW<float> &index, const void *q, size_t k, size_t ef, bool allow,
              hnswlib::BaseFilterFunctor *filter) {
    index.setEf(ef);
    index.setAllowEfBelowK(allow);
    return index.searchKnn(q, k, filter);
}

void check(hnswlib::HierarchicalNSW<float> &index, const std::vector<float> &query, hnswlib::BaseFilterFunctor *filter,
           unsigned int divisor) {
    for (size_t j = 0; j < nq; j++) {
        const void *q = query.data() + j * d;

        // ef >= k: unchanged results.
        const size_t ks[] = {1, 10, 100};
        for (size_t k : ks) {
            const size_t efs[] = {k, 2 * k};
            for (size_t ef : efs)
                assert(nearest_first(search(index, q, k, ef, true, filter)) ==
                       nearest_first(search(index, q, k, ef, false, filter)));
        }

        // ef < k: k results, nearest first, whose first ef are the default results with k = ef.
        const size_t pairs[][2] = {{100, 10}, {1000, 50}, {1000, 200}};
        for (size_t p = 0; p < 3; p++) {
            const size_t k = pairs[p][0], ef = pairs[p][1];
            std::vector<std::pair<float, idx_t>> got = nearest_first(search(index, q, k, ef, true, filter));
            std::vector<std::pair<float, idx_t>> walk = nearest_first(search(index, q, ef, ef, false, filter));
            assert(got.size() > ef && got.size() <= k);
            assert(walk.size() == ef);
            for (size_t i = 0; i < ef; i++)
                assert(got[i].first == walk[i].first);
            for (size_t i = 0; i < got.size(); i++) {
                assert(i == 0 || got[i - 1].first <= got[i].first);
                assert(got[i].second % divisor == 0);
                assert(!index.isMarkedDeleted(index.label_lookup_[got[i].second]));
                assert(got[i].first == index.fstdistfunc_(q, index.getDataByInternalId(index.label_lookup_[got[i].second]),
                                                          index.dist_func_param_));
            }
        }

        // ef = 0 is raised to 1.
        assert(!search(index, q, 10, 0, true, filter).empty());
    }
}

}  // namespace

int main() {
    std::mt19937 rng(47);
    std::uniform_real_distribution<> distrib;
    std::vector<float> data(n * d), query(nq * d);
    for (size_t i = 0; i < data.size(); i++)
        data[i] = distrib(rng);
    for (size_t i = 0; i < query.size(); i++)
        query[i] = distrib(rng);

    hnswlib::L2Space space(d);
    hnswlib::HierarchicalNSW<float> index(&space, n, 16, 200, 100);
    for (size_t i = 0; i < n; i++)
        index.addPoint(data.data() + i * d, i);

    check(index, query, nullptr, 1);
    std::cout << "Bare-bone search: ok" << std::endl;

    PickDivisibleIds filter(2);
    check(index, query, &filter, 2);
    std::cout << "Filtered search: ok" << std::endl;

    for (size_t i = 0; i < n; i += 5)
        index.markDelete(i);
    check(index, query, nullptr, 1);
    check(index, query, &filter, 2);
    std::cout << "Search with deleted elements: ok" << std::endl;

    std::cout << "All ef below k tests passed" << std::endl;
    return 0;
}
