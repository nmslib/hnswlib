// This is a test file for testing the filtering feature

#include "../hnswlib/hnswlib.h"

#include <assert.h>

#include <vector>
#include <iostream>

namespace
{

using idx_t = hnswlib::labeltype;

bool pickIdsDivisibleByThree(unsigned int ep_id) {
    return ep_id % 3 == 0;
}

bool pickIdsDivisibleBySeven(unsigned int ep_id) {
    return ep_id % 7 == 0;
}

template<typename filter_func_t>
void test(filter_func_t filter_func, size_t div_num) {
    int d = 4;
    idx_t n = 100;
    idx_t nq = 10;
    size_t k = 10;
   
    std::vector<float> data(n * d);
    std::vector<float> query(nq * d);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;

    for (idx_t i = 0; i < n * d; ++i) {
        data[i] = distrib(rng);
    }
    for (idx_t i = 0; i < nq * d; ++i) {
        query[i] = distrib(rng);
    }
      

    hnswlib::L2Space space(d);
    hnswlib::AlgorithmInterface<float>* alg_brute  = new hnswlib::BruteforceSearch<float,hnswlib::FILTERFUNC>(&space, 2 * n);
    hnswlib::AlgorithmInterface<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float,hnswlib::FILTERFUNC>(&space, 2 * n);

    for (size_t i = 0; i < n; ++i) {
        alg_brute->addPoint(data.data() + d * i, i);
        alg_hnsw->addPoint(data.data() + d * i, i);
    }

    // test searchKnnCloserFirst of BruteforceSearch with filtering
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * d;
        auto gd = alg_brute->searchKnn(p, k, filter_func);
        auto res = alg_brute->searchKnnCloserFirst(p, k, filter_func);
        assert(gd.size() == res.size());
        size_t t = gd.size();
        while (!gd.empty()) {
            assert(gd.top() == res[--t]);
            assert((gd.top().second % div_num) == 0);
            gd.pop();
        }
    }

    // test searchKnnCloserFirst of hnsw with filtering
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * d;
        auto gd = alg_hnsw->searchKnn(p, k, filter_func);
        auto res = alg_hnsw->searchKnnCloserFirst(p, k, filter_func);
        assert(gd.size() == res.size());
        size_t t = gd.size();
        while (!gd.empty()) {
            assert(gd.top() == res[--t]);
            assert((gd.top().second % div_num) == 0);
            gd.pop();
        }
    }
    
    delete alg_brute;
    delete alg_hnsw;
}

} // namespace

int main() {
    std::cout << "Testing ..." << std::endl;
    test(pickIdsDivisibleByThree, 3);
    test(pickIdsDivisibleBySeven, 7);
    std::cout << "Test ok" << std::endl;

    return 0;
}
