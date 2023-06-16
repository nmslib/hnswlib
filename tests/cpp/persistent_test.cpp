#include "../../hnswlib/hnswlib.h"

#include <assert.h>

#include <vector>
#include <iostream>

namespace {

using idx_t = hnswlib::labeltype;

void testPersistentIndex() {
    int d = 1;
    idx_t n = 6;
    idx_t nq = 10;
    size_t k = 10;

    std::vector<float> data(n * d);
    std::vector<float> query(nq * d);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;


    for (idx_t i = 0; i < n * d; i++) {
        data[i] = distrib(rng);
    }
    for (idx_t i = 0; i < nq * d; ++i) {
        query[i] = distrib(rng);
    }

    hnswlib::InnerProductSpace space(d);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, 2 * n, 16, 200, 100, false, false, true, "test.bin", 10);

    for (size_t i = 0; i < n; i++) {
        alg_hnsw->addPoint(data.data() + d * i, i);
    }

    // start a timer
    auto start = std::chrono::high_resolution_clock::now();
    alg_hnsw->persistDirty();
    // stop timer and print time
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "persistDirty() took " << elapsed.count() << " seconds" << std::endl;

    hnswlib::HierarchicalNSW<float>* alg_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, "test.bin", false, 2 * n, false, false, true, 100);

    // Check that all data is the same
    for (size_t i = 0; i < n; i++) {
        std::vector<float> actual = alg_hnsw2->template getDataByLabel<float>(i);
        for (size_t j = 0; j < d; j++) {
            // print got and expected
            std::cout << "actual: " << actual[j] << " expected: " << data[d * i + j] << std::endl;
            // Check that abs difference is less than 1e-6
            assert(std::abs(actual[j] - data[d * i + j]) < 1e-6);
        }
    }

    // Compare to in-memory index
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * d;
        // auto gd = alg_hnsw->searchKnn(p, k);
        // std::cout << "gd.size(): " << gd.size() << std::endl;
        auto res = alg_hnsw2->searchKnn(p, k);
        // std::cout << "res.size(): " << res.size() << std::endl;
        // assert(gd.size() == res.size());
        // int missed = 0;
        // for (size_t i = 0; i < gd.size(); i++) {
        //     std::cout << "gd.top().first: " << gd.top().first << " res.top().first: " << res.top().first << std::endl;
        //     assert(std::abs(gd.top().first - res.top().first) < 1e-6);
        //     assert(gd.top().second == res.top().second);
        //     gd.pop();
        //     res.pop();
        // }
    }

    delete alg_hnsw;
}
}

int main() {
    std::cout << "Testing ..." << std::endl;
    testPersistentIndex();
    std::cout << "Test testPersistentIndex ok" << std::endl;
    return 0;
}


