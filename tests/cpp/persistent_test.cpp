#include "../../hnswlib/hnswlib.h"

#include <assert.h>

#include <vector>
#include <iostream>

namespace {

using idx_t = hnswlib::labeltype;

void testPersistentIndex() {
    int d = 1536;
    idx_t n = 100;
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
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, 2 * n, 16, 200, 100, false, false, true, ".", 10);

    for (size_t i = 0; i < n; i++) {
        alg_hnsw->addPoint(data.data() + d * i, i);
        // if (i % 10 == 0)
            // alg_hnsw->persistDirty();
    }
    alg_hnsw->persistDirty();

    hnswlib::HierarchicalNSW<float>* alg_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, ".", false, 2 * n, false, false, true, 100);

    // Check that all data is the same
    for (size_t i = 0; i < n; i++) {
        std::vector<float> actual = alg_hnsw2->template getDataByLabel<float>(i);
        for (size_t j = 0; j < d; j++) {
            // Check that abs difference is less than 1e-6
            if (!(std::abs(actual[j] - data[d * i + j]) < 1e-6)){
                std::cout << "actual: " << actual[j] << " expected: " << data[d * i + j] << std::endl;
            }
            assert(std::abs(actual[j] - data[d * i + j]) < 1e-6);
        }
    }

    // Compare to in-memory index
    for (size_t j = 0; j < nq; ++j) {
        const void* p = query.data() + j * d;
        auto gd = alg_hnsw->searchKnn(p, k);
        auto res = alg_hnsw2->searchKnn(p, k);
        assert(gd.size() == res.size());
        int missed = 0;
        for (size_t i = 0; i < gd.size(); i++) {
            assert(std::abs(gd.top().first - res.top().first) < 1e-6);
            assert(gd.top().second == res.top().second);
            gd.pop();
            res.pop();
        }
    }

    delete alg_hnsw;
}


void testResizePersistentIndex() {
    int d = 1536;
    idx_t n = 400;
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
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, n / 4, 16, 200, 100, false, false, true, ".", 10);

    // Add a quarter of the data
    for (size_t i = 0; i < n / 4; i++) {
        alg_hnsw->addPoint(data.data() + d * i, i);
        if (i % 9 == 0)
            alg_hnsw->persistDirty();
    }
    alg_hnsw->persistDirty();

    // Resize index and another quarter of the data
    alg_hnsw->resizeIndex(n / 2);
    for (size_t i = n / 4; i < n / 2; i++) {
        alg_hnsw->addPoint(data.data() + d * i, i);
        if (i % 9 == 0)
            alg_hnsw->persistDirty();
    }
    alg_hnsw->persistDirty();

    // Load the resized index with n / 2 elements
    hnswlib::HierarchicalNSW<float>* alg_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, ".", false, n / 2, false, false, true, 100);    
    // Check that the added half of the data is the same
    for (size_t i = 0; i < n / 2; i++) {
        std::vector<float> actual = alg_hnsw2->template getDataByLabel<float>(i);
        for (size_t j = 0; j < d; j++) {
            assert(std::abs(actual[j] - data[d * i + j]) < 1e-6);
        }
    }

    // Resize the index and add all the data
    alg_hnsw2->resizeIndex(n);
    for (size_t i = n / 2; i < n; i++) {
        alg_hnsw2->addPoint(data.data() + d * i, i);
        if (i % 9 == 0)
            alg_hnsw2->persistDirty();
    }
    alg_hnsw2->persistDirty();
    
    // Load the resized index with n elements
    hnswlib::HierarchicalNSW<float>* alg_hnsw3 = new hnswlib::HierarchicalNSW<float>(&space, ".", false, n, false, false, true, 100);
    // Check that all the data is the same
    for (size_t i = 0; i < n; i++) {
        std::vector<float> actual = alg_hnsw3->template getDataByLabel<float>(i);
        for (size_t j = 0; j < d; j++) {
            assert(std::abs(actual[j] - data[d * i + j]) < 1e-6);
        }
    }
    
    delete alg_hnsw;
    delete alg_hnsw2;
    delete alg_hnsw3;
}


void testAddUpdatePersistentIndex() {
    int d = 1536;
    idx_t n = 100;
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
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, n, 16, 200, 100, false, false, true, ".", 10);

    for (size_t i = 0; i < n; i++) {
        alg_hnsw->addPoint(data.data() + d * i, i);
        if (i % 10 == 0)
            alg_hnsw->persistDirty();
    }
    alg_hnsw->persistDirty();

    // Generate random updates to the index
    float update_prob = 0.2;
    for (size_t i = 0; i < n; i++) {
        if (distrib(rng) < update_prob) {
            std::vector<float> new_data(d);
            for (size_t j = 0; j < d; j++) {
                new_data[j] = distrib(rng);
            }
            alg_hnsw->addPoint(new_data.data(), i);
            if (i % 10 == 0)
                alg_hnsw->persistDirty();
        }
    }
    alg_hnsw->persistDirty();

    // Load the index with n elements
    hnswlib::HierarchicalNSW<float>* alg_hnsw2 = new hnswlib::HierarchicalNSW<float>(&space, ".", false, n, false, false, true, 100);

    // Check that all the data is the same
    for (size_t i = 0; i < n; i++) {
        std::vector<float> actual = alg_hnsw2->template getDataByLabel<float>(i);
        std::vector<float> expected = alg_hnsw->template getDataByLabel<float>(i);
        for (size_t j = 0; j < d; j++) {
            assert(std::abs(actual[j] - expected[j]) < 1e-6);
        }
    }

}
}

int main() {
    std::cout << "Testing ..." << std::endl;
    testPersistentIndex();
    std::cout << "Test testPersistentIndex ok" << std::endl;
    testResizePersistentIndex();
    std::cout << "Test testResizePersistentIndex ok" << std::endl;
    return 0;
}


