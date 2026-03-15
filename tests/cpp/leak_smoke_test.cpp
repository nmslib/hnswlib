#include "assert.h"
#include "../../hnswlib/hnswlib.h"
#include <random>
#include <vector>

int main() {
    int dim = 16;
    int max_elements = 1000;
    int M = 16;
    int ef_construction = 200;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distrib(0.0f, 1.0f);

    std::vector<float> data(dim * max_elements);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = distrib(rng);
    }

    for (int iter = 0; iter < 5; iter++) {
        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(
            &space, max_elements, M, ef_construction, 42 + iter);

        for (int i = 0; i < max_elements; i++) {
            alg_hnsw->addPoint(data.data() + (i * dim), i);
        }

        for (int i = 0; i < 50; i++) {
            auto result = alg_hnsw->searchKnn(data.data() + (i * dim), 10);
            assert(result.size() == 10);
        }

        delete alg_hnsw;
    }

    return 0;
}
