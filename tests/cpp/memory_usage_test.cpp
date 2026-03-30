#include "assert.h"
#include "../../hnswlib/hnswlib.h"
#include <random>
#include <vector>

int main() {
    int dim = 16;
    int max_elements = 1000;
    int M = 16;
    int ef_construction = 200;

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction, 42);

    size_t mem_before = alg_hnsw->getMemoryUsage();
    size_t file_before = alg_hnsw->indexFileSize();
    assert(mem_before > 0);
    assert(mem_before >= file_before);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distrib(0.0f, 1.0f);
    std::vector<float> data(dim * max_elements);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = distrib(rng);
    }
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data.data() + (i * dim), i);
    }

    size_t mem_after = alg_hnsw->getMemoryUsage();
    size_t file_after = alg_hnsw->indexFileSize();
    assert(mem_after > 0);
    assert(mem_after >= mem_before);
    assert(mem_after >= file_after);

    delete alg_hnsw;
    return 0;
}
