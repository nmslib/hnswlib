#include "../../hnswlib/hnswlib.h"

int main() {
    int dim = 1024;               // Dimension of the elements
    int max_elements = 10;   // Maximum number of elements, should be known beforehand
    int M = 1024;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    float nonzero_prob = 0.1;

    // Initing index
    hnswlib::SparseCosSpace space{};
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Generate random data
    // Generate number with 10% probability
    std::mt19937 rng;
    rng.seed(74);
    std::uniform_real_distribution<> distrib_real;
    hnswlib::SparseVector* vectors = new hnswlib::SparseVector[max_elements];
    float* temp_array = new float[dim];
    for (int i = 0; i < max_elements; i++) {
        memset(temp_array, 0, dim * sizeof(float));
        size_t size = 0;
        for (int j = 0; j < dim; j++) {
            if (distrib_real(rng) < nonzero_prob) {
                float obj = distrib_real(rng);
                if (obj == 0) continue;
                temp_array[j] = obj;
                size++;
            }
        }

        vectors[i].entries = new hnswlib::SparseVectorEntry[size];
        size_t spv_count = 0;
        for (unsigned int j = 0; j < dim; j++) {
            if (temp_array[j] != 0) {
                vectors[i].entries[spv_count] = {j, temp_array[j]};
                spv_count++;
            }
        }
        vectors[i].num_entries = spv_count;
    }

    delete[] temp_array;

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(vectors + i, i);
    }

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(vectors + i, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";

    // Serialize index
    std::string hnsw_path = "hnsw.bin";
    alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;

    // Deserialize index and check recall
    alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(vectors + i, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    recall = (float)correct / max_elements;
    std::cout << "Recall of deserialized index: " << recall << "\n";

    for (int i = 0; i < max_elements; i++) {
        delete[] vectors[i].entries;
    }
    delete[] vectors;
    delete alg_hnsw;
    return 0;
}
