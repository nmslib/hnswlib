#include "../../hnswlib/hnswlib.h"

#include <chrono>

void normalize_data_for_inner_product(float* data, int dim, int max_elements) {
    for (int i = 0; i < max_elements; i++) {
        double norm_sq = 0.0;

        for (int j = 0; j < dim; j++) {
            float value = data[i * dim + j];
            norm_sq += static_cast<double>(value) * value;
        }

        double norm = std::sqrt(norm_sq);

        if (norm > 0.0) {
            for (int j = 0; j < dim; j++) {
                data[i * dim + j] =
                    static_cast<float>(data[i * dim + j] / norm);
            }
        }
    }
}

template <typename SpaceType>
void run_test(const std::string& name, SpaceType& space, float* data, int dim, int max_elements) {
    int M = 16;
    int ef_construction = 200;

    std::cout << "Running " << name << " test...\n";

    // Initing index
    hnswlib::HierarchicalNSW<float>* alg_hnsw =
        new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    auto build_start = std::chrono::high_resolution_clock::now();

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }

    auto build_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> build_seconds = build_end - build_start;
    auto build_ms =
        std::chrono::duration<double, std::milli>(build_end - build_start);

    std::cout << name << " index build time: "
              << build_seconds.count() << " seconds ("
              << build_ms.count() << " ms)\n";

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
            alg_hnsw->searchKnn(data + i * dim, 1);

        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }

    float recall = correct / max_elements;
    std::cout << name << " Recall: " << recall << "\n";

    // Serialize index
    std::string hnsw_path = name + "_hnsw.bin";
    alg_hnsw->saveIndex(hnsw_path);
    delete alg_hnsw;

    // Deserialize index and check recall
    alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);

    correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
            alg_hnsw->searchKnn(data + i * dim, 1);

        hnswlib::labeltype label = result.top().second;
        if (label == i)  correct++;
    }

    recall = correct / max_elements;
    std::cout << name << " Recall of deserialized index: "
              << recall << "\n\n";

    delete alg_hnsw;
}

int main() {
    int dim = 16;
    int max_elements = 10000;

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<float> distrib_real(0.0f, 1.0f);

    float* data_l2 = new float[dim * max_elements];
    float* data_ip = new float[dim * max_elements];

    for (int i = 0; i < dim * max_elements; i++) {
        float value = distrib_real(rng);
        data_l2[i] = value;
        data_ip[i] = value;
    }

    hnswlib::L2Space l2_space(dim);
    run_test("l2", l2_space, data_l2, dim, max_elements);

    normalize_data_for_inner_product(data_ip, dim, max_elements);

    hnswlib::InnerProductSpace ip_space(dim);
    run_test("ip_normalized", ip_space, data_ip, dim, max_elements);

    delete[] data_l2;
    delete[] data_ip;

    return 0;
}
