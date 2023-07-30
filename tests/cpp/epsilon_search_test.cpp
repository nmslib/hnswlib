#include "assert.h"
#include "../../hnswlib/hnswlib.h"

typedef unsigned int docidtype;
typedef float dist_t;

int main() {
    int dim = 16;               // Dimension of the elements
    int max_elements = 10000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    int num_quries = 50;
    int epsilon = 2.0;          // Distance to query

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::BruteforceSearch<dist_t>* alg_brute = new hnswlib::BruteforceSearch<dist_t>(&space, max_elements);
    hnswlib::HierarchicalNSW<dist_t>* alg_hnsw = new hnswlib::HierarchicalNSW<dist_t>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;

    size_t data_point_size = space.get_data_size();
    char* data = new char[data_point_size * max_elements];
    for (int i = 0; i < max_elements; i++) {
        char* point_data = data + i * data_point_size;
        for (int j = 0; j < dim; j++) {
            char* vec_data = point_data + j * sizeof(float);
            float value = distrib_real(rng);
            *(float*)vec_data = value;
        }
    }

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        hnswlib::labeltype label = i;
        char* point_data = data + i * data_point_size;
        alg_hnsw->addPoint(point_data, label);
        alg_brute->addPoint(point_data, label);
    }

    // Query random vectors
    for (int i = 0; i < num_quries; i++) {
        char* query_data = new char[data_point_size];
        for (int j = 0; j < dim; j++) {
            size_t offset = j * sizeof(float);
            char* vec_data = query_data + offset;
            float value = distrib_real(rng);
            *(float*)vec_data = value;
        }
        hnswlib::EpsilonSearchStopCondition<dist_t> stop_condition(epsilon);
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result_hnsw =
            alg_hnsw->searchStopCondition(query_data, max_elements, nullptr, &stop_condition);
        
        // check that returned results are in epsilon region
        size_t num_vectors = result_hnsw.size();
        std::unordered_set<hnswlib::labeltype> hnsw_labels;
        while (!result_hnsw.empty()) {
            float dist = result_hnsw.top().first;
            hnswlib::labeltype label = result_hnsw.top().second;
            hnsw_labels.insert(label);
            assert(dist <= epsilon);
            result_hnsw.pop();
        }
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result_brute =
            alg_brute->searchKnn(query_data, max_elements);
        
        // check recall
        std::unordered_set<hnswlib::labeltype> gt_labels;
        while (!result_brute.empty()) {
            float dist = result_brute.top().first;
            hnswlib::labeltype label = result_brute.top().second;
            if (dist < epsilon) {
                gt_labels.insert(label);
            }
            result_brute.pop();
        }
        float correct = 0;
        for (const auto& hnsw_label: hnsw_labels) {
            if (gt_labels.find(hnsw_label) != gt_labels.end()) {
                correct += 1;
            }
        }
        float recall = correct / gt_labels.size();
        assert(recall > 0.99);
        delete[] query_data;
    }

    delete[] data;
    delete alg_brute;
    delete alg_hnsw;
    return 0;
}
