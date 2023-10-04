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

    int num_quries = 100;
    float epsilon = 1.0;                 // Squared distance to query
    int max_candidates = max_elements;   // Upper bound on the number of returned elements in the epsilon region
    int min_candidates = 2000;           // Minimum number of candidates to search in the epsilon region
                                         // this parameter is similar to ef

    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::BruteforceSearch<dist_t>* alg_brute = new hnswlib::BruteforceSearch<dist_t>(&space, max_elements);
    hnswlib::HierarchicalNSW<dist_t>* alg_hnsw = new hnswlib::HierarchicalNSW<dist_t>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;

    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // Add data to index
    std::cout << "Building index ...\n";
    for (int i = 0; i < max_elements; i++) {
        hnswlib::labeltype label = i;
        float* point_data = data + i * dim;
        alg_hnsw->addPoint(point_data, label);
        alg_brute->addPoint(point_data, label);
    }
    std::cout << "Index is ready\n";

    // Query random vectors
    for (int i = 0; i < num_quries; i++) {
        float* query_data = new float[dim];
        for (int j = 0; j < dim; j++) {
            query_data[j] = distrib_real(rng);
        }
        hnswlib::EpsilonSearchStopCondition<dist_t> stop_condition(epsilon, min_candidates);
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result_hnsw =
            alg_hnsw->searchStopCondition(query_data, max_candidates, nullptr, &stop_condition);
        
        // check that returned results are in epsilon region
        size_t num_vectors = result_hnsw.size();
        std::unordered_set<hnswlib::labeltype> hnsw_labels;
        while (!result_hnsw.empty()) {
            float dist = result_hnsw.top().first;
            hnswlib::labeltype label = result_hnsw.top().second;
            hnsw_labels.insert(label);
            assert(dist >=0 && dist <= epsilon);
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
        float recall = correct / std::max((size_t)1, gt_labels.size());
        assert(recall > 0.97);
        delete[] query_data;
    }
    std::cout << "Recall is OK\n";

    // Query the elements for themselves and check that query can be found
    float epsilon_small = 0.0001f;
    int min_candidates_small = 500;
    for (size_t i = 0; i < max_elements; i++) {
        hnswlib::EpsilonSearchStopCondition<dist_t> stop_condition(epsilon_small, min_candidates_small);
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = 
            alg_hnsw->searchStopCondition(alg_hnsw->getDataByInternalId(i), max_candidates, nullptr, &stop_condition);
        size_t num_vectors = result.size();
        // get closest distance
        float dist = -1;
        while (!result.empty()) {
            dist = result.top().first;
            result.pop();
        }
        assert(dist == 0);
    }
    std::cout << "Small epsilon search is finished\n";

    delete[] data;
    delete alg_brute;
    delete alg_hnsw;
    return 0;
}
