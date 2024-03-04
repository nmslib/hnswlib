#include "../../hnswlib/hnswlib.h"
#include <thread>
#include <chrono>
#include <iomanip>

int main() {
    std::cout << "Running replace same label back-to-back test" << std::endl;
    int d = 16;
    int num_elements = 50;
    int max_elements = 2 * num_elements;

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    std::uniform_int_distribution<> distrib_int(0, max_elements - 1);

    hnswlib::InnerProductSpace space(d);

    std::cout << "Generating random data" << std::endl;
    std::cout << "Initial batch size: " << max_elements << std::endl;
    float* initial_batch = new float[d * max_elements];
    for (int i = 0; i < d * max_elements; i++) {
        initial_batch[i] = distrib_real(rng);
    }

    std::cout << "Update batch size: " << num_elements << std::endl;
    float* update_batch = new float[d * num_elements];
    for (int i = 0; i < d * num_elements; i++) {
        update_batch[i] = distrib_real(rng);
    }

    std::cout << "Building index" << std::endl;
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, 1024, 16, 200, 456, true);

    std::vector<int> labels;

    std::cout << "Adding initial batch" << std::endl;
    for(int row = 0; row < max_elements; row++) {
        int label = distrib_int(rng);
        labels.push_back(label);
        alg_hnsw->addPoint((void*)(initial_batch + d * row), label, true);
    };

    std::cout << "Deleting half of the initial batch" << std::endl;
    for (int i = 0; i < labels.size() / 2; i++) {
        if(!alg_hnsw->isMarkedDeleted(alg_hnsw->label_lookup_[labels[i]]))
            alg_hnsw->markDelete(labels[i]);
    }
    
    std::cout << "Updating index size" << std::endl;
    size_t curr_ele_count = alg_hnsw->getCurrentElementCount();
    if(curr_ele_count + max_elements > alg_hnsw->getMaxElements()) {
            alg_hnsw->resizeIndex((curr_ele_count + max_elements) * 1.3);
    }

    std::cout << "Adding update batch" << std::endl;
    for(int row; row < num_elements; row++) {
        alg_hnsw->addPoint((void*)(update_batch + d * row), 42, true);
    };


    std::cout << "Searching for 10 nearest neighbors" << std::endl;
    auto results = alg_hnsw->searchKnnCloserFirst((void*)(initial_batch), 10);
    
    // check if the search results contain duplicate labels
    std::cout << "Checking search results for duplicate labels" << std::endl;
    std::unordered_set<int> labels_set;
    for (int i = 0; i < results.size(); i++) {
        labels_set.insert(results[i].second);
    }
    if (labels_set.size() != 10) {
        std::cout << "Search results contain duplicate labels" << std::endl;
        throw std::runtime_error("Search results contain duplicate labels");
    }

    delete[] initial_batch;
    delete[] update_batch;
    delete alg_hnsw;
    return 0;
}
