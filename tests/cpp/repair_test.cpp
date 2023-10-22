#include "../../hnswlib/hnswlib.h"
#include <thread>


bool is_indegree_ok(hnswlib::HierarchicalNSW<float>* alg_hnsw) {
    bool is_ok_flag = true;
    std::vector<int> indegree(alg_hnsw->cur_element_count);

    for (int level = alg_hnsw->maxlevel_; level >=0 ; level--) {
        std::fill(indegree.begin(), indegree.end(), 0);
        int num_elements = 0;
        // calculate in-degree
        for (int internal_id = 0; internal_id < alg_hnsw->cur_element_count; internal_id++) {
            int element_level = alg_hnsw->element_levels_[internal_id];
            if (element_level < level) {
                continue;
            }
            std::vector<hnswlib::tableint> neis = alg_hnsw->getConnectionsWithLock(internal_id, level);
            for (hnswlib::tableint nei : neis) {
                indegree[nei] += 1;
            }
            num_elements += 1;
        }
        // skip levels with 1 element
        if (num_elements <= 1) {
            continue;
        }

        // check in-degree
        for (int internal_id = 0; internal_id < alg_hnsw->cur_element_count; internal_id++) {
            int element_level = alg_hnsw->element_levels_[internal_id];
            if (element_level < level) {
                continue;
            }
            if (indegree[internal_id] == 0) {
                std::cout << "zero in-degree node found, level=" << level << " id=" << internal_id << "\n" << std::flush;
                is_ok_flag = false;
            }
        }
    }

    return is_ok_flag;
}


int main() {
    int dim = 4;                // Dimension of the elements
    int n = 100;                // Maximum number of elements, should be known beforehand
    int M = 8;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int num_test_iter = 50;

    int test_id = 0;

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    while (test_id < num_test_iter) {
        // Initing index
        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, n * 3, M, ef_construction, 100, true);

        // Generate random data
        float* data = new float[dim * n];
        for (int i = 0; i < dim * n; i++) {
            data[i] = distrib_real(rng);
        }

        // Add data to index
        for (int i = 0; i < n; i++) {
            alg_hnsw->addPoint(data + i * dim, i, true);
        }
        std::cout << test_id << " Index is ready\n";

        std::vector<std::thread> threads;
        
        // mix new inserts with modifications (50% of operations are new)
        for(int i = 0; i < n; i += 10) {
            threads.emplace_back([alg_hnsw, data, i, dim, &distrib_real, &rng]() {
                for(auto j = 0; j < 10; j++) {
                    auto actual_index = i + j;
                    auto id = ( actual_index % 2 != 0) ? actual_index + 10000 : actual_index;
                    std::vector<float> values;
                    for (size_t j = 0; j < dim; j++) {
                        values.push_back(distrib_real(rng) + 0.01);
                    }
                    alg_hnsw->addPoint(values.data(), id, true);
                }
            });
        }

        // add repair method to check concurrency
        threads.emplace_back([alg_hnsw] {
            alg_hnsw->repair_zero_indegree();
        });

        for(auto& t: threads) {
            t.join();
        }

        bool is_ok_before_flag = is_indegree_ok(alg_hnsw);
        // fix in-degree if it is broken
        if (!is_ok_before_flag) {
            alg_hnsw->repair_zero_indegree();
        }
        bool is_ok_after_flag = is_indegree_ok(alg_hnsw);
        assert(is_ok_after_flag);
        test_id += 1;

        delete[] data;
        delete alg_hnsw;
    }
    return 0;
}
