#include "../hnswlib/hnswlib.h"
#include <thread>

int main() {
    int d = 16;
    int max_elements = 100;

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;

    hnswlib::L2Space space(d);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements);

    int num_threads = 40;
    int num_ids = 1;

    int num_iterations = 10;
    int start_id = 0;

    while (true) {
        std::uniform_int_distribution<> distrib_int(start_id, start_id + num_ids - 1);
        std::vector<std::thread> threads;
        for (size_t thread_id = 0; thread_id < num_threads; thread_id++) {
            threads.push_back(
                std::thread(
                    [&] {
                        for (int iter = 0; iter < num_iterations; iter++) {
                            std::vector<float> data(d);
                            int id = distrib_int(rng);
                            //std::cout << id << std::endl;
                            for (int i = 0; i < d; i++) {
                                data[i] = distrib_real(rng);
                            }
                            alg_hnsw->addPoint(data.data(), id);
                        }
                    }
                )
            );
        }
        for (auto &thread : threads) {
            thread.join();
        }
        //std::cout << alg_hnsw->cur_element_count << std::endl;
        if (alg_hnsw->cur_element_count > max_elements - num_ids) {
            //std::cout << "Exit" << std::endl;
            break;
        }
        start_id += num_ids;
    }
    
    return 0;
}
