#include "../../hnswlib/hnswlib.h"
#include <thread>
#include <chrono>
#include <atomic>


int main() {
    std::cout << "Running multithread load test" << std::endl;
    int d = 16;
    int max_elements = 1000;

    hnswlib::L2Space space(d);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, 2 * max_elements);
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw_holder(alg_hnsw);

    int num_threads = 40;

    std::mt19937 seeding_rng(314159265);

    std::cout << "Building index" << std::endl;
    int num_labels = 10;

    int num_iterations = 10;
    int start_label = 0;

    // run threads that will add elements to the index
    // about 7 threads (the number depends on num_threads and num_labels)
    // will add/update element with the same label simultaneously
    while (true) {
        // add elements by batches
        std::vector<std::thread> threads;
        for (size_t thread_id = 0; thread_id < num_threads; thread_id++) {
            unsigned int rng_seed = seeding_rng();
            threads.push_back(
                std::thread(
                    [=] {
                        std::uniform_int_distribution<> distrib_int(
                            start_label, start_label + num_labels - 1);
                        std::uniform_real_distribution<> distrib_real;
                        std::mt19937 rng(rng_seed);
                        for (int iter = 0; iter < num_iterations; iter++) {
                            std::vector<float> data(d);
                            hnswlib::labeltype label = distrib_int(rng);
                            for (int i = 0; i < d; i++) {
                                data[i] = distrib_real(rng);
                            }
                            alg_hnsw->addPoint(data.data(), label);
                        }
                    }
                )
            );
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (alg_hnsw->cur_element_count > max_elements - num_labels) {
            break;
        }
        start_label += num_labels;
    }

    // insert remaining elements if needed
    std::uniform_real_distribution<> main_distrib_real;
    std::mt19937 main_rng(seeding_rng());
    for (hnswlib::labeltype label = 0; label < max_elements; label++) {
        auto search = alg_hnsw->label_lookup_.find(label);
        if (search == alg_hnsw->label_lookup_.end()) {
            std::cout << "Adding " << label << std::endl;
            std::vector<float> data(d);
            for (int i = 0; i < d; i++) {
                data[i] = main_distrib_real(main_rng);
            }
            alg_hnsw->addPoint(data.data(), label);
        }
    }

    std::cout << "Index is created" << std::endl;

    std::atomic<bool> stop_threads{false};
    std::vector<std::thread> threads;

    // create threads that will do markDeleted and unmarkDeleted of random elements
    // each thread works with specific range of labels
    std::cout << "Starting markDeleted and unmarkDeleted threads" << std::endl;
    num_threads = 20;
    int chunk_size = max_elements / num_threads;
    for (size_t thread_id = 0; thread_id < num_threads; thread_id++) {
        unsigned int rng_seed = seeding_rng();
        threads.push_back(
            std::thread(
                [=, &stop_threads] {
                    std::mt19937 rng(rng_seed);
                    std::uniform_int_distribution<> distrib_int(0, chunk_size - 1);
                    int start_id = thread_id * chunk_size;
                    std::vector<bool> marked_deleted(chunk_size);
                    while (!stop_threads) {
                        int id = distrib_int(rng);
                        hnswlib::labeltype label = start_id + id;
                        if (marked_deleted[id]) {
                            alg_hnsw->unmarkDelete(label);
                            marked_deleted[id] = false;
                        } else {
                            alg_hnsw->markDelete(label);
                            marked_deleted[id] = true;
                        }
                    }
                }
            )
        );
    }

    // create threads that will add and update random elements
    std::cout << "Starting add and update elements threads" << std::endl;
    num_threads = 20;
    for (size_t thread_id = 0; thread_id < num_threads; thread_id++) {
        unsigned int rng_seed = seeding_rng();
        threads.push_back(
            std::thread(
                [=, &stop_threads] {
                    std::mt19937 rng(rng_seed);
                    std::uniform_int_distribution<> distrib_int_add(
                        max_elements, 2 * max_elements - 1);
                    std::uniform_real_distribution<> distrib_real;
                    std::vector<float> data(d);
                    while (!stop_threads) {
                        hnswlib::labeltype label = distrib_int_add(rng);
                        for (int i = 0; i < d; i++) {
                            data[i] = distrib_real(rng);
                        }
                        alg_hnsw->addPoint(data.data(), label);
                        std::vector<float> data = alg_hnsw->getDataByLabel<float>(label);
                        float max_val = *max_element(data.begin(), data.end());
                        // never happens but prevents compiler from deleting unused code
                        if (max_val > 10) {
                            HNSWLIB_THROW_RUNTIME_ERROR("Unexpected value in data");
                        }
                    }
                }
            )
        );
    }

    std::cout << "Sleep and continue operations with index" << std::endl;
    int sleep_ms = 60 * 1000;
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    std::cout << "Stopping threads and waiting for them to join" << std::endl;
    stop_threads = true;
    for (auto &thread : threads) {
        thread.join();
    }

    std::cout << "Finish" << std::endl;
    return 0;
}
