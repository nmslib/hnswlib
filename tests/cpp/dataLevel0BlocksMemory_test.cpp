#include "../../hnswlib/hnswlib.h"

#include <assert.h>

#include <vector>
#include <iostream>
#include <cstdio>
#include <thread>
#include <chrono>

namespace {

const size_t M = 32;
const size_t ef_construction = 500;
const size_t random_seed = 100;
const bool allow_replace_deleted = false;

const size_t dimension = 1024;
const size_t total_items = 100 * 10000;
const size_t num_query = 500 * 10000;
size_t topk = 10;
const size_t max_thread_num = 48;
const std::string index_path = "./hnsw.index";

std::vector<float> data(total_items * dimension);
std::vector<float> query(num_query * dimension);


void check_knn_closer(hnswlib::AlgorithmInterface<float>* alg_hnsw) {
    for (size_t j = 0; j < num_query; ++j) {
        const void* p = query.data() + j * dimension;
        auto gd = alg_hnsw->searchKnn(p, topk);
        auto res = alg_hnsw->searchKnnCloserFirst(p, topk);
        assert(gd.size() == res.size());
        size_t t = gd.size();
        while (!gd.empty()) {
            assert(gd.top() == res[--t]);
            gd.pop();
        }
    }
    std::cout << "test hnsw search knn closer first success..." << std::endl;
}

void test_compatibility(bool hnsw_first_use_blocks_memory,
                        bool hnsw_second_use_blocks_memory) {

    std::cout << "================== test compatibility ==================" << std::endl;
    hnswlib::L2Space space(dimension);
    hnswlib::AlgorithmInterface<float>* alg_hnsw_first = new hnswlib::HierarchicalNSW<float>(&space, 2 * total_items,
            M, ef_construction, random_seed, allow_replace_deleted, hnsw_first_use_blocks_memory);

    for (size_t i = 0; i < total_items; ++i) {
        alg_hnsw_first->addPoint(data.data() + dimension * i, i);
    }
    check_knn_closer(alg_hnsw_first);

    // save hnsw index
    std::remove(index_path.data());
    alg_hnsw_first->saveIndex(index_path);
    std::cout << "save hnsw(use_small_blocks_memory = " << hnsw_first_use_blocks_memory << ") index success" << std::endl;
    delete alg_hnsw_first;

    // load hnsw index
    hnswlib::AlgorithmInterface<float>* alg_hnsw_second = new hnswlib::HierarchicalNSW<float>(&space, false,
            0, allow_replace_deleted, hnsw_second_use_blocks_memory);
    std::cout << "load hnsw(use_small_blocks_memory = " << hnsw_second_use_blocks_memory << ") index success" << std::endl;
    std::remove(index_path.data());
    check_knn_closer(alg_hnsw_second);

    delete alg_hnsw_second;
}

void test_performace(bool use_small_blocks_memory) {
    if (total_items == 0) {
      return;
    }

    std::cout << "================== test preformace("
              << "dimension: " << dimension
              << ", M: " << M
              << ", ef_construction: " << ef_construction
              << ", topk: " << topk
              << ", use_small_blocks_memory: " << (use_small_blocks_memory ? "ture" : "false" )
              << ") ==================" << std::endl;
    hnswlib::L2Space space(dimension);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, 2 * total_items,
            M, ef_construction, random_seed, allow_replace_deleted, use_small_blocks_memory);

    std::vector<std::thread> threads;
    size_t num_threads = (total_items >= max_thread_num ? max_thread_num : total_items);
    size_t batch_num = (total_items / (num_threads <= 1 ? 1 : (num_threads - 1))) + 1;
    auto start_time = std::chrono::system_clock::now();
    for (size_t idx = 0; idx < total_items; idx += batch_num) {
        size_t start = idx;
        size_t end = std::min(idx + batch_num, total_items);
        threads.push_back(
            std::thread(
                [alg_hnsw, start, end] {
                    for (size_t i = start; i < end; i++) {
                       alg_hnsw->addPoint(data.data() + i * dimension, i);
                    }
                }
            )
        );
    }
    for (auto &thread : threads) {
       thread.join();
    }
    threads.clear();
    auto end_time = std::chrono::system_clock::now();
    double duration_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    double duration_in_seconds = static_cast<double>((std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time)).count()) / 1000.0;
    size_t qps = (duration_in_seconds == 0 ? total_items : total_items / duration_in_seconds);
    double latency = (total_items == 0 ? 0 : duration_in_ms / total_items);
    std::cout << "Start " << num_threads << " thread to add " << total_items << " items to hnsw index, cost "
              << duration_in_seconds << " seconds, qps: " << qps  << ", latency: " << latency << "ms" << std::endl;


    num_threads = (num_query >= max_thread_num ? max_thread_num : num_query);
    batch_num = (num_query / (num_threads <= 1 ? 1 : (num_threads - 1))) + 1;
    start_time = std::chrono::system_clock::now();
    for (size_t idx = 0; idx < num_query; idx += batch_num) {
        size_t start = idx;
        size_t end = std::min(idx + batch_num, num_query);
        threads.push_back(
            std::thread(
                [alg_hnsw, start, end] {
                    for (size_t i = start; i < end; i++) {
                        const void* p = query.data() + i * dimension;
                        auto gd = alg_hnsw->searchKnn(p, topk);
                    }
                }
            )
        );
    }
    for (auto &thread : threads) {
       thread.join();
    }
    threads.clear();
    end_time = std::chrono::system_clock::now();
    duration_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    duration_in_seconds = static_cast<double>((std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time)).count()) / 1000.0;
    qps = (duration_in_seconds == 0 ? num_query : num_query / duration_in_seconds);
    latency = (num_query == 0 ? 0 : duration_in_ms / num_query);
    std::cout << "Start " << num_threads << " thread to exec " << num_query << " searchKnn, cost "
              << duration_in_seconds << " seconds, qps: " << qps  << ", latency: " << latency << "ms" << std::endl;

    delete alg_hnsw;
}

}  // namespace

int main() {

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;

    for (size_t i = 0; i < total_items * dimension; ++i) {
        data[i] = distrib(rng);
    }
    for (size_t i = 0; i < num_query * dimension; ++i) {
        query[i] = distrib(rng);
    }

    test_compatibility(true, true);
    test_compatibility(false, false);
    test_compatibility(true, false);
    test_compatibility(false, true);

    test_performace(true);
    test_performace(false);

    return 0;
}
