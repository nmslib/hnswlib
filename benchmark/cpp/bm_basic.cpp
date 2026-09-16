#include <benchmark/benchmark.h>
#include <vector>
#include <random>

#include "hnswlib/hnswalg.h"

// hnsw build benchmark

void l2_normalize(float* arr, size_t dim) {
    float norm = 0;
    for (size_t i = 0; i < dim; ++i) {
        norm += arr[i] * arr[i];
    }
    norm = std::sqrt(norm);
    for (size_t i = 0; i < dim; ++i) {
        arr[i] /= norm;
    }
}
void l2_normalize_batch(float* arr, size_t dim, size_t batch_size) {
    for(size_t i = 0; i < batch_size; ++i){
        l2_normalize(arr + i*dim, dim);
    }
}
void prepare_data(std::vector< std::vector<float> >& embeddings, size_t dim, size_t x_data_size, bool need_l2_normalize) {
    std::mt19937 rng(42); // same seed to ensure reproducibility
    std::vector<float> datas(x_data_size*dim);
    std::generate(datas.begin(), datas.end(), rng);
    if (need_l2_normalize) {
        l2_normalize_batch(datas.data(), dim, x_data_size);
    }
    for(size_t i=0; i<x_data_size; ++i) {
        auto& emb  = embeddings[i];
        memcpy(emb.data(), datas.data() + i*dim, dim*sizeof(float));
    }
}


static void BM_HnswIPAddPointWholeTimeBench(benchmark::State& state) {
    size_t M = state.range(0);
    size_t ef_construction = state.range(1);
    size_t dim = state.range(2);
    size_t x_data_size = state.range(3);

    std::vector< std::vector<float> > embeddings(x_data_size, std::vector<float>(dim, 0.0f));
    prepare_data(embeddings, dim, x_data_size, true);

    for (auto _: state) {
        auto space = std::make_shared<hnswlib::InnerProductSpace>(dim);
        auto index = std::make_shared<hnswlib::HierarchicalNSW<float>>(space.get(), x_data_size);
        for (size_t i = 0; i < x_data_size; i++) {
            auto& emb = embeddings[i];
            index->addPoint(emb.data(), i);
        }
        benchmark::DoNotOptimize(index);
    }

    state.SetComplexityN(state.range(0)*state.range(1)*state.range(2)*state.range(3));
}

void RegisterHnswBasicBenchmarks() {
    BENCHMARK(BM_HnswIPAddPointWholeTimeBench)
        ->ArgsProduct({
            {16,32},
            {200,400},
            {32, 128},
            {500,5000}
        });
}