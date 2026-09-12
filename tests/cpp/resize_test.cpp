#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <cmath>

#include "hnswlib/hnswlib.h"

// Performs the index resize test.
void TestRandomSelf() {

    constexpr int kDim = 16;
    constexpr int kNumElements = 10000;
    constexpr int kM = 16;
    constexpr int kEfConstruction = 100;
    constexpr int kEfSearch = 20;

    // Set up a random number generator.
    std::mt19937 rng;
    std::uniform_real_distribution<float> distrib_real;

    // Generate random data.
    std::vector<float> data(kNumElements * kDim);
    for (int i = 0; i < kNumElements * kDim; ++i) {
      data[i] = distrib_real(rng);
    }

    // Initialize the HNSW index.
    hnswlib::L2Space space(kDim);
    // Initialize with half the maximum elements.
    auto* alg_hnsw = new hnswlib::HierarchicalNSW<float>(
        &space, kNumElements / 2, kM, kEfConstruction);
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw_holder(alg_hnsw);

    alg_hnsw->setEf(kEfSearch);

    // Add the first half of the data to the index.
    const int first_batch_size = kNumElements / 2;
    std::cout << "Adding first batch of " << first_batch_size << " elements."
              << std::endl;
    for (int i = 0; i < first_batch_size; ++i) {
      alg_hnsw->addPoint(data.data() + (i * kDim), i);
    }

    // Resize the index and add the second batch
    std::cout << "Resizing the index to " << kNumElements << "." << std::endl;
    alg_hnsw->resizeIndex(kNumElements);

    const int second_batch_size = kNumElements - first_batch_size;
    std::cout << "Adding the second batch of " << second_batch_size
              << " elements." << std::endl;
    for (int i = first_batch_size; i < kNumElements; ++i) {
      alg_hnsw->addPoint(data.data() + (i * kDim), i);
    }

    // Final validation - ensure all points are retrievable
    std::cout << "Final validation of all elements..." << std::endl;
    for (int i = 0; i < kNumElements; ++i) {
        auto result = alg_hnsw->searchKnn(data.data() + (i * kDim), 1);
        assert(!result.empty() && result.top().second == i);
    }

    std::cout << "Resize test completed successfully!" << std::endl;
}

int main() {
  TestRandomSelf();
  std::cout << "\nAll test runs completed successfully!" << std::endl;
  return 0;
}
