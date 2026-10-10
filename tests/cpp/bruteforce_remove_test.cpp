#include <cassert>
#include <iostream>
#include <vector>

#include "hnswlib/hnswlib.h"

// Regression test for BruteforceSearch::removePoint reading an erased
// unordered_map iterator (issue #626). Run under AddressSanitizer to catch the
// use-after-free; without it the test still checks remove/add/search results.
int main() {
    const size_t dim = 4;
    const size_t n = 8;
    hnswlib::L2Space space(dim);
    hnswlib::BruteforceSearch<float> index(&space, n + 2);

    std::vector<float> data(n * dim, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        data[i * dim] = static_cast<float>(i);
        index.addPoint(data.data() + i * dim, 100 + i);
    }

    // Remove an element from the middle; the last element is moved into its slot.
    index.removePoint(103);
    assert(index.cur_element_count == n - 1);

    // The moved element (label 107) must still be found at its own vector.
    auto moved = index.searchKnn(data.data() + 7 * dim, 1);
    assert(moved.size() == 1 && moved.top().second == 107);
    assert(moved.top().first == 0.0f);

    // A new element added after the removal must be searchable.
    std::vector<float> query(dim, 10.0f);
    index.addPoint(query.data(), 9001);
    auto result = index.searchKnn(query.data(), 1);
    assert(result.size() == 1 && result.top().second == 9001);

    std::cout << "BruteforceSearch removePoint test passed" << std::endl;
    return 0;
}
