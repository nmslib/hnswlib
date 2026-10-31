#include "../../hnswlib/hnswlib.h"

#include <cstdlib>
#include <sstream>

void require(bool condition) {
    if (!condition) std::abort();
}

void check_update(hnswlib::SpaceInterface<float>& space, size_t capacity,
                  bool roundtrip) {
    float first[] = {0.0f, 0.0f};
    float second[] = {10.0f, 10.0f};
    float replacement[] = {1.0f, 1.0f};
    hnswlib::BruteforceSearch<float> original(&space, capacity);
    original.addPoint(first, 101);
    original.addPoint(second, 4096);
    std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
    require(original.saveIndexNoExceptions(saved).ok());
    hnswlib::BruteforceSearch<float> loaded(&space);
    if (roundtrip) loaded.loadIndex(saved, &space);
    auto& index = roundtrip ? loaded : original;
    require(index.addPointNoExceptions(replacement, 101).ok());
    require(index.cur_element_count == 2);
    require(index.addPointNoExceptions(replacement, 4096).ok());
    require(index.cur_element_count == 2);
    auto results = index.searchKnnCloserFirst(replacement, 2);
    require(results.size() == 2);
    require(results[0].second != results[1].second);
    for (const auto& result : results) {
        require(result.second == 101 || result.second == 4096);
        // Both updated vectors equal the query: squared L2 = 0, 1-dot = -1.
        const float expected = space.get_dist_func()(replacement, replacement,
                                                    space.get_dist_func_param());
        require(result.first == expected);
    }
    // A later roundtrip must retain the newly updated vectors and label map.
    std::stringstream again(std::ios::in | std::ios::out | std::ios::binary);
    require(index.saveIndexNoExceptions(again).ok());
    hnswlib::BruteforceSearch<float> restored(&space);
    restored.loadIndex(again, &space);
    require(restored.addPointNoExceptions(first, 101).ok());
    require(restored.cur_element_count == 2);
    results = restored.searchKnnCloserFirst(first, 2);
    require(results.size() == 2 && results[0].second != results[1].second);
}

int main() {
    hnswlib::L2Space l2(2);
    hnswlib::InnerProductSpace ip(2);
    for (size_t capacity : {size_t(2), size_t(3)}) {
        check_update(l2, capacity, false);
        check_update(l2, capacity, true);
        check_update(ip, capacity, false);
        check_update(ip, capacity, true);
    }
    hnswlib::BruteforceSearch<float> empty(&l2, 2);
    std::stringstream bytes(std::ios::in | std::ios::out | std::ios::binary);
    require(empty.saveIndexNoExceptions(bytes).ok());
    hnswlib::BruteforceSearch<float> loaded(&l2);
    loaded.loadIndex(bytes, &l2);
    float vector[] = {2, 3};
    require(loaded.addPointNoExceptions(vector, 7001).ok());
    require(loaded.cur_element_count == 1);
    require(loaded.searchKnnCloserFirst(vector, 1)[0].second == 7001);
}
