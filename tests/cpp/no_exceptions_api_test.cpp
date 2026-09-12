#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

#include "hnswlib/hnswlib.h"

namespace {

void testAddPointReportsCapacityErrorWhenReplaceHasNoVacancy() {
    const int dim = 4;
    const size_t max_elements = 2;
    std::vector<float> a(dim, 1.0f);
    std::vector<float> b(dim, 2.0f);
    std::vector<float> c(dim, 3.0f);

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, max_elements, /*M=*/16,
                                          /*ef_construction=*/16, /*random_seed=*/100,
                                          /*allow_replace_deleted=*/true);

    assert(index.addPointNoExceptions(a.data(), 1).ok());
    assert(index.addPointNoExceptions(b.data(), 2).ok());

    hnswlib::Status status = index.addPointNoExceptions(c.data(), 3, /*replace_deleted=*/true);
    assert(!status.ok());
    assert(std::string(status.message()) ==
           "The number of elements exceeds the specified limit");
    assert(index.getCurrentElementCount() == max_elements);
}

void testBruteforceSaveIndexNoExceptionsDoesNotThrow() {
    const int dim = 4;
    const char *path = "no_exceptions_api_test.bin";
    std::vector<float> a(dim, 1.0f);

    hnswlib::L2Space space(dim);
    hnswlib::BruteforceSearch<float> index(&space, 8);
    assert(index.addPointNoExceptions(a.data(), 10).ok());

    std::remove(path);
    hnswlib::Status status = index.saveIndexNoExceptions(path);
    assert(status.ok());
    std::remove(path);
}

}  // namespace

int main() {
    testAddPointReportsCapacityErrorWhenReplaceHasNoVacancy();
    testBruteforceSaveIndexNoExceptionsDoesNotThrow();
    return 0;
}
