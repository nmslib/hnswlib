#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <ostream>
#include <sstream>
#include <streambuf>
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

void testLoadIndexNoExceptionsDoesNotClearOnUnopenedStream() {
    const int dim = 4;
    std::vector<float> a(dim, 1.0f);

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, 8);
    assert(index.addPointNoExceptions(a.data(), 1).ok());
    assert(index.getCurrentElementCount() == 1);

    std::ifstream missing("hnswlib_rc_review_missing_index.bin", std::ios::binary);
    assert(!missing.is_open());
    hnswlib::Status status = index.loadIndexNoExceptions(missing, &space);
    assert(!status.ok());
    assert(index.getCurrentElementCount() == 1);

    // Default-constructed streams are often still good() on libc++; the loader
    // must not treat "never opened" as an empty successful index.
    std::ifstream never_opened;
    hnswlib::Status status_unopened = index.loadIndexNoExceptions(never_opened, &space);
    assert(!status_unopened.ok());
    assert(index.getCurrentElementCount() == 1);

    std::vector<float> restored = index.getDataByLabel<float>(1);
    assert(restored.size() == static_cast<size_t>(dim));
    assert(restored[0] == 1.0f);
}

class FailingBuf : public std::streambuf {
 protected:
    int overflow(int) override { return traits_type::eof(); }
    std::streamsize xsputn(const char*, std::streamsize) override { return 0; }
};

void testSaveIndexNoExceptionsDoesNotThrowOnWriteFailure() {
    const int dim = 4;
    std::vector<float> a(dim, 1.0f);

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, 8);
    assert(index.addPointNoExceptions(a.data(), 1).ok());

    FailingBuf buf;
    std::ostream out(&buf);
    hnswlib::Status status = index.saveIndexNoExceptions(out);
    assert(!status.ok());
}

void testAddPointIntegerLevelIsNotReplaceDeleted() {
    const int dim = 4;
    std::vector<float> a(dim, 1.0f);
    std::vector<float> b(dim, 2.0f);

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(
        &space, /*max_elements=*/8, /*M=*/16, /*ef_construction=*/16,
        /*random_seed=*/100, /*allow_replace_deleted=*/false);
    index.addPoint(a.data(), 1);
    // Integer 3 is a graph level. Binding it to bool replace_deleted would
    // throw because replacement is disabled.
    index.addPoint(b.data(), 2, 3);
    assert(index.getCurrentElementCount() == 2);
}

void testReloadDropsStaleLabelsAndDeletedSet() {
    const int dim = 4;
    std::vector<float> a(dim, 1.0f);
    std::vector<float> b(dim, 2.0f);
    std::vector<float> c(dim, 3.0f);

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> live(
        &space, /*max_elements=*/8, /*M=*/16, /*ef_construction=*/16,
        /*random_seed=*/100, /*allow_replace_deleted=*/true);
    assert(live.addPointNoExceptions(a.data(), 1).ok());
    assert(live.addPointNoExceptions(b.data(), 2).ok());
    live.markDelete(1);
    assert(live.getDeletedCount() == 1);

    hnswlib::HierarchicalNSW<float> replacement(&space, 8);
    assert(replacement.addPointNoExceptions(c.data(), 10).ok());
    std::ostringstream saved(std::ios::binary);
    assert(replacement.saveIndexNoExceptions(saved).ok());

    std::istringstream in(saved.str(), std::ios::binary);
    assert(live.loadIndexNoExceptions(in, &space).ok());
    assert(live.getCurrentElementCount() == 1);
    assert(live.getDeletedCount() == 0);

    auto missing = live.getDataByLabelNoExceptions<float>(2);
    assert(!missing.ok());
    std::vector<float> got = live.getDataByLabel<float>(10);
    assert(got.size() == static_cast<size_t>(dim));
    assert(got[0] == 3.0f);
}

void testCorruptLoadDoesNotClearLiveIndex() {
    const int dim = 4;
    std::vector<float> a(dim, 1.0f);

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, 8);
    assert(index.addPointNoExceptions(a.data(), 1).ok());

    std::istringstream truncated("HNSW", std::ios::binary);
    hnswlib::Status status = index.loadIndexNoExceptions(truncated, &space);
    assert(!status.ok());
    assert(index.getCurrentElementCount() == 1);
    std::vector<float> restored = index.getDataByLabel<float>(1);
    assert(restored[0] == 1.0f);
}

void testBruteforceLoadRebuildsLabelMap() {
    const int dim = 4;
    std::vector<float> a(dim, 1.0f);
    std::vector<float> b(dim, 2.0f);
    std::vector<float> a2(dim, 9.0f);

    hnswlib::L2Space space(dim);
    hnswlib::BruteforceSearch<float> index(&space, 8);
    assert(index.addPointNoExceptions(a.data(), 10).ok());
    assert(index.addPointNoExceptions(b.data(), 20).ok());
    assert(index.cur_element_count == 2);

    std::ostringstream saved(std::ios::binary);
    assert(index.saveIndexNoExceptions(saved).ok());

    hnswlib::BruteforceSearch<float> loaded(&space, 8);
    std::istringstream in(saved.str(), std::ios::binary);
    assert(loaded.loadIndexNoExceptions(in, &space).ok());
    assert(loaded.cur_element_count == 2);
    assert(loaded.addPointNoExceptions(a2.data(), 10).ok());
    assert(loaded.cur_element_count == 2);
}

void testBruteforceLoadIndexNoExceptionsDoesNotMutateOnFailure() {
    const int dim = 4;
    std::vector<float> a(dim, 1.0f);

    hnswlib::L2Space space(dim);
    hnswlib::BruteforceSearch<float> index(&space, 8);
    assert(index.addPointNoExceptions(a.data(), 10).ok());
    assert(index.cur_element_count == 1);

    std::ifstream missing("hnswlib_rc_review_missing_bf.bin", std::ios::binary);
    assert(!missing.is_open());
    assert(!index.loadIndexNoExceptions(missing, &space).ok());
    assert(index.cur_element_count == 1);

    std::ifstream never_opened;
    assert(!index.loadIndexNoExceptions(never_opened, &space).ok());
    assert(index.cur_element_count == 1);

    std::istringstream truncated("HNSW", std::ios::binary);
    assert(!index.loadIndexNoExceptions(truncated, &space).ok());
    assert(index.cur_element_count == 1);
}

void testLoadRejectsCurCountGreaterThanMaxElements() {
    const int dim = 4;
    std::vector<float> a(dim, 1.0f);
    std::vector<float> b(dim, 2.0f);

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> saved(&space, 8);
    assert(saved.addPointNoExceptions(a.data(), 1).ok());
    assert(saved.addPointNoExceptions(b.data(), 2).ok());
    std::ostringstream out(std::ios::binary);
    assert(saved.saveIndexNoExceptions(out).ok());
    std::string blob = out.str();
    assert(blob.size() >= 2 * sizeof(size_t));
    size_t patched_max = 1;
    std::memcpy(&blob[sizeof(size_t)], &patched_max, sizeof(size_t));

    hnswlib::HierarchicalNSW<float> live(&space, 8);
    assert(live.addPointNoExceptions(a.data(), 9).ok());
    std::istringstream in(blob, std::ios::binary);
    assert(!live.loadIndexNoExceptions(in, &space).ok());
    assert(live.getCurrentElementCount() == 1);
}

void testClearResetsCapacitySoAddPointDoesNotWriteNull() {
    const int dim = 4;
    std::vector<float> a(dim, 1.0f);

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, 8);
    assert(index.addPointNoExceptions(a.data(), 1).ok());
    index.clear();
    hnswlib::Status status = index.addPointNoExceptions(a.data(), 1);
    assert(!status.ok());
}

void testStatusCopiesStackMessage() {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "stack-%d", 7);
    hnswlib::Status st(buf);
    buf[0] = 'X';
    assert(!st.ok());
    assert(std::string(st.message()) == "stack-7");

    hnswlib::Status ok;
    assert(ok.ok());
    assert(ok.message() == nullptr);

    hnswlib::Status from_string(std::string("owned"));
    assert(!from_string.ok());
    assert(std::string(from_string.message()) == "owned");
}

void testSearchKnnCloserFirstIsConst() {
    const int dim = 4;
    std::vector<float> a(dim, 1.0f);

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, 8);
    index.addPoint(a.data(), 1);
    const hnswlib::HierarchicalNSW<float>& cref = index;
    auto res = cref.searchKnnCloserFirst(a.data(), 1);
    assert(res.size() == 1);
    assert(res[0].second == 1);
}

}  // namespace

int main() {
    testAddPointReportsCapacityErrorWhenReplaceHasNoVacancy();
    testBruteforceSaveIndexNoExceptionsDoesNotThrow();
    testLoadIndexNoExceptionsDoesNotClearOnUnopenedStream();
    testSaveIndexNoExceptionsDoesNotThrowOnWriteFailure();
    testReloadDropsStaleLabelsAndDeletedSet();
    testCorruptLoadDoesNotClearLiveIndex();
    testBruteforceLoadRebuildsLabelMap();
    testBruteforceLoadIndexNoExceptionsDoesNotMutateOnFailure();
    testAddPointIntegerLevelIsNotReplaceDeleted();
    testLoadRejectsCurCountGreaterThanMaxElements();
    testClearResetsCapacitySoAddPointDoesNotWriteNull();
    testStatusCopiesStackMessage();
    testSearchKnnCloserFirstIsConst();
    return 0;
}
