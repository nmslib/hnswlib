#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "hnswlib/hnswlib.h"

namespace {

// std::isfinite can be optimized away under CMake's -Ofast flags.
bool finiteBits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

bool finiteBits(double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Assertion failed: " << message << std::endl;
        std::abort();
    }
}

template<typename Result>
void expectNonFiniteError(const Result& result) {
    require(!result.ok(), "eligible non-finite distance must return an error Status");
    require(std::strstr(result.status().message(), "non-finite distance") != nullptr,
            "error Status must identify a non-finite distance");
}

#if defined(__EXCEPTIONS) || _HAS_EXCEPTIONS == 1
template<typename Search>
void expectRuntimeError(Search search) {
    bool caught = false;
    try {
        search();
    } catch (const std::runtime_error& error) {
        caught = true;
        require(std::strstr(error.what(), "non-finite distance") != nullptr,
                "throwing wrapper must preserve the non-finite distance error");
    }
    require(caught, "throwing wrapper must translate the error to std::runtime_error");
}
#endif

void expectNear(float actual, double expected) {
    require(finiteBits(actual), "finite control must return a finite distance");
    const double difference = static_cast<double>(actual) - expected;
    require(difference >= -expected * 0.000002 &&
            difference <= expected * 0.000002,
            "finite control must preserve its squared L2 distance");
}

class ExcludeLabel : public hnswlib::BaseFilterFunctor {
 public:
    explicit ExcludeLabel(hnswlib::labeltype excluded) : excluded_(excluded), calls(0) {}

    bool operator()(hnswlib::labeltype label) override {
        ++calls;
        return label != excluded_;
    }

 private:
    hnswlib::labeltype excluded_;

 public:
    size_t calls;
};

void testReportedL2Case(size_t dim, float alpha, bool expect_error) {
    std::vector<float> p1(dim, 0.0f);
    std::vector<float> p2(dim, 0.0f);
    std::vector<float> query(dim, 0.0f);
    p1[0] = alpha;
    p2[1] = alpha;
    query[0] = 0.6f * alpha;
    query[1] = 0.8f * alpha;
    for (size_t i = 0; i < dim; ++i) {
        require(finiteBits(p1[i]) && finiteBits(p2[i]) && finiteBits(query[i]),
                "reported overflow fixture must have only finite coordinates");
    }

    hnswlib::L2Space space(dim);
    hnswlib::BruteforceSearch<float> index(&space, 2);
    require(index.addPointNoExceptions(p1.data(), 1).ok(), "p1 insertion must succeed");
    require(index.addPointNoExceptions(p2.data(), 2).ok(), "p2 insertion must succeed");

    std::cout << "Reported L2 case: dim=" << dim << ", alpha=" << alpha << std::endl;
    auto result = index.searchKnnNoExceptions(query.data(), 2);
    if (expect_error) {
        expectNonFiniteError(result);
        expectNonFiniteError(index.searchKnnNoExceptions(query.data(), 1));
        expectNonFiniteError(index.searchKnnCloserFirstNoExceptions(query.data(), 2));
        return;
    }

    require(result.ok(), "finite reported case must succeed");
    auto heap = std::move(result).value();
    require(heap.size() == 2, "finite reported case must return both labels");
    const double scale = static_cast<double>(alpha) * static_cast<double>(alpha);
    require(heap.top().second == 1, "p1 must be the farther finite result");
    expectNear(heap.top().first, 0.8 * scale);
    heap.pop();
    require(heap.top().second == 2, "p2 must be the nearest finite result");
    expectNear(heap.top().first, 0.4 * scale);

    auto closest = index.searchKnnCloserFirstNoExceptions(query.data(), 2);
    require(closest.ok(), "finite closer-first search must succeed");
    require(closest.value().size() == 2 &&
            closest.value()[0].second == 2 && closest.value()[1].second == 1,
            "finite closer-first search must preserve nearest-first label order");
}

void testL2OverflowShapes() {
    const float zero[] = {0.0f, 0.0f};

    // A single squared difference exceeds FLT_MAX.
    const float individual_square[] = {3e19f};
    require(finiteBits(individual_square[0]), "individual-square coordinate must be finite");
    require(!finiteBits(individual_square[0] * individual_square[0]),
            "individual-square fixture must overflow in one square");
    hnswlib::L2Space scalar_space(1);
    hnswlib::BruteforceSearch<float> scalar_index(&scalar_space, 1);
    require(scalar_index.addPointNoExceptions(individual_square, 1).ok(),
            "individual-square insertion must succeed");
    expectNonFiniteError(scalar_index.searchKnnNoExceptions(zero, 1));

    // Each square is finite (2.25e38); their sum (4.5e38) exceeds FLT_MAX.
    const float accumulation_only[] = {1.5e19f, 1.5e19f};
    for (size_t i = 0; i < 2; ++i) {
        require(finiteBits(accumulation_only[i]), "accumulation coordinate must be finite");
        require(finiteBits(accumulation_only[i] * accumulation_only[i]),
                "each accumulation-only square must remain finite");
    }
    hnswlib::L2Space accumulation_space(2);
    hnswlib::BruteforceSearch<float> accumulation_index(&accumulation_space, 1);
    require(accumulation_index.addPointNoExceptions(accumulation_only, 1).ok(),
            "accumulation-only insertion must succeed");
    expectNonFiniteError(accumulation_index.searchKnnNoExceptions(zero, 1));
}

void testL2FilterIncludesOverflowAfterFiniteNeighbor() {
    const float query[] = {0.0f, 0.0f};
    const float finite_neighbor[] = {1.0f, 0.0f};
    const float overflow_neighbor[] = {3e19f, 0.0f};
    hnswlib::L2Space space(2);
    hnswlib::BruteforceSearch<float> index(&space, 2);
    require(index.addPointNoExceptions(finite_neighbor, 10).ok(), "finite insertion must succeed");
    require(index.addPointNoExceptions(overflow_neighbor, 20).ok(), "overflow insertion must succeed");

    ExcludeLabel exclude_overflow(20);
    auto finite_result = index.searchKnnNoExceptions(query, 1, &exclude_overflow);
    require(finite_result.ok(), "filtered-out overflow must not cause an error");
    require(finite_result.value().size() == 1 &&
            finite_result.value().top().second == 10 &&
            finite_result.value().top().first == 1.0f,
            "filter must retain the finite nearest neighbor");

    ExcludeLabel include_both(0);
    expectNonFiniteError(index.searchKnnNoExceptions(query, 1, &include_both));
}

// A custom numeric space isolates the search's handling of distance values,
// including double, from the built-in float-only L2 distance implementation.
template<typename Distance>
class NumericDistanceSpace : public hnswlib::SpaceInterface<Distance> {
 public:
    NumericDistanceSpace() : calls(0) {}

    size_t get_data_size() override {
        return sizeof(Distance);
    }

    hnswlib::DISTFUNC<Distance> get_dist_func() override {
        return distance;
    }

    void* get_dist_func_param() override {
        return &calls;
    }

    size_t calls;

 private:
    static Distance distance(const void*, const void* point, const void* parameter) {
        ++*const_cast<size_t*>(static_cast<const size_t*>(parameter));
        Distance result;
        std::memcpy(&result, point, sizeof(result));
        return result;
    }
};

// Build special values from IEEE bit patterns so Clang's fast-math warning
// does not reject compile-time infinity constants.
template<typename Distance>
struct NonFiniteFixtures;

template<>
struct NonFiniteFixtures<float> {
    float values[3];

    NonFiniteFixtures() {
        const uint32_t bits[] = {
            UINT32_C(0x7f800000),  // +infinity
            UINT32_C(0xff800000),  // -infinity
            UINT32_C(0x7fc00000)   // quiet NaN
        };
        std::memcpy(values, bits, sizeof(values));
    }
};

template<>
struct NonFiniteFixtures<double> {
    double values[3];

    NonFiniteFixtures() {
        const uint64_t bits[] = {
            UINT64_C(0x7ff0000000000000),  // +infinity
            UINT64_C(0xfff0000000000000),  // -infinity
            UINT64_C(0x7ff8000000000000)   // quiet NaN
        };
        std::memcpy(values, bits, sizeof(values));
    }
};

template<typename Distance>
void testNonFiniteDistances() {
    const NonFiniteFixtures<Distance> fixtures;
    const Distance* invalid = fixtures.values;
    const Distance query = 0;
    const Distance finite_neighbor = 1;
    for (size_t i = 0; i < 3; ++i) {
        require(!finiteBits(invalid[i]), "non-finite distance fixture must retain its bit pattern");
        NumericDistanceSpace<Distance> space;
        hnswlib::BruteforceSearch<Distance> index(&space, 2);
        require(index.addPointNoExceptions(&finite_neighbor, 10).ok(),
                "custom finite insertion must succeed");
        require(index.addPointNoExceptions(&invalid[i], 20).ok(),
                "custom non-finite insertion must succeed");

        // Even after the heap is full, an eligible invalid candidate is an error.
        expectNonFiniteError(index.searchKnnNoExceptions(&query, 1));
        expectNonFiniteError(index.searchKnnCloserFirstNoExceptions(&query, 1));

        ExcludeLabel exclude_invalid(20);
        auto result = index.searchKnnNoExceptions(&query, 1, &exclude_invalid);
        require(result.ok(), "filtered-out non-finite distance must not cause an error");
        require(result.value().size() == 1 && result.value().top().second == 10 &&
                result.value().top().first == 1,
                "filter must preserve the finite custom-distance result");

#if defined(__EXCEPTIONS) || _HAS_EXCEPTIONS == 1
        expectRuntimeError([&]() { index.searchKnn(&query, 1); });
        expectRuntimeError([&]() { index.searchKnnCloserFirst(&query, 1); });
#endif
    }
}

template<typename Distance>
void testZeroKDoesNotEvaluateDistancesOrFilters() {
    NumericDistanceSpace<Distance> space;
    hnswlib::BruteforceSearch<Distance> index(&space, 1);
    const Distance query = 0;
    const NonFiniteFixtures<Distance> fixtures;
    const Distance invalid = fixtures.values[2];
    ExcludeLabel include_all(0);

    auto empty = index.searchKnnNoExceptions(&query, 0, &include_all);
    require(empty.ok() && empty.value().empty(), "empty index with k=0 must succeed");
    require(index.addPointNoExceptions(&invalid, 20).ok(), "k=0 fixture insertion must succeed");

    auto heap = index.searchKnnNoExceptions(&query, 0, &include_all);
    require(heap.ok() && heap.value().empty(), "k=0 must return an empty heap");
    auto closest = index.searchKnnCloserFirstNoExceptions(&query, 0, &include_all);
    require(closest.ok() && closest.value().empty(), "k=0 must return an empty closer-first vector");
    require(index.searchKnn(&query, 0, &include_all).empty(), "throwing k=0 wrapper must return empty");
    require(index.searchKnnCloserFirst(&query, 0, &include_all).empty(),
            "throwing closer-first k=0 wrapper must return empty");
    require(space.calls == 0, "k=0 must not evaluate any distance");
    require(include_all.calls == 0, "k=0 must not invoke the label filter");
}

void testFiniteCustomDoubleDistances() {
    NumericDistanceSpace<double> space;
    hnswlib::BruteforceSearch<double> index(&space, 3);
    const double query = 0;
    const double points[] = {-4.0, 1.0, 9.0};
    for (size_t i = 0; i < 3; ++i) {
        require(index.addPointNoExceptions(&points[i], i + 1).ok(),
                "finite double insertion must succeed");
    }
    auto result = index.searchKnnCloserFirstNoExceptions(&query, 2);
    require(result.ok(), "finite custom double distances must remain supported");
    require(result.value().size() == 2 &&
            result.value()[0].first == -4.0 && result.value()[0].second == 1 &&
            result.value()[1].first == 1.0 && result.value()[1].second == 2,
            "finite custom double distances must preserve their values and order");
}

void testIntegerDistancesRemainSupported() {
    hnswlib::L2SpaceI space(4);
    hnswlib::BruteforceSearch<int> index(&space, 2);
    const unsigned char query[] = {0, 0, 0, 0};
    const unsigned char p1[] = {1, 2, 0, 0};
    const unsigned char p2[] = {3, 4, 0, 0};
    require(index.addPointNoExceptions(p1, 11).ok(), "integer p1 insertion must succeed");
    require(index.addPointNoExceptions(p2, 22).ok(), "integer p2 insertion must succeed");
    auto result = index.searchKnnCloserFirstNoExceptions(query, 2);
    require(result.ok(), "integer distance searches must remain supported");
    require(result.value().size() == 2 &&
            result.value()[0].first == 5 && result.value()[0].second == 11 &&
            result.value()[1].first == 25 && result.value()[1].second == 22,
            "integer L2 distances must preserve values and nearest-first order");
}

}  // namespace

int main() {
    const size_t dims[] = {2, 4, 16, 17};
    testFiniteCustomDoubleDistances();
    testIntegerDistancesRemainSupported();
    for (size_t dim : dims) {
        testReportedL2Case(dim, 1.0f, false);
        testReportedL2Case(dim, 2e19f, false);
    }
    for (size_t dim : dims) {
        testReportedL2Case(dim, 3e19f, true);
        testReportedL2Case(dim, 2.5e19f, true);
    }
    testL2OverflowShapes();
    testL2FilterIncludesOverflowAfterFiniteNeighbor();
    testNonFiniteDistances<float>();
    testNonFiniteDistances<double>();
    testZeroKDoesNotEvaluateDistancesOrFilters<float>();
    testZeroKDoesNotEvaluateDistancesOrFilters<double>();
    std::cout << "Test ok" << std::endl;
    return 0;
}
