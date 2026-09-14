#include "../../hnswlib/visited_list_pool.h"

#include <cassert>
#include <limits>
#include <stdexcept>
#include <type_traits>

static_assert(!std::is_copy_constructible<hnswlib::VisitedListLease>::value,
              "VisitedListLease must not be copy constructible");
static_assert(!std::is_copy_assignable<hnswlib::VisitedListLease>::value,
              "VisitedListLease must not be copy assignable");

int main() {
    hnswlib::VisitedListPool pool(1, 8);

    hnswlib::VisitedList *leased_pointer = nullptr;
    try {
        auto lease = pool.getFreeVisitedListLease();
        leased_pointer = lease->get();
        assert(leased_pointer->curV == 1);

        leased_pointer->mass[2] = leased_pointer->curV;
        hnswlib::vl_type previous_generation = leased_pointer->curV;
        assert(lease->next() == leased_pointer);
        assert(leased_pointer->curV == previous_generation + 1);
        assert(leased_pointer->mass[2] != leased_pointer->curV);

        for (unsigned int i = 0; i < leased_pointer->numelements; ++i) {
            leased_pointer->mass[i] = 123;
        }
        leased_pointer->curV = std::numeric_limits<hnswlib::vl_type>::max();
        lease->next();
        assert(leased_pointer->curV == 1);
        for (unsigned int i = 0; i < leased_pointer->numelements; ++i) {
            assert(leased_pointer->mass[i] == 0);
        }

        throw std::runtime_error("exercise exceptional lease return");
    } catch (const std::runtime_error &) {
    }

    auto returned_lease = pool.getFreeVisitedListLease();
    assert(returned_lease->get() == leased_pointer);
    assert(returned_lease->get()->curV == 2);
    return 0;
}
