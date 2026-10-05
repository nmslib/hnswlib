#pragma once

#include <memory>
#include <string.h>
#include <vector>

namespace hnswlib {
typedef unsigned short int vl_type;

class VisitedList {
 public:
    vl_type curV;
    vl_type *mass;
    unsigned int numelements;

    VisitedList(int numelements1) {
        curV = -1;
        numelements = numelements1;
        mass = new vl_type[numelements];
    }

    void reset() {
        curV++;
        if (curV == 0) {
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV++;
        }
    }

    ~VisitedList() { delete[] mass; }
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
    int numelements;

    struct PerThreadState {
        std::vector<std::unique_ptr<VisitedList>> pool;
    };

    static PerThreadState& getThreadState() {
        static thread_local PerThreadState state;
        return state;
    }

 public:
    VisitedListPool(int /*initmaxpools*/, int numelements1) : numelements(numelements1) {}

    VisitedList *getFreeVisitedList() {
        auto& state = getThreadState();
        if (!state.pool.empty()) {
            std::unique_ptr<VisitedList> vl = std::move(state.pool.back());
            state.pool.pop_back();
            if (vl->numelements < static_cast<unsigned int>(numelements)) {
                vl.reset(new VisitedList(numelements));
            }
            vl->reset();
            return vl.release();
        }
        auto* vl = new VisitedList(numelements);
        vl->reset();
        return vl;
    }

    void releaseVisitedList(VisitedList *vl) {
        if (!vl) return;
        getThreadState().pool.emplace_back(vl);
    }

    ~VisitedListPool() = default;
};
}  // namespace hnswlib
