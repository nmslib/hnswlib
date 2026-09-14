#pragma once

#include <mutex>
#include <string.h>
#include <deque>
#include <memory>

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

class VisitedListPool;

// Owns one VisitedList for the duration of a batch worker. Acquisition
// prepares the first search generation; next() prepares each later search.
class VisitedListLease {
    VisitedListPool *pool_;
    VisitedList *visited_list_;

    VisitedListLease(VisitedListPool *pool, VisitedList *visited_list)
        : pool_(pool), visited_list_(visited_list) {}

    friend class VisitedListPool;

 public:
    VisitedListLease(const VisitedListLease &) = delete;
    VisitedListLease &operator=(const VisitedListLease &) = delete;

    VisitedList *get() const {
        return visited_list_;
    }

    VisitedList *next() {
        visited_list_->reset();
        return visited_list_;
    }

    ~VisitedListLease();
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

class VisitedListPool {
    std::deque<VisitedList *> pool;
    std::mutex poolguard;
    int numelements;

 public:
    VisitedListPool(int initmaxpools, int numelements1) {
        numelements = numelements1;
        for (int i = 0; i < initmaxpools; i++)
            pool.push_front(new VisitedList(numelements));
    }

    VisitedList *getFreeVisitedList() {
        VisitedList *rez;
        {
            std::unique_lock <std::mutex> lock(poolguard);
            if (pool.size() > 0) {
                rez = pool.front();
                pool.pop_front();
            } else {
                rez = new VisitedList(numelements);
            }
        }
        rez->reset();
        return rez;
    }

    void releaseVisitedList(VisitedList *vl) {
        std::unique_lock <std::mutex> lock(poolguard);
        pool.push_front(vl);
    }

    std::unique_ptr<VisitedListLease> getFreeVisitedListLease() {
        return std::unique_ptr<VisitedListLease>(new VisitedListLease(this, getFreeVisitedList()));
    }

    ~VisitedListPool() {
        while (pool.size()) {
            VisitedList *rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    }
};

inline VisitedListLease::~VisitedListLease() {
    pool_->releaseVisitedList(visited_list_);
}
}  // namespace hnswlib
