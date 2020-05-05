#pragma once

#include <mutex>
#include <string.h>

namespace hnswlib {
    /* visited_list type */
    typedef unsigned short int vl_type;

    /*
     * 访问过的List
     */
    class VisitedList {
    public:
        // 该list被用过的次数
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
        };

        ~VisitedList() { delete[] mass; }
    };
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

    class VisitedListPool {
        std::deque<VisitedList *> pool;
        std::mutex poolguard;
        // visted_list size
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
                // 在用完之后再回收
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
        };

        void releaseVisitedList(VisitedList *vl) {
            // 用完后回收
            std::unique_lock <std::mutex> lock(poolguard);
            pool.push_front(vl);
        };

        ~VisitedListPool() {
            while (pool.size()) {
                VisitedList *rez = pool.front();
                pool.pop_front();
                delete rez;
            }
        };
    };
}

