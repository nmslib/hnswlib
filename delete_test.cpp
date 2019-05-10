#include <iostream>
#include <cstdlib>
#include <ctime>
#include <set>
#include "hnswlib/hnswlib.h"

int testDelete() {
    hnswlib::HierarchicalNSW<int>* appr_alg;
    int dim = 128, size = 2000;
    hnswlib::L2SpaceI l2space(dim);
    appr_alg = new hnswlib::HierarchicalNSW<int>(&l2space, size + 10, 128, 200, 100);

    srand(time(NULL));
    int* buffer = new int[dim];
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < dim; j++) {
            buffer[j] = rand() % 10000;
        }
        appr_alg->addPoint((void*) buffer, i + 1);
    }

    int* sample = new int[dim];
    for (int j = 0; j < dim; j++) {
        sample[j] = rand() % 10000;
    }

    std::set<hnswlib::labeltype> deletedSet;

    for (int i = 0; i < 2; i++) {
        std::priority_queue<std::pair<int, hnswlib::labeltype>> result = appr_alg->searchKnn((void*) sample, 200);
        hnswlib::labeltype toDel;
        while (!result.empty()) {
            if (result.size() == 1) {
                std::cout << ' ' << result.top().second << "(" << result.top().first << ") ";
                if (deletedSet.find(result.top().second) != deletedSet.end()) {
                    std::cerr << "delete failed!" << std::endl;
                    return -1;
                }
                toDel = result.top().second;
            }
            result.pop();
        }
        std::cout << std::endl;
        appr_alg->markDelete(toDel);
        deletedSet.insert(toDel);
        try {
            appr_alg->template getDataByLabel<int>(toDel);
            std::cerr << "delete failed!" << std::endl;
            return -1;
        } catch (std::runtime_error e) {
        }
    }

    appr_alg->recycle_in_test();
    std::cout << "reusable starts at " << appr_alg->reusable_entry << std::endl;

    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < dim; j++) {
            buffer[j] = rand() % 10000;
        }
        appr_alg->addPoint((void*) buffer, i + 1 + size);

        std::priority_queue<std::pair<int, hnswlib::labeltype>> result = appr_alg->searchKnn((void*) buffer, 5);
        while (!result.empty()) {
            std::cout << ' ' << result.top().second << "(" << result.top().first << ") ";
            result.pop();
        }
        std::cout << std::endl;
    }

    delete appr_alg;
    delete[] buffer;
    delete[] sample;
    return 0;
};

