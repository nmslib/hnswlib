#pragma once
#include "hnswlib.h"
#include <cmath>

namespace hnswlib {
// struct for sparse vector
struct SparseVectorEntry {
    size_t index;
    float val;
};
struct SparseVector {
    size_t num_entries;
    // entries is array of SparseVectorEntry of size num_entries sorted by index
    SparseVectorEntry* entries;
};

static float SparseNorm(SparseVector *sVect) {
    float res = 0;
    SparseVectorEntry *entries = sVect->entries;
    for (size_t i = 0; i < sVect->num_entries; i++) {
        res += entries[i].val * entries[i].val;
    }
    return std::sqrt(res);
}

// d = 1.0 - sum(Ai*Bi) / sqrt(sum(Ai*Ai) * sum(Bi*Bi))
static float SparseCos(const void *sVect1v, const void *sVect2v, const void *) {
    // third argument unused
    SparseVector *sVect1 = (SparseVector *) sVect1v;
    SparseVector *sVect2 = (SparseVector *) sVect2v;
    float numer = 0;
    float denom = SparseNorm(sVect1) * SparseNorm(sVect2);

    if (denom == 0) {
        return 0;
    }

    // to compute sum(Ai * Bi), intersect!
    size_t c1 = 0;
    size_t c2 = 0;
    SparseVectorEntry *e1 = sVect1->entries;
    SparseVectorEntry *e2 = sVect2->entries;
    while (c1 < sVect1->num_entries && c2 < sVect2->num_entries) {
        if (e1->index == e2->index) {
            numer += e1->val * e2->val;
        } 

        // Note: if the indices were equal, we need to increment both
        if (e1->index <= e2->index) {
            // e1 smaller, inc to catch up to e2
            e1++;
            c1++;
        }
        if (e1->index >= e2->index) {
            // e2 smaller, inc e2
            e2++;
            c2++;
        }
    }
    float d = 1.0 - numer / denom;
    return d;
}

class SparseCosSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;

    public:
    SparseCosSpace() {
        fstdistfunc_ = SparseCos;
        // data_size_ is for the use of `addPoint` to know how large the
        //     the data structure representing a single point is 
        //     for the sake of copying
        data_size_ = sizeof(SparseVector);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        // UNUSED
        return NULL;
    }

    void save_data_to_output(std::ofstream output, void* memory_block, size_t cur_element_count) {
        /*
            Block format:
            num_entries (size_t)
            entries SparseVectorEntry[num_entries]
        */

        for (size_t i = 0; i < cur_element_count; i++) {
            SparseVector* sparse_vector = memory_block[i];
            writeBinaryPOD(output, sparse_vector->num_entries)
            output.write(sparse_vector->entries, sparse_vector->num_entries * sizeof(SparseVectorEntry));
        }
    }

    void read_data_to_memory(std::ifstream input, char* memory_block, size_t cur_element_count) {
        // TODO: need to think about memory management (how are the vector entries freed?)
        // input.read(data_level0_memory_, cur_element_count * size_data_per_element_);
    }

    ~SparseCosSpace() {}
};
}
