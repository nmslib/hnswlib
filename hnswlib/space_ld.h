#pragma once
#include "hnswlib.h"

namespace hnswlib {
    static int
    LevenshteinDistance(const void *pVect1, const void *pVect2, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t l1 = strlen((char *) pVect1);
        size_t l2 = strlen((char *) pVect2);
        char *p2 = (char *)pVect2;

        int i, j, t, track;
        int dist[qty][qty];

        char *s1 = (char *) pVect1;
        char *s2 = (char *) pVect2;

        for (i=0; i<=l1; i++) {
            dist[0][i] = i;
        }
        for (j=0; j<=l2; j++) {
            dist[j][0] = j;
        }
        for (j=1; j<=l1; j++) {
            for (i=1; i<=l2; i++) {
                if(s1[i-1] == s2[j-1]) {
                    track= 0;
                } else {
                    track = 1;
                }
                t = std::min((dist[i-1][j]+1), (dist[i][j-1]+1));
                dist[i][j] = std::min(t, (dist[i-1][j-1]+track));
            }
        }
        return dist[l2][l1];
    }

    class LD2Space : public SpaceInterface<int> {

        DISTFUNC<int> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        LD2Space(size_t dim) {
            fstdistfunc_ = LevenshteinDistance;
            dim_ = dim;
            data_size_ = dim_ * sizeof(unsigned char);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<int> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }
    };
}