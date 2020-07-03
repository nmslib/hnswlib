#pragma once
#include "hnswlib.h"

namespace hnswlib {

    template<typename vec_t, typename dist_t>
    static dist_t L2Sqr(const vec_t *pVect1, const vec_t *pVect2, const size_t qty) {
        dist_t res = 0;
        for (size_t i = 0; i < qty; i++) {
            dist_t t = pVect1[i] - pVect2[i];
            res += t * t;
        }
        return (res);
    }

#if defined(USE_AVX)

    // Favor using AVX if available.
    static float
    L2SqrSIMD16Ext(const float *pVect1, const float *pVect2, const size_t qty) {
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m256 diff, v1, v2;
        __m256 sum = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }

        _mm256_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    }

#elif defined(USE_SSE)

    static float
    L2SqrSIMD16Ext(const float *pVect1, const float *pVect2, const size_t qty) {
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;

        const float *pEnd1 = pVect1 + (qty16 << 4);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }

        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }
#endif

#if defined(USE_SSE) || defined(USE_AVX)
    static float
    L2SqrSIMD16ExtResiduals(const float *pVect1v, const float *pVect2v, const size_t qty) {
        size_t qty16 = qty >> 4 << 4;
        float res = L2SqrSIMD16Ext(pVect1v, pVect2v, qty16);
        float *pVect1 = (float *) pVect1v + qty16;
        float *pVect2 = (float *) pVect2v + qty16;

        size_t qty_left = qty - qty16;
        float res_tail = L2Sqr<float, float>(pVect1, pVect2, qty_left);
        return (res + res_tail);
    }
#endif


#ifdef USE_SSE
    static float
    L2SqrSIMD4Ext(const float *pVect1, const float *pVect2, const size_t qty) {
        float PORTABLE_ALIGN32 TmpRes[8];

        size_t qty4 = qty >> 2;

        const float *pEnd1 = pVect1 + (qty4 << 2);

        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }

    static float
    L2SqrSIMD4ExtResiduals(const float *pVect1v, const float *pVect2v, const size_t qty) {
        size_t qty4 = qty >> 2 << 2;

        float res = L2SqrSIMD4Ext(pVect1v, pVect2v, qty4);
        size_t qty_left = qty - qty4;

        float res_tail = L2Sqr<float, float>(pVect1v + qty4, pVect2v + qty4, qty_left);

        return (res + res_tail);
    }
#endif

    class L2Space : public SpaceInterface<float, float> {
        float (*fstdistfunc_)(const float *pVect1, const float *pVect2, const size_t qty);
    public:
        L2Space(size_t dim) : SpaceInterface<float, float>(dim) {
#if defined(USE_SSE) || defined(USE_AVX)
            if (dim % 16 == 0)
                fstdistfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                fstdistfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                fstdistfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                fstdistfunc_ = L2SqrSIMD4ExtResiduals;
            else
#endif
            fstdistfunc_ = L2Sqr<float, float>;
        }

        float calculate_distance(const float *pVect1, const float *pVect2) {
            return fstdistfunc_(pVect1, pVect2, dim_);
        }

        ~L2Space() {}
    };

    class L2SpaceI : public SpaceInterface<unsigned char, int> {
    public:
        using SpaceInterface<unsigned char, int>::SpaceInterface;

        int calculate_distance(const unsigned char *a, const unsigned char *b) {
            //Ideally should return L2Sqr<unsigned char, int>(a, b, dim_);
            //Using loop unrolled version for improving sift_1b performance
            int qty = dim_ >> 2;
            int res = 0;
            for (size_t i = 0; i < qty; i++) {
                res += ((*a) - (*b)) * ((*a) - (*b));
                a++;
                b++;
                res += ((*a) - (*b)) * ((*a) - (*b));
                a++;
                b++;
                res += ((*a) - (*b)) * ((*a) - (*b));
                a++;
                b++;
                res += ((*a) - (*b)) * ((*a) - (*b));
                a++;
                b++;
            }

            return res;
        }

        ~L2SpaceI() {}
    };
}
