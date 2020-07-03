#pragma once
#include <cmath>
#include <limits>
#include "hnswlib.h"

namespace hnswlib {

    static float
    InnerProduct(const float *pVect1, const float *pVect2, const size_t qty) {
        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            res += pVect1[i] * pVect2[i];
        }
        return (1.0f - res);
    }

#if defined(USE_AVX)

// Favor using AVX if available.
    static float
    InnerProductSIMD4Ext(const float *pVect1, const float *pVect2, const size_t qty) {
        float PORTABLE_ALIGN32 TmpRes[8];

        size_t qty16 = qty / 16;
        size_t qty4 = qty / 4;

        const float *pEnd1 = pVect1 + 16 * qty16;
        const float *pEnd2 = pVect1 + 4 * qty4;

        __m256 sum256 = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

            __m256 v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            __m256 v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
        }

        __m128 v1, v2;
        __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

        while (pVect1 < pEnd2) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        }

        _mm_store_ps(TmpRes, sum_prod);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];;
        return 1.0f - sum;
}

#elif defined(USE_SSE)

    static float
    InnerProductSIMD4Ext(const float *pVect1, const float *pVect2, const size_t qty) {
        float PORTABLE_ALIGN32 TmpRes[8];

        size_t qty16 = qty / 16;
        size_t qty4 = qty / 4;

        const float *pEnd1 = pVect1 + 16 * qty16;
        const float *pEnd2 = pVect1 + 4 * qty4;

        __m128 v1, v2;
        __m128 sum_prod = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        }

        while (pVect1 < pEnd2) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        }

        _mm_store_ps(TmpRes, sum_prod);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

        return 1.0f - sum;
    }

#endif

#if defined(USE_AVX)

    static float
    InnerProductSIMD16Ext(const float *pVect1, const float *pVect2, const size_t qty) {
        float PORTABLE_ALIGN32 TmpRes[8];

        size_t qty16 = qty / 16;


        const float *pEnd1 = pVect1 + 16 * qty16;

        __m256 sum256 = _mm256_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

            __m256 v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            __m256 v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
        }

        _mm256_store_ps(TmpRes, sum256);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

        return 1.0f - sum;
    }

#elif defined(USE_SSE)

      static float
      InnerProductSIMD16Ext(const float *pVect1, const float *pVect2, const size_t qty) {
        float PORTABLE_ALIGN32 TmpRes[8];

        size_t qty16 = qty / 16;

        const float *pEnd1 = pVect1 + 16 * qty16;

        __m128 v1, v2;
        __m128 sum_prod = _mm_set1_ps(0);

        while (pVect1 < pEnd1) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
        }
        _mm_store_ps(TmpRes, sum_prod);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

        return 1.0f - sum;
    }

#endif

#if defined(USE_SSE) || defined(USE_AVX)
    static float
    InnerProductSIMD16ExtResiduals(const float *pVect1v, const float *pVect2v, const size_t qty) {
        size_t qty16 = qty >> 4 << 4;
        float res = InnerProductSIMD16Ext(pVect1v, pVect2v, qty16);
        float *pVect1 = (float *) pVect1v + qty16;
        float *pVect2 = (float *) pVect2v + qty16;

        size_t qty_left = qty - qty16;
        float res_tail = InnerProduct(pVect1, pVect2, qty_left);
        return res + res_tail - 1.0f;
    }

    static float
    InnerProductSIMD4ExtResiduals(const float *pVect1v, const float *pVect2v, const size_t qty) {
        size_t qty4 = qty >> 2 << 2;

        float res = InnerProductSIMD4Ext(pVect1v, pVect2v, qty4);
        size_t qty_left = qty - qty4;

        float *pVect1 = (float *) pVect1v + qty4;
        float *pVect2 = (float *) pVect2v + qty4;
        float res_tail = InnerProduct(pVect1, pVect2, qty_left);

        return res + res_tail - 1.0f;
    }
#endif

    class InnerProductSpace : public SpaceInterface<float, float> {

        float (*fstdistfunc_)(const float *pVect1, const float *pVect2, const size_t qty);
    public:
        InnerProductSpace(size_t dim) : SpaceInterface<float, float>(dim) {
            fstdistfunc_ = InnerProduct;
#if defined(USE_AVX) || defined(USE_SSE)
            if (dim % 16 == 0)
                fstdistfunc_ = InnerProductSIMD16Ext;
            else if (dim % 4 == 0)
                fstdistfunc_ = InnerProductSIMD4Ext;
            else if (dim > 16)
                fstdistfunc_ = InnerProductSIMD16ExtResiduals;
            else if (dim > 4)
                fstdistfunc_ = InnerProductSIMD4ExtResiduals;
#endif
        }

        float calculate_distance(const float *pVect1, const float *pVect2) {
            return fstdistfunc_(pVect1, pVect2, dim_);
        }

        ~InnerProductSpace() {}
    };

    class FloatNormalizer : public NormalizerInterface<float, float> {
    public:
        using NormalizerInterface<float, float>::NormalizerInterface;

        void normalize_vector(const float *data, float *norm_array) {
            float norm = 0.0f;
            for(size_t i = 0; i < dim_; i++)
                norm += data[i] * data[i];
            norm = 1.0f / (sqrtf(norm) + 1e-30f);
            for (size_t i = 0; i < dim_; i++)
                norm_array[i] = data[i] * norm;
        }
    };
}
