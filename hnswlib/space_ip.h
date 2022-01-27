#pragma once
#include "hnswlib.h"

namespace hnswlib {
    /**
     * For a given loop unrolling factor K, distance type dist_t, and data type data_t,
     * calculate the inner product distance between two vectors.
     * The compiler should automatically do the loop unrolling for us here and vectorize as appropriate.
     */
    template<typename dist_t, typename data_t = dist_t, int K = 1>
    static dist_t
    InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        dist_t res = 0;
        data_t *a = (data_t *) pVect1;
        data_t *b = (data_t *) pVect2;

        qty = qty / K;

        for (size_t i = 0; i < qty; i++) {
            for (size_t j = 0; j < K; j++) {
                const size_t index = (i * K) + j;
                const dist_t _a = a[index];
                const dist_t _b = b[index];
                res += _a * _b;
            }
        }

        return (static_cast<dist_t>(1.0f) - res);
    }

    template<typename dist_t, typename data_t = dist_t, int K>
    static dist_t
    InnerProductAtLeast(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {
        size_t k = K;
        size_t remainder = *((size_t *) qty_ptr) - K;

        data_t *a = (data_t *) pVect1;
        data_t *b = (data_t *) pVect2;

        return InnerProduct<dist_t, data_t, K>(a, b, &k)
             + InnerProduct<dist_t, data_t, 1>(a + K, b + K, &remainder);
    }

#if defined(USE_AVX)

// Favor using AVX if available.
    static float
    InnerProductSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

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
    InnerProductSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

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


#if defined(USE_AVX512)

    static float
    InnerProductSIMD16Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN64 TmpRes[16];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        size_t qty16 = qty / 16;


        const float *pEnd1 = pVect1 + 16 * qty16;

        __m512 sum512 = _mm512_set1_ps(0);

        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

            __m512 v1 = _mm512_loadu_ps(pVect1);
            pVect1 += 16;
            __m512 v2 = _mm512_loadu_ps(pVect2);
            pVect2 += 16;
            sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
        }

        _mm512_store_ps(TmpRes, sum512);
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] + TmpRes[13] + TmpRes[14] + TmpRes[15];

        return 1.0f - sum;
    }

#elif defined(USE_AVX)

    static float
    InnerProductSIMD16Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

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
      InnerProductSIMD16Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

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

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    static float
    InnerProductSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        float res = InnerProductSIMD16Ext(pVect1v, pVect2v, &qty16);
        float *pVect1 = (float *) pVect1v + qty16;
        float *pVect2 = (float *) pVect2v + qty16;

        size_t qty_left = qty - qty16;
        float res_tail = InnerProduct<float, float>(pVect1, pVect2, &qty_left);
        return res + res_tail - 1.0f;
    }

    static float
    InnerProductSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty4 = qty >> 2 << 2;

        float res = InnerProductSIMD4Ext(pVect1v, pVect2v, &qty4);
        size_t qty_left = qty - qty4;

        float *pVect1 = (float *) pVect1v + qty4;
        float *pVect2 = (float *) pVect2v + qty4;
        float res_tail = InnerProduct<float, float>(pVect1, pVect2, &qty_left);

        return res + res_tail - 1.0f;
    }
#endif

    template<typename dist_t, typename data_t = dist_t>
    class InnerProductSpace : public SpaceInterface<dist_t> {

        DISTFUNC<dist_t> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        InnerProductSpace(size_t dim) : dim_(dim), data_size_(dim * sizeof(data_t)) {
            if (dim % 128 == 0)
                fstdistfunc_ = InnerProduct<dist_t, data_t, 128>;
            else if (dim % 64 == 0)
                fstdistfunc_ = InnerProduct<dist_t, data_t, 64>;
            else if (dim % 32 == 0)
                fstdistfunc_ = InnerProduct<dist_t, data_t, 32>;
            else if (dim % 16 == 0)
                fstdistfunc_ = InnerProduct<dist_t, data_t, 16>;
            else if (dim % 8 == 0)
                fstdistfunc_ = InnerProduct<dist_t, data_t, 8>;
            else if (dim % 4 == 0)
                fstdistfunc_ = InnerProduct<dist_t, data_t, 4>;

            else if (dim > 128)
                fstdistfunc_ = InnerProductAtLeast<dist_t, data_t, 128>;            
            else if (dim > 64)
                fstdistfunc_ = InnerProductAtLeast<dist_t, data_t, 64>;
            else if (dim > 32)
                fstdistfunc_ = InnerProductAtLeast<dist_t, data_t, 32>;
            else if (dim > 16)
                fstdistfunc_ = InnerProductAtLeast<dist_t, data_t, 16>;
            else if (dim > 8)
                fstdistfunc_ = InnerProductAtLeast<dist_t, data_t, 8>;
            else if (dim > 4)
                fstdistfunc_ = InnerProductAtLeast<dist_t, data_t, 4>;
            else
                fstdistfunc_ = InnerProduct<dist_t, data_t>;
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<dist_t> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }
        ~InnerProductSpace() {}
    };

    template<> InnerProductSpace<float, float>::InnerProductSpace(size_t dim) : dim_(dim), data_size_(dim * sizeof(float)) {
        fstdistfunc_ = InnerProduct<float, float>;
    #if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
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

}
