#pragma once
#include "hnswlib.h"

namespace hnswlib {

    static float
    L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);

        float res = 0;
        for (size_t i = 0; i < qty; i++) {
            float t = *pVect1 - *pVect2;
            pVect1++;
            pVect2++;
            res += t * t;
        }
        return (res);
    }

#if defined(USE_AVX)

    // Favor using AVX if available.
    static float
    L2SqrSIMD16Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
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
    L2SqrSIMD16Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);
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

#ifdef USE_NEON
  static float
  L2SqrSIMD16Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    float32x4_t diff, v1, v2;
    float32x4_t sum0 = vdupq_n_f32(0);
    float32x4_t sum1 = vdupq_n_f32(0);
    float32x4_t sum2 = vdupq_n_f32(0);
    float32x4_t sum3 = vdupq_n_f32(0);

    while (pVect1 < pEnd1) {
      v1 = vld1q_f32(pVect1);
      pVect1 += 4;
      v2 = vld1q_f32(pVect2);
      pVect2 += 4;
      diff = vsubq_f32(v1, v2);
      sum0 = vfmaq_f32(sum0,diff,diff);

      v1 = vld1q_f32(pVect1);
      pVect1 += 4;
      v2 = vld1q_f32(pVect2);
      pVect2 += 4;
      diff = vsubq_f32(v1, v2);
      sum1 = vfmaq_f32(sum1,diff,diff);

      v1 = vld1q_f32(pVect1);
      pVect1 += 4;
      v2 = vld1q_f32(pVect2);
      pVect2 += 4;
      diff = vsubq_f32(v1, v2);
      sum2 = vfmaq_f32(sum2,diff,diff);

      v1 = vld1q_f32(pVect1);
      pVect1 += 4;
      v2 = vld1q_f32(pVect2);
      pVect2 += 4;
      diff = vsubq_f32(v1, v2);
      sum3 = vfmaq_f32(sum3,diff,diff);
    }

    return vaddvq_f32(vaddq_f32(vaddq_f32(sum0,sum1),vaddq_f32(sum2,sum3)));
  }
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_NEON)
    static float
    L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
        float *pVect1 = (float *) pVect1v + qty16;
        float *pVect2 = (float *) pVect2v + qty16;

        size_t qty_left = qty - qty16;
        float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
        return (res + res_tail);
    }
#endif


#ifdef USE_SSE
    static float
    L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        float *pVect1 = (float *) pVect1v;
        float *pVect2 = (float *) pVect2v;
        size_t qty = *((size_t *) qty_ptr);


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
#endif


#ifdef USE_NEON
  static float
  L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);


    size_t qty4 = qty >> 2;

    const float *pEnd1 = pVect1 + (qty4 << 2);

    float32x4_t diff, v1, v2;
    float32x4_t sum = vdupq_n_f32(0);

    while (pVect1 < pEnd1) {
      v1 = vld1q_f32(pVect1);
      pVect1 += 4;
      v2 = vld1q_f32(pVect2);
      pVect2 += 4;
      diff = vsubq_f32(v1, v2);
      sum = vfmaq_f32(sum, diff, diff);
    }
    return vaddvq_f32(sum);
  }
#endif


#if defined(USE_SSE) || defined(USE_NEON)
    static float
    L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        size_t qty = *((size_t *) qty_ptr);
        size_t qty4 = qty >> 2 << 2;

        float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
        size_t qty_left = qty - qty4;

        float *pVect1 = (float *) pVect1v + qty4;
        float *pVect2 = (float *) pVect2v + qty4;
        float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

        return (res + res_tail);
    }
#endif


    class L2Space : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2Space(size_t dim) {
            fstdistfunc_ = L2Sqr;
        #if defined(USE_SSE) || defined(USE_AVX) || defined(USE_NEON)
            if (dim % 16 == 0)
                fstdistfunc_ = L2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                fstdistfunc_ = L2SqrSIMD4Ext;
            else if (dim > 16)
                fstdistfunc_ = L2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                fstdistfunc_ = L2SqrSIMD4ExtResiduals;
        #endif
            dim_ = dim;
            data_size_ = dim * sizeof(float);
        }

        size_t get_data_size() {
            return data_size_;
        }

        DISTFUNC<float> get_dist_func() {
            return fstdistfunc_;
        }

        void *get_dist_func_param() {
            return &dim_;
        }

        ~L2Space() {}
    };

    static int
    L2SqrI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {

        size_t qty = *((size_t *) qty_ptr);
        int res = 0;
        unsigned char *a = (unsigned char *) pVect1;
        unsigned char *b = (unsigned char *) pVect2;

        qty = qty >> 2;
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
        return (res);
    }

#ifdef USE_NEON

#ifdef USE_NEON_DOTPROD

#define DOT_ISA_ATTRIBUTE __attribute__((target("dotprod")))

#if !defined(__ARM_FEATURE_DOTPROD)
  DOT_ISA_ATTRIBUTE
  static uint32x4_t vdotq_u32(uint32x4_t __p0, uint8x16_t __p1, uint8x16_t __p2) {
    uint32x4_t __ret;
    __ret = (uint32x4_t) __builtin_neon_vdotq_v((int8x16_t)__p0, (int8x16_t)__p1, (int8x16_t)__p2, 50);
    return __ret;
  }
#endif

  DOT_ISA_ATTRIBUTE
  static uint32x4_t ssd_16_dot(uint32x4_t o , uint8x16_t a,uint8x16_t b) {
    uint8x16_t t = vabdq_u8(a,b);
    return vdotq_u32(o,t,t);
  }

#else
#define DOT_ISA_ATTRIBUTE

  DOT_ISA_ATTRIBUTE
  static uint32x4_t ssd_16_dot(uint32x4_t o , uint8x16_t a,uint8x16_t b) {
    uint8x16_t t = vabdq_u8(a,b);
    uint16x8_t m0 = vmull_u8(vget_low_u8(t),vget_low_u8(t));
    uint16x8_t m1 = vmull_high_u8(t,t);
    return vaddq_u32(o, vpaddq_u32(vpaddlq_u16(m0), vpaddlq_u16(m1)));
  }

#endif

  DOT_ISA_ATTRIBUTE
  static int L2SqrI_neon(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr)
  {
    uint32x4_t s0 = {0};
    uint32x4_t s1 = {0};
    size_t qty = *((size_t *) qty_ptr);
    unsigned char *a = (unsigned char *) pVect1;
    unsigned char *b = (unsigned char *) pVect2;

    unsigned int loops = qty/32;
    for (int i=0;i<loops;i++)
    {
      uint8x16_t _a = *(uint8x16_t *)a;
      uint8x16_t _b = *(uint8x16_t *)b;
      s0 = ssd_16_dot(s0,_a,_b);
      a += 16;
      b += 16;
      _a = *(uint8x16_t *)a;
      _b = *(uint8x16_t *)b;
      s1 = ssd_16_dot(s1,_a,_b);
      a += 16;
      b += 16;
    }
    if (qty & 16)
    {
      uint8x16_t va = *(uint8x16_t *)a;
      uint8x16_t vb = *(uint8x16_t *)b;
      s0 = ssd_16_dot(s0, va, vb);
      a += 16; b += 16;
    }
    if (qty & 15)
    {
      uint64x2_t ta = {0},tb = {0};

      if (qty & 8)
      {
        ta[0] = *(uint64_t *)a;
        tb[0] = *(uint64_t *)b;
        a += 8; b += 8;
      }

      uint64_t lefta = 0,leftb = 0;
      if (qty & 4)
      {
        lefta = *(uint32_t *)a;
        leftb = *(uint32_t *)b;
        a += 4; b += 4;
      }

      qty &= 3;
      for (int i=0;i<qty;i++)
      {
        lefta = (lefta << 8) | *a++;
        leftb = (leftb << 8) | *b++;
      }
      ta[1] = lefta;
      tb[1] = leftb;
      s0 = ssd_16_dot(s0, ta, tb);
    }
    return vaddvq_u32(vaddq_u32(s0,s1));

  }
#undef DOT_ISA_ATTRIBUTE
#endif


    static int L2SqrI(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr) {
        size_t qty = *((size_t*)qty_ptr);
        int res = 0;
        unsigned char* a = (unsigned char*)pVect1;
        unsigned char* b = (unsigned char*)pVect2;

        for(size_t i = 0; i < qty; i++)
        {
            res += ((*a) - (*b)) * ((*a) - (*b));
            a++;
            b++;
        }
        return (res);
    }

    class L2SpaceI : public SpaceInterface<int> {

        DISTFUNC<int> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2SpaceI(size_t dim) {
#ifdef USE_NEON
            fstdistfunc_ = L2SqrI_neon;
#else
            if(dim % 4 == 0) {
                fstdistfunc_ = L2SqrI4x;
            }
            else {
                fstdistfunc_ = L2SqrI;
            }
#endif

            dim_ = dim;
            data_size_ = dim * sizeof(unsigned char);
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

        ~L2SpaceI() {}
    };


}
