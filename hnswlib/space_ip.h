#pragma once
#include "hnswlib.h"
#include <sys/syscall.h> 
#include <sys/time.h>
#include <sys/syscall.h> 
#include <unistd.h>
namespace hnswlib {

#if defined(USE_AMX)

#define u64 unsigned long long
#define u8  unsigned char
#define u16 unsigned short int

#define XFEATURE_XTILECFG           17
#define XFEATURE_XTILEDATA          18
#define XFEATURE_MASK_XTILECFG      (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA     (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE         (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM         0x1022
#define ARCH_REQ_XCOMP_PERM         0x1023        

int enable_amx() {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status) {
        std::cout << "SYS_arch_prctl(READ) error" << std::endl;
        return 0;
    }
    if (bitmask & XFEATURE_MASK_XTILEDATA) {
        return 1;
    }
    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (0 != status) {
        std::cout << "SYS_arch_prctl(WRITE) error" << std::endl;
        return 0;
    }
    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA)) {
        std::cout << "SYS_arch_prctl(READ) error" << std::endl;
        return 0;
    }
    return 1;
}

#endif


inline float bf162float(uint16_t data) {
  int t = (data<<16);
  auto a= *reinterpret_cast<float*>(&t);
  return a;
}

static float
InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    float res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((float *) pVect1)[i] * ((float *) pVect2)[i];
    }
    return res;
}

static float
InnerProductDistance(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    return 1.0f - InnerProduct(pVect1, pVect2, qty_ptr);
}



#if defined(USE_AVX)

// Favor using AVX if available.
static float
InnerProductSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
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
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    return sum;
}

static float
InnerProductDistanceSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD4ExtAVX(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE)

static float
InnerProductSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
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

    return sum;
}

static float
InnerProductDistanceSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD4ExtSSE(pVect1v, pVect2v, qty_ptr);
}

#endif


#if defined(USE_AVX512)

static float
InnerProductSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN64 TmpRes[16];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;


    const float *pEnd1 = pVect1 + 16 * qty16;

    __m512 sum512 = _mm512_set1_ps(0);

    size_t loop = qty16 / 4;
    
    while (loop--) {
        __m512 v1 = _mm512_loadu_ps(pVect1);
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v3 = _mm512_loadu_ps(pVect1);
        __m512 v4 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v5 = _mm512_loadu_ps(pVect1);
        __m512 v6 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        __m512 v7 = _mm512_loadu_ps(pVect1);
        __m512 v8 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;

        sum512 = _mm512_fmadd_ps(v1, v2, sum512);
        sum512 = _mm512_fmadd_ps(v3, v4, sum512);
        sum512 = _mm512_fmadd_ps(v5, v6, sum512);
        sum512 = _mm512_fmadd_ps(v7, v8, sum512);
    }

    while (pVect1 < pEnd1) {
        __m512 v1 = _mm512_loadu_ps(pVect1);
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect1 += 16;
        pVect2 += 16;
        sum512 = _mm512_fmadd_ps(v1, v2, sum512);
    }

    float sum = _mm512_reduce_add_ps(sum512);
    return sum;
}

static float
InnerProductDistanceSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtAVX512(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_AVX)

static float
InnerProductSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
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

    return sum;
}

static float
InnerProductDistanceSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtAVX(pVect1v, pVect2v, qty_ptr);
}

#endif


#if defined(USE_SSE)

static float
InnerProductSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
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

    return sum;
}

static float
InnerProductDistanceSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtSSE(pVect1v, pVect2v, qty_ptr);
}

#endif


#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static DISTFUNC<float> InnerProductSIMD16Ext = InnerProductSIMD16ExtSSE;
static DISTFUNC<float> InnerProductSIMD4Ext = InnerProductSIMD4ExtSSE;
static DISTFUNC<float> InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtSSE;
static DISTFUNC<float> InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtSSE;
  

static float
InnerProductDistanceSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = InnerProductSIMD16Ext(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *) pVect1v + qty16;
    float *pVect2 = (float *) pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);
    return 1.0f - (res + res_tail);
}

static float
InnerProductDistanceSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = InnerProductSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *) pVect1v + qty4;
    float *pVect2 = (float *) pVect2v + qty4;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);

    return 1.0f - (res + res_tail);
}
#endif
static float InnerProductDistanceBf16AVX512(const void* a, const void* b, const void *qty_ptr) {
  float result[16] = {0.0f}; // 用于存储中间结果

  uint16_t *x = (uint16_t *)a;
  uint16_t *y = (uint16_t *)b;
  __m512 vr_f32 = _mm512_setzero_ps(); // 初始化累积寄存器为0

  size_t dim = * (size_t*) qty_ptr ;

  size_t i = 0;
  // 每次处理32个元素（16个__bf16元素在__m512bh寄存器中存储为32个uint16_t）
  for (; i + 32 <= dim; i += 32) {
      // 加载32个uint16_t到__m512i类型的临时寄存器
      __m512i temp_x = _mm512_loadu_si512(x + i);
      __m512i temp_y = _mm512_loadu_si512(y + i);

      // 强制转换为__m512bh类型
      __m512bh v1_f16 = reinterpret_cast<__m512bh&>(temp_x);
      __m512bh v2_f16 = reinterpret_cast<__m512bh&>(temp_y);

      // 计算BF16的点积，并将结果累加到vr_f32
      vr_f32 = _mm512_dpbf16_ps(vr_f32, v1_f16, v2_f16);
  }
  
  // 累加result数组的所有元素，获得最终的点积结果
  float dot_product = _mm512_reduce_add_ps(vr_f32);

  // 处理剩余的元素（小于32的部分）
  for (; i < dim; i++) {
      float x_val = bf162float(x[i]);
      float y_val = bf162float(y[i]);
      dot_product += x_val * y_val;
  }
  //printf("%d %f ",dim,dot_product);
  return dot_product;
}

static float InnerProductDistanceBf16AVX512Ext(const void* a, const void* b, const void *qty_ptr){
  return 1.0f - InnerProductDistanceBf16AVX512(a, b, qty_ptr);
}
static float InnerProductDistanceBf16(const void* a, const void* b, const void *qty_ptr) {
  uint16_t *x = (uint16_t *)a;
  uint16_t *y = (uint16_t *)b;
 // __m512 vr_f32 = _mm512_setzero_ps(); // 初始化累积寄存器为0

  size_t dim = * (size_t*) qty_ptr;

  float dot_product = 0.0f;

  for (int i=0; i < dim; i++) {
      float x_val = bf162float(x[i]);
      float y_val = bf162float(y[i]);
      dot_product += x_val * y_val;
  }  
  return 1-dot_product;
}
#if defined(USE_AMX)






float amx_inner_product_matrix_fp32( char **floatLibraryMatrix, char  *floatQueryMatrix, uint64_t dims,uint64_t batchSizeA,
                              uint64_t batchSizeB, float *results){
    int DIM=32;
    int blockCount=(dims)/DIM;
    int tailCount=dims%DIM;
    unsigned char maBf16[1024] __attribute__((aligned(64)));
    unsigned char mbBf16[1024] __attribute__((aligned(64)));

    //thread_local unsigned char *mbBf16=NULL;
    thread_local char *preQuery=NULL;
    thread_local char cfg[64]={0};
    thread_local bool init_mem=false;
 
    if(!init_mem){

/*         if(!mbBf16){
          mbBf16 =(unsigned char *)aligned_alloc(64,sizeof(char)*dims*4);
        } */
        cfg[0]=1;
        cfg[16]=DIM*2;
        cfg[48] = 16;  // row->M
        // matrix B need a layout rearragement
        cfg[16+1*2] = batchSizeB*2*2;   // col = N*4
        cfg[48+1]   = DIM/2;   // row = K/4
 
        cfg[22]=DIM*2;
        cfg[51] = 16;  // row->M
        // matrix B need a layout rearragement
        cfg[24] = batchSizeB*2*2;   // col = N*4
        cfg[52]   = DIM/2;   // row = K/4
 
        cfg[26]=DIM*2;
        cfg[53] = 16;  // row->M
        // matrix B need a layout rearragement
        cfg[28] = batchSizeB*2*2;   // col = N*4
        cfg[54]   = DIM/2;   // row = K/4
 
        cfg[16+2*2] = (batchSizeB*4); // N*sizeof(int32)
        cfg[48+2] = 16;
 
        init_mem = true;
 
        _tile_loadconfig((void *)cfg);
    }
    __m512i high_bits;
    __m512i low_bits;
    __m512i all_bits;
    int i=0;
    for( i = 0; i < blockCount/3; i+=1) {
        int index=3*i;
        for(int j = 0; j < batchSizeA; j++) {
          high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatLibraryMatrix[j] +  index * DIM * 4),16);
          low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  index * DIM * 4 + 64);
          all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
          _mm512_store_si512(maBf16 + j * DIM * 2 , all_bits);
        }

        high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  index * DIM * 4),16);
        low_bits = _mm512_loadu_si512(floatQueryMatrix +  index * DIM * 4 + 64);
        all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
        _mm512_store_si512(mbBf16 , all_bits);

        _tile_loadd(0,maBf16, 64);
        _tile_loadd(1,mbBf16 , 4);

        for(int j = 0; j < batchSizeA; j++) {
          high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatLibraryMatrix[j] +  (index+1) * DIM * 4),16);
          low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  (index+1) * DIM * 4 + 64);
          all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
          _mm512_store_si512(maBf16 + j * DIM * 2 , all_bits);
        }

        high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  (index+1) * DIM * 4),16);
        low_bits = _mm512_loadu_si512(floatQueryMatrix +  (index+1) * DIM * 4 + 64);
        all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
        _mm512_store_si512(mbBf16 , all_bits);

        _tile_loadd(3,maBf16, 64);
        _tile_loadd(4,mbBf16 , 4);

        for(int j = 0; j < batchSizeA; j++) {
          high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatLibraryMatrix[j] +  (index+2) * DIM * 4),16);
          low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  (index+2) * DIM * 4 + 64);
          all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
          _mm512_store_si512(maBf16 + j * DIM * 2 , all_bits);
        }

        high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  (index+2) * DIM * 4),16);
        low_bits = _mm512_loadu_si512(floatQueryMatrix +  (index+2) * DIM * 4 + 64);
        all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
        _mm512_store_si512(mbBf16 , all_bits);

        _tile_loadd(5,maBf16, 64);
        _tile_loadd(6,mbBf16 , 4);

        _tile_dpbf16ps(2,0,1);  
        _tile_dpbf16ps(2,3,4);
        _tile_dpbf16ps(2,5,6);
    }
    switch(blockCount%3){
      case 0: break;
      case 1:
        for(int j = 0; j < batchSizeA; j++) {
          high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatLibraryMatrix[j] +  3 * i * DIM * 4),16);
          low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  3 * i * DIM * 4 + 64);
          all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
          _mm512_store_si512(maBf16 + j * DIM * 2 , all_bits);
        }

        high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  3*i * DIM * 4),16);
        low_bits = _mm512_loadu_si512(floatQueryMatrix +  3*i * DIM * 4 + 64);
        all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
        _mm512_store_si512(mbBf16 , all_bits);

        _tile_loadd(0,maBf16, 64);
        _tile_loadd(1,mbBf16, 4);
        _tile_dpbf16ps(2,0,1);  
        break;
 
      case 2:        
        for(int j = 0; j < batchSizeA; j++) {
          high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatLibraryMatrix[j] +  3*i * DIM * 4),16);
          low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  3*i * DIM * 4 + 64);
          all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
          _mm512_store_si512(maBf16 + j * DIM * 2 , all_bits);
        }

        high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  3*i * DIM * 4),16);
        low_bits = _mm512_loadu_si512(floatQueryMatrix +  3*i * DIM * 4 + 64);
        all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
        _mm512_store_si512(mbBf16 , all_bits);

        _tile_loadd(0,maBf16, 64);
        _tile_loadd(1,mbBf16, 4);

        for(int j = 0; j < batchSizeA; j++) {
          high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatLibraryMatrix[j] +  (3*i+1) * DIM * 4),16);
          low_bits = _mm512_loadu_si512(floatLibraryMatrix[j] +  (3*i+1) * DIM * 4 + 64);
          all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
          _mm512_store_si512(maBf16 + j * DIM * 2 , all_bits);
        }

        high_bits = _mm512_srli_epi32(_mm512_loadu_si512(floatQueryMatrix +  (3*i+1) * DIM * 4),16);
        low_bits = _mm512_loadu_si512(floatQueryMatrix +  (3*i+1) * DIM * 4 + 64);
        all_bits= _mm512_mask_blend_epi16(0x55555555,low_bits , high_bits);
        _mm512_store_si512(mbBf16 , all_bits);

        _tile_loadd(3,maBf16, 64);
        _tile_loadd(4,mbBf16, 4);
        _tile_dpbf16ps(2,0,1);  
        _tile_dpbf16ps(2,3,4);
        break;
    }
  
   
    _tile_stored(2, results, batchSizeB*2*2);
    _tile_zero(2);
   
    // printf("tailCount=%d\n",tailCount);
    if (tailCount != 0) {
        int32_t offset= dims/DIM*DIM;
        for (int k = 0; k < batchSizeA; k++) {
            for (int l = 0; l < batchSizeB; l++) {
              for (int m = 0; m < tailCount; m += 1) {
                results[k * batchSizeB + l] += (*(float *)(floatLibraryMatrix[k]  + 4*(offset+m))) * (*(float *)(floatQueryMatrix + 4*(offset+m)));
              }
            }
        }
    }
 
    return 0;
}

static float InnerProductBatchExtAMX(const void **pVect1v, const void *pVect2v, const void *qty_ptr, size_t nSize, size_t mSize, float * results_amx){

    unsigned int dims= *(unsigned int*)qty_ptr;
    char **floatLibraryMatrix = (char**) pVect1v;
    char *floatQueryMatrix = (char*) pVect2v;


    int batchSizeA = 16, batchSizeB = 16;
    int batchCountA = (nSize - 1) / batchSizeA + 1;
    int batchCountB = (mSize - 1) / batchSizeB + 1;

    int lastBatchSizeA = (nSize % batchSizeA == 0) ? batchSizeA : nSize % batchSizeA;
    int lastBatchSizeB = (mSize % batchSizeB == 0) ? batchSizeB : mSize % batchSizeB;

    int offsetA = batchSizeA * dims * 4;
    int offsetB = batchSizeB * dims * 4;

    float *results_ptr = results_amx;

    for (int i = 0; i < batchCountA; i++) {
        int currentBatchSizeA = (i == batchCountA - 1) ? lastBatchSizeA : batchSizeA;
        char **currentLibraryMatrixPtr = floatLibraryMatrix + i * 16;

        for (int j = 0; j < batchCountB; j++) {
            int currentBatchSizeB = (j == batchCountB - 1) ? lastBatchSizeB : batchSizeB;
            char *currentQueryMatrixPtr = floatQueryMatrix + j * offsetB;

            amx_inner_product_matrix_fp32(currentLibraryMatrixPtr, currentQueryMatrixPtr, dims, currentBatchSizeA, currentBatchSizeB, results_ptr);

            results_ptr += currentBatchSizeB * currentBatchSizeA;
        }
    }

    return 0;
}

float amx_inner_product_matrix_bf16( char **floatLibraryMatrix, char  *floatQueryMatrix, uint64_t dims,uint64_t batchSizeA,
                              uint64_t batchSizeB, float *results_ptr){
    int DIM=32;
    int blockDim = 96;
    int blockCount=((dims))/blockDim;
    size_t tailCount=dims%DIM;
    int tailBlock=dims%blockDim;

    thread_local char cfg[64]={0};
    thread_local bool init_mem=false;

    unsigned char ma1Bf16[1024] __attribute__((aligned(64)));
    unsigned char ma2Bf16[1024] __attribute__((aligned(64)));
    unsigned char ma3Bf16[1024] __attribute__((aligned(64)));

    float results[16*16] __attribute__((aligned(64)))={0};
 
    if(!init_mem){
        cfg[0]=1;
        cfg[16]=DIM*2;
        cfg[48] = 16;  // row->M
        // matrix B need a layout rearragement
        cfg[16+1*2] = batchSizeB*2*2;   // col = N*4
        cfg[48+1]   = DIM/2;   // row = K/4
 
        cfg[22]=DIM*2;
        cfg[51] = 16;  // row->M
        // matrix B need a layout rearragement
        cfg[24] = batchSizeB*2*2;   // col = N*4
        cfg[52]   = DIM/2;   // row = K/4
 
        cfg[26]= DIM*2;
        cfg[53] = 16;  // row->M
        // matrix B need a layout rearragement
        cfg[28] = batchSizeB*2*2;   // col = N*4
        cfg[54]   = DIM/2;   // row = K/4
 
        cfg[16+2*2] = (batchSizeB*4); // N*sizeof(int32)
        cfg[48+2] = 16;
        init_mem = true;
 
        _tile_loadconfig((void *)cfg);
    }
    //memset(maBf16,0,16*DIM*2);
 
    int i=0;
    for(int i=0;i<blockCount;i++){

      //int32_t stride=i*DIM;
      __m512i sa;
      size_t offset = i * blockDim *2;
      
      for(int j=0;j<batchSizeA;j++){  
        size_t destOffset1 = j * DIM * 2;

        _mm512_store_si512(ma1Bf16 + destOffset1, _mm512_loadu_si512(floatLibraryMatrix[j] + offset));
        _mm512_store_si512(ma2Bf16 + destOffset1, _mm512_loadu_si512(floatLibraryMatrix[j] + offset + 64));
        _mm512_store_si512(ma3Bf16 + destOffset1, _mm512_loadu_si512(floatLibraryMatrix[j] + offset + 128));
      } 

      _tile_loadd(1,floatQueryMatrix + offset , 4);
      _tile_loadd(4,floatQueryMatrix + offset + 64 , 4);
      _tile_loadd(6,floatQueryMatrix + offset + 128, 4);
      _tile_loadd(0,ma1Bf16, 64);
      _tile_loadd(3,ma2Bf16, 64);
      _tile_loadd(5,ma3Bf16, 64);
      _tile_dpbf16ps(2,3,4);
      _tile_dpbf16ps(2,0,1);
      _tile_dpbf16ps(2,5,6);
    //amx_int8_mul((u64*) cfg, maInt8,queryMatrix+stride,DIM,batchSizeB*4,(void*)results);
    }
    if(tailBlock >= DIM){
      for(int i=0;i<tailBlock/DIM;i++){
        __m512i sa;
        for(int j=0;j<batchSizeA;j++){  
          sa=_mm512_loadu_si512(floatLibraryMatrix[j]+blockCount*blockDim * 2 + i * DIM * 2 );
          _mm512_store_si512(ma1Bf16+j*DIM*2,sa);
        }
        _tile_loadd(0,ma1Bf16, 64);
        _tile_loadd(1,floatQueryMatrix + blockCount*blockDim*2 + i * DIM*2 , 4);
        _tile_dpbf16ps(2,0,1);
      }
    }
    _tile_stored(2, results, batchSizeB*2*2);
    _tile_zero(2);
   
    if (tailCount != 0) {
        int32_t offset= dims/DIM*DIM;
        for (int k = 0; k < batchSizeA; k++) {
            for (int l = 0; l < batchSizeB; l++) {
                for (int m = 0; m < tailCount; m += 1) {
                  //blockDim*blockCount+tailBlock/DIM*DIM+i
                  
                  results[k * batchSizeB + l] += bf162float(*(uint16_t *)(floatLibraryMatrix[k]  + 2*(offset+m))) * bf162float(*(uint16_t *)(floatQueryMatrix + 2*(offset+m)));
                    // __m512 lib_vec = _mm512_loadu_ps((float *)(floatLibraryMatrix[k]  + 2*(DIM * blockCount + i)));
                    // __m512 query_vec = _mm512_loadu_ps((float *)(floatQueryMatrix + 2*(DIM * blockCount + i)));
                    // result_vec = _mm512_fmadd_ps(lib_vec, query_vec, result_vec);
                }
            }
        }
    }
    memcpy(results_ptr, results, batchSizeA * batchSizeB * sizeof(float));
 
    return 0;
}

static float
InnerProductDistanceBatchExtAMX(const void **pVect1v, const void *pVect2v, const void *qty_ptr, size_t nSize, size_t mSize, float * results_amx) {

  InnerProductBatchExtAMX(pVect1v, pVect2v, qty_ptr,nSize,mSize,results_amx);
  for(int i=0;i<nSize;i++){
    results_amx[i]=1.0f-results_amx[i];
    // printf("%f ",results_amx[i]);
  }
    return 0;
}

static float InnerProductBatchExtAMXBF16(const void **pVect1v, const void *pVect2v, const void *qty_ptr, size_t nSize, size_t mSize, float * results_amx){ 
    unsigned int dims= *(unsigned int*)qty_ptr;
    char **floatLibraryMatrix = (char**) pVect1v;
    char *floatQueryMatrix = (char*) pVect2v;
    
    int batchSizeA = 16, batchSizeB = 16;
    int batchCountA = (nSize - 1) / batchSizeA + 1;
    int batchCountB = (mSize - 1) / batchSizeB + 1;

    int lastBatchSizeA = (nSize % batchSizeA == 0) ? batchSizeA : nSize % batchSizeA;
    int lastBatchSizeB = (mSize % batchSizeB == 0) ? batchSizeB : mSize % batchSizeB;

    int offsetA = batchSizeA * dims * 2;
    int offsetB = batchSizeB * dims * 2;

    float *results_ptr = results_amx;

    for (int i = 0; i < batchCountA; i++) {
        int currentBatchSizeA = (i == batchCountA - 1) ? lastBatchSizeA : batchSizeA;
        char **currentLibraryMatrixPtr = floatLibraryMatrix + i * 16;

        for (int j = 0; j < batchCountB; j++) {
            int currentBatchSizeB = (j == batchCountB - 1) ? lastBatchSizeB : batchSizeB;
            char *currentQueryMatrixPtr = floatQueryMatrix + j * offsetB;

            amx_inner_product_matrix_bf16(currentLibraryMatrixPtr, currentQueryMatrixPtr, dims, currentBatchSizeA, currentBatchSizeB, results_ptr);

            results_ptr += currentBatchSizeB * currentBatchSizeA;
        }
    }

    return 0;
}
static float InnerProductDistanceBatchExtAMXBF16(const void **pVect1v, const void *pVect2v, const void *qty_ptr, size_t nSize, size_t mSize, float * results_amx) {
  InnerProductBatchExtAMXBF16(pVect1v, pVect2v, qty_ptr,nSize,mSize,results_amx);
  for(int i=0;i<nSize;i++){
    results_amx[i]=1.0f-results_amx[i];
  }
    return 0;
}

static float
InnerProductDistanceBatchExtAMXBF16Residuals(const void **pVect1v, const void *pVect2v, const void *qty_ptr, size_t nSize, size_t mSize, float * results_amx) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty32 = qty >> 5 << 5;

    InnerProductBatchExtAMXBF16(pVect1v, pVect2v, &qty32,nSize,mSize,results_amx);

    size_t qty_left = qty - qty32;

    uint16_t *pVect2 = (uint16_t *) pVect2v + qty32;
    for(size_t i = 0; i < nSize; i++) {
        uint16_t *pVect1 = (uint16_t *) pVect1v[i] + qty32;
        results_amx[i] += InnerProductDistanceBf16AVX512(pVect1, pVect2, &qty_left);
    }
    for(size_t i = 0; i < nSize; i++) {
        results_amx[i] = 1.0f - results_amx[i];
    }
    return 0;
}
static AMXDISTFUNC<float> InnerProductDistanceBatchExt = InnerProductDistanceBatchExtAMX;
static AMXDISTFUNC<float> InnerProductDistanceBatchBf16Ext = InnerProductDistanceBatchExtAMXBF16;

#endif
static DISTFUNC<float> InnerProductDistanceBF16Ext = InnerProductDistanceBf16;
class InnerProductSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
#ifdef USE_AMX
    AMXDISTFUNC<float> amxdistfunc_;
#endif

    size_t data_size_;
    size_t dim_;

 public:
    InnerProductSpace(size_t dim) {
        fstdistfunc_ = InnerProductDistance;
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX512;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX512;
        } else if (AVXCapable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        }
    #elif defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
            InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtAVX;
        }
    #endif
    #if defined(USE_AVX)
        if (AVXCapable()) {
            InnerProductSIMD4Ext = InnerProductSIMD4ExtAVX;
            InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtAVX;
        }
    #endif

        

        if (dim % 16 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = InnerProductDistanceSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = InnerProductDistanceSIMD4ExtResiduals;
#endif
#if defined(USE_AMX)
    if (AMXCapable()) {
      amxdistfunc_=InnerProductDistanceBatchExtAMX;
    }

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
#if defined(USE_AMX)
    AMXDISTFUNC<float> get_amx_dist_func(){
      return amxdistfunc_;
    }
#endif

    void *get_dist_func_param() {
        return &dim_;
    }

~InnerProductSpace() {}
};


class Bf16InnerProductSpace : public hnswlib::SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
#ifdef USE_AMX
    AMXDISTFUNC<float> amxdistfunc_;
#endif
    size_t data_size_;
    size_t dim_;
 public:
    Bf16InnerProductSpace(size_t dim) {
        fstdistfunc_ = InnerProductDistanceBf16;
#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable()) {
            //InnerProductSIMD16Ext = InnerProductDistanceBf16AVX512;
            InnerProductDistanceBF16Ext = InnerProductDistanceBf16AVX512Ext;
        } else if (AVXCapable()) {
            //InnerProductSIMD16Ext = InnerProductDistanceBf16;
            InnerProductDistanceBF16Ext = InnerProductDistanceBf16;
        }
    #else 
        //InnerProductSIMD16Ext = InnerProductDistanceBf16;
        InnerProductDistanceBF16Ext = InnerProductDistanceBf16;
    #endif
        fstdistfunc_=InnerProductDistanceBF16Ext;     
#endif
#if defined(USE_AMX)
    if (AMXCapable()) {
        if (dim%32!=0){
          InnerProductDistanceBatchBf16Ext=InnerProductDistanceBatchExtAMXBF16Residuals;
        }else{
          InnerProductDistanceBatchBf16Ext=InnerProductDistanceBatchExtAMXBF16;
        }
          
    }

    amxdistfunc_ = InnerProductDistanceBatchBf16Ext;
#endif
        dim_ = dim ;
        data_size_ = dim  * sizeof(uint16_t);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

#if defined(USE_AMX)
    AMXDISTFUNC<float> get_amx_dist_func(){
      return amxdistfunc_;
    }
#endif
    void *get_dist_func_param() {
        return &dim_;
    }
    ~Bf16InnerProductSpace() {}
};
}  // namespace hnswlib

