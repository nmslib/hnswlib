#include "../../hnswlib/hnswlib.h"
#include <thread>


// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading

template<typename MTYPE>
using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);
static int8_t vector_dot_product(const void* a, const void* b, const void *qty_ptr) {

    uint32_t length = * (uint32_t*)qty_ptr;

    int32_t final_result = 0;
    size_t i = 0;
    int8_t *a_tmp=(int8_t *)a;
    int8_t *b_tmp=(int8_t *)b;
    if(length>=64){
        __m512i sum = _mm512_setzero_si512();  // 初始化累加和为 0
        for (; i+64 <= length; i += 64) {
            // 加载数据
            __m512i va = _mm512_loadu_si512((void*)&a_tmp[i]);
            __m512i vb = _mm512_loadu_si512((void*)&b_tmp[i]);
            // 执行点积运算
            //std::cout << "we are 32 lines" <<std::endl;
            sum = _mm512_dpbusd_epi32(sum, va, vb);
        }

        // 将 SIMD 寄存器中的结果累积到一个标量值
        int32_t result[16]={0};
        _mm512_storeu_si512((void*)result, sum);
        // 累加所有部分结果
        for (int j = 0; j < 16; ++j) {
            final_result += result[j];
        }
    }
    for (; i < length; i++) {
        final_result += a_tmp[i] * b_tmp[i];
    }
    //printf("%d ",final_result);
    return final_result;
}
static int8_t vector_dot_product_opt_avx512(const void* a, const void* b, const void *qty_ptr) {
  const uint8_t* pvec_u8 = (const uint8_t*)a;
  const int8_t* pvec_s8 = (const int8_t*)a;
  size_t qty32 = *((size_t*)qty_ptr) / 64;
  const uint8_t* pend_u8 = pvec_u8 + 64 * qty32;
  // calc dot
  __m512i sum512 = _mm512_setzero_si512();
  __m512i v1, v2,v3;

  __m128i one = _mm_set1_epi16(1);
  __m512i agg_base = _mm512_broadcastw_epi16(one);
  while (pvec_u8 < pend_u8) {
    v1 = _mm512_loadu_si512(pvec_u8);
    v2 = _mm512_loadu_si512(pvec_s8);
    v3 = _mm512_maddubs_epi16(v1, v2);
    sum512 = _mm512_add_epi32(sum512, _mm512_madd_epi16(v3, agg_base));
    pvec_u8 += 64;
    pvec_s8 += 64;
  }
  int8_t dotsum = _mm512_reduce_add_epi32(sum512);
  // fetch from u8 qcode
/*   float* flt_u8 = (float*)((char*)qcode_u8 + dim);
  float scale_u8 = *flt_u8;
  flt_u8++;
  float offset_u8 = *flt_u8;
  // fetch from s8 qcode
  float* flt_s8 = (float*)((char*)qcode_s8 + dim);
  float scale_s8 = *flt_s8;
  flt_s8++;
  float sum_s8 = *flt_s8;
  float score = scale_u8 * scale_s8 * dotsum + offset_u8 * sum_s8; */
  return dotsum;
}
static float vector_dot_product_bf16(const void* a, const void* b, const void *qty_ptr) {
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

    // 将vr_f32寄存器的值存入result数组
    _mm512_storeu_ps(result, vr_f32);

    // 累加result数组的所有元素，获得最终的点积结果
    float dot_product = 0.0f;
    for (int j = 0; j < 16; j++) {
        dot_product += result[j];
    }

    // 处理剩余的元素（小于32的部分）
/*     for (; i < dim; i++) {
        float x_val = bf162float(x[i]);
        float y_val = bf162float(y[i]);
        dot_product += x_val * y_val;
    } */
    //printf("%d %f ",dim,dot_product);
    return 1-dot_product;
}


static int8_t fvec_inner_product_int8_avx2int8(const void* a, const void* b, const void *qty_ptr) {
  //exit(-1);
  const uint8_t* pvec_u8 = (const uint8_t*)a;
    const int8_t* pvec_s8 = (const int8_t*)b;
    size_t qty32 = *((size_t*)qty_ptr) / 32;
    const uint8_t* pend_u8 = pvec_u8 + 32 * qty32;

    // 初始化累加和为 0
    __m256i sum256 = _mm256_setzero_si256();
    __m256i v1, v2, v3;

    // 创建一个包含 1 的 128 位向量
    __m128i one = _mm_set1_epi16(1);
    // 广播 1 到 256 位向量
    __m256i agg_base = _mm256_broadcastw_epi16(one);

    while (pvec_u8 < pend_u8) {
        v1 = _mm256_loadu_si256((__m256i*)pvec_u8);
        v2 = _mm256_loadu_si256((__m256i*)pvec_s8);
        v3 = _mm256_maddubs_epi16(v1, v2);
        sum256 = _mm256_add_epi32(sum256, _mm256_madd_epi16(v3, agg_base));
        pvec_u8 += 32;
        pvec_s8 += 32;
    }

    // 处理剩余数据
    for (size_t i = 32 * qty32; i < *((size_t*)qty_ptr); i++) {
        sum256 = _mm256_add_epi32(sum256, _mm256_set1_epi32(pvec_u8[i] * pvec_s8[i]));
    }

    // 将 SIMD 寄存器中的结果累积到一个标量值
    int32_t result[8];
    _mm256_storeu_si256((__m256i*)result, sum256);

    int8_t dotsum = 0;
    for (int i = 0; i < 8; ++i) {
        dotsum += result[i];
    }
    std::cout<<dotsum<< " ";
    return dotsum;
}
static int8_t
Int8InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    int32_t res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((int8_t *) pVect1)[i] * ((int8_t *) pVect2)[i];
    }
        // 如果需要，可以在这里进行截断或饱和处理
    if(res>127) res=127;
    else if(res<-128) res=-128; 
    return static_cast<int8_t>(res);
}

class Int8InnerProductSpace : public hnswlib::SpaceInterface<int8_t> {
    DISTFUNC<int8_t> fstdistfunc_;
    size_t data_size_;
    size_t dim_;
 public:
    Int8InnerProductSpace(size_t dim) {
        fstdistfunc_ = vector_dot_product_opt_avx512;
        dim_ = dim;
        data_size_ = dim * sizeof(int8_t);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<int8_t> get_dist_func() {
        return fstdistfunc_;
    }
    void *get_dist_func_param() {
        return &dim_;
    }
    ~Int8InnerProductSpace() {}
};
void setThreadAffinity(std::thread::native_handle_type handle, size_t cpuId) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpuId, &cpuset);

    int rc = pthread_setaffinity_np(handle, sizeof(cpuset), &cpuset);
    if (rc != 0) {
        throw std::system_error(rc, std::generic_category(), "pthread_setaffinity_np");
    }
}
template<class Function>
inline void ParallelFor_Build(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;


        int dimSizeperThread = (end-start)/numThreads;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                //setThreadAffinity(pthread_self(), threadId);
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;


        int dimSizeperThread = (end-start)/numThreads;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

int call_scalar(hnswlib::HierarchicalNSW<int8_t>* alg_hnsw,Int8InnerProductSpace & space,int8_t* data,int dim, size_t max_elements,int top_k,int num_threads){
    std::vector<hnswlib::labeltype> neighbors(max_elements);
    ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
        std::priority_queue<std::pair<int8_t, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + dim * row, 1);
        hnswlib::labeltype label = result.top().second;
        neighbors[row] = label;
    });
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        hnswlib::labeltype label = neighbors[i];
        if (label == i) correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";
    return 0;
}

// int call_AMX(hnswlib::HierarchicalNSW<int8_t>* alg_hnsw,Int8InnerProductSpace & space,int8_t* data,int dim, size_t max_elements,int top_k,int num_threads){
//     //init_onednn();
//     std::vector<hnswlib::labeltype> neighbors(max_elements);
//     ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
//         std::priority_queue<std::pair<int8_t, hnswlib::labeltype>> result = alg_hnsw->searchKnnAMX(data + dim * row, 1);
//         hnswlib::labeltype label = result.top().second;
//         neighbors[row] = label;
//     });
//     float correct = 0;
//     for (int i = 0; i < max_elements; i++) {
//         hnswlib::labeltype label = neighbors[i];
//         if (label == i) correct++;
//     }
//     float recall = correct / max_elements;
//     std::cout << "Recall: " << recall << "\n";
//     return 0;
// }

int call_scalar_fp32(hnswlib::HierarchicalNSW<float>* alg_hnsw,hnswlib::InnerProductSpace& space,float* data,int dim, size_t max_elements,int top_k,int num_threads){
    std::vector<hnswlib::labeltype> neighbors(max_elements);
    ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + dim * row, 1);
        hnswlib::labeltype label = result.top().second;
        neighbors[row] = label;
    });
    float correct = 0;
    for (size_t i = 0; i < max_elements; i++) {
        hnswlib::labeltype label = neighbors[i];
        if (label == i) correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";
    return 0;
}
// int call_scalar_bf16(hnswlib::HierarchicalNSW<float>* alg_hnsw,Bf16InnerProductSpace& space,float* data,int dim, size_t max_elements,int top_k,int num_threads){
//     std::vector<hnswlib::labeltype> neighbors(max_elements);
//     ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
//         std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + dim * row, 1);
//         hnswlib::labeltype label = result.top().second;
//         neighbors[row] = label;
//     });
//     float correct = 0;
//     for (int i = 0; i < max_elements; i++) {
//         hnswlib::labeltype label = neighbors[i];
//         if (label == i) correct++;
//     }
//     float recall = correct / max_elements;
//     std::cout << "Recall: " << recall << "\n";
//     return 0;
// }


int call_AMX_fp32(hnswlib::HierarchicalNSW<float>* alg_hnsw,hnswlib::InnerProductSpace & space,float* data,int dim, size_t max_elements,int top_k,int num_threads){
    //init_onednn();
    std::vector<hnswlib::labeltype> neighbors(max_elements);
    ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + dim * row, 1);
        hnswlib::labeltype label = result.top().second;
        neighbors[row] = label;
    });
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        hnswlib::labeltype label = neighbors[i];
        if (label == i) correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";
    return 0;
}

int call_AMX_bf16(hnswlib::HierarchicalNSW<float>* alg_hnsw,hnswlib::Bf16InnerProductSpace & space,float* data,int dim, size_t max_elements,int top_k,int num_threads){
    //init_onednn();
    std::vector<hnswlib::labeltype> neighbors(max_elements);
    ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + dim * row, 1);
        hnswlib::labeltype label = result.top().second;
        neighbors[row] = label;
    });
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        hnswlib::labeltype label = neighbors[i];
        if (label == i) correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";
    return 0;
}
int main() {
    int true_dim=1024;
    int dim = true_dim/2;               // Dimension of the elements
    size_t max_elements = 10*1024;   // Maximum number of elements, should be known beforehand
    int M = 32;                 // Tightly connected with internal dimensionality of the data
    size_t nq = max_elements;
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int num_threads = 1;       // Number of threads for operations with index

    int top_k=1;

    int iteration=3;
    float correct = 0;

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data_fp32 = (float* )aligned_alloc(64,true_dim * max_elements*sizeof(float));
    float* query_fp32 = (float* )aligned_alloc(64,true_dim * nq*sizeof(float));
    float* data_bf16 = (float* )aligned_alloc(64,dim * max_elements*sizeof(float));

    const char* amx_bf16_env = std::getenv("BF16_AMX");
    bool amx_enable_bf16 = amx_bf16_env ? std::stoi(amx_bf16_env) : false;

    const char* amx_fp32_env = std::getenv("FP32_AMX");
    bool amx_enable_fp32 = amx_fp32_env ? std::stoi(amx_fp32_env) : false;

    const char* def_fp32_env = std::getenv("FP32_DEF");
    bool def_enable_fp32 = def_fp32_env ? std::stoi(def_fp32_env) : false;    
    
    // const char* avx512_bf16_env = std::getenv("BF16_AVX512");
    // bool avx512_enable_bf16 = avx512_bf16_env ? std::stoi(avx512_bf16_env) : false;



    uint16_t *bf_data = (uint16_t* ) data_bf16;
    for (size_t i = 0; i < true_dim * max_elements; i++) {
        float tmp =  (distrib_real(rng));
        data_fp32[i] = tmp;
        query_fp32[i] = tmp;
        uint32_t *int32_data =(uint32_t *) &tmp;
        bf_data[i]=*int32_data >> 16;
    }

    hnswlib::InnerProductSpace space_fp32(true_dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw_fp32 = new hnswlib::HierarchicalNSW<float>(&space_fp32, max_elements, M, ef_construction);
    // Add data to index
    ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
        alg_hnsw_fp32->addPoint((void*)(data_fp32 + true_dim * row), row);
    });

    // Query the elements for themselves and measure recall

    hnswlib::Bf16InnerProductSpace space_bf16(true_dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw_bf16 = new hnswlib::HierarchicalNSW<float>(&space_bf16, max_elements, M, ef_construction);
    ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
        alg_hnsw_bf16->addPoint((void*)(bf_data + true_dim * row), row);
    });



    std::chrono::_V2::system_clock::time_point start_scalar_fp32,end_scalar_fp32,
                                               start_scalar_bf16,end_scalar_bf16,
                                               start_AMX_fp32,end_AMX_fp32,
                                                start_AMX_bf16,end_AMX_bf16;
    if(def_enable_fp32){
        std::cout << "Default FP32 search start." <<"\n";
        start_scalar_fp32 = std::chrono::high_resolution_clock::now();
        for(int i=0;i<iteration;i++){
          call_scalar_fp32(alg_hnsw_fp32,space_fp32,data_fp32,true_dim,nq,top_k,num_threads);
        }
        end_scalar_fp32 = std::chrono::high_resolution_clock::now();
        std::cout << "Default FP32 search end." <<"\n-----------------------------------------------\n\n";
    }


    // if(avx512_enable_bf16){
    //     std::cout << "BF16 with AVX512 search start." <<"\n";
    //     start_scalar_bf16 = std::chrono::high_resolution_clock::now();
    //     for(int i=0;i<iteration;i++){
    //       call_scalar_bf16(alg_hnsw_bf16,space_bf16,data_bf16,dim,nq,top_k,num_threads);
    //     }
    //     end_scalar_bf16 = std::chrono::high_resolution_clock::now();
    //     std::cout << "BF16 with AVX512 search end." <<"\n-----------------------------------------------\n\n";
    // }


    if(amx_enable_fp32){
      std::cout << "FP32 with AMX search start." <<"\n";
      start_AMX_fp32 = std::chrono::high_resolution_clock::now();
      for(int i=0;i<iteration;i++){
        call_AMX_fp32(alg_hnsw_fp32,space_fp32,data_fp32,true_dim,nq,top_k,num_threads);
      }
      end_AMX_fp32 = std::chrono::high_resolution_clock::now();
      std::cout << "FP32 with AMX search end." <<"\n-----------------------------------------------\n\n";
    }


    if(amx_enable_bf16){
      std::cout << "BF16 with AMX search start." <<"\n";
      start_AMX_bf16 = std::chrono::high_resolution_clock::now();
      for(int i=0;i<iteration;i++){
        call_AMX_bf16(alg_hnsw_bf16,space_bf16,data_bf16,dim,nq,top_k,num_threads);
      }
      end_AMX_bf16 = std::chrono::high_resolution_clock::now();
      std::cout << "BF16 with AVX512 search end." <<"\n-----------------------------------------------\n\n";
    }
  
    std::chrono::duration<double,std::milli> duration_scalar_fp32 = end_scalar_fp32 - start_scalar_fp32;
    std::chrono::duration<double,std::milli> duration_scalar_bf16 = end_scalar_bf16 - start_scalar_bf16;
    std::chrono::duration<double,std::milli> duration_AMX_fp32 = end_AMX_fp32 - start_AMX_fp32;
    std::chrono::duration<double,std::milli> duration_AMX_bf16 = end_AMX_bf16 - start_AMX_bf16;


    if(def_enable_fp32) std::cout << "Time taken for default fp32:" << duration_scalar_fp32.count()/iteration/nq<<std::endl;
    // if(avx512_enable_bf16) std::cout << "Time taken for bf16 with AVX512:" << duration_scalar_bf16.count()/iteration/nq<<std::endl;
    if(amx_enable_fp32) std::cout << "Time taken for fp32 with AMX:" << duration_AMX_fp32.count()/iteration/nq<<std::endl;
    if(amx_enable_bf16) std::cout << "Time taken for bf16 with AMX:" << duration_AMX_bf16.count()/iteration/nq<<std::endl;
    fflush(stdout);

/*     delete[] data;
    delete alg_hnsw; */
    return 0;


}
