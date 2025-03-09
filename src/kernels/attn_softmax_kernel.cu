#include <src/kernels/attn_softmax_kernel.h>

template <typename T>
struct SumOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a + b; }
};

template <typename T>
struct MaxOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return max(a, b); }
};

template <template <typename> class ReductionOp, typename T>
__inline__ __device__ T warpReduce(T val)
{
    for (int mask = 32 / 2; mask > 0; mask /= 2)
    {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template <template <typename> class ReductionOp, typename T>
__inline__ __device__ T blockReduce(T val)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_num = (blockDim.x + 32 - 1) / 32;
    static __shared__ T warp_res[64]
    val = warpReduce<ReductionOp, T>(val);
    if(lane_id == 0) {
        warp_res[lane_id] = val;
    }
    __syncthreads();
    float warp_val = tid < warp_num ? warp_res[tid] : T(0);
    return warpReduce<ReductionOp, T>(warp_val)
}

// float
__global__ void maskScaleAndSoftMax_kernel(float* attn_score,
                                      float* qk,
                                      float* mask,
                                      int batch_size,
                                      int head_num,
                                      int q_len,
                                      int k_len,
                                      float scale)
{
    // dim3 grid_size(bs, head_num, max_q_len);
    /*
    qk: The result of q*k [bs, head_num, max_q_len, max_k_len]
    mask: Attenmask: [bs, max_q_len, max_k_len]
    attn_score: 输出结果 [bs, head_num, max_q_len, max_k_len]
    scale: sqrt(head_size)
    */
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int qid = blockIdx.z;
    int tid = threadIdx.x; 
    float thread_sum = 0.0f;

    int offset = head_num * q_len * k_len * batch_id +
                 q_len * k_len * head_id +
                 k_len * qid +
                 tid * 4;
    // int vec_size = Vec<float>::size;
    // using Vec_t = Vec<float>::Type;

    // Vec_t* begin = reinterpret_cast<Vec_t*>(qk + offset); 
    // if(tid*4 < k_len)
    // {
    //     // 向量化读取
    //     Vec_t vec = begin[tid];
    //     thread_sum += expf(vec.x);
    //     thread_sum += expf(vec.y);
    //     thread_sum += expf(vec.z);
    //     thread_sum += expf(vec.w);
    // }

    if(threadIdx.x >= k_len) return; // 感觉很傻逼




    
    // thread_sum = blockReduce(thread_sum);
}


#define LAUNCH_SOFTMAX(dtype, vec_size)                                                                         \
    if (block.x > 2048 && block.x <= 4096)                                                                      \
    {                                                                                                           \
        constexpr int NUMS_PER_THREAD_PER_ROW = 4;                                                              \
        block.x /= 4 * vec_size;                                                                                \
        block.x = (block.x + 32 - 1) / 32 * 32;                                                                 \
        assert(block.x < 1024);                                                                                 \
        ScaleMaskAndSoftmax_##dtype<dtype, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((dtype *)attn_score->data, \
                                                                                     (dtype *)qk->data,         \
                                                                                     (dtype *)mask->data,       \
                                                                                     batch_size,                \
                                                                                     head_nums,                 \
                                                                                     q_length,                  \
                                                                                     k_length,                  \
                                                                                     scale);                    \
    }                                                                                                           \
    else if (block.x > 1024)                                                                                    \
    {                                                                                                           \
        constexpr int NUMS_PER_THREAD_PER_ROW = 2;                                                              \
        block.x /= 2 * vec_size;                                                                                \
        \                                    
        block.x = (block.x + 32 - 1) / 32 * 32;                                                                 \
        \   
        assert(block.x < 1024);                                                                                 \
        ScaleMaskAndSoftmax_##dtype<dtype, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((dtype *)attn_score->data, \
                                                                                     (dtype *)qk->data,         \
                                                                                     (dtype *)mask->data,       \
                                                                                     batch_size,                \
                                                                                     head_nums,                 \
                                                                                     q_length,                  \
                                                                                     k_length,                  \
                                                                                     scale);                    \
    }                                                                                                           \
    else                                                                                                        \
    {                                                                                                           \
        \ 
        constexpr int NUMS_PER_THREAD_PER_ROW = 1;                                                              \
        block.x /= vec_size;                                                                                    \
        assert(block.x < 1024);                                                                                 \
        ScaleMaskAndSoftmax_##dtype<dtype, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((dtype *)attn_score->data, \
                                                                                     (dtype *)qk->data,         \
                                                                                     (dtype *)mask->data,       \
                                                                                     batch_size,                \
                                                                                     head_nums,                 \
                                                                                     q_length,                  \
                                                                                     k_length,                  \
                                                                                     scale);                    \
        \      
                                                                                                     \
    }

template<typename T>
void launchMaskScaleAndSoftMax(TensorWrapper<T>* qk,
                               TensorWrapper<T>* mask,
                               TensorWrapper<T>* attn_score,
                               float scale)  
{
    /*
    qk: The result of q*k [bs, head_num, max_q_len, max_k_len]
    mask: Attenmask: [bs, max_q_len, max_k_len]
    attn_score: 输出结果 [bs, head_num, max_q_len, max_k_len]
    scale: sqrt(head_size)
    */
   int max_q_len = qk->shape[2];
   int max_k_len = qk -> shape[3];
   int batch_size = qk->shape[0];
   int head_num = qk->shape[1];

   // 为什么 max_q_len放在第一维? 有什么说法吗? 
   dim3 grid(max_q_len, batch_size, head_num);
   dim3 block((max_k_len + 32 - 1) / 32 * 32); // 保证thread数目是32的整数倍
   LAUNCH_SOFTMAX(float, 1);
}