#include <src/kernels/rmsnorm.h>

//TODO: 测试warpReduce结束之后是否每个thread内部的val都一样
template<typename T>
__device__ T warpReduce(T val)
{
    // do warp reduce
    for(int i = 16; i > 0; i >>= 1)
    {
        val += __shfl_xor_sync(0xFFFFFFFF, val, i);
    }
    return val;
}

template<typename T>
__device__ T blockReduce(T val)
{
    int tid = threadIdx.x;
    int wid = tid / 32;
    int laneid = tid % 32; // warp 内部的 id
    int warp_num = (blockDim.x + 32 - 1) / 32;
    static __shared__ T reduce_res[64];

    val = warpReduce(val);
    if(laneid == 0) 
    {
        reduce_res[wid] = val;
    }
    __syncthreads();

    T sum = tid < warp_num ? reduce_res[tid] : T(0);
    sum = warpReduce<T>(sum);
    return sum;
}

template<typename T>
__global__ void rmsNorm(T* decoder_in, T* residual, T* weight, float eps, int num_tokens, int hidden_units)
{
    // hidden units 我理解为 hidden size
    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    Vec_t* out = reinterpret_cast<Vec_t*>(decoder_in + blockDim.x * hidden_units); // 每个block负责一个token对应的embedding
    Vec_t* rsd = reinterpret_cast<Vec_t*>(residual + blockDim.x * hidden_units);
    float thread_sum = 0.0f;
    for(int idx = threadIdx.x; idx * vec_size < hidden_units; idx += blockDim.x)
    {
        Vec_t vec = out[idx];
        thread_sum += vec.x * vec.x;
        thread_sum += vec.y * vec.y;
        thread_sum += vec.z * vec.z;
        thread_sum += vec.w * vec.w;

        // 向量化存储
        rsd[idx] = vec;

        // store decoder_in to residual
        residual[4 * idx ] = vec.x;
        residual[4 * idx + 1] = vec.y;
        residual[4 * idx + 2] = vec.z;
        residual[4 * idx + 3] = vec.w;
    } 
    __syncthreads(); // 确保thread_sum 计算完成

    // do reduce at the block level
    thread_sum = blockReduce(thread_sum);

    __shared__ float factor;
    
    if(threadIdx.x == 0) 
    {
        factor = rsqrtf((float)thread_sum / hidden_units + eps);
    }
    __syncthreads(); // 同步以确保每一个 thread 都能够正确获取 factor的值

    // 向量化读取 weights
    Vec_t* w = reinterpret_cast<Vec_t*>(weight);
    for(int idx = threadIdx.x; idx * vec_size < hidden_units; idx += blockDim.x)
    {
        Vec_t tmp = out[idx];

        out[idx].x = tmp.x * factor * w[idx].x;
        out[idx].y = tmp.x * factor * w[idx].y;
        out[idx].z = tmp.x * factor * w[idx].z;
        out[idx].w = tmp.x * factor * w[idx].w;
    }
}

template<typename T>
void launchRMSNorm(TensorWrapper<T>* decoder_in, TensorWrapper<T>* decoder_residual, LayerNormWeight<T>* norm_weight, float eps)
{
    int num_tokens = decoder_in -> shape[0];
    int hidden_units = decoder_in -> shape[1];
    int num_threads = hidden_units / 4;
    T* residual = decoder_residual -> data;
    int num_blocks = num_tokens;
    
    dim3 block_size(num_threads);
    dim3 grid_size(num_blocks);

    rmsNorm<T><<<grid_size, block_size>>>(decoder_in -> data, decoder_residual -> data, norm_weight -> gamma, 
                                            eps, num_tokens, hidden_units);
}

// 实例化
template void launchRMSNorm(TensorWrapper<float>* decoder_in, TensorWrapper<float>* decoder_residual, LayerNormWeight<float>* norm_weight, float eps);