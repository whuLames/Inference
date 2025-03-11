#include <src/kernels/addresidual_rmsnorm.h>

template<typename T>
__device__ void warpReduce(T val)
{
    for(int offset = 32>>1; offset > 0; offset >>= 1)
    {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<typename T>
__device__ void blockReduce(T val)
{
    static __shared__ float warpRes[64]; // 存放block内部各个warp reduce的结果
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    val = warpReduce(val); 
    if(lane_id == 0) warpRes[warp_id] = val;
    __syncthreads();
    int warp_num = (blockDim.x + 31) / 32; // 如果保证是整数倍, 这里没必要做向上取整的处理?

    T sum = tid < warp_num ? warpRes[tid] : T(0);
    sum = warpReduce(sum);
    return sum;
}

template<typename T>
__global__ void addResidualAndRMSNorm_kernel(T* residual, 
                                             T* decoder_out,
                                             T* bias,
                                             T* gamma,
                                             float eps,
                                             int token_num,
                                             int hidden_unit)
{
    int token_id = blockIdx.x;
    int tid = threadIdx.x;
    using Vec_t = Vec<float>::Type; 

    Vec_t* beg_residual = reinterpret_cast<Vec_t*>(residual + hidden_unit * token_id); 
    Vec_t* beg_decoder = reinterpret_cast<Vec_t*>(decoder_out + hidden_unit * token_id);
    
    Vec_t* thread_residual = beg_residual[tid];
    Vec_t* thread_decoder = beg_decoder[tid];

    // add residual to decoder_out
    thread_decoder.x += thread_residual.x;
    thread_decoder.y += thread_residual.y;
    thread_decoder.z += thread_residual.z;
    thread_decoder.w += thread_residual.w;

    // store [decoder + residual] as final residual
    thread_residual.x =  thread_decoder.x;
    thread_residual.y =  thread_decoder.y;
    thread_residual.z =  thread_decoder.z;
    thread_residual.w =  thread_decoder.w;

    float thread_sum = 0.0f;
    thread_sum += thread_decoder.x * thread_decoder.x;
    thread_sum += thread_decoder.y * thread_decoder.y;
    thread_sum += thread_decoder.z * thread_decoder.z;
    thread_sum += thread_decoder.w * thread_decoder.w;

    // block reduce
    thread_sum = blockReduce<T>(thread_sum);
    __shared__ float factor;

    if(tid == 0)
    {
        factor = rsqrtf(float(thread_sum) / hidden_unit + eps);
    }
    __syncthreads();

    Vec_t* beg_weight = reinterpret_cast<Vec_t*>(gamma);
    Vec_t* thread_weight = beg_weight[tid];

    thread_decoder.x *= factor * thread_weight.x;
    thread_decoder.y *= factor * thread_weight.y;
    thread_decoder.z *= factor * thread_weight.z;
    thread_decoder.w *= factor * thread_weight.w;
}


template<typename T>
void launchAddresidualAndRMSNorm(TensorWrapper<T>* residual,
                                 TensorWrapper<T>* decoder_out,
                                 BaseWeight<T>* bias,
                                 T* gamma, // 这里就不封装为LayerNormWeight了
                                 float eps)
{
    /*
    residual: 刚开始输入的token对应的embedding  [num_token, hidden_units]
    decoder_out: 经过context attention的输出embedding [num_token, hidden_units]
    bias: context attention最后一个linear的bias [hidden_units]
    gamma: RMSNorm 的参数 [hidden_units]
    eps: RMSNorm 计算常量
    */
    int token_num = residual->shape[0];
    int hidden_units = decoder_out->shape[1];
    int thread_num = hidden_units / 4; // 向量化读取
    // int thread_num = (thread_num + 31) / 32 * 32; // 保证选取warp的数量是32的倍数
    dim3 grid_size(token_num);
    dim3 block_size(thread_num);

    addResidualAndRMSNorm_kernel<T><<<grid_size, block_size>>>(residual->data,
                                 decoder_out->data,
                                 bias->data,
                                 gamma,
                                 eps,
                                 token_num,
                                 hidden_units);

}