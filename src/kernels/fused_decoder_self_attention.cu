#include <src/kernels/fused_decoder_self_attention.h>

__device__ float2 GetRopEfrec(int zid, int rotary_embedding_dim, float rotary_embedding_base, float t_step) 
{
    // zid = 2(i - 1) = 0, 2, 4...
    float mtheta = t_step / powf(rotary_embedding_base, float(zid) / rotary_embedding_dim);
    return {cos(mtheta), sin(mtheta)};
}

__device__ float2 GetRopEres(float data, float data_rotate, float2 coef)
{
    float2 res;
    res.x = data * coef.x + data_rotate * coef.y;
    res.y = data * coef.x - data_rotate * coef.y;
    return res;
}

template<typename T>
__device__ T warpReduce(T val)
{
    for(int offset = 32>>1; offset > 0; offset >>= 1)
    {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }

    return val;
}


template<typename T>
__device__ T blockReduce(T val)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_num = (blockDim.x + 32 - 1) / 32;
    __shared__ float warpRes[64];

    val = warpReduce(val);

    if(lane_id == 0) warpRes[warp_id] = val;
    __syncthreads()

    T sum = warp_id < warp_num ? warpRes[warp_id] : T(0);

    return warpReduce<T>(sum);
}

template<typename T>
__global__ void fuser_decoder_self_attn_kernel(T* q,
                                               T* k,
                                               T* v,
                                               T* qkv_bias,
                                               T* k_cache,
                                               T* v_cache,
                                               T* mha_output,
                                               const int batch_size,
                                               const int head_num,
                                               const int kv_head_num,
                                               const int max_seq_len,
                                               const int head_size,
                                               const int step,
                                               int   rotary_embedding_dim,
                                               float rotary_embedding_base)
{
    /*
    // 1. RoPE: 不做RoPE, 假设此时的输入已经做完了位置编码
    // 2. concat kv cache: 不做concat kv 假设此时已经实现了concat KV, 传入的k_cache和v_cache已经是
    3. Qk gemv
    4. Scale and SoftMax
    5. Qk*v gemv

    grid_size: head_num * batch_size
    block_size: head_size
    */

    // a thread is responsible for a head_size
    int tid = threadIdx.x
    int q_batch_id = blockIdx.x / head_num;
    int q_head_id = blockIdx.x % head_num;

    int kv_batch_id = q_batch_id;
    int kv_head_id = q_head_id / (head_num / kv_head_num);

    

    
    float2 

}

template<typename T>
void launchDecoderMaskedMHA(TensorWrapper<T>* qkv_buf,
                            BaseWeight<T>& qkv,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<T>* k_cache,
                            TensorWrapper<T>* v_cache,
                            TensorWrapper<bool>* finished,
                            TensorWrapper<int>* step,
                            TensorWrapper<T>* mha_output,
                            LLaMAAttentionStaticParams& static_params)
{
    /*
    qkv_buf: [bs, qkv_head_num, head_size]
    qkv: ?
    k_cache: [layer, bs, qv_head_num, max_seq_len, head_size]
    v_cache:
    step: [bs] The length of processed token in each batch
    mha_output: The output of the function
    static_params: some params in llama2
    */
    int batch_size = qkv_buf->shape[0];
    int qkv_head_num = qkv_buf->shape[1];
    int head_size = qkv_buf->shape[2];
    int kv_head_num = k_cache->shape[2];
    int head_num = qkv_head_num - 2 * kv_head_num;
    int cur_step = step->getVal();
    int cur_layer = layer_id->getVal();
    int max_seq_len = qkv_buf->shape[3];
    int layer_offset = batch_size * qkv_head_num * max_seq_len * head_size;

    T* qkv_data = qkv -> data;
    T* q = qkv_data;
    T* k = qkv_data + head_num * head_size;
    T* v = qkv_data + (head_num + kv_head_num) * head_size;

    int rotary_embedding_dim = static_params.rotary_embedding_dim; // head_size?
    float rotary_embedding_base = static_params.rotary_embedding_base;
    bool use_dynamic_ntk = static_params.use_dynamic_ntk;

    size_t smem_size_bytes = head_size * sizeof(T) + cur_step * sizeof(float); // cur_step is used for what ?
    dim3 grid_size(head_num * batch_size);
    dim3 block_size(head_size);
    fuser_decoder_self_attn_kernel<T><<<grid_size, block_size, smem_size_bytes>>>(q,
                                                                                  k,
                                                                                  v,
                                                                                  qkv->data,
                                                                                  k_cache->data,
                                                                                  v_cache->data,
                                                                                  mha_output->data,
                                                                                  batch_size,
                                                                                  head_num,
                                                                                  kv_head_num,
                                                                                  max_seq_len,
                                                                                  head_size,
                                                                                  cur_step,
                                                                                  rotary_embedding_dim,
                                                                                  rotary_embedding_base);
}