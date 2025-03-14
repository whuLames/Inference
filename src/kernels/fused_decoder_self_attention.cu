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
__device__ T warpReduceSum(T val)
{
    for(int offset = 32>>1; offset > 0; offset >>= 1)
    {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<typename T>
__device__ T warpReduceMax(T val)
{
    for(int offset = 32>>1; offset > 0; offset >>= 1)
    {
        val = max(__shfl_xor_sync(0xFFFFFFFF, val, offset), val);
    }
    return val;
}

template<typename T>
__device__ T blockReduceMax(T val)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_num = (blockDim.x + 32 - 1) / 32;
    __shared__ float warpRes[64];

    val = warpReduceMax(val);

    if(lane_id == 0) warpRes[warp_id] = val;
    __syncthreads()

    T sum = warp_id < warp_num ? warpRes[warp_id] : T(0);

    return warpReduceMax<T>(sum);
}

template<typename T>
__device__ T blockReduceSum(T val)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_num = (blockDim.x + 32 - 1) / 32;
    __shared__ float warpRes[64];

    val = warpReduceSum(val);

    if(lane_id == 0) warpRes[warp_id] = val;
    __syncthreads()

    T sum = warp_id < warp_num ? warpRes[warp_id] : T(0);

    return warpReduceSum<T>(sum);
}

template<typename T>
__global__ void fused_decoder_self_attn_kernel(T* q,
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
    // 1. RoPE: 不做RoPE, 假设此时的输入已经做完了位置编码, 与向量化读取不匹配
    2. concat kv cache: 不做concat kv 假设此时已经实现了concat KV, 传入的k_cache和v_cache已经是
    3. Qk gemv
    4. Scale and SoftMax
    5. Qk*v gemv

    grid_size: head_num * batch_size
    block_size: head_size
    k,v cache: [layer, bs, kv_head_num, max_seq_len, head_size]
    */

    // a thread is responsible for a head_size of q
    int tid = threadIdx.x;
    int q_batch_id = blockIdx.x / head_num;
    int q_head_id = blockIdx.x % head_num;

    int kv_batch_id = q_batch_id;
    int kv_head_id = q_head_id / (head_num / kv_head_num);

    int batch_stride = (head_num + 2 * kv_head_num) * head_size; /* head_num * head_size*/
    int kv_batch_stride = kv_head_num * head_size;
    int head_stride = head_size;

    int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
    int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;     

    int vec_size = Vec<float>::size;
    
    int q_offset_vec = q_batch_id * batch_stride + q_head_id * head_stride + tid * vec_size;
    int k_offset_vec = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid * vec_size;
    int cache_offset = kv_batch_id * kv_head_num * max_seq_len * head_size +
                       kv_head_id * max_seq_len * head_size +
                       tid * vec_size;
    
    int step_stride = head_size; // step_stride is used for what?
    float scale = rsqrt(float(head_size)); // 用于后续softMax
    using Vec_t = Vec<float>::Type;

    Vec_t qvec, kvec, vvec;
    const T* q_mem = q;
    const T* k_mem = k;
    const T* v_mem = v;

    if(tid * vec_size < head_size)
    {
        // const_cast 用于去除变量的 const 修饰符
        qvec = reinterpret_cast<Vec_t*>(const_cast<T*>(&q_mem[q_offset_vec])); 
        kvec = reinterpret_cast<Vec_t*>(const_cast<T*>(&k_mem[k_offset_vec])); 
        vvec = reinterpret_cast<Vec_t*>(const_cast<T*>(&v_mem[k_offset_vec])); 
    }

    extern __shared__ char sqk[]; // 声明一块动态共享内存, 其大小由function configuration 参数决定
    T* sq_scalar = reinterpret_cast<T*>(sqk); // 存放q
    float* logits = reinterpret_cast<float*>(sqk + head_size);
    Vec_t* sq = reinterpret_cast<Vec_t*>(sq_scalar);
    if(tid * vec_size < head_size) {
        sq[tid] = qvec;
    }
    __syncthreads();
    
    float zero = 0.0f;
    Vec_t zero_f4 = scalar_cast_vec<Vec_t, T>(zero);
    float4 scale_f4 = scalar_cast_vec<float4, float>(scale);

    // do q * k
    for(int i = 0; i < step; i ++)
    {
        Vec_t kvec_qk = tid * vec_size < head_size ? *reinterpret_cast<Vec_t*>(&k_cache[iter * step_stride + cache_offset]) : zero_f4;
        if(iter == step - 1 && tid * vec_size < head_size) {
            *reinterpret_cast<Vec_t*>(&k_cache[iter * step_stride + cache_offset]) = kvec; // concat kv cache
            kvec_qk = kvec;
        }
        Vec_t qk = zero_f4; // 记录q*k的计算结果
        qk.x = tid * vec_size < head_size ? sq[tid].x * kvec_qk.x * scale_f4.x : zero;  //根本没必要将scale扩充为float4类型
        qk.y = tid * vec_size < head_size ? sq[tid].y * kvec_qk.y * scale_f4.y : zero; 
        qk.z = tid * vec_size < head_size ? sq[tid].z * kvec_qk.z * scale_f4.z : zero; 
        qk.w = tid * vec_size < head_size ? sq[tid].w * kvec_qk.w * scale_f4.w : zero; 
        T qk_acc = qk.x + qk.y + qk.z + qk.w;
        T attn_score = blockReduceSum<T>(qk_acc);

        if(tid == 0) {
            logits[iter] = attn_score;
        }
        __syncthreads()
    }

    T local_logits = tid < step ? logits[step] : T(0);
    // softmax
    __shared__ float max_val, sum_val;
    
    T block_max = blockReduceMax<T>(local_logits);

    if(tid == 0) {
        max_val = block_max;
    }
    __syncthreads();

    T local_val = tid < step ? expf(local_logits - max_val) : 0;
    T global_sum_val = blockReduceSum<T>(local_val);

    if(tid == 0) {
        sum_val = global_sum_val + 1e-6;
    }
    __syncthreads();

    if(tid < step) {
        logits[step] = local_val / sum_val; 
    }
    __syncthreads();
    
    // do logits * v
    if(tid * vec_size < head_size) {
        Vec_t O = scalar_cast_vec<Vec_t, T>(0.0f);
        for(int i = 0; i < step; i ++) {
            Vec_t vvec_qkv = *reinterpret_cast<Vec_t*>(&v_cache[iter * step_stride + cache_offset]);
            if(iter == step - 1) { // concat v_cache 
                *reinterpret_cast<Vec_t*>(&v_cache[iter * step_stride + cache_offset]) = vvec;
                vvec_qkv = vvec;
            }
            O.x += logits[i] * vvec_qkv.x;
            O.y += logits[i] * vvec_qkv.y;
            O.z += logits[i] * vvec_qkv.z;
            O.w += logits[i] * vvec_qkv.w;
        }
        *reinterpret_cast<Vec_t*>(&mha_output[q_offset_vec]) = O;
    }
    
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
    mha_output: The output of the function [bs, head_num, head_size]
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
    fused_decoder_self_attn_kernel<T><<<grid_size, block_size, smem_size_bytes>>>(q,
                                                                                  k,
                                                                                  v,
                                                                                  qkv->data,
                                                                                  k_cache->data + layer_offset,
                                                                                  v_cache->data + layer_offset,
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