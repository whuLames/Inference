#include <math.h>
#include <stdio.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/qkv_bias_and_RoPE.h"


inline __device__ float2 GetRoPefreq(int zid, int rot_embed_dim, float base, float t_step)
{
    /*
    计算 旋转位置 编码中的一些常量数据：Cos(m Theta_i) and Sin(m Theta_i), 其中 1 <= i <= d/2, d即为query和key 向量的 head_num * head_size(head_size?)
    rot_embed_dim = d
    zid = range(0, d, 2): [0, 2, 4 ... d /2) = 2(i - 1)
    t_step = m, 表示该向量的位置
    base = 10000
    */
   float inv_value = powf(base, (float)zid / rot_embed_dim);
   return {cos(t_step * inv_value), sin(t_step * inv_value)};
}

inline __device__ float2 GetRoPEres(float data, float data_rotate, const float2 coef)
{
    float2 rot_v;
    rot_v.x = data * coef.x - data_rotate * coef.y;
    rot_v.y = data * coef.y + data_rotate * coef.x;
    return rot_v;
} 

template <typename T>
__global__ void add_fusedQKV_bias_transpose_kernel(T *q_buf,
                                                   T *k_buf,
                                                   T *v_buf,
                                                   T *QKV,
                                                   /*optional*/const T *qkv_bias,
                                                   const int *padding_offset, // created before qkv linear
                                                   const int *history_length,
                                                   const int *input_length, // actual length of each seq
                                                   const int batch_size,
                                                   const int seq_len, // max_seq_len to pad to
                                                   const int token_num,
                                                   const int head_num,
                                                   const int kv_head_num,
                                                   const int head_size,
                                                   const int rotary_embedding_dim,
                                                   float rotary_embedding_base, // default 10000 in llama
                                                   int max_position_embeddings, /*default 2048 in llama*/
                                                   bool use_dynamic_ntk /*placeholder for ntk RoPE*/)
{
    /*
    q_buf: [bs, head_num, seq_len, head_size]
    k_buf and v_buf: [bs, kv_head_num, seq_len, head_size]
    QKV: [num_tokens, qkv_head_num, head_size]
    padding_offset: [bs, seq_len]
    history_length: [bs]
    input_length: [bs]
    grid_size: [token_num, head_num]
    block_size: [head_size]
    */
   
    // 每个block 负责 一个token的一个head的 q和k的位置编码的计算
    int token_id = blockIdx.x;
    int head_id = blockIdx.x;
    int tid = threadIdx.x;

    int qkv_head_num = kv_head_num * 2 + head_num;
    int ele_num_per_token = qkv_head_num * head_size; //建立在qkv的head_size相同的情况下，否则需要分开计算
    int ele_num_per_head = head_size * 3;
    int rotary_size = head_size / 2; // rotary_embedding_dim / 2
    // 当前线程需要处理的元素即为 qkv[q_id] and qkv[q_id + rotary_embedding_dim / 2]
    int q_id = token_id * ele_num_per_token + head_id * head_size + tid; 
    int k_id = token_id * ele_num_per_token + head_num * head_size + head_id * head_size + tid;
    int v_id = token_id * qkv_head_num * head_size + head_id * head_size + tid + head_num * head_size + kv_head_num * head_size;

    int padding_token_id = token_id + padding_offset[token_id];
    int batch_id = padding_token_id / seq_len;
    int history_token_num = history_length[batch_id];

    // TODO: int local_token_id = input_length[batch_id];
    int local_token_id = padding_token_id % seq_len;

    int q_dest_id = head_num * seq_len * head_size * batch_id + seq_len * head_size * head_id + head_size * local_token_id + tid; 
    int kv_dest_id = kv_head_num * seq_len * head_size * batch_id + seq_len * head_size * head_id + head_size * local_token_id + tid;
    
    // get freq
    int zid = tid * 2;
    // int t_step = padding_token_id % seq_len // 这个是不对的, 其padding_token_id所在行前面的元素依然有可能是 padding token
    int t_step = history_token_num + local_token_id; // 计算当前token在seq_token中的位置(包括上下文信息)

    float2 rope_freq;
    float2 rope_q_res;
    float2 rope_k_res;

    if(tid < rotary_size)  // 只有一半的线程会计算
    {
        rope_freq = GetRoPefreq(zid, rotary_embedding_dim, base, t_step);
        
        rope_q_res = GetRoPEres(qkv[q_id], qkv[q_id + rotary_size], rope_freq);
        rope_k_res = GetRoPEres(qkv[k_id], qkv[k_id + rotary_size], rope_freq);

        // 将添加位置信息的value 写回到 q_buf and k_buf
        q_buf[q_dest_id] = rope_q_res.x;
        q_buf[q_dest_id + rotary_size] = rope_q_res.y;
        
        if(head_id < kv_head_num)
        {
            k_buf[k_dest_id] = rope_k_res.x;
            k_buf[kv_dest_id + rotary_size] = rope_k_res.y;
            v_buf[kv_dest_id] = v[v_id];
            v_buf[kv_dest_id + rotary_size] = v[v_id + rotary_size]
        }

    }










    


    
}


template<typename T>
void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<T>* q_buf,
                                           TensorWrapper<T>* k_buf,
                                           TensorWrapper<T>* v_buf,
                                           TensorWrapper<T>* QKV,
                                           BaseWeight<T>& qkv,
                                           //Tensor* qkv_bias,
                                           TensorWrapper<int>* padding_offset,
                                           TensorWrapper<int>* history_length,
                                           TensorWrapper<int>* input_length,
                                           LLaMAAttentionStaticParams& params)
{
    /*
    q_buf: [bs, head_num, seq_len, head_size]
    k_buf and v_buf: [bs, kv_head_num, seq_len, head_size]
    QKV: [num_tokens, qkv_head_num, head_size]
    // padding_offset: [bs, seq_len] 这个shape是错误的
    padding_offset: [seq_num, max_seq_len]
    history_length: [bs]
    input_length: [bs]

    seq_len: 所有batch中句子的最长长度，包括history_length + current_length
    */
    int bs = q_buf->shape[0];
    int seq_len = q_buf->shape[2];  // 对应batch中, 历史context的token长度
    int token_num = qkv->shape[0];
    int head_num = q_buf->shape[1];
    int kv_head_num = k_buf->shape[1];
    kv_head_num = (qkv_head_num - head_num) / 2;
    int head_size = qkv-shape[2];
    int rotary_embedding_dim = params.rotary_embedding_dim;
    float rotary_embedding_base = params.rotary_embedding_base;
    int max_position_embeddings = params.max_position_embeddings;
    bool use_dynamic_ntk = params.use_dynamic_ntk;

    dim3 grid_size(token_num, head_num);
    dim3 block_size(head_size);

    add_fusedQKV_bias_transpose_kernel<T><<<grid, block>>>(q_buf->data,
                                                           k_buf->data,
                                                           v_buf->data,
                                                           QKV->data,
                                                           /*optional*/qkv.bias,
                                                           padding_offset->data,
                                                           history_length->data,
                                                           input_length->data,
                                                           batch_size,
                                                           seq_len,
                                                           token_num,
                                                           head_num,
                                                           kv_head_num,
                                                           head_size,
                                                           params.rotary_embedding_dim,
                                                           params.rotary_embedding_base,
                                                           params.max_position_embeddings,
                                                           params.use_dynamic_ntk);
   
}


template<typename T>
__global__ void rope_kernel_for_self_decoder(T* q,
                    T* k,
                    const int batch_size,
                    const int head_num,
                    const int kv_head_num,
                    const int head_size,
                    const int step,
                    int   rotary_embedding_dim,
                    float rotary_embedding_base){
    int tid = threadIdx.x;
    int q_head_id = blockIdx.x;
    int q_batch_id = blockIdx.y;
    // (RussWong)note: !!should add () to head_num / kv_head_num, or res is wrong
    int kv_head_id = q_head_id / (head_num / kv_head_num);
    int kv_batch_id = q_batch_id;

    int batch_stride = head_num * head_size;
    int kv_batch_stride = kv_head_num * head_size;
    int head_stride = head_size;
    int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
    int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;
    if (tid >= rotary_embedding_dim / 2) {
        return;
    }
    // RoPE
    float k_reg = k[k_offset];
    float k_rotate_reg = k[k_offset + head_size / 2];
    float2 cos_sin = GetRoPEfreq(tid * 2, rotary_embedding_dim, rotary_embedding_base, step - 1);
    float2 q_rotate = GetRoPEres(q[q_offset], q[q_offset + head_size / 2], cos_sin);
    float2 k_rotate = make_float2(0,0);
    k_rotate.x = cos_sin.x * k_reg - cos_sin.y * k_rotate_reg;
    k_rotate.y = cos_sin.x * k_rotate_reg + cos_sin.y * k_reg;

    q[q_offset] = q_rotate.x;
    q[q_offset + head_size / 2] = q_rotate.y;
    k[k_offset] = k_rotate.x;
    k[k_offset + head_size / 2] = k_rotate.y;
}
// TODO: fp16 self decoder rope has not implemented yet
template<>
__global__ void rope_kernel_for_self_decoder(half* q,
                    half* k,
                    const int batch_size,
                    const int head_num,
                    const int kv_head_num,
                    const int head_size,
                    const int step, // >= 1
                    int   rotary_embedding_dim,
                    float rotary_embedding_base){}

// note: all TensorWrapper's shape cant see here, we can see it in context_decoder.cpp or self_decoder.cpp
template<typename T>
void launchRoPE(TensorWrapper<T>* qkv_buf,
                TensorWrapper<int>* step,
                LLaMAAttentionStaticParams& static_params){
    const int batch_size = qkv_buf->shape[0];
    const int qkv_head_num = qkv_buf->shape[1];
    int head_num = 32; // only for llama
    const int head_size = qkv_buf->shape[2];
    LLM_CHECK(batch_size == 1);
    LLM_CHECK(qkv_head_num == 96);
    LLM_CHECK(head_size == 128);
    const int cur_step = step->getVal();
    T* qkv_data = qkv_buf->data;
    T* q = qkv_data;
    T* k = qkv_data + head_num * head_size;

    int   rotary_embedding_dim = static_params.rotary_embedding_dim;
    float rotary_embedding_base = static_params.rotary_embedding_base;
    int   max_position_embeddings = static_params.max_position_embeddings;
    dim3 grid(head_num, batch_size);
    dim3 block(head_size); 
    rope_kernel_for_self_decoder<T><<<grid, block>>>(q,
                                                    k,
                                                    batch_size,
                                                    head_num,
                                                    head_num, // only for llama, kv head = head
                                                    head_size,
                                                    cur_step,
                                                    rotary_embedding_dim,
                                                    rotary_embedding_base);
}

template void launchRoPE(TensorWrapper<float>* qkv_buf,
                        TensorWrapper<int>* step,
                        LLaMAAttentionStaticParams& static_params);
template void launchRoPE(TensorWrapper<half>* qkv_buf,
                        TensorWrapper<int>* step,
                        LLaMAAttentionStaticParams& static_params);