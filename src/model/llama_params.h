#pragma once
struct LLaMAAttentionStaticParams {
    int   rotary_embedding_dim;
    float rotary_embedding_base; 
    int   max_position_embeddings;
    bool  use_dynamic_ntk; // for dyn scaling rope
};

// (RussWong)note: llama类模型里面动态改变的变量, 注意非全部必需
struct LLaMAAttentionDynParams {
    int batch_size;
    int num_tokens;
    int max_q_len;
    int max_k_len;
    int num_layers;
    bool is_ctx = false;
};

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

}


