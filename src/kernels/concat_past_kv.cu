#include <src/kernels/concat_past_kv.h>

// k(v) shape: [bs, kv_head_num, max_q_len, head_size]
// k(v) cache shape: [layer_num, bs, kv_head_num, max_seq_len, head_size]


template <typename T>
__global__ void append_k_cache(T* k_dest, const size_t lay_offset, const T* k_src, const int kv_head_num,
                        const int head_size, const int *cur_query_length, const int* history_length, const int max_q_len,
                        const int max_seq_len)
{
    // grid_size: [bs, kv_head_num, max_q_len, head_size]
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int token_id = blockIdx.x;
    int tid = threadIdx.x;
    int batch_size = blockDim.y;
    int seq_id = history_length[token_id] + token_id;
    T* k_cache_dest = k_dest + lay_offset;

    if(token_id < cur_query_length[batch_id])
    {
        int dest_offset = kv_head_num * max_q_len * head_size * batch_id +
                          max_seq_len * head_size * head_id +
                          head_size * seq_id + 
                          tid;

        int src_offset = kv_head_num * max_q_len * head_size * batch_id +
                         max_seq_len * head_size * head_id +
                         head_size * token_id +
                         tid; 

        k_cache_dest[dest_offset] = k_src[src_offset];
    }

}

template <typename T>
__global__ void append_v_cache(T* v_dest, const size_t lay_offset, const T* v_src, const int kv_head_num,
                        const int *cur_query_length, const int* history_length, const int max_q_len,
                        const int max_seq_len)
{
    // grid_size: [bs, kv_head_num, max_q_len, head_size]
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int token_id = blockIdx.x;
    int tid = threadIdx.x;
    int batch_size = blockDim.y;
    int seq_id = history_length[token_id] + token_id;
    T* v_cache_dest = v_dest + lay_offset;

    if(token_id < cur_query_length[batch_id])
    {
        int dest_offset = kv_head_num * max_q_len * head_size * batch_id +
                          max_seq_len * head_size * head_id +
                          head_size * seq_id + 
                          tid;

        int src_offset = kv_head_num * max_q_len * head_size * batch_id +
                         max_seq_len * head_size * head_id +
                         head_size * token_id +
                         tid; 

        v_cache_dest[dest_offset] = v_src[src_offset];
    }

}

template<typename T>
void launchConcatKVCache(TensorWrapper<T> *k_src, // from qkv bias and rope
                          TensorWrapper<T> *v_src,
                          TensorWrapper<int> *layer_id,         // layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
                          TensorWrapper<int> *cur_query_length, // current epoch or local input length,[batchsize]
                          TensorWrapper<int> *history_length,
                          TensorWrapper<T> *k_dst,
                          TensorWrapper<T> *v_dst)
{
    int batch_size = k_src->shape[0];
    int max_seq_len = k_dst->shape[3];
    int kv_head_num = k_src->shape[1];
    int max_q_len = k_src->shape[2];
    int head_size = k_src->shape[3];
    int blockSize = head_size;
    int layer = layer_id->getVal();
    size_t layer_offset = layer * batch_size * kv_head_num * max_seq_len * head_size;
    dim3 grid(max_q_len, batch_size, kv_head_num);
    append_key_cache<T><<<grid, blockSize>>>(k_dst->data,
                                             layer_offset,
                                             k_src->data,
                                             kv_head_num,
                                             head_size,
                                             /*(int*)*/ cur_query_length->data,
                                             /*(int*)*/ history_length->data,
                                             max_q_len,
                                             max_seq_len);

    append_value_cache<T><<<grid, blockSize>>>(v_dst->data,
                                               layer_offset,
                                               v_src->data,
                                               kv_head_num,
                                               head_size,
                                               /*(int*)*/ cur_query_length->data,
                                               /*(int*)*/ history_length->data,
                                               max_q_len,
                                               max_seq_len);
}

template void launchConcatKVCache(TensorWrapper<float> *k_src, // from qkv bias and rope
                                  TensorWrapper<float> *v_src,
                                  TensorWrapper<int> *layer_id,         // layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
                                  TensorWrapper<int> *cur_query_length, // current epoch or local input length,[batchsize]
                                  TensorWrapper<int> *history_length,
                                  TensorWrapper<float> *k_dst,
                                  TensorWrapper<float> *v_dst);