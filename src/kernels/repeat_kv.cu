#include <src/kernels/repeat_kv.h>


template<typename T>
__global__ void repeat_k_kernel(T *v_dst, // [bs, q_head_num, max_k_len, head_size]
                                const T *v_src, // [layer_num, bs, kv_head_num, max_seq_len, head_size]
                                const size_t layer_offset,
                                const int head_num,
                                const int q_head_per_kv,
                                const int head_size,
                                const int *context_length,
                                const int max_k_len,
                                const int max_seq_len)
{
    /*
    layer_offset: layer_offset, layer层面的偏移量
    head_num: kv_head_num
    q_head_per_kv: 每一对kv被多少头的q所共享
    head_size: head_size
    max_k_len: 历史最大token长度
    max_seq_len: 多轮对话中的token上限

    一个block负责广播一份指定的k(v)
    */

    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int token_id = blockIdx.x;
    if(token_id >= max_k_len) return;
    
    int tid = threadIdx.x;
    if(tid >= head_size) return ;

    int k_head_num = head_num / q_head_per_kv;
    int k_head_id = head_id / q_head_per_kv;

    int src_index = layer_offset +
                    k_head_num * max_seq_len * head_size * batch_id +
                    max_seq_len * head_size * k_head_id +
                    head_size * token_id +
                    tid;

    int dst_index = head_num * max_k_len * head_size * batch_id +
                    max_k_len * head_size * head_id +
                    head_size * token_id +
                    tid;

    v_dst[dst_index] = v_src[src_index];

}

template<typename T>
__global__ void repeat_v_kernel(T *v_dst, // [bs, q_head_num, max_k_len, head_size]
                                const T *v_src, // [layer_num, bs, kv_head_num, max_seq_len, head_size]
                                const size_t layer_offset,
                                const int head_num,
                                const int q_head_per_kv,
                                const int head_size,
                                const int *context_length,
                                const int max_k_len,
                                const int max_seq_len)
{
    /*
    layer_offset: layer_offset, layer层面的偏移量
    head_num: kv_head_num
    q_head_per_kv: 每一对kv被多少头的q所共享
    head_size: head_size
    max_k_len: 历史最大token长度
    max_seq_len: 多轮对话中的token上限

    一个block负责广播一份指定的k(v)
    */

    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;

    int idx = blockDim.x * blockIdx.x + threadIdx; // 这里主要是一个block分配的线程数为128，而不是head_size，如果是head_size，更简单一些
    // 此时 token_id = blockIdx.x

    int token_id = idx / head_size;
    
    if(token_id < max_k_len)
    {
        int v_head_num = head_num / q_head_per_kv;
        int v_head_id = head_id / q_head_per_kv;

        int src_index = layer_offset +
                        v_head_num * max_seq_len * head_size * batch_id +
                        max_seq_len * head_size * v_head_id +
                        head_size * token_id +
                        tid;

        int dst_index = head_num * max_k_len * head_size * batch_id +
                        max_k_len * head_size * head_id +
                        head_size * token_id +
                        tid;

        v_dst[dst_index] = v_src[src_index];
    }
    

    

}


template<typename T>
void launchRepeatKVCache(TensorWrapper<T> *k_cache_src,
                         TensorWrapper<T> *v_cache_src,
                         TensorWrapper<int> *context_length,
                         TensorWrapper<int> *layer_id,
                         TensorWrapper<T> *k_cache_dst,
                         TensorWrapper<T> *v_cache_dst)
{
    int batch_size = k_cache_src->shape[1];
    int kv_head_num = k_cache_src->shape[2];
    int head_num = k_cache_dst->shape[1];
    int head_size = k_cache_src->shape[4];
    int max_seq_len = k_cache_src->shape[3];
    int max_k_len = k_cache_dst->shape[2];
    int blockSize = 128;
    dim3 block_size(blockSize); // 其实就是head_size
    
    dim3 grid_size((max_k_len * head_size + blockSize - 1) / blockSize , batch_size, head_num);

    int layer = layer_id->getVal();
    size_t layer_offset = layer * batch_size * kv_head_num * max_seq_len * head_size;
    int q_head_per_kv = head_num / kv_head_num;

    repeat_k_kernel<T><<<grid_size, block_size>>>(k_cache_dst->data,
                                                  k_cache_src->data,
                                                  layer_offset,
                                                  q_head_per_kv,
                                                  head_size,
                                                  context_length->data,
                                                  max_k_len,
                                                  max_seq_len);

    repeat_v_kernel<T><<<grid_size, block_size>>>(v_cache_dst->data,
                                                  v_cache_src->data,
                                                  layer_offset,
                                                  q_head_per_kv,
                                                  head_size,
                                                  context_length->data,
                                                  max_k_len,
                                                  max_seq_len);
}


// 模板示例化
template void launchRepeatKVCache(TensorWrapper<float> *k_cache_src,
                                    TensorWrapper<float> *v_cache_src,
                                    TensorWrapper<int> *context_length,
                                    TensorWrapper<int> *layer_id,
                                    TensorWrapper<float> *k_cache_dst,
                                    TensorWrapper<float> *v_cache_dst);