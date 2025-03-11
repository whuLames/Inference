#include <src/kernels/transpose_and_remove_padding.h>

template<typename T>
__global__ void transpose_removepadding_kernel(T* data_in,  // [bs, head_num, max_q_len, head_size]
                                               T* data_out, // [num_token, q_hidden_size]
                                               int* padding_offset,
                                               int batch_size,
                                               int head_num,
                                               int max_q_len,
                                               int head_size)
{
    // do computation
    // grid: [token]  block: [q_hidden_unit]
    int token_id = blockIdx.x;
    int dst_token_id = token_id + padding_offset[token_id];
    int batch_id = dst_token_id / max_q_len; // 行号
    int local_token_id = dst_token_id % max_q_len;  // 列号
    int thread_num = blockDim.x;
    int tid = threadIdx.x;
    int q_hidden_size = head_num * head_size;
    
    while(tid < q_hidden_size)
    {
        int dst_index =  q_hidden_size * token_id + tid;
        int head_id = tid / head_size;
        int head_size_id = tid % head_size;

        int src_index = head_num * max_q_len * head_size * batch_id +
                        max_q_len * head_size * head_id +
                        head_size * local_token_id +
                        head_size_id;

        data_out[dst_index] = data_in[src_index];
        tid += thread_num;
    }

}



template<typename T>
void launchTransposeAndRemovePadding(TensorWrapper<T> data_w_padding,
                                     TensorWrapper<int> padding_offset,
                                     TensorWrapper<T> data_wo_padding)
{
    /*
    data_w_padding: 输入数据, 即padding形式下的token attention [bs, head_num, max_q_len, head_size]
    data_wo_padding: 输出数据, 即非padding下的token attention [num_token, q_hidden_size]
    padding_offset: [batch_size, max_q_len]
    */
    int batch_size = data_w_padding->shape[0];
    int head_num = data_w_padding->shape[1];
    int max_q_len = data_w_padding->shape[2];
    int head_size = data_w_padding->shape[3];
    int num_token = data_wo_padding->shape[0];
    int q_hidden_size = data_wo_padding->shape[1]; // == head_num * head_size

    dim3 grid_size(num_token);
    dim3 block_size(std::min(q_hidden_size, 1024));

}

// 模板实例化
template void launchTransposeAndRemovePadding(TensorWrapper<float> data_w_padding,
                                              TensorWrapper<int> padding_offset,
                                              TensorWrapper<float> data_wo_padding);