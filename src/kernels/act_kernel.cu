#include <src/kernels/act_kernel.h>

template<typename T>
__device__ T silu(T val)
{
    return val / (1.0f + expf(-val));
}
template<typename T>
__global__ void act_kernel(T* input, T* out, int inter_size)
{
    int tid = threadIdx.x;
    int row_id = blockIdx.x;   // row_id 为 batch_id 或 num_tokens
    while(tid < inter_size)
    {
        int src_index_1 = 2 * inter_size * row_id + tid;            
        int src_index_2 = 2 * inter_size + inter_size + tid;
        int dst_index = inter_size * row_id;
        out[dst_index] = silu<T>(input[src_index_1]) * input[src_index_2];
        tid += blockDim.x;
    }
}

template<typename T>
void launchAct(TensorWrapper<T>* input, TensorWrapper<T>* out)
{
    /*
    input: [bs/num_tokens, 2, inter_size]
    out: [bs/num_tokens, inter_size]

    context_decoder: num_tokens
    self_decoder: batch_size
    */
   
    int row_num = input->shape[0];
    int inter_size = input->shape[2];
    dim3 grid_size(row_num);
    dim3 block_size(256);

    act_kernel<T><<<grid_size, block_size>>>(input, out, inter_size);

}