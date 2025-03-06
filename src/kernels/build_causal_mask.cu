#include <src/kernels/build_causal_mask.h>


// 需要注意的一点是，batch中每个输入的context包括其当前轮次对话的输入
template <typename T>
__global__ void buildCasualMask(T* mask, int* seqs_len, int* contexts_len, int max_seq_len, int max_context_len) 
{
    // 一个block负责 一个 batch的 mask 的 building
    int bid = blockIdx.x;  // block id
    int tid = threadIdx.x; // thread id
    int matrix_size_each_seq = max_seq_len * max_context_len;

    int s_len = seqs_len[bid], c_len = contexts_len[bid];
    
    while(tid < matrix_size_each_seq)
    {
        int row_id =  id / max_context_len;
        int col_id =  id % max_context_len;
        int index = matrix_size_each_seq * bid + tid;

        // if(row_id >= s_len) mask[index] = T(0);
        // else 
        // {
        //     if(col_id > max_context_len - max_seq_len + row_id) mask[index] = T(0);
        //     else mask[index] = T(1);
        // } 

        // if(row_id >= s_len || col_id > max_context_len - max_seq_len + row_id) mask[index] = T(0);
        // else mask[index] = T(1);

        // 这样的写法可以进一步避免分支的产生，以避免可能出现的warp_divergence
        bool is_one = row_id < s_len && col_id <= c_len - s_len + row_id;
        mask[index] = static_cast<T>(is_one); 
        tid += blockDim.x;
    }
}

template <typename T>
void launchBuildCasualMask(TensorWarpper<T>* mask, TensorWarpper<int>* seqs_len, TensorWarpper<int>* contexts_len)
{
    int batch_size = mask -> data[0];
    int max_seq_len = mask -> data[1];
    int max_context_len = mask -> data[2];

    dim3 grid_size(batch_size);
    dim3 block_size(256);

    buildCasualMask<T><<<grid_size, block_size>>>(mask -> data, seq_len -> data, context_len -> data, max_seq_len, max_context_len);
}