/*
Get the input embedding for given id
*/
#include <src/kernels/embedding.h>


template<typename T> 
__global__ void embeddingFactor(const int* input_ids, T* embeddingTables, T* out, int token_num, int hidden_size)
{
    // get the embedding for specific token indicated by input_ids
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    int total_elements = token_num * hidden_size;
    
    while(idx < total_elements)
    {
        int input_id = input_ids[idx / hidden_size];
        int element_id = idx % hidden_size;
        int element = embeddingTables[input_id * hidden_size + element_id];
        out[index] = element; 
        idx += num_threads;
    }
}

template<typename T>
void launchEmbeddingFactor(TensorWarpper<int>* input_ids, EmbeddingWeight<T>* embeddingTables, TensorWarpper<T>* out)
{
    int token_num = input_ids -> shape[0];
    int hidden_size = out -> shape[1];
    dim3 block_size;
    dim3 grid_size;

    block_size.x = 256;
    // grid_size.x = (out->shape[0] * out->shape[1] + block_size.x - 1) / block_size.x to gurantee one threads for one element
    grid_size.x = 2048;
    LLM_CHECK_WITH_INFO(token_num == out -> shape[0], "The inputs shape is not match with the output shape");
    embeddingFactor<T><<<grid_size, block_size>>(input_ids -> data, embeddingTables -> data, out -> data, token_num, hidden_size); 
}

// 实例化
/*
实例化的作用：
1. 如果函数实现放在.cu 或 .cpp文件中, 该函数可能会被其他文件调用，比如在main.cpp中，此时编译时无法链接到main.cpp所需要的模板函数的对应实现，而如果把
模板函数放在头文件中，会把所有可能的模板函数的实例进行实例化，此时主函数链接到所需要的一种实例即可

2. 对于上述函数，其有cuda语义符号 即 <<<>>>，即使其放在.h头文件中，c++语义也无法对其进行处理，所以需要在此进行实例化
*/ 
template void launchEmbeddingFactor(TensorWarpper<int>* input_ids, EmbeddingWeight<float>* embeddingTables, TensorWarpper<float>* out);
