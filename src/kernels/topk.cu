#include <src/kernels/topK.h>
#include <cub/cub.cuh>

template<typename T, int K>
__device__ topK<T,K> reduceFunc(const topK<T, K> a, const topK<T, K> b)
{
    topK<T, K> res = a;
    for(int i = 0; i < K; i ++) {
        res.insertHeap(b.val[i], b.id[i]);
    }
    return res;
}

template<typename T, int K, int BlockSize, int BlockPerBeam>
__global__ void topK_kernel_1(T* probs, int vocab_size, int* tmp_ids, T* tmp_vals)
{
    /*
    probs: [bs, beam_width, vocab_size]
    tmp_ids & tmp_vals: [bs, beam_width, blockPerBeam, K]
    grid_size: (bs * beam_width * blockPerBeam)
    block_size:  256
    */
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int batch_id = bid / BlockPerBeam; // batch_id 也是 row_id
    int lane_id = bid % BlockPerBeam; 
    topK<T, K> threadTopk;
    threadTopk.init();
    
    int batch_offset = beam_width * vocab_size;
    // build the topK within the same thread
    for(int data_index = lane_id * blockSize + tid; data_index < vocab_size; data_index += BlockSize * BlockPerBeam) {
        int data_offset = batch_offset * batch_id + data_index
        T data = probs[data_offset];
        threadTopk.insertHeap(data, data_offset); // 这里的id是batch化下的整体id 或者 id == data_index, 即每个batch内部的ID
    }
    typedef cub::BlockReduce<topK<T, K>, blockSize> blockreduce;
    __shared__ typename blockreduce::TempStorage tmp_storage;
    topK<T, K> block_topk = blockreduce(tmp_storage).Reduce(thread_topK, reduce_functor<T, K>); // 借助cub实现自定义的Reduce

    // blockreduce 的结果
    if(tid == 0) {
        for(int i = 0; i < K; i ++) {
            int dst_offset = batch_id * BlockPerBeam * K + lane_id * K + i;
            tmp_ids[dst_offset] = block_topk.id[i];
            tmp_vals[dst_offset] = block_topk.val[i];
        }
    }


}

template <typename T, int K, int BlockSize, int BlockPerBeam>
__global__ void topK_kernel_2(int* tmp_ids, T* tmp_vals, int final_ids, T* final_vals)
{
    /*
    tmp_ids & tmp_vals: [bs, beam_width, BlockPerBeam, K]
    final_ids & final_vals: [bs, K]
    grid_size: bs * beam_width
    */

    typedef cub::BlockReduce<topK<T, K>, blockSize> blockreduce;
    __shared__ typename blockreduce::TempStorage tmp_storage;
    int tid = threadIdx.x;
    int bid = blockIdx.x; // batch_id

    topK<T, K> threadTopk;
    threadTopk.init();
    for(int i = tid; i < BlockPerBeam * k; i += blockSize) {
        int index = bid * BlockPerBeam * K + tid;  // 没有考虑 beam_width的长度
        threadTopk.insertHeap(tmp_ids[index], tmp_vals[index]);
    }

    topK<T, K> block_topk = blockreduce(tmp_storage).Reduce(thread_topK, reduce_functor<T, K>);

    if(tid == 0) {
        for(int i = 0; i < K; i ++) {
            int dst_offset = bid * K + i;
            final_ids[dst_offset] = block_topk.id[i];
            final_vals[dst_offset] = block_topk.val[i];
        }
    }
}


template<typename T>
void launchTokforBeamSearch(TensorWrapper<T>* probs, // input
                            TensorWrapper<int>* topk_ids, // output
                            TensorWrapper<T>* topk_vals, // output
                            TensorWrapper<int>* final_topk_ids, // output
                            TensorWrapper<T>* final_topk_vals) // output
{
    /*
    probs: [bs, beam_width, vocab_size]
    topk_ids: [bs, beam_width, block_per_beam, K]
    topk_vals: [bs, beam_width, block_per_beam, K]
    final_topk_ids: [bs, beam_width, K]
    final_topk_vals: [bs, beam_width, K]
    */
    int batch_size = probs->shape[0];
    int beam_width = 1;
    int vocab_size = probs->shape[2];
    constexpr int BlockPerBeam = 8; // why 8 ?
    constexpr int K = 5;
    int* tmp_ids = topk_ids->data;
    T* tmp_vals = topk_vals->data;
    int* final_ids = final_topk_ids->data;
    T* final_vals = final_vals->data;

    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int max_block_num = deviceProp.maxGridSize[0]; 

    int block_nums_1 = std::min(batch_size * beam_width * BlockPerBeam, max_block_num); // 完全没必要啊
    int block_nums_2 = std::min(batch_size * beam_width, max_block_num);

    dim3 grid_size_1(block_nums_1);
    dim3 grid_size_2(block_nums_2);
    dim3 block_size(256);

    topK_kernel_1<T, K, 256, BlockPerBeam><<<grid_size_1, block_size>>>(probs->data, vocab_size, tmp_ids, tmp_vals);
    topK_kernel_2<T, K, 256, BlockPerBeam><<<grid_size_2, block_size>>>(tmp_ids, tmp_vals, final_ids, final_vals);
}