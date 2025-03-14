#include <src/kernels/sampling.h>

template<typename T>
__global__ void sampling_kernel(int* topk_id,
                                T* topk_val, //[bs, K] from topK
                                int* output_id, //[bs]
                                int* seqlen, //cumulated seq len,[bs]
                                bool* is_finished, //[bs]
                                int K,
                                int rand_num, // step
                                int end_id, // when initialize llama model, we will init it, and this is a fixed val
                                int vocab_size)
{
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    if(is_finished[batch_id]) return;

    int max_val = topk_val[batch_id * K];
    int index = batch_id * K + tid;
    int val = topk_val[index];
    topk_val[index] = (T)expf((float)val - (float)max_val);

    // samping only for the thread whose id is 0
    if(tid == 0) {
        float sum, threshold;
        for(int i = 0; i < K; i ++) sum += topk_val[batch_id * K + i];
        // 生成随机数
        curandState_t state;
        curand_init((unsigned long long)rand_num,(unsigned long long)bid, (unsigned long long)0, &state);
        threshold = (float)curand_uniform(&state) * sum; // for a block
        output_id[batch_id] = topk_id[batch_id * K] % vocab_size;  // 初始化
        for(int i = 0; i < K; i ++) {
            threshold -= topk_id[batch_id * K + i];
            if(threshold < 0) {
                output_id[batch_id] = topk_id[batch_id * K + i] % vocab_size;
                break;
            }
        }
        seqlen[batch_id] = is_finished[batch_id] ? seqlen[batch_id] : seqlen[batch_id] + 1;
        is_finished[batch_id] = output_id[batch_id] == end_id ? true : false;
    }
}
template<typename T>
void launchSampling(TensorWrapper<int>* topk_id,
                    TensorWrapper<T>* topk_val,
                    TensorWrapper<int>* seqlen,
                    TensorWrapper<bool>* is_finished,
                    TensorWrapper<int>* output_id,
                    IntDict& params)
{
    /*
    topk_id: 每个batch中topk元素的id  [bs, k]
    topk_val: 每个batch中topk元素的value [bs, k]
    */
    int batch_size = topk_id->shape[0];
    int K = topk_id->shape[1];
    int vocab_size = params["vocab_size"];
    int step = params["step"];
    int end_id = params["end_id"];

    dim3 grid_size(batch_size);
    dim3 block_size(K);
    sampling_kernel<T><<<grid_size, block_size>>>(topk_id->data,
                                                  topk_val->data,
                                                  seqlen->data,
                                                  is_finished->data,
                                                  K,
                                                  step,
                                                  end_id,
                                                  vocab_size);
}

// 模板实例化
template void launchSampling(TensorWrapper<int>* topk_id,
                             TensorWrapper<float>* topk_val,
                             TensorWrapper<int>* seqlen,
                             TensorWrapper<bool>* is_finished,
                             TensorWrapper<int>* output_id,
                             IntDict& params);