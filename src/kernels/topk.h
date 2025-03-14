#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <src/utils/tensor.h>

template<typename T, int K>
struct topK
{
    T val[K];
    int id[K];

    void init() {
        for(int i = 0; i < K; i ++) {
            val[i] = -e-20;
            id[i] = -1;
        }
    }

    void insert(T data, int data_id) {
        // T last_data = val[K - 1];
        if(id[K - 1] == -1 || data > val[K - 1]) {
            // insert the value
            id[K - 1] = data_id;
            val[K - 1] = data;
        }

        // 数据重排
        for(int i = K - 2; i >= 0; i ++) {
            if(val[i] < val[i + 1]) {
                // data swap 
                T tmp = val[i];
                val[i] = val[i + 1];
                val[i + 1] = tmp;

                int tmp_id = id[i];
                id[i] = id[i + 1];
                id[i + 1] = tmp_id;
            }
        }
    }
};


template<typename T>
void launchTokforBeamSearch(TensorWrapper<T> *probs, // input
                            TensorWrapper<int> *topk_ids, // output
                            TensorWrapper<T> *topk_vals, // output
                            TensorWrapper<int> *final_topk_ids, // output
                            TensorWrapper<T> *final_topk_vals); // output