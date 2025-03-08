#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <src/utils/check.h>
#include <src/utils/tensor.h>
#include <src/weights/base_weights.h>
#include <src/utils/vectorize_utils.h> // 用于向量化读取
#include <src/model/llama_params.h>

/// @brief RoPE Computation for self decoder
/// @tparam T 
/// @param qkv_buf 
/// @param step 
/// @param static_params 
template<typename T>
void launchRoPE(TensorWrapper<T>* qkv_buf, TensorWrapper<int>* step, LLaMAAttentionStaticParams& static_params);


/// @brief 
/// @tparam T 
/// @param q_buf 
/// @param k_buf 
/// @param v_buf 
/// @param QKV 
/// @param qkv 
/// @param padding_offset 
/// @param history_length 
/// @param input_length 
/// @param params 
template<typename T>
void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<T>* q_buf,
                                           TensorWrapper<T>* k_buf,
                                           TensorWrapper<T>* v_buf,
                                           TensorWrapper<T>* QKV,
                                           BaseWeight<T>& qkv,
                                           //Tensor* qkv_bias,
                                           TensorWrapper<int>* padding_offset,
                                           TensorWrapper<int>* history_length,
                                           TensorWrapper<int>* input_length,
                                           LLaMAAttentionStaticParams& params);