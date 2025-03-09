#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <src/utils/tensor.h>
#include <src/utils/check.h>
#include <src/utils/vectorize_utils.h>

template<typename T>
void launchMaskScaleAndSoftMax(TensorWrapper<T>* qk,
                               TensorWrapper<T>* mask,
                               TensorWrapper<T>* attn_score,
                               float scale);

