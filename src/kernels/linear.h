#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include <src/kernels/cublas.h>
#include <src/utils/tensor.h>
#include <src/weights/base_weights.h>
#include <src/utils/check.h>

/// @brief Input is A, weights is B, output is C
/// @tparam T 
/// @param input 
/// @param weights 
/// @param output 
/// @param culbas_wrapper 
/// @param trans_a 
/// @param tran_b 
template<typename T>
void launchLinearGemm(TensorWrapper<T>* input, TensorWrapper<T>* weights, TensorWrapper<T>* output,
                        cublasWrapper* cublas_wrapper, bool trans_a = false, bool tran_b = false);


template<typename T>
void launchLinearStrideBatchGemm(TensorWrapper<T>* input1, TensorWrapper<T>* input2, TensorWrapper<T>* output,
                                    cublasWrapper* cublas, bool trans_1 = false, bool trans_2 = false);