#include <cuda.h>
#include <src/weights/norm_weights.h>
#include <src/utils/tensor.h>
#include <src/utils/check.h>
#include <src/utils/vectorize_utils.h>
template<typename T>
void launchRMSNorm(TensorWrapper<T>* in, TensorWrapper<T>* residual, LayerNormWeight<T> norm_weight, float eps);