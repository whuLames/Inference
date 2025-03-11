#include <src/utils/tensor.h>
#include <src/utils/check.h>
#include <src/weights/base_weights.h>
#include <src/utils/vectorize_utils.h>

template<typename T>
void launchAddresidualAndRMSNorm(TensorWrapper<T>* residual,
                                 TensorWrapper<T>* decoder_out,
                                 BaseWeight<T>* bias,
                                 T* gamma, // 这里就不封装为LayerNromWeight了
                                 float eps);