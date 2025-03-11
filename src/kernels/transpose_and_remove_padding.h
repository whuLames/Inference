#include <src/utils/tensor.h>
#include <src/utils/check.h>


template<typename T>
void launchTransposeAndRemovePadding(TensorWrapper<T> data_w_padding,
                                     TensorWrapper<int> padding_offset,
                                     TensorWrapper<T> data_wo_padding);