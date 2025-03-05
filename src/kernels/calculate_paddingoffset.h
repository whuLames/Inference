#include <src/utils/tensor.h>
#include <src/utils/check.h>

#include <cuda_runtime.h>

/// @brief 
/// @param padding_offset : the calculated padding_offset to output
/// @param cum_seq_len : the accumulate input sequence length
/// @param input_length : the input sequence length
void launchCalPaddingOffset(TensorWrapper<int>* padding_offset, TensorWrapper<int>* cum_seq_len, TensorWrapper<int>* input_length);