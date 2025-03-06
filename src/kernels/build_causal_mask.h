#include <cuda.h>
#include <src/utils/check.h>
#include <src/utils/tensor.h>

/// @brief : Build the casual matrix mask for context enconder
/// @tparam T : The data type
/// @param mask :
/// @param seqs_len ： 每一个句子的 token 长度
/// @param contexts_len: 每一个 batch 对应历史上下文 token 长度(包括当前句子本身)
template<typename T>
void launchBuildCasualMask(TensorWrapper<T>* mask, TensorWrapper<int>* seqs_len, TensorWrapper<int>* contexts_len);