/*
Get the input embedding for given id
*/

#include <src/utils/tensor.h>
#include <src/utils/check.h>
#include <src/weights/embedding_weights.h>

template <typename T>
void launchEmbeddingFactor(TensorWarpper<int>* input_ids, EmbeddingWeight<T>* embeddingTables, TensorWarpper<T>* out);