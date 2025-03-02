#include <src/weights/base_weights.h>

/*
attention weights: 存储attention计算相关的weights
*/
template<typename T>
struct AttentionWeights {
    BaseWeight<T> q;
    BaseWeight<T> k;
    BaseWeight<T> v;
    BaseWeight<T> qkv;
    BaseWeight<T> output;
};