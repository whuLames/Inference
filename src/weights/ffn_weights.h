#include <src/weights/base_weights.h>

/*
attention weights: 存储attention计算相关的weights
*/
template<typename T>
struct FFNweights {
    BaseWeight<T> gate;
    BaseWeight<T> up;
    BaseWeight<T> down;
    BaseWeight<T> gateAndUp;
};