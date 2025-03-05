#pragma once
#include <cuda.h>
#include <cuda_fp16.h>

template<typename T>
struct Vec {
    using Type = T;
    static constexpr int size = 0;
};

// float32类型特化
template<>
struct Vec<float> {
    using Type = float4;
    static constexpr int size = 4;
};