#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>
// #include "src/utils/macro.h"

/*
模板偏特化实现二进制文件读取
偏特化目的: 判断二进制文件类型和输出文件类型是否相符，对应不同实现
因为C++不支持function 偏特化，所以我们使用模板偏特化来实现不同参数组合下的数据读取
 */
template<typename T_OUT, typename T_FILE, bool isSame = std::is_same<T_OUT, T_FILE>::value>
struct LoadWeightFromBin {
public:
    /// @brief Read data from a given binary file
    /// @param ptr 
    /// @param shape 
    /// @param file_path 
    /// @return 
    static bool internalFun(T_OUT* ptr, std::vector<int>shape, std::string file_path);
    static void print(int val);
};