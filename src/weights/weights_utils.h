#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>
#include "src/utils/macro.h"

template<typename T_OUT, typename T_FILE, bool isSame = std::is_same<T_OUT, T_FILE>::value>
struct LoadWeightFromBin {
public:
    static bool internalFun(T_OUT ptr, std::vector<int> std::string file_path);
};