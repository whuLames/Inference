#include <src/weights/weights_utils.h>
#include <src/utils/check.h>
#include <vector>
/// @brief The warpped GPU Malloc function
/// @tparam T 
/// @param ptr 
/// @param size in byte
template<typename T>
void GPUMalloc(T** ptr, size_t size) 
{
    LLM_CHECK_WITH_INFO(size >= (size_t)0, "The Malloc size " + std::to_string(size) + " is invalid");
    CHECK(cudaMalloc((void**)ptr, size));
}

template<typename T>
void GPUFree(T* ptr) 
{
    if(ptr != nullptr) {
        CHECK(cudaFree(ptr));
        ptr = nullptr;
    }
}
// 模板实例化
template void GPUMalloc(float** ptr, size_t size);
template void GPUFree(float* ptr);

template<typename T>
void cudaH2Dcpy(T* tgt, const T* src, const size_t size)
{
    CHECK(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyHostToDevice));
}

template void cudaH2Dcpy(float* tgt, const float* src, const size_t size);
// template void cudaH2Dcpy(half* tgt, const half* src, const size_t size);


/// @brief read data from given binary file as the dtype T
/// @tparam T 
/// @param file_name 
/// @param shape
/// @return The read res stored in vector<T>
template<typename T>
std::vector<T> loadWeightsFromBin(std::vector<int> shape, std::string file_name) {
    std::cout << "file name: " << file_name << std::endl;
    if(shape.size() > 2 || shape.size() == 0) {
        std::cout << "The dim of shape must less than 2 and more than 0" << std::endl;
        // return res;
        return std::vector<T>();
    }
    
    long ele_cnt = shape.size() == 1 ? shape[0] : shape[0] * shape[1];

    std::ifstream in_file(file_name, std::ios::in | std::ios::binary);
    if(!in_file) {std::cout << "The file: " << file_name << "Open Failed" << std::endl; return std::vector<T>();}

    // get the size of the binary file
    in_file.seekg(0, std::ios::end);
    std::streamsize file_size = in_file.tellg();
    in_file.seekg(0, std::ios::beg);
    std::cout << "file size: " << file_size << std::endl;

    if(std::is_same<float, T>::value) std::cout << "The data type is: float" << std::endl; 
    if(sizeof(T) * ele_cnt != file_size) {
        std::cout << "The required size dont match the file size !" << std::endl;
        std::cout << "Ele cnt is : " << ele_cnt << " Required size is: " << sizeof(T) * ele_cnt << " file size is: " << file_size << std::endl;  
        return std::vector<T>();
    }
    std::vector<T> res(ele_cnt);
    // res.resize(ele_cnt);

    // read data
    in_file.read((char*)res.data(), file_size);
    T* p = res.data();
    for(int i = 0; i < res.size(); i ++) std::cout << p[i] << " ";
    std::cout << std::endl;
    // close the ifstream
    in_file.close();

    return res;
}

/*
LoadWeightFromBin 偏特化
*/

// 不需要类型转换
template<typename T_OUT, typename T_FILE>
struct LoadWeightFromBin<T_OUT, T_FILE, true> {
public:
    /// @brief The weight load function
    /// @param ptr : a gpu ptr point to the data
    /// @param shape 
    /// @param file_path 
    static void internalFun(T_OUT* ptr, std::vector<int>shape, std::string file_path) {
        std::vector<T_OUT> host_array = loadWeightsFromBin<T_OUT>(shape, file_path);
        if(host_array.size() == 0) return ;
        size_t malloc_size = host_array.size() * sizeof(T_FILE);
        // for(auto val : host_array) std::cout << val << " ";
        T_OUT* p = host_array.data();
        // for(int i = 0; i < host_array.size(); i ++) std::cout << p[i] << " ";
        // ptr = p;
        // std::cout << "address of ptr: " << ptr << std::endl;
        // std::cout << "address of vector: " << host_array.data() << std::endl;
        // return;
        
        // GPU Memory Allocate
        GPUMalloc<T_OUT>(&ptr, malloc_size);
        // Memory Copy
        cudaH2Dcpy(ptr, host_array.data(), malloc_size);
    }
    static void print(int val) {
        std::cout << "This is 偏特化 For true" << std::endl;
        std::cout << "val: " << val << std::endl;
    }
};

// 需要类型转换
template<typename T_OUT, typename T_FILE>
struct LoadWeightFromBin<T_OUT, T_FILE, false> {
public:
    static void internalFun(T_OUT* ptr, std::vector<int> shape, std::string file_path) {
        //TODO: add the implementation when T_OUT not as same as T_FILE
        std::cout << "偏特化 FOR FALSE" << std::endl;
        return ;
    }
    static void print(int val) {
        std::cout << "This is 偏特化 For false" << std::endl;
        std::cout << "val: " << val << std::endl;
    }
};

// structure 实例化
template struct LoadWeightFromBin<float, float, true>;
template struct LoadWeightFromBin<float, int, false>;