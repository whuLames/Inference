#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

class BaseAllocator
{
public:
    virtual ~BaseAllocator(){}; // 析构函数建议声明为虚函数
    
    virtual void UnifyFree();

    template<typename T>
    T* Malloc(T* ptr, size_t size, bool is_host) {
        return (T*)UnifyMalloc((void*)ptr, size, is_host);
    }
    virtual void* UnifyMalloc(void* ptr, size_t size, bool is_host=false) = 0; // 纯虚函数, 子类必须给出实现

    template<typename T>
    void Free(T* ptr, bool is_host=false) {
        UnifyFree((void*)ptr, is_host);
    }
    virtual void UnifyFree(void* ptr, bool is_host) = 0;
}; 