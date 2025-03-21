#pragma once
#include <unordered_map>
#include <map>
#include <vector>
#include <iostream>

#include <src/utils/check.h>
#include <src/memory/allocator/base_allocator.h>


struct CudaBigBlock 
{
    void* data;
    size_t size;
    bool is_allocated;
    CudaBigBlock() = default;
    CudaBigBlock(void* data_, size_t size_, bool is_allocated_): data(data_), size(size_), is_allocated(is_allocated_) {};
};


struct CudaSmallBlock
{
    void* data;
    size_t size;
    bool is_allocated;
    CudaSmallBlock() = default;
    CudaSmallBlock(void* data_, size_t size_, bool is_allocated_): data(data_), size(size_), is_allocated(is_allocated_) {};
};

class CudaAllocator: public BaseAllocator {
private:
    std::map<int, std::vector<CudaBigBlock>> cudaBigBlockMap;
    std::map<int, std::vector<CudaSmallBlock>> cudaSmallBlockMap;
    std::map<int, size_t> FreeSize;  
    int dev_id;
    size_t total_allocated_size = 0;
public:
    CudaAllocator() {
        cudaGetDevice(&dev_id);
    }
    ~CudaAllocator() {

    };

    void* UnifyMalloc(void* ptr, size_t size, bool is_host) {
        size = (size + 32 - 1) / 32;
        if(is_host) {
            ptr = malloc(size);
            memset(ptr, 0, size);
            return ptr;
        }
        
        // 寻找 big buffer
        if(size > 1024 * 1024) {
            int block_id = -1;
            std::vector<CudaBigBlock> &bigblocks = cudaBigBlockMap[dev_id];
            int big_block_num = bigblocks.size();
            for(int i = 0; i < big_block_num; i ++) {
                if(bigblocks[i].size >= size && !bigblocks[i].is_allocated && bigblocks[i].size - size <= 1024 * 1024) {
                    if(block_id == -1 || bigblocks[block_id].size > bigblocks[i].size) {
                        block_id = i;
                    }
                }
            }

            if(block_id != -1) { // 找到一个符合要求的block
                bigblocks[block_id].is_allocated = true;
                return bigblocks[block_id].data;
            }
            // 没找到符合要求的 big block, 我们需要新建一个
            void* new_buffer;
            cudaMalloc((void**)&new_buffer, size);
            CudaBigBlock new_block(new_buffer, size, false);
            cudaBigBlockMap[dev_id].push_back(new_block);
            total_allocated_size += size;
            return new_buffer;
        }

        // 寻找 small buffer
        auto &smallblocks = cudaSmallBlockMap[dev_id];
        for(int i = 0; i < smallblocks.size(); i ++) {
            if(smallblocks[i].size >= size && !smallblocks[i].is_allocated) {
                smallblocks[i].is_allocated = true;
                FreeSize[i] += smallblocks[i].size; // 未分配的小buffer的大小
                return smallblocks[i].data;
            }
        }

        void* new_buffer;
        CHECK(cudaMalloc(&new_buffer, size));
        smallblocks.push_back(CudaSmallBlock(new_buffer, size, true));
        return new_buffer;
    }

    void UnifyFree(void* ptr, bool is_host) {
        if(ptr == nullptr) return;
        if(is_host) {
            free(ptr);
            return;
        }
        for(auto &it : cudaSmallBlockMap) {
            if (FreeSize[it.first] > 1024 * 1024 * 1024) {
                auto &cudaBlocks = it.second;
                std::vector<CudaSmallBlock> temp;
                for (int i = 0; i < cudaBlocks.size(); i++) {
                    if (!cudaBlocks[i].is_allocated) {
                        cudaSetDevice(it.first);
                        cudaFree(cudaBlocks[i].data);
                    } else {
                        temp.push_back(cudaBlocks[i]);
                    }
                }
                cudaBlocks.clear();
                it.second = temp;
                FreeSize[it.first] = 0;
            }
        }

        for (auto &it: cudaSmallBlockMap) {
            auto &cudaBlocks = it.second;
            for (int i = 0; i < cudaBlocks.size(); i++) {
                if (cudaBlocks[i].data == ptr) {
                    FreeSize[it.first] += cudaBlocks[i].size;
                    cudaBlocks[i].is_allocated = false;
                    return;
                }
            }
            // 若是大block，不归还到OS, 即不加入FreeSize
            auto &bigBlocks = cudaBigBlockMap[it.first];
            for (int i = 0; i < bigBlocks.size(); i++) {
                if (bigBlocks[i].data == ptr) {
                    bigBlocks[i].is_allocated = false;
                    return;
                }
            }
        }

        // 什么时候会出现找不到的情况? 扯淡
        cudaFree(ptr); 
    }
};