#include<cuda.h>
#include <iostream>

__global__ void test_kernel()
{
    int tid = threadIdx.x;
    // printf("tid %d, num: %d", tid, blockDim.x);
    if(tid == blockDim.x - 1) printf("The element is %d \n", tid);
}


int main(int argc, char** argv)
{
    int grid_size = 1;
    int block_size = atoi(argv[1]);
    std::cout << "block_size: " << block_size << std::endl;
    test_kernel<<<grid_size, block_size>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    // cudaDeviceSynchronize();
}