#include <algorithm>
#include <iostream>
#include <vector>
#include <random>

// 生成host端的随机数据
struct GeneRdData {
    // 这样的写法是错误的，成员变量的初始化不能直接调用函数
    std::random_device rd;
    std::mt19937 gen;
    std::uniform_int_distribution<> dis_int;
    std::uniform_real_distribution<> dis_real;

    // 使用初始化列表的性能更好，避免默认构造函数的调用以及后续的赋值操作
    GeneRdData(int int_l, int int_r, float float_l, float float_r) : gen(rd()) {
        dis_int = std::uniform_int_distribution<>(int_l , int_r);
        dis_real = std::uniform_real_distribution<>(float_l, float_r);
    }

    int gen_int_data(std::vector<int> shape, int** output)
    {
        int size = std::accumulate(shape.begin(), shape.end(), int(1), std::multiplies<int>());
        *output = (int*)malloc(size * sizeof(int));
        for(int i = 0; i < size; i ++)
        {
            (*output)[i] = dis_int(gen);
            std::cout << "int :" << i << "-th: " << (*output)[i] << std::endl;
        }
        return size;
    }

    int gen_real_data(std::vector<int> shape, float** output)
    {
        int size = std::accumulate(shape.begin(), shape.end(), int(1), std::multiplies<int>());
        *output = (float*)malloc(size * sizeof(float));
        for(int i = 0; i < size; i ++)
        {
            (*output)[i] = dis_real(gen);
            std::cout << "real :" << i << "-th: " << (*output)[i] << std::endl;
        }
        return size;
    }
};

// template<typename T>
// __inline__ void rdDataGene(std::vector<int> shape, T* out) {
//     if(shape.size() == 0) return ;
//     int size = std::accumulate(shape.begin(), shape.end(), int(1), std::multiplies<int>());
//     out = (T*)malloc(size * sizeof(T));
//     // 随机生成数据
// }