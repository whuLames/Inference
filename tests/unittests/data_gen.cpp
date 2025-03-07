#include <tests/unittests/data_gen.h>

int main(int argc, char** argv)
{
    std::vector<int> shape;
    for(int i = 1; i < argc; i ++)
    {
        shape.push_back(std::atoi(argv[i]));
    }
    GeneRdData gen_data(1, 3, 1.0f, 2.0f);
    int* int_data;
    float* real_data;
    int int_size = gen_data.gen_int_data(shape, (int**)&int_data);
    int real_size = gen_data.gen_real_data(shape, (float**)&real_data);
    std:: cout << "size: " << int_size << " real size: " << real_size << std::endl;

    for(int i = 0; i < int_size; i ++) std::cout << int_data[i] << std::endl;
    for(int i = 0; i < real_size; i ++) std::cout << real_data[i] << std::endl;
}