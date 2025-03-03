#include <src/weights/weights_utils.h>
#include <vector>
#include <fstream>
template<typename T_OUT, typename T_FILE>
void testFileReader(std::vector<int> shape, std::string file_name) {
    
    //LoadWeightFromBin<float, float>::internalFun(ptr, shape, file_name);
    // int val = 10;
    // LoadWeightFromBin<T_OUT, T_FILE>::print(val);
    T_OUT* ptr;
    LoadWeightFromBin<T_OUT, T_FILE>::internalFun(ptr, shape, file_name);
    // LoadWeightFromBin<T_OUT, T_FILE>::test(ptr, shape, file_name);
    int size = shape[0] * shape[1];
    std::cout << "size: " << size << std::endl;
    std::cout << "address of ptr: " << ptr << std::endl;
    for(int i = 0; i < size; i ++) std::cout << ptr[i] << " ";
}

std::vector<float> readData(float** p, std::string filename)
{
    std::ifstream in_file(filename, std::ios::in | std::ios::binary);
    in_file.seekg(0, std::ios::end);
    std::streamsize file_size = in_file.tellg();
    in_file.seekg(0, std::ios::beg);
    std::cout << "File size: " << file_size << std::endl;

    int ele_cnt = file_size / 4;
    std::vector<float> res(ele_cnt);

    in_file.read((char*)res.data(), file_size);

    *p = (float*)malloc(sizeof(float) * ele_cnt);
    memcpy(*p, res.data(), sizeof(float) * ele_cnt);
    //for(int i = 0; i < ele_cnt; i ++) std::cout << res[i] << " ";

    for(int i = 0; i < ele_cnt; i ++) std::cout << p[i] << " ";
    // p = ptr;

    std::cout << "done!" << std::endl;
    return res;
}

int main(){ 
    std::vector<int> shape = {5, 15};
    std::string file_name = "/data/coding/Inference/test/test.bin";

    // testFileReader<float, float>(shape, file_name);
    float* p = nullptr;
    readData(&p, file_name);
    
    for(int i = 0; i < 75; i ++) std::cout << p[i] << " ";
    return 0;
}