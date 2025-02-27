#include <iostream>
#include <vector>
#include <numeric>
#include <unordered_map>
#include <src/utils/string_utils.h>
#include <src/utils/check.h>


enum Device
{
    CPU,
    GPU
};

enum DataType
{
    FP32,
    FP16,
    INT32,
    INT8,
    BOOL,
    BYTES,
    UNKOWN
};

template<typename T>
DataType getTensorType()
{
    if (std::is_same<T, float>::value || std::is_same<T, const float>::value) {
        return FP32;
    }
    else if (std::is_same<T, half>::value || std::is_same<T, const half>::value) {
        return FP16;
    }
    else if (std::is_same<T, int>::value || std::is_same<T, const int>::value) {
        return INT32;
    }
    else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value) {
        return INT8;
    }
    else if (std::is_same<T, bool>::value || std::is_same<T, const bool>::value) {
        return BOOL;
    }
    else if (std::is_same<T, char>::value || std::is_same<T, const char>::value) {
        return BYTES;
    }
    else {
        return UNSUPPORTED;
    }
}

// The TensorWarpper is convinent for the implementation of Tensor-Map
template<typename T>
class TensorWarpper;

struct Tensor {
    Device device;  // The device of the Tensor located
    DataType dtype; // The DataType of the Tensor
    std::vector<int> shape; // The shape of the Tensor

    // 构造函数
    Tensor() = default;
    Tensor(const Device _device, const DataType _dtype, const std::vector<int> _shape): device(_device), dtype(_dtype), shape(_shape) {}
    
    virtual int size() const {
        if(shape.size() == 0) return 0;
        return std::accumulate(shape.begin(), shape.end(), int(1), std::multiplies<int>()); 
    }

    template<typename T>
    TensorWarpper<T>* as() {
        return static_cast<TensorWarpper<T>*>(this);
    }

    std::string DeviceString() {
        return device == CPU ? "CPU" : "GPU";
    }

    virtual std::string toString() {
        std::string device_str = DeviceString();
        std::unordered_map<DataType, std::string> dtype2str {
            {FP32, "Float32"},
            {FP16, "Float16"},
            {INT32, "Int32"},
            {INT8, "Int8"},
            {BOOL, "Bool"},
            {BYTES, "Bytes"},
            {UNKOWN, "Unkown"}
        };

        return fmtstr("Tensor[device=%s, dtype=%s, shape=%s]", device_str.c_str(), dtype2str[dtype].c_str(), vec2str(shape).c_str());
    }

};


template<typename T>
class TensorWarpper: public Tensor {
    T* data;
    int size_cache = -1;

    // 构造函数
    TensorWarpper() = default;
    TensorWarpper(Device _device, DataType _dtype, std::vector<int> _shape): Tensor(_device, _dtype, _shape) {}
    TensorWarpper(Device _device, DataType _dtype, std::vector<int> _shape, T* _data): Tensor(_device, _dtype, _shape), data(_data) {
        DataType in_dtype = getTensorType<T>();
        LLM_CHECK_WITH_INFO(in_dtype == _dtype, "when build TensorWrapper, the passed in data type should be same as dtype in params");
    }

    virtual int size() const {
        if(~size_cache) return size_cache;

        if(data == nullptr || shape.size() == 0) size_cache = 0;
        else size_cache =  std::accumulate(shape.begin(), shape.end(), int(1), std::multiplies<int>());
        
        return size_cache;
    }

    // get val
    T* getVal(int index) {
        LLM_CHECK(device == CPU);
        return data[index];

    }

    // get ptr
    T* getPtr() {
        return (T*) data; // 和 return data 有区别吗？
    }

    // get ptr by offset
    T* getPtrByOffset(int offset) {
        int sz = ~i ? size_cache : size();
        LLM_CHECK(offset < sz);
        return (T*)(data + offset);
    }

};

struct TensorMap {
    std::unordered_map<std::string, Tensor*> tensor_map_;

    TensorMap() = default;
    TensorMap(std::initializer_list<std::pair<std::string, Tensor*>> tensor_map){
        for (auto& pair : tensor_map) {
            if (isValid(pair.second)) {
                insert(pair.first, pair.second);
            }
            else {
                // std::cout << "this is not a valid tensor, skip to insert into tensormap" << std::endl;
                LLM_CHECK_WITH_INFO(isValid(pair.second),fmtstr("%s is not a valid tensor, skipping insert into TensorMap", pair.first.c_str()));
            }
        }
    }

    TensorMap(const std::unordered_map<std::string, Tensor*>& tensor_map) {
        // C++ 11 traverse
        // for (auto& kv : tensor_map) {
        // C++ 98 traverse
        for(auto it = tensor_map_.begin(); it != tensor_map_.end(); it++) {
            // if (isValid(kv.second)) {
            //     insert(kv.first, kv.second);
            // }
            if (isValid(it->second)) {
                insert(it->first, it->second);
            }
            else {
                // TODO: add a reminder info
            }
        }        
    };

    ~TensorMap(){
        tensor_map_.clear();
    }

    inline size_t size() const
    {
        return tensor_map_.size();
    }

    inline bool isExist(const std::string& key) const
    {
        return tensor_map_.find(key) != tensor_map_.end();
    }

    inline bool isValid(const Tensor* tensor)
    {
        return tensor->size() > 0;
    }
    // 增
    inline void insert(const std::string& key, Tensor* value)
    {
        // TODO: add a check to check key is unique and value is valid
        // tensor_map_.insert({key, value});
        tensor_map_[key] = value;
    }

    inline void insert(std::pair<std::string, Tensor*> p)
    {
        tensor_map_.insert(p);
    }
    //删

    //改
    inline void modify(std::pair<std::string, Tensor*> p)
    {
        if(isValid(p.second)) tensor_map_[p.first] = p.second;
    }

    //查
    inline Tensor* at(const std::string& key)
    {
         // TODO: add a check to check key is existed
        LLM_CHECK_WITH_INFO(isExist(key), fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key);
        
    }

    // 运算符重载
    inline Tensor* operator[](const std::string& key)
    {
        LLM_CHECK_WITH_INFO(isExist(key), fmtstr("Cannot find a tensor of name %s in the tensor map    (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key);
    }
    
    std::vector<std::string> keys() const
    {
        std::vector<std::string> key_names;
        for (auto& kv : tensor_map_) {
            key_names.push_back(kv.first);
        }
        return key_names;
    }

    std::string toString()
    {
        std::stringstream ss;
        ss << "{";
        std::vector<std::string> key_names = keys();
        for (size_t i = 0; i < tensor_map_.size(); ++i) {
            ss << key_names[i] << ": " << at(key_names[i])->toString();
            if (i < tensor_map_.size() - 1) {
                ss << ", ";
            }
        }
        ss << "}";
        return ss.str();
    }
};