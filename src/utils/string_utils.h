#include <string>
#include <vector>
#include <sstream>
#include <memory>

template <typename T>
inline std::string vec2str(std::vector<T> vec) {
    // convert a 1-dimensional vector to string
    std::stringstream ss;
    int size = vec.size();
    ss << "(";
    if(size != 0) {
        for(int i = 0; i < size; i ++) {
        ss << vec[i] << ", ";
        }
        ss << vec.back();
    }
    ss << ")";
    return ss.str();
}

template<typename... Args>
inline std::string fmtstr(const std::string& format, Args... args)
{
    // This function came from a code snippet in stackoverflow under cc-by-1.0
    //   https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf

    // Disable format-security warning in this function.
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;  // Extra space for '\0'
    if (size_s <= 0) {
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args...);
    return std::string(buf.get(), buf.get() + size - 1);  // We don't want the '\0' inside
}