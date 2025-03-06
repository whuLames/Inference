#include <iostream>

template<typename T1, typename T2, bool isSame = std::is_same<T1, T2>::value>
struct base {
public:
    static void print();
};

//模板偏特化

template<typename T1, typename T2>
struct base<T1, T2, true> {
public:
    static void print() {
        std::cout << "偏特化 for True " << std::endl;
    }
};

template<typename T1, typename T2>
struct base<T1, T2, false> {
public:
    static void print() {
        std::cout << "偏特化 for False" << std::endl;
    }
};

template struct base<float, float, true>;
template struct base<float, int, false>;

