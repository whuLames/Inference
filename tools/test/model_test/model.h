#include <iostream>

template<typename T1, typename T2, bool isSame = std::is_same<T1, T2>::value>
struct base {
public:
    static void print();
};