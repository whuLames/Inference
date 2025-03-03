#include <test/template_test.h>

template<typename T>
void TemplateTest<T>::getType() {
    if(std::is_same<int, T>::value) std::cout << "The data type is Int" << std::endl;
    else if (std::is_same<float, T>::value) std::cout << "The data type is Float" << std::endl;
    std::cout << "yes" << std::endl;
}

int main()
{
    TemplateTest<int> test;
    test.getType();
}

