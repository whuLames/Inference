# Inference 过程中的一些小问题
## qkv_bias_and_ROPE
- 关于向量化读取: float4类型和声明一个struct指针, struct里面有四个float, 这两种读取方式的速度一样吗？ 为什么向量化读取要一次是4个float？是128 byte 这样的参数设置和硬件的规定有关系吗？

- 在Launch函数中，通常class类对象传递指针，而struct对象传递引用，这样做的目的是？

- Llama中的 dynamic_NTK 参数的含义是什么 ？

- 在function的参数传递时，声明某一个参数为const，除了表达上的语义信息，其带来的好处是什么？

- 在CPU上malloc一段空间，如果不赋值的话，其值为0还是随机？