#include <src/kernels/linear.h>

// 一个矩阵的行存 = 其转置矩阵的
// For a matrix A, the layout of A adopted row-major == A^T adopted the col-major
// 参考链接: https://zhuanlan.zhihu.com/p/646999661
// TODO: 传引用和传指针的区别 ？


// TODO: 为什么这里函数声明

template<typename T>
void launchLinearGemm(TensorWrapper<T>* input, TensorWrapper<T>* weights, TensorWrapper<T>* output,
                        cublasWrapper* cublas_wrapper, bool trans_a, bool trans_b)
{
    /*
    这种写法只考虑在列存情况下, 真正输入的矩阵的形状，在这里直接考虑转置问题，在后面传参的时候就不用考虑转置的问题了。
    个人感觉这种写法更直接，或者说更符合本人目前的思维逻辑
    */

    // int Am = trans_b ? input->shape[0] : input->shape[1];
    // int Ak = trans_b ? input->shape[1] : weights->shape[0];
    // int Bk = trans_a ? weights->shape[0] : weights->shape[1];
    // int Bn = trans_a ? weights->shape[1] : weights->shape[0];
    // int Cm = output -> shape[1];
    // int Cn = output -> shape[0];

    int Am = weights -> shape[1];
    int Ak = weights -> shape[0];
    int Bk = input -> shape[1];
    int Bn = input -> shape[0];
    int Cm = output -> shape[1];
    int Cn = output -> shape[0];

    Bk = input -> shape.size() == 3 ? input->shape[1] * input->shape[2] : input -> shape[1];
    Cm = output->shape.size() == 3 ? output->shape[1] * output->shape[2] : output->shape[1]; 
    int lda = Am;
    int ldb = Bk;
    int ldc = Cm;

    cublasOperation_t transA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublas_wrapper->Gemm(transA,
                         transB,
                         trans_b ? Ak : Am, // m
                         Cn,                // n, when load real weight, lmhead weight is same as pre embedding, which shape = [vocab, hidden], so here should transpose b
                         Bk,
                         weights->data,  // A, cur_input_len is for context decoder lmhead
                         lda,          // lda
                         input->data,  // B
                         ldb,          // ldb
                         output->data, // C
                         ldc,          // ldc
                         1.0f,
                         0.0f);
}


// input1: A input2:B output:C
template<typename T>
void launchLinearStrideBatchGemm(TensorWrapper<T>* input1, TensorWrapper<T>* input2, TensorWrapper<T>* output,
                                    cublasWrapper* cublas, bool trans_1, bool trans_2)
{
    int Am = input2 -> shape[3];
    int Ak = input2 -> shape[2];

    int Bk = input1 -> shape[3];
    int Bn = input1 -> shape[2];

    int Cm = output -> shape[3];
    int Cn = output -> shape[2];

    int lda = Am, ldb = Bk, ldc = Cm;
    // input1[0]*input1[1] = input2[0]*input2[1] = output[0]*output[1]
    int batch_count = output->shape[0] * output->shape[1];
    int64_t stridA = Am * Ak, stridB = Bk * Bn, stridC = Cm * Cn;

    cublasOperation_t transA = trans_2 ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t transB = trans_1 ? CUBLAS_OP_T : CUBLAS_OP_N;

    cublas -> stridedBatchedGemm(transA,
                                        transB,
                                        trans_2 ? Ak : Am,
                                        Cn,
                                        Bk,
                                        input2->data,
                                        lda,
                                        stridA,
                                        input1->data,
                                        ldb,
                                        stridB,
                                        output->C,
                                        ldc,
                                        stridC,
                                        batch_count,
                                        1.0f,
                                        0.0f);
}