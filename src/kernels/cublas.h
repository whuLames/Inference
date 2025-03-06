#include <src/utils/check.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

/// @brief 封装cublas对象，实现GEMM
class cublasWrapper {
    private:
        cublasHandle_t   cublas_handle_;
        cublasLtHandle_t cublaslt_handle_;     

        cudaDataType_t Atype_;
        cudaDataType_t Btype_;
        cudaDataType_t Ctype_;
        cudaDataType_t computeType_;   
    
    public:
        cublasWrapper(cublasHandle_t cublas_handle_,
                      cublasLtHandle_t cublaslt_handle_);
                      // BaseAllocator* allocator); enable it when we use cublasLt API

        ~cublasWrapper();
        void setFP32GemmConfig();
        void setFP16GemmConfig();
        //for proj matmul
        void Gemm(cublasOperation_t transa,
                cublasOperation_t transb,
                const int         m,
                const int         n,
                const int         k,
                const void*       A,
                const int         lda,
                const void*       B,
                const int         ldb,
                void*             C,
                const int         ldc,
                float             alpha,
                float             beta);
        // for qk*v and q*k
        void stridedBatchedGemm(cublasOperation_t transa,
                                cublasOperation_t transb,
                                const int         m,
                                const int         n,
                                const int         k,
                                const void*       A,
                                const int         lda,
                                const int64_t     strideA,
                                const void*       B,
                                const int         ldb,
                                const int64_t     strideB,
                                void*             C,
                                const int         ldc,
                                const int64_t     strideC,
                                const int         batchCount,
                                float             f_alpha,
                                float             f_beta);
};
