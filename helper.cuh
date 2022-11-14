#include <cstdio>
#include <cfloat>
#include <cuda_runtime.h>
#include <curand.h>
#include <iostream>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
// Helper function to convert row major to column major
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))

namespace helper
{

    __device__ int d_scale = 28; // 28 30;
    template <typename D>
    class my_data
    {

    public:
        int size_;
        D *d_ptr_;
        my_data()
        {
            d_ptr_ = NULL;
            size_ = 0;
        }
        ~my_data()
        {
            if (d_ptr_ != NULL)
                cudaFree(d_ptr_);
        }
        void init(int size)
        {

            size_ = size * sizeof(D);
            //  h_ptr_ = (D *)new D[size_];
            cudaMalloc(&d_ptr_, size_);
            // int ctr=0;
        }
    };

    __global__ void A_2_int8(float *d_ptr_float, int8_t *d_ptr_int, int elements)
    {
        int gl_id = threadIdx.x + blockDim.x * blockIdx.x;
        if (gl_id < elements)
        {
            d_ptr_int[gl_id] = (__float2int_rn(d_ptr_float[gl_id] * d_scale));
            int x = __float2int_rn(d_ptr_float[gl_id] * d_scale);
            if (x != d_ptr_int[gl_id])
            {
                d_ptr_int[gl_id] = 127;
                if (x < 0)
                    d_ptr_int[gl_id] = -127;
                //    printf("\n id:%d,  float:  %f, int8:%d, float2int:%d", gl_id, d_ptr_float[gl_id], d_ptr_int[gl_id], __float2int_rn(d_ptr_float[gl_id] * d_scale));
                //  if ()
            }
        }
    }
    template <typename D>
    __global__ void B_2_int_x(float *d_ptr_float, D *d_ptr_int, int elements)
    {
        int gl_id = threadIdx.x + blockDim.x * blockIdx.x;
        if (gl_id < elements)
        {
            d_ptr_int[gl_id] = (D)(d_ptr_float[gl_id]);
            // if (gl_id < 10)
            //   printf("\n %d", d_ptr_int[gl_id]);
        }
    }

    template <typename D>
    __global__ void print_gpu(D *mat, int32_t *mat2)
    {
        printf("\nth:%d, fl:%f, int:%f", threadIdx.x, mat[threadIdx.x], float(mat2[threadIdx.x]) / d_scale);
    }

    my_data<int8_t> A[500], B, B1pas;
    // my_data<int16_t> A_i16[500], B_i16;
    my_data<int32_t> C;

    static const char *cublasGetErrorEnum(cublasStatus_t error)
    {
        switch (error)
        {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";

        default:
            return "<unknown>";
        }
    }
    inline void cublasCheck(cublasStatus_t status, int iLine, const char *szFile)
    {
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr << "Cublas error " << cublasGetErrorEnum(status) << " at line " << iLine << " in file "
                      << szFile << std::endl;
        }
    }
#define cublasCk(call) cublasCheck(call, __LINE__, __FILE__)
    template <typename D>
    void ExIgemmTensor(int m,
                       int n,
                       int k,
                       const int8_t *A,
                       int lda,
                       const int8_t *B,
                       int ldb,
                       D *CC,
                       int ldc, int _precision, cublasHandle_t cublasHandle)
    {
        int alpha = 1;
        float alp = 1;
        float bet = 0;
        int beta = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        // std::cout<<cublasLtGetVersion()<<std::endl;
        cudaEventRecord(start, 0);
        if (typeid(D) == typeid(int8_t))
        {
            cublasCk(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, n, k,
                                  &alpha,
                                  A, CUDA_R_8I, m,
                                  B, CUDA_R_8I, k,
                                  &beta,
                                  CC, CUDA_R_32I, m,
                                  CUDA_R_32I,
                                  // CUBLAS_GEMM_DEFAULT
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        else
        {
            //  cout << "\n FP32 C \n";
            cublasCk(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  m, n, k,
                                  &alp,
                                  A, CUDA_R_8I, m,
                                  B, CUDA_R_8I, k,
                                  &bet,
                                  CC, CUDA_R_32F, m,
                                  CUDA_R_32F,
                                  // CUBLAS_GEMM_DEFAULT
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
        cudaEventRecord(stop, 0);
    }

}