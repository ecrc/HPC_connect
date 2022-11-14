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
namespace helper_fp16
{
    __device__ __half H_partial_mul[400];
    __global__ void half_vecMul_v1(float *A, float *gpu_r_adf_in, int nb_node_parallel, int dim1, int pas, int level, float *gpu_y_modulation)
    {
        int tid = threadIdx.x;

        __shared__ __half sh_b[1200];

        int ind = (dim1 * 2) - 2 * level - (2 * pas);
        int j = blockIdx.x % (2 * pas);
        int i = blockIdx.x / (2 * pas);

        __half va = __float2half(gpu_r_adf_in[(j + ind) * dim1 * 2 + tid]);
        __half vb = __float2half(A[i * dim1 * 2 + tid]);

        if (level == 0)
            vb = 0;
        sh_b[tid] = __hmul(va, vb);
        __syncthreads();
        if (2 * dim1 > 512)
        {
            if ((tid < 512))
                if ((tid + 512) < 2 * dim1)
                {

                    sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 512]);
                }
            __syncthreads();
        }
        if (2 * dim1 > 256)
        {
            if ((tid < 256))
                if ((tid + 256) < 2 * dim1)
                {
                    sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 256]);
                }
            __syncthreads();
        }
        if (2 * dim1 > 128)
        {
            if ((tid < 128))
                if ((tid + 128) < 2 * dim1)
                {
                    sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 128]);
                }
            __syncthreads();
        }
        if (2 * dim1 > 64)
        {
            if ((tid < 64))
                if ((tid + 64) < 2 * dim1)
                {
                    sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 64]);
                }
            __syncthreads();
        }
        if (2 * dim1 > 32)
        {
            if ((tid < 32))
                if ((tid + 32) < 2 * dim1)
                {
                    sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 32]);
                }
            __syncthreads();
        }
        if (2 * dim1 > 16)
        {
            if ((tid < 16))
                if ((tid + 16) < 2 * dim1)
                {
                    sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 16]);
                }
            __syncthreads();
        }
        if (2 * dim1 > 8)
        {
            if ((tid < 8))
                if ((tid + 8) < 2 * dim1)
                {
                    sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 8]);
                }
            __syncthreads();
        }
        if (2 * dim1 > 4)
        {
            if ((tid < 4))
                if ((tid + 4) < 2 * dim1)
                {
                    sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 4]);
                }
            __syncthreads();
        }
        if (2 * dim1 > 2)
        {
            if ((tid < 2))
                if ((tid + 2) < 2 * dim1)
                {
                    sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 2]);
                }
            __syncthreads();
        }
        if (2 * dim1 > 1)
        {
            if ((tid < 1))
                if ((tid + 1) < 2 * dim1)
                {
                    sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 1]);
                }
            __syncthreads();
        }

        if (tid == 0)
        {

            H_partial_mul[i * 2 * pas + j] = __hsub(sh_b[0], __float2half(gpu_y_modulation[ind + j]));
        }
    }

    int initialization_half(__half **_C, __half **_A, __half **A_half, __half **B_half, __half **B1_half, int lz, int pas, int parallel_nodes, int dim1, vector<float> float_comb_v, vector<float> float_comb_v2)
    {
        cudaError_t cudaStat1;
        cudaStat1 = cudaMalloc(_C, 2 * lz * pas * 2 * parallel_nodes * sizeof(__half));
        if (cudaStat1 != cudaSuccess)
        {
            printf("device memory allocation failed matrix half C");
            return EXIT_FAILURE;
        }

        for (int i7 = 0; i7 < (dim1 / pas + (dim1 % pas)); i7++)
        {
            cudaStat1 = cudaMalloc(&_A[i7], 2 * pas * pas * 2 * sizeof(__half));
            if (cudaStat1 != cudaSuccess)
            {
                printf("device memory allocation failed half matrix A");
                return EXIT_FAILURE;
            }
        }

        cudaStat1 = cudaMalloc(A_half, 2 * pas * pas * 2 * sizeof(__half));
        if (cudaStat1 != cudaSuccess)
        {
            printf("device memory allocation failed half matrix A");
            return EXIT_FAILURE;
        }

        cudaStat1 = cudaMallocManaged(B_half, float_comb_v.size() * sizeof(__half));
        if (cudaStat1 != cudaSuccess)
        {
            printf("device memory allocation failed half matrix A");
            return EXIT_FAILURE;
        }

        cudaStat1 = cudaMallocManaged(B1_half, float_comb_v2.size() * sizeof(__half));
        if (cudaStat1 != cudaSuccess)
        {
            printf("device memory allocation failed half matrix A");
            return EXIT_FAILURE;
        }
    }

}