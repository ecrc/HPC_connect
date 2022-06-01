/*
 GPU-multi-level version with Real matrix representation
we are not doing batching here
single precision gemm ... works
half precision gemm .... works .... to enable it uncomment #define FP16MM
qam modulation ..... here
optimised gpu kernels for mat a,b,c  (2*pas x 2*pas)// 2*pas x l111*nb_parallel // 2*pasx *l111*nb_parallel
norme plus sorting kernel
multiple selected nodes at at time
works well
*/
#include <iomanip>
#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>
#include <iostream>
#include <cuComplex.h>
//#include <cublas.h>
#include <boost/multi_array.hpp>
#include <mkl.h>
#include <mkl_lapacke.h>
//#include <list>
#include <vector>
#include <iterator>
#include <cublas_v2.h>
#include <omp.h>
#include <algorithm>
#include <list>

#include <vector>

#include <iterator>

#include <cuda_fp16.h>

#define FP16MM

using namespace std;

int modulation = 2; // 2 4 16 64
int nTx, nRx, Iter = 0, steps = 1, nb_nodes = 300, nb_threads = 1;
int incremental = 1;
int parallel_nodes; // nodes treated in parallel and sorting size
vector<cuFloatComplex> BPSK, qam4, qam16, qam64;
vector<cuFloatComplex> Constellation_symbols;
int strategy_exploration = 1;
int sort_size = 10;
struct timeval mul1, mul11, mul22, mul2;
struct position
{
    int indice;
    float score;
};

class tree_node
{
public:
    vector<int8_t> s;
    int level = 0;
    float evaluation = 0;
    tree_node() {}
    tree_node(vector<int8_t> ss, int lev, float eval)
    {
        s = ss;
        level = lev;
        evaluation = eval;
    }
};
// FP16MM
__device__ float partial_mul[400];
__device__ __half H_partial_mul[400];
__global__ void vecMul_v1(float *A, float *gpu_r_adf_in, int nb_node_parallel, int dim1, int pas, int level, float *gpu_y_modulation)
{
    int tid = threadIdx.x;
#ifndef FP16MM
    __shared__ float sh_b[1200];
#else
    __shared__ __half sh_b[1200];
#endif

    int ind = (dim1 * 2) - 2 * level - (2 * pas);
    int j = blockIdx.x % (2 * pas);
    int i = blockIdx.x / (2 * pas);
#ifndef FP16MM
    float va = gpu_r_adf_in[(j + ind) * dim1 * 2 + tid];
    float vb = A[i * dim1 * 2 + tid];
#else
    __half va = __float2half(gpu_r_adf_in[(j + ind) * dim1 * 2 + tid]);
    __half vb = __float2half(A[i * dim1 * 2 + tid]);
#endif

    if (level == 0)
        vb = 0;
#ifndef FP16MM
    sh_b[tid] = va * vb;
#else
    sh_b[tid] = __hmul(va, vb);
#endif
    __syncthreads();
    if (2 * dim1 > 512)
    {
        if ((tid < 512))
            if ((tid + 512) < 2 * dim1)
            {
#ifndef FP16MM
                sh_b[tid] = sh_b[tid] + sh_b[tid + 512];
#else
                sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 512]);
#endif
            }
        __syncthreads();
    }
    if (2 * dim1 > 256)
    {
        if ((tid < 256))
            if ((tid + 256) < 2 * dim1)
            {
#ifndef FP16MM
                sh_b[tid] = sh_b[tid] + sh_b[tid + 256];
#else
                sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 256]);
#endif
            }
        __syncthreads();
    }
    if (2 * dim1 > 128)
    {
        if ((tid < 128))
            if ((tid + 128) < 2 * dim1)
            {
#ifndef FP16MM
                sh_b[tid] = sh_b[tid] + sh_b[tid + 128];
#else
                sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 128]);
#endif
            }
        __syncthreads();
    }
    if (2 * dim1 > 64)
    {
        if ((tid < 64))
            if ((tid + 64) < 2 * dim1)
            {
#ifndef FP16MM
                sh_b[tid] = sh_b[tid] + sh_b[tid + 64];
#else
                sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 64]);
#endif
            }
        __syncthreads();
    }
    if (2 * dim1 > 32)
    {
        if ((tid < 32))
            if ((tid + 32) < 2 * dim1)
            {
#ifndef FP16MM
                sh_b[tid] = sh_b[tid] + sh_b[tid + 32];
#else
                sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 32]);
#endif
            }
        __syncthreads();
    }
    if (2 * dim1 > 16)
    {
        if ((tid < 16))
            if ((tid + 16) < 2 * dim1)
            {
#ifndef FP16MM
                sh_b[tid] = sh_b[tid] + sh_b[tid + 16];
#else
                sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 16]);
#endif
            }
        __syncthreads();
    }
    if (2 * dim1 > 8)
    {
        if ((tid < 8))
            if ((tid + 8) < 2 * dim1)
            {
#ifndef FP16MM
                sh_b[tid] = sh_b[tid] + sh_b[tid + 8];
#else
                sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 8]);
#endif
            }
        __syncthreads();
    }
    if (2 * dim1 > 4)
    {
        if ((tid < 4))
            if ((tid + 4) < 2 * dim1)
            {
#ifndef FP16MM
                sh_b[tid] = sh_b[tid] + sh_b[tid + 4];
#else
                sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 4]);
#endif
            }
        __syncthreads();
    }
    if (2 * dim1 > 2)
    {
        if ((tid < 2))
            if ((tid + 2) < 2 * dim1)
            {
#ifndef FP16MM
                sh_b[tid] = sh_b[tid] + sh_b[tid + 2];
#else
                sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 2]);
#endif
            }
        __syncthreads();
    }
    if (2 * dim1 > 1)
    {
        if ((tid < 1))
            if ((tid + 1) < 2 * dim1)
            {
#ifndef FP16MM
                sh_b[tid] = sh_b[tid] + sh_b[tid + 1];
#else
                sh_b[tid] = __hadd(sh_b[tid], sh_b[tid + 1]);
#endif
            }
        __syncthreads();
    }

    if (tid == 0)
    {

#ifndef FP16MM
        partial_mul[i * 2 * pas + j] = sh_b[0] - gpu_y_modulation[ind + j];
#else
        H_partial_mul[i * 2 * pas + j] = __hsub(sh_b[0], __float2half(gpu_y_modulation[ind + j]));
#endif
    }
}

__global__ void print_gpu_mat_colomn_major(float *mat, int nbrows, int nb_columns)
{ // one thread only
    printf("\ncolomn major gpu begin\n");
    for (int i = 0; i < nbrows; i++)
    {
        printf("\n");
        for (int j = 0; j < nb_columns; j++)
        {
            printf(" %f", mat[j * nbrows + i]);
        }
    }
    printf("\ninside the gpu end\n");
}

__global__ void prepare_mat_A(float *vec, float *d_mat_B, int nb_copies, int level, int pas, int dim1, int dim2, cuFloatComplex *gpu_comb, float *gpu_r_adf_in, float *d_matrix_A, float *d_matrix_A1, float *y_cpu_modulation, int modulation, float *d_matrix_C, int nb_node_parallel)
{
    int gl_id = threadIdx.x + blockDim.x * blockIdx.x;
    // int tid = threadIdx.x;
    // if(tid==0)for (int i=0;i<10;i++)printf("pmul:%f ", partial_mul[i]);
    if (gl_id < 2 * dim1)
    {

        int ind = (dim1 * 2) - 2 * level - (2 * pas);

        for (int j = 0; j < 2 * pas; j++)
        {
            int target = (j + ind) * dim1 * 2 + gl_id;
            d_matrix_A1[gl_id * 2 * pas + j] = gpu_r_adf_in[target];
        }
    }
}
__global__ void p_new_mat_A(float *new_mat_A, int level, int pas, int dim1, int dim2, float *gpu_r_adf_in, __half *A)
{
    int gl_id = threadIdx.x + blockDim.x * blockIdx.x;
    //  int tid = threadIdx.x;

    if (gl_id < 2 * pas)
    {

        int ind = (dim1 * 2) - 2 * level - (2 * pas);

        for (int j = 0; j < 2 * pas; j++)
        {
            int target = (j + ind) * dim1 * 2 + (2 * dim1 - 2 * level - 2 * pas) + gl_id;
#ifndef FP16MM
            new_mat_A[gl_id * 2 * pas + j] = gpu_r_adf_in[target];
#else
            A[gl_id * 2 * pas + j] = __float2half(gpu_r_adf_in[target]);
#endif
        }
    }
}


__global__ void p_new_mat_C(int nb_copies, int level, int pas, int dim1, int dim2, float *y_cpu_modulation, float *new_mat_C, int nb_node_parallel, __half *C)
{

    int gl_id = threadIdx.x + blockDim.x * blockIdx.x;
    int r_p = gl_id % (2 * pas);
    int max_threads = blockDim.x * gridDim.x;
    int nb_b = max_threads / (pas * 2);
    int ind = (dim2 * 2) - 2 * level - (2 * pas);
    float x = 0;

    if (gl_id < (nb_copies * nb_node_parallel) * pas * 2)
        x = y_cpu_modulation[ind + r_p];
    __syncthreads();
    int d = 0;
    int r = 0;
    while (d < nb_copies * nb_node_parallel)
    {
        r = nb_b;

        if (d + nb_b > nb_copies * nb_node_parallel)
        {
            r = (nb_copies * nb_node_parallel) - d;
        }
        if (gl_id < (r)*pas * 2)
        { // x=local[tid];
            __syncthreads();
#ifndef FP16MM
            new_mat_C[d * pas * 2 + gl_id] = x;
#else
            C[d * pas * 2 + gl_id] = __float2half(x);
#endif
        }

        d = d + nb_b;
        // __syncthreads();
    }
}

__global__ void prepare_mat_C(float *vec, float *d_mat_B, int nb_copies, int level, int pas, int dim1, int dim2, float *y_cpu_modulation, float *d_matrix_C, int nb_node_parallel)
{
    int gl_id = threadIdx.x + blockDim.x * blockIdx.x;
    //  int tid = threadIdx.x;
    int r_p = gl_id % (2 * pas);
    int max_threads = blockDim.x * gridDim.x;
    int nb_b = max_threads / (pas * 2);
    int ind = (dim2 * 2) - 2 * level - (2 * pas);
    float x = 0;

    if (gl_id < (nb_copies * nb_node_parallel) * pas * 2)
        x = y_cpu_modulation[ind + r_p];
    __syncthreads();
    int d = 0;
    while (d < nb_copies * nb_node_parallel)
    {
        int r = nb_b;

        if (d + nb_b > nb_copies * nb_node_parallel)
        {
            r = (nb_copies * nb_node_parallel) - d;
        }
        if (gl_id < (r)*pas * 2)
        { // x=local[tid];
            __syncthreads();
            d_matrix_C[d * pas * 2 + gl_id] = x;
        }

        d = d + nb_b;
    }
}
__global__ void prepare_mat_B1(float *vec, float *d_mat_B, float *d_mat_B_1, int nb_copies, int level, int pas, int dim1, int dim2, cuFloatComplex *gpu_comb, float *gpu_r_adf_in, float *d_matrix_A, float *d_matrix_A1, float *y_cpu_modulation, int modulation, float *d_matrix_C, int nb_node_parallel)
{
    int gl_id = threadIdx.x + blockDim.x * blockIdx.x;
    int r_p = gl_id % (2 * dim1);
    int max_threads = blockDim.x * gridDim.x;
    int nb_b = max_threads / (dim1 * 2);
    int p0 = 2 * dim1 - 2 - (2 * level) + 1;
    int p1 = 2 * dim1 - 2 - (2 * level) - 2 * (pas - 1);
    for (int j = 0; j < nb_node_parallel; j++)
    {
        float x = 0;
        if (level != 0)
            x = vec[j * (2 * dim1) + r_p];
        int d = 0;
        while (d < (nb_copies))
        {
            int r = nb_b;

            if (d + nb_b > (nb_copies))
            {
                r = (nb_copies)-d;
            }
            if (gl_id < (r)*dim1 * 2)
            {
                if (r_p >= p1 && r_p <= p0)
                {
                    int position = d + gl_id / (2 * dim1);

                    int i1 = p0 - r_p;

                    float xx;
                    cuFloatComplex l = gpu_comb[(int)(position)*pas + i1 / 2];
                    xx = cuCrealf(l);
                    if (i1 % 2 == 0)
                        xx = cuCimagf(l);
                    x = xx;
                }

                __syncthreads();
                d_mat_B[j * nb_copies * dim1 * 2 + d * dim1 * 2 + gl_id] = x;
            }

            d = d + nb_b;
        }
    }
}

__global__ void prepare_mat_C(float *vec, float *d_mat_B, int nb_copies, int level, int pas, int dim1, int dim2, cuFloatComplex *gpu_comb, float *gpu_r_adf_in, float *d_matrix_A, float *d_matrix_A1, float *y_cpu_modulation, int modulation, float *d_matrix_C, int nb_node_parallel)
{
    int gl_id = threadIdx.x + blockDim.x * blockIdx.x;
    int r_p = gl_id % (2 * pas);
    int max_threads = blockDim.x * gridDim.x;
    int nb_b = max_threads / (pas * 2);
    int ind = (dim2 * 2) - 2 * level - (2 * pas);
    float x = 0;

    if (gl_id < (nb_copies * nb_node_parallel) * pas * 2)
        x = y_cpu_modulation[ind + r_p];
    __syncthreads();
    int d = 0;
    while (d < nb_copies * nb_node_parallel)
    { 
        int r = nb_b;

        if (d + nb_b > nb_copies * nb_node_parallel)
        {
            r = (nb_copies * nb_node_parallel) - d;
        }
        if (gl_id < (r)*pas * 2)
        {
            __syncthreads();
            d_matrix_C[d * pas * 2 + gl_id] = x;
        }

        d = d + nb_b;
    }
}

__global__ void prepare_matrices(float *vec, float *d_mat_B, int nb_copies, int level, int pas, int dim1, int dim2, cuFloatComplex *gpu_comb, float *gpu_r_adf_in, float *d_matrix_A, float *d_matrix_A1, float *y_cpu_modulation, int modulation, float *d_matrix_C, int nb_node_parallel)
{

    int gl_id = threadIdx.x + blockDim.x * blockIdx.x;

    if (gl_id < 2 * dim1)
    {

        int ind = (dim1 * 2) - 2 * level - (2 * pas);

        for (int j = 0; j < 2 * pas; j++)
        {
            int target = (j + ind) * dim1 * 2 + gl_id;
            d_matrix_A1[gl_id * 2 * pas + j] = gpu_r_adf_in[target];
        }
    }

    /* Matrix B combinations*/

    __shared__ float node_vec[1000];
    if (2 * 2 * dim1 > 1000)
        printf("\n erreur size of reserved shared memory is not enough \n");
    __syncthreads();

    for (int j = 0; j < nb_node_parallel; j++)
    {
        if (gl_id < 2 * dim1)
        {
            float x = vec[j * (2 * dim1) + gl_id];
            // node_vec[threadIdx.x]=x;

            for (int i = 0; i < nb_copies; i++)
            {
                int ind = j * (nb_copies * (2 * dim1)) + i * dim1 * 2 + gl_id;
                d_mat_B[ind] = x;
            }
        }
        __syncthreads();
    }

    __syncthreads();
    if (gl_id < 2 * 2 * dim1)
    {

        for (int j = 0; j < nb_node_parallel; j++)
        {
            for (int i = 0; i < nb_copies / blockDim.x + 1; i++)
            {
                int position = (gl_id + (i * blockDim.x));
                if (position < nb_copies)
                { // printf("\n position:%d\n", position);
                    for (int i1 = 0; i1 < pas; i1++)
                    {
                        int p1 = 0, p2 = 0;
                        float f = 1;
                        // if(position%2==0)
                        {
                            p1 = j * nb_copies * (2 * dim1) + position * 2 * dim1 + 2 * dim1 - 2 - (2 * level) - 2 * i1;
                            p2 = j * nb_copies * (2 * dim1) + position * 2 * dim1 + 2 * dim1 - 2 - (2 * level) - 2 * i1 + 1;
                            f = 1;
                        }
                        d_mat_B[p1] = cuCrealf(gpu_comb[(int)(position)*pas + i1]);
                        d_mat_B[p2] = f * cuCimagf(gpu_comb[(int)(position)*pas + i1]);
                    }
                }
            }
        }
    }

    // Matrix C
    __syncthreads();

    int ind = (dim2 * 2) - 2 * level - (2 * pas);
    if (gl_id < 2 * pas)
    {
        node_vec[gl_id] = y_cpu_modulation[ind + gl_id];
        node_vec[gl_id + 2 * pas] = node_vec[gl_id];
        node_vec[gl_id + 4 * pas] = node_vec[gl_id];
        node_vec[gl_id + 6 * pas] = node_vec[gl_id];
    }

    __syncthreads();

    if (gl_id < 8 * pas)
    {
        for (int f = 0; f < nb_copies * nb_node_parallel; f = f + 4)
        {

            __syncthreads();
            d_matrix_C[f * 4 * 2 * pas + gl_id] = node_vec[gl_id];
        }
    }
}

__device__ void max(float &a, float &a1, int &ia, int &ia1, float &b, float &b1, int &ib, int &ib1, int swap)
{
    float c = a, c1 = a1;
    int ic = ia, ic1 = ia1;
    float d = b, d1 = b1;
    int id = ib, id1 = ib1;

    // if(a==0)a=1000000; if(a1==0)a1=1000000; if(b==0)b=1000000; if(b1==0)b1=1000000;
    if (c > b)
    {
        c1 = a;
        ic1 = ia;
        c = b;
        ic = ib;
        if (c1 > b1)
        {
            c1 = b1;
            ic1 = ib1;
        }
    }
    else if (c1 > b)
    {
        c1 = b;
        ic1 = ib;
    }

    if (swap == 1)
    {

        if (ic1 == ib1)
        {
            d = a;
            id = ia;
            d1 = a1;
            id1 = ia1;
        }
        else if (ic1 == ia1)
        {
            d = b;
            id = ib;
            d1 = b1;
            id1 = ib1;
        }
        else if (a1 < b1)
        {
            d = a1;
            id = ia1;
            d1 = b1;
            id1 = ib1;
        }
        else if (a1 > b1)
        {
            d = b1;
            id = ib1;
            d1 = a1;
            id1 = ia1;
        }
    }
    a = c;
    a1 = c1;
    ia = ic;
    ia1 = ic1;
    b = d;
    b1 = d1;
    ib = id;
    ib1 = id1;

    /*if(a-b>0.000001){
        a1=a;ia1=ia; a=b;ia=ib;
        if(a1-b1>0.000001){a1=b1;ia1=ib1;}
     }else if(a1-b>0.000001){a1=b;ia1=ib;}
    */
}

// template <unsigned int blockSize>
__device__ void warpReduce(volatile float *local_norme, volatile int *local_id, unsigned int tid, int swap, unsigned int blockSize)
{
    if (blockSize >= 64)
    {
        float v1 = local_norme[tid];
        float v2 = local_norme[tid + 32];
        int iv1 = local_id[tid];
        int iv2 = local_id[tid + 32];
        float v11 = local_norme[tid + blockDim.x];
        float v22 = local_norme[tid + 32 + blockDim.x];
        int iv11 = local_id[tid + blockDim.x];
        int iv22 = local_id[tid + 32 + blockDim.x];

        max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, 0);

        local_norme[tid] = v1;
        local_id[tid] = iv1;
        local_norme[tid + blockDim.x] = v11;
        local_id[tid + blockDim.x] = iv11;
        /*if(v1>v2){
               local_norme[tid]=v2;local_norme[tid+32]=v1;
             local_id[tid]=iv2; local_id[tid+32]=iv1;
        }*/
    }

    if (blockSize >= 32)
        if (tid < 16)
        {
            float v1 = local_norme[tid];
            float v2 = local_norme[tid + 16];
            int iv1 = local_id[tid];
            int iv2 = local_id[tid + 16];
            float v11 = local_norme[tid + blockDim.x];
            float v22 = local_norme[tid + 16 + blockDim.x];
            int iv11 = local_id[tid + blockDim.x];
            int iv22 = local_id[tid + 16 + blockDim.x];
            max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, 0);
            local_norme[tid] = v1;
            local_id[tid] = iv1;
            local_norme[tid + blockDim.x] = v11;
            local_id[tid + blockDim.x] = iv11;
            /* if(v1>v2){
                    local_norme[tid]=v2;local_norme[tid+16]=v1;
                  local_id[tid]=iv2; local_id[tid+16]=iv1;
             }*/
        }
    if (blockSize >= 16)
        if (tid < 8)
        {
            float v1 = local_norme[tid];
            float v2 = local_norme[tid + 8];
            int iv1 = local_id[tid];
            int iv2 = local_id[tid + 8];
            float v11 = local_norme[tid + blockDim.x];
            float v22 = local_norme[tid + 8 + blockDim.x];
            int iv11 = local_id[tid + blockDim.x];
            int iv22 = local_id[tid + 8 + blockDim.x];

            max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, 0);
            local_norme[tid] = v1;
            local_id[tid] = iv1;
            local_norme[tid + blockDim.x] = v11;
            local_id[tid + blockDim.x] = iv11;
            /*    if(v1>v2){
                     local_norme[tid]=v2;local_norme[tid+8]=v1;
                     local_id[tid]=iv2; local_id[tid+8]=iv1;
                }*/
        }
    if (blockSize >= 8)
        if (tid < 4)
        {
            float v1 = local_norme[tid];
            float v2 = local_norme[tid + 4];
            int iv1 = local_id[tid];
            int iv2 = local_id[tid + 4];
            float v11 = local_norme[tid + blockDim.x];
            float v22 = local_norme[tid + 4 + blockDim.x];
            int iv11 = local_id[tid + blockDim.x];
            int iv22 = local_id[tid + 4 + blockDim.x];

            max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, 0);
            local_norme[tid] = v1;
            local_id[tid] = iv1;
            local_norme[tid + blockDim.x] = v11;
            local_id[tid + blockDim.x] = iv11;
            /*    if(v1>v2){
                       local_norme[tid]=v2;local_norme[tid+4]=v1;
                     local_id[tid]=iv2; local_id[tid+4]=iv1;
                }*/
        }
    if (blockSize >= 4)
        if (tid < 2)
        {
            float v1 = local_norme[tid];
            float v2 = local_norme[tid + 2];
            int iv1 = local_id[tid];
            int iv2 = local_id[tid + 2];
            float v11 = local_norme[tid + blockDim.x];
            float v22 = local_norme[tid + 2 + blockDim.x];
            int iv11 = local_id[tid + blockDim.x];
            int iv22 = local_id[tid + 2 + blockDim.x];

            max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, 0);
            local_norme[tid] = v1;
            local_id[tid] = iv1;
            local_norme[tid + blockDim.x] = v11;
            local_id[tid + blockDim.x] = iv11;

            /*         local_norme[tid+2]=v2;
                     local_id[tid+2]=iv2;
                    local_norme[tid+2+blockDim.x]=v22;
                    local_id[tid+2+blockDim.x]=iv22;*/
            /*if(v1>v2){
                local_norme[tid+2]=v1;
                 local_id[tid+2]=iv1; }*/
        }
    if (blockSize >= 2)
        if (tid < 1)
        {
            float sv1 = local_norme[tid];
            int siv1 = local_id[tid];
            float v1 = local_norme[tid];
            float v2 = local_norme[tid + 1];
            int iv1 = local_id[tid];
            int iv2 = local_id[tid + 1];
            float v11 = local_norme[tid + blockDim.x];
            float v22 = local_norme[tid + 1 + blockDim.x];
            int iv11 = local_id[tid + blockDim.x];
            int iv22 = local_id[tid + 1 + blockDim.x];
            // if(swap==1){//printf("\n 1 %f %f %d %d", v1, v11,iv1, iv11);
            // printf("\n 2 %f %f %d %d", v2, v22,iv2, iv22);}
            max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, swap);
            // if(swap==1){
            // printf("\n %f %f %d %d", v2, v22,iv2, iv22);}
            local_norme[tid] = v1;
            local_id[tid] = iv1;
            local_norme[tid + blockDim.x] = v11;
            local_id[tid + blockDim.x] = iv11;

            local_norme[tid + 1] = v2;
            local_id[tid + 1] = iv2;
            local_norme[tid + 1 + blockDim.x] = v22;
            local_id[tid + 1 + blockDim.x] = iv22;

            // local_norme[tid+blockDim.x]=v11;float local_norme[tid+1+blockDim.x]=v22;
            //   local_id[tid+blockDim.x]=iv11; local_id[tid+1+blockDim.x]=iv22;

            /* if(tid==-1) if(local_norme[tid+1]-local_norme[tid]<0.00001){
                        //local_norme[tid]=v2;
                        local_norme[tid+1]=sv1;
                    //local_id[tid]=iv2;
                     local_id[tid+1]=siv1;
               } */
        }
}

__device__ float N_best[600000];
__device__ int Ni_best[600000];

__global__ void final_NORM(float *vec, int *best_nd, float *eval1, int sort_size,
                           const float *d_mat_B,
                           const float *new_mat_B, int level, int l111, int pas, int dim1, unsigned int blockSize)
{
    __shared__ float local_norme[2048];
    __shared__ int local_id[2048];
    unsigned int tid = threadIdx.x;
    float my_normes[2] = {
        100000,
        100000};
    int my_ids[2] = {
        tid,
        tid};
    int nb = blockSize / blockDim.x;

    for (int i = 0; i < nb; i++)
    {

        float vf = N_best[i * blockDim.x + tid];
        int i_vf = Ni_best[i * blockDim.x + tid];
        if (vf < my_normes[0])
        {
            //  my_normes[2]=my_normes[1]; my_ids[2]=my_ids[1];
            my_normes[1] = my_normes[0];
            my_ids[1] = my_ids[0];
            my_normes[0] = vf;
            my_ids[0] = i_vf;
        }
        else if (vf < my_normes[1])
        {
            //  my_normes[2]=my_normes[1]; my_ids[2]=my_ids[1];
            my_normes[1] = vf;
            my_ids[1] = i_vf;
        }
        __syncthreads();
    }
    if (tid < blockSize)
    {
        local_norme[tid] = my_normes[0];
        local_id[tid] = my_ids[0];
        local_norme[tid + blockDim.x] = my_normes[1];
        local_id[tid + blockDim.x] = my_ids[1];
    }
    else
    {
        local_norme[tid] = 1000;
        local_id[tid] = 1000;
        local_norme[tid + blockDim.x] = 1000;
        local_id[tid + blockDim.x] = 1000;
    }

    //  if(threadIdx.x==0)for(int j=0;j<blockDim.x+100;j++) printf("\n j:%d lid:%d ln:%f ",j,local_id[j], local_norme[j]);
    //     local_norme[threadIdx.x+2*blockDim.x]=my_normes[2];
    __syncthreads();
    // if(tid==0)for(int i=0;i<blockSize;i++){ printf("%f %d::", local_norme[i],local_id[i]); printf("\n");

    if (blockSize >= 1024)
    {
        if (tid < 512)
        {
            float v1 = local_norme[tid];
            float v2 = local_norme[tid + 512];
            int iv1 = local_id[tid];
            int iv2 = local_id[tid + 512];
            float v11 = local_norme[tid + blockDim.x];
            float v22 = local_norme[tid + 512 + blockDim.x];
            int iv11 = local_id[tid + blockDim.x];
            int iv22 = local_id[tid + 512 + blockDim.x];

            max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, 0);
            local_norme[tid] = v1;
            local_id[tid] = iv1;
            local_norme[tid + blockDim.x] = v11;
            local_id[tid + blockDim.x] = iv11;
        }
        __syncthreads();
    }

    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            float v1 = local_norme[tid];
            float v2 = local_norme[tid + 256];
            int iv1 = local_id[tid];
            int iv2 = local_id[tid + 256];
            float v11 = local_norme[tid + blockDim.x];
            float v22 = local_norme[tid + 256 + blockDim.x];
            int iv11 = local_id[tid + blockDim.x];
            int iv22 = local_id[tid + 256 + blockDim.x];

            max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, 0);
            local_norme[tid] = v1;
            local_id[tid] = iv1;
            local_norme[tid + blockDim.x] = v11;
            local_id[tid + blockDim.x] = iv11;
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            float v1 = local_norme[tid];
            float v2 = local_norme[tid + 128];
            int iv1 = local_id[tid];
            int iv2 = local_id[tid + 128];
            float v11 = local_norme[tid + blockDim.x];
            float v22 = local_norme[tid + 128 + blockDim.x];
            int iv11 = local_id[tid + blockDim.x];
            int iv22 = local_id[tid + 128 + blockDim.x];

            max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, 0);
            local_norme[tid] = v1;
            local_id[tid] = iv1;
            local_norme[tid + blockDim.x] = v11;
            local_id[tid + blockDim.x] = iv11;

            __syncthreads();
        }
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
        {
            float v1 = local_norme[tid];
            float v2 = local_norme[tid + 64];
            int iv1 = local_id[tid];
            int iv2 = local_id[tid + 64];
            float v11 = local_norme[tid + blockDim.x];
            float v22 = local_norme[tid + 64 + blockDim.x];
            int iv11 = local_id[tid + blockDim.x];
            int iv22 = local_id[tid + 64 + blockDim.x];

            max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, 0);
            local_norme[tid] = v1;
            local_id[tid] = iv1;
            local_norme[tid + blockDim.x] = v11;
            local_id[tid + blockDim.x] = iv11;
        }
        __syncthreads();
    }

    if (tid < 32)
        warpReduce(local_norme, local_id, tid, 1, blockSize);
    // if (tid == 0)printf("\n fin sorting: "); //g_odata[blockIdx.x] = local_norme[0];

    // __syncthreads();   if(threadIdx.x==0)
    //  for(int i=0;i<sort_size;i++){
    //  if(threadIdx.x==0)printf("\n3: %f l_id: %d", local_norme[i],local_id[i]);
    // } if(tid==0)printf("\n");
    __syncthreads();
    if (tid == 0)
    {
        int nb = sort_size / 2 + sort_size % 2;
        for (int i = 0; i < nb; i = i + 1)
        {
            int id = nb - 1 - i;
            local_norme[id * 2] = local_norme[id];
            local_norme[id * 2 + 1] = local_norme[id + blockDim.x];
            local_id[id * 2] = local_id[id];
            local_id[id * 2 + 1] = local_id[id + blockDim.x];
            // printf("\n3:id:%d %f l_id: %d",id, local_norme[id],local_id[id]);
            // printf("\n31: %f l_id: %d", local_norme[id+blockDim.x],local_id[id+blockDim.x]);
        }
    }
    __syncthreads();
    if (tid < sort_size)
    {
        eval1[tid] = local_norme[tid];
        best_nd[tid] = local_id[tid];
        //  printf("\n4: %d %f l_id: %d", tid, local_norme[tid], local_id[tid]);
    }

    //   int todo = (dim1 * 2 * sort_size) / blockDim.x + 1;
    //     if(gl_id==0){
    //      for(int i=0;i<10;i++) if(best_nd[i]>=0) printf("\n sorting gpu local norme:%f id:%d, ev:%f, todo:%d",local_norme[i],best_nd[i],d_norme[best_nd[i]], todo);
    //     }
    //
    // todo=0;
    // if(threadIdx.x<sort_size){if(best_nd[threadIdx.x]<sort_size*l111){ eval1[threadIdx.x]= local_norme[threadIdx.x]; //d_norme[best_nd[gl_id]];
    //}}
    // else{eval1[gl_id]=100000;} }

    __syncthreads();

    for (int f = 0; f < sort_size; f++)
    {
        if (tid < 2 * dim1)
        {
            int nd = best_nd[f] / l111;
            // printf("\n i:%d nd:%d . ln:%d vec :%d",blockDim.x, nd, 2*dim1*tid+i,nd*2*dim1+i);

            if (level == 0)
            {
                local_norme[2 * dim1 * f + tid] = 0; // vec[2*dim1*f+tid]=0;
            }
            else
            {
                local_norme[2 * dim1 * f + tid] = vec[nd * 2 * dim1 + tid];
            }
        }
        __syncthreads();
    }

    __syncthreads();
    for (int f = 0; f < sort_size; f++)
    {
        if (tid < 2 * dim1)
        {
            //  for (int i = 0; i < 2 * dim1; i++)
            {

                vec[2 * dim1 * f + tid] = local_norme[2 * dim1 * f + tid];
            }
        }
    }
    __syncthreads();
    if (tid < sort_size)
    {
        for (int i = 0; i < 2 * pas; i++)
        {
            vec[tid * 2 * dim1 + 2 * dim1 - 2 * level - 2 * pas + i] = new_mat_B[best_nd[tid] * 2 * pas + i];
            local_norme[tid * 2 * dim1 + 2 * dim1 - 2 * level - 2 * pas + i] = new_mat_B[best_nd[tid] * 2 * pas + i];
        }
    }

    // __syncthreads();*/
    //   if(gl_id==0){ for(int i=0;i<2*dim1*sort_size;i++){ if(i%(2*dim1)==0)printf("\n best :%d\n",i/(2*dim1));if(vec[i]!=local_norme[i]) printf("\nblock: %d i %d vec:%f ln:%f",blockDim.x, i, vec[i], local_norme[i]);}
    // printf ("\n =========================================final sort fin===============\n");}
}

__global__ void NORM(float *d_mat_C, int l111, int l1, int pas, float *d_norme, float eval, int *best_nd, float *eval1, int sort_size,
                     const float *d_mat_B, float *vec, int dim1, unsigned int blockSize, float *new_mat_C,
                     const float *new_mat_B, int level, __half *C_half)
{
    //  int best = -1;
    __shared__ float local_norme[128];
    __shared__ int local_id[128];
    // int p = 0;
    int gl_id = threadIdx.x + blockDim.x * blockIdx.x;
    //  if(gl_id==0)printf("\n G GPU norme in");
    int nb = 1;
    float my_normes[2] = {
        100000,
        100000};
    int my_ids[2] = {
        gl_id,
        gl_id};
    int nb_th = blockDim.x * gridDim.x;
    unsigned int tid = threadIdx.x;
    nb = l111 * l1 / nb_th;
    if (nb == 0)
        nb = 1;
    for (int i = 0; i < nb; i++)
    {
        int ind = i * nb_th + gl_id;
        float summ = 100000;
        if (ind < l111 * l1)
            summ = eval1[ind / l111];
        if (ind < l111 * l1)
            for (int h = 0; h < 2 * (pas); h++)
            {

                int p_mul = (ind / l111) * 2 * pas + h;
#ifndef FP16MM
                float x = new_mat_C[ind * 2 * pas + h] - partial_mul[p_mul];
#else
                __half x = __hsub(C_half[ind * 2 * pas + h], H_partial_mul[p_mul]); //__half2float(C_half[ind * 2 * pas + h]) - __half2float(H_partial_mul[p_mul]);
                                                                                    //    if(gl_id==0) printf("v:%f \n",__half2float(C_half[ind*2*pas+h]));
#endif
                //  if(ind==100280)printf("\nnew kernel ind 100280 h:%d ids:%d posiyion:%d , max:%d", h,ind_Block, ind_Block+threadIdx.x, l111*l1*2*pas);

#ifndef FP16MM
                summ = summ + x * x;
#else

                summ = summ + __half2float(__hmul(x, x));
#endif
            }

        if (summ < my_normes[0])
        {
            //  my_normes[2]=my_normes[1]; my_ids[2]=my_ids[1];
            my_normes[1] = my_normes[0];
            my_ids[1] = my_ids[0];
            my_normes[0] = summ;
            my_ids[0] = ind;
        }
        else if (summ < my_normes[1])
        {
            //  my_normes[2]=my_normes[1]; my_ids[2]=my_ids[1];
            my_normes[1] = summ;
            my_ids[1] = ind;
        }
    }
    //  __syncthreads();
    if (tid < l111 * l1)
    {
        local_norme[threadIdx.x] = my_normes[0];
        local_norme[threadIdx.x + blockDim.x] = my_normes[1];
        //     local_norme[threadIdx.x+2*blockDim.x]=my_normes[2];

        local_id[threadIdx.x] = my_ids[0];
        local_id[threadIdx.x + blockDim.x] = my_ids[1];
    }

    __syncthreads();
    if (blockSize >= 128)
    {
        if (tid < 64)
        {
            float v1 = local_norme[tid];
            float v2 = local_norme[tid + 64];
            int iv1 = local_id[tid];
            int iv2 = local_id[tid + 64];
            float v11 = local_norme[tid + blockDim.x];
            float v22 = local_norme[tid + 64 + blockDim.x];
            int iv11 = local_id[tid + blockDim.x];
            int iv22 = local_id[tid + 64 + blockDim.x];

            max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, 0);
            local_norme[tid] = v1;
            local_id[tid] = iv1;
            local_norme[tid + blockDim.x] = v11;
            local_id[tid + blockDim.x] = iv11;
        }
        __syncthreads();
    }

    if (tid < 32 && tid < l111 * l1)
        warpReduce(local_norme, local_id, tid, 1, blockSize);
    __syncthreads();
    if (tid == 0)
        if (gridDim.x == 1)
        {
            int nb = sort_size / 2 + sort_size % 2;
            for (int i = 0; i < nb; i = i + 1)
            {
                int id = nb - 1 - i;
                local_norme[id * 2] = local_norme[id];
                local_norme[id * 2 + 1] = local_norme[id + blockDim.x];
                local_id[id * 2] = local_id[id];
                local_id[id * 2 + 1] = local_id[id + blockDim.x];
            }
        }

    __syncthreads();

    if (tid == 0 && gridDim.x > 1)
    {
        N_best[blockIdx.x * 2 + tid] = local_norme[tid];
        Ni_best[blockIdx.x * 2 + tid] = local_id[tid];
        N_best[blockIdx.x * 2 + tid + 1] = local_norme[tid + blockDim.x];
        Ni_best[blockIdx.x * 2 + tid + 1] = local_id[tid + blockDim.x];
    }
}

__global__ void G_GPU_norme_4(float *d_mat_C, int l111, int l1, int pas, float *d_norme, float eval, int *best_nd, float *eval1, int sort_size,
                              const float *d_mat_B, float *vec, int dim1, unsigned int blockSize, float *new_mat_C,
                              const float *new_mat_B, int level, __half *C_half)
{
    //  int best = -1;
    extern __shared__ float local_norme[];
    __shared__ int local_id[2048];
    // int p = 0;
    int gl_id = threadIdx.x + blockDim.x * blockIdx.x;
    //  if(gl_id==0)printf("\n G GPU norme in");
    int nb = 1;
    float my_normes[2] = {
        100000,
        100000};
    int my_ids[2] = {
        gl_id,
        gl_id};
    int nb_th = blockDim.x * gridDim.x;
    unsigned int tid = threadIdx.x;
    // if(l111*l1-blockDim.x>1)

    // if(gl_id==0)for(int i=0;i<200;i++) printf("\n half C %f:",__half2float(C_half[i]));

    nb = l111 * l1 / nb_th;
    if (nb == 0)
        nb = 1;
    for (int i = 0; i < nb; i++)
    {

        int ind = i * nb_th + gl_id;
        float summ = 100000;
        if (ind < l111 * l1)
            summ = eval1[ind / l111];
        if (ind < l111 * l1)
            for (int h = 0; h < 2 * (pas); h++)
            {
                float x = 0;
                int p_mul = (ind / l111) * 2 * pas + h;
#ifndef FP16MM
                x = new_mat_C[ind * 2 * pas + h] - partial_mul[p_mul];
#else
                x = __half2float(C_half[ind * 2 * pas + h]) - __half2float(H_partial_mul[p_mul]);

#endif

                summ = summ + x * x;
            }

        if (summ < my_normes[0])
        {
            //  my_normes[2]=my_normes[1]; my_ids[2]=my_ids[1];
            my_normes[1] = my_normes[0];
            my_ids[1] = my_ids[0];
            my_normes[0] = summ;
            my_ids[0] = ind;
        }
        else if (summ < my_normes[1])
        {
            //  my_normes[2]=my_normes[1]; my_ids[2]=my_ids[1];
            my_normes[1] = summ;
            my_ids[1] = ind;
        }
    }

    __syncthreads();
    if (tid < l111 * l1)
    {
        local_norme[threadIdx.x] = my_normes[0];
        local_norme[threadIdx.x + blockDim.x] = my_normes[1];
        //     local_norme[threadIdx.x+2*blockDim.x]=my_normes[2];

        local_id[threadIdx.x] = my_ids[0];
        local_id[threadIdx.x + blockDim.x] = my_ids[1];
    }

    __syncthreads();
    if (blockSize >= 1024)
    {
        if (tid < 512)
        {
            float v1 = local_norme[tid];
            float v2 = local_norme[tid + 512];
            int iv1 = local_id[tid];
            int iv2 = local_id[tid + 512];
            float v11 = local_norme[tid + blockDim.x];
            float v22 = local_norme[tid + 512 + blockDim.x];
            int iv11 = local_id[tid + blockDim.x];
            int iv22 = local_id[tid + 512 + blockDim.x];

            max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, 0);
            local_norme[tid] = v1;
            local_id[tid] = iv1;
            local_norme[tid + blockDim.x] = v11;
            local_id[tid + blockDim.x] = iv11;
        }
        __syncthreads();
    }

    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            float v1 = local_norme[tid];
            float v2 = local_norme[tid + 256];
            int iv1 = local_id[tid];
            int iv2 = local_id[tid + 256];
            float v11 = local_norme[tid + blockDim.x];
            float v22 = local_norme[tid + 256 + blockDim.x];
            int iv11 = local_id[tid + blockDim.x];
            int iv22 = local_id[tid + 256 + blockDim.x];

            max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, 0);
            local_norme[tid] = v1;
            local_id[tid] = iv1;
            local_norme[tid + blockDim.x] = v11;
            local_id[tid + blockDim.x] = iv11;
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            float v1 = local_norme[tid];
            float v2 = local_norme[tid + 128];
            int iv1 = local_id[tid];
            int iv2 = local_id[tid + 128];
            float v11 = local_norme[tid + blockDim.x];
            float v22 = local_norme[tid + 128 + blockDim.x];
            int iv11 = local_id[tid + blockDim.x];
            int iv22 = local_id[tid + 128 + blockDim.x];

            max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, 0);
            local_norme[tid] = v1;
            local_id[tid] = iv1;
            local_norme[tid + blockDim.x] = v11;
            local_id[tid + blockDim.x] = iv11;

            __syncthreads();
        }
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
        {
            float v1 = local_norme[tid];
            float v2 = local_norme[tid + 64];
            int iv1 = local_id[tid];
            int iv2 = local_id[tid + 64];
            float v11 = local_norme[tid + blockDim.x];
            float v22 = local_norme[tid + 64 + blockDim.x];
            int iv11 = local_id[tid + blockDim.x];
            int iv22 = local_id[tid + 64 + blockDim.x];

            max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, 0);
            local_norme[tid] = v1;
            local_id[tid] = iv1;
            local_norme[tid + blockDim.x] = v11;
            local_id[tid + blockDim.x] = iv11;
        }
        __syncthreads();
    }

    if (tid < 32 && tid < l111 * l1)
        warpReduce(local_norme, local_id, tid, 1, blockSize);

    __syncthreads();
    if (tid == 0)
        if (gridDim.x == 1)
        {
            int nb = sort_size / 2 + sort_size % 2;
            for (int i = 0; i < nb; i = i + 1)
            {
                int id = nb - 1 - i;

                local_norme[id * 2] = local_norme[id];
                local_norme[id * 2 + 1] = local_norme[id + blockDim.x];
                local_id[id * 2] = local_id[id];
                local_id[id * 2 + 1] = local_id[id + blockDim.x];
            }
        }

    __syncthreads();

    if (tid == 0 && gridDim.x > 1)
    {
        eval1[sort_size + blockIdx.x * 4 + tid] = local_norme[tid];
        best_nd[sort_size + blockIdx.x * 4 + tid] = local_id[tid];
        eval1[sort_size + blockIdx.x * 4 + tid + 1] = local_norme[tid + blockDim.x];
        best_nd[sort_size + blockIdx.x * 4 + tid + 1] = local_id[tid + blockDim.x];

        eval1[sort_size + blockIdx.x * 4 + tid + 2] = local_norme[tid + 1];
        best_nd[sort_size + blockIdx.x * 4 + tid + 2] = local_id[tid + 1];
        eval1[sort_size + blockIdx.x * 4 + tid + 3] = local_norme[tid + 1 + blockDim.x];
        best_nd[sort_size + blockIdx.x * 4 + tid + 3] = local_id[tid + 1 + blockDim.x];
        // printf("\nblockid:%d position :%d norme %f %f l_id: %d",blockIdx.x, sort_size+ blockIdx.x*sort_size ,eval1[sort_size+ blockIdx.x*sort_size+tid],local_norme[tid],local_id[tid]);
    }

    if (gridDim.x == 1)
    {
        if (tid < sort_size)
        {
            eval1[blockIdx.x * sort_size + tid] = local_norme[tid];
            best_nd[blockIdx.x * sort_size + tid] = local_id[tid];
        }

        __syncthreads();

        for (int f = 0; f < sort_size; f++)
        {
            if (tid < 2 * dim1)
            {
                int nd = best_nd[f] / l111;
                // printf("\n i:%d nd:%d . ln:%d vec :%d",blockDim.x, nd, 2*dim1*tid+i,nd*2*dim1+i);

                if (level == 0)
                {
                    local_norme[2 * dim1 * f + tid] = 0; // vec[2*dim1*f+tid]=0;
                }
                else
                {
                    local_norme[2 * dim1 * f + tid] = vec[nd * 2 * dim1 + tid];
                }
            }
            __syncthreads();
        }

        __syncthreads();
        for (int f = 0; f < sort_size; f++)
        {
            if (tid < 2 * dim1)
            {
                for (int i = 0; i < 2 * dim1; i++)
                {

                    vec[2 * dim1 * f + tid] = local_norme[2 * dim1 * f + tid];
                }
            }
        }
        __syncthreads();
        if (tid < sort_size)
        {
            for (int i = 0; i < 2 * pas; i++)
            {
                vec[tid * 2 * dim1 + 2 * dim1 - 2 * level - 2 * pas + i] = new_mat_B[best_nd[tid] * 2 * pas + i];
                local_norme[tid * 2 * dim1 + 2 * dim1 - 2 * level - 2 * pas + i] = new_mat_B[best_nd[tid] * 2 * pas + i];
            }
        }
    }
}

__global__ void final_sort(float *vec, int *best_nd, float *eval1, int sort_size,
                           const float *d_mat_B,
                           const float *new_mat_B, int level, int l111, int pas, int dim1, unsigned int blockSize)
{
    __shared__ float local_norme[2024];
    __shared__ int local_id[2024];

    unsigned int tid = threadIdx.x;

    if (tid < blockSize)
    {
        local_norme[tid] = eval1[sort_size + tid];
        local_id[tid] = best_nd[sort_size + tid];
        local_norme[blockDim.x + tid] = 100000;
    }
    else
    {
        local_norme[tid] = 10000;
        local_id[tid] = 10000;
        local_norme[blockDim.x + tid] = 100000;
    }

    __syncthreads();

    if (blockSize >= 1024)
    {
        if (tid < 512)
        {
            float v1 = local_norme[tid];
            float v2 = local_norme[tid + 512];
            int iv1 = local_id[tid];
            int iv2 = local_id[tid + 512];
            float v11 = local_norme[tid + blockDim.x];
            float v22 = local_norme[tid + 512 + blockDim.x];
            int iv11 = local_id[tid + blockDim.x];
            int iv22 = local_id[tid + 512 + blockDim.x];

            max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, 0);
            local_norme[tid] = v1;
            local_id[tid] = iv1;
            local_norme[tid + blockDim.x] = v11;
            local_id[tid + blockDim.x] = iv11;
        }
        __syncthreads();
    }

    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            float v1 = local_norme[tid];
            float v2 = local_norme[tid + 256];
            int iv1 = local_id[tid];
            int iv2 = local_id[tid + 256];
            float v11 = local_norme[tid + blockDim.x];
            float v22 = local_norme[tid + 256 + blockDim.x];
            int iv11 = local_id[tid + blockDim.x];
            int iv22 = local_id[tid + 256 + blockDim.x];

            max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, 0);
            local_norme[tid] = v1;
            local_id[tid] = iv1;
            local_norme[tid + blockDim.x] = v11;
            local_id[tid + blockDim.x] = iv11;
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            float v1 = local_norme[tid];
            float v2 = local_norme[tid + 128];
            int iv1 = local_id[tid];
            int iv2 = local_id[tid + 128];
            float v11 = local_norme[tid + blockDim.x];
            float v22 = local_norme[tid + 128 + blockDim.x];
            int iv11 = local_id[tid + blockDim.x];
            int iv22 = local_id[tid + 128 + blockDim.x];

            max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, 0);
            local_norme[tid] = v1;
            local_id[tid] = iv1;
            local_norme[tid + blockDim.x] = v11;
            local_id[tid + blockDim.x] = iv11;

            __syncthreads();
        }
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
        {
            float v1 = local_norme[tid];
            float v2 = local_norme[tid + 64];
            int iv1 = local_id[tid];
            int iv2 = local_id[tid + 64];
            float v11 = local_norme[tid + blockDim.x];
            float v22 = local_norme[tid + 64 + blockDim.x];
            int iv11 = local_id[tid + blockDim.x];
            int iv22 = local_id[tid + 64 + blockDim.x];

            max(v1, v11, iv1, iv11, v2, v22, iv2, iv22, 0);
            local_norme[tid] = v1;
            local_id[tid] = iv1;
            local_norme[tid + blockDim.x] = v11;
            local_id[tid + blockDim.x] = iv11;
        }
        __syncthreads();
    }

    if (tid < 32)
        warpReduce(local_norme, local_id, tid, 1, blockSize);

    __syncthreads();
    if (tid == 0)
    {
        int nb = sort_size / 2 + sort_size % 2;
        for (int i = 0; i < nb; i = i + 1)
        {
            int id = nb - 1 - i;
            local_norme[id * 2] = local_norme[id];
            local_norme[id * 2 + 1] = local_norme[id + blockDim.x];
            local_id[id * 2] = local_id[id];
            local_id[id * 2 + 1] = local_id[id + blockDim.x];
        }
        __syncthreads();
        if (tid < sort_size)
        {
            eval1[tid] = local_norme[tid];
            best_nd[tid] = local_id[tid];
        }

        __syncthreads();

        for (int f = 0; f < sort_size; f++)
        {
            if (tid < 2 * dim1)
            {
                int nd = best_nd[f] / l111;
                // printf("\n i:%d nd:%d . ln:%d vec :%d",blockDim.x, nd, 2*dim1*tid+i,nd*2*dim1+i);

                if (level == 0)
                {
                    local_norme[2 * dim1 * f + tid] = 0; // vec[2*dim1*f+tid]=0;
                }
                else
                {
                    local_norme[2 * dim1 * f + tid] = vec[nd * 2 * dim1 + tid];
                }
            }
            __syncthreads();
        }

        __syncthreads();
        for (int f = 0; f < sort_size; f++)
        {
            if (tid < 2 * dim1)
            {
                for (int i = 0; i < 2 * dim1; i++)
                {

                    vec[2 * dim1 * f + tid] = local_norme[2 * dim1 * f + tid];
                }
            }
        }
        __syncthreads();
        if (tid < sort_size)
        {
            for (int i = 0; i < 2 * pas; i++)
            {
                vec[tid * 2 * dim1 + 2 * dim1 - 2 * level - 2 * pas + i] = new_mat_B[best_nd[tid] * 2 * pas + i];
                local_norme[tid * 2 * dim1 + 2 * dim1 - 2 * level - 2 * pas + i] = new_mat_B[best_nd[tid] * 2 * pas + i];
            }
        }
    }
}

/* GPU Code end! */

vector<tree_node> list_temp;
vector<tree_node> branch_nodes(vector<tree_node> vec_node, int8_t *A2, float d, int pas, int l111)
{
    list_temp.resize(l111 * vec_node.size());
    // cout<<"\n vec size"<< vec_node.size();
    int ind = 0;

    for (int i = 0; i < vec_node.size(); i++)
    {
        tree_node node = vec_node[i];
        //  vec_node.pop_back();
        if (node.evaluation < d)
        {
            int level = (int)node.level; // cuCrealf(etat);
            for (int jf = 0; jf < l111; jf++)
            {
                tree_node r = node;
                r.level = level + pas;
                for (int jj1 = 0; jj1 < pas; jj1++)
                {
                    r.s[(nTx - 1) - level - jj1] = A2[jf * pas + jj1];
                    if (pas == 1)
                        r.s[(nTx - 1) - level - jj1] = jf;
                }
                // list_temp.push_back(r);
                list_temp[ind] = r;
                ind++;
            }
        }
    }
    vec_node.clear();
    return list_temp;
}
int nerb(float is, float ir, int nb)
{
    int e = 0;
    int v1[nb], v2[nb];
    if (nb == 1)
    {
        if (is == -1)
        {
            v1[0] = 0;
        }
        else
        {
            v1[0] = 1;
        }
        if (ir == -1)
        {
            v2[0] = 0;
        }
        else
        {
            v2[0] = 1;
        }
    }
    else if (nb == 2)
    {
        if (is == -3)
        {
            v1[1] = 0;
            v1[0] = 0;
        }
        else if (is == -1)
        {
            v1[1] = 0;
            v1[0] = 1;
        }
        else if (is == 1)
        {
            v1[1] = 1;
            v1[0] = 0;
        }
        else if (is == 3)
        {
            v1[1] = 1;
            v1[0] = 1;
        }
        else
        {
            printf("\n not a match");
        }

        if (ir == -3)
        {
            v2[1] = 0;
            v2[0] = 0;
        }
        else if (ir == -1)
        {
            v2[1] = 0;
            v2[0] = 1;
        }
        else if (ir == 1)
        {
            v2[1] = 1;
            v2[0] = 0;
        }
        else if (ir == 3)
        {
            v2[1] = 1;
            v2[0] = 1;
        }
        else
        {
            printf("\n not a match1");
        }
    }
    else if (nb == 3)
    {
        if (is == -7)
        {
            v1[2] = 0;
            v1[1] = 0;
            v1[0] = 0;
        }
        else if (is == -5)
        {
            v1[2] = 0;
            v1[1] = 0;
            v1[0] = 1;
        }
        else if (is == -3)
        {
            v1[2] = 0;
            v1[1] = 1;
            v1[0] = 0;
        }
        else if (is == -1)
        {
            v1[2] = 0;
            v1[1] = 1;
            v1[0] = 1;
        }
        else if (is == 1)
        {
            v1[2] = 1;
            v1[1] = 0;
            v1[0] = 0;
        }
        else if (is == 3)
        {
            v1[2] = 1;
            v1[1] = 0;
            v1[0] = 1;
        }
        else if (is == 5)
        {
            v1[2] = 1;
            v1[1] = 1;
            v1[0] = 0;
        }
        else if (is == 7)
        {
            v1[2] = 1;
            v1[1] = 1;
            v1[0] = 1;
        }

        if (ir == -7)
        {
            v2[2] = 0;
            v2[1] = 0;
            v2[0] = 0;
        }
        else if (ir == -5)
        {
            v2[2] = 0;
            v2[1] = 0;
            v2[0] = 1;
        }
        else if (ir == -3)
        {
            v2[2] = 0;
            v2[1] = 1;
            v2[0] = 0;
        }
        else if (ir == -1)
        {
            v2[2] = 0;
            v2[1] = 1;
            v2[0] = 1;
        }
        else if (ir == 1)
        {
            v2[2] = 1;
            v2[1] = 0;
            v2[0] = 0;
        }
        else if (ir == 3)
        {
            v2[2] = 1;
            v2[1] = 0;
            v2[0] = 1;
        }
        else if (ir == 5)
        {
            v2[2] = 1;
            v2[1] = 1;
            v2[0] = 0;
        }
        else if (ir == 7)
        {
            v2[2] = 1;
            v2[1] = 1;
            v2[0] = 1;
        }
    }

    for (int i = 0; i < nb; i++)
        if (v1[i] != v2[i])
            e++;
    return e;
}

int error(cuFloatComplex s, cuFloatComplex rs, int nbit)
{
    int err = 0, nb = nbit / 2;
    if (nb == 0)
        nb++;
    if (cuCrealf(s) != cuCrealf(rs))
    {
        err = err + nerb(cuCrealf(s), cuCrealf(rs), nb);
    }
    if (cuCimagf(s) != cuCimagf(rs))
    {
        err = err + nerb(cuCimagf(s), cuCimagf(rs), nb);
    }
    return err;
}

vector<int_fast8_t> Branching_combinations(int mod, int pas)
{
    if (mod == 2)
    {
        BPSK.push_back(make_cuFloatComplex(-1, 0));
        BPSK.push_back(make_cuFloatComplex(1, 0));
    }
    else if (mod == 4)
    {
        qam4.push_back(make_cuFloatComplex(-1, -1));
        qam4.push_back(make_cuFloatComplex(-1, 1));
        qam4.push_back(make_cuFloatComplex(1, -1));
        qam4.push_back(make_cuFloatComplex(1, 1));
    }
    else if (mod == 16)
    {
        qam16.push_back(make_cuFloatComplex(-3, -3));
        qam16.push_back(make_cuFloatComplex(-3, -1));
        qam16.push_back(make_cuFloatComplex(-3, 1));
        qam16.push_back(make_cuFloatComplex(-3, 3));
        qam16.push_back(make_cuFloatComplex(-1, -3));
        qam16.push_back(make_cuFloatComplex(-1, -1));
        qam16.push_back(make_cuFloatComplex(-1, 1));
        qam16.push_back(make_cuFloatComplex(-1, 3));

        qam16.push_back(make_cuFloatComplex(1, -3));
        qam16.push_back(make_cuFloatComplex(1, -1));
        qam16.push_back(make_cuFloatComplex(1, 1));
        qam16.push_back(make_cuFloatComplex(1, 3));

        qam16.push_back(make_cuFloatComplex(3, -3));
        qam16.push_back(make_cuFloatComplex(3, -1));
        qam16.push_back(make_cuFloatComplex(3, 1));
        qam16.push_back(make_cuFloatComplex(3, 3));
    }
    else if (mod == 64)
    {
        int a0 = -7;
        for (int i = 0; i < 8; i++)
        {
            float aa = a0 + (2 * i);
            for (int i1 = 0; i1 < 8; i1++)
                qam64.push_back(make_cuFloatComplex(aa, a0 + 2 * i1));
        }
    }
    vector<cuFloatComplex> symbols;
    if (mod == 2)
    {
        symbols = BPSK;
        Constellation_symbols = BPSK;
    }
    else if (mod == 4)
    {
        symbols = qam4;
        Constellation_symbols = qam4;
    }
    else if (mod == 16)
    {
        symbols = qam16;
        Constellation_symbols = qam16;
    }
    else if (mod == 64)
    {
        symbols = qam64;
        Constellation_symbols = qam64;
    }

    vector<int_fast8_t> B1;
    vector<int_fast8_t> constellation_codification;
    for (int i = 0; i < mod; i++)
    {
        int_fast8_t f = i;
        constellation_codification.push_back(f);
    }

    vector<vector<int_fast8_t>> A2, A1;
    vector<int_fast8_t> temp;

    for (int i = 0; i < pas; i++)
    {
        int_fast8_t f = 0;
        temp.push_back(f);
    }
    for (int i = 0; i < pas; i++)
    {
        A1 = A2;
        A2.clear();
        if (A1.size() == 0)
            A1.push_back(temp);
        //        printf("\n A1 size:%d,Modulation:%d ", A1.size(), mod);
        for (int j = 0; j < A1.size(); j++)
        {
            temp = A1[j];
            for (int jf = 0; jf < mod; jf++)
            { // jf
                vector<int_fast8_t> r = temp;
                r[i] = constellation_codification[jf];
                A2.push_back(r);
            }
        }
    }
    for (int j = 0; j < A2.size(); j++)
    {
        /*printf("\n v:%d",j);*/
        for (int i = 0; i < pas; i++)
        {
            B1.push_back(A2[j][i]);
        }
    }

    return B1;
}

vector<cuFloatComplex> int8_to_complex(vector<int_fast8_t> mat, vector<cuFloatComplex> Constellation_symbols)
{
    vector<cuFloatComplex> complex_mat;

    for (int i = 0; i < mat.size(); i++)
    {
        complex_mat.push_back(Constellation_symbols[(int)mat[i]]);
    }

    return complex_mat;
}

vector<cuFloatComplex> constellation(int mod, int pas)
{
    vector<cuFloatComplex> B1;
    if (mod == 2)
    {
        BPSK.push_back(make_cuFloatComplex(-1, 0));
        BPSK.push_back(make_cuFloatComplex(1, 0));
    }
    else if (mod == 4)
    {
        qam4.push_back(make_cuFloatComplex(-1, -1));
        qam4.push_back(make_cuFloatComplex(-1, 1));
        qam4.push_back(make_cuFloatComplex(1, -1));
        qam4.push_back(make_cuFloatComplex(1, 1));
    }
    else if (mod == 16)
    {
        qam16.push_back(make_cuFloatComplex(-3, -3));
        qam16.push_back(make_cuFloatComplex(-3, -1));
        qam16.push_back(make_cuFloatComplex(-3, 1));
        qam16.push_back(make_cuFloatComplex(-3, 3));
        qam16.push_back(make_cuFloatComplex(-1, -3));
        qam16.push_back(make_cuFloatComplex(-1, -1));
        qam16.push_back(make_cuFloatComplex(-1, 1));
        qam16.push_back(make_cuFloatComplex(-1, 3));
        qam16.push_back(make_cuFloatComplex(1, -3));
        qam16.push_back(make_cuFloatComplex(1, -1));
        qam16.push_back(make_cuFloatComplex(1, 1));
        qam16.push_back(make_cuFloatComplex(1, 3));
        qam16.push_back(make_cuFloatComplex(3, -3));
        qam16.push_back(make_cuFloatComplex(3, -1));
        qam16.push_back(make_cuFloatComplex(3, 1));
        qam16.push_back(make_cuFloatComplex(3, 3));
    }
    else if (mod == 64)
    {
        int a0 = -7;
        for (int i = 0; i < 8; i++)
        {
            float aa = a0 + (2 * i);
            for (int i1 = 0; i1 < 8; i1++)
                qam64.push_back(make_cuFloatComplex(aa, a0 + 2 * i1));
        }
    }
    vector<cuFloatComplex> symbols;
    if (mod == 2)
    {
        symbols = BPSK;
        Constellation_symbols = BPSK;
    }
    else if (mod == 4)
    {
        symbols = qam4;
        Constellation_symbols = qam4;
    }
    else if (mod == 16)
    {
        symbols = qam16;
        Constellation_symbols = qam16;
    }
    else if (mod == 64)
    {
        symbols = qam64;
        Constellation_symbols = qam64;
    }
    vector<vector<cuFloatComplex>> A2, A1;

    vector<cuFloatComplex> temp;
    for (int i = 0; i < pas; i++)
        temp.push_back(make_cuFloatComplex(0, 0));

    for (int i = 0; i < pas; i++)
    {
        A1 = A2;
        A2.clear();
        if (A1.size() == 0)
            A1.push_back(temp);
        //  printf("\n A1 size:%d,Modulation:%d ", A1.size(), mod);
        for (int j = 0; j < A1.size(); j++)
        {
            temp = A1[j];
            for (int jf = 0; jf < mod; jf++)
            { // jf
                vector<cuFloatComplex> r = temp;
                r[i] = symbols[jf];
                A2.push_back(r);
            }
        }
    }

    for (int j = 0; j < A2.size(); j++)
    {
        /*printf("\n v:%d",j);*/
        for (int i = 0; i < pas; i++)
        {
            /*printf(" %f+%fi ",cuCrealf(A2[j][i]),  cuCimagf(A2[j][i]));*/
            B1.push_back(A2[j][i]);
        }
    }
    for (int j = 0; j < A2.size(); j++)
    {
        printf("\n  adel constelations in complex:%d", j);
        for (int i = 0; i < pas; i++)
        {
            printf(" %f+%fi ", cuCrealf(B1[j * pas + i]), cuCimagf(B1[j * pas + i]));
        }
    }
    Branching_combinations(mod, pas);
    return B1;
}

float kernel1(cuFloatComplex *G, int k, int pas, int j, int l1, int l11) // calcul de la norme au carre
{
    float sum = 0;
    //  #pragma unroll
    //  #pragma omp parallel for
    for (int h = 0; h < (k + pas); h++) // sum = sum + pow(cuCrealf(G[h*(l1*l11)+j]),2) + pow(cuCimagf(G[h*(l1*l11)+j]),2);
    {
        sum = sum + pow(cuCrealf(G[h * (l1 * l11) + j]), 2) + pow(cuCimagf(G[h * (l1 * l11) + j]), 2);
    }
    return sum;
}

bool myfunction(const position &i,
                const position &j)
{
    return (i.score > j.score);
}
int IDX2C(int i, int j, int ld)
{

    return (((j) * (ld)) + (i));
}
vector<int> norme2(cuFloatComplex *G3, float *Poids, int k, int pas, float d, int l1, int l11, float eval1, int exploration)
{

    vector<int> next;
    vector<position> next_position;
    float x = 0.0;

    for (int j = 0; j < l11; j++)
    {
        x = kernel1(G3, k, pas, j, l1, l11);
        x = x + eval1; // version inc update

        Poids[j] = x;
        if (x < d)
        {
            if (exploration != 1)
            {
                next.push_back(j);
            }
            else
            {
                position p;
                p.indice = j;
                p.score = x; /* next.push_back(j);*/
                next_position.push_back(p);
            }
        }
    }
    // if (exploration == 1)
    sort(next_position.begin(), next_position.end(), myfunction);
    for (int i = 0; i < next_position.size(); i++)
        next.push_back(next_position[i].indice);
    return next;
}
vector<cuFloatComplex> complex_v_best;
vector<position> real_norme3(float *G, float *Poids, int k, int pas, float d, int l1, int l11, float *eval1, int exploration, float *d_matrix_C, float *d_norme, int *best_nd, int nb_node_par,
                             const float *d_mat_B, float *vec, int dim1, int level, float *new_mat_C,
                             const float *new_mat_B, int l111,
                             const float *new_mat_B_1pas, __half *C_half)
{
    // float best1 = d, best2 = d, best3 = d;q
    vector<int> next;
    vector<position> next_position;
    vector<position> next_position2;
    // float x = 0.0;
    float *best_v = (float *)malloc(2 * dim1 * sizeof(float));

    int l0 = l11 * nb_node_par;
    int size = 2;
    size = l0;
    if (size > 1024)
        size = 1024;

    int size_shared = size + sort_size;
    if (size_shared < dim1 * nb_node_par)
        size_shared = 2 * dim1 * nb_node_par;
    unsigned int nb_block = (l11 * nb_node_par) / size;
    if (nb_block == 0)
        nb_block = 1;
    if (nb_block > 160)
        nb_block = 128; // it should be studied not all numbers work
    gettimeofday(&mul1, NULL);

    int nb_t = size;
    if (size < 2 * dim1)
        nb_t = 2 * dim1;

    int nit = 1024;
    if (l11 < 10000)
        nit = 4;
    if (l11 < 64)
        nit = 1;
    if (pas != 1)
    {
        if (l11 > 1024)
        {
            NORM<<<(l11 * nb_node_par) / nit, 64>>>(d_matrix_C, l11, nb_node_par, pas, d_norme, 0, best_nd, eval1, sort_size, d_mat_B, vec, dim1, size, new_mat_C, new_mat_B, level, C_half);
        }
        else
        {
            G_GPU_norme_4<<<nb_block, nb_t, 2 * size_shared * sizeof(float)>>>(d_matrix_C, l11, nb_node_par, pas, d_norme, 0, best_nd, eval1, sort_size, d_mat_B, vec, dim1, size, new_mat_C, new_mat_B, level, C_half);
        }
    }
    else
    {
        G_GPU_norme_4<<<nb_block, nb_t, 2024 * sizeof(float)>>>(d_matrix_C, l11, nb_node_par, pas, d_norme, 0, best_nd, eval1, sort_size, d_mat_B, vec, dim1, size, new_mat_C, new_mat_B_1pas, level, C_half);
    }
    if (nb_block > 1)
    {
        int nbtt = 1024;
        if (((l11 * nb_node_par) / nit) < 1024)
            nbtt = (l11 * nb_node_par) / nit;

        if (l11 > 1024)
        {
            final_NORM<<<1, nbtt>>>(vec, best_nd, eval1, sort_size, d_mat_B, new_mat_B, level, l111, pas, dim1, 2 * (l11 * nb_node_par) / nit);
        }
    }

    gettimeofday(&mul2, NULL);

    if (level + pas >= dim1)
    {
        cudaMemcpy(best_v, vec, 2 * dim1 * sizeof(float), cudaMemcpyDeviceToHost);
        {
            complex_v_best.clear();
            for (int i = 0; i < 2 * dim1; i = i + 2)
                complex_v_best.push_back(make_cuFloatComplex(best_v[i], best_v[i + 1]));
        }
    }

    // for(int i=0;i<sort_size;i++)
    {

        { // printf("\n :%d", i);
            position p;
            p.indice = 0; // h_best_nd[i];
            p.score = 1;  // h_norme[h_best_nd[i]];
            next_position2.push_back(p);
        }
    }

    for (int i = 0; i < next_position2.size(); i++)
        next_position.push_back(next_position2[next_position2.size() - 1 - i]);

    free(best_v);

    return next_position;
}

//
int found = 0;

cublasHandle_t handle;
float real_node_evaluation(vector<tree_node> &list_node, vector<tree_node> &list_temp, vector<cuFloatComplex> &best_ad, int level, float *eval1, int l111, int dim1, int dim2, int pas, float d1, float *d_norme, int *best_nd, int nb_node_parallel, float *vec,
                           const float *new_mat_A,
                           const float *new_mat_B, float *new_mat_C, float *new_mat_B_1pas,
                           const __half *A_half,
                           const __half *B_half, __half *C_half,
                           const __half *B1_half)
{

    // printf ("\n evaluation nb par;%d l111:%d pas:%d\n", nb_node_parallel,l111,pas);

    float *result_vec;
    //(float *)malloc(nb_node_parallel * 2 * l111 * dim1 * 2 * sizeof(float));

    float dd = d1;
    float *Poids2; //= (float *)malloc(nb_node_parallel* l111 * sizeof(float));
    vector<int> next;
    vector<position> next22;

    gettimeofday(&mul11, NULL);

    const float alf = -1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    const __half alf_H = __float2half(-1);
    const __half bet_H = __float2half(0);
    const __half *H_alpha = &alf_H;
    const __half *H_beta = &bet_H;

    /*matrix matrix multiplication on GPU single and half precision*/

    if (pas != 1)
    {
#ifndef FP16MM
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2 * pas, l111 * nb_node_parallel, 2 * pas, alpha, new_mat_A, 2 * pas, new_mat_B, 2 * pas, beta, new_mat_C, 2 * pas);
//   cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2 * pas, l111*nb_node_parallel/batch_count ,2 * pas,alpha,aa, 2*pas,strideA,new_mat_B, 2*pas,strideB, beta, new_mat_C,2*pas,strideC, batch_count);
#else
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2 * pas, l111 * nb_node_parallel, 2 * pas, H_alpha, A_half, 2 * pas, B_half, 2 * pas, H_beta, C_half, 2 * pas);
        //    cublasHgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2 * pas, l111*nb_node_parallel/batch_count ,2 * pas,H_alpha,A_half, 2*pas,strideA,B_half, 2*pas,strideB, H_beta, C_half,2*pas,strideC,batch_count);
#endif
    }
    else
    {
#ifndef FP16MM
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2 * pas, l111 * nb_node_parallel, 2 * pas, alpha, new_mat_A, 2 * pas, new_mat_B_1pas, 2 * pas, beta, new_mat_C, 2 * pas);
#else
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2 * pas, l111 * nb_node_parallel, 2 * pas, H_alpha, A_half, 2 * pas, B1_half, 2 * pas, H_beta, C_half, 2 * pas);
#endif
    }
    gettimeofday(&mul22, NULL);

    {
        /*Norm and sorting phase*/
        next22 = real_norme3(result_vec, Poids2, 0, pas, d1, nb_node_parallel, l111, eval1, strategy_exploration, new_mat_C, d_norme, best_nd, nb_node_parallel, new_mat_B, vec, dim1, level, new_mat_C, new_mat_B, l111, new_mat_B_1pas, C_half);
    }

    if (level + pas >= dim1 /*&& eval1[0] > 0*/)
    {
        {
            dd = 1; // eval1[0];
            found = 100;
            best_ad = complex_v_best;
        }
    }

    int nb = next22.size();

    { // explored_nodes++;

        {
            int ind = next22[0].indice;
        }

        {

            list_temp[0].evaluation = 0; // Poids2[ind];
            list_node.push_back(list_temp[0]);
        }
    }

    //   free(Poids2);

    return dd;
}

vector<float> r_mat;

vector<double> v;
vector<double> v1;
vector<float> S_to_real(vector<tree_node> mat)
{
    r_mat.clear();
    int ind = 0;
    // v.reserve(2*mat[0].s.size());
    r_mat.resize(2 * mat[0].s.size() * 2 * mat.size());
    // v1.reserve(2*mat[0].s.size());
    for (int i = 0; i < mat.size(); i++)
    {
        // v.clear();v1.clear();

        for (int j = 0; j < mat[i].s.size(); j++)
        {
            cuFloatComplex c;
            if (mat[i].s[j] >= 0)
            {
                c = Constellation_symbols[mat[i].s[j]];
            }
            else
            {
                c = make_cuFloatComplex(0, 0);
            }
            float re_c = cuCrealf(c);
            float im_c = cuCimagf(c);
            r_mat[ind] = re_c;
            ind++;
            r_mat[ind] = im_c;
            ind++;
        }
        for (int j = 0; j < mat[i].s.size(); j++)
        {
            cuFloatComplex c;
            if (mat[i].s[j] >= 0)
            {
                c = Constellation_symbols[mat[i].s[j]];
            }
            else
            {
                c = make_cuFloatComplex(0, 0);
            }
            float re_c = cuCrealf(c);
            float im_c = cuCimagf(c);

            r_mat[ind] = -im_c;
            ind++;
            r_mat[ind] = re_c;
            ind++;
            //   r_mat.push_back(re_c);
        }
    }

    return r_mat;
}

// Prints the contents of a matrix to standard output
template <class M>
void printMatrix2(const M &matrix, int rows, int columns)
{
    int height = rows;   // matrix.shape()[0];
    int width = columns; // matrix.shape()[1];
    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            std::cout << matrix[row * width + col] << " ";
        }
        std::cout << "\n";
    }
}

template <class M>
void printMatrix(const M &matrix)
{
    int height = matrix.shape()[0];
    int width = matrix.shape()[1];
    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            std::cout << matrix[row][col] << " ";
        }
        std::cout << "\n";
    }
}

vector<float> R_to_real(vector<cuFloatComplex> mat, int nb_rows, int nb_colomns)
{
    vector<vector<float>> r_mat;
    r_mat.reserve(1000);
    vector<float> rt;
    rt.reserve(1000);
    for (int i = 0; i < nb_rows; i++)
    {
        vector<float> v;
        v.reserve(2 * nb_colomns);
        vector<float> v1;
        v1.reserve(2 * nb_colomns);
        for (int j = 0; j < nb_colomns; j++)
        {
            cuFloatComplex c = mat[i * nb_colomns + j];
            float re_c = cuCrealf(c);
            float im_c = cuCimagf(c);
            v.push_back(re_c);
            v.push_back(-im_c);
            v1.push_back(im_c);
            v1.push_back(re_c);
        }
        rt.insert(rt.end(), v.begin(), v.end());
        rt.insert(rt.end(), v1.begin(), v1.end());
    }

    return rt;
}
vector<float> C_colomn_major(cuFloatComplex *vec, int dim)
{
    vector<float> r_vec;

    // r_vec.reserve(100);

    for (int i = 0; i < dim; i++)
    {

        cuFloatComplex c = vec[i];
        float re_c = cuCrealf(c);
        float im_c = cuCimagf(c);

        r_vec.push_back(re_c);
        r_vec.push_back(im_c);
    }
    for (int i = 0; i < dim; i++)
    {

        cuFloatComplex c = vec[i];
        float re_c = cuCrealf(c);
        float im_c = cuCimagf(c);

        r_vec.push_back(-im_c);
        r_vec.push_back(re_c);
    }
    // for(int i=0;i<r_vec.size();i++)cout<<" "<< r_vec[i];
    return r_vec;
}

cuFloatComplex *real_to_complex(double *mat, int nb_rows, int nb_columns)
{
    vector<cuFloatComplex> r_mat;
    for (int i = 0; i < nb_rows; i = i + 2)
    {
        for (int j = 0; j < nb_columns; j + 2)
        {
            r_mat.push_back(make_cuFloatComplex((float)mat[i * nb_columns + j], (float)-1 * mat[i * nb_columns + (j + 1)]));
        }
    }
    cuFloatComplex *r = &r_mat[0];
    return r;
}

void QR_decomposition(cuFloatComplex *Q, cuFloatComplex *R, cuFloatComplex *R1, int dim1, int dim2)
{
    int mat_size = dim1 * dim2;
    lapack_complex_float *tau, *Rl;
    tau = (lapack_complex_float *)malloc(dim1 * sizeof(lapack_complex_float));
    Rl = (lapack_complex_float *)malloc(mat_size * sizeof(lapack_complex_float));

    /** QR decomposition of H = R1 **/
    for (int j = 0; j < mat_size; j++)
    {
        float ql = cuCrealf(R1[j]);
        float ql1 = cuCimagf(R1[j]);
        Rl[j].real = ql;
        Rl[j].imag = ql1;
    }
    LAPACKE_cgeqrf(LAPACK_ROW_MAJOR, dim2, dim1, Rl, dim1, tau);
    for (int j = 0; j < mat_size; j++)
    {
        float ql = Rl[j].real;
        float ql1 = Rl[j].imag;
        R1[j] = make_cuFloatComplex(ql, ql1);
    }

    memset(R, 0, mat_size * sizeof(cuFloatComplex));
    // memset(Q1, 0, dim2 * dim2 * sizeof(cuFloatComplex));
    for (int i = 0; i < dim1; i++)
        for (int j = i; j < dim1; j++)
            R[i * dim1 + j] = R1[i * dim1 + j];
    LAPACKE_cungqr(LAPACK_ROW_MAJOR, dim2, dim1, dim1, Rl, dim1, tau);
    for (int j = 0; j < mat_size; j++)
    {
        float ql = Rl[j].real;
        float ql1 = Rl[j].imag;
        R1[j] = make_cuFloatComplex(ql, ql1);
    }
    for (int i = 0; i < dim2; i++)
        for (int j = 0; j < dim2; j++)
            Q[j * dim2 + i] = make_cuFloatComplex(cuCrealf(R1[i * dim2 + j]), -cuCimagf(R1[i * dim2 + j]));

    /** end QR decomposition of H = R1 **/
}

void Channel_matrix_noise_generation(cuFloatComplex *H, cuFloatComplex *Noise, int dim1, int dim2)
{
    int mat_size = dim1 * dim2;
    float a, b;
    cuFloatComplex ss = make_cuFloatComplex(1 / sqrt(2), 0);
    for (int i = 0; i < mat_size; i++)
    {
        a = (float)rand() / (float)(RAND_MAX);
        b = (float)rand() / (float)(RAND_MAX);
        H[i] = cuCmulf(ss, make_cuFloatComplex(a, b)); // matrice canal
    }
    for (int i = 0; i < dim2; i++)
    {
        a = (float)rand() / (float)(RAND_MAX);
        b = (float)rand() / (float)(RAND_MAX);
        Noise[i] = make_cuFloatComplex(a, b);
    }
}

int initialization(__half **_C, float **CC, __half **_A, float **AA, __half **A_half, __half **B_half, __half **C_half, float **new_mat_A, float **new_mat_B, float **new_mat_C, float **new_mat_B_1pas, __half **B1_half, int lz, int pas, int parallel_nodes, int dim1, vector<float> float_comb_v, vector<float> float_comb_v2)
{
    cudaError_t cudaStat1;
    cudaStat1 = cudaMalloc(CC, 2 * lz * pas * 2 * parallel_nodes * sizeof(float));
    if (cudaStat1 != cudaSuccess)
    {
        printf("device memory allocation failed matrix half C");
        return EXIT_FAILURE;
    }
    cudaStat1 = cudaMalloc(_C, 2 * lz * pas * 2 * parallel_nodes * sizeof(__half));
    if (cudaStat1 != cudaSuccess)
    {
        printf("device memory allocation failed matrix half C");
        return EXIT_FAILURE;
    }

    for (int i7 = 0; i7 < (dim1 / pas + (dim1 % pas)); i7++)
    {
#ifndef FP16MM
        // if(i7*pas<100)

        cudaStat1 = cudaMalloc(&AA[i7], 2 * pas * pas * 2 * sizeof(float));
        if (cudaStat1 != cudaSuccess)
        {
            printf("device memory allocation failed half matrix A");
            return EXIT_FAILURE;
        }
#else
        // if(i7*pas<100)

        cudaStat1 = cudaMalloc(&_A[i7], 2 * pas * pas * 2 * sizeof(__half));
        if (cudaStat1 != cudaSuccess)
        {
            printf("device memory allocation failed half matrix A");
            return EXIT_FAILURE;
        }
#endif
    }

    cudaStat1 = cudaMalloc(A_half, 2 * pas * pas * 2 * sizeof(__half));
    if (cudaStat1 != cudaSuccess)
    {
        printf("device memory allocation failed half matrix A");
        return EXIT_FAILURE;
    }

    cudaStat1 = cudaMalloc(C_half, 2 * lz * pas * 2 * parallel_nodes * sizeof(__half));
    if (cudaStat1 != cudaSuccess)
    {
        printf("device memory allocation failed matrix C");
        return EXIT_FAILURE;
    }

    cudaStat1 = cudaMallocManaged(B_half, float_comb_v.size() * sizeof(__half));
    if (cudaStat1 != cudaSuccess)
    {
        printf("device memory allocation failed half matrix A");
        return EXIT_FAILURE;
    }

    cudaStat1 = cudaMalloc(new_mat_A, 2 * pas * pas * 2 * sizeof(float));
    if (cudaStat1 != cudaSuccess)
    {
        printf("device memory allocation failed matrix A");
        return EXIT_FAILURE;
    }
    cudaMalloc(new_mat_B, float_comb_v.size() * sizeof(float));

    cudaStat1 = cudaMalloc(new_mat_C, 2 * lz * pas * 2 * parallel_nodes * sizeof(float));
    if (cudaStat1 != cudaSuccess)
    {
        printf("device memory allocation failed matrix C");
        return EXIT_FAILURE;
    }
    cudaMalloc(new_mat_B_1pas, float_comb_v2.size() * sizeof(float));

    cudaStat1 = cudaMallocManaged(B1_half, float_comb_v2.size() * sizeof(__half));
    if (cudaStat1 != cudaSuccess)
    {
        printf("device memory allocation failed half matrix A");
        return EXIT_FAILURE;
    }
}

float multi_level(std::vector<cuFloatComplex> R_ad, std::vector<int8_t> nn1,
                  std::vector<cuFloatComplex> nn2, vector<tree_node> &list_node, vector<cuFloatComplex> &best_ad, int nb_node_par, double d, int dim1, int dim2, int pas,
                  const float *new_mat_A,
                  const float *new_mat_B, float *new_mat_C, float *new_mat_B_1pas,
                  const __half *A_half,
                  const __half *B_half, __half *C_half,
                  const __half *B1_half,
                  struct timeval &branch_start,
                  struct timeval &branch_end,
                  struct timeval &end_pr,
                  double &elapsed_mul, double &elapsed_mul11, double &elapsed_branch,
                  cuFloatComplex *Y_cpu1,
                  __half *_C, __half *_A[500],
                  float *CC, float *AA[500], struct timeval &start_ad, int_fast8_t *A2, cuFloatComplex *R)
{
    float d_ad = 0;
    int l111 = (int)pow(modulation, pas);
    // root node
    tree_node nn0(nn1, 0, 0);
    list_node.push_back(nn0);
    best_ad = nn2;
    R_ad.clear();
    d_ad = d;
    //    cout<< "\n matrix R\n";
    for (int ie = 0; ie < dim1; ie++)
    {
        for (int je = 0; je < dim1; je++)
        {
            R_ad.push_back(R[ie * dim1 + je]);
            // cout<<" "<< cuCrealf(R[ie * dim1 + je])<<"+i"<<cuCimagf(R[ie * dim1 + je]);
        }
        //   cout<<"\n";
    }

    vector<float> R_adf = R_to_real(R_ad, dim1, dim1);

    vector<float> colomn_cpu = C_colomn_major(Y_cpu1, dim2);

    float *gpu_y_modulation;
    float *gpu_r_adf_in;
    cudaError_t cudaStat;
    //     cout << "\nradf_in size" << R_adf.size();
    cudaMalloc(&gpu_r_adf_in, R_adf.size() * sizeof(float));
    cudaStat = cudaMemcpy(gpu_r_adf_in, R_adf.data(), R_adf.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStat != cudaSuccess)
    {
        printf("device memory allocation failed GPU r in");
        printf("device memory allocation failed GPU r in");
        return EXIT_FAILURE;
    }
    float *H_D_eval;
    cudaStat = cudaMalloc(&H_D_eval, 1000 * parallel_nodes * sizeof(float));
    cudaMemset(H_D_eval, 0, 1000);
    if (cudaStat != cudaSuccess)
    {
        printf("device memory allocation failed GPU H_D_eval in");
        return EXIT_FAILURE;
    }

    cudaStat = cudaMalloc(&gpu_y_modulation, colomn_cpu.size() * sizeof(float));
    if (cudaStat != cudaSuccess)
    {
        printf("device memory allocation failed y_modulation");
        return EXIT_FAILURE;
    }

    cudaMemcpy(gpu_y_modulation, colomn_cpu.data(), colomn_cpu.size() * sizeof(float), cudaMemcpyHostToDevice);

    int nj = 0;
    for (int i9 = 0; i9 < (dim1 / pas); i9++)
    {
        p_new_mat_A<<<1, 2 * dim1>>>(AA[i9], i9 * pas, pas, dim1, dim2, gpu_r_adf_in, _A[i9]);
        nj++;
    }
    int nf = dim1 % pas;
    if (nf != 0)
    {
        for (int i9 = nj * pas; i9 < (dim1); i9++)
        {
            p_new_mat_A<<<1, 2 * dim1>>>(AA[nj], i9, 1, dim1, dim2, gpu_r_adf_in, _A[nj]);

            nj++;
        }
    }

    float *gpu_vec;
    cudaStat = cudaMalloc(&gpu_vec, nb_node_par * 2 * 2 * dim1 * sizeof(float));
    if (cudaStat != cudaSuccess)
    {
        printf("device memory allocation failed gpu vec");
        return EXIT_FAILURE;
    }

    float *d_norme;
    int *best_nd;

    cudaStat = cudaMalloc(&best_nd, ((l111 * nb_node_par) / 64 + 1) * sort_size * sizeof(int));
    if (cudaStat != cudaSuccess)
    {
        printf("device memory allocation failed for norme calculation");
        return EXIT_FAILURE;
    }

    found = 0;

    int jk = 0;
    vector<tree_node> list_temp;
    int pas_org = steps;
    int l111_org = l111;
    gettimeofday(&end_pr, NULL);
    gettimeofday(&start_ad, NULL);
    int nb_node_parallel = nb_node_par;
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    while (list_node.size() != 0)
    {
        pas = steps;
        l111 = l111_org;

        tree_node node;

        if (list_node.size() > 0)
        {
            node = list_node.back();
            list_node.pop_back();
        }
        if (node.evaluation < d)
        {
            int level = (int)node.level;
            if (level + pas_org > dim1)
            {
                pas = 1;
                l111 = modulation;
            }
            //       cout << "\n jk" << jk << " nd" << node.evaluation;
            vector<tree_node> vec;
            vec.push_back(node);
            list_temp.clear();

            nb_node_parallel = nb_node_par; //
            if (jk == 0)
                nb_node_parallel = 1;
            gettimeofday(&branch_start, NULL);
            /*matrix vector multiplication */
            if (pas == steps)
            {
                vecMul_v1<<<nb_node_parallel * 2 * pas, 2 * dim1, 0, stream1>>>(gpu_vec, gpu_r_adf_in, nb_node_parallel, dim1, pas, level, gpu_y_modulation);
            }
            else if (pas == 1)
            {

                vecMul_v1<<<2 * pas * nb_node_parallel, 2 * dim1, 0, stream1>>>(gpu_vec, gpu_r_adf_in, nb_node_parallel, dim1, pas, level, gpu_y_modulation);
            }
            // cudaDeviceSynchronize();
            gettimeofday(&branch_end, NULL);
            list_temp = branch_nodes(vec, A2, d, pas, 1);
            /*Matrix matrix multiplication + sorting phase*/
            float new_d = real_node_evaluation(list_node, list_temp, best_ad, level, H_D_eval, l111, dim1, dim2, pas, d, d_norme, best_nd, nb_node_parallel, gpu_vec, AA[jk], new_mat_B, CC, new_mat_B_1pas, _A[jk], B_half, _C, B1_half); // for cpu evaluation

            elapsed_mul = elapsed_mul + (mul2.tv_sec - mul1.tv_sec) + ((mul2.tv_usec - mul1.tv_usec) / 1000000.0);

            elapsed_mul11 = elapsed_mul11 + (mul22.tv_sec - mul11.tv_sec) + ((mul22.tv_usec - mul11.tv_usec) / 1000000.0);

            elapsed_branch = elapsed_branch + (branch_end.tv_sec - branch_start.tv_sec) + ((branch_end.tv_usec - branch_start.tv_usec) / 1000000.0);
            if (level + pas >= dim1)
                if (new_d < d - 0.000001)
                {
                    //   cout << "\n new d :" << new_d << " d: " << d << " level:" << level << " pas:" << pas;
                    d = new_d; // found=100;
                }
        }
        jk++;
        if (found != 0 && jk > 2)
            list_node.clear();

        pas = steps;
        nb_node_parallel = nb_node_par;
    }

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    cudaFree(gpu_r_adf_in);
    // cudaFree(gpu_y_modulation);
    cudaFree(gpu_vec);
    cudaFree(best_nd);
    cudaFree(H_D_eval);
}

int simulation_muti_level(int nTx, int nRx, int Iter, int steps, int nb_nodes, int nb_threads, int argc, char **argv)
{
    int usr = 1;
    int dim1 = nTx;
    int dim2 = nRx;
    int dim3 = usr;
    int matrix_size1 = dim1 * dim2;
    int matrix_size2 = dim1 * dim3;
    int matrix_size3 = dim2 * dim3;
    int_fast8_t *A2;
    cuFloatComplex *H, *H_initial, *R, *R1, *Q1, *S, *Noise, *Noise_initial, *Y_cpu, *Y_cpu1, alpha1, beta1, ss, *Decdata;
    // GPU pointers
    cuFloatComplex *dH, *dQ1, *dS, *dY_cpu, *dY_cpu1;
    int l = (int)pow(2, dim1);
    int *iteration;
    float a, b;
    double *SNRdb;
    double *BER;
    double d = 0;
    int pas = steps;
    std::vector<tree_node> list_node;
    list_node.reserve(500); // search tree
    std::vector<int8_t> nn1;
    std::vector<cuFloatComplex> nn2;
    std::vector<cuFloatComplex> R_ad;
    std::vector<cuFloatComplex> best_ad;
    std::vector<cuFloatComplex> best_ad2;
    cuFloatComplex v_n1 = make_cuFloatComplex(0, 0);
    int8_t v_0 = -1;
    for (int it = 0; it < dim1; it++)
    {
        nn1.push_back(v_0);
        nn2.push_back(v_n1);
    }
    float d_ad = 0;
    double *BER_ad, *T_ad;
    struct timeval start_ad, end_ad;

    BER_ad = (double *)malloc(20 * sizeof(double)); // Symbol error rate for each SNR
    memset(BER_ad, 0.0, 20 * sizeof(double));
    T_ad = (double *)malloc(20 * sizeof(double)); // time for each SNR
    memset(T_ad, 0, 20 * sizeof(double));

    // norm and number of bits transmitted by each antenna

    int norm = 1, nbit = 1;
    if (modulation == 2)
    {
    }
    else if (modulation == 4)
    {
        norm = 2;
        nbit = 2;
    }
    else if (modulation == 16)
    {
        norm = 10;
        nbit = 4;
    }
    else if (modulation == 64)
    {
        norm = 42;
        nbit = 6;
    }
    else
    {
        printf("\n constellation not supported");
        exit(0);
    }

    vector<int_fast8_t> BB = Branching_combinations(modulation, pas);
    vector<cuFloatComplex> comb_g = int8_to_complex(BB, Constellation_symbols);
    vector<float> float_comb_v;
    vector<float> float_comb_v2;
    for (int u = 0; u < parallel_nodes; u++)
        for (int f = 0; f < comb_g.size(); f += pas)
        {
            for (int j1 = pas - 1; j1 >= 0; j1--)
            {
                cuFloatComplex tm = comb_g[f + j1];
                float_comb_v.push_back(cuCrealf(tm));
                float_comb_v.push_back(cuCimagf(tm));
            }
        }
    for (int u = 0; u < parallel_nodes; u++)
        for (int f = 0; f < Constellation_symbols.size(); f++)
        {
            cuFloatComplex tm = Constellation_symbols[f];
            float_comb_v2.push_back(cuCrealf(tm));
            float_comb_v2.push_back(cuCimagf(tm));
        }
    cudaError_t cudaStat1;
    cuFloatComplex *gpu_comb, *gpu_comb_1pas;
    float *new_mat_B, *new_mat_B_1pas;
    __half *_C, *_A[500];
    float *CC, *AA[500];
    __half *A_half, *B_half, *B1_half, *C_half;
    float *new_mat_A;
    float *new_mat_C;
    int lz = (int)pow(modulation, pas);
    // initialization of Multi-level approach
    initialization(&_C, &CC, &_A[0], &AA[0], &A_half, &B_half, &C_half, &new_mat_A, &new_mat_B, &new_mat_C, &new_mat_B_1pas, &B1_half, lz, pas, parallel_nodes, dim1, float_comb_v, float_comb_v2);
    cudaMemcpy(new_mat_B, float_comb_v.data(), float_comb_v.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(new_mat_B_1pas, float_comb_v2.data(), float_comb_v2.size() * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < float_comb_v.size(); i++)
    {
        B_half[i] = __float2half(float_comb_v[i]);
    }
    for (int i = 0; i < float_comb_v2.size(); i++)
    {
        B1_half[i] = __float2half(float_comb_v2[i]);
    }

    A2 = &BB[0];

    // allocate and initialize cpu memory
    H = (cuFloatComplex *)malloc(matrix_size1 * sizeof(cuFloatComplex)); // matrice du canal
    H_initial = (cuFloatComplex *)malloc(matrix_size1 * sizeof(cuFloatComplex));
    R = (cuFloatComplex *)malloc(matrix_size1 * sizeof(cuFloatComplex));  // triangular matrix R de la decomposition QR
    R1 = (cuFloatComplex *)malloc(matrix_size1 * sizeof(cuFloatComplex)); // Orthogonal matrix Q de la decomposition QR
    Q1 = (cuFloatComplex *)malloc(dim2 * dim2 * sizeof(cuFloatComplex));
    S = (cuFloatComplex *)malloc(matrix_size2 * sizeof(cuFloatComplex));             // modulation, generation des symboles complex
    Noise = (cuFloatComplex *)malloc(matrix_size3 * sizeof(cuFloatComplex));         // le bruit gaussien
    Noise_initial = (cuFloatComplex *)malloc(matrix_size3 * sizeof(cuFloatComplex)); // initial noise
    Y_cpu = (cuFloatComplex *)malloc(matrix_size3 * sizeof(cuFloatComplex));         // le vecteur recu y = Hs + n
    Y_cpu1 = (cuFloatComplex *)malloc(matrix_size3 * sizeof(cuFloatComplex));        // Q'*y
    Decdata = (cuFloatComplex *)malloc(matrix_size2 * sizeof(cuFloatComplex));       // le signal decode apres le sphere decoder

    // allocate and initialize gpu memory
    cudaMalloc((void **)&dH, matrix_size1 * sizeof(cuFloatComplex));
    cudaMalloc((void **)&dS, matrix_size2 * sizeof(cuFloatComplex));
    cudaMalloc((void **)&dY_cpu, matrix_size3 * sizeof(cuFloatComplex));
    cudaMalloc((void **)&dY_cpu1, matrix_size3 * sizeof(cuFloatComplex));
    cudaMalloc((void **)&dQ1, dim2 * dim2 * sizeof(cuFloatComplex));

    // SNR
    SNRdb = (double *)malloc(20 * sizeof(double));
    SNRdb[0] = 0.0;
    SNRdb[1] = 4.0;
    SNRdb[2] = 8.0;
    SNRdb[3] = 12.0;
    SNRdb[4] = 16.0;
    SNRdb[5] = 20.0;
    SNRdb[6] = 24.0;
    SNRdb[7] = 26.0;
    SNRdb[8] = 30.0;
    SNRdb[9] = 31.0;
    SNRdb[10] = 32.0;
    SNRdb[11] = 23;
    SNRdb[12] = 28.0;
    SNRdb[12] = 33.5;
    SNRdb[13] = 34.0;
    SNRdb[14] = 34.5;
    SNRdb[15] = 35.0;
    // Ber
    BER = (double *)malloc(20 * sizeof(double));
    memset(BER, 0.0, 13 * sizeof(double));
    struct timeval branch_start, branch_end;
    struct timeval start_pr, end_pr;
    double elapsed_mul = 0, elapsed_mul11 = 0, elapsed_branch = 0;
    int l11 = (int)pow(modulation, pas);
    gettimeofday(&start_ad, NULL);
    cublasCreate(&handle);
    // cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    //  if (Iter != 0)

    for (int jj = 0; jj < Iter; jj++) // Montecarlo simulations
    {
        if (jj % 500 == 0)
            printf("\n iter = %d \n", jj);
        /* Channel matrix, noise vector, and QR decomposition */
        Channel_matrix_noise_generation(H_initial, Noise_initial, dim1, dim2);

        for (int i = 0; i < matrix_size1; i++)
        {
            H[i] = H_initial[i];
            R1[i] = H[i];
        }
        QR_decomposition(Q1, R, R1, dim1, dim2);
        /* end of Channel matrix, noise vector, and QR decomposition */
       // int ii = 0;
       // if (jj > 2000)
       //     ii = 4;
        for (int ii=0; ii < 9; ii++) // Changing the SNR
        {

            double variance = dim1 * pow(10.0, -((SNRdb[ii]) / 10)); // noise according to SNR

            d = dim2 * variance + modulation + 10000;
            d = 10000000000000;

            ss = make_cuFloatComplex(1 / sqrt(2), 0);
            memset(Y_cpu, 0, matrix_size3 * sizeof(cuFloatComplex));
            memset(Y_cpu1, 0, matrix_size3 * sizeof(cuFloatComplex));
            memset(Decdata, 0, matrix_size2 * sizeof(cuFloatComplex));

            /*Generating the data to send number of symbols = number of antennas*/
            for (int i = 0; i < matrix_size2; i++)
            {
                a = rand() % modulation;
                if (modulation == 2)
                { // S11[i] =BPSK[a];
                    S[i] = BPSK[a];
                }
                else if (modulation == 4)
                { // S11[i] =qam4[a];
                    S[i] = qam4[a];
                }
                else if (modulation == 16)
                { // S11[i] =qam16[a];
                    S[i] = qam16[a];
                }
                else if (modulation == 64)
                { // S11[i] =qam64[a];
                    S[i] = qam64[a];
                }
            }
            /*normalization of symbols*/
            for (int i = 0; i < matrix_size2; i++)
            {
                S[i] = cuCmulf(S[i], make_cuFloatComplex(1 / sqrt(norm), 0));
            }

            //  float lsnr = (pow(10, ((SNRdb[ii]) / 10)));
            float lsnr = (pow(10, ((SNRdb[ii]) / 10)));
            // float Es = 1;
            float noisevar = 1 / lsnr;

            /*Adding the noise*/
            for (int i = 0; i < matrix_size3; i++)
            {
                // Noise[i] = make_cuFloatComplex(a * sqrt(noisevar * 0.5), b * sqrt(noisevar * 0.5));
                Noise[i] = cuCmulf(make_cuFloatComplex(sqrt(noisevar * 0.5), 0.0), Noise_initial[i]); // bruit
                Y_cpu[i] = Noise[i];
            }

            /** y = Hs on GPU **/
            alpha1 = make_cuFloatComplex(1, 0);
            beta1 = make_cuFloatComplex(1, 0);
            cudaMemcpy(dH, H, matrix_size1 * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
            cudaMemcpy(dS, S, matrix_size2 * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
            cudaMemcpy(dY_cpu, Y_cpu, matrix_size3 * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

            const cuFloatComplex *alpha2 = &alpha1;
            const cuFloatComplex *beta2 = &beta1;

            cublasCgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, dim2, dim3, dim1, alpha2, dH, dim2, dS, dim1, beta2, dY_cpu, dim2);

            cudaMemcpy(Y_cpu, dY_cpu, matrix_size3 * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
            /** end y = Hs on GPU **/

            /** Q'*y +n on GPU  **/
            alpha1 = make_cuFloatComplex(1, 0);
            beta1 = make_cuFloatComplex(0, 0);
            cudaMemcpy(dQ1, Q1, dim2 * dim2 * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
            cudaMemcpy(dY_cpu1, Y_cpu1, matrix_size3 * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

            alpha2 = &alpha1;
            beta2 = &beta1;

            cublasCgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, dim2, dim3, dim2, alpha2, dQ1, dim2, dY_cpu, dim2, beta2, dY_cpu1, dim2);

            cudaMemcpy(Y_cpu1, dY_cpu1, matrix_size3 * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

            for (int i = 0; i < matrix_size2; i++)
            {
                Y_cpu1[i] = cuCmulf(Y_cpu1[i], make_cuFloatComplex(sqrt(norm), 0)); // S[i]=S11[i];
            }
            /** end Q'*y +n on GPU  **/

            for (int i = 0; i < matrix_size2; i++)
            {
                S[i] = cuCmulf(S[i], make_cuFloatComplex(sqrt(norm), 0)); // printf("\nS: %f %f ",cuCrealf(S[i]), cuCimagf(S[i]));
            }

            int nb_node_par = parallel_nodes;
            gettimeofday(&start_pr, NULL);
            /* starting of multi-level approach*/
            multi_level(R_ad, nn1,
                        nn2, list_node, best_ad, nb_node_par, d, dim1, dim2, pas,
                        new_mat_A,
                        new_mat_B, new_mat_C, new_mat_B_1pas,
                        A_half,
                        B_half, C_half,
                        B1_half,
                        branch_start,
                        branch_end,
                        end_pr,
                        elapsed_mul, elapsed_mul11, elapsed_branch,
                        Y_cpu1,
                        _C, _A,
                        CC, AA, start_ad, A2, R);
            /* end of multi-level approach*/
            gettimeofday(&end_ad, NULL);
            double elapsed_ad = (end_ad.tv_sec - start_ad.tv_sec) + ((end_ad.tv_usec - start_ad.tv_usec) / 1000000.0);
            double t1_ad = elapsed_ad;

            double elapsed_preparation = (end_pr.tv_sec - start_pr.tv_sec) + ((end_pr.tv_usec - start_pr.tv_usec) / 1000000.0);
            double t1_pr = elapsed_preparation;
            if (jj == 10 & ii==0)
                cout << "\n elapsed preparation:" << t1_pr;

            d = d_ad;
            if (jj > 0)
                T_ad[ii] = T_ad[ii] +
                           t1_ad;

            if (found != 0)
            {
                for (int il = 0; il < matrix_size2; il++)
                {
                    double d1 = cuCrealf(S[il]);
                    double d2 = cuCrealf(best_ad[il]);
                    double d3 = cuCimagf(S[il]);
                    double d4 = cuCimagf(best_ad[il]);
                    double dd = d1 - d2;
                    double dd1 = d3 - d4;
                    if (dd > 0.25 || dd1 > 0.25)
                    {
                        BER_ad[ii] = BER_ad[ii] + 1;
                    }
                }
            }
            else
            {
                for (int il = 0; il < 1; il++)
                    printf(" not found");
            }
        }
    }
   
    cout
        << left
        << setw(20)
        << "\n SNR"
        << left
        << setw(25)
        << "Complexity"
        << left
        << setw(28)
        << "Error rate"
        << endl;

    for (int ii = 0; ii < 9; ii++)
    {      T_ad[ii] = T_ad[ii] / (Iter);
        BER_ad[ii] = BER_ad[ii] / (dim1 * Iter * 1 * nbit / nbit);
          cout
            << left
            << setw(20)
            << SNRdb[ii]
            << left
            << setw(25)
            << T_ad[ii] 
            << left
            << setw(28)
            << BER_ad[ii] 
            << endl;
   
        //   cout << "\nmul: " << elapsed_mul / Iter << " branch:" << elapsed_branch / (Iter);
        // cout << "\n matrix multiplication: " << elapsed_mul11 / (Iter) << " branch:" << elapsed_branch / (Iter);
     //   printf("\n time:%f ii %d", T_ad[ii], ii);
      //  printf("\n Ber_ad[%d(%fdb)] = %.15f ", ii, SNRdb[ii], BER_ad[ii]);
        //    }
    }
    cudaFree(gpu_comb);
    cudaFree(gpu_comb_1pas);
    cudaFree(new_mat_B);
    cudaFree(new_mat_B_1pas);
    cudaFree(dH);
    cudaFree(dS);
    cudaFree(dY_cpu);
    cudaFree(dY_cpu1);
    cudaFree(dQ1);
    cudaFree(new_mat_A);
    cudaFree(new_mat_C);
    cublasDestroy(handle);
    cudaFree(_C);
    cudaFree(CC);
    for (int i = 0; i < (dim1 / steps + (dim1 % steps)); i++)
    {
#ifndef FP16MM

        cudaFree(AA[i]);
#else

        cudaFree(_A[i]);
#endif
    }
    return 0;
}

// CPU code
int main(int argc, char *argv[])
{
    sscanf(argv[1], "%d", &nTx);
    sscanf(argv[2], "%d", &nRx);
    sscanf(argv[3], "%d", &steps);
    sscanf(argv[4], "%d", &Iter);
    sscanf(argv[5], "%d", &modulation);
    sscanf(argv[6], "%d", &parallel_nodes);

#ifndef FP16MM
    cout << "\nrunning cublasSgemm test\n"
         << endl;
#else
    cout << "\nrunning half precision mode cublasHgemm test\n"
         << endl;
#endif

    sort_size = parallel_nodes;
    printf("\nnew GPU multi-level MIMO System : %dx%d ", nTx, nRx);
    if (strategy_exploration == 3)
        printf("\nexploration strategy: Braedth first  %d", strategy_exploration);
    if (strategy_exploration == 2)
        printf("\nexploration strategy: depth first   %d", strategy_exploration);
    if (strategy_exploration == 1)
        printf("\nexploration strategy: best first  %d", strategy_exploration);

    printf("\n incrimental 1=yes 0=no ::: %d", incremental);
    printf("\n levels: %d", steps);
    printf("\n number iteration:%d", Iter);
    printf("\n number threads:%d", nb_threads);
    printf("\n modulation:%d", modulation);
    printf("\n parallel nodes:%d", parallel_nodes);
    int dev = 0;
    cudaError_t err = cudaSetDevice(dev);
    if (err == cudaSuccess)
    {
        cout << "\n GPU device " << dev << " set  succesfully";
    }
    if (nTx == nRx && nTx > 5)
    {
        simulation_muti_level(nTx, nRx, Iter, steps, nb_nodes, nb_threads, argc, argv);
    }
    else
    {
        cout << "\n Number of transmit  antennas must equal number of recieve antennas >5";
    }

    return 0;
}