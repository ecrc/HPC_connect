

# cuda root .. change as needed
#_CUDA_ROOT_ = /usr/local/cuda

# mkl root .. change as needed
#_MKL_ROOT_ = /opt/intel/mkl
#_ICC_ROOT_ = /opt/intel/compiler

#NVCC=nvcc
CC= nvcc -arch=sm_70 --fmad=true -std=c++14 --disable-warnings -lmkl_intel_ilp64  -lcudart -lcublas  -lcublas_static -g -O3 # degugage 
CCOPTS= -DMKL_ILP64 -m64
CFLAGS = -Wall -std=gnu99 $(CCOPTS)

# include and lib paths
#INCLUDES=-I${_CUDA_ROOT_}/include -I$(_MKL_ROOT_)/include
#LIB_PATH=-L${_CUDA_ROOT_}/lib64 -L$(_MKL_ROOT_)/lib/intel64 -L$(_ICC_ROOT_)/lib/intel64

# librmoduaries to link against
LIB= -lcudart -lcublas

# mkl multi threaded
LIB+=-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5  -lpthread -lm -ldl  -DMKL_ILP64 -Xcompiler -fopenmp    #-liomp5  

#C_SRC=SD_new_exploration.cu
#C_SRC=serial_sd.cu
#C_SRC=CPU_SD.cu
#C_SRC=opt_CPU_sd.cu
C_SRC=GPU_sd.cu
#C_SRC=multi-GPU-sd.cu
#C_SRC=master-multi-GPU-sd.cu
#C_SRC=multi-level-gemm.cu

C_OBJ=$(C_SRC:.cu=.o)
ALL_OBJ=$(C_OBJ)

EXE=$(C_SRC:.cu=)

%.o: %.cu
	$(CC) -c $(INCLUDES) $< -o $@

$(EXE): $(C_OBJ)
	$(CC) $(LIB_PATH) $(LIB) $(C_SRC) -o  $(EXE)

all: $(EXE)

clean:
	rm -f *.o $(EXE)

#nvcc -std=c++11 -lmkl_intel_ilp64  -lcudart -lcublas  -lcublas_static  -O3 -lmkl_intel_thread -lmkl_core -liomp5  -lpthread -lm    SD_sgemm_for_gpu_dev.cu -o SD_sgemm_for_gpu_dev
#nvcc -std=c++11 -lcublas_static  -O3   -lcudart -lcublas -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5  -lpthread -lm    SD_sgemm_for_gpu_dev.cu -o  SD_sgemm_for_gpu_dev
#nvcc -std=c++11 -lcublas_static  -O3  SD_sgemm_for_gpu_dev.o -o SD_sgemm_for_gpu_dev  -lcudart -lcublas -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5  -lpthread -lm  