CC= nvcc -arch=sm_80 --fmad=true -std=c++14 --disable-warnings -lmkl_intel_ilp64  -lcudart -lcublas  -lcublas_static -g -O3 # degugage 
CCOPTS= -DMKL_ILP64 -m64
CFLAGS = -Wall -std=gnu99 $(CCOPTS)

# librmoduaries to link against
LIB= -lcudart -lcublas

# mkl multi threaded
LIB+=-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5  -lpthread -lm -ldl  -DMKL_ILP64 -Xcompiler -fopenmp    #-liomp5  

C_SRC=GPU_sd.cu

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
