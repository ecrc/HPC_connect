# HPC_connect Project
## 1. What is HPC_connect?
HPC-connect project is a parallel framework that exploits GPUs to accelerate the Massive MIMO detection physical layer. 
To this aim, the project introduces a scalable non-linear approach, called the GPU-Multi-Level approach, that can support a massive number of antennas at both the transmitter and the receiver.    

## 2. Vision of HPC_connect
HPC_connect is a collaboration between the Extreme Computing Research Center (ECRC) and the Communication Theory Lab (CTL) at KAUST. Its main goal is to reduce the latency requirements of next-generation wireless communication networks. More specifically, the project goal is to provide full HPC-based massive-MIMO physical layers (channel estimation, detection, precoding), which helps the deployment of a massive number of antennas in Next-G basestation while operating within the power budget of the basestation.     

## 3. Current Version: 0.0.1

## 4. Dependencies  and test on NVIDIA GPUs
We use cmake@3.21.0, MKL, and cuBLAS to build the code. 

module load cuda/11.4 

module load mkl/2020.0.166

## 5. Compilation 
###### Single-precision mode (FP32) (A100,V100 GPUs)
make -f makefile

For older GPU architectures update the makefile accordengly. For instance: CC= nvcc -v -gencode arch=compute_35,... 

## 6. Tests
To execute the code please specify the MIMO configuration in Execution_script_multi_level.sh
###### Single-precision mode (FP32)
bash Execution_script_multi_level.sh 

## 7. Handout


