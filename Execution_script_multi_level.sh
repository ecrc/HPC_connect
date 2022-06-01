#!/bin/bash
module load mkl
module load cuda/11.4
module load cmake
# MIMO configuration
M=100            # Number of antennas in MIMO MxM.  M>5 
modulation=64    # Modulation used BPSK(2), 4-QAM(4), 16-QAM(16), 64-QAM(64)   
nb_simulation=200  # Number of bits = nb_simulation*M*bits_per_symbol
GPU_device=0       # Id of GPU device
nb_level=3         # Number of combined levels between 1 and 5 
precision=1      # 1 for single-precision and 2 for half-precision
SNR_min=0
SNR_max=30 

if (($precision==1))
then
./GPU_sd $M $M $nb_level $nb_simulation $modulation 1 >output.txt
fi
echo "End of the simulation please see output.txt for error rate and complexity results"
