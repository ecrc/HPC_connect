#!/bin/bash
# MIMO configuration
M=100            # Number of antennas in MIMO MxM.  M>5 
modulation=64    # Modulation used BPSK(2), 4-QAM(4), 16-QAM(16), 64-QAM(64)   
nb_simulation=200  # Number of bits = nb_simulation*M*bits_per_symbol
GPU_device=0       # Id of GPU device
nb_level=4         # Number of combined levels between 1 and 5 
precision=s      # s for single-precision, h for half-precision, and i for INT8 precision
SNR_min=0
SNR_max=30 

./GML $M $M $nb_level $nb_simulation $modulation $precision >output.txt
echo "End of the simulation please see output.txt for error rate and complexity results"
