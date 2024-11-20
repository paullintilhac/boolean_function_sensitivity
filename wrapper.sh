#!/bin/bash

#SBATCH --account=temfom0  # Specify the account to charge

#SBATCH --job-name=my_job  # Job name

#SBATCH --output=my_job_%j.out  # Standard output and error log

#SBATCH --error=my_job_%j.err  
#SBATCH --time=24:00:00  # Time limit hrs:min:sec

#SBATCH --partition=gpuq  # Specify the partition to submit to
# echo $(hostname -s)
# nvcc --version
# source activate gpuq
# echo conda env:
# conda info --env

# python exp_test.py \
#         --N 30 \
#         --width 1 \
#         --dim 120 \
#         --l 2 \
#         --h 1 \
#         --f 128 \
#         --bs 2 \
#         --epochs 2000 \
#         --num_samples 1000 \
#         --repeat 5


python exp_refactor.py  
    --N 30      \
       --width 1  \
              --dim 120      \
                 --l 2     \
                     --h 1     \
                         --f 128     \
                             --bs 2048      \
                                --epochs 2000      \
                                   --num_samples 65536    \
                                        --repeat 1 \
                                        --lr "5e-5"