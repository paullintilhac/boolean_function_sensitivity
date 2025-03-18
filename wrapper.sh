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


python exp_refactor.py  --N 20 \
 --dim 2 \
   --l 1 \
     --h 1 \
       --f 32  \
        --bs 256  \
         --epochs 1500 \
           --num_samples 32768  \
             --repeat 1 \
             --lr "4e-3" \
             --dropout .5 \
             --wd .1 \
             --world_size 8 \
              --backend nccl \
              --stop_loss .02 \
              --save_every 50 \