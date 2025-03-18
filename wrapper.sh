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

# python exp_refactor.py  --N 20 \
#  --dim 2 \
#    --l 1 \
#      --h 1 \
#        --f 32  \
#         --bs 32  \
#          --epochs 1500 \
#            --num_samples 8192  \
#              --repeat 1 \
#              --lr "4e-3" \
#              --dropout .2 \
#              --wd .01 \
#              --world_size 8 \
#               --backend nccl \
#               --stop_loss .000002 \
#               --save_every 50 \
#               --save_checkpoints

python exp_refactor.py  --N 20 \
 --dim 2 \
 --dim2 2 \
   --l 1 \
     --h 1 \
       --f 128  \
        --bs 32  \
         --epochs 1500 \
           --num_samples 8192  \
             --repeat 1 \
             --lr "4e-3" \
             --dropout 0.2 \
             --wd .00001 \
             --world_size 8 \
              --backend nccl \
              --stop_loss .0002 \
              --save_every 10 \
