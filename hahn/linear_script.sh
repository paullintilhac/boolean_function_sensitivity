#!/bin/bash

#SBATCH --account=temfom0  # Specify the account to charge

#SBATCH --job-name=my_job  # Job name

#SBATCH --output=my_job_%j.out  # Standard output and error log

#SBATCH --error=my_job_%j.err  
#SBATCH --time=12:00:00  # Time limit hrs:min:sec

#SBATCH --partition=gpuq  # Specify the partition to submit to
#SBATCH --gres=gpu:1
echo $(hostname -s)
nvcc --version
source activate gpuq
echo conda env:
conda info --env

while getopts d:f:l:h:i: flag
do
    case "${flag}" in
        d) hidden_dim=${OPTARG};;
        f) ff_dim=${OPTARG};;
        l) layers=${OPTARG};;
        h) heads=${OPTARG};;
        i) iters=${OPTARG};;
    esac
done
echo "hidden_dim: $hidden_dim";
echo "ff_dim: $ff_dim";
echo "layers: $layers";
echo "heads: $heads";
echo "iterations: $iters"
echo filename in script
echo "d_$hidden_dim-f_$ff_dim-l_$layers-h_$heads"

python ../losses_linear_spectrum.py --d $hidden_dim --f $ff_dim --l $layers --h $heads --i $iters > log_linear
