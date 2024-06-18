#!/bin/bash

echo before sbatch opts 

#SBATCH --account=temfom0  # Specify the account to charge

#SBATCH --job-name=my_job  # Job name

#SBATCH --output=my_job_%j.out  # Standard output and error log

#SBATCH --error=my_job_%j.err  
#SBATCH --time=12:00:00  # Time limit hrs:min:sec

#SBATCH --partition=gpuq  # Specify the partition to submit to
#SBATCH --gres=gpu:1

echo before parse args in script file
while getopts d:f:l:h: flag
do
    case "${flag}" in
        d) hidden_dim=${OPTARG};;
        f) ff_dim=${OPTARG};;
        l) layers=${OPTARG};;
        h) heads=${OPTARG};;
    esac
done
echo after parse args
echo "hidden_dim: $hidden_dim";
echo "ff_dim: $ff_dim";
echo "layers: $layers";
echo "heads: $layers";


echo "d_$hidden_dim-f_$ff_dim-l_$layers-h_$heads"

python ../losses_linear_spectrum.py --d $hidden_dim --f $ff_dim --l $layers --h $heads > log_1
