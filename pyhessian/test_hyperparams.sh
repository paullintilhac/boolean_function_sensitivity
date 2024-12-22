#!/bin/bash

python exp_refactor.py       --world_size 8      --N 20            --dim 30         --l 1         --h 1         --f 128         --bs 32         --epochs 100000         --num_samples 8192      --repeat 1 --lr "1e-4" --save_every 200 --dropout .5 --save_checkpoints --backend nccl

python exp_refactor.py       --world_size 8      --N 20            --dim 60         --l 1         --h 1         --f 128         --bs 32         --epochs 100000         --num_samples 8192      --repeat 1 --lr "1e-4" --save_every 200 --dropout .5 --save_checkpoints --backend nccl

python exp_refactor.py       --world_size 8      --N 20            --dim 120         --l 1         --h 1         --f 256         --bs 32         --epochs 100000         --num_samples 8192      --repeat 1 --lr "1e-4" --save_every 200 --dropout .5--save_checkpoints --backend nccl


python exp_refactor.py      --world_size 8   --N 20            --dim 30         --l 1         --h 1         --f 128         --bs 64         --epochs 100000         --num_samples 16384      --repeat 1 --lr "2e-4" --save_every 200 --dropout .5 --save_checkpoints --backend nccl

python exp_refactor.py        --world_size 8     --N 20            --dim 60         --l 1         --h 1         --f 128         --bs 64         --epochs 100000         --num_samples 16384      --repeat 1 --lr "2e-4" --save_every 200 --dropout .5 --save_checkpoints --backend nccl

python exp_refactor.py       --world_size 8      --N 20            --dim 120         --l 1         --h 1         --f 256         --bs 64         --epochs 100000         --num_samples 16384      --repeat 1 --lr "2e-4" --save_every 200 --dropout .5 --save_checkpoints --backend nccl

