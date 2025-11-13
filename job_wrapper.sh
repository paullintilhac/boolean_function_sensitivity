#!/bin/bash
# --- Normalize CUDA visibility (fix truncated UUIDs from scheduler) ---
# If CUDA_VISIBLE_DEVICES contains GPU UUIDs (possibly truncated), map them to indices.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && echo "$CUDA_VISIBLE_DEVICES" | grep -q 'GPU-'; then
  echo "Normalizing CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  MAP="$(nvidia-smi --query-gpu=uuid,index --format=csv,noheader)"
  RES=""
  IFS=',' read -ra TOKS <<< "$CUDA_VISIBLE_DEVICES"
  for t in "${TOKS[@]}"; do
    t="$(echo "$t" | xargs)"  # trim spaces
    idx="$(echo "$MAP" | awk -v u="$t" '$1 ~ u {print $2}')"
    [ -n "$idx" ] && RES="${RES}${RES:+,}$idx"
  done
  if [ -n "$RES" ]; then
    export CUDA_VISIBLE_DEVICES="$RES"
  else
    # If mapping failed (e.g., tokens too truncated), unmask to show all GPUs assigned
    unset CUDA_VISIBLE_DEVICES
  fi
  echo "CUDA_VISIBLE_DEVICES -> ${CUDA_VISIBLE_DEVICES:-<unset>}"
fi

# Use PCI bus order so indices are stable and match NCCL expectations
export CUDA_DEVICE_ORDER=PCI_BUS_ID
desired_name="boolean_function_sensitivity"
current_dir_name=$(echo "${PWD##*/}")

if [ "$current_dir_name" = "$desired_name" ]; then
    echo "Current directory name is '$desired_name'. skipping cloning"
else
    echo "Current directory name is not '$desired_name'. assuming repo has not been cloned yet"
    python3 -m ensurepip --default-pip
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    python3 -m pip install pyhessian pandas
    git clone https://github.com/paullintilhac/boolean_function_sensitivity.git
    cd boolean_function_sensitivity
    git fetch
    git checkout hardcoded
fi

python3 exp_refactor.py  --N 20 \
 --dim 2 \
     --h 1 \
       --f 128 \
        --bs 64  \
         --epochs 100000 \
           --num_samples 8192  \
             --repeat 1 \
             --lr "4e-3" \
             --dropout 0.1 \
             --wd .0001 \
             --world_size 8 \
              --backend nccl \
              --stop_loss .02 \
              --save_every 10 \
              --sam 
#watch -n 1 nvidia-smi

