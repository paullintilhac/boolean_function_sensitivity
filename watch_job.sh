#!/bin/bash
tmux new -d -s job

# Pane 1 (left): GPU monitor that auto-refreshes
tmux send-keys  -t job 'watch -n1 "nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,noheader"' C-m

# Split to make Pane 2 (right)
tmux split-window -h -t job

# In Pane 2: cd into your work dir and start your long training
# (edit the path/command below to your actual job)
tmux send-keys  -t job.right 'bash wrapper.sh' C-m

# Attach (you’ll see both panes)
tmux attach -t job

