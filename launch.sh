#!/usr/bin/env bash
# start_job.sh — launch your command in tmux so it persists after logout

set -euo pipefail

# name your tmux session
SESSION_NAME="run"

# the command you actually want to run:
#   - output goes to out.log
#   - stderr goes to err.log
#   - stdin is closed so it won't hang
CMD="bash -lc 'exec ./wrapper.sh </dev/null > out.log 2> err.log'"

# create or reattach the tmux session
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists. Attaching..."
    tmux attach -t "$SESSION_NAME"
else
    echo "Starting new tmux session '$SESSION_NAME'..."
    tmux new-session -d -s "$SESSION_NAME" "$CMD"
    echo "Started. Logs: out.log (stdout), err.log (stderr)"
    echo "You can attach anytime with: tmux attach -t $SESSION_NAME"
fi

