#!/usr/bin/env bash
set -euo pipefail

# Usage: ./start_tmux.sh /path/to/wrapper.sh
WRAPPER_REL="${1:-./wrapper.sh}"

WRAPPER_ABS="$(readlink -f "$WRAPPER_REL")"
WRAPPER_DIR="$(dirname "$WRAPPER_ABS")"
OUT="$WRAPPER_DIR/out.log"
ERR="$WRAPPER_DIR/err.log"

# put tmux socket under the job dir (not /tmp)
SOCK_DIR="${_CONDOR_SCRATCH_DIR:-$WRAPPER_DIR}"
SOCK_PATH="$SOCK_DIR/tmux.sock"
SESSION="run"

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found in PATH" >&2; exit 1
fi
if [[ ! -f "$WRAPPER_ABS" ]]; then
  echo "ERROR: wrapper.sh not found at $WRAPPER_ABS" >&2; exit 1
fi
chmod +x "$WRAPPER_ABS" || true

# start a persistent login shell session (using custom socket)
if ! tmux -S "$SOCK_PATH" has-session -t "$SESSION" 2>/dev/null; then
  tmux -S "$SOCK_PATH" new-session -d -s "$SESSION" "bash -l"
fi

# cd to the wrapper dir and launch the job with explicit logs; close stdin
tmux -S "$SOCK_PATH" send-keys -t "$SESSION" "cd \"$WRAPPER_DIR\"" C-m
tmux -S "$SOCK_PATH" send-keys -t "$SESSION" \
  "bash -lc 'exec \"$WRAPPER_ABS\" </dev/null > \"$OUT\" 2> \"$ERR\"'" C-m

echo "Launched: $WRAPPER_ABS"
echo "  stdout -> $OUT"
echo "  stderr -> $ERR"
echo "Attach:   tmux -S \"$SOCK_PATH\" attach -t $SESSION   (Ctrl-b then d to detach)"

