#!/usr/bin/env bash
set -euo pipefail

# Prefer a system python3 if present
if command -v python3 >/dev/null 2>&1; then
  PY=python3
# Common location in ML images (conda-based)
elif [ -x /opt/conda/bin/python ]; then
  export PATH="/opt/conda/bin:$PATH"
  PY=python
# Fallback: some images only have /usr/bin/python
elif [ -x /usr/bin/python3 ]; then
  PY=/usr/bin/python3
else
  echo "No python found (tried python3, /opt/conda/bin/python, /usr/bin/python3)"
  exit 127
fi

#echo "Using: $($PY -V)"
#$PY your_script.py  # <-- change to your entrypoint

