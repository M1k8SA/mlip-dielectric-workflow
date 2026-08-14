#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python_exe="$script_dir/.venv/bin/python"
if [[ ! -x "$python_exe" ]]; then
    echo "ERROR: project environment not found: $python_exe" >&2
    echo "Run: bash $script_dir/setup_environment.sh" >&2
    exit 1
fi

# Conservative CPU defaults; users can override any of these before launch.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS:-4}"
export DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS:-1}"
export PATH="$script_dir/.venv/bin:$PATH"

exec "$python_exe" "$script_dir/run_dielectric.py" "$@"
