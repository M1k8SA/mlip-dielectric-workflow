#!/bin/bash
#SBATCH -p debug
#SBATCH -J BFGS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:30:00
#SBATCH -o BFGS.out
#SBATCH -e BFGS.err

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SLURM_SUBMIT_DIR:-$script_dir}"

# Override when submitting on a cluster with, e.g.:
#   PYTHON=/path/to/deepmd/bin/python sbatch BFGS.slurm
PYTHON="${PYTHON:-$script_dir/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment not found or not executable: $PYTHON" >&2
    exit 1
fi

# The workflow is model-agnostic: select the model explicitly when submitting,
# e.g. MODEL=/path/to/model.pth sbatch BFGS.slurm
MODEL="${MODEL:-}"
if [[ -z "$MODEL" ]]; then
    echo "MODEL is required. Submit with: MODEL=/path/to/model sbatch BFGS.slurm" >&2
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export DP_INTRA_OP_PARALLELISM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export DP_INTER_OP_PARALLELISM_THREADS=1

"$PYTHON" "$script_dir/relax_mlip.py" \
    --input pw.in \
    --model "$MODEL" \
    --output relaxed_pw.in \
    --xyz-output relaxed.xyz \
    --trajectory BFGS.traj \
    --logfile BFGS.log \
    --fmax 0.005 \
    --max-steps 500 \
    --optimizer bfgs \
    --timeout 1500

"$PYTHON" -c "import sys; print('Python:', sys.executable)"
