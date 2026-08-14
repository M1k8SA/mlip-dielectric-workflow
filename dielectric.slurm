#!/bin/bash
#SBATCH --job-name=dielectric
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Add cluster-specific directives when needed, for example:
#   #SBATCH --partition=compute
#   #SBATCH --gres=gpu:1
# Submit one case:
#   MODEL=/abs/path/model.pth sbatch dielectric.slurm --relax-fixed-cell
# Submit an array (CASE_LIST contains one case directory per non-comment line):
#   MODEL=/abs/path/model.pth CASE_LIST=/abs/path/cases.txt \
#     sbatch --array=0-99%10 dielectric.slurm --relax-fixed-cell

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
submit_dir="${SLURM_SUBMIT_DIR:-$PWD}"
submit_dir="$(cd -- "$submit_dir" && pwd -P)"

absolute_from_submit() {
    local value="$1"
    if [[ "$value" = /* ]]; then
        printf '%s\n' "$value"
    else
        printf '%s/%s\n' "$submit_dir" "$value"
    fi
}

# A cluster environment may be selected explicitly, e.g.
#   PYTHON=/path/to/deepmd-env/bin/python sbatch ...
python_setting="${PYTHON:-$script_dir/.venv/bin/python}"
if [[ "$python_setting" == */* ]]; then
    python_exe="$(absolute_from_submit "$python_setting")"
    # The project-local default is relative to script_dir, not submit_dir.
    if [[ -z "${PYTHON:-}" ]]; then
        python_exe="$script_dir/.venv/bin/python"
    fi
else
    python_exe="$(command -v -- "$python_setting" || true)"
fi
if [[ -z "$python_exe" || ! -x "$python_exe" ]]; then
    echo "ERROR: Python environment not found or not executable: $python_setting" >&2
    echo "Set PYTHON=/path/to/deepmd-env/bin/python when submitting." >&2
    exit 1
fi
export PATH="$(dirname -- "$python_exe"):$PATH"

# Without an array, calculate CASE_DIR (default: the sbatch submission directory).
# With an array, array index N selects line N after blank/comment lines are removed.
case_dir_setting="${CASE_DIR:-$submit_dir}"
if [[ "$case_dir_setting" != /* ]]; then
    case_dir_setting="$submit_dir/$case_dir_setting"
fi
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if [[ -z "${CASE_LIST:-}" ]]; then
        echo "ERROR: CASE_LIST is required for a Slurm array job." >&2
        exit 1
    fi
    case_list="$(absolute_from_submit "$CASE_LIST")"
    if [[ ! -f "$case_list" ]]; then
        echo "ERROR: CASE_LIST not found: $case_list" >&2
        exit 1
    fi
    mapfile -t cases < <(
        sed 's/\r$//' "$case_list" | grep -vE '^[[:space:]]*(#|$)'
    )
    array_index="$SLURM_ARRAY_TASK_ID"
    if [[ ! "$array_index" =~ ^[0-9]+$ ]] || (( array_index >= ${#cases[@]} )); then
        echo "ERROR: array index $array_index is outside CASE_LIST (0..$((${#cases[@]} - 1)))." >&2
        exit 1
    fi
    case_entry="${cases[$array_index]}"
    if [[ "$case_entry" = /* ]]; then
        case_dir_setting="$case_entry"
    else
        case_dir_setting="$(dirname -- "$case_list")/$case_entry"
    fi
fi

if [[ ! -d "$case_dir_setting" ]]; then
    echo "ERROR: calculation directory not found: $case_dir_setting" >&2
    exit 1
fi
case_dir="$(cd -- "$case_dir_setting" && pwd -P)"
cd "$case_dir"

input_file="${INPUT:-pw.in}"
born_file="${BORN_FILE:-BORN}"
work_dir="${WORK_DIR:-dielectric_work}"
cores="${CORES:-${SLURM_CPUS_PER_TASK:-1}}"
if [[ ! "$cores" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: CORES must be a positive integer: $cores" >&2
    exit 1
fi

# Keep CPU thread counts consistent with the Slurm allocation. A CUDA-enabled
# DeepMD environment may use a requested GPU while these CPUs handle postprocess.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$OMP_NUM_THREADS}"
export DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS:-$OMP_NUM_THREADS}"
export DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS:-1}"

workflow_args=(
    --input "$input_file"
    --born "$born_file"
    --work-dir "$work_dir"
    --cores "$cores"
)

if [[ -z "${MODEL:-}" && -z "${FORCE_MODEL:-}" ]]; then
    echo "ERROR: set MODEL, or at least FORCE_MODEL when relaxation is skipped/reused." >&2
    exit 1
fi

for model_var in MODEL RELAX_MODEL FORCE_MODEL; do
    model_value="${!model_var:-}"
    if [[ -n "$model_value" ]]; then
        model_path="$(absolute_from_submit "$model_value")"
        if [[ ! -s "$model_path" ]]; then
            echo "ERROR: $model_var file not found or empty: $model_path" >&2
            exit 1
        fi
        case "$model_var" in
            MODEL) workflow_args+=(--model "$model_path") ;;
            RELAX_MODEL) workflow_args+=(--relax-model "$model_path") ;;
            FORCE_MODEL) workflow_args+=(--force-model "$model_path") ;;
        esac
    fi
done

if [[ -n "${EPSIL_SCRIPT:-}" ]]; then
    epsil_path="$(absolute_from_submit "$EPSIL_SCRIPT")"
    if [[ ! -s "$epsil_path" ]]; then
        echo "ERROR: EPSIL_SCRIPT not found or empty: $epsil_path" >&2
        exit 1
    fi
    workflow_args+=(--epsil-script "$epsil_path")
fi

echo "[slurm] job=${SLURM_JOB_ID:-local} array_index=${SLURM_ARRAY_TASK_ID:-none}"
echo "[slurm] host=$(hostname) case=$case_dir"
echo "[slurm] python=$python_exe cores=$cores"
echo "[slurm] start=$(date --iso-8601=seconds)"

exit_code=0
"$python_exe" "$script_dir/run_dielectric.py" "${workflow_args[@]}" "$@" || exit_code=$?

echo "[slurm] finish=$(date --iso-8601=seconds) exit_code=$exit_code"
if (( exit_code != 0 )); then
    echo "[slurm] workflow failed; inspect $case_dir/$work_dir and the Slurm error log." >&2
fi
exit "$exit_code"
