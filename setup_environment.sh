#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tools_dir="$script_dir/.tools"
venv_dir="$script_dir/.venv"
cache_dir="$script_dir/.uv-cache"
uv="$tools_dir/uv"

uv_version="0.11.32"
python_exe="${PYTHON_EXE:-python3}"

if [[ ! -x "$uv" ]]; then
    mkdir -p "$tools_dir"
    installer="$(mktemp)"
    trap 'rm -f "$installer"' EXIT
    curl -LsSf "https://astral.sh/uv/${uv_version}/install.sh" -o "$installer"
    UV_UNMANAGED_INSTALL="$tools_dir" sh "$installer"
fi

export UV_CACHE_DIR="$cache_dir"
export UV_LINK_MODE=copy

if [[ ! -x "$venv_dir/bin/python" ]]; then
    "$uv" venv --python "$python_exe" "$venv_dir"
fi

# Install a CPU-only PyTorch runtime for DeepMD models using the PyTorch
# backend. For CUDA or another DeepMD backend, use a compatible custom env.
"$uv" pip install \
    --python "$venv_dir/bin/python" \
    "torch==2.10.0" \
    --index-url https://download.pytorch.org/whl/cpu

"$uv" pip install \
    --python "$venv_dir/bin/python" \
    "deepmd-kit==3.1.3" \
    "phonopy==4.4.0" \
    "ase==3.29.0" \
    "numpy==2.2.6" \
    "mpich==5.0.1.post1"

"$venv_dir/bin/python" -c \
    "import ase, deepmd, numpy, phonopy, torch; print('Environment OK:', deepmd.__version__, phonopy.__version__, torch.__version__)"

echo "Environment created: $venv_dir"
echo "Run workflow: bash $script_dir/run_dielectric.sh"
