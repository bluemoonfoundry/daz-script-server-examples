#!/usr/bin/env bash
# Runs pose_transfer_photo.py against the pinocchio-ik conda-forge
# environment, without requiring a manual `conda activate` step first.
#
# `--backend pinocchio` needs the real Pinocchio C++/Eigen library. Windows
# has no PyPI wheel at all (see environment.yml / pinocchio_ik.py's module
# docstring); conda-forge is the reliable cross-platform source, so this
# wrapper uses `conda run` the same way on macOS/Linux as run_pinocchio.ps1
# does on Windows, rather than assuming the caller has already activated
# the right environment (or has the right DLLs/shared libs on their
# loader path if invoked by raw interpreter path instead).
#
# All arguments after the script name are forwarded verbatim to
# pose_transfer_photo.py.
#
# Usage:
#   ./run_pinocchio.sh photo.jpg --backend pinocchio
#   ./run_pinocchio.sh --batch photos/ --backend pinocchio --save-poses --stats-csv
#
# If `conda` isn't already resolvable on PATH, set PINOCCHIO_IK_CONDA_ROOT
# to your miniforge/miniconda install root once (e.g. in your shell rc)
# rather than editing this script:
#   export PINOCCHIO_IK_CONDA_ROOT="$HOME/miniforge3"

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_conda() {
    if command -v conda >/dev/null 2>&1; then
        command -v conda
        return
    fi
    local root="${PINOCCHIO_IK_CONDA_ROOT:-$HOME/miniforge3}"
    local candidate="$root/condabin/conda"
    if [ -x "$candidate" ]; then
        echo "$candidate"
        return
    fi
    echo "Could not find a conda executable. Either put conda on PATH" >&2
    echo "(e.g. source your miniforge3/etc/profile.d/conda.sh), or set" >&2
    echo "PINOCCHIO_IK_CONDA_ROOT to your install root. Tried: $candidate" >&2
    exit 1
}

conda_exe="$(resolve_conda)"

if ! "$conda_exe" env list | grep -q "pinocchio-ik"; then
    echo "The 'pinocchio-ik' conda environment doesn't exist yet. Create it" >&2
    echo "once with:" >&2
    echo "" >&2
    echo "    conda env create -f environment.yml" >&2
    echo "" >&2
    echo "(run from this directory: $script_dir)" >&2
    exit 1
fi

exec "$conda_exe" run -n pinocchio-ik --no-capture-output --cwd "$script_dir" \
    python "$script_dir/pose_transfer_photo.py" "$@"
