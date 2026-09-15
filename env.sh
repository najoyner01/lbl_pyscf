# Source me: `source env.sh` (from any directory).
# Sets up the gpu4pyscf runtime: NERSC python + CUDA toolkit modules, the repo's
# venv, and the PYTHONPATH that makes `import gpu4pyscf` resolve to this checkout.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# lmod is not initialized in non-interactive shells (e.g. `bash -c`, some CI)
if ! type module >/dev/null 2>&1; then
    . /etc/profile.d/zzz-lmod.sh
fi

module load python/3.13 cudatoolkit    # cudatoolkit -> 13.2 (cuBLAS/cuSOLVER for cupy, nvcc for cmake)
source "$REPO/venv/bin/activate"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUPY_ACCELERATORS=cub,cutensor
export OMP_NUM_THREADS=8               # PySCF's CPU-side work; don't oversubscribe a shared node
