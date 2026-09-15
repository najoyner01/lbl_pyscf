#!/bin/bash
#SBATCH -A m4884_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -G 1
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH -J ecp_bench
#SBATCH -o logs/ecp_bench_%j.out
#SBATCH -e logs/ecp_bench_%j.err

# ECP screening benchmark, single A100. --exclusive (rather than -q shared)
# so no co-tenant job perturbs the timings.

set -euo pipefail

module load cudatoolkit

REPO=/global/u1/n/najoyner/lbl_pyscf
source "$REPO/venv/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export CUPY_ACCELERATORS=cub,cutensor
export OMP_NUM_THREADS=8          # matches lib.num_threads(8) in benchmark_ecp.py

# Both benchmark scripts resolve the molecule path relative to CWD, not to the
# script, so run from this directory.
cd "$REPO/benchmarks/gto"

nvidia-smi
python -c "import cupy; print('cupy', cupy.__version__, cupy.cuda.Device(0).compute_capability)"

srun -n 1 --gpus-per-task=1 python benchmark_ecp_screening.py --n 10 20 40 80 --ip
