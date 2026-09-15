#!/bin/bash -l
#SBATCH -A m4884_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -G 1
#SBATCH --exclusive
#SBATCH -t 03:00:00
#SBATCH -J ecp_ptable
#SBATCH -o logs/ecp_ptable_%j.out
#SBATCH -e logs/ecp_ptable_%j.err

# ECP single-point benchmark over the def2-ECP elements, GPU vs CPU.
#
#   mkdir -p logs && sbatch benchmarks/ecp_ptable/run_bench.sbatch
#
# --exclusive so no co-tenant job perturbs the CPU timings; the CPU side is
# still pinned to 16 threads (see below) because that is the comparison point,
# not because that is all the node has.
#
# Resubmitting is safe: --resume skips (system, functional, device) triples
# already present in results.jsonl, so a job that hits the wall clock can be
# continued rather than restarted.

set -euo pipefail

REPO=/global/u1/n/najoyner/lbl_pyscf
source "$REPO/env.sh"
cd "$REPO"

# env.sh sets OMP_NUM_THREADS=8; the benchmark's comparison point is 16 threads.
export OMP_NUM_THREADS=16

nvidia-smi -L
python -c "import cupy; print('cupy', cupy.__version__, 'cc', cupy.cuda.Device(0).compute_capability)"

srun -n 1 --gpus-per-task=1 python benchmarks/ecp_ptable/run_bench.py \
    --basis def2-tzvp \
    --cpu-threads 16 \
    --grid 75 302 \
    --conv-tol 1e-9 \
    --resume

python benchmarks/ecp_ptable/make_report.py
