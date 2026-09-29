#!/bin/bash -l
#SBATCH -A m4884_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -G 1
#SBATCH --exclusive
#SBATCH -t 05:00:00
#SBATCH -J inorg_neutral
#SBATCH -o logs/inorg_neutral_%j.out
#SBATCH -e logs/inorg_neutral_%j.err

# Neutral / open-shell variant of the inorganic ECP benchmark, GPU vs CPU.
#
#   mkdir -p logs && sbatch benchmarks/inorganic_benchmark/run_bench_neutral.sbatch
#
# Same 36 def2-ECP systems as run_bench.sbatch, but every species is neutral:
# the six anions of systems.py become open-shell neutrals (TcO4, ReO4, RhCl6,
# IrCl6, PdCl4, PtCl4) and run UKS instead of RKS. See systems_neutral.py for
# the spin assignments -- they are formal, not verified ground states.
#
# --exclusive so no co-tenant job perturbs the CPU timings; the CPU side is
# still pinned to 16 threads (see below) because that is the comparison point,
# not because that is all the node has.
#
# Longer wall clock than the closed-shell job: UKS doubles the Fock builds for
# the six open-shell systems and those are the largest molecules in the set.
#
# Resubmitting is safe: --resume skips (system, functional, basis, device,
# method) tuples already present in results_neutral.jsonl, so a job that hits
# the wall clock can be continued rather than restarted.
#
# Writes results_neutral.jsonl / report_neutral.{md,csv,png,pdf} -- it does not
# touch the closed-shell results.jsonl or report.*.

set -euo pipefail

REPO=/global/u1/n/najoyner/lbl_pyscf
source "$REPO/env.sh"
cd "$REPO"

# env.sh sets OMP_NUM_THREADS=8; the benchmark's comparison point is 16 threads.
export OMP_NUM_THREADS=16

nvidia-smi -L
python -c "import cupy; print('cupy', cupy.__version__, 'cc', cupy.cuda.Device(0).compute_capability)"

srun -n 1 --gpus-per-task=1 python benchmarks/inorganic_benchmark/run_bench.py \
    --systems neutral \
    --basis def2-tzvp \
    --cpu-threads 16 \
    --grid 75 302 \
    --conv-tol 1e-9 \
    --resume

python benchmarks/inorganic_benchmark/make_report.py \
    --results benchmarks/inorganic_benchmark/results_neutral.jsonl \
    --prefix report_neutral
