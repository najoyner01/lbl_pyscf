#!/bin/bash -l
#SBATCH -A nstaff
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -G 1
#SBATCH --exclusive
#SBATCH -t 10:00:00
#SBATCH -J inorg_tm_ladder
#SBATCH -o logs/inorg_tm_%j.out
#SBATCH -e logs/inorg_tm_%j.err

# Transition-metal basis-set ladder, GPU vs CPU.
#
#   mkdir -p logs && sbatch benchmarks/inorganic_benchmark/run_bench_tm.sbatch
#
# The 19 closed-shell 4d/5d transition-metal complexes of systems.py (Y-Cd,
# Hf-Hg), each run at def2-SVP, def2-TZVP and def2-QZVP (DZ/TZ/QZ) with PBE,
# B3LYP and PBE0: 19 x 3 x 3 = 171 (system, basis, functional) points, each on
# both devices -> 342 SCFs. See systems_tm.py for the table and for why there
# is no 5Z rung (gpu4pyscf's ECP kernels stop at g functions).
#
# Basis is the OUTER loop in run_bench.py, so a job that hits the wall clock
# leaves complete rungs behind rather than a ragged ladder. The QZ rung is the
# expensive one: 142-492 AO with g functions on the metal, and the CPU leg of
# the largest hexachlorides (492 AO, B3LYP/PBE0) runs several minutes each.
#
# --exclusive so no co-tenant job perturbs the CPU timings; the CPU side is
# still pinned to 16 threads (see below) because that is the comparison point,
# not because that is all the node has.
#
# Resubmitting is safe: --resume skips (system, functional, basis, device,
# method) tuples already present in results_tm.jsonl.
#
# Writes results_tm.jsonl / report_tm.{md,csv,png,pdf} / figures/tm_ladder*.
# It does not touch results.jsonl, results_neutral.jsonl or results_lanl.jsonl.

set -euo pipefail

REPO=/global/u1/n/najoyner/lbl_pyscf
source "$REPO/env.sh"
cd "$REPO"

# env.sh sets OMP_NUM_THREADS=8; the benchmark's comparison point is 16 threads.
export OMP_NUM_THREADS=16

nvidia-smi -L
python -c "import cupy; print('cupy', cupy.__version__, 'cc', cupy.cuda.Device(0).compute_capability)"

srun -n 1 --gpus-per-task=1 python benchmarks/inorganic_benchmark/run_bench.py \
    --systems tm \
    --bases def2-svp def2-tzvp def2-qzvp \
    --functionals pbe b3lyp pbe0 \
    --cpu-threads 16 \
    --grid 75 302 \
    --conv-tol 1e-9 \
    --resume

python benchmarks/inorganic_benchmark/make_report.py \
    --results benchmarks/inorganic_benchmark/results_tm.jsonl \
    --prefix report_tm
python benchmarks/inorganic_benchmark/make_figures.py --sets tm_ladder
