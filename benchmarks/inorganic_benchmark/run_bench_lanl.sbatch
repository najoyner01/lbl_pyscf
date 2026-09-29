#!/bin/bash -l
#SBATCH -A m4884_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 128
#SBATCH -G 1
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH -J lanl_fblock
#SBATCH -o logs/lanl_fblock_%j.out
#SBATCH -e logs/lanl_fblock_%j.err

# LANL f-element benchmark, GPU vs CPU.
#
#   mkdir -p logs && sbatch benchmarks/inorganic_benchmark/run_bench_lanl.sbatch
#
# 39 lanthanide and actinide coordination complexes from lanl_benchmark.json at
# their experimental geometries -- 16 to 211 atoms, 570 to 3747 AO, 33 of them
# open shell, charges -2..+1. See systems_lanl.py for the table.
#
# BASIS. def2-TZVP throughout, the same as run_bench.sbatch, with exactly one
# substitution: the f-block metal takes the Stuttgart-Koeln RSC ECP and valence
# basis, because PySCF's def2 sets stop before the f block and cover neither
# the eight lanthanides nor the six actinides in this set. Iodine keeps its
# def2 ECP.
#
# DENSITY FITTING is on (the default for this set). Direct SCF at 3747 AO is
# not tractable on the CPU side. The def2 auxiliary sets also omit the
# actinides, so the aux basis is generated per molecule by make_auxbasis.
#
# CPU AO CAP. --max-nao-cpu 1800 runs the CPU leg for 21 of the 39 systems and
# records a skip for the rest, so the GPU still sweeps everything while the
# paired GPU/CPU comparison stays inside the wall clock. Raise the cap and
# resubmit to pair more of them -- a skip does not count as done, so --resume
# will pick those up rather than step over them.
#
# --exclusive so no co-tenant job perturbs the CPU timings; the CPU side is
# still pinned to 16 threads (see below) because that is the comparison point,
# not because that is all the node has.
#
# Longer wall clock than either def2 job: these are 100x the AO count of the
# 36-element sets, most are unrestricted, and 31 of the 39 rows are flagged in
# the source spreadsheet as hard to converge (those get a level shift, damping
# and a raised cycle cap, applied identically on both devices).
#
# Resubmitting is safe: --resume skips (system, functional, basis, device,
# method) tuples already recorded, so a job killed at the wall clock loses
# nothing. Writes results_lanl.jsonl / report_lanl.{md,csv,png,pdf} and does
# not touch results.jsonl or results_neutral.jsonl.
#
# READING THE RESULTS. The two devices build their own initial guesses, and on
# an open-shell f element that is often enough to land them in different SCF
# solutions. A large dE here usually means "different solution", not "broken
# GPU kernel" -- check `converged`, `cycles` and <S^2> before concluding
# otherwise.

set -euo pipefail

REPO=/global/u1/n/najoyner/lbl_pyscf
source "$REPO/env.sh"
cd "$REPO"

# env.sh sets OMP_NUM_THREADS=8; the benchmark's comparison point is 16 threads.
export OMP_NUM_THREADS=16

nvidia-smi -L
python -c "import cupy; print('cupy', cupy.__version__, 'cc', cupy.cuda.Device(0).compute_capability)"

srun -n 1 --gpus-per-task=1 python benchmarks/inorganic_benchmark/run_bench.py \
    --systems lanl \
    --basis def2-tzvp \
    --cpu-threads 16 \
    --grid 75 302 \
    --conv-tol 1e-8 \
    --max-nao-cpu 1800 \
    --resume

python benchmarks/inorganic_benchmark/make_report.py \
    --results benchmarks/inorganic_benchmark/results_lanl.jsonl \
    --prefix report_lanl
