#!/bin/bash -l
#SBATCH -A nstaff
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -G 1
#SBATCH -t 00:30:00
#SBATCH -J prof_tm
#SBATCH -o prof_%j.out
#SBATCH -e prof_%j.err
set -uo pipefail
source /global/u1/n/najoyner/lbl_pyscf/env.sh
export OMP_NUM_THREADS=16
cd /global/u1/n/najoyner/lbl_pyscf/benchmarks/inorganic_benchmark/profiling
nvidia-smi -L
srun -n 1 --gpus-per-task=1 python prof_components.py
