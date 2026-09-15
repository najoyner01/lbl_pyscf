#!/bin/bash -l
#SBATCH -A m4884_g
#SBATCH -C gpu
#SBATCH -q shared              # single GPU, no need for a whole node; use `regular --exclusive` for timings
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -G 1
#SBATCH -t 00:15:00
#SBATCH -J i2_ecp
#SBATCH -o logs/i2_ecp_%j.out
#SBATCH -e logs/i2_ecp_%j.err

# I2 / PBE0 / aug-cc-pVDZ-PP + ECP28MDF. See examples/ecp_i2_dft.py.
# Submit from the repo root:  mkdir -p logs && sbatch examples/run_i2_ecp.sbatch

set -euo pipefail

REPO=/global/u1/n/najoyner/lbl_pyscf
source "$REPO/env.sh"
cd "$REPO"

nvidia-smi -L
python -c "import cupy; print('cupy', cupy.__version__, 'cc', cupy.cuda.Device(0).compute_capability)"

srun -n 1 --gpus-per-task=1 python examples/ecp_i2_dft.py --cpu
