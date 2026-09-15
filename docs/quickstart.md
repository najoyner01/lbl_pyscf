# GPU4PySCF quickstart — clone to first calculation (NERSC Perlmutter)

Verified 2026-09-10 on a Perlmutter login node: A100-PCIE-40GB (`sm_80`),
driver CUDA 13.2, `cudatoolkit/13.2`, NERSC `python/3.13`, cmake 3.28.
Total time: a few minutes of typing plus one long `cmake --build`.

gpu4pyscf is a **plugin** for PySCF, not a fork: you install PySCF from pip and
add gpu4pyscf's compiled CUDA libraries on top. There is a prebuilt wheel
(`pip install gpu4pyscf-cuda13x`), but for development you build from source and
import the checkout via `PYTHONPATH` — that is what this guide does.

---

## 1. Modules

```sh
module load python/3.13 cudatoolkit    # cudatoolkit -> 13.2; cmake is already in /usr/bin
```

`cudatoolkit` is what makes this work without conda: it puts both the toolkit
and NVIDIA's math libs (`libcublas.so.13`, `libcusolver.so.13`, …) on
`LD_LIBRARY_PATH`, which is where the CuPy wheel finds them. It also sets
`CUDA_HOME`, which cmake needs.

## 2. Clone

```sh
cd $HOME                                    # or /pscratch/sd/<i>/<user> if $HOME quota is tight
git clone https://github.com/najoyner01/lbl_pyscf.git
cd lbl_pyscf
git remote add upstream https://github.com/pyscf/gpu4pyscf.git   # optional, for rebases
export REPO=$PWD
```

(For plain upstream work, clone `https://github.com/pyscf/gpu4pyscf.git` instead —
identical from here on.)

## 3. Python environment

```sh
python3 -m venv $REPO/venv
source $REPO/venv/bin/activate

pip install --upgrade pip
pip install --no-cache-dir \
  pyscf==2.14.0 \
  cupy-cuda13x cutensor-cu13 \
  gpu4pyscf-libxc-cuda12x==0.8.1 \
  pyscf-dispersion==1.5.0 geometric==1.1.0 basis-set-exchange==0.11 \
  pytest
```

Notes:

- `gpu4pyscf-libxc-cuda12x` is correct despite the `cuda12x` name — it ships a
  host-side `libxc.so` into `site-packages/gpu4pyscf/lib/deps/lib/`, which is
  where `gpu4pyscf/dft/libxc.py` looks for it. This is why you build with
  `-DBUILD_LIBXC=OFF` below (libxc takes ~20 min to compile otherwise).
- `requirements.txt` pins the **CUDA 12** stack (`cupy-cuda12x==13.4.1` +
  `cutensor-cu12==2.2.0`) — see the cuTENSOR note in §7 before choosing.

## 4. Build the CUDA extensions (login node, no GPU needed)

```sh
cd $REPO
cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf -DCUDA_ARCHITECTURES=80-real -DBUILD_LIBXC=OFF
cmake --build build/temp.gpu4pyscf -j 16
```

- `80-real` = A100 only. Building the default `70;80;90` triples the time for
  nothing on Perlmutter.
- This is the slow step (tens of minutes — the integral kernels are heavily
  templated). It is also the only step you ever repeat: **pure-Python edits need
  no rebuild**, only changes under `gpu4pyscf/lib/**/*.cu|*.h` do.
- Result: `gpu4pyscf/lib/lib{gint,gvhf,gvhf_rys,gvhf_md,gdft,gecp,pbc,solvent,cupy_helper,…}.so`

## 5. Per-session environment

Write it once and source it forever:

```sh
cat > $REPO/env.sh <<'EOF'
# usage: source env.sh   (from anywhere)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module load python/3.13 cudatoolkit
source "$REPO/venv/bin/activate"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUPY_ACCELERATORS=cub,cutensor
export OMP_NUM_THREADS=8          # PySCF's CPU-side parts; don't oversubscribe
EOF
```

`PYTHONPATH` is what makes `import gpu4pyscf` resolve to your checkout (no
`pip install -e`), so edits are live.

## 6. Verify

```sh
source $REPO/env.sh
python -c "import cupy; d=cupy.cuda.Device(0); print('cc', d.compute_capability); print((cupy.arange(10.)*2).sum())"
python -c "import pyscf, gpu4pyscf; print(pyscf.__version__, gpu4pyscf.__version__, gpu4pyscf.__file__)"
```

Expect `cc 80`, `90.0` (this forces a real NVRTC compile), and a `gpu4pyscf`
path pointing **inside your checkout**.

## 7. First calculation

```python
# first_calc.py — water, B3LYP/def2-tzvpp, density fitting
import time, pyscf
from gpu4pyscf.dft import rks

mol = pyscf.M(atom='''O  0.000  0.000  0.117
                      H -0.757  0.000 -0.469
                      H  0.757  0.000 -0.469''',
              basis='def2-tzvpp', verbose=4)

mf = rks.RKS(mol, xc='b3lyp').density_fit()
mf.grids.atom_grid = (99, 590)

t0 = time.perf_counter()
e = mf.kernel()
print(f"E = {e:.9f} Ha   [{time.perf_counter()-t0:.2f} s]")

g = mf.Gradients().kernel()
print("max |grad| =", abs(g).max())
```

```sh
python first_calc.py
```

Reference result on an A100 (measured):

```
E = -76.466676721 Ha   [4.9 s]     # most of that is one-time JIT; reruns are faster
max |grad| = 0.00342135387
```

The equivalent starting from a CPU PySCF object is `mf = pyscf.dft.RKS(mol,
xc='b3lyp').density_fit().to_gpu()`; `to_cpu()` goes back. More patterns in
`examples/`.

**Expect this warning with the CUDA 13 stack:**

```
gpu4pyscf/lib/cutensor.py:154: UserWarning: using cupy as the tensor contraction engine.
```

It is not a misconfiguration on your side: the `cupy-cuda13x` 13.6.0 wheel does
not ship the `cupy_backends.cuda.libs.cutensor` binding at all, so `contract()`
falls back to `cupy.einsum`. Fine for SCF/DF-SCF; slower for contraction-heavy
paths (CC, TDDFT, Hessians). To get real cuTENSOR you need the pinned CUDA 12
stack — `module load cudatoolkit/12.4`, `pip install -r requirements.txt`
(`cupy-cuda12x==13.4.1` + `cutensor-cu12==2.2.0`, whose wheel *does* contain the
binding), and a **full rebuild** so the `.so` files link against CUDA 12. That
combination is upstream's tested one but is not verified in this guide.

## 8. Running for real

Login nodes have a single shared A100 — OK for the smoke test above, not for
anything timed or long. Interactive:

```sh
salloc -A <project>_g -C gpu -q interactive -t 60 -N 1 -G 1
source $REPO/env.sh
pytest gpu4pyscf/gto/tests/test_ecp.py -v
```

Batch (see `benchmarks/gto/run_ecp_bench.sbatch` for a real one):

```sh
#!/bin/bash
#SBATCH -A <project>_g -C gpu -q regular -N 1 -n 1 -c 32 -G 1 -t 00:30:00
#SBATCH -o logs/%x_%j.out -e logs/%x_%j.err
set -euo pipefail
source /path/to/lbl_pyscf/env.sh
srun -n 1 --gpus-per-task=1 python first_calc.py
```

Add `--exclusive` instead of a shared queue whenever you are measuring timings.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ImportError: libcublas.so.13: cannot open shared object file` | `module load cudatoolkit` missing in that shell/job script |
| `cannot open source file "vector_types.h"` from an NVRTC compile | `CUDA_HOME`/`CUDA_PATH` points at a toolkit without headers; `rm -rf ~/.cupy/kernel_cache` after fixing |
| `import gpu4pyscf` resolves to site-packages, not your checkout | `PYTHONPATH` not set, or a stray `pip install gpu4pyscf-cuda13x` in the venv — uninstall it |
| `libgint.so` not found | build step never ran, or you cloned fresh over an old `build/` — `rm -rf build && cmake …` |
| `using cupy as the tensor contraction engine` | expected on CUDA 13; see §7 |
| OOM on a 40 GB A100 | drop `.density_fit()` for direct SCF, or coarsen `mf.grids.atom_grid` |

Related: `docs/perlmutter-setup.md` documents an alternative conda-based setup
(conda-forge `cuda-toolkit` instead of the `cudatoolkit` module) — use it only if
you need the CUDA 12 / cuTENSOR path and the module route gives you trouble.
`CLAUDE.md` has the porting conventions and the GPU primitive cheat-sheet.
