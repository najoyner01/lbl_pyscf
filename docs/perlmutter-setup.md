# Building & testing gpu4pyscf on NERSC Perlmutter

Validated 2026-09 on a Perlmutter GPU node (A100-PCIE-40GB, `sm_80`, CUDA driver
13.x). Compile on a login node; run tests inside a `salloc -C gpu` job.

`requirements.txt` is unchanged — these are conda/environment steps that sit
*under* the pinned pip packages, not new dependencies.

## Why the extra steps

`cupy-cuda12x` does **not** bundle the CUDA toolkit. On Perlmutter, NERSC's
`cudatoolkit` module keeps the CUDA math libraries (cuBLAS/cuSOLVER/…) in a
directory separate from the core toolkit, so cupy can't find `libcublasLt.so.12`
/ `libcusolver.so.11`. Simplest fix: get the whole toolkit from conda-forge into
the same env.

conda-forge then lays CUDA out under `$CONDA_PREFIX/targets/x86_64-linux/`
(not `$CONDA_PREFIX/include` / `$CONDA_PREFIX/lib`), and cupy's NVRTC does
`-I $CUDA_PATH/include` — so `CUDA_PATH` must point at the `targets` dir or every
cupy JIT kernel fails with `cannot open source file "vector_types.h"`.

## One-time setup (login node)

```sh
module load conda cmake            # cmake >= 3.19 required

conda create -n gpu4pyscf python=3.11 -y
conda activate gpu4pyscf

cd <repo>
pip install -r requirements.txt pytest
conda install -c conda-forge cuda-toolkit=12.4      # nvcc + headers + libs + nvrtc

# sanity: this path must exist
ls $CONDA_PREFIX/targets/x86_64-linux/include/vector_types.h
```

`cuda-toolkit=12.4` only needs to match `cupy-cuda12x`'s CUDA **major** version
(12); it does not need to match the system driver.

## Environment (conda activate / deactivate hooks)

conda sources every file in `$CONDA_PREFIX/etc/conda/activate.d/` on
`conda activate` and every file in `.../deactivate.d/` on `conda deactivate`.
Put the gpu4pyscf environment in an `activate.d` script, and a matching
`deactivate.d` script that restores the prior values so the settings don't leak
into other environments.

What each variable is for:

| Variable | Why |
|----------|-----|
| `CUDA_HOME` | points cmake / build tooling at the conda toolkit |
| `CUDA_PATH` | cupy's NVRTC does `-I $CUDA_PATH/include`; must be the `targets/x86_64-linux` dir (see above) |
| `LD_LIBRARY_PATH` | so `libcutensor` / gpu4pyscf's `.so` files find `libcublasLt` etc. in the conda `targets` + main `lib` dirs |
| `PYTHONPATH` | import `gpu4pyscf` from the source checkout (no `pip install -e`) |
| `CUPY_ACCELERATORS` | enable the cuB / cuTENSOR contraction backends |

Edit `REPO` below to your checkout path, then:

```sh
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/gpu4pyscf-env.sh" <<'EOF'
#!/bin/sh
REPO=/pscratch/sd/n/namehta4/Quantum/lbl_pyscf

# stash prior values so deactivate can restore them
export _GPU4PYSCF_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export _GPU4PYSCF_OLD_PYTHONPATH="${PYTHONPATH:-}"

export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX/targets/x86_64-linux"
export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUPY_ACCELERATORS="cub,cutensor"
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/gpu4pyscf-env.sh" <<'EOF'
#!/bin/sh
export LD_LIBRARY_PATH="$_GPU4PYSCF_OLD_LD_LIBRARY_PATH"
export PYTHONPATH="$_GPU4PYSCF_OLD_PYTHONPATH"
[ -z "$LD_LIBRARY_PATH" ] && unset LD_LIBRARY_PATH
[ -z "$PYTHONPATH" ] && unset PYTHONPATH
unset _GPU4PYSCF_OLD_LD_LIBRARY_PATH _GPU4PYSCF_OLD_PYTHONPATH
unset CUDA_HOME CUDA_PATH CUPY_ACCELERATORS
EOF
```

The scripts are `source`d, not executed — no `chmod` needed. They take effect on
the next activation:

```sh
conda deactivate && conda activate gpu4pyscf
echo "$CUDA_PATH"          # .../targets/x86_64-linux
echo "$LD_LIBRARY_PATH"    # starts with the two conda dirs, no leading ':'
python -c "import sys; print('REPO on path:', any('lbl_pyscf' in p for p in sys.path))"
```

In a batch job or fresh shell, `module load conda && conda activate gpu4pyscf` is
enough — the hooks run automatically, nothing extra to `source`.

## Build the CUDA extensions (login node — no GPU needed)

```sh
cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf -DCUDA_ARCHITECTURES=80-real -DBUILD_LIBXC=OFF
cmake --build build/temp.gpu4pyscf -j 16
```

`80-real` = A100 only (faster than the default `70;80;90`). `-DBUILD_LIBXC=OFF`
because the `gpu4pyscf-libxc-cuda12x` wheel from `requirements.txt` provides
`libxc` at runtime.

## Verify

```sh
python -c "import cupy; x=cupy.arange(10.); print((x+x).sum())"   # -> 90.0 (forces an NVRTC compile)
python -c "import cupy; cupy.show_config()" 2>&1 | grep -i warn   # expect no cutensor preload warning
python -c "import cutensor; print('cutensor', cutensor.__version__)"  # expect 2.2.0 (matches requirements.txt)
```

If cupy JIT still can't find headers, `rm -rf ~/.cupy/kernel_cache` and re-check
`echo $CUDA_PATH`.

## Run tests / benchmarks (GPU node)

```sh
salloc -A <project>_g -C gpu -q interactive -t 60 -N 1 -G 1
module load conda ; conda activate gpu4pyscf
cd <repo>

pytest gpu4pyscf/gpu4pyscf/gto/tests/test_ecp.py gpu4pyscf/gpu4pyscf/gto/tests/test_ecp_screening.py -v
python benchmarks/gto/benchmark_ecp_screening.py
```

## Notes

- Pure-Python edits need no rebuild; only re-run `cmake --build` after touching
  `lib/**/*.cu` / `*.h`.
- Keep the conda env off `$HOME` if quota is tight — `/global/common/software/<project>/`
  is NERSC's recommended spot for many-small-file software trees.
