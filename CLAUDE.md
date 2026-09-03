# CLAUDE.md — PySCF → GPU porting project

## Purpose

Port as much of PySCF's functionality to GPU as practical, inside the
**`gpu4pyscf`** plugin. Immediate targets, in order:

1. **ECP** (effective core potentials) — finish the remaining gaps.
2. **Coupled cluster** (CCSD → CCSD(T) → Λ/RDM/gradients → UCCSD → EOM-CC).

See `docs/STRATEGY.md` for the phased roadmap and rationale.

## Repos in this workspace

| Path | Role | Notes |
|------|------|-------|
| `pyscf/` | **Reference CPU implementation.** Read-only. | The source of truth for algorithms, APIs, and numerical results to validate against. Never edit. |
| `gpu4pyscf/` | **The target.** All new code goes here. | Python package `gpu4pyscf/gpu4pyscf/`, CUDA in `gpu4pyscf/gpu4pyscf/lib/`. |

`gpu4pyscf` depends on `pyscf>=2.8` and imports from it heavily (mol parsing,
basis data, class scaffolding, small CPU kernels). It is a *plugin*, not a fork.

## Build & test

```sh
# Build CUDA extensions (needs CUDA toolkit + nvcc; sm_70 minimum, CUDA 11/12/13)
cd gpu4pyscf
cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf
cmake --build build/temp.gpu4pyscf -j 8
export PYTHONPATH="$PWD:$PYTHONPATH"

# Runtime deps
pip3 install -r requirements.txt   # pinned cupy + cutensor combo — do not mix arbitrary versions

# Tests (pytest / unittest); each new module gets tests under <module>/tests/
pytest gpu4pyscf/gpu4pyscf/cc/tests -v
pytest gpu4pyscf/gpu4pyscf/gto/tests/test_ecp.py -v

# Lint — CI runs both, keep clean
ruff check --config gpu4pyscf/.ruff.toml --unsafe-fixes gpu4pyscf/gpu4pyscf
flake8 --config gpu4pyscf/.flake8 gpu4pyscf/gpu4pyscf
```

A GPU is required to run anything. If no GPU/toolkit is present in the dev
environment, you can still write code and reason against the CPU reference, but
say so — do not claim tests pass when they were not run.

## Porting conventions (from `gpu4pyscf/CONTRIBUTING.md` + observed practice)

- **Performance is the point.** API divergence from PySCF is acceptable when it
  buys speed. Correctness vs. the CPU reference is not negotiable.
- **CuPy arrays are the default container.** Functions must tolerate mixed
  cupy/numpy inputs (PySCF hands you numpy). Convert explicitly with
  `cupy.asarray` / `.get()`. Many numpy ufuncs reject mixed operands.
- **Mirror PySCF names & signatures** where a counterpart exists
  (`CCSD`, `update_amps`, `ao2mo`, `kernel`, `energy`, `nocc`, `nmo`, ...).
- **Class inheritance from PySCF is optional.** If you do inherit, disable
  unsupported methods explicitly: assign `NotImplemented` or `None`, or raise
  `NotImplementedError`. See `cc/ccsd_incore.py:CCSDBase` for the pattern.
- **Provide `to_cpu` / `to_gpu`.** Use `gpu4pyscf.lib.utils.to_cpu` /
  `to_gpu`; `_patch_pyscf.py` grafts `to_gpu` onto the PySCF classes. Round-trip
  must be tested.
- Use `gpu4pyscf.lib.logger` (same API as `pyscf.lib.logger`) and derive
  classes from `pyscf.lib.StreamObject`.
- Keep `_keys` sets accurate — `to_cpu` uses them to decide what to copy back.

## GPU primitive cheat-sheet (`gpu4pyscf.lib.cupy_helper`)

| Need | Use |
|------|-----|
| Tensor contraction / einsum | `contract('ijab,jb->ia', A, B)` — cuTENSOR-backed, the workhorse for CC |
| Attach metadata to an array | `tag_array(arr, foo=...)` → `CPArrayWithTag` |
| Triangular pack/unpack | `pack_tril`, `unpack_tril`, `unpack_4fold` (in pyscf) |
| Cartesian ↔ spherical | `cart2sph`, `block_c2s_diag` |
| Symmetrize | `transpose_sum`, `hermi_triu` |
| Fancy indexing (2D) | `take_last2d`, `takebak` |
| Available device memory | `get_avail_mem()` |
| Batched GEMM | `grouped_dot`, `grouped_gemm` |
| Eigensolver | `eigh` (cuSolver); also `gpu4pyscf.lib.cusolver` |
| DIIS | `gpu4pyscf.lib.diis.DIIS` |
| Multi-GPU | `gpu4pyscf.lib.multi_gpu`, `reduce_to_device`, `broadcast_to_devices` |

Einsum engine can be swapped (see `gpu4pyscf/examples/13-einsum_engine.py`).
cuTENSOR is strongly recommended; a warning at import means it is misconfigured.

## Calling a custom CUDA kernel from Python (ctypes pattern)

```python
from gpu4pyscf.lib.cupy_helper import load_library
libfoo = load_library('libfoo')                 # gpu4pyscf/lib/foo/ -> libfoo.so
libfoo.FOO_kernel.argtypes = [ctypes.c_void_p, ctypes.c_int, ...]

err = libfoo.FOO_kernel(out.data.ptr, n, arr.data.ptr, ...)  # cupy array -> .data.ptr
if err != 0:
    raise RuntimeError('FOO CUDA kernel failed.')
```

- One CMake target per lib dir: `gpu4pyscf/gpu4pyscf/lib/<mod>/CMakeLists.txt`,
  `LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}`, add it to
  `gpu4pyscf/gpu4pyscf/lib/CMakeLists.txt`.
- Kernels return an `int` error code; caller raises on non-zero.
- sm_70 minimum. Do **not** use sm_80-only features unconditionally. Must
  compile under CUDA 11 and 12. Watch the shared-memory cap (`__config__.shm_size`).
- Integral code works in the **sorted / grouped-basis** frame: `group_basis(mol)`
  returns `(sorted_mol, coeff, uniq_l_ctr, l_ctr_counts)`; compute in cartesian
  (`ao_loc_nr(cart=True)`) then transform back with `coeff.T @ M @ coeff`.

## Validation requirements (every ported feature)

1. **Value test vs. PySCF CPU** to ~8–10 significant digits on a small system
   (energies to 1e-8; amplitudes/matrix elements to 1e-6..1e-9).
2. **`to_cpu` / `to_gpu` round-trip** produces the same result.
3. **Mixed numpy/cupy input** handled (objects built by PySCF flow in).
4. Add a benchmark under `gpu4pyscf/benchmarks/` for anything perf-relevant;
   store baselines in `gpu4pyscf/gpu4pyscf/tests/benchmark_results/`.

## ECP — status & remaining work

**Already implemented** (`gpu4pyscf/gpu4pyscf/gto/ecp.py` + `lib/ecp/*.cu`, built
as `libgecp`):

- Scalar ECP integrals `get_ecp(mol)` — type 1 (local `U_L`) + type 2 (semilocal).
- 1st derivatives `get_ecp_ip` (`ip`) → analytic **gradients** (`grad/rhf.py`).
- 2nd derivatives `get_ecp_ipip` (`ipipv`, `ipvip`) → analytic **Hessian**
  (`hessian/rhf.py`).
- Wired into `scf/hf.py` `get_hcore`; tests in `gto/`, `scf/`, `dft/`, `df/`
  `tests/`; benchmark in `benchmarks/gto/benchmark_ecp.py`.
- Angular-momentum-templated kernels with a general fallback; cart→sph by
  transform; Bessel + Gauss–Chebyshev radial quadrature.

**Gaps to close (this project):**

| Gap | Where | Notes |
|-----|-------|-------|
| **Spin-orbit ECP (SO-ECP)** | CPU: `pyscf/gto/ecp.py:so_by_shell`, `ECPso_spinor` | GPU code explicitly drops SO terms (`sort_ecp_basis` filters `SO_TYPE_OF != 0`). Needed for X2C/SOC, spinor GHF. New kernel `ecp_so.cu` + spinor assembly. |
| ~~**Screening**~~ | `gto/ecp.py` | DONE on branch `ecp-screening` (A100-validated, screened==unscreened to 1e-13). Two-level `check_3c_overlap`-style screen; toggle `ecp.SCREEN_ECP`. Benchmark (`benchmarks/gto/ecp_screening.md`): 14x/55x/182x on Cu-chain N=20/40/80. Pending: merge. |
| **ECP-atom slicing in grad/hess** | `grad/rhf.py:344`, `gto/ecp.py` | `# TODO: slice ecp_atoms`; currently over-computes. |
| **PBC ECP** | `pyscf/pbc/gto/ecp.py`, `pyscf/lib/pbc/nr_ecp.c` | No GPU path. Lower priority. |
| **Validation breadth** | — | Sweep basis L up to g, all `crenbl`/`def2`/`ccECP` sets, f-in-core; compare to PySCF to 1e-10; benchmark vs CPU across system sizes. |
| Blackwell / CUDA<13.1 nvcc bug | `lib/ecp/CMakeLists.txt` | Known miscompile; currently worked around by disabling opt. Track a real fix. |

## Coupled cluster — status & remaining work

**Already implemented** (`gpu4pyscf/gpu4pyscf/cc/ccsd_incore.py`):

- `CCSD` for **RHF, closed shell, incore only** — full `(pq|rs)` AO→MO transform
  held in GPU memory (`_make_eris_incore`, needs room for ~2× t2).
- GPU `update_amps` (cupy + `contract`), DIIS, `_direct_ovvv_vvvv` (direct AO
  contraction of the O(nvir⁴) ladder term via `libgint`), `to_cpu`/`to_gpu`.
- Test: single H₂O/cc-pVDZ energy + amplitudes vs PySCF.

**Everything else is `NotImplemented`:** `solve_lambda`, `ccsd_t`, `make_rdm1/2`,
`nuc_grad_method`, `density_fit`, all EOM (`ipccsd`/`eaccsd`/`eeccsd`).
No UCCSD/GCCSD, no DF-CCSD, no PBC.

**Reference modules in `pyscf/pyscf/cc/`:** `ccsd.py` (RHF driver),
`rintermediates.py` / `uintermediates.py` (Woo/Wvv/Wovoo… builders — port these
almost verbatim as `contract` calls), `ccsd_t.py` + `_ccsd` (triples),
`ccsd_lambda.py`, `ccsd_rdm.py`, `uccsd.py`, `gccsd.py`, `dfccsd.py`,
`eom_rccsd.py` / `eom_uccsd.py`.

**Build order (see `docs/STRATEGY.md` for detail):**

1. **DF-RCCSD** — `(ia|P)` 3-center ERIs from `gpu4pyscf.df` instead of incore
   `vvvv`; block over occ; the practical large-system path. Reuse
   `gpu4pyscf/mp/dfmp2.py` for the integral pipeline and MP2 initial guess.
2. **RCCSD(T)** — triples correction; custom-tiled kernel over occupied triples.
3. **Λ equations + 1-/2-RDM → RCCSD analytic gradient.**
4. **UCCSD / UCCSD(T)** (open shell).
5. **EOM-EE / IP / EA-CCSD** — `matvec` + Davidson on GPU.
6. GCCSD, DF-CCSD polish, PBC-CCSD (much later).

CC is dominated by tensor contractions: express intermediates with `contract`,
watch memory (`t2`, `eris.ovvv`, `eris.ovov` are the big arrays — block the
occupied index, spill to host or shard across GPUs when needed).

## Gotchas

- **cupy/numpy mixing** silently produces wrong results with some ufuncs — assert
  array types at boundaries.
- The cupy mempool limit is set at import (`__config__.py`, 90% of VRAM) and a
  conditional small-alloc mempool is installed in `__init__.py`. Large transient
  arrays can still OOM — check `get_avail_mem()` before allocating O(N⁴).
- Integral kernels assume the **grouped/sorted-basis** frame — mixing sorted and
  unsorted `ao_loc` gives subtly wrong integrals.
- cuTENSOR/cupy versions are tightly coupled — use `requirements.txt`, do not
  bump one alone.
- `to_cpu` only copies attributes named in `_keys`; a forgotten key → silent data
  loss on round-trip.
- PySCF is vendored here at a specific commit; check `pyscf/` for the actual API,
  not memory of a newer/older release.

## Where to put things

```
gpu4pyscf/gpu4pyscf/
  cc/            # coupled cluster  -> ccsd.py (DF), ccsd_t.py, ccsd_lambda.py,
                 #   ccsd_rdm.py, uccsd.py, eom_rccsd.py, tests/
  gto/ecp.py     # ECP Python driver (extend for SO-ECP, screening)
  lib/ecp/       # ECP CUDA kernels  -> add ecp_so.cu; CMakeLists.txt
  lib/cc/        # (new) any custom CC CUDA kernels, e.g. triples
  grad/ , hessian/   # wire analytic derivatives in here
  mp/            # DF-MP2 — reference pattern + dependency for CC
```
