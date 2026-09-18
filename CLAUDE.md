# CLAUDE.md — PySCF → GPU porting project

## Current focus (as of 2026-09-17) — read this first

The real driving goal is **replacing ADF** for actinide-series research
(molecular; periodic is a separate, descoped track) — see
`docs/adf-parity-strategy.md`. Status: Tier 1 (SOC energies/gradients,
charges, geometry opt, solvation, thermochemistry) is DONE. Currently mid an
**actinide-readiness gate**: an all-electron X2C-SOC nuclear gradient just
landed (`grad/x2c.py`, atomic approximation) and unblocks variational
actinide SOC geometry optimization at GHF level. **Top open item:** that same
validation found a curvature bug in `grad/gks.py:_gks_xc_grad` for strongly
non-collinear densities (GKS/DFT SOC frequencies on heavy-SOC systems aren't
trustworthy yet) — see `docs/adf-parity-strategy.md` §4.5 item 3g for the
diagnostic protocol already written up. Fix that, then Stage 2b (open-shell
actinide, `NpO₂²⁺`/`[UCl₆]²⁻`) is next. Coupled cluster (below) and Tier
2/3 ADF-parity items (ETS-NOCV, QTAIM, EPR/ESR, …) are untouched, independent
tracks — pick up whenever.

## Next task (ready to run on Perlmutter) — fix `_gks_xc_grad`

The prompt below is drafted and ready to hand to a Perlmutter session
verbatim. It is the top open item (see above).

```
You are working on gpu4pyscf at /pscratch/sd/n/namehta4/Quantum/lbl_pyscf, branch
`gpu-porting`. PySCF→GPU port. `pyscf/` is READ-ONLY reference — never edit it.
$SCRATCH = /pscratch/sd/n/namehta4.

# GOAL

Fix a curvature error in `gpu4pyscf/grad/gks.py:_gks_xc_grad` (the
multi-collinear XC nuclear-gradient term) that shows up on STRONGLY
non-collinear 2c densities. Read `docs/ghf-gradient-design.md` §5.3 and
`hessian/tests/results/x2c_soc_grad_A100.md` §V3 first — that's where it was
found and isolated.

This is CORRECTNESS work on an already-merged, partially-validated function.
Do not guess-and-check: localize the bug with the diagnostics below BEFORE
changing code, then fix, then re-validate broadly (the fix must not break the
cases that already pass).

# THE FINDING (from x2c_soc_grad_A100.md V3)

On uranyl (UO2 2+) with GKS/pbe0 + X2C-SOC (`approx='atom1e'`), at r(U=O) =
1.66 / 1.68 / 1.70 Å:
  - the analytic X2C-SOC hcore derivative is FD-exact (V2u, ~1e-7 vs FD of the
    forward get_hcore — no SCF, no XC grid involved);
  - GHF uranyl (no XC term at all) is FD-exact (V2, 4.4e-5 at a displaced,
    large-gradient geometry);
  - the SO-off reduction matches pyscf's `sfx2c1e_grad` to 1.3e-10;
  - **but** the GKS analytic gradient's implied PES curvature is ~4x too weak
    vs central-FD of `e_tot` (analytic k ≈ 1.4 vs FD k ≈ 5.5 Eh/Å²), and the
    gap GROWS with displacement over that 0.04 Å span;
  - **the gap is UNCHANGED between `grid_response=True` and `False`.**

That last point is the key isolating clue: `grid_response=False` runs ONLY
`_xc_grad_orbital`; `grid_response=True` runs `_xc_grad_full_response`, which
recomputes an equivalent orbital-response term (`de_rho`) plus an ADDITIONAL
grid-weight-derivative term (`de_weight`). If toggling grid_response doesn't
change the error, the bug is most likely in code common to both paths — the
shared "channel" contraction pattern:

    for c in range(4):                          # c = rho, mx, my, mz
        wv[c]  (from mcfun's vxc, weight-scaled)
        vtmp = <AO-derivative> . wv[c]           # rks_grad._d1_dot_ / _gga_grad_sum_
        contribution += Tr[ vtmp . dm_ch[c] ]

which appears (nearly identically) inside both `_xc_grad_orbital` and the
`de_rho` accumulation of `_xc_grad_full_response`. Start there, not in
`de_weight`.

# WHY IT ONLY SHOWS UP NOW

The existing GKS+SOC gradient tests (`test_gks_grad.py`, and
`test_ghf_grad.py::TestGHFGradSOC` on HI/crenbl) validated `_gks_xc_grad` by:
  (a) exact collinear reduction to `grad/uks.py` (D_ab = 0, m_x=m_y=0 — never
      exercises those two channels at all);
  (b) FD on H2O (collinear, same blind spot) and HI/crenbl+SOC, where
      `|D_ab| ~ 3e-2` — a SMALL non-collinear admixture on a light system —
      at a tiny FD step (h=1e-4 bohr, one point).
Uranyl + X2C-SOC has a much larger, genuinely non-collinear m_x/m_y (heavy-atom
SOC), and V3 spans a WIDE geometry range. A working hypothesis: there is a
wrong/missing factor specific to the **m_x or m_y channel** (c=1,2) that is
small in absolute terms for weak non-collinearity (swamped by the tolerance in
the earlier tests) but large and geometry-dependent for uranyl — consistent
with "error grows with displacement" if the m_x/m_y magnitude itself changes
over that geometry range.

**One concrete thing to re-derive, not assume:** the GGA branch does
`wv[:, 0] *= .5` UNIFORMLY across all 4 channels
(`_xc_grad_orbital`/`_xc_grad_full_response`), copied from the ENERGY-side
`numint2c.py:_mcol_gga_vxc_mat`'s `if hermi: wv[:,0] *= .5  # because of
v+v.conj().T in r_vxc`. On the energy side that halving is compensated by an
explicit "+ Hermitian conjugate" step elsewhere in the Fock-matrix build. On
the gradient side, the contraction `Tr[vtmp . dm_ch[c]]` is NOT a Fock-matrix
build — there may be no equivalent completion, or it may need a different one
for the off-diagonal (m_x, m_y) channels specifically (which came from
`D_ab`/`D_ba`, an off-diagonal *block* pair, vs the diagonal `D_aa`/`D_bb`
pair that the standard RKS/UKS gradient already handles correctly via
`grad/uks.py`). Verify by direct re-derivation against `grad/uks.py`'s
GGA gradient term and the `_mcol_gga_vxc_mat` Fock build — do not assume the
convention transfers unchanged.

# DIAGNOSTIC PROTOCOL — localize before fixing

1. **Reproduce the discrepancy** as a regression baseline: uranyl X2C-SOC,
   r = 1.66/1.68/1.70 Å, reproduce the ~4x curvature mismatch from
   `x2c_soc_grad_A100.md` V3 exactly (same settings) before touching code.
2. **Channel isolation.** In `_gks_xc_grad`'s energy AND gradient evaluation,
   zero out one channel of `dm_ch` at a time (rho, mx, my, mz) and separately
   at a time, artificially zero the corresponding `vxc`/`wv` channel — compare
   analytic-vs-FD-of-`e_tot` for the truncated problem. This should show
   whether the error is confined to a single channel (m_x or m_y expected) or
   spread across the assembly. Careful: zeroing a channel changes the physics,
   so compare analytic-vs-FD for the *truncated* Hamiltonian at each geometry,
   not against the full result.
3. **Non-collinearity magnitude scan.** Use the global-spin-rotation trick
   from `test_ghf_grad.py::test_ghf_spin_rotation_invariance` — but recall
   SOC breaks that invariance, so instead: take a FIXED converged uranyl X2C-
   SOC density at ONE geometry, rotate the spin quantization axis by angle θ
   (this moves weight between the diagonal and m_x/m_y channels while leaving
   the physical state equivalent), and compare the analytic gradient before
   vs after rotation. If `_gks_xc_grad` is correct, the *total* gradient must
   be invariant under a pure spin rotation at fixed geometry (rotating the spin
   frame doesn't change any nuclear force) — a MUCH cheaper and sharper
   diagnostic than a geometry scan, no SCF re-convergence needed, and it
   isolates exactly the m_x/m_y-dependence you're hunting.
4. Once localized, re-derive that piece term-by-term against
   `numint2c.py:_mcol_lda_vxc_mat`/`_mcol_gga_vxc_mat` (the validated energy
   side) and `grad/uks.py` (the validated collinear gradient), the same way
   `grad/ghf.py`'s design doc derived the JK term — write the derivation down
   before editing code.

# FIX + VALIDATION

- Fix `_gks_xc_grad` (LDA and GGA branches; check MGGA is still correctly
  NotImplemented, don't accidentally half-fix it).
- Re-run V3's exact setup (uranyl, r=1.66/1.68/1.70) — curvature must now
  agree with FD of `e_tot` to a tight tolerance (define one, e.g. <5% or
  <1e-5 Eh/Å² depending on what FD-noise allows).
- Add a NEW test exercising strong non-collinearity directly and cheaply: the
  spin-rotation-invariance-at-fixed-geometry check from step 3 above, made
  permanent in `grad/tests/test_gks_grad.py` (target ~1e-8, no SCF needed).
- Regression: `grad/tests/test_gks_grad.py`, `test_ghf_grad.py` (incl.
  `TestGHFGradSOC`), `test_soc_geomopt.py`, `solvent/tests/test_pcm_soc.py`,
  `hessian/tests/test_fd_hessian.py` must all still pass — the fix must not
  perturb the already-correct collinear/weak-non-collinear cases.
- Re-run `x2c_soc_grad_A100.md`'s V3/V5 uranyl checks post-fix and update that
  doc with the corrected curvature numbers.

# CONSTRAINTS

- Never edit pyscf/. Fix stays in `grad/gks.py`; if the bug is in a shared
  helper (`rks_grad._gga_grad_sum_` etc.), fix at the call site in `gks.py`
  unless the helper itself is provably wrong for the multi-collinear case —
  flag that explicitly rather than changing a shared RKS/UKS helper blindly.
- Do not touch the X2C hcore-derivative term (`grad/x2c.py`) — it's validated
  and out of scope here.
- Commit trailer:
      Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
  (add your own Claude-Session line).
- Cannot push from Perlmutter. When done:
      git bundle create $SCRATCH/gks-xc-grad-fix.bundle origin/gpu-porting..HEAD
      git bundle verify $SCRATCH/gks-xc-grad-fix.bundle
  then tell me: which channel/term was wrong and why (the actual mechanism,
  not just "fixed it"), the corrected uranyl curvature vs FD, the
  spin-rotation-invariance number for the new test, and the regression
  results across the suites listed above.
```

## Purpose (original framing — still the two intended tracks, CC is dormant)

Port as much of PySCF's functionality to GPU as practical, inside the
**`gpu4pyscf`** plugin. Immediate targets, in order:

1. **ECP** (effective core potentials) — feature-complete for molecular work
   (energy + gradient, incl. actinide integrals); periodic gradient/stress and
   perf items remain, not on the critical path. See `docs/adf-parity-strategy.md`
   for the actinide-SOC caveat (variational 2c-ECP-SOC is dead-ended there;
   X2C-SOC is the active replacement).
2. **Coupled cluster** (CCSD → CCSD(T) → Λ/RDM/gradients → UCCSD → EOM-CC) —
   dormant since the initial incore RHF CCSD; not touched during the ADF-parity
   work.

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

**Status — all five gaps closed, merged to `gpu-porting`:**

| Gap | Where | Notes |
|-----|-------|-------|
| ~~**Screening**~~ | `gto/ecp.py` | Two-level `check_3c_overlap` screen (`_screen_block`, `SCREEN_ECP` toggle). A100: screened==unscreened to 1e-13; 14x/55x/189x on Cu-chain N=20/40/80 (`benchmarks/gto/ecp_screening.md`). |
| ~~**ECP-atom slicing in grad/hess**~~ | `gto/ecp.py`, `grad/rhf.py`, `hessian/rhf.py`, `lib/cuest_wrapper.py` | `loop_ecp_ip`/`loop_ecp_ipip` + `get_ecp_ip_sum`/`get_ecp_ipip_sum` batch over ECP atoms; the `[n_ecp_atm,3or9,nao,nao]` tensor is never materialized. `get_ecp_ip`/`get_ecp_ipip` kept as concatenating wrappers. 23 A100 tests. |
| ~~**Spin-orbit ECP (SO-ECP)**~~ | `gto/ecp.py`, `lib/ecp/ecp_so.cu`; `docs/ecp-so-design.md` | `get_ecp_so → [3,nao,nao]` real + `get_soc_1e → [2nao,2nao]`. `so_cart` = `type2_cart` + `L^a` transform of `omegaj` (`so_ang_matrix.cu`), **same prefactor as scalar type-2** (no complex kernel). A100: == `mol.intor('ECPso')`/GHF block to 1e-10 on `ul` + explicit S–G projectors. **Gradient done** (2026-09-08): bra-derivative kernel `ECP_so_ip_cart` (`lib/ecp/ecp_so_ip.cu`), `gto/ecp.py:loop_ecp_so_ip`/`get_ecp_so_ip`, wired into `grad/ghf.py:_soc_hcore_grad` for `with_soc=True` gradients at **HF (`grad/ghf.py`)** and **DFT (`grad/gks.py`)** level. `grad/gks.py` = ghf grad + hybrid-scaled K + multi-collinear XC gradient (LDA/GGA/global-hybrid + grid response; RSH/MGGA/NLC raise). FD-validated: SO-ECP integral ≲5e-11, GHF+SOC non-collinear 2.6e-9, GKS+SOC 7.6e-9, GKS collinear→UKS 1e-11. **SOC geometry optimization works** (2026-09-08): `pyscf.geomopt.geometric_solver.optimize` converges for `GHF/GKS + with_soc=True` — pass `g.as_scanner()` (not a bare `Gradients`), `g.grid_response=True` for GKS; `test_soc_geomopt.py`. **Solvated SOC works** (2026-09-09): `GHF/GKS.PCM()`/`.SMD()` incl. `with_soc=True` — reaction field couples to `Re(D_αα+D_ββ)`, block-diagonal into the 2c Fock; `solvent/tests/test_pcm_soc.py` (FD ≲3e-8, ΔG_solv sane, optimize converges). **FD Hessian + thermo** (2026-09-09): `hessian/fd.py` — `GHF/GKS.Hessian()`, `(natm,natm,3,3)` for `pyscf.hessian.thermo`; O(6N) SCF+grad, small systems only. Reduces to analytic UHF/UKS Hessians to ~2e-6. `spin_samples` **RESOLVED** (2026-09-09): irrelevant for GKS+SO-ECP — `spin_samples=50` reproduces 1202 to <1e-6 cm⁻¹ for both collinear (HI) and genuinely non-collinear (I atom ²P, ∫\|m\|=1.0) densities; the ~180 cm⁻¹ spread once blamed on it was geometry (dω/dr ≈ 53 cm⁻¹ per 0.01 Å near the HI minimum). `hessian/fd.py` `_SPIN_SAMPLES_MIN` guard removed; see `hessian/tests/results/spin_samples_{convergence,noncollinear}.md`, `docs/ghf-gradient-design.md` §5.6.1. Untested: a true 5f actinide open-shell case. **Actinide SO-ECP integrals** (2026-09-09): the ECP60MWB-SO set ships in vendored pyscf as `ecpds60mwbso` (Ac–Lr; AREP identical to `stuttgart_rsc`). `get_ecp_so`/`get_soc_1e` reproduce `mol.intor('ECPso')` to ~1e-13 for Th–Am incl. f/g (l=3,4) projectors — `so_ang_matrix.cu`/`ECP_LMAX=4` already cover it, no regen. Accessor `gto/actinide_ecp.py`, test `gto/tests/test_actinide_ecp.py`. **But** a *self-consistent* 2c-GKS+`ecpds60mwbso` collapses ~5 Eh (the explicit, strongly SO-split U 6p semicore over-couples the unbounded semilocal SO projector; these sets are for restricted SO-CI, not variational 2c) — not a gpu4pyscf bug (one-shot Fock == pyscf CPU to 1e-12). Use a Dirac-fitted 2c ECP or X2C-SOC for variational actinide 2c; see `hessian/tests/results/uranyl_soc_2a.md`. Follow-ups: templated fast paths; X2C-SOC & RSH/MGGA-GKS gradients; **analytic** SOC Hessian. |
| ~~**Validation breadth**~~ | `gto/tests/test_ecp_sweep.py` | 1526 A100 subtests: get_ecp/_ip/_ipip/_so/get_soc_1e vs `mol.intor` across ~12 scalar + 6 SO ECP sets × ~19 elements Na–Bi, cart+sph, s..g probe basis. |
| ~~**PBC ECP**~~ | `pbc/gto/ecp.py`, `pbc/scf/hf.py`; `docs/ecp-pbc-design.md` | `ecp_int(cell, kpts=None, intor='ECPscalar'\|'ECPso')` — supmol lattice sum reusing `libgecp.ECP_cart`/`ECP_so_cart`, no new CUDA; ket images batched by free memory. Wired into `pbc/scf/hf.get_hcore`. A100: 10/10 tests, == `pyscf.pbc.gto.ecp.ecp_int` (scalar 5e-9, SO 5e-9). Only follow-up left: PBC ECP gradient/stress (phase 5). |
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
