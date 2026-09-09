# FD nuclear Hessian for GHF / GKS — A100 validation run (V1–V7)

Captured run of `gpu4pyscf/hessian/tests/test_fd_hessian.py` (the V1–V6 + wiring
gates for `gpu4pyscf/hessian/fd.py`) plus the full analytic-Hessian regression
suite (V7).  Raw pytest console logs are not committed (`*.log` is gitignored);
this file is the curated record.

| | |
|---|---|
| host | `login28`, NVIDIA A100-PCIE-40GB (40 GB) |
| commit under test | `c9927b5ef` (`hessian: finite-difference nuclear Hessian for GHF / GKS`) |
| software | pyscf 2.14.0, cupy 13.4.1, geomeTRIC 1.1, pytest 9.1.1, Python 3.11.16 |
| settings | `spin_samples=50`, `disp=1e-3` Bohr, `conv_tol ≤ 1e-12`, `grids.level=3` |
| dates (UTC) | FD suite + regression 2026-09-09T17:56Z…19:xxZ; parallel corroboration 2026-09-09T19:02Z |

## V1–V6 + wiring — `test_fd_hessian.py`

`8 passed in 236.78s (0:03:56)` — `FD_SUITE_RC=0`

| gate | test | measured | assertion | verdict |
|---|---|---|---|---|
| **V1** | `test_v1_ghf_fd_vs_uhf_analytic` — FD GHF Hessian (real block-diagonal, UHF MOs embedded, no SOC) vs analytic UHF Hessian, H₂O/STO-3G | `‖H_fd − H_uhf‖ / ‖H_uhf‖ = 1.73e-06` | `< 5e-4` | PASS |
| **V2** | `test_v2_gks_fd_vs_uks_analytic` — FD GKS(pbe0) vs analytic UKS(pbe0, `grid_response=True`), H₂O/STO-3G | `‖H_fd − H_uks‖ / ‖H_uks‖ = 2.62e-06` | `< 5e-4` | PASS |
| **V3 GHF+SOC** | `test_v3_ghf_soc_hi_frequencies` — HI/CRENBL, `with_soc=True`, at the optimized geometry, unprojected `harmonic_analysis` | 6 freqs (cm⁻¹) = `[0.00, 0.00, 0.00, 0.64, 7.23, 2441.36]` → 5 modes `|ν| < 50`, stretch **2441.36** | 5 near-zero + 1 stretch in 2000–2600 | PASS |
| **V3 GKS+SOC** | `test_v3_gks_soc_hi_frequencies` — HI/CRENBL, GKS(pbe0), `with_soc=True`, optimized geometry | 6 freqs (cm⁻¹) = `[0.00, 0.00, 0.00, 0.00, 0.17, 2252.81]` → 5 modes `|ν| < 50`, stretch **2252.81** | 5 near-zero + 1 stretch in 2000–2600 | PASS |
| **V4** | `test_v4_soc_shifts_the_stretch` — HI GHF stretch with vs without `with_soc`, common fixed geometry | SOC = **2407.96**, no-SOC = **2413.21**, shift = **5.252 cm⁻¹** | shift `∈ (1e-2, 100)` cm⁻¹ | PASS |
| **V5** | `test_v5_thermo_gks_soc_and_pcm` — `harmonic_and_thermo`, HI GKS(pbe0), `with_soc=True`; then `.PCM()` | GKS+SOC: `G_tot = −111.93637728`, `H_tot = −111.91294472`, `S_tot = 7.859317e-05`, `ZPE = 0.00553616`.  GKS+SOC+PCM: `G_tot = −111.93962572` | all finite | PASS |
| **V6** | `test_v6_step_size_convergence` — GHF+SOC HI, `disp` sensitivity | `‖H(1e-3) − H(2e-3)‖ = 5.79e-06`, `‖H(1e-3) − H(5e-3)‖ = 9.77e-06` (Ha/Bohr²) | `‖H(1e-3) − H(2e-3)‖ < 1e-3` | PASS |
| wiring | `test_hessian_method_and_conv_tol_guard` — `scf.GHF(mol).Hessian()` / `dft.GKS(mol).Hessian()` return `hessian.fd.Hessian`; `conv_tol=1e-9` raises `ValueError` | — | — | PASS |

Notes:
- The GKS+SOC stretch (V3) is `2252.81 cm⁻¹` in this run vs `~2430` recorded on
  an earlier draw — the mcfun non-collinear XC integration with `spin_samples=50`
  is stochastic, so the GKS stretch wanders run-to-run by ~O(100 cm⁻¹).  The GHF
  path (no Monte-Carlo grid) reproduces `2441 cm⁻¹` tightly, and V4/V6 are
  unaffected (GHF).  All still satisfy the 2000–2600 cm⁻¹ gate.

## V7 — analytic-Hessian regression suite unaffected

`pytest gpu4pyscf/hessian/tests/ --deselect test_fd_hessian.py`

`93 passed, 1 skipped, 8 deselected, 17 warnings in 4771.71s (1:19:31)` — `REGRESSION_RC=0`

- 8 deselected = the `test_fd_hessian.py` FD gates (run separately above).
- 1 skipped = `test_large_exponent.py::test_hessian_large_exp_methylbromide_rks`
  — a pre-existing conditional skip, unrelated to this change.
- 0 failures.  Covers analytic RHF / RKS / UHF / UKS Hessians, grid-response
  RKS/UKS, level-shift, large-exponent, D3/D4 dispersion, and the full VV10
  (non-local correlation) Hessian family.

### Parallel corroboration

The 29 slowest V7 files (`test_uks_hessian_grid_response.py` +
`test_vv10_hessian.py`) were additionally run in a second concurrent process
sharing the same GPU:

`29 passed in 723.61s (0:12:03)` — `PARALLEL_RC=0` (GPU footprint ~3 GB / 40 GB
with both processes; no OOM / CUDA contention error).

## Summary

The change is one new module (`gpu4pyscf/hessian/fd.py`) plus one import line in
`gpu4pyscf/hessian/__init__.py`; it does not touch any analytic-Hessian code
path.  V1–V6 + wiring: **8/8 pass**.  V7: **93 pass, 1 pre-existing skip, 0 fail**
(+ 29/29 on the parallel re-run).  An analytic 2-component / SOC Hessian remains
future work.
