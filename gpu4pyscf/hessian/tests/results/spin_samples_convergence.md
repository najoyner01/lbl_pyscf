# mcfun `spin_samples` for GKS + SO-ECP on a **closed-shell** density (HI)

**Result: `spin_samples` is irrelevant for HI GKS(pbe0)+SOC — `spin_samples = 50`
reproduces 1202 to < 1e-6 cm⁻¹.  The ~180 cm⁻¹ "spread" once blamed on it was a
geometry-convergence artifact (the two readings were at different bond lengths).**

A100-PCIE-40GB, pyscf 2.14.0, cupy 13.4.1, `origin/gpu-porting`.
`xc=pbe0`, `with_soc=True`, `conv_tol ≤ 1e-12`, `grids.level=3`, `disp=1e-3` Bohr,
`grid_response=True` unless noted.

> Record transcribed from the Perlmutter run report; the driver script for this
> closed-shell study was not captured in a bundle (the follow-up non-collinear
> study, `spin_samples_noncollinear.md` + `spin_samples_noncollinear.py`, is
> committed in full and reaches the same conclusion).

## Why this study

The V3 gate of the FD-Hessian validation (`fd_hessian_A100.md`) recorded the HI
GKS(pbe0)+SOC stretch at **2253 cm⁻¹**, where an earlier draw had given
**≈2430 cm⁻¹** — both nominally at `spin_samples=50`.  Three hypotheses:

- **H1** — a coarse Lebedev spin grid leaves `E_xc` weakly dependent on the
  (physically arbitrary) spin-quantization-axis orientation, which the SCF lands
  on differently run to run.
- **H2** — the two runs converged to genuinely different SCF solutions.
- **H3** — the two runs were read at slightly different geometries.

## Phase 1 — spin-rotation invariance of `E_xc` vs `spin_samples`

HI GKS(pbe0)+SOC, fixed geometry.  Converged density has `∫|m(r)| = 3.2e-8`
(collinear).  Global SU(2) rotation of the converged spinor MOs, **no
re-convergence**:

| spin_samples | max\|ΔE_xc(θ)\| (Eh) | max\|ΔE_tot(θ)\| (Eh) |
|---:|---:|---:|
| 50   | 5.3e-15 | 1.842e-2 |
| 110  | 5.3e-15 | 1.842e-2 |
| 194  | 3.6e-15 | 1.842e-2 |
| 302  | 3.6e-15 | 1.842e-2 |
| 434  | 3.6e-15 | 1.842e-2 |
| 590  | 5.3e-15 | 1.842e-2 |
| 770  | 5.3e-15 | 1.842e-2 |
| 974  | 3.6e-15 | 1.842e-2 |
| 1202 | 5.3e-15 | 1.842e-2 |

`ΔE_xc` is at the double-precision floor at every `spin_samples` — nothing to
converge.  `ΔE_tot` (1.84e-2 Eh) is the physical `L·S` term (it *should* change
under a spin-only rotation with SOC on), and it is itself `spin_samples`-flat.
**H1 fails immediately.**

## Phase 2 — FD stretch frequency vs `spin_samples` (fixed geometry)

| spin_samples | stretch (cm⁻¹) | ZPE (Eh) | e_tot (Eh) |
|---:|---:|---:|---:|
| 50   | 2252.4304173 | 0.00513141406 | −111.9221668694 |
| 110  | 2252.4304173 | 0.00513141406 | −111.9221668694 |
| 194  | 2252.4304173 | 0.00513141406 | −111.9221668694 |
| 302  | 2252.4304173 | 0.00513141406 | −111.9221668694 |
| 434  | 2252.4304173 | 0.00513141406 | −111.9221668694 |
| 590  | 2252.4304173 | 0.00513141406 | −111.9221668694 |
| 770  | 2252.4304173 | 0.00513141406 | −111.9221668694 |
| 1202 | 2252.4304173 | 0.00513141406 | −111.9221668694 |

Flat to < 1e-6 cm⁻¹.  `grids.level` 3 → 5: stretch moves **0.004 cm⁻¹**.
`grid_response` on/off: **0**.  Second system I₂/CRENBL, `spin_samples`
50 / 194 / 770 → all **216.5002244 cm⁻¹**.

### The ~180 cm⁻¹ was geometry

FD stretch vs bond length near the HI minimum:

| r (Å) | 1.60 | 1.62 | 1.6425 | 1.66 | 1.68 |
|---|---:|---:|---:|---:|---:|
| stretch (cm⁻¹) | 2486.65 | 2374.46 | 2252.43 | 2160.48 | 2058.45 |

`dω/dr ≈ −53 cm⁻¹ per 0.01 Å`.  A reading of ~2430 cm⁻¹ corresponds to
`r ≈ 1.61 Å` — the unoptimized `_hi(r=1.61)` start point, **not a minimum**.

## Phase 3 — ΔG error at `spin_samples = 50` vs 770

| | ΔG error at ss=50 | G_tot (ss=50) | G_tot (ss=770) |
|---|---:|---:|---:|
| gas | 8.0e-11 kcal/mol | −111.9372010 | −111.9372010 |
| PCM | 2.7e-11 kcal/mol | −111.9404482 | −111.9404482 |

`G_tot` bit-identical.  `spin_samples = 50` is numerically exact for this
workflow, not merely fast.

## Verdict

| hypothesis | outcome |
|---|---|
| **H1** coarse spin grid → spurious spin-axis dependence of `E_xc` | **REFUTED** — `spin_samples` is a deterministic Lebedev order; converged density is collinear (`m ≈ 0`); `ΔE_xc = 5e-15` at ss=50, flat to 1202. |
| **H2** different SCF solution | **REFUTED** — 4 perturbed-guess SCF runs agree in `e_tot` to 2e-13; FD stretch reproducible to 7 decimals. |
| **H3** different geometry | **CONFIRMED** — frequency is `spin_samples`/grid/run-independent at fixed geometry; the ~180 cm⁻¹ spread = a ~0.03 Å geometry difference. |

## Recommendation

`spin_samples = 50` is adequate for closed-shell / collinear-regime SOC
thermochemistry (the ADF-parity solvated-ΔG workflow).  The `hessian/fd.py`
`_SPIN_SAMPLES_MIN` guard is **removed** — no measured case supports it.  What
controls the HI frequency is (1) a tightly converged geometry (`grms ≤ 1e-5`)
and (2) `grid_response=True` (already enforced by `fd.py`).

**Still open at this point:** a genuinely non-collinear SOC density
(open-shell, `∫|m| ~ O(1)`) — addressed in `spin_samples_noncollinear.md`
(same conclusion).
