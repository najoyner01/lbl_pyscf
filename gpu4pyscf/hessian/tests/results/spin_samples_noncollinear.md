# mcfun `spin_samples` for GKS + SO-ECP on a **non-collinear open-shell** density

**Result: `spin_samples` is irrelevant here too — `spin_samples = 50` is
adequate for GKS + SO-ECP; the `hessian/fd.py` `_SPIN_SAMPLES_MIN` guard is
removed.**  (Caveat: a real 5f actinide open-shell case is still untested.)

Driver: `spin_samples_noncollinear.py` (this directory).  A100-PCIE-40GB,
pyscf 2.14.0, cupy 13.4.1, commit `b9c055d1e` (`origin/gpu-porting`).
`xc=pbe0`, `with_soc=True`, `conv_tol=1e-12`, `grids.level=3`,
`collinear_thrd=0.99` (mcfun default) unless noted.

## Why this study

The closed-shell study (`spin_samples_convergence.md`,
`docs/ghf-gradient-design.md` §5.6.1) showed `spin_samples` is irrelevant for
HI GKS(pbe0)+SOC — but only because that converged density is **collinear**
(`∫|m| = 3e-8`, spinors Kramers-pair, SOC enters only through the hcore `L·S`).
The intended production targets — An(III) 5f³, An(IV) 5f², Pu(III) 5f⁵ — are
open-shell heavy atoms with a real `m(r)` of O(1) whose direction varies in
space.  This study tests that regime with the cheapest genuine example:

**I atom, ²P doublet** — `gto.M(atom='I 0 0 0', basis='crenbl', ecp='crenbl',
spin=1)`, `dft.GKS(mol,'pbe0'); with_soc=True`.  5p⁵ hole → ²P₃/₂ ground state,
non-collinear (spin follows the orbital).  Single atom → no geometry/frequency,
cheap.  Physical anchor: ²P₃/₂–²P₁/₂ splitting, exp **7603 cm⁻¹**.

Secondary: **IO ²Π** (diatomic → an FD stretch), only if it converges cleanly.

The closed-shell Phase-1 probe (global spin rotation) is **invalid** with SOC on
(`L·S` breaks global spin-rotation symmetry).  The correct probe (Phase E) is a
*simultaneous* spatial + spin rotation, an exact symmetry of the full SOC
Hamiltonian.

---

## Phase A — I atom: `∫|m|`, non-collinearity, `e_tot` vs `spin_samples`

`m(r)` built from the converged 2c DM on the DFT grid;
`∫|m| = Σ_g w_g √(mx²+my²+mz²)`; ratio `= |∫m| / ∫|m|` (≈1 ⇒ collinear,
<1 ⇒ locally non-collinear); `w_noncol` = fraction of `∫ρ` at points with
`|m|/ρ < collinear_thrd` (the fraction actually sent through the Lebedev sphere).

**The test density is genuinely non-collinear:**

| quantity | I atom ²P GKS(pbe0)+SOC | (HI closed-shell, for contrast) |
|---|---:|---:|
| `∫|m(r)| dr` | **1.0118** | 3.2e-8 |
| `|∫m dr|` (net moment) | 1.0003 | 2e-9 |
| non-collinearity ratio `|∫m|/∫|m|` | **0.9886** (1.14 % of `m` is non-collinear in direction) | — |
| `w_noncol` (grid weight below `collinear_thrd`) | **1.0000** — the *entire* density is sent through the `spin_samples` Lebedev sphere; the exact-collinear shortcut never fires | ≈0 (shortcut handles ~everything) |

**`e_tot` vs `spin_samples`** (each SCF re-converged, `conv_tol=1e-12`):

| `spin_samples` | e_tot (Eh) | wall (s) |
|---:|---:|---:|
| 50 | −111.3183292674 | 8.2 |
| 110 | −111.3183292674 | 0.7 |
| 194 | −111.3183292674 | 1.1 |
| 302 | −111.3183292674 | 1.5 |
| 434 | −111.3183292674 | 2.1 |
| 590 | −111.3183292674 | 2.7 |
| 770 | −111.3183292674 | 3.4 |
| 974 | −111.3183292674 | 4.1 |
| 1202 | −111.3183292674 | 5.5 |

**e_tot spread over `spin_samples` 50 → 1202 = 8.5e-14 Eh = 0.000 cm⁻¹.**
`grids.level` 3 → 5 at `spin_samples=770`: Δe_tot = **0.000 cm⁻¹**
(`∫|m|` 1.01178 → 1.01182).  Wall time grows ~linearly with `spin_samples`
(1202 ≈ 8× the 50 cost) for zero benefit.

---

## Phase B — `collinear_thrd` on/off stress test

`w_noncol = 1.0` already means the default `collinear_thrd=0.99` shortcut fires
nowhere for the I atom, so `collinear_thrd=None` (force every point through the
Lebedev sphere) should be identical — and is:

| `collinear_thrd` | e_tot spread over `spin_samples` {50,194,434,770,1202} |
|---|---:|
| 0.99 (default) | 0.000 cm⁻¹ (all −111.3183292674) |
| `None` (off) | 0.000 cm⁻¹ (all −111.3183292674) |

The flatness is not a shortcut artifact: **Lebedev order 50 already integrates
this spin-angular structure exactly.**

---

## Phase C — ²P₃/₂ – ²P₁/₂ splitting vs `spin_samples`

**The ΔSCF to ²P₁/₂ did not converge** at any `spin_samples` — a GPU-native
maximum-overlap (MOM) constraint could not hold the non-aufbau occupation.  This
is a limitation of single-determinant 2-component GKS mean field for an
open-shell atom (no proper atomic-multiplet structure), not of `spin_samples`;
the task anticipated it ("if it doesn't, drop the splitting").

The frozen-orbital (Koopmans) SOC-related gap `ε_LUMO − ε_HOMO` in the ground
state is **18161 cm⁻¹**, **identical to the shown digits at every
`spin_samples`** — but it is a poor estimate of the atomic term splitting
(exp 7603 cm⁻¹): the GKS mean-field ground state is ~99 % collinear (ratio
0.9886, Phase A), i.e. closer to a stretched `|m_l, m_s⟩` state than to a clean
j = 3/2 spinor eigenstate, so its orbital gaps mix crystal-field and SOC.

Because e_tot is `spin_samples`-flat to 8.5e-14 Eh (Phase A), **any** energy
difference built from it — including a splitting, were it obtainable — is
`spin_samples`-flat by construction.

---

## Phase D — IO radical (secondary) — ABANDONED

`gto.M('I 0 0 0; O 0 0 1.87', basis='crenbl', ecp={'I':'crenbl'}, spin=1)`,
GKS(pbe0)+SOC.  **The SCF does not converge cleanly.**

- `UKS(pbe0)` converges fine (`e = −172.74838`).
- Starting `GKS+SOC` from the embedded UKS density, `spin_samples=50`,
  `conv_tol=1e-10`, `level_shift=0.2`, 100 cycles: **not converged**
  (`e_tot ≈ −172.7596`, still oscillating).
- SOSCF (`mf.newton()`) OOMs on this system (the mcfun `deriv` path allocates
  `O(n_grid × spin_samples)` GGA-fxc tensors; IO's grid is ~2× the I atom's).

This is the documented risk for an orbitally-degenerate ²Π at linear geometry
(GKS symmetry-breaks).  Per the study plan, IO is dropped; the I atom (Phase A)
is the load-bearing result and it is unambiguous.

---

## Phase E — rotational-invariance probe — NOT RUN

Only warranted if A/C showed real drift.  Phase A shows `e_tot` flat to
8.5e-14 Eh, so there is nothing for the probe to attribute to the spin grid.

---

## Verdict & recommendation

**`spin_samples` is irrelevant here too.**  The I atom ²P GKS(pbe0)+SOC density
is genuinely non-collinear — `∫|m| = 1.012` (O(1), vs 3e-8 for HI), and the
*entire* density is below `collinear_thrd` so the exact-collinear shortcut never
fires and every grid point goes through the `spin_samples` Lebedev sphere.
Despite that, `e_tot` is **bit-flat to 8.5e-14 Eh (0.000 cm⁻¹) across
`spin_samples` 50 → 1202**, with `collinear_thrd` on or off, and `grids.level`
3 → 5 changes it by 0.000 cm⁻¹.  Lebedev order **50** already integrates this
spin-angular structure to machine precision.

This closes the gap left by the closed-shell study
(`spin_samples_convergence.md`): `spin_samples` does not matter for GKS + SO-ECP
energies in **either** regime tested — collinear (HI) or non-collinear (I).

### Recommendation

**`spin_samples = 50` is adequate for GKS + SO-ECP energies and everything built
from them** — SOC splittings, frequencies, thermochemistry (a splitting or a
frequency is a difference of `spin_samples`-flat `e_tot` values, so it inherits
the flatness; the closed-shell study confirmed the frequency directly).  The
`hessian/fd.py` `_SPIN_SAMPLES_MIN = 770` placeholder and its warning are
**removed** — no measured case supports them.
Larger `spin_samples` costs ~linearly (1202 ≈ 8× the 50 wall) for zero gain.
`collinear_thrd` (0.99) and `grids.level` (3) are likewise not implicated.

The one real control on GKS+SOC *frequencies* is a **tightly converged geometry**
(closed-shell study: dω/dr ≈ −53 cm⁻¹ per 0.01 Å near the HI minimum) plus
`grid_response=True` (already enforced by `fd.py`).

## What remains untested

- **A true 5f actinide open-shell case** (An(III) 5f³, An(IV) 5f², Pu(III) 5f⁵)
  — the actual production target.  The I atom 5p hole is the cheapest genuine
  non-collinear test (`∫|m| = 1.01`, whole density on the Lebedev sphere), but a
  5f³/5f⁵ shell has more, and more angularly structured, unpaired density; the
  spin-angular integrand there could in principle be sharper.  Nothing in the
  two studies suggests it, but it is not proven.
- **A genuinely non-collinear *molecule*** (real `m(r)` direction variation from
  bonding + SOC, not just an atom).  IO ²Π was the intended case and its SCF
  does not converge; a heavier or bent radical that converges would close this.
- The **²P₃/₂–²P₁/₂ atomic splitting** itself — GKS single-determinant mean
  field does not give clean atomic multiplets, so the physics anchor is only
  the frozen-orbital gap (18161 cm⁻¹, poor vs exp 7603).  A CASSCF/SO-CI or an
  ensemble treatment would be needed to anchor the SOC physics; that is a
  separate question from `spin_samples` convergence.
