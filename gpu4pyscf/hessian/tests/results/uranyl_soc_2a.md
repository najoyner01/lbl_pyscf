# Stage 2a — actinide validation on uranyl (UO₂²⁺)

**Status: COMPLETE.**
Scalar GKS + ECP machinery validated on uranium.  The actinide **spin-orbit
ECP integrals** are present and validated to ~1e-13 (incl. the f/g projectors
the Na–Bi sweep never stressed).  A **self-consistent** GKS + SO-ECP geometry /
frequency run **cannot** be done for uranyl — not for lack of data or a code
bug, but because the only available actinide SO-ECP (60-core `ecpds60mwbso`) is
variationally unstable in 2-component SCF.  Details below.

Driver: `uranyl_soc_2a.py` (this directory).  A100-PCIE-40GB, pyscf 2.14.0,
cupy 13.4.1, branch `uranyl-2a` off `origin/gpu-porting` (= `319a3a48a`).

## Headline

1. **The actinide SO-ECP *does* ship in this pyscf** — as `ecpds60mwbso`
   (`pyscf/gto/basis/soecp/ECPDS60MWBSO.dat`), covering **Ac–Lr**.  It is the
   ECP60MWB-**SO** set (Küchle 1994 / Cao 2003) — the spin-orbit partner of the
   scalar `stuttgart_rsc` (ECP60MWB) that Stage 2a already used.  The earlier
   "no actinide SO-ECP" conclusion came from trying only the scalar name.
   With `ecp={'U':'ecpds60mwbso'}`: `mol.has_ecp_soc()` → **True**, SO
   projectors l = 1,2,3,4; the AREP (scalar) columns are **byte-identical** to
   `stuttgart_rsc` (Δ = 0.0).  New accessor: `gpu4pyscf/gto/actinide_ecp.py`
   (a thin documented wrapper over the vendored data — no parameter
   re-transcription).

2. **gpu4pyscf's SO-ECP kernel is correct for the actinide f (l=3) and g (l=4)
   projectors.**  `get_ecp_so` vs `mol.intor('ECPso')`, U/Np/Pu/Th/Am,
   spherical **and** cartesian: **max |Δ| = 7.9e-14** (target 1e-9).
   `get_soc_1e` vs the pyscf spinor assembly: **4.0e-14**.  The angular table
   `lib/ecp/so_ang_matrix.cu` (`_l_op`, l = 0..4) and `ECP_LMAX = 4`
   (`lib/ecp/ecp.h`) already cover l ≤ 4 — **no regeneration, no libgecp
   rebuild.**  `gto/tests/test_actinide_ecp.py`: **7 passed**.

3. **A self-consistent 2-component GKS + SO-ECP calculation on uranyl is
   variationally unstable.**  Seeded from the converged scalar RKS density
   (the `minao` guess diverges outright), GKS + `ecpds60mwbso` converges — but
   to **−631.6496 Eh, 5.077 Eh *below* the scalar −626.5722 Eh**.  A 5-Hartree
   "SOC shift" on a closed-shell 5f⁰ ion is unphysical (the real effect is
   ~0.01–0.1 Eh).  Every SCF strategy available for gpu4pyscf GKS (plain DIIS,
   level shift 0.2/0.5/1.0, ramped) lands in the same collapsed state; SOSCF is
   `NotImplemented` for GKS.
   * **Not a gpu4pyscf bug.**  One-shot GKS+SOC `hcore` / `veff` / `energy`
     from a fixed DM match pyscf CPU to **1e-12**.  The collapse also
     reproduces in ordinary Rayleigh–Schrödinger PT2 (full-space
     E₂(SOC) ≈ −5.7 Eh).
   * **Root cause: the ECP.**  The 60-core actinide set carries the U 5s5p5d
     6s6p semicore explicitly.  Its P/D spin-orbit projectors (coefficients
     ≈ −60, +15) were parameterised for a *restricted-active-space spin-orbit
     CI*, not for a variational 2c-SCF one-electron operator.  The deeply
     bound (ε ≈ −8 Eh), strongly SO-split **6p semicore** over-couples through
     the (unbounded-below) semilocal SO projector.  Restricting the same PT2 to
     the frontier valence window gives a **physical E₂(SOC) ≈ −0.10 Eh**
     (~2.8 kcal/mol); the top full-space contributors are exactly the ε ≈ −8 Eh
     (6p) orbitals.
   * There is **no large-core actinide SO-ECP** in this pyscf to move 6p into
     the core (only `ECPDS60MWBSO` covers actinides).

4. Scalar results (unchanged from the first pass; the `gto/ecp.py` fix
   `0d9a363ad` this study depends on is committed separately):
   r(U=O) = **1.6791 Å**, ν₁/ν₂/ν₃ = **1092 / 181 / 1183 cm⁻¹**, no imaginary
   modes, SCF needs no aids.

## Basis / ECP

| | choice | note |
|---|---|---|
| U ECP, scalar | `stuttgart_rsc` (ECP60MWB, `ncore=60`) | Stage 2a scalar validation |
| U ECP, +SO | **`ecpds60mwbso`** (ECP60MWB-SO, `ncore=60`) | same AREP (Δ=0), adds SO projectors l = 1–4 |
| U basis | `stuttgart_rsc` valence | matched |
| O basis | `def2-TZVP` | task spec |
| accessor | `gpu4pyscf.gto.actinide_ecp.get_actinide_so_ecp('U')` → `'ecpds60mwbso'` | + `actinide_basis`, `source`, `has_actinide_so_ecp`; covers Th, Pa, U, Np, Pu, Am, Cm (and the rest of Ac–Lr) |

Provenance (module docstring, per element): W. Küchle, M. Dolg, H. Stoll,
H. Preuss, *J. Chem. Phys.* **100**, 7535 (1994) [ECP + SO potential];
X. Cao, M. Dolg, H. Stoll, *J. Chem. Phys.* **118**, 487 (2003) [valence basis].
Both are reproduced verbatim in the header of `soecp/ECPDS60MWBSO.dat`.

## Phase 0-SOC — SO-ECP prerequisites

| check | sph | cart |
|---|---|---|
| `mol.has_ecp_soc()` | True | True |
| SO projector l values | 1, 2, 3, 4 | 1, 2, 3, 4 |
| `‖get_ecp_so − mol.intor('ECPso')‖_max` | **7.9e-14** | **7.9e-14** |
| `‖get_soc_1e − pyscf spinor assembly‖_max` | **4.0e-14** | 4.0e-14 |
| `‖get_ecp(ecpds60mwbso) − get_ecp(stuttgart_rsc)‖_max` (AREP unchanged) | **0.0** | 0.0 |

`get_ecp_so` vs `mol.intor('ECPso')`, wider actinide set (probe basis, r = 2.1 Å
U–F): **U 7.9e-14, Np 7.5e-14, Pu 9.0e-14, Th 5.6e-14, Am 9.9e-14** — sph and
cart identical to 2 sig figs.

**Angular machinery (Phase 2, step 2 of the spec):** the SO projectors run
l = 1..4.  `lib/ecp/so_ang_matrix.cu` is generated for l = 0..4
(`_l_op_s..._l_op_g`, `*_l_op[5]`), `ECP_LMAX` in `lib/ecp/ecp.h` is 4 —
**both already cover the actinide f/g SO projectors**, so no
`generate_so_ang_matrix.py` re-run and no `libgecp` rebuild were needed.  (The
Na–Bi sweep did exercise l = 3,4 SO projectors via `ecpds28mwbso`/Ce and
`ecpds60mdfso`/Bi; what was untested before this study is an actinide 60-core
*radial* SO parameter set.)

**Physical anchor (Phase 2, step 5).**  Projecting `get_ecp_so` onto a single
U 5f radial function and diagonalising the 14×14 spinor SO operator gives the
exact L·S level pattern — a 6-fold (j = 5/2) level below an 8-fold (j = 7/2)
level with eigenvalue ratio **−2 : +3/2** — confirming the angular operator.
The magnitude ζ_5f is radial-shape dependent (6 → 6300 cm⁻¹ across trial
exponents 0.3 → 2.0); a physically sized 5f (exp ≈ 1.2) gives ζ_5f ≈ 1300 cm⁻¹,
bracketing the literature ζ_5f(U) ≈ 1900–2200 cm⁻¹.  A matched-core control on
iodine (`crenbl`, the p-block SO-ECP that *is* variationally stable) gives
ζ_5p ≈ 5.9×10³ cm⁻¹ vs the experimental ²P₃/₂–²P₁/₂ ⇒ ζ_5p ≈ 5.1×10³ cm⁻¹.
The SO-ECP integrals are physically scaled; the uranyl problem (headline 3) is
the *semicore in the valence space under a variational treatment*, not the
integrals.

## Phase 1 — scalar geometry (RKS/PBE0)   [unchanged]

`optimize(mf, convergence_grms=1e-4, convergence_gmax=2e-4)`, geomeTRIC
(LinearAngle internal coords).

| | value |
|---|---|
| **r(U=O)** | **1.67909 Å** (both bonds, symmetric to 7e-7 Å) |
| **O=U=O** | **180.000°** |
| steps / wall | 5 / 202 s |
| E(min) | −626.57224190 Eh; fresh ‖grad‖_max = 9.8e-7 |
| SCF aids | none, every step |

vs literature bare UO₂²⁺: CASPT2 ≈1.71 Å; hybrid DFT 1.68–1.70 Å.  PBE0/ECP60MWB
1.679 Å — within ~0.01 Å of hybrid-DFT, ~0.03 Å below CASPT2 (inside the
0.02–0.05 Å target).

## Phase 2 — SOC geometry

**Not runnable variationally** (headline 3).  The driver's `phase2` records the
stability probe rather than an optimisation:

| quantity | value |
|---|---|
| scalar RKS e_tot | −626.57224190 Eh |
| GKS+SOC e_tot, seeded, plain DIIS | −631.64956051 Eh (converged) |
| GKS+SOC e_tot, seeded, level_shift 1.0 / 0.5 | −631.64956051 Eh (both, converged) |
| GKS+SOC from `minao` | diverges (‖g‖ ~ 80, energy −360…−480 Eh) |
| Δ(GKS+SOC − scalar) | **−5.0773 Eh** (unphysical) |
| RS-PT2 E₂(SOC), full valence space | **−5.6928 Eh** |
| RS-PT2 E₂(SOC), frontier window (±0.4 Eh; 7 occ × 14 vir) | **−0.1019 Eh** (physical) |
| dominant full-space PT2 contributors | U 6p semicore, ε ≈ −8.1 Eh |

**Consequence for the intended deliverable.**  There is no variationally
meaningful GKS+SOC minimum, so `optimize(g.as_scanner())` cannot produce
r_soc(U=O) and Phase 3-SOC has no geometry to take a Hessian at.  The *physical*
SOC effect on uranyl is (i) exactly **zero at first order** (closed shell:
`Tr(D·h_SO) = 0` by spin symmetry) and (ii) small at second order
(≈ −0.1 Eh total; the SO-induced change in r(U=O) / ν₃ from a restricted-space
treatment is expected at the ~0.01 Å / ~1 % level from the literature, below
the FD-Hessian noise floor of this workflow).

## Phase 3 — scalar frequencies   [unchanged]

FD Hessian (`finite_diff_hessian`, disp 1e-3, `conv_tol` 1e-12) →
`pyscf.hessian.thermo.harmonic_analysis`, 721 s.

| mode | symmetry | this work | imag? |
|---|---|---:|:---:|
| ν₂ bend (×2) | πu | **180.5 cm⁻¹** | no |
| ν₁ sym str | σg⁺ | **1092.0 cm⁻¹** | no |
| ν₃ asym str | σu⁺ | **1183.3 cm⁻¹** | no |

No imaginary vibration → true minimum.  vs CASPT2 (Gagliardi et al. 2001)
≈1120 / ≈170 / ≈1260: ν₁ −3 %, ν₃ −6 % (inside the 5–10 % target); loose gates
(ν₃ ∈ 900–1300, |ν₃−ν₁| = 91 < 100, bend > 0) all met.

## Phase 3-SOC — SOC frequencies

**Blocked** (no stable variational GKS+SO-ECP state; headline 3 / Phase 2).

## Phase 4 — aqueous (C-PCM), scalar   [unchanged]

| quantity | gas | aqueous C-PCM | shift |
|---|---:|---:|---:|
| r(U=O) | 1.6791 Å | 1.6928 Å | +0.014 |
| ν₁ | 1092 | 1048 cm⁻¹ | −44 |
| ν₂ | 181 | 206 cm⁻¹ | +25 |
| ν₃ | 1183 | 1112 cm⁻¹ | −71 |

Correct trends (bond elongates, stretches soften, bend stiffens); continuum
alone recovers ~25–35 % of the measured gas→aqueous ν₃ drop — the balance needs
explicit equatorial waters.  Known model limitation, not a machinery problem.

## Verdict

**Scalar GKS + ECP machinery: VALIDATED on uranium** (geometry to ~0.03 Å of
CASPT2, ν₃ within 6 %, true minimum, no SCF aids; C-PCM composes).

**Actinide SO-ECP integrals: VALIDATED.**  `ecpds60mwbso` (Ac–Lr) is present in
vendored pyscf; gpu4pyscf reproduces `mol.intor('ECPso')` to ~1e-13 for Th–Am
including the f (l=3) and g (l=4) projectors, `get_soc_1e` to 4e-14, AREP
unchanged from `stuttgart_rsc`.  No kernel or angular-table change required.
Accessor + tests added (`gpu4pyscf/gto/actinide_ecp.py`,
`gto/tests/test_actinide_ecp.py`).

**Self-consistent 2-component actinide SOC: BLOCKED by an ECP limitation.**
Variational GKS + `ecpds60mwbso` on uranyl collapses ~5 Eh (the explicit,
strongly SO-split 6p semicore over-couples through the unbounded semilocal SO
projector; these Stuttgart sets are for restricted SO-CI, not variational 2c).
Confirmed *not* a gpu4pyscf bug (one-shot Fock == pyscf CPU to 1e-12; collapse
reproduces in RS-PT2).  No large-core actinide SO-ECP exists to avoid it, and
gpu4pyscf has no restricted-space / frozen-semicore SOC facility.  **The
scalar-vs-SO geometry/frequency comparison that defines Stage 2a Phase 2/3-SOC
therefore cannot be produced.**  First-order SOC on closed-shell uranyl is zero
by symmetry; a restricted-space second-order estimate is ≈ −0.1 Eh.

**Recommendation for actinide SOC in gpu4pyscf:** use a Dirac-fitted 2-component
ECP (dhf-type) or all-electron X2C-SOC for *variational* 2c work; the
Stuttgart SO-ECP integrals validated here are for *perturbative* /
restricted-active-space use.  (X2C-SOC nuclear gradients remain the open
SOC-gradient gap — `docs/adf-parity-strategy.md` §4.2.)

## SCF convergence aids

*Scalar:* none — every scalar SCF (single point, 5 geometry steps, ~19
FD-Hessian displacements, PCM re-opt + Hessian) converged from `minao` with no
level shift / SOSCF / damping.

*SOC:* a scalar-RKS density seed is **mandatory** (`minao` diverges); even then
the converged state is the −631.65 Eh collapse, not a physical minimum.  Level
shifting (0.2–1.0) does not recover a physical state; SOSCF is `NotImplemented`
for GKS.
