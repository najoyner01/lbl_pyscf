# Replacing ADF with gpu4pyscf — capability gap analysis & roadmap

Status: 2026-09-04. Motivation: actinide-series research (periodic materials +
molecular complexes), currently done in ADF, to be replaced by a GPU-accelerated
gpu4pyscf workflow with "as much functionality as possible."

## 0. Framing — ADF's technology is not gpu4pyscf's technology

ADF is a **molecular** DFT suite (periodic systems are a separate SCM program,
BAND) built on **Slater-type orbitals (STO)** with **ZORA** as its default
relativistic treatment (scalar and spin-orbit variants), plus X2C/DKH as
alternatives. gpu4pyscf is **Gaussian-type orbital (GTO)**, and already covers
both molecular and periodic (`gpu4pyscf/pbc/`).

"Replace ADF" therefore means **matching capability, not numerics**:
- STO vs GTO: a basis-technology difference, not a gap to close. GTO basis sets
  (def2, ANO-RCC-derived, etc.) reach comparable accuracy; no action needed.
- ZORA has no GTO/pyscf equivalent. The two available relativistic routes are:
  - **ECP** — core electrons replaced by a pseudopotential. Cheap, mature in
    this codebase (this project's whole ECP effort — screening, gradients,
    Hessian, spin-orbit, PBC — is now GPU-complete and validated).
  - **X2C** — exact two-component, all-electron. The closer numerical analog
    to ADF's ZORA(+SO); more expensive (all electrons); this is the route that
    matters most for matching ADF-level accuracy on actinides.
  Both are legitimate; ECP is the pragmatic/cheap default, X2C is the
  high-accuracy one. Neither should be discarded — but **X2C, not ECP+GHF, is
  the right target for "SOC during optimization" if ADF-level accuracy is the
  bar**, because that's what ADF itself is doing under the hood.

## 1. Current gpu4pyscf inventory vs ADF features

| ADF capability | gpu4pyscf status |
|---|---|
| DFT (LDA/GGA/meta-GGA/hybrid/range-separated) | ✅ `dft/` — full libxc coverage, RKS/UKS/GKS, PBC RKS/UKS |
| Scalar-relativistic ECP | ✅ done this project (screening, grad, Hessian, molecular + PBC) |
| Spin-orbit ECP | ✅ done this project (molecular + PBC energies); **no gradients yet** |
| Spin-orbit X2C | ⚠️ **partial** — `x2c/x2c.py:SpinOrbitalX2CHelper` wired to **GHF only** (energy); not wired to GKS/DFT; no gradients; PBC X2C exists (`pbc/x2c/x2c1e.py`) but scalar-only, not checked for SOC |
| Geometry optimization / TS search | ✅ geomeTRIC (molecular), ASE-based cell optimizer (PBC) |
| Frequencies (Hessian) | ✅ `hessian/` — RHF/UHF/RKS/UKS, molecular; PBC coverage partial |
| TDDFT / excited states | ✅ `tdscf/` incl. a RIS (reduced-scaling) variant, spin-flip |
| NMR shielding | ✅ `properties/shielding.py` |
| EPR/ESR (g-tensor, A-tensor, D-tensor) | ❌ **not found** — real gap if magnetic-resonance properties matter for open-shell actinide work |
| IR / Raman | ✅ `properties/ir.py`, `properties/raman.py` |
| Polarizability | ✅ `properties/polarizability.py` |
| Dispersion (D3/D4) | ✅ `dispersion/` |
| Solvation: COSMO/implicit | ✅ PCM + SMD (`solvent/`) — not literally COSMO but same class of model |
| Solvation: FDE, 3D-RISM | ❌ not found |
| Bonding analysis: EDA / ETS-NOCV | ⚠️ **partial** — `properties/eda.py` exists but is a simpler decomposition, not full ETS-NOCV |
| QTAIM | ❌ not found |
| Population analysis (Mulliken/ESP) | ✅ `pop/esp.py`; Mulliken exists on the PySCF CPU side (inherited) |
| QM/MM | ✅ `qmmm/` |
| Nonadiabatic dynamics / surface hopping | ✅ `md/` (Tully FSSH) — **beyond** ADF's core scope, a plus |
| Periodic DFT (ADF's BAND) | ✅ `pbc/` — DF, k-points, gradients+stress, now periodic ECP (scalar+SO); PBC X2C partial |

**Bottom line: coverage is already broad.** The concrete, actinide-relevant
gaps are narrower than "replace ADF" sounds:

1. **X2C spin-orbit not wired to DFT (GKS)** — only to GHF. Real work needed
   before DFT+SOC geometry optimization is possible via X2C.
2. **No gradients for any spin-orbit method** (ECP-SOC or X2C-SOC), molecular
   or periodic. This blocks "SOC during optimization" regardless of which
   relativistic route is used.
3. **No EPR/ESR properties.** Only matters if magnetic-resonance spectroscopy
   is part of the workflow (plausible for paramagnetic actinide species — worth
   confirming).
4. ETS-NOCV / QTAIM / FDE — bonding-analysis tools some ADF users lean on
   heavily; only relevant if your workflow uses them.

## 2. Recommended path to "SOC during geometry optimization"

Revising the earlier plan (which targeted ECP+GHF gradients specifically):

**Phase A — GKS + spin-orbit X2C (energies).** Wire `SpinOrbitalX2CHelper`
into `dft/gks.py` (mirror `x2c1e_ghf` → an `x2c1e_gks`), validate SCF energies
vs PySCF CPU X2C-GKS. This is the direct ADF-ZORA-SOC-DFT analog and the
biggest accuracy lever for actinides. ECP+GHF-SOC (already done) remains
available as the cheap alternative / cross-check.

**Phase B — X2C-SOC gradients (molecular).** Nuclear gradient of the X2C
Hamiltonian is more involved than a normal GTO gradient (differentiate the
X2C decoupling transformation itself) — genuinely new engineering, this is
the long pole. PySCF CPU has `pyscf.grad.x2c` (spin-free) as a partial
porting reference; spin-orbit X2C gradients may not exist anywhere in PySCF
either, worth checking before committing to a timeline.

**Phase C — decide on periodic.** Since ADF-proper is molecular-only, confirm
whether periodic actinide work is (a) required to replace an ADF+BAND
combination, or (b) a separate materials-science need. If needed: PBC X2C-SOC
+ PBC GKS-SOC + PBC gradients/stress — a large stack building on the
(already-done) PBC ECP-SOC pattern from this project.

**ECP-SOC gradients** (the original ask) remain worth doing regardless — cheap
relative to X2C, useful cross-check, and this project already has all the
non-SOC ECP gradient infrastructure to extend. Lower accuracy ceiling than
X2C for actinides, but much less new engineering. Could be phase A' in
parallel with X2C work as the "good enough, ships sooner" path.

## 3. Open questions for prioritization

- Is periodic work actually replacing ADF+BAND, or a separate need? Changes
  whether phase C is in scope now.
- Does the workflow need EPR/ESR (g/A/D-tensors)? Not started; would be new
  scope on top of this roadmap.
- Is ETS-NOCV / QTAIM-level bonding analysis load-bearing, or is the simpler
  `properties/eda.py` enough?
- Given X2C-SOC gradients are genuinely open-ended (may not exist in any GTO
  code yet), is ECP-SOC-gradients-as-interim-solution + X2C-SOC-energies (no
  gradients yet, single-point/spectroscopy use) an acceptable near-term
  target, while X2C gradients are scoped separately?
