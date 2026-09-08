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
| Spin-orbit ECP | ✅ done this project (molecular + PBC energies); GKS+SO-ECP energies validated 2026-09-08 (`test_gks_x2c_soc.py`); **no gradients yet** |
| Spin-orbit X2C | ⚠️ **partial** — `x2c/x2c.py:SpinOrbitalX2CHelper` energies validated at **GHF and GKS** (molecular, 2026-09-08); no gradients; PBC X2C exists (`pbc/x2c/x2c1e.py`) but scalar-only, not checked for SOC |
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

## 1b. Scope locked in 2026-09-04

User confirmed: **periodic is separate materials work, not in scope for
ADF-replacement** (molecular only from here). EPR/ESR and full ETS-NOCV/QTAIM
are both needed. Accepted near-term target: **X2C-SOC energies only (no
gradient yet) + ECP-SOC gradients as the interim geometry-optimization path.**

Investigation findings that revise the plan below:

- **GKS + spin-orbit X2C energies is very likely already functional**, not new
  engineering. `gpu4pyscf.dft.gks.GKS(rks.KohnShamDFT, GHF)` — GKS *is-a* GHF
  — and `GHF.x2c1e()`/`.x2c()` (→ `x2c1e_ghf`) only asserts
  `isinstance(mf, ghf.GHF)`, which GKS satisfies; neither `gks.py` nor `rks.py`
  override `x2c`/`x2c1e`. The `X2C1E_GSCF` mixin only replaces `get_hcore`
  (with the 2-component X2C Hamiltonian, spin-free + spin-dependent terms
  folded in via `SpinOrbitalX2CHelper`) and leaves GKS's `get_veff`/XC
  machinery untouched. So `dft.gks.GKS(mol, xc=...).x2c1e()` should already
  give spin-orbit X2C-DFT. **Needs a validation test on Perlmutter, not a new
  implementation** — treat as unverified-but-likely-working until tested.
  Requires an all-electron relativistic basis (e.g. `x2c-tzvpall`,
  ANO-RCC-derived) for the actinide; not compatible with `mol.has_ecp()` by
  design (X2C is all-electron; ECP and X2C are alternatives, never combined).
- **No CPU pyscf oracle exists for GHF nuclear gradients** (`pyscf/grad/`
  has no `ghf.py`) **or for SO-ECP gradient integrals** (only the scalar
  `ECPso` energy `intor` exists, no derivative). But `pyscf/grad/dhf.py`
  (Dirac-HF gradient) shows the *general* pattern is standard and simple:
  hcore-derivative + JK-derivative + Pulay overlap term, real part of a
  complex spinor-density trace — GHF is actually a simplification of DHF (2c,
  no small-component/kinetic-balance term). The gap is that nobody wrote the
  GHF-specific driver, not that the theory is missing. Validation for both new
  pieces falls back to finite-difference (of the SCF energy for the GHF
  gradient driver; of the `get_ecp_so` integral matrix directly for the
  SO-ECP derivative kernel) — the same standard PySCF's own test suite uses
  when landing a genuinely new derivative.

Revised near-term sequence:
1. Validate GKS+X2C-SOC on an actinide test case (Perlmutter). If it doesn't
   already work, fix forward from there rather than building from scratch.
2. `grad/ghf.py` — GHF nuclear gradient (hcore/JK/overlap Pulay-force pattern,
   no SOC yet). Validate vs finite difference and vs RHF-gradient reduction on
   a closed-shell system (GHF without SOC should reduce to RHF).
3. SO-ECP gradient integrals (extend `lib/ecp/ecp_so.cu`, mirroring how
   `ecp_type2_ip.cu` extends `ecp_type2.cu` for the scalar case). Validate vs
   finite difference of `get_ecp_so`.
4. Wire 2+3 together: ECP-SOC contribution in `GHF.Gradients()` → the agreed
   interim optimization path. Validate full SCF+gradient vs finite difference
   of the GHF+ECP-SOC total energy.
5. EPR/ESR (g/A/D-tensor) and full ETS-NOCV/QTAIM — separate, large tracks;
   scope after 1-4 land, since geometry optimization is a workflow
   prerequisite for property/bonding-analysis calculations in practice.

## 2. Recommended path to "SOC during geometry optimization" (superseded by §1b above, kept for the original relativistic-method reasoning)

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

## 3. Prioritization answers (2026-09-04)

- Periodic is **separate materials work**, not replacing ADF+BAND — out of
  scope for this roadmap. (The already-done PBC-ECP work stands on its own.)
- EPR/ESR (g/A/D-tensors): **needed.**
- Full ETS-NOCV / QTAIM: **needed** (the simpler `properties/eda.py` is not
  enough).
- Near-term target **accepted**: X2C-SOC energies only (no gradient yet) +
  ECP-SOC gradients as the interim geometry-optimization path.

## 4. Actinide + ligand-separation workflow — feature gap list

Framing: solvated, often open-shell An(III/IV)–organic-extractant complexes
(TODGA, CMPO, BTP/BTBP-type). Deliverables: binding free energies, An/Ln
selectivity (ΔΔG), geometries, and covalency-based rationalization of the
selectivity. "ECP level" = small-core or f-in-core relativistic ECP
(Stuttgart RSC/MWB), the pragmatic route for 50–150-atom complexes (all-
electron X2C is too expensive at that size).

### 4.1 Usable today (scalar-relativistic ECP path)

Covers the core of most computational separation studies:

- UKS / ROKS with small-core or f-in-core RECP + the completed ECP
  integrals/screening/gradients/Hessian from this project.
- f-element SCF convergence aids: second-order SCF (`scf/soscf.py`,
  `mf.newton()`), Fermi/Gaussian smearing (`scf/smearing.py`),
  constrained-DFT + SOSCF (`dft/cdft_soscf.py`) for broken-symmetry /
  charge-localized states.
- Solvation: PCM + SMD, **with gradients and Hessian** (`solvent/grad`,
  `solvent/hessian`) → solvated geometry optimization + frequencies.
- Dispersion D3/D4 + gCP (`dispersion/`) — essential for second-sphere
  ligand binding and BTP/BTBP π-stacking.
- Thermochemistry: `pyscf.hessian.thermo` consumes the GPU Hessian (see
  `drivers/dft_3c_driver.py`) → **ΔG of complexation, An/Ln ΔΔG selectivity**
  — the primary deliverable.
- ESP/RESP/CHELPG charges, NMR shielding, IR/Raman, TDDFT (no SOC).
- SOC as a **single-point correction** on scalar-relativistic geometries
  (GHF + SO-ECP energies; `gto/ecp.py:get_soc_1e`).

Net: solvated ΔG / ΔΔG separation-energetics work can start now. SOC in the
optimization loop and the ADF-style covalency analysis are the gaps.

### 4.2 Tier 1 — blocks common workflows

| Gap | Impact | Effort |
|---|---|---|
| **SOC-level gradients** (ECP or X2C) | no SOC in the geometry-opt loop or SOC-ΔG. Design in `docs/ghf-gradient-design.md`; not built. | medium–large; needs finite-difference validation on a non-collinear case |
| ~~**GKS + SO-ECP / GKS + X2C-SOC validation**~~ | ✅ **DONE 2026-09-08.** Both routes validated vs CPU PySCF on A100 — `x2c/tests/test_gks_x2c_soc.py`, 7/7 (H₂O/cc-pvdz + I heavy-atom; `e_tot` to 1e-6–1e-7, `mo_energy` to 1e-4–1e-5). Confirmed the GKS-is-a-GHF inheritance path needs no GKS-specific SOC code. Bonus fix: `dft/gks.py:GKS.__init__` now defaults `collinear='mcol'` (the only 2-component XC scheme gpu4pyscf implements; plain `'col'` always raised). CPU cross-checks need `pip install mcfun`; the GPU path does not. | ~~small~~ done |
| **Hirshfeld / CM5 charges** | standard in the An/Ln separation literature; only ESP/RESP/CHELPG exist now. | small |

### 4.3 Tier 2 — ADF-signature covalency toolkit (needed, per Sec 3)

Separation *selectivity* is rationalized through An–L bond covalency
(5f vs 4f differentiation) — ADF's analysis tools are what do this:

| Gap | Notes |
|---|---|
| **ETS-NOCV** | fragment bond-energy decomposition (An³⁺ + ligand → complex) + NOCV deformation-density channels. The most-cited ADF analysis for f-element bonding. Only `properties/eda.py` (simpler) exists. Large new track. |
| **QTAIM** | bond-critical-point properties, delocalization indices at An–L bonds. Missing. Large new track (real-space density topology). |
| **Mayer / Wiberg bond orders** | not found in gpu4pyscf. Small. |
| **NBO / NPA** | external (NBO6/7); PySCF can write the interface file — usable now, not GPU/integrated. |

### 4.4 Tier 3 — advanced / situational

| Gap | Relevant if… |
|---|---|
| **EPR/ESR (g/A/D-tensor)** | paramagnetic An(III/IV) EPR is in the workflow. Not implemented anywhere in gpu4pyscf (nor this PySCF checkout's `pyscf.prop`). Large new track. |
| **SOC-TDDFT** | f–f / LMCT transitions with spin-orbit (UV-vis, luminescence). TDDFT exists, not with SOC. |
| **X2C actinide basis sets** | going all-electron X2C instead of ECP — SARC-DKH2 covers Z=90–103; `x2c-*` family coverage spottier. Non-issue for the ECP route. |
| **Multireference (CASSCF / NEVPT2)** | genuinely multiconfigurational An states (some U(IV)/U(V), bridged multi-metal). Scoped out (GKS+SOC deemed sufficient); CASSCF is CPU-only in PySCF, not GPU-accelerated here. |

### 4.5 Recommended order for separation-chemistry readiness

1. ~~Validate GKS+X2C-SOC and GKS+SO-ECP energies~~ — ✅ DONE 2026-09-08
   (merged to `gpu-porting`). Trustworthy DFT+SOC single points now unblocked.
2. Hirshfeld/CM5 charges — small, immediately useful for An/Ln analysis.
3. GHF/GKS + SO-ECP gradients (`docs/ghf-gradient-design.md`) — enables SOC
   geometry optimization and SOC-ΔG.
4. ETS-NOCV — the highest-value bonding-analysis gap for selectivity
   rationalization.
5. QTAIM.
6. EPR/ESR, SOC-TDDFT — as the specific spectroscopy needs arise.
