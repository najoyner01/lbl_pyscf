# Uranyl campaign — all-electron X2C-SOC go/no-go

**Status: COMPLETE.**  Verdict: **conditional GO** for the X2C-SOC analytic
gradient — the 2c X2C-SOC SCF is variationally sound on actinides, GPU-faithful,
and cheap; bare X2C-1e over-estimates SOC energies (add SNSO/AMFI for
quantitative work) but the geometry-relevant SOC force is clean.

Driver: `uranyl_x2c_soc.py` (this directory).  A100-PCIE-40GB, pyscf 2.14.0,
cupy 13.4.1, branch `uranyl-x2c-soc` off `origin/gpu-porting` (= `319a3a48a`).

Fixed geometry: r(U=O) = 1.6791 Å (Stage 2a scalar-ECP minimum), linear D∞h;
PBE0; `spin_samples = 50`; all-electron ANO-RCC-VDZP.

## Why this study

2c-GKS + the actinide **SO-ECP** `ecpds60mwbso` **variationally collapses** ~5 Eh
on uranyl (`uranyl_soc_2a.md`).  All-electron **X2C** builds a variationally
*bounded* 2-component operator — the candidate route for actinide SOC.  This is
the go/no-go for an **X2C-SOC analytic gradient** project.

## Phase 0 — basis / build

`x2c-SVPall`/`x2c-TZVPall` have **no U/Th** in this pyscf.  `ANO-RCC-VDZP` (Roos
relativistic ANO) covers U, Th, O and is compact — **used** for both U and O.

| system | `has_ecp` | n(elec) | nao (contracted) | nao (primitive) |
|---|:-:|---:|---:|---:|
| UO₂²⁺ | False | 106 | 112 | 438 |
| ThO   | False | 98  | 98  | 402 |

X2C decoupling runs at the primitive dimension: 2 × 438 = 876 (complex) for uranyl.

## Phase 1 — uranyl GKS + X2C-SOC SCF

| quantity | value |
|---|---|
| non-relativistic RKS `e_tot` | −21169.26199 Eh |
| **X2C-SOC GKS `e_tot`** | **−28121.32271 Eh** |
| converged from **`minao`, no aids** | **yes** — identical energy from an RKS-seeded 2c DM |
| X2C `get_hcore` build | **0.46 s** (GPU) |
| SCF wall (1e-9) | 95 s;  tight 1e-11 → 165 s (plateaus ~1e-10) |
| U core MO energies (Eh) | 1s −4255.45, 2s −797.80, **2p₁/₂ −750.88, 2p₃/₂ −623.58** |

**No variational collapse.**  Bounded, converges from the trivial guess — the
exact contrast with the SO-ECP.  The U core MO spectrum shows a
**2p₁/₂–2p₃/₂ splitting ≈ 127 Eh**, comparable to the ≈139 Eh U L₂–L₃ X-ray
separation — the spin-orbit operator is active and roughly the right size at the
core.

## Phase 2 — cross-checks

| check | result |
|---|---|
| **2.1** X2C `get_hcore` GPU vs CPU (pyscf) | **5.6e-8 relative** (abs 2.7e-4 on a ~4900 Eh U-1s element); GPU 0.2 s / CPU 2.7 s |
| **2.3** reduction: 2c GKS fed the spin-free X2C `hcore` vs 1c RKS `sfx2c1e()` | **|Δ| = 2.2e-11** — the GKS 2c SCF reduces to the scalar 1c result exactly |
| **2.2** E(X2C-1e 2c) − E(spin-free X2C-1e) | **+29.18 Eh (+794 eV)** |
| **2.2 within-scheme** E(X2C-1e 2c, SO `w` **on**) − E(same, SO `w` **off**) | **+29.18 Eh** — and "SO off" = `sfx2c1e()` to 6e-10, so this *is* the bare X2C-1e SO contribution, no scheme artifact |
| **2.1 (SCF)** GPU vs CPU pyscf full X2C-SOC SCF `e_tot` | GPU −28121.3227145 / CPU −28121.3227101 → **\|Δ\| = 4.4e-6 Eh** (rel 1.6e-10); CPU SCF 53 s |
| CPU pyscf E(X2C-2c) − E(sfX2C-1e) | **+29.1792 Eh** — identical to GPU; the +29 Eh is pyscf's bare X2C-1e, reproduced by gpu4pyscf, *not* a GPU bug |

**The X2C-1e spin-orbit contribution to the uranyl total energy is +29 Eh
(destabilising) and dominated by the U core.**  This is **bare one-electron
X2C-1e** — no two-electron spin-orbit / SNSO (Boettger) / AMFI screening.  The
2e-SO interaction a production treatment adds partially cancels the 1e-SO
(typically 10–60 % for splittings), so bare X2C-1e is not a quantitative SOC
energy for a heavy element — but it is a well-defined operator, reproduced
exactly by pyscf CPU (Phase 2.1), converged variationally.  The +29 Eh is a
core effect (U 1s–3d), essentially geometry-independent — Phase 4 shows it
cancels in the U=O bond-coordinate derivative.

## Phase 3 — ThO corroboration

ThO (closed-shell ¹Σ⁺, Th Z=90, r = 1.84 Å), ANO-RCC-VDZP, 98 AO / 402 primitive.

| quantity | value |
|---|---|
| GPU X2C-SOC `e_tot` | −26515.56681 Eh (conv, `get_hcore` ~0 s, SCF 84 s) |
| CPU pyscf X2C-SOC `e_tot` | −26515.56681 Eh — **\|Δe\| = 2.0e-6**, **\|Δ mo_energy\| = 5.5e-5** |
| spin-free X2C-1e RKS | −26541.20417 Eh |
| E(X2C-2c) − E(sfX2C-1e) | **+25.64 Eh (+698 eV)** — same bare-X2C-1e overestimate, slightly below U's +29 Eh (Z-scaling) |

The actinide X2C-SOC path is **not uranyl-specific**: converges cleanly, GPU
matches CPU, same bare-X2C-1e SOC-energy behaviour.

## Phase 4 — X2C-SOC PES scan

X2C-SOC and spin-free X2C single points, r(U=O) = 1.66 / 1.68 / 1.70 Å:

| r (Å) | X2C-SOC `e_tot` (Eh) | spin-free X2C `e_tot` (Eh) | E_SOC = E(2c) − E(sf) (Eh) |
|---|---|---|---|
| 1.66 | −28121.323638 | −28150.500804 | **+29.177166** |
| 1.68 | −28121.322624 | −28150.501943 | **+29.179319** |
| 1.70 | −28121.319585 | −28150.501010 | **+29.181425** |

Both PESs are **smooth** (spin-free curvature ≈ 5.2 Eh/Å²).  E_SOC(r) is
essentially **linear**: +29.18 Eh offset (core, geometry-independent to 99.6 %)
plus a slope of **d E_SOC/dr ≈ +0.107 Eh/Å** — the physically meaningful part.

Minima: spin-free X2C ≈ **1.681 Å**; X2C-SOC ≈ **1.660 Å**.  So bare X2C-1e SOC
**contracts r(U=O) by ≈ 0.021 Å** — correct sign (SOC is known to contract the
uranyl U=O bond) and the right order of magnitude (literature ~0.01–0.02 Å; the
slight over-shoot is consistent with bare X2C-1e over-estimating SOC).  The huge
core term cancels in the bond-coordinate derivative, leaving a clean, sensible
SOC force — exactly what an analytic gradient must reproduce.

## Verdict — X2C-SOC analytic-gradient project: **conditional GO**

**GO on the machinery.**  All-electron X2C-SOC on an actinide:
- **converges cleanly from `minao`, no aids** — uranyl and ThO — to a bounded,
  physical energy.  The exact contrast with the SO-ECP, which diverges from
  `minao` and collapses ~5 Eh even when seeded.
- **GPU reproduces CPU pyscf**: `e_tot` |Δ| = 4.4e-6 (uranyl) / 2.0e-6 (ThO),
  `mo_energy` |Δ| ≤ 5.5e-5, X2C `get_hcore` 5.6e-8 relative.  First actinide
  check of the GPU X2C path — passes.
- **cheap**: X2C `get_hcore` is **0.46 s** and mostly GPU (only `int1e_spnucsp`
  on CPU pyscf); the SCF is ~1–2 min at 112 AO.  An FD-validated gradient
  (~6N ≈ 54 SCF+`get_hcore` evaluations for uranyl) is hours, not days.
- **PES is smooth** and the SOC effect on geometry is physical: E_SOC(r) is
  linear with slope +0.107 Eh/Å, the +29 Eh core term cancels in the derivative,
  and SOC contracts r(U=O) by ≈0.021 Å (right sign, right magnitude).  A
  gradient has a well-behaved quantity to reproduce.

**Condition: bare X2C-1e is not a quantitative SOC energy for actinides.**  The
SOC contribution is +29 Eh (uranyl) / +26 Eh (ThO), *destabilising* and
core-dominated — **bare one-electron X2C-1e** with no two-electron SO / SNSO /
AMFI screening.  It is faithfully reproduced by pyscf CPU (identical +29 Eh),
**not a gpu4pyscf bug**.  A production actinide-SOC path must add the
screened-nuclear-SO (SNSO / Boettger) factor or atomic mean-field (AMFI) SO
integrals — a well-defined, separable addition on top of the working X2C
machinery, and itself a candidate follow-up.  For *relative* quantities where
the core SOC cancels (bond lengths, frequencies, ΔG) bare X2C-1e already gives a
sensible SOC effect (Phase 4: correct-sign ≈0.02 Å bond contraction).

**Bottom line:** pursue the X2C-SOC analytic gradient.  The 2-component X2C-SOC
SCF is variationally sound on actinides, GPU-faithful, and cheap; the gradient's
new work is the decoupling-response derivation plus GPU `pVp`/`pVxp` derivative
integrals (currently `int1e_spnucsp` is a CPU pyscf call — ~2.7 s once, but its
*derivative* would want a GPU kernel for larger systems).  Plan SNSO/AMFI as a
parallel workstream for quantitative accuracy.
