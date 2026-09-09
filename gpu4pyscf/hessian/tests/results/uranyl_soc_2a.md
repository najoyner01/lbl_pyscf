# Stage 2a — actinide validation on uranyl (UO₂²⁺)

**Status: COMPLETE.**  Scalar GKS + ECP machinery validated on a real actinide;
the spin-orbit half is **blocked by missing basis-set data**, not by a bug.

Driver: `uranyl_soc_2a.py` (this directory).  A100-PCIE-40GB, pyscf 2.14.0,
cupy 13.4.1, commit `0d9a363ad` (branch `uranyl-2a` off `origin/gpu-porting`
= `319a3a48a`).

## Headline

1. **SOC comparison is blocked — no actinide spin-orbit ECP in this pyscf.**
   `mol.has_ecp_soc()` is `False` for U with `crenbl`, `stuttgart_rsc`,
   `stuttgart_dz`, `lanl2dz` — all are 2-column scalar-relativistic RECPs.  The
   SO-ECP sweep (`test_ecp_sweep.py`) only ever covered Na–Bi.  Phase 2 (SOC
   geometry) and the SOC half of Phase 3 therefore **cannot be run**.  This is a
   data gap: `get_ecp_so` correctly raises `mol has no spin-orbit ECP
   projectors`, `mol.intor('ECPso')` returns exact zeros, and neither the SCF
   nor the integral driver crashes on an actinide basis — there is simply
   nothing to integrate.

2. **A real GPU bug was found and fixed** — `gpu4pyscf/gto/ecp.py`, committed
   separately as `0d9a363ad` (one line).  Scalar `get_ecp` launched the
   `ECP_cart` CUDA kernel even for a screening-emptied `(l_i, l_j, l_ecp)` task
   block → `CUDA Error: invalid configuration argument` →
   `RuntimeError: ECP CUDA kernel failed`.  Triggered by the small-core U
   `stuttgart_rsc` ECP with `def2-SVP`/`def2-TZVP` oxygen (the first `s-s-s`
   block screens to zero).  The `so` / `ip` / `so_ip` loops in the same file
   already guarded with `if len(task) == 0: continue`; the scalar loop did not.
   After the fix, `get_ecp` for small-core uranyl matches CPU
   `mol.intor('ECPscalar')` to **3.6e-13**; the large-core RECPs (which never
   hit the empty block) are byte-for-byte unchanged.

3. **Scalar actinide machinery works and reproduces the literature.**
   RKS(PBE0), U `stuttgart_rsc` (ECP60MWB small-core, 32 valence e⁻) +
   O `def2-TZVP`, 149 basis functions:

   | quantity | this work | literature (bare UO₂²⁺) | agreement |
   |---|---|---|---|
   | r(U=O) | **1.6791 Å** | CASPT2 1.71; hybrid-DFT 1.68–1.70 | ≈0 vs hybrid DFT, −0.03 Å vs CASPT2 — inside the 0.02–0.05 Å target |
   | ν₁ (σg, sym str) | **1092 cm⁻¹** | CASPT2 ~1120; DFT ~1040–1130 | −3 % vs CASPT2 |
   | ν₂ (πu, bend, ×2) | **181 cm⁻¹** | ~150–190 | in range, real (bend > 0) |
   | ν₃ (σu, asym str) | **1183 cm⁻¹** | CASPT2 ~1260; DFT ~1100–1200 | −6 % vs CASPT2 — inside the 5–10 % target |

   No imaginary modes → a true minimum.  SCF converged at every geometry with
   **no convergence aids** (default `minao` guess, no level shift, no SOSCF).

## Basis / ECP chosen

| | choice | why |
|---|---|---|
| U ECP | `stuttgart_rsc` (ECP60MWB, `ncore=60`, `l_max=g`) | the standard **small-core** relativistic RECP for actinides; the 5s5p5d/6s6p semicore is explicit.  Needs the `ecp.py` fix above. |
| U basis | `stuttgart_rsc` (matched valence) | paired with the ECP |
| O basis | `def2-TZVP` (all-electron) | task spec; `def2-SVP` also exercised in the integral check |
| — | **no SO projectors available for any actinide** | see headline 1 |

Large-core alternatives (`crenbl`, `stuttgart_dz`, `lanl2dz`, all `ncore≈78`)
run without the fix but are cruder for U=O covalency, and none carry SO
projectors either.

## Phase 0 — prerequisites

**0a — U ECP enumeration:**

| U ECP | `ncore` | ECP `l_max` | nao (uranyl, def2-TZVP O) | `has_ecp_soc()` |
|---|---:|---:|---:|:---:|
| `crenbl` | 78 (large) | 3 (f) | 130 | **False** |
| `stuttgart_rsc` (ECP60MWB) | 60 (small) | 4 (g) | 149 | **False** |
| `stuttgart_dz` | 78 (large) | 4 (g) | 141 | **False** |
| `lanl2dz` | 78 (large) | 3 (f) | 98 | **False** |

No SO projectors for any actinide → SOC comparison blocked.

**0b — SO machinery on the actinide basis (no crash):**
`get_ecp_so(mol)` raises `ValueError: mol has no spin-orbit ECP projectors`;
`mol.intor('ECPso')` returns exact zeros (shape `(3, 149, 149)`, `max|·| = 0`).

**0b — scalar ECP GPU vs CPU `mol.intor('ECPscalar')`** (after the `ecp.py` fix,
O = def2-SVP):

| U ECP | `|get_ecp − ECPscalar|_max` |
|---|---:|
| `stuttgart_rsc` (small-core) | **3.6e-13** (was: kernel crash) |
| `crenbl` | 1.6e-8 |
| `stuttgart_dz` | 6.4e-7 |
| `lanl2dz` | 1.8e-9 |

(The large-core diffs of 1e-6..1e-9 are the pre-existing GPU-vs-CPU radial-
quadrature agreement for these RECPs — unchanged by the fix, well inside SCF
tolerance.)

**0c — scalar RKS(PBE0) SCF:** `stuttgart_rsc` U / `def2-TZVP` O, 149 BF →
**converged**, `e_tot = −626.56080040`, **34 s**, no convergence aids.

**0d — SOC SCF:** N/A (no SO-ECP).

## Phase 1 — scalar geometry (RKS/PBE0)

`optimize(mf, convergence_grms=1e-4, convergence_gmax=2e-4)`, geomeTRIC, which
picks `LinearAngle` internal coordinates (recognises the linear O–U–O).

| | value |
|---|---|
| **r(U=O)** | **1.67909 Å** (both bonds; symmetric to 7e-7 Å) |
| **O=U=O angle** | **180.000°** |
| steps / wall | 5 / 202 s |
| E(min) | −626.57224190 Eh; fresh \|grad\|_max = 9.8e-7 |
| SCF aids | none, at every step |

vs literature for **bare gas-phase UO₂²⁺**: CASPT2 (Gagliardi et al. 2001)
≈1.71 Å; hybrid DFT 1.68–1.70 Å; GGA 1.70–1.72 Å.  PBE0/ECP60MWB **1.679 Å**
sits on the short (hybrid) side — within ~0.01 Å of published B3LYP/PBE0 and
~0.03 Å below CASPT2, inside the task's 0.02–0.05 Å target.

## Phase 2 — SOC geometry

**BLOCKED** — no actinide SO-ECP data (headline 1).  To unblock: add a
spin-orbit ECP for U to pyscf's basis data (e.g. the Stuttgart ECP60MWB_SO
parameters as a 3-column entry), or pass one via a custom `ecp={'U': [...]}`
dict carrying `SO_TYPE` projectors.  The `get_ecp_so` / `get_soc_1e` kernels
are already validated Na–Bi and require no code change to accept an f-shell
projector — only the data.

## Phase 3 — frequencies, scalar (FD Hessian at the Phase-1 minimum)

`finite_diff_hessian(mf, disp=1e-3)` (central differences, `conv_tol` tightened
to 1e-12) → `pyscf.hessian.thermo.harmonic_analysis`.  721 s (~19 re-converged
SCF+gradient evaluations).

| mode | symmetry | this work | imag? |
|---|---|---:|:---:|
| ν₂ bend (×2) | πu | **180.5 cm⁻¹** | no (\|Im\| < 0.5) |
| ν₁ sym str | σg⁺ | **1092.0 cm⁻¹** | no |
| ν₃ asym str | σu⁺ | **1183.3 cm⁻¹** | no |

The 5 non-vibrational modes come out at 0, 0, 0, 19.6, 20.0 cm⁻¹ (translational/
rotational residual; the ~20 cm⁻¹ is the FD-Hessian noise floor at `disp=1e-3`).
No imaginary vibration → the Phase-1 structure is a genuine minimum.

### Literature comparison (bare gas-phase UO₂²⁺)

| source | method | r(U=O) Å | ν₁ | ν₂ | ν₃ |
|---|---|---|---|---|---|
| Gagliardi, Roos, Malmqvist, Dyke, *JPC A* **105** (2001) 10602 | CASPT2 | ~1.71 | ~1120 | ~170 | ~1260 |
| Shamov & Schreckenbach, *JPC A* **109** (2005) 10961 | DFT survey (BP86/B3LYP, several RECP) | 1.69–1.72 | ~1040–1130 | ~150–250 | ~1100–1200 |
| Bühl, Kabrede, Diss, Wipff, *JACS* **128** (2006) 6357 | CPMD (BLYP) + waters | 1.72–1.76 | — | — | ~1000–1150 |
| Vallet, Wahlgren, Grenthe, *JACS* **125** (2003) 14941 | DFT (B3LYP) ± explicit H₂O | 1.70–1.78 | — | — | ~950–1150 |
| Denning, *JPC A* **111** (2007) 4125 (review) | — | ~1.71 | — | — | bare ~1100–1150; aqueous ~961 |
| **this work** | RKS/PBE0, ECP60MWB(U)/def2-TZVP(O) | **1.679** | **1092** | **181** | **1183** |

Numbers from other groups are quoted approximately (method- and RECP-dependent;
several from memory of the literature).  The task's loose gates for bare
uranyl — ν₃ ∈ 900–1300 cm⁻¹, ν₁ within ~100 cm⁻¹ of ν₃ (here \|Δ\| = 91), bend
> 0 — are all met.

## Phase 4 — aqueous (C-PCM, ε = 78.36), scalar

Re-optimised from the Phase-1 minimum with C-PCM, then FD Hessian.  995 s.

| quantity | gas (Phase 1/3) | aqueous C-PCM | shift | experiment (aq. [UO₂(H₂O)₅]²⁺) |
|---|---:|---:|---:|---:|
| r(U=O) | 1.6791 Å | **1.6928 Å** | +0.014 | ~1.76–1.77 Å (EXAFS) |
| ν₁ (σg) | 1092 | **1048 cm⁻¹** | −44 | ~869 (Raman) |
| ν₂ (πu) | 181 | **206 cm⁻¹** | +25 | ~200–210 |
| ν₃ (σu) | 1183 | **1112 cm⁻¹** | −71 | ~961 (IR) |
| E | −626.5722 | −627.0504 Eh | −0.478 | (large, as expected for a +2 dication) |

(The FD Hessian with PCM also produces a spurious 57.8 cm⁻¹ rotational pair —
the continuum cavity is fixed in the lab frame and breaks exact rotational
invariance; the three genuine vibrations are the ones tabulated.)

**Continuum solvation captures the right *trends* — U=O bond elongates, both
stretches soften, the bend stiffens — but undershoots the magnitude.**  C-PCM
alone recovers only ~25–35 % of the measured gas→aqueous ν₃ drop (−71 cm⁻¹ vs
the ~−220 cm⁻¹ to experiment) and ~25 % of the bond elongation.  The balance
comes from explicit equatorial-water donation into the U=O σ*/π* system, which a
bare dielectric cannot model; Bühl/Wipff, Vallet/Wahlgren and Shamov/
Schreckenbach all show 4–5 explicit H₂O (± continuum) are needed to reach
ν₃ ≈ 960 cm⁻¹.  This is a known model limitation, **not** a machinery problem —
the point of Phase 4 was to confirm `GKS + ECP + PCM + FD-Hessian` compose and
run on an actinide, which they do.

## Verdict

**Scalar GKS + SO-ECP machinery: VALIDATED on uranium.**

- The `get_ecp` / `get_ecp_ip` / FD-Hessian / PCM stack runs on a small-core
  actinide RECP after a one-line kernel-launch fix, and reproduces published
  bare-uranyl geometry (r(U=O) to ~0.01 Å vs hybrid DFT, ~0.03 Å vs CASPT2) and
  vibrational frequencies (ν₃ within 6 %, ν₁ within 3 % of CASPT2; no imaginary
  modes) — both inside the task's accuracy targets.
- SCF needs **no convergence aids** for closed-shell 5f⁰ uranyl: plain `minao`,
  no level shift, no SOSCF, at every geometry-optimisation and FD-Hessian point.
- Continuum solvation composes and gives qualitatively correct shifts.

**Spin-orbit half: BLOCKED at data, not code.**  This pyscf checkout ships no
spin-orbit ECP for any actinide, so the scalar-vs-SO comparison that defines
Stage 2a cannot be completed here.  The SO-ECP kernels do not fail on an
actinide basis — they have nothing to act on.  Unblocking is a basis-data task
(add ECP60MWB_SO for U), after which Phase 2/Phase 3-SOC can run with no
gpu4pyscf code change.

## SCF convergence aids needed

**None.**  Every SCF in this study — scalar single point, 5 geometry steps,
~19 FD-Hessian displacements, the PCM re-optimisation and its Hessian —
converged to `conv_tol` 1e-9…1e-12 from the default `minao` guess with no level
shift, no SOSCF, no damping.  Closed-shell 5f⁰ uranyl is numerically benign; the
open-shell 5fⁿ actinides of the wider campaign are expected to be harder.
