# X2C-SOC analytic nuclear gradient — validation (atomic approximation)

Analytic `d h_X2C / dR` for `GHF/GKS + mf.x2c1e()` under the **atomic (local)
X2C approximation** (`with_x2c.approx = 'atom1e'`), **one-electron spin-orbit
only**.  Unblocks variational all-electron actinide SOC geometry optimization —
the SO-ECP route is dead-ended by variational collapse (`uranyl_soc_2a.md`); A1
(`uranyl_x2c_soc.md`) gave the conditional GO.

New code: `gpu4pyscf/grad/x2c.py`; one branch in `grad/ghf.py:grad_elec`;
`X2C1E_GSCF.nuc_grad_method` / `Gradients` in `x2c/x2c.py`.
Tests: `gpu4pyscf/grad/tests/test_x2c_soc_grad.py`.

Env: A100 (Perlmutter GPU node), CUDA 12.8, cupy 13.4.1, pyscf 2.14.0, `mcfun`,
`geometric`.  `conv_tol` 1e-10…1e-11, central FD, `spin_samples = 50` for GKS.

## Method

Structural port of `pyscf/x2c/sfx2c1e_grad.py` (`gen_sf_hfw`; JCP 135, 084114
(2011)) from the spin-free scalar case to the spin-orbital (complex,
`σ·pVp`) case.  Half-transformed form:

    h_X2C = c_fw0† h0 c_fw0 ,   c_fw0 = [ R0 ; X0 R0 ] ,
    h0    = [[ V, T ], [ T, W/4c² − T ]] ,   W = σ·(pVp)

with `X0` block-atomic and geometry independent in the atomic approximation
(`dX0/dR = 0`), so

    dh = dc_fw0† h0 c_fw0  +  h.c.  +  c_fw0† dh0 c_fw0

and `dc_fw0` needs only `dR0`, obtained in closed form from `_get_r1` (the
spectral divided-difference derivative of `_get_r`, generalised to Hermitian).
**No CPHF / Sylvester response solve** — the molecular decoupling response is a
separate project and is out of scope; `hcore_deriv_generator` raises on any
non-`atom1e` approx.  `pVp` / `pVxp` derivative integrals (`int1e_ipspnucsp`,
`int1e_ipsprinvsp`) evaluated with CPU pyscf.

**A spin-free X2C gradient template exists** — `pyscf/x2c/sfx2c1e_grad.py`
(`hcore_grad_generator`), fully wired into `mol.RHF().sfx2c1e().Gradients()`.
V1 is a hard reduction check against it, not a finite difference.

## Results

### V1 — spin-free reduction (SO term of W zeroed) vs pyscf `sfx2c1e_grad`

| system | ‖ analytic(SO off) − pyscf `sfx2c1e_grad` ‖ |
|---|---|
| HCl / 6-31g (αα & ββ blocks; αβ ≡ 0) | **3.4e-14** |
| uranyl / ANO-RCC-VDZP, r = 1.66 Å | **1.3e-10** |
| uranyl / ANO-RCC-VDZP, r = 1.70 Å | **1.6e-10** |

The full linear-algebra recipe (`_get_r1`, `c_fw1`, block assembly, contraction)
is exact against the reference; the SO term is the only addition, and it holds
on the decontracted actinide basis.

### V2 — analytic gradient vs central finite difference of `e_tot`

| SCF | system | geometry | analytic dE/dr | ‖ analytic − FD ‖ |
|---|---|---|---|---|
| GHF | HCl / sto-3g | r = 2.4 bohr (≈ rₑ) | +0.0385 Eh/bohr | **5.1e-9** |
| GHF | HCl / sto-3g | r = 3.0 bohr (displaced) | −0.1184 | **3.4e-9** |
| GHF | HCl / sto-3g | r = 3.6 bohr (displaced) | −0.1369 | **6.6e-9** |
| GKS/pbe0 | HCl / sto-3g | r = 2.4 bohr | — | **2.6e-7** |
| GKS/pbe0 | HI / ANO-RCC-VDZP | r = 3.05 bohr (≈ rₑ) | — | **2.7e-6** |
| GHF | **uranyl** / ANO-RCC-VDZP | **r = 1.72 Å (displaced)** | **+0.2725 Eh/bohr** | **4.4e-5** (rel 1.6e-4, FD-h-limited) |

The GHF-uranyl point is the decisive composition check: pure Hartree-Fock has
**no XC grid**, so the FD of `e_tot` is trustworthy to ~1e-8.  At a strongly
displaced geometry with a large gradient the analytic X2C-SOC hcore derivative
+ overlap Pulay + J/K compose correctly for the actinide.
(`dE/dr(U=O)` here is the symmetric-stretch derivative `g[O₁,z] − g[O₂,z]`.)

### V2u — uranyl `d h_X2C/dR` vs central FD of the forward `get_hcore`

Pure one-electron operator — no SCF, no XC grid, no mcfun quadrature noise.

| geometry | worst ‖ analytic − FD(get_hcore) ‖ | rel |
|---|---|---|
| uranyl, r = 1.66 Å | **6.5e-7** | 4.3e-8 |
| uranyl, r = 1.70 Å | **2.2e-7** | 1.5e-8 |

FD-truncation-limited at both geometries — the hcore-derivative matrix itself is
exact on the actinide, at and away from the minimum.

### V3 — uranyl dE/dr(U=O), GKS/pbe0, `atom1e`, seeded SCF

* **Analytic gradient zero-crossing:** r = **1.6642 Å** (A1 X2C-SOC minimum
  ≈ 1.66; V5 optimizer independently → 1.6642 Å).
* **Analytic `dE_SOC/dr`** (analytic total − numerical spin-free) at r = 1.66:
  **+0.110 Eh/Å** — reproduces A1's numerical **+0.107 Eh/Å**.
* The geometry-independent +29 Eh core SOC term differentiates to ≈ 0:
  `d/dr (E_soc − E_sf)` is **+0.088 Eh/Å, constant** across r = 1.66/1.68/1.70
  (linear, as A1 found) — the analytic force is the ~0.1 Eh/Å physical part, not
  the core term.
* **Analytic total dE/dr vs FD of `e_tot`:** the GKS PES curvature from the
  analytic gradient is ~4× weaker than from FD of `e_tot`
  (analytic k ≈ 1.4 vs FD k ≈ 5.5 Eh/Å²), the gap growing with displacement.
  This is **not** the X2C hcore-derivative term:
  - V2u: `d h_X2C/dR` is FD-exact on uranyl at r = 1.66 **and** 1.70;
  - V1: `with_soc=False` == pyscf `sfx2c1e_grad` to 1.3e-10 on uranyl;
  - **GHF** uranyl (no XC): analytic == FD to 4.4e-5 at displaced r = 1.72;
  - unchanged by `grid_response=True`; not an FD step-size artefact
    (stable for h = 1e-3 … 1e-2 Å).

  It is the pre-existing multi-collinear **mcfun XC nuclear gradient**
  (`grad/gks.py:_gks_xc_grad`) — no CPU reference, validated only by the
  collinear reduction and small-molecule FD; the uranyl 2c density is genuinely
  non-collinear.  Outside this task's scope ("the only new term is the hcore
  derivative"); a fix belongs in `_gks_xc_grad`.

### V4 — regression

Plain (non-X2C) GHF gradient path unchanged (HF / sto-3g, analytic vs FD
< 1e-6).  `x2c/tests/test_gks_x2c_soc.py` (energies) and the existing
`grad/tests/test_{ghf,gks}_grad.py` are untouched by the `with_x2c` branch.

### V5 — end-to-end geometry optimization (uranyl)

`optimize(mf.x2c1e().nuc_grad_method().as_scanner(), maxsteps=20)`,
`g.grid_response = True`, SCF seeded from a converged non-relativistic RKS 2c
density.  Start r(U=O) = 1.72 Å.

* geomeTRIC **converged in 4 steps** → **r(U=O) = 1.6642 Å** (A1 X2C-SOC
  minimum ≈ 1.66; SOC contracts the U=O bond, right sign and magnitude).
* final ‖grad‖ = **5.3e-5** (< 3e-4), RMS-Grad 3.0e-5 / Max-Grad 3.9e-5, linear.

## Notes / limitations

- **Atomic approximation only.**  `hcore_deriv_generator` raises on any
  non-`atom1e` approx.  The molecular decoupling response `dX/dR` (a
  Sylvester/CPHF solve) is a separate project.
- **One-electron SOC only** — no 2e-SO / SNSO (Boettger) / AMFI.  The bare
  X2C-1e absolute SOC energy over-estimates (+29 Eh core term on uranyl, A1),
  but that term is geometry-independent to ~99.6 % and differentiates to ≈ 0
  (V3), so the gradient is the clean physical part.
- **GKS on actinides:** use `grid_response=True`; the PES *curvature* from the
  GKS analytic gradient is currently softened by `_gks_xc_grad` (see V3).  The
  minimum location and the SOC force slope are sound; GHF is exact.
- A perf bug found during validation: `np.einsum('pi,xpq,qj->xij', …)` without
  `optimize=` takes an O(n⁴) Python path (2418 s for the HI gradient).
  `grad/x2c.py` uses explicit BLAS matmuls (5.8 s, identical numerics).
- `Gradients.to_cpu` for a KS base still has no pyscf `grad/gks.py` counterpart
  (pre-existing; noted in `grad/gks.py`).  The X2C hcore-derivative term carries
  no new round-trip state.
