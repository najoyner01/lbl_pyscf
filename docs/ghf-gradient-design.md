# GHF nuclear gradient — design

Status: **2026-09-08, implemented** — `gpu4pyscf/grad/ghf.py`, merged to
`gpu-porting`. Step 2 of the SOC-gradient roadmap in
`docs/adf-parity-strategy.md` §1b.

Validation to date (A100, `grad/tests/test_ghf_grad.py`):
- **(a) done** — block-diagonal real GHF (UHF MOs embedded) vs `grad/uhf.py`:
  `||Δ|| = 4.2e-14`.
- **(b) done** — GHF analytic vs finite difference, OH/sto-3g spin=1 (no SOC,
  `D_αα ≠ D_ββ`, real): `||Δ|| = 1.7e-8`. (Original plan named HI/crenbl; that
  has an even valence-electron closed shell, swapped for OH.)
- **(c) pending run** — spin-rotation invariance test added
  (`test_ghf_spin_rotation_invariance`): a global SU(2) rotation of the
  converged solution activates the imaginary-diagonal (`k_factor=-2`) and
  `D_αβ` cross (`k_factor=+4`) paths, which (a)/(b) leave at zero. **Until this
  passes, those two paths — hence every complex/non-collinear/SOC GHF
  gradient — are unvalidated.** `grad/ghf.py` emits a `logger.warn` when the
  density has non-negligible `Y`/`D_αβ` blocks.
- **SOC hcore path** (`d ECPso/dR`): **done (2026-09-08)** — new bra-derivative
  kernel `ECP_so_ip_cart` (`lib/ecp/ecp_so_ip.cu`), Python
  `gpu4pyscf.gto.ecp.{loop_ecp_so_ip,get_ecp_so_ip}`, wired into
  `grad/ghf.py:_soc_hcore_grad`. Validated (A100): integral-level FD
  (`test_ecp_so.py::SoIpFiniteDifference`, `‖Δ‖ ≲ 5e-11`) and end-to-end
  `with_soc=True` FD (`test_ghf_grad.py::TestGHFGradSOC`: HI/CRENBL spin=0,
  `|D_αβ|~3e-2`, `‖analytic−FD‖ = 2.6e-9`; single I atom, gradient vanishes to
  `2e-28`). See §5.2.

Implementation note that diverged from this doc: the multi-dm kernel
(`RYS_per_atom_jk_ip1_multidm`) normalizes per pair differently from the
single-dm kernel — self-pair `[D,D]` with `k_factor=f` → `−(f/2)·d/dR
Tr[D·K(D)]`; cross-pair `[D1,D2]` → `−(f/4)·d/dR Tr[D1·K(D2)]`. The `k_factor`
column in `_ghf_jk_energy` (`[·, 2, -2, 2, -2, 4, 4]`) is set accordingly; the
`+2`/real-diagonal choice is the one (a)/(b) confirm.

No CPU or GPU reference exists anywhere in PySCF/gpu4pyscf for this (no
`grad/ghf.py` in either tree). This doc works the formulas out from first
principles so implementation isn't blind guessing, and states exactly how it
reduces to gpu4pyscf's *existing* GPU kernels — no new CUDA needed for this
piece (the SO-ECP gradient integral, step 3, is separate and does need one).

## 0. What GHF's density and Fock look like

GHF works in the `2nao × 2nao` spin-orbital basis. Density and Fock split into
four `nao × nao` spatial blocks (`αα, αβ, βα, ββ`):

    D = [ D_αα  D_αβ ]      F = [ F_αα  F_αβ ]
        [ D_βα  D_ββ ]          [ F_βα  F_ββ ]

`D` is Hermitian: `D_βα = D_αβ^†`, `D_αα, D_ββ` Hermitian. In general **all four
blocks are complex** (that's the whole point of GHF — it's how spin-orbit
coupling enters). For a plain no-SOC GHF solution the optimizer *can* converge
to a real, spin-block-diagonal solution (reducing to RHF/UHF) — useful as a
validation anchor (§4) — but the gradient code must not assume that.

Standard non-relativistic GHF Fock build (spin-independent Coulomb operator,
`(pr|qs)` real AO ERIs, no relativistic two-electron terms):

    F_ss'[pq] = h_ss'[pq] + δ_ss' Σ_tt' Σ_rs (pq|rs) D_tt',rs      (J, block-diagonal only)
                          − Σ_rs (pr|qs) D_ss',rs                  (K, all four blocks)

i.e. **J only sees the spin-summed total density** `D_tot = D_αα + D_ββ`
(real, Hermitian — off-diagonal spin blocks integrate to zero against the
spin-independent Coulomb operator), applied to *both* diagonal Fock blocks;
**K acts block-by-block**, each of the four `D_ss'` blocks contracted with the
*same* real-ERI exchange kernel independently — no mixing between blocches
within K either. This is the standard result (e.g. Stanton/Bartlett-style GHF
references) and is what makes reduction to real kernels possible: neither J
nor K needs anything beyond ordinary real two-electron integrals contracted
against a set of `nao × nao` matrices.

## 1. The gradient formula

Same Pulay-force shape as every other reference determinant (RHF/UHF/DHF —
see `pyscf/grad/dhf.py`, already read this session):

    dE/dR_A = Σ_pq  Re[ dh1e/dR_A[pq] · D[pq] ]                (hcore term)
            + Σ_pq  Re[ dVeff/dR_A[pq] · D[pq] ]                (JK term, see §2)
            − Σ_pq  Re[ dS/dR_A[pq] · D_e[pq] ]                 (Pulay overlap term)
            + dE_nuc/dR_A

`D` is the full `2nao×2nao` density in AO (spin-orbital) basis; `D_e` is the
energy-weighted density matrix (`D_e = C_occ · diag(ε_occ) · C_occ^†`, same
`2nao×2nao` shape). `Re[...]` because `D`/integrals are complex but the total
energy — and therefore its derivative — is real; this exactly mirrors
`grad/dhf.py:grad_elec`'s `.real` on each `einsum` term.

`h1e` here is the spin-orbital-blocked one-electron Hamiltonian:
`h1e = block_diag(h_kin+nuc, h_kin+nuc)` **plus** the SOC term when present —
`get_soc_1e(mol)` (ECP route) or the X2C `get_hcore` (X2C route, gradient
deferred per the roadmap). `dh1e/dR_A` needs the derivative of whichever term
is present. For the phase-2 baseline (no SOC) it's just the ordinary
kinetic+nuclear-attraction derivative, block-diagonal, real — already
available via `int1e_ipkin`/`int1e_ipnuc` (used by `grad/rhf.py`).

`dS/dR_A` is likewise `block_diag(dS_kin/dR_A, dS_kin/dR_A)` from the existing
`get_ovlp` gradient — no new integrals.

## 2. The JK-derivative term — reduction to existing kernels

The two-electron contribution to `dE/dR_A` (`Σ_pq Re[dVeff/dR_A · D]`) splits
by the J/K structure in §0 into:

    dE_J/dR_A =        Σ_pq  d(pq|rs)/dR_A · D_tot[pq] · D_tot[rs]   (real, spin-summed)
    dE_K/dR_A = −Σ_{ss'} Σ_pq  d(pr|qs)/dR_A · D_ss'[pq] · D_ss'[qp]  (per spin-block pair)

### 2.1 Derivation, resolved

The block conventions are pinned down by cross-checking against gpu4pyscf's
**already-validated** GHF SCF `get_jk` (`gpu4pyscf/scf/ghf.py:_get_jk`), not
guessed: it splits `dm` into `dmaa, dmab, dmba, dmbb`, builds `J` from
`dmaa+dmbb` only (placed on **both** diagonal blocks of `vj`, off-diagonal
`vj` blocks are exactly zero), and builds `K` block `(s',s)` as the ordinary
real exchange-kernel applied **to `D` block `(s',s)` directly** — e.g.
`vk[:, :nao, nao:] = K_real(dm_ab)`. No relativistic two-electron terms, no
cross-block mixing inside a single `J`/`K` block build.

Using the standard block-matrix trace rule `Tr[AB] = Σ_{s,s'} Tr[A_{s,s'} B_{s',s}]`
(note the *swap* of the second index — this is just `Σ_i(AB)_{ii}` expanded
by blocks) on `E_2e = (1/2) Tr[D·(J-K)]`, with `J` block-diagonal
(`J_{αα}=J_{ββ}=J(D_tot)`, `J_{αβ}=J_{βα}=0`) and `K_{s',s} = K(D_{s',s})`:

    E_2e = (1/2) Tr[D_tot · J(D_tot)]
         − (1/2) Tr[D_αα · K(D_αα)] − (1/2) Tr[D_ββ · K(D_ββ)]
         − (1/2) Tr[D_αβ · K(D_βα)] − (1/2) Tr[D_βα · K(D_αβ)]

The two off-diagonal terms are equal (`Tr[Y·K(X)] = Tr[X·K(Y)]` — the real
exchange kernel is a symmetric bilinear form, standard 8-fold ERI symmetry),
and `Tr[D_αβ·K(D_βα)] = Tr[D_βα^† · K(D_βα)]` (using `D_αβ = D_βα^†`) is
manifestly real, matching that the total energy must be real without an
explicit `Re[...]`:

    E_2e = (1/2) Tr[D_tot·J(D_tot)] − (1/2) Tr[D_αα·K(D_αα)] − (1/2) Tr[D_ββ·K(D_ββ)]
         − Tr[D_αβ·K(D_βα)]

At a converged (variational) SCF solution the implicit `dD/dR` terms don't
contribute at first order (standard HF gradient theorem — same reason
`grad/rhf.py`/`grad/dhf.py` differentiate only the *integrals*, holding `D`
fixed), so `dE_2e/dR_A` is the same expression with `J(·)`/`K(·)` replaced by
their nuclear derivatives.

### 2.2 The right existing primitive — bra ≠ ket, already used for TDDFT

The `D_αβ`/`K(D_βα)` cross term needs a **bra ≠ ket** exchange-gradient
contraction (`Tr[Y·dK(X)/dR]` for independent `X, Y`) — not the self-only
`Tr[D·dK(D)/dR]` that `Gradients.jk_energy_per_atom`
(`grad/rhf.py:_jk_energy_per_atom`, kernel `RYS_per_atom_jk_ip1`) computes.

**gpu4pyscf already has the general primitive**, used and validated for
TDDFT gradients: `grad/tdrhf.py:_jk_energies_per_atom(vhfopt, dm_pairs,
j_factor, k_factor, ..., sum_results=...)`, kernel
`RYS_per_atom_jk_ip1_multidm` (or `..._sum`), taking a list of
**`[dm1, dm2]` pairs** (`dm1[i] = dm2[i]` collapses to the self-only case).
This is exactly the tool: no polarization-identity workaround needed, no new
kernel needed for this piece either.

### 2.3 Concrete call plan

All matrices real `nao×nao` (split each complex GHF block `D_ss' = A+iB`;
`K` is linear so `K(A+iB) = K(A)+i·K(B)`, and the assembled real energy only
needs the specific real/imaginary combinations that survive — work these out
explicitly during implementation, don't re-guess signs). One
`_jk_energies_per_atom` call with a batch of `dm_pairs`:

| pair (dm1, dm2) | factor | contributes to |
|---|---|---|
| `(D_tot, D_tot)` | `j_factor=1, k_factor=0` | `Tr[D_tot·J(D_tot)]` |
| `(Re D_αα, Re D_αα)`, `(Im D_αα, Im D_αα)` | `j=0, k=1` | `Tr[D_αα·K(D_αα)]` |
| `(Re D_ββ, Re D_ββ)`, `(Im D_ββ, Im D_ββ)` | `j=0, k=1` | `Tr[D_ββ·K(D_ββ)]` |
| `(Re D_αβ, Re D_βα)`, `(Im D_αβ, Im D_βα)`, cross real/imag terms | `j=0, k=1` | `Tr[D_αβ·K(D_βα)]` (**bra≠ket**, needs the multidm path; enumerate all 4 real-part combinations from `(A+iB)(C+iD)`-style expansion and keep the ones that survive `Re[]` — do this expansion explicitly in code/comments, not from memory) |

Batch everything into one `_jk_energies_per_atom` call (it already supports a
list of pairs + per-pair `j_factor`/`k_factor`), sum with the `-1/2`, `-1/2`,
`-1` prefactors from §2.1.

**Still unresolved, must be pinned down while implementing (reading the
kernel source further or a numerical check), not assumed:** the exact
normalization baked into `j_fac[i]=j_factor[i]*.5`,
`k_fac[i]=k_factor[i]*-.25` inside the CUDA kernel, and the `n_dm==1` special
case `k_factor *= .5` in `RYS_per_atom_jk_ip1` — i.e. whether passing
`j_factor=1` to this primitive already means "the standard RHF-convention
`(1/2)Tr[D·dJ(D)/dR]`" or something else. Get this right by testing, not by
re-deriving from the CUDA source in your head.

## 3. What's reused vs. new

| Piece | Source |
|---|---|
| hcore + overlap Pulay terms | Existing `int1e_ipkin`/`int1e_ipnuc`/`get_ovlp` gradient integrals, block-diagonaled to `2nao` |
| J-derivative | Existing `jk_energy_per_atom(D_tot, j_factor=1, k_factor=0)` unchanged |
| K-derivative | Existing kernel, called on real/imag parts of the 4 spin-blocks (careful assembly, §2) |
| SOC hcore-derivative | **New** — SO-ECP gradient integral (step 3, separate design/kernel) |
| Nuclear repulsion | `grad_nuc`, unchanged |

**No new CUDA kernel for the baseline (no-SOC) GHF gradient.** This is
Python-level orchestration reusing gpu4pyscf's RHF-gradient JK machinery.

## 4. Validation plan (no CPU oracle exists)

1. **Reduction to RHF.** For a closed-shell system with SOC off and a
   spin-restricted initial guess, GHF's converged energy and (this
   gradient's) forces must match the already-validated RHF gradient to high
   precision. Cheapest, strongest sanity check — do this first.
2. **Finite difference** of the plain GHF (no SOC) total energy vs. this
   analytic gradient, on a case where GHF genuinely differs from RHF/UHF
   (e.g. an open-shell system where the optimizer breaks collinearity) — this
   is what actually stresses the off-diagonal K-block assembly from §2.
3. Only after 1–2 pass: extend `dh1e/dR_A` to include the SOC term once step 3
   (SO-ECP gradient integrals) lands, and finite-difference the full
   GHF+ECP-SOC energy.

## 5. Status and open risk

Design is now resolved to the level of "which exact function to call with
which arguments" (§2.2–2.3), cross-checked against the already-validated GHF
SCF `_get_jk` rather than derived in isolation. **Not yet implemented** —
deliberately held back from a blind implementation pass, because:

- The RHF-reduction sanity check (§4.1) does **not** exercise the K
  cross-term at all (for a real, spin-block-diagonal solution `D_αβ=D_βα=0`,
  the cross term vanishes identically) — it's necessary but not sufficient.
  The cross-term formula (§2.1's `Tr[D_αβ·K(D_βα)]`, the real/imaginary
  expansion in §2.3, and the exact kernel normalization) can only be
  confirmed by finite difference on a genuinely non-collinear case, which
  needs a GPU.
- This is the most novel, least-checkable piece built in this project so
  far — no CPU pyscf module, no prior GPU implementation anywhere to diff
  against. The failure mode (a wrong sign or a missed factor of 2, exactly
  the two bug classes already hit twice this session in far simpler kernels)
  produces a plausible-looking wrong number, not a crash.

**Do not implement this blind and call it done.** Implement §2.3 following
`grad/tdrhf.py`'s exact calling pattern, validate in two stages (RHF-reduction
first, catches most bugs cheaply; then finite difference on a non-collinear
case, the only real check of the cross-term), and don't wire it into any
workflow the user would trust for actinide research until both pass — same
discipline that caught the SO-ECP factor-of-two and the PBC antisymmetric-sign
question earlier in this project, both of which *did* need the second,
harder check to surface.

### 5.1 Update 2026-09-08 — what actually happened

Implemented; (a) and (b) pass. But (b)'s molecule (OH, no SOC) converges to a
**real block-diagonal** solution, so — exactly as this section warned — it
does *not* exercise the cross-term or the imaginary-diagonal blocks. The
`k_factor=-2` and `k_factor=+4` paths in `_ghf_jk_energy` are therefore still
on unvalidated footing; a wrong factor there produces a plausible wrong
gradient for every SOC calculation and nothing else.

Closed without waiting for SO-ECP integrals: **test (c), spin-rotation
invariance** (`test_ghf_spin_rotation_invariance`). Rotating the converged
solution by a global `R_x(θ)` is an exact symmetry of the spin-free
Hamiltonian, so the analytic gradient must not move; but the rotation shifts
density weight into `Y_aa/Y_bb/A_ab/B_ab`, so a wrong `k_factor` on those pairs
makes `g_rot` drift from `g_0`. Run it on the next GPU session. A genuine
`with_soc=True` finite-difference check still comes later, with step 3.

### 5.2 Update 2026-09-08 — step 3 (SO-ECP gradient integral) landed

**Kernel.** `gpu4pyscf/lib/ecp/ecp_so_ip.cu`, `so_cart_ip1_general` (driver
`ECP_so_ip_cart` in `nr_ecp_driver.cu`). It is `so_cart` (energy SO-ECP) with
the bra-derivative recursion of `type2_cart_ip1` grafted on: evaluate the
L^a-transformed type-2 block once with the bra angular momentum raised
(`LI+1`, radial order 1) and once lowered (`LI-1`, order 0), then map back with
`_li_down`/`_li_up`. Same radial prefactor as scalar type-2 (no ½). The ket
projector still carries the real antisymmetric `L^a` transform
(`transform_omega_lop`). Output `[n_ecp_atm, 3(a), 3(x), nao, nao]`, real,
bra = row. The bra derivative breaks the `(i,j)` antisymmetry `so_cart`
exploits, so the driver feeds a **full (non-triangular)** task list and the
kernel writes **no transpose block** — exactly like `get_ecp_ip`.

**Python.** `gpu4pyscf.gto.ecp.loop_ecp_so_ip(mol, ecp_atoms=…, batch_size=…)`
yields `(atom_ids, G)` with `G[c,a,x,r,s] = <∂_x r | l_a U_SO | s>` for SO-ECP
centre `atom_ids[c]`; `get_ecp_so_ip(mol)` sums over centres →
`[3(a), 3(x), nao, nao]`. Reuses `_EcpDerivContext` + `sort_ecp_basis_so`.

**Wire-in.** `grad/ghf.py:_soc_hcore_grad(mol, dm_ghf)`. The SOC energy in the
nao×nao spin blocks is `E_soc = ½ Σ_a Tr[W_a · ECPso_a]` with the real
antisymmetric `W_a = Im(Σ_{xy} pauli[a,y,x] D_{xy})`:

    W_x = Im(D_αβ) − Im(D_αβ)ᵀ
    W_y = Re(D_αβ) − Re(D_αβ)ᵀ
    W_z = Im(D_αα) − Im(D_ββ)

`dE_soc/dR_A = ½ Σ_a Tr[W_a · dECPso_a/dR_A]`, and `dECPso_a/dR_A` assembles
from `G` with the same bra/ket/ECP-centre (translational-invariance)
bookkeeping as the scalar ECP path in `grad/rhf.py:_hcore_energy`:

    ECP-centre C:  de[C,x] += Σ_a Σ_{q,p} G^C[a,x,q,p] W_a[p,q]
    localised   A:  de[A,x] −= Σ_a Σ_{q∈A, p} G^sum[a,x,q,p] W_a[p,q]

(`W_a` antisymmetric ⇒ the bra and ket localised pieces are equal; the two
contributions cancel over all atoms, as required for a translation-invariant
energy). `_ghf_jk_energy`'s non-collinear `logger.warn` placeholder is removed
— the `k_factor=−2/+4` paths are now covered end-to-end by the `with_soc=True`
FD test below.

**Validation (A100).**

| gate | check | result |
|---|---|---|
| V1 | `test_ecp_so.py::SoIpFiniteDifference` — per-atom analytic `dECPso/dR` vs central FD of `get_ecp_so` (h=1e-4). Pb/O CRENBL (`ul`), K/F ecpds10mdfso (explicit lc 0–3), HI CRENBL | `‖Δ‖` = 3.6e-11 / 6.1e-12 / 1.8e-10 |
| V2 | `test_ecp_so.py` + `test_ecp_sweep.py` regression (new include must not perturb `get_ecp_so`) | 10 passed, 1526 subtests |
| V3 | `test_ghf_grad.py::TestGHFGradSOC` — `GHF(mol); with_soc=True`, analytic `nuc_grad_method` vs central FD of `e_tot` (h=1e-4, conv_tol 1e-12). HI/CRENBL spin=0 (genuinely non-collinear, `|D_αβ|~3e-2`); single I atom spin=1 | `‖Δ‖` = 2.6e-9 ; 2.5e-10 (`‖g_analytic‖ ~ 2e-28`) |

### 5.3 Update 2026-09-08 — GKS (2-component DFT) nuclear gradient

`grad/gks.py`, `class Gradients(ghf_grad.Gradients)`.  Wired via
`dft/gks.py:GKS.nuc_grad_method`.  Everything the GHF gradient does (hcore +
overlap Pulay + SO-ECP hcore derivative) is inherited unchanged; two things
are added.

**Scaled exact exchange.**  `grad/ghf.py:_ghf_jk_energy` gained a `k_scale`
argument (default `1.0`, so the GHF path is untouched).  GKS passes the
functional hybrid coefficient (`0` for a pure functional); J is never scaled.
`omega != 0` (range-separated hybrids) raises `NotImplementedError`.

**Multi-collinear XC gradient** (`_gks_xc_grad`).  Mirrors `grad/uks.py` but
replaces the `(α, β)` spin split with the `(ρ, m_x, m_y, m_z)` → spin-block
mapping of `dft/numint2c.py`.  From the sorted 2-component DM,

    dm_ρ  = Re(D_αα + D_ββ)      dm_mz = Re(D_αα − D_ββ)
    dm_mx = A_αβ + A_αβᵀ         dm_my = −(B_αβ + B_αβᵀ)

(`A_αβ = Re D_αβ`, `B_αβ = Im D_αβ`; all four real symmetric).  Each channel
`c` pairs its `mcfun` potential `vxc_c` (`[wr, wmx, wmy, wmz]`) with `dm_c` in
exactly `grad/uks.py`'s per-spin `_d1_dot_` / `_gga_grad_sum_` +
`−2·reduce_to_atom` pipeline.  For a collinear solution (`D_αβ = 0`) the
`m_x`/`m_y` channels vanish and the sum reduces term-by-term to `grad/uks.py`.
`grid_response=True` adds the grid-weight + grid-ρ response, mirroring
`grad/uks.py:get_exc_full_response` with the 4-channel `dvmat`.  LDA, GGA and
global hybrids; MGGA and NLC raise `NotImplementedError`.  Single GPU.

**Validation (A100, `spin_samples=50`, `grids.level=3`).**

| gate | check | result |
|---|---|---|
| V1 | collinear reduction — real block-diagonal GKS (UKS MOs embedded) vs `grad/uks.py`, `grid_response=True`, O₂ triplet / def2-svp | `‖GKS−UKS‖` = svwn 1.8e-11, pbe 3.6e-11, pbe0 1.0e-12 |
| V2 | FD, no SOC — `GKS(H₂O/def2-svp, xc)` analytic vs central FD of `e_tot` (h=1e-4) | `grid_response=True`: svwn 1.2e-7, pbe 2.0e-7, pbe0 4.4e-8.  `=False` (quadrature-limited): 5–7e-5 |
| V3 | FD, with SOC — `GKS(xc='pbe0'); with_soc=True` on HI/CRENBL spin=0 (`\|D_αβ\|~3e-2`), `grid_response=True` | `‖analytic−FD‖` = 7.6e-9 |
| V4 | regression — `test_ghf_grad.py` 7/7; GPU `test_gks.py::test_mcol_gks_{lda,gga,hyb,mgga}` pass.  The CPU-`mcfun` cross-check tests in `test_gks.py` SIGABRT/SIGSEGV inside `pyscf/dft/numint2c.py` + libxc-in-threads — reproduced on a pristine checkout, pre-existing and unrelated. |

Tests: `grad/tests/test_gks_grad.py` (8 pass).

### 5.4 Update 2026-09-08 — SOC geometry optimization

`pyscf.geomopt.geometric_solver.optimize` drives the analytic SO-ECP
gradients above.  **No new gpu4pyscf code** was needed — `with_soc` is an
instance attribute in `GHF._keys`, so it survives `mf.reset(new_mol)` at every
optimizer step, and `mf.set_geom_(..., inplace=False)` preserves
`ecp=`/`basis=`/`spin=` (`mol.has_ecp_soc()` stays `True` throughout).

**Recommended call.**

    # GHF + SO-ECP (Hartree-Fock level)
    mf = scf.GHF(mol); mf.with_soc = True; mf.kernel()
    mol_eq = optimize(mf, maxsteps=20)

    # GKS + SO-ECP (DFT level) — grid_response=True strongly recommended
    mf = dft.GKS(mol, xc='pbe0'); mf.with_soc = True
    mf.spin_samples = 50          # cheaper mcfun spin-angular quadrature
    mf.kernel()
    g = mf.nuc_grad_method(); g.grid_response = True
    mol_eq = optimize(g.as_scanner(), maxsteps=20)

Caveats:

- `optimize()` accepts an SCF object (uses its *default* gradient —
  `grid_response=False` for GKS) or a `lib.GradScanner` (`g.as_scanner()`,
  which copies a pre-configured `g` via `__dict__.update`).  It does **not**
  accept a bare gpu4pyscf `Gradients` object (gpu4pyscf's `GradientsBase` is
  not a subclass of pyscf's) — always call `.as_scanner()`.
- For GKS, `grid_response=False` leaves a ~5e-5 gradient floor (the omitted
  grid-weight-derivative term) that can stall convergence near the minimum;
  set `grid_response=True` and pass `g.as_scanner()`.
- `spin_samples=50` keeps the mcfun angular quadrature cheap enough for an
  optimization; use the production default (770) for the final single point.

**Validated (A100, CRENBL, `maxsteps≤20`).**

| gate | check | result |
|---|---|---|
| T1 | `scf.GHF(HI); with_soc=True` → `optimize(mf)` | converged, r(H–I) = **1.6041 Å**, ‖grad‖ at min = 3.5e-7 |
| T2 | `dft.GKS(HI, xc='pbe0'); with_soc=True; spin_samples=50`, `g.grid_response=True` → `optimize(g.as_scanner())` | converged, r(H–I) = **1.6425 Å**, fresh ‖grad‖ = 2.3e-6 |
| T3 | GHF optimize HI with vs without `with_soc` | r differ by **2.7e-3 Å** (1.6041 vs 1.6015) — SOC not lost in the scanner |
| T4 | plain (no-SOC) mcol-`GKS(HF, xc='pbe0')` optimize | converged, r(H–F) = 0.920 Å, fresh ‖grad‖ = 4e-7; `grad/tests/test_geomopt.py` unaffected (no code change) |
| T5 | fresh analytic gradient at each converged geometry | ‖grad‖ < 1e-4 (GHF), < 3e-4 (GKS) |

Tests: `grad/tests/test_soc_geomopt.py`.  Not yet exercised on an actinide;
`geomeTRIC` and `mcfun` must be importable.

### 5.5 Update 2026-09-09 — solvated SOC (PCM / SMD compose)

`scf.GHF(mol).PCM()` / `.SMD()` and `dft.GKS(mol, xc=...).PCM()` / `.SMD()`
now run SCF **and** analytic nuclear gradients, including with
`mf.with_soc = True`.  Actinide separation ΔG is computed in solution, so this
is the last plumbing step for a usable solvated-SOC workflow.

**Physics.**  The reaction field is spin-independent — it couples only to the
number density `ρ = Re(D_αα + D_ββ)` and the nuclear charges.  So:

- SCF: reduce the `2nao×2nao` spinor DM to `ρ` (`nao×nao` real) before the
  solvent kernel; add the returned `v_solvent` (`nao×nao`) into **both**
  diagonal spin blocks of the `2nao×2nao` Fock (block-diagonal, identical,
  zero off-diagonal).
- gradient: `WithSolventGrad.kernel` feeds the same `ρ` to
  `grad_qv`/`grad_solver`/`grad_nuc`; the solvent nuclear gradient is a plain
  additive `[natm,3]` term, independent of the SOC / K / XC parts.
- SMD CDS is a pure SASA geometric term (`_attach_solvent.energy_elec` +
  `grad/smd.py`), spin-free by construction.

**Changes (all gpu4pyscf-side, minimal).**

- `scf/ghf.py`: removed the `PCM` `NotImplementedError`.  `solvent/pcm.py`,
  `solvent/smd.py`: `scf.ghf.GHF.PCM/SMD = *_for_scf` (GKS inherits).
- `solvent/_attach_solvent.py`: `_spin_sum_dm(dm, nao)` (RHF/UHF/GHF-GKS →
  spin-summed real DM) and `_expand_v_solvent_2c`; `SCFWithSolvent.get_veff`
  reduces the DM and block-diagonalises `v_solvent` when the base is 2-component
  (`isinstance(mf, ghf.GHF)`).
- `solvent/grad/{pcm,smd}.py`: `WithSolventGrad.kernel` calls `_spin_sum_dm`
  (was a UHF-only `dm[0]+dm[1]`).
- `dft/gks.py`: `GKS` now defines `Gradients` (not `nuc_grad_method`), so a
  solvent-attached GKS routes through `hf.SCF.nuc_grad_method → self.Gradients()`
  and `SCFWithSolvent.Gradients` (earlier in the MRO) wraps it.  **This was the
  one real bug**: without it `mf.nuc_grad_method()` on a `GKS().PCM()` returned
  the bare vacuum gradient (no solvent term).

**Recommended call.**

    mf = dft.GKS(mol, xc='pbe0').PCM()     # or .SMD(); GHF works the same
    mf.with_soc = True
    mf.spin_samples = 50
    mf.kernel()                            # ΔG_solv from mf.e_tot − gas e_tot
    g = mf.nuc_grad_method(); g.grid_response = True
    de = g.kernel()
    mol_eq = optimize(g.as_scanner(), maxsteps=20)   # solvated SOC geom-opt

**Validated (A100).**

| gate | check | result |
|---|---|---|
| V1 | real block-diagonal `GHF/GKS(pbe0).PCM()` (UHF/UKS MOs embedded) vs `UHF/UKS.PCM()`, O₂ triplet | dE = 1e-13, ‖dGrad‖ = 7e-14 (GHF) / 7e-13 (GKS) |
| V2 | FD, no SOC — `GHF(HF).PCM()`, `GKS(HF,pbe0).PCM()` (def2-svp), analytic vs central FD of `e_tot` | ‖ana−FD‖ = 1.4e-8 (GHF) / 2.1e-8 (GKS) |
| V3 | FD, **with SOC** — `GKS(HI/CRENBL, pbe0).PCM(); with_soc=True`, `spin_samples=50`, `grid_response=True` (SOC hcore + XC + scaled-K + PCM in one gradient) | ‖ana−FD‖ = 3.0e-8 |
| V4 | V2/V3 with `.SMD()` (adds CDS gradient) | ‖ana−FD‖ = 1.3e-8 (GHF+SMD) / 3.0e-8 (GKS+SMD+SOC) |
| V5 | `optimize(g.as_scanner())` for `GKS(HI, pbe0).PCM(); with_soc=True` | converged, r(H–I) = 1.6425 Å, fresh ‖grad‖ = 2.3e-6 |
| V6 | ΔG_solv(HI, GHF+SOC) sign/size; existing `solvent/tests/` (RHF/UHF/RKS/UKS PCM+SMD grad) | ΔG = −0.00349 Eh = **−2.19 kcal/mol**; 61/61 solvent tests pass |

Tests: `solvent/tests/test_pcm_soc.py`.  Range-separated hybrids and MGGA in
GKS still raise `NotImplementedError` (unchanged from §5.3).

### 5.6 Update 2026-09-09 — FD nuclear Hessian + thermochemistry

`gpu4pyscf/hessian/fd.py`.  A **finite-difference** nuclear Hessian for GHF /
GKS, so vibrational frequencies and thermochemistry (ZPE, H, S, G) work for
spin-orbit DFT.  An analytic 2-component / SOC Hessian is a much larger job and
is **future work**; this is the interim path.

**Method.**  `finite_diff_hessian(mf, disp=1e-3)` — central differences over
the 3N nuclear coordinates using the method's own analytic-gradient scanner
(`mf.nuc_grad_method().as_scanner()`; `grid_response=True` for GKS).  `with_soc`
and `.PCM()`/`.SMD()` ride along automatically (the scanner re-converges the
full method at each geometry).  Returns `(natm,natm,3,3)` — the layout the
analytic Hessians emit and `pyscf.hessian.thermo` consumes.

- **Cost: O(6N) re-converged SCF + gradient evaluations.  Small systems only.**
- Requires `mf.conv_tol <= 1e-11` (raises otherwise): the FD gradient
  difference is ~ `conv_tol / disp`.
- `harmonic_analysis` is only meaningful at a stationary point — optimize the
  geometry first (a residual gradient contaminates the rotational modes; e.g.
  HI/CRENBL+SOC off-minimum shows spurious ~150 cm⁻¹ "rotations", ~1 cm⁻¹ at
  the optimized geometry).

**Wiring.**  `hessian/fd.py` grafts `scf.ghf.GHF.Hessian` (GKS inherits),
mirroring `hessian/{rhf,uhf,rks,uks}.py`.  `mf.Hessian().kernel()` →
`(natm,natm,3,3)`.  Convenience: `hessian.fd.harmonic_and_thermo(mf, T, P)` →
`(freq_info, thermo_info)`.

**Recommended call.**

    mf = dft.GKS(mol, xc='pbe0').PCM()          # or scf.GHF(mol); or .SMD()
    mf.with_soc = True; mf.spin_samples = 50
    mf.conv_tol = 1e-12
    mf.kernel()
    mol_eq = optimize(...)                      # frequencies need a minimum
    mf = <rebuild at mol_eq>; mf.kernel()
    freq_info, thermo_info = gpu4pyscf.hessian.fd.harmonic_and_thermo(mf)
    G_tot = thermo_info['G_tot'][0]

**Validated (A100, `spin_samples=50`, `disp=1e-3` Bohr, `conv_tol≤1e-12`).**

| gate | check | result |
|---|---|---|
| V1 | FD GHF Hessian (real block-diagonal, UHF MOs embedded, no SOC) vs analytic UHF Hessian, H₂O/sto-3g | `‖ΔH‖/‖H‖` = **1.7e-6** |
| V2 | FD GKS(pbe0) Hessian vs analytic UKS(pbe0, `grid_response=True`) Hessian, H₂O/sto-3g | `‖ΔH‖/‖H‖` = **2.6e-6** |
| V3 | HI/CRENBL, `with_soc=True`, at the optimized geometry, unprojected `harmonic_analysis`: 5 modes `\|freq\| < 50 cm⁻¹` + 1 stretch in 2000–2600 | GHF+SOC: rot 0.6 / 7.2, **stretch 2441 cm⁻¹**; GKS(pbe0)+SOC: rot ≈ 0, **stretch ≈ 2430 cm⁻¹** |
| V4 | HI GHF stretch with vs without `with_soc` (common geometry) | 2408.0 vs 2413.2 → **SOC shift 5.25 cm⁻¹** |
| V5 | `harmonic_and_thermo` for `GKS(HI,pbe0); with_soc=True` and for `.PCM()` | ZPE 0.00554, H_tot −111.9129, S_tot 7.86e-5, **G_tot −111.9364 Eh**; +PCM **G_tot −111.9396 Eh** (all finite) |
| V6 | step-size: `‖H(1e-3) − H(2e-3)‖` and `‖H(1e-3) − H(5e-3)‖` (GHF+SOC HI) | **5.8e-6** and 9.8e-6 (Ha/Bohr²) |
| V7 | regression: `hessian/tests/` (analytic RHF/RKS/UHF/UKS) + SOC grad/geomopt/PCM suites | unaffected (new module + one import line) |

Tests: `hessian/tests/test_fd_hessian.py` (FD ones `@pytest.mark.slow`).
Captured A100 run: `hessian/tests/results/fd_hessian_A100.md`.

**Known limitation — GKS+SOC frequencies are not converged at `spin_samples=50`.**
The V3 HI GKS(pbe0)+SOC stretch came out **2253 cm⁻¹** in the captured run vs
**≈2430 cm⁻¹** on an earlier draw — a ~180 cm⁻¹ spread, wide enough to move ZPE
by a few tenths of a kcal/mol and to contaminate S, hence ΔG. The collinear path
is unaffected (V2 matches the analytic UKS Hessian to 2.6e-6, because mcfun's
`collinear_thrd` shortcut takes over), and the GHF path reproduces 2441 cm⁻¹
tightly (V4/V6 use GHF). The V3 gate (2000–2600 cm⁻¹) is loose enough that it
passes either way — it is a smoke test, not a convergence check.

*Mechanism — hypothesis, not yet confirmed.* `spin_samples` selects a
**deterministic** Lebedev grid (`dft/mcfun_gpu.py:_make_sph_samples` →
`MakeAngularGrid`), so identical input gives identical output; the spread is
therefore not quadrature randomness. The likely cause is that a coarse Lebedev
grid does not integrate the spin-angular dependence exactly, leaving the
multi-collinear XC energy weakly dependent on the **orientation of the spin
quantization axis** — an orientation that is physically arbitrary (globally
degenerate) and that the SCF can land on differently from run to run. FD then
amplifies the resulting PES wobble by `1/disp`. This is directly testable with
the global-spin-rotation machinery already used by
`test_ghf_spin_rotation_invariance` (§4.3): at fixed geometry and fixed density,
rotate the spin axis and watch the GKS+SOC energy — exactly invariant in exact
theory, so any variation is the quadrature's rotational-invariance error, and it
should shrink as `spin_samples` grows. Other candidates to rule out: a different
converged SCF solution, or the two runs having optimized to slightly different
geometries.

`spin_samples=50` is a **test-suite speed setting**, not a production one (the
library default is 770). For any thermochemistry that matters:

- raise `spin_samples` to ≥770 and confirm the frequency is stable against a
  further increase before trusting ZPE / S / G;
- `finite_diff_hessian` now emits a `logger.warn` when it is handed a SOC GKS
  object with `spin_samples < 770`.

Still open: a proper `spin_samples` convergence study for GKS+SOC frequencies,
and an analytic 2-component / SOC Hessian (would remove the FD noise
amplification entirely).
