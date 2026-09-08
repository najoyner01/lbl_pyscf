# GHF nuclear gradient — design

Status: 2026-09-04, design only (not implemented). Step 2 of the SOC-gradient
roadmap in `docs/adf-parity-strategy.md` §1b.

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
