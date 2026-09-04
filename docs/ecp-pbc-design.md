# PBC ECP on GPU — design

Status: design + first-cut Γ/k-point scalar `ecp_int` on branch `ecp-pbc`;
needs GPU + PBC validation.

## What PBC ECP is

The short-range ECP contribution to the periodic core Hamiltonian:

    H^ECP_k[i,j] = Σ_L e^{i k·L} Σ_{L_U} ⟨ φ_i(0) | Û_ECP(L_U) | φ_j(L) ⟩

`L` = lattice vectors, `L_U` = images of the ECP centres. ECP is short-ranged,
so both sums truncate at `cell.rcut`.

PySCF reference: `pyscf/pbc/gto/ecp.py:ecp_int(cell, kpts, intor)` — handles
`intor in ('ECPscalar', 'ECPso')`. It reuses the **molecular** C kernels
(`ECPtype_scalar_cart` / `ECPtype_so_cart` from `pyscf/lib/gto/nr_ecp.c`) inside
a BVK lattice sum (`PBCECP_loop` in `pyscf/lib/pbc/nr_ecp.c`), assembled through
`pbc.df.incore.Int3cBuilder` with a fake auxiliary s-function. The SO branch
applies the same `einsum('sxy,kspq->kxpyq', -1j·½·σ, ·)` Pauli step as the
molecular code.

Consumed by `pbc/scf/hf.py:get_hcore` — currently
`raise NotImplementedError('ECP in PBC SCF')`.

## GPU approach — supmol + `libgecp`

The molecular GPU ECP kernels (`libgecp`: `ECP_cart`, `ECP_so_cart`, validated)
already do everything the per-image integral needs. A periodic ECP matrix is a
lattice sum of those, so:

1. **Image list.** `Ls = cell.get_lattice_Ls(rcut=cell.rcut)` — lattice vectors
   within the ECP range. `nL = len(Ls)`.
2. **Supmol.** Build a `Mole` whose basis is the reference-cell shells
   ("bra", `nao_ref`) followed by the reference shells translated by every
   `L ∈ Ls` ("ket images", `nL·nao_ref`), and whose `_ecpbas` is the
   reference-cell projectors translated by every `L ∈ Ls` (ECP images). All in
   one `_atm/_bas/_env`.
3. **Kernel.** Reuse the molecular driver: `group_basis` + `sort_ecp_basis` +
   `make_full_tasks`, restricted to tasks `(ish ∈ bra groups, jsh ∈ ket groups,
   ksh ∈ ecp groups)`. `libgecp.ECP_cart` writes the `[i_bra, j_ket]` block
   (the mirror write is harmless — we slice it off).
4. **Fold + phase.** The result block `M[nao_ref, nL·nao_ref]` reshaped to
   `[nao_ref, nL, nao_ref]`; then
       H_k = Σ_L e^{i k·L} · M[:, L, :]
   Real at Γ.

Cost: one molecular-scale ECP kernel launch over a task list ~`nL×` the
molecular one — acceptable for a first landing; `nL` is small for real cells
(ECP rcut ~ few Å). Screening (`make_tasks` distance/exponent screen, already in
the tree) prunes most cross-image triples.

### Why one supmol call, not `nL` molecular calls

`libgecp.ECP_cart` already loops a task list of `(ish,jsh,ksh)` shell triples
with one CUDA block per triple — feeding it the full cross-image task list in a
single launch is both simpler and faster than `nL` separate launches.

## Phasing

1. **(this branch)** Γ + k-point **scalar** `ecp_int(cell, kpts=None)` →
   `[nkpts, nao, nao]` complex (real at Γ). Reuses `libgecp.ECP_cart` unchanged
   via the supmol; no new CUDA. Validate vs `pyscf.pbc.gto.ecp.ecp_int`.
2. Wire into `gpu4pyscf/pbc/scf/hf.py:get_hcore` (drop the `NotImplementedError`).
3. **SO** PBC ECP: same supmol, `libgecp.ECP_so_cart` → `[nkpts, 3, nao, nao]`
   real, then the Pauli step → `[nkpts, 2nao, 2nao]`.
4. Rectangular kernel entry `ECP_cart_rect` (separate bra/ket `ao_loc`, no
   mirror, output `[nao_ref, nL·nao_ref]`) to avoid the `[nao_sup, nao_sup]`
   allocation — perf/memory, not correctness.
5. Derivatives: PBC ECP gradient / stress (for `pbc/grad`), much later.

## Validation

- `ecp_int(cell)` vs `pyscf.pbc.gto.ecp.ecp_int(cell)` at Γ to 1e-9, small
  cell with a heavy-atom ECP (e.g. a 1-D chain of I or a small Si/GaAs-like
  cell with `def2` ECP on the heavy site).
- k-point: a 2×1×1 or 2×2×2 `kpts` mesh, per-k comparison; Hermiticity of each
  `H_k`; `H_{-k} == H_k.conj()`.
- Convergence in `rcut` (increase → result stable to 1e-9).
- Γ result is real (imag part < 1e-12).

## Risks / notes

- **Double-counting the ECP images.** The bra is pinned to cell 0; both the ket
  and the ECP centre are summed over `Ls`. Need `⟨i(0)|U(L_U)|j(L)⟩` for every
  `(L_U, L)` with both within range of `i(0)` — i.e. the ecp-image and ket-image
  sums are independent, not locked. Get this wrong → factor errors vs the
  reference. Cross-check term counts against `PBCECP_loop`.
- `get_lattice_Ls` rcut: ECP `cell.rcut` may be tuned for AO overlap, not the
  ECP operator range. Use `max(cell.rcut, ecp_rcut)` where `ecp_rcut` comes from
  the smallest ECP exponent + `cell.precision`.
- Low-dimensional cells (`cell.dimension < 3`): `get_lattice_Ls` handles it, but
  test 1-D/2-D separately.
- The `[nao_sup, nao_sup]` intermediate (phase 1) caps cell size; phase 4 fixes.
