# Spin-orbit ECP (SO-ECP) on GPU — design

Status: design + Python scaffold landed on branch `ecp-so`; CUDA kernel pending.

## What SO-ECP is

An ECP with spin-orbit projectors carries extra semilocal terms

    H^SO = Σ_l  ΔU^SO_l(r) · |l⟩ ( l·s ) ⟨l|

with `s = ½σ` (Pauli). PySCF stores the SO projectors in `mol._ecpbas` rows
flagged `SO_TYPE_OF == 1` (scalar projectors are `== 0`); the GPU scalar path
(`sort_ecp_basis` in `gpu4pyscf/gto/ecp.py`) currently **drops** those rows.

`mol.has_ecp_soc()` is true iff any `_ecpbas[:, SO_TYPE_OF] == 1`.

## How PySCF evaluates it (the reference)

Two layers — the GPU port mirrors this split:

1. **Real cartesian kernel** `ECPtype_so_cart` (`pyscf/lib/gto/nr_ecp.c`) →
   `mol.intor('ECPso')` returns a **real** array `[3, nao, nao]` (sph):
   the three components `⟨i| l_a ΔU^SO |j⟩`, `a ∈ {x,y,z}`.
   It is the scalar type-2 semilocal machinery with four changes:
   - iterate only projectors with `SO_TYPE_OF == 1`;
   - `lc == -1` (the `ul` term) is treated as `lc = max_l(atom) + 1`;
   - the reduced radial integral is scaled by `½` (`prad[i] *= .5`);
   - **`transform_angj`**: the shell-`j` angular factor `angj[j, m, q]`
     (`m` = projector component, `0..2lc`) is contracted with the
     angular-momentum operator matrix `L^a_{m'm}` (`_angular_moment_matrix[lc]`,
     shape `[3, 2lc+1, 2lc+1]`), giving three output components instead of one:
         jmm_angj[a, j, m', q] = Σ_m  angj[j, m, q] · L^a_{m' m}
     then three separate `gctr` accumulators `gctrx/y/z`.

2. **Spinor / Pauli assembly** (Python, trivial):

   GHF `get_hcore` (`pyscf/scf/ghf.py`):
   ```python
   s = .5 * lib.PauliMatrices                       # [3,2,2] complex
   ecpso = einsum('sxy,spq->xpyq', -1j*s, mol.intor('ECPso'))  # [2,nao,2,nao]
   hcore += ecpso.reshape(2*nao, 2*nao)
   ```
   `ECPso_spinor` (`nr_ecp.c`) does the same contraction to produce integrals
   directly in the spinor basis (`cart2spinor` + Pauli), with an extra `½`
   discussed in pyscf issue #378.

**Consequence for the GPU port: no complex CUDA kernel is needed.** The kernel
produces the real `[3, nao_cart, nao_cart]` tensor; the Pauli/spinor step is a
handful of CuPy `einsum`s in Python.

## GPU plan

### Kernel — `gpu4pyscf/lib/ecp/ecp_so.cu`

Adapt `type2_cart` from `ecp_type2.cu` (1 task = `(ish, jsh, ksh)` per block,
128 threads = quadrature points). Changes:

| type2_cart | ecp_so |
|---|---|
| scalar projectors (driver filters `SO_TYPE_OF == 0`) | driver filters `SO_TYPE_OF == 1` |
| `gctr[i + j*nao]` single component | `gctr[(a*nao + i) + j*nao*3]` — 3 components `a` |
| final contract `(k+l)pq,kimp,ljmq->ij` | insert `L^a_{m'm}` on the `angj` m-index before the `m` reduction; unroll `a=0,1,2` |
| — | multiply accumulator by `0.5` (radial ½ factor) |
| `lc` from `ecpbas[ANG_OF]` | driver passes `lc = -1 → maxl(atom)+1`; skip `lc > ECP_LMAX` (=4) with an error, matching CPU |

`L^a` constants: `gpu4pyscf/lib/ecp/so_ang_matrix.cu` — `__constant__` arrays
`_l_op_{s,p,d,f,g}[3*(2l+1)*(2l+1)]`, transcribed verbatim from
`_angular_moment_matrix_*` in `pyscf/lib/gto/nr_ecp.c` (provenance: the
`angular_moment_matrix(l)` docstring there; `l·1j` is real and antisymmetric,
`M = -Mᵀ`, which `generate_so_ang_matrix.py` asserts).

### Driver entry — `nr_ecp_driver.cu`

`extern "C" int ECP_so_cart(double *gctr, const int *ao_loc, int nao,
const int *tasks, int ntasks, const int *ecpbas, const int *ecploc,
const int *atm, const int *bas, const double *env, int li, int lj, int lc)`
with the same `li*100 + lj*10 + lc` templated fast-path switch + general
fallback as `ECP_cart`. `gctr` layout `[3, nao, nao]`, C-contiguous.

Add `ecp_so.cu` (and `so_ang_matrix.cu`) to the `#include` list in
`nr_ecp_driver.cu`; no `CMakeLists.txt` change (single `gecp` target already
globs via the driver).

### Python driver — `gpu4pyscf/gto/ecp.py`

- `_sort_ecp_basis_so(_ecpbas)` — like `sort_ecp_basis` but **keeps**
  `SO_TYPE_OF == 1`, drops `== 0`, and rewrites `lc == -1` rows to
  `maxl(atom)+1` before grouping.
- `get_ecp_so(mol) -> cp.ndarray [3, nao, nao]` (sph, real) — mirrors
  `get_ecp`: `group_basis`, task list (reuse `make_tasks` + screening),
  `libecp.ECP_so_cart` per `(li,lj,lc)` group, `coeff.T @ M @ coeff` per
  component.
- `get_soc_1e(mol, comp='so') -> cp.ndarray [2*nao, 2*nao] complex` — the GHF
  contribution: `einsum('sxy,spq->xpyq', -1j*0.5*pauli, get_ecp_so(mol))`.

### Consumers (later, separate change)

`gpu4pyscf/scf/ghf.py` (if/when GHF+SOC is wired) and `gpu4pyscf/x2c/`.
Out of scope for the first landing — `get_ecp_so` + tests first.

## Validation

1. `get_ecp_so(mol)` vs `mol.intor('ECPso')` (CPU), all three components, to
   1e-10, on a small heavy-atom case with a real SO-ECP set
   (e.g. `ecp='crenbl'` on I or Pb, or an explicit `parse_ecp` with SO blocks).
2. `get_soc_1e(mol)` vs the CPU GHF `get_hcore` SOC block
   (`einsum('sxy,spq->xpyq', -1j*.5*PauliMatrices, mol.intor('ECPso'))`).
3. Screened == unscreened (SO uses the same `make_tasks` screen).
4. Basis-L sweep s..g on both AO and projector `lc`.
5. `lc == -1` (`ul`-as-Lmax) path exercised.

## Phasing

1. **(this branch, done)** design doc, `_l_op` constants + generator/verifier,
   Python `get_ecp_so` / `get_soc_1e` skeleton, test file (xfail until kernel).
2. `ecp_so.cu` general (non-templated) kernel + `ECP_so_cart` driver entry;
   get test 1 green on Perlmutter.
3. Templated `(li,lj,lc)` fast paths; screening; basis-L sweep.
4. Spinor-basis path + GHF/x2c wiring (separate branch).

## Notes / risks

- `ECP_LMAX = 4` cap: SO projectors with `lc > 4` (incl. `ul`-fallback that
  pushes past 4) are unsupported — match the CPU `fprintf` + skip.
- The `½` factor: CPU applies it in `ECPtype_so_cart` radial accumulation.
  Keep it in the kernel (not Python) so `get_ecp_so` matches `mol.intor('ECPso')`
  directly.
- Sign/`-1j`: lives entirely in the Python Pauli assembly; the kernel is real.
- cuTENSOR path for the `L^a` contraction is unnecessary — it's a
  `(2lc+1)×(2lc+1)` matrix, do it in-kernel.
