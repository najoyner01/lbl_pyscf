# ECP on GPU — a big-picture-first guide

Status date: 2026-09-09. For orientation, not as a replacement for
`CLAUDE.md`, `docs/STRATEGY.md`, `docs/ecp-so-design.md`, `docs/ecp-pbc-design.md`,
or `docs/adf-parity-strategy.md` — this stitches those together into one
learning path and adds "what's left" as of right now (including the
uncommitted diff sitting in your working tree).

Read the layers in order. Each layer adds detail to the one before it; you
can stop at any layer and already have a correct (if coarse) mental model.

---

## Layer 0 — the one-paragraph picture

PySCF computes ECP (effective core potential) matrix elements with a CPU C
loop, one shell-triple `(bra shell, ket shell, ECP shell)` at a time.
`gpu4pyscf` replaces that loop with CUDA kernels that process many
shell-triples at once, one per thread block. Everything else — screening
which triples matter, batching over atoms, spin-orbit, periodic boundary
conditions — is a variation on "build a list of triples, launch one kernel
over the whole list, transform the result back to spherical AOs." **This
part of the project is essentially finished**: energies, gradients, Hessians,
spin-orbit, and periodic ECP are all implemented and validated to
1e-9–1e-13 against CPU PySCF. What's left is a handful of well-scoped
follow-ups (gradients for spin-orbit ECP, a couple of performance/robustness
items), not new architecture.

---

## Layer 1 — the architecture, in words

Three places hold the actual work; everything else is wiring.

```
gpu4pyscf/gto/ecp.py         Python driver — molecular case
gpu4pyscf/pbc/gto/ecp.py     Python driver — periodic case (reuses the same CUDA)
gpu4pyscf/lib/ecp/*.cu       CUDA kernels, built as one shared library `libgecp`
```

Consumers wire these in:

```
scf/hf.py        get_hcore()        -> get_ecp(mol)                  (energy)
grad/rhf.py       Gradients          -> get_ecp_ip_sum(mol)           (1st derivative)
hessian/rhf.py    Hessian            -> get_ecp_ipip_sum(mol)         (2nd derivative)
pbc/scf/hf.py     get_hcore()        -> pbc.gto.ecp.ecp_int(cell,...) (periodic energy)
x2c / ghf (SOC)   get_hcore()        -> get_soc_1e(mol)               (spin-orbit)
```

**The recurring pattern**, used by every one of the above:

1. Sort/group the basis by angular momentum (`group_basis` / `sort_ecp_basis`)
   — GPU kernels are templated per `(li, lj, lc)` triple of angular momenta,
   so shells of the same `(l, contraction)` are batched together. This is
   why you'll see "sorted/grouped-basis frame" mentioned constantly — mixing
   a sorted `ao_loc` with an unsorted one silently gives wrong integrals.
2. Build a task list — every `(ish, jsh, ksh)` shell-triple that needs an
   integral (`make_tasks` / `make_full_tasks`), after dropping triples that
   screening says are negligible.
3. Launch one CUDA kernel per `(li, lj, lc)` group, one thread block per task
   in that group's task list (`dim3 blocks(ntasks)`).
4. Transform the cartesian result back to spherical AOs (`coeff.T @ M @ coeff`)
   and un-sort back to the caller's AO ordering.

Once this pattern clicks, every "gap" below reads as: *what do we batch
differently, and what does the kernel compute differently* — not a new
design.

---

## Layer 2 — what actually changed vs. vanilla PySCF (conceptually)

| PySCF (CPU) | gpu4pyscf (GPU) | Why |
|---|---|---|
| C loop over shell triples, one at a time, per-thread nothing (single-core work unit = one triple) | Task list of triples; one CUDA thread **block** per triple, 128 threads = quadrature points within that triple | Exposes parallelism at two levels: across triples (blocks) and across the radial quadrature inside one triple (threads) |
| No batching needed — CPU already loops shells one at a time | Basis must be **grouped by angular momentum first** (`group_basis`) so a single kernel launch can be templated on `(li,lj,lc)` | Kernels use compile-time/templated fast paths per angular momentum for speed; a general fallback exists for untemplated `l` |
| Spin-orbit ECP (`ECPso_spinor`) computed directly in the complex spinor basis by the C kernel | Kernel stays **real**, produces `[3, nao, nao]` (the `⟨i\|l_a ΔU^SO\|j⟩` components); the complex Pauli/spinor assembly (`-i·½·σ` contraction) happens afterward in a few lines of CuPy `einsum` | No complex-valued CUDA kernel needed — this was the single biggest simplification in the whole SO-ECP port (see `docs/ecp-so-design.md`) |
| Periodic ECP: dedicated PBC C kernel (`PBCECP_loop`) doing its own lattice sum | **No new CUDA at all** — build a "supermolecule" (`_ecp_supmol`) whose basis is the reference cell plus every lattice image, then call the *same* molecular kernel (`ECP_cart` / `ECP_so_cart`) on the enlarged task list, then fold with the Bloch phase `e^{ik·L}` | Reuses validated, already-fast molecular kernels instead of writing/validating a second implementation |
| No screening TODO — CPU cost model is different | Two-level Schwarz-like screen (`_build_screen_data`, `_screen_block`, `_screen_grid`) that prunes the task list *before* the kernel launch | On GPU, a shell-triple that contributes ~0 is dead work occupying a whole thread block; pruning it from the task list is a direct speedup (measured 14×–189× on a Cu-chain benchmark) |
| Gradient/Hessian just differentiate the same C loop | `loop_ecp_ip` / `loop_ecp_ipip` batch derivative integrals **over ECP atoms** and accumulate on the fly (`get_ecp_ip_sum`, `get_ecp_ipip_sum`) rather than materializing a `[n_ecp_atoms, 3-or-9, nao, nao]` tensor | That tensor is the memory bottleneck for a large-atom-count Hessian; batching avoids ever holding it in full |

---

## Layer 3 — the five gaps, in detail (all closed, now merged to `gpu-porting`)

Each row of `CLAUDE.md`'s ECP table maps to one of these. Read them in this
order — they build on each other (screening → atom slicing reuses the same
task lists → SO-ECP reuses the type-2 kernel → validation exercises all of
it → PBC reuses the molecular kernel again).

### 3.1 Screening — `gto/ecp.py`
- `_build_screen_data(sorted_mol, ecpbas, ecp_loc)` precomputes per-shell
  exponent/extent data once.
- `_screen_block` / `_screen_grid` decide, per `(ish_range, jsh_range,
  ksh_range)` block, whether any triple in it can exceed `mol.precision`;
  whole blocks are dropped before ever building a task array.
- Toggle: `SCREEN_ECP`. Benchmark: `benchmarks/gto/ecp_screening.md`
  (screened == unscreened to 1e-13; 14×/55×/189× speedup on Cu-chain N=20/40/80).

### 3.2 ECP-atom slicing in derivatives — `gto/ecp.py`, `grad/rhf.py`, `hessian/rhf.py`
- `loop_ecp_ip(mol, ip_type='ip', ecp_atoms=None, batch_size=None)` and
  `loop_ecp_ipip(...)` are generators that yield derivative-integral batches
  per ECP-atom chunk, sized by `_ecp_atom_batch_size` against free GPU memory.
- `get_ecp_ip_sum` / `get_ecp_ipip_sum` consume the generator and accumulate
  directly into the gradient/Hessian contraction — the full
  `[n_ecp_atoms, comp, nao, nao]` tensor never exists at once.
- `get_ecp_ip` / `get_ecp_ipip` still exist as simple concatenating wrappers
  for callers that want the whole tensor (small systems, tests).

### 3.3 Spin-orbit ECP (SO-ECP) — `gto/ecp.py`, `lib/ecp/ecp_so.cu`, `docs/ecp-so-design.md`
This is the one genuinely new piece of physics in the port. Worth
understanding, not just trusting:
- `mol._ecpbas` rows are flagged `SO_TYPE_OF`: `0` = scalar projector
  (what the rest of `ecp.py` uses), `1` = spin-orbit projector.
  `sort_ecp_basis` (scalar path) discards the SO rows; `sort_ecp_basis_so`
  (`gto/ecp.py:337`) does the opposite — keeps only SO rows, and rewrites
  the special "`ul`" marker `lc == -1` to `max_l(atom) + 1` (a CPU
  convention this port has to replicate exactly).
- The kernel (`ecp_so.cu`) is `type2_cart` (the ordinary semilocal kernel)
  plus one extra step: before reducing the projector's angular index, it
  contracts with the angular-momentum operator matrix `L^a` (constants in
  `lib/ecp/so_ang_matrix.cu`, transcribed from PySCF's
  `_angular_moment_matrix`), producing **three** output components
  `a ∈ {x,y,z}` instead of one. Output layout `[3, nao, nao]`, still real.
- **A real bug worth knowing about, because it teaches the debugging
  method**: an early version was uniformly off by a factor of 2 vs.
  `mol.intor('ECPso')`. The CPU code applies a `prad[i] *= .5` that looks
  physical but is CPU-quadrature bookkeeping specific to its iterative
  radial integration — porting it literally double-counted a factor already
  present elsewhere. Diagnosed by noticing the ratio was *exactly* 0.5
  everywhere (uniform ratio ⇒ a missing/extra constant, not a real bug in
  the angular part) — a generally useful heuristic: a uniform scalar
  mismatch across many test cases means "constant factor," a
  case-dependent mismatch means "a term is wrong."
- Python entry points: `get_ecp_so(mol) -> [3, nao, nao]` real, and
  `get_soc_1e(mol) -> [2·nao, 2·nao]` complex (`gto/ecp.py:429`), which is
  the actual GHF-consumable Pauli/spinor assembly:
  `einsum('sxy,spq->xpyq', -1j·0.5·PauliMatrices, get_ecp_so(mol))`.
- Validated to 1e-10 against `mol.intor('ECPso')` and the CPU GHF
  `get_hcore` SOC block, including the explicit-SO-projector case and an
  s..g basis sweep.

### 3.4 Validation breadth — `gto/tests/test_ecp_sweep.py`
- 1526 A100 subtests: `get_ecp` / `_ip` / `_ipip` / `_so` / `get_soc_1e` vs.
  `mol.intor`, across ~12 scalar + 6 SO ECP sets × ~19 elements (Na–Bi) ×
  cartesian + spherical × probe-basis angular momentum s..g.
- This is the thing to run first whenever you touch `lib/ecp/*.cu` or
  `gto/ecp.py` — it's the fastest way to know if a change broke something,
  and it's what caught the SO-ECP factor-of-2 bug above.

### 3.5 PBC ECP — `pbc/gto/ecp.py`, `pbc/scf/hf.py`, `docs/ecp-pbc-design.md`
- `_ecp_supmol(Ls, ket_img_ids, sorted_mol, sorted_ecpbas)` builds one `Mole`
  containing the reference cell shells ("bra") followed by every lattice
  image of those shells ("ket images") and every lattice image of the ECP
  centers — then it's just a (bigger) molecular ECP problem.
- `_lattice_ecp_cart(cell, intor)` batches over ket images
  (`_image_batch_size`, sized against `get_avail_mem()`) so the transient
  `[comp, nao_ref·(1+batch), nao_ref·(1+batch)]` matrix never exceeds GPU
  memory for large cells — the ECP-image sum stays unbatched (cheap; ECP
  is short-ranged so few images matter).
- `ecp_int(cell, kpts=None, intor='ECPscalar'|'ECPso')` folds the batched
  result with the Bloch phase `Σ_L e^{ik·L}` and is what `pbc/scf/hf.py`
  calls from `get_hcore`.
- Validated at Γ and on k-point meshes to 5e-9 against
  `pyscf.pbc.gto.ecp.ecp_int`; Hermiticity and `H_{-k} = H_k.conj()` checked;
  Γ-point result checked to be real to <1e-12.
- **The one subtlety that will bite you if you re-derive this**: the ket
  images and the ECP-center images are summed *independently* — you need
  `⟨i(0)|Û(L_U)|j(L)⟩` for every `(L_U, L)` pair in range, not a single
  locked-together sum. Getting this wrong produces a systematic
  double-counting/undercounting error, not a crash — cross-check term
  counts against the CPU reference, don't just eyeball the energy.

---

## Layer 4 — what's actually left to finish the GPU ECP port

Ranked by what blocks what. Nothing here is architecture-changing; it's all
extending patterns already proven in Layer 3.

### 4.0 Right now, uncommitted in your working tree
`git status` on `gpu-porting` shows unstaged edits to `build.sh`,
`gpu4pyscf/gto/ecp.py`, and `gpu4pyscf/lib/ecp/nr_ecp_driver.cu`, plus an
untracked `benchmarks/gto/run_ecp_bench.sbatch`. The `.cu`/`.py` edits are a
real bug fix, not exploratory: aggressive screening can empty a task group
entirely, and launching a CUDA kernel with `dim3 blocks(0)` is undefined
behavior. The fix adds an `ntasks == 0 → return 0` early-out to all four
`ECP_*_cart` driver entry points, and a matching `if len(tasks_all[i,j,k]) ==
0: continue` skip in the Python task-loop in `ecp.py`. **Action:** rebuild
`libgecp`, rerun `test_ecp_sweep.py` (this is exactly the kind of edge case
it's built to catch on small/pathological systems), then commit — this is a
correctness fix that should land before anything else, since it's a latent
crash/UB risk in the screening path that's already merged.

### 4.1 Spin-orbit ECP gradients — the biggest real gap
- SO-ECP **energies** work (molecular + PBC). SO-ECP **derivatives** do not
  exist anywhere — not in PySCF's own C library either (only the scalar
  `ECPso` energy integral exists on the CPU side; PySCF's Dirac/spinor
  gradient code (`grad/dhf.py`) doesn't cover this case). This blocks
  geometry optimization for any spin-orbit-ECP method (relevant for heavy
  elements / actinides, per `docs/adf-parity-strategy.md`).
- Plan (sketched in `docs/adf-parity-strategy.md` §3 and mirrored from how
  `ecp_type2_ip.cu` extends `ecp_type2.cu` for the scalar case): extend
  `lib/ecp/ecp_so.cu` with an `so_ip` variant the same way the scalar kernel
  was differentiated, then wire it through a `get_ecp_so_ip` and into
  `GHF.Gradients()`. Validate against finite difference of `get_ecp_so`
  itself (there's no CPU analytic reference to check against here — this is
  new physics, not a port, so finite-difference is the *only* independent
  check; be more paranoid than usual).
- This is squarely GPU-ECP work (not the CC track), and it's the one item
  standing between "SOC energies work" and "SOC geometry optimization
  works."

### 4.2 SO-ECP performance follow-ups
- `so_cart` currently runs the **general** (non-templated) kernel path only.
  The scalar kernels have `(li,lj,lc)`-templated fast paths; SO-ECP doesn't
  yet. Low risk, mechanical, same pattern as the scalar templating —  worth
  doing once SO-ECP is used in any performance-sensitive workflow.
- Spinor-basis / GHF/x2c wiring beyond the `get_soc_1e` 2-component form
  (needed only if a true 4-component/spinor SCF path is ever wired up —
  check whether the science actually needs this before investing).

### 4.3 PBC ECP gradient / stress
- Explicitly deferred (`docs/ecp-pbc-design.md` phase 5) — needed for
  periodic geometry optimization / MD with ECP, not needed for periodic
  single-point energies (which already work). Lowest priority unless a
  periodic-ECP optimization workflow is on the near-term roadmap.

### 4.4 Blackwell / CUDA < 13.1 nvcc miscompile
- Tracked in `lib/ecp/CMakeLists.txt` as a known miscompile, currently
  worked around by disabling an optimization flag rather than fixed at the
  root. Fine for now on Perlmutter's toolchain, but revisit before deploying
  on newer hardware/toolchains — "disable optimization" is a workaround, not
  a fix, and could silently regress if the guard condition is ever slightly
  wrong.

### 4.5 What "done" means here, concretely
Per `CLAUDE.md`'s validation bar, ECP clears all four requirements already:
value-tested vs. CPU PySCF (1e-9 to 1e-13, well past the usual 1e-8/1e-6
bar), `to_cpu`/`to_gpu` round-trips exercised, mixed numpy/cupy inputs
handled, and benchmarks committed under `benchmarks/gto/`. Nothing in
Layer 4 needs to happen before you consider the *scalar* ECP path
production-ready for SCF/DFT energies, gradients, and Hessians, molecular or
periodic. Layer 4 items are specifically about the spin-orbit extension and
a couple of hardening/perf items, not about the core port.

---

## Suggested reading/rebuild order if you want to get hands dirty

1. Rebuild and run the sweep test first, before reading further code — it's
   your ground truth and the fastest way to confirm your environment works:
   ```sh
   cd gpu4pyscf
   cmake --build build/temp.gpu4pyscf -j 8
   pytest gpu4pyscf/gto/tests/test_ecp_sweep.py -v
   ```
2. Read `gto/ecp.py` top to bottom once, ignoring the CUDA — it's plain
   Python/CuPy and it *is* the algorithm (task lists, grouping, screening,
   transform-back). This gives you Layer 1–2 for free.
3. Read one CUDA kernel, the simplest: `lib/ecp/ecp_type1.cu` (the local
   `U_L` term — no angular coupling, so it's the least code). Then
   `ecp_type2.cu` (semilocal — this is what `ecp_so.cu` extends).
4. Read `docs/ecp-so-design.md` end to end — it's short, and it documents
   the factor-of-2 bug hunt, which is a better teacher than the working code.
5. Read `docs/ecp-pbc-design.md` — by this point the "supmol, reuse the
   molecular kernel" trick will feel obvious rather than clever.
6. Land the uncommitted zero-task fix (§4.1 above) as your first real
   contribution — it's small, it's already written, and testing it forces
   you through the build/test loop end to end.
