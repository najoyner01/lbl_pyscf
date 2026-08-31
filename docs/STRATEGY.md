# Strategy: porting PySCF ECP and Coupled Cluster to GPU

Status date: 2026-08-31. Companion to `../CLAUDE.md`.

---

## 0. Guiding decisions

- **Target repo is `gpu4pyscf`.** Work *with* the plugin architecture: reuse its
  integral engines (`libgint`, `libgvhf`, `gpu4pyscf.df`, `gpu4pyscf.scf.int4c2e`),
  its `contract`/cuTENSOR layer, DIIS, logger, multi-GPU helpers, and the
  `to_cpu`/`to_gpu` machinery. Do not rebuild what exists.
- **CPU PySCF is the oracle.** Every phase ships with a value test against
  `pyscf/pyscf/...` to 1e-8 (energies) / 1e-6..1e-9 (tensors).
- **Density fitting is the default** for correlated methods on GPU. It caps
  memory at O(N³) aux integrals, matches how `gpu4pyscf.dft`/`mp` already work,
  and sidesteps the explicit `vvvv` tensor. Conventional (incore/AO-direct)
  paths are kept only for small systems and cross-checks.
- **Contraction-first.** CC intermediates are ported as `contract(...)` calls
  translated almost line-for-line from `pyscf/cc/rintermediates.py` and
  `uintermediates.py`. Drop to custom CUDA only where a contraction cannot be
  tiled efficiently (the `(T)` triples, possibly `Wvvvv`).
- **Land small, vertically.** Each phase is an end-to-end working method with
  tests and a benchmark, not a layer across all methods.

---

## Part A — ECP

### A0. Baseline (already in tree)

`gpu4pyscf/gpu4pyscf/gto/ecp.py` + `lib/ecp/*.cu` (built as `libgecp`) already
provide scalar ECP, first derivatives (gradient), and second derivatives
(Hessian), wired into `scf/hf.py`, `grad/rhf.py`, `hessian/rhf.py`, with tests
and a benchmark. **ECP is mostly done.** The work below closes specific gaps.

### A1. Screening (perf) — ~1 week

- **Problem:** `make_tasks` / `make_full_tasks` enumerate every
  (shell-pair × ECP-group) triple; `# TODO: Add screening`.
- **Do:** precompute per-shell-pair Schwarz-like bounds and shell–ECP-center
  distances; drop triples whose estimated contribution < `mol.precision`.
  Build the task list on host, ship the pruned index array to the kernel
  (kernels already take a `tasks` array — no kernel change needed for a first
  cut).
- **Validate:** energies/gradients unchanged to 1e-10 on the existing ECP test
  molecules; benchmark on a large ECP system (e.g. a metal cluster,
  100+ heavy atoms) — expect the ECP fraction of SCF to drop substantially.

### A2. ECP-atom slicing in derivatives (perf + correctness hygiene) — ~3 days

- `grad/rhf.py:344` and `gto/ecp.py:get_ecp_ip/ipip` accept `ecp_atoms` but the
  gradient path passes all of them. Thread the active-atom subset through so
  per-atom derivative passes only touch relevant ECP centers.

### A3. Spin-orbit ECP (SO-ECP) — ~3–5 weeks (largest ECP item)

- **Reference:** `pyscf/pyscf/gto/ecp.py:so_by_shell`, C kernel `ECPso_spinor`
  in `pyscf/pyscf/lib/gto/nr_ecp.c`; basis flag `gto.SO_TYPE_OF`.
  Currently `gpu4pyscf` `sort_ecp_basis` **discards** SO projectors.
- **Deliverable:** `get_ecp_so(mol)` returning the SO-ECP contribution in the
  spinor (or 2-component Pauli) basis, matching `so_by_shell` assembled over
  shells to 1e-9.
- **Plan:**
  1. New CUDA kernel `lib/ecp/ecp_so.cu` — reuse `type2_ang_nuc` radial/angular
     machinery; add the `i/2 ⟨σ·l U(r)⟩` angular factor and complex output.
  2. Python driver: keep SO projectors in `sort_ecp_basis`, add a spinor
     cart→spinor transform (see `gpu4pyscf/x2c/x2c.py` for the spinor plumbing).
  3. Wire into `gpu4pyscf/x2c` and any GHF/GKS spinor SCF path.
- **Risk:** complex-valued kernels + spinor indexing are error-prone; test
  shell-by-shell against `so_by_shell` before the full matrix.

### A4. PBC ECP — ~3–4 weeks — **deferred** (after CC phase 1–2)

- Reference `pyscf/pyscf/pbc/gto/ecp.py`, `pyscf/pyscf/lib/pbc/nr_ecp.c`:
  lattice-sum of the molecular ECP over image cells.
- Approach: reuse the molecular `libgecp` kernels per image, sum with the
  existing `gpu4pyscf/pbc` k-point / lattice-sum infrastructure. Screen images
  by ECP decay.

### A5. Validation & benchmark sweep — ~1 week, ongoing

- Matrix of {`crenbl`, `lanl2dz`, `def2-ECP`, `ccECP`, f-in-core} ×
  {basis L = s..g} × {energy, grad, Hessian} vs PySCF to 1e-10.
- Publish a `benchmarks/gto/benchmark_ecp.py` extension with CPU-vs-GPU speedup
  across 10–200 heavy atoms; baseline JSON in `tests/benchmark_results/`.

**ECP exit criteria:** SO-ECP matches PySCF; screening gives >5× on large ECP
systems with zero accuracy loss; validation matrix green.

---

## Part B — Coupled Cluster

Reference tree: `pyscf/pyscf/cc/`. GPU baseline: `gpu4pyscf/gpu4pyscf/cc/ccsd_incore.py`
(RHF closed-shell CCSD, incore ERIs, `update_amps` already on GPU).

### B1. DF-RCCSD — ~6–8 weeks  ← **start here**

**Goal:** closed-shell RCCSD that scales to ~1500 basis functions on one GPU.

- **ERIs:** build `Lov`, `Loo`, `Lvv` (`(pq|P)` half-transformed 3-center) with
  `gpu4pyscf.df` / the `gpu4pyscf/mp/dfmp2.py` pipeline. Form `ovov`, `oovv`,
  `ovvo`, `oooo`, `ovoo` on the fly from `L`. Never materialize `vvvv`:
  contract the ladder term as `(L_vv, L_vv) → Wvvvv·t2` via batched GEMM, or via
  the existing `_direct_ovvv_vvvv` AO path as a fallback / check.
- **Amplitude update:** extend `ccsd_incore.update_amps`; port
  `pyscf/cc/rintermediates.py` intermediate-by-intermediate as `contract`
  calls. `frozen` core/virtual support.
- **Driver:** `kernel` with DIIS (`gpu4pyscf.lib.diis`), MP2 `init_amps`
  (reuse `gpu4pyscf.mp`), `energy`, `e_corr`/`e_tot`, `_finalize`,
  `to_cpu`/`to_gpu`. Class `gpu4pyscf.cc.dfccsd.RCCSD` (or fold into `cc/ccsd.py`
  with `.density_fit()`).
- **Memory:** block over occupied index `i`; keep two `t2` blocks resident;
  `get_avail_mem()` guard; optional host spill for `ovvv`.
- **Tests:** vs `pyscf.cc.dfccsd` and `pyscf.cc.ccsd` (canonical) — `e_corr` to
  1e-8, `t1`/`t2` to 1e-6; `to_cpu` round-trip; H₂O, C₆H₆/cc-pVDZ, a
  ~40-atom case. Benchmark vs CPU CCSD.

### B2. RCCSD(T) — ~4–6 weeks

- Reference `pyscf/cc/ccsd_t.py` (+ `_ccsd` C helper).
- O(N⁷) triples. First cut: tiled loop over occupied triples `(i,j,k)` with
  `contract` on the `(nvir³)` blocks. If bandwidth-bound, write
  `lib/cc/ccsd_t.cu` — register-tiled `(vvv)` contraction, one triple per block,
  accumulate `e_(T)`. This is the highest-value custom-kernel opportunity in the
  whole project (large speedups reported by other GPU CC implementations).
- **Tests:** `e_(T)` vs `pyscf.cc.ccsd_t` to 1e-8 on H₂O, and a 20–30 atom case
  for timing.

### B3. Λ equations + RDMs + RCCSD analytic gradient — ~6–8 weeks

- Reference `ccsd_lambda.py`, `ccsd_rdm.py`, `pyscf/grad/ccsd.py`.
- `solve_lambda` reuses the B1 intermediates (transpose structure). `make_rdm1`,
  `make_rdm2` (with `ao_repr` option). Gradient: contract RDMs with
  derivative integrals — reuse `gpu4pyscf.df.grad` and `gpu4pyscf.grad`
  machinery; **ECP gradient term already exists** (`get_ecp_ip`).
- Wire `nuc_grad_method` / `Gradients`; geometry-optimization smoke test via
  `geomeTRIC`.
- **Tests:** RDM traces, `e_tot` from RDMs, gradient vs `pyscf` CCSD gradient to
  1e-6; finite-difference check.

### B4. UCCSD / UCCSD(T) — ~6–8 weeks

- Reference `uccsd.py`, `uintermediates.py`, `uccsd_t.py`, `dfuccsd.py`.
- Spin-block the B1/B2 machinery (aa/ab/bb). Largest single porting effort by
  line count; mechanical once B1 patterns are established.
- **Tests:** vs `pyscf.cc.uccsd` on radicals (e.g. •OH, •CH₃) to 1e-8.

### B5. EOM-CCSD (EE / IP / EA) — ~8–10 weeks

- Reference `eom_rccsd.py` (2100 loc), later `eom_uccsd.py`.
- Port the `matvec` (sigma-vector) builds + diagonal preconditioner; run
  Davidson on GPU (`gpu4pyscf.lib` Krylov / cupy). Reuse B1 intermediates.
- Order: EE-RCCSD → IP/EA-RCCSD → UCCSD variants.
- **Tests:** excitation / ionization energies vs `pyscf` to 1e-6 on small
  molecules.

### B6. Later / opportunistic

- GCCSD (`gccsd.py`) — needed for some EOM and relativistic paths.
- BCCD, QCISD, CC2/CCSDT-n — thin wrappers once CCSD infra exists.
- PBC-CCSD (`pyscf/pbc/cc/`) — major, own project.
- FP32/mixed-precision CC (cf. `dfmp2.py` already exposes `fp_type`).
- Multi-GPU t2 sharding for CCSD / (T).

**CC exit criteria per phase:** method matches PySCF to tolerance, `to_cpu`
round-trips, benchmark shows net speedup vs CPU PySCF on ≥20-atom systems,
tests + benchmark committed.

---

## Suggested sequencing (single developer + Claude)

| Quarter | ECP | CC |
|---------|-----|-----|
| 1 | A1 screening, A2 slicing, A5 validation | B1 DF-RCCSD |
| 2 | A3 SO-ECP | B1 finish, B2 RCCSD(T) |
| 3 | A4 PBC-ECP (if needed) | B3 Λ/RDM/gradient |
| 4 | — | B4 UCCSD, start B5 EOM |

ECP items are small and parallelizable against the long CC push. If SO-ECP is
not needed for the science driving this project, do A1/A2/A5 only and put all
remaining effort into CC.

---

## Cross-cutting risks

| Risk | Mitigation |
|------|-----------|
| GPU memory ceiling on CC | DF by default; block occ index; `get_avail_mem` guards; host spill; multi-GPU later |
| cuTENSOR/cupy version fragility | pin via `requirements.txt`; CI on the pinned combo |
| Numerical drift vs CPU accumulating over CC iterations | test `e_corr` to 1e-8 *and* converged amplitudes to 1e-6; Kahan/compensated sums only if a real regression shows |
| Custom `(T)` kernel complexity | ship the tiled-`contract` version first; kernel is an optimization, not a blocker |
| nvcc miscompiles (seen in `lib/ecp/CMakeLists.txt` for Blackwell/CUDA<13.1) | keep arch/toolchain guards; test on ≥2 CUDA versions |
| SO-ECP complex/spinor indexing bugs | validate shell-by-shell vs `so_by_shell` before assembling the full operator |
| Scope creep across methods | land one vertical method fully (tests+bench) before starting the next |
