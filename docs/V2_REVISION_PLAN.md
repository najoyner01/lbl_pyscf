# gpu4pyscf v2 revision plan — engine efficiency

Status date: 2026-10-01. Companion to `STRATEGY.md` and `../CLAUDE.md`.
Input data: `benchmarks/inorganic_benchmark/results_tm.jsonl` (transition-metal
basis ladder, 19 complexes × DZ/TZ/QZ × PBE/B3LYP/PBE0, A100-40GB vs 16 CPU
threads) plus the component profiles in
`benchmarks/inorganic_benchmark/profiling/` (jobs 59166484, 59167336, dedicated
A100-SXM4 node). **No code has been changed yet; this is the plan.**

---

## 1. What the ladder says

All 342 runs converged; worst GPU/CPU energy disagreement is 2.4e-9 Ha. The
problem is purely speed:

| rung | AO range | median speedup PBE | median B3LYP / PBE0 | GPU faster in |
|---|---|---|---|---|
| DZ | 46–140 | 0.61× | 0.15× | 0 / 57 |
| TZ | 58–262 | 0.45× | 0.23× | 10 / 57 |
| QZ | 142–492 | 1.44× | 0.67× | 23 / 57 |

Three structural facts, all visible in the raw timings:

1. **Hybrids are 3–7× slower than GGAs on the GPU at the same size, but equal on
   the CPU.** AgCl/TZVP: PBE 1.5 s vs B3LYP 12.6 s on GPU; 0.39 vs 0.39 s on CPU.
   The exchange build is the whole story.
2. **Cost does not track AO count — it tracks contraction depth.** AgCl/QZVP
   (142 AO, 2 atoms) costs 31 s with B3LYP; MoF6/SVP (115 AO, 7 atoms) costs
   5 s. The difference is Cl's 10-primitive s shell and 8-primitive p shell.
3. **Per-call fixed costs are ~1 s** before any SCF cycle runs: B3LYP
   `SCF initialization` is 1.2 s at 77 AO and 2.3 s at 492 AO, versus a full
   CPU SCF of 0.4 s / 118 s respectively.

## 2. Measured component breakdown (dedicated A100, ms)

| component | AgCl SVP 49 | AgCl TZ 77 | AgCl QZ 142 | MoF6 TZ 226 | IrCl6 QZ 492 |
|---|---:|---:|---:|---:|---:|
| `jk.get_k` (one K build, shipped) | 412 | 729 | 1560 | 179 | 1691 |
| `jk.get_k`, fully decontracted (expt.) | **7** | **8.7** | **28** | **24** | **159** |
| `j_engine.get_j` (one J build) | 18 | 34 | 129 | 140 | 503 |
| `nr_rks` vxc, B3LYP | 8 | 8 | 15 | 21 | 24 |
| `jk._VHFOpt.build` (K screening setup) | 98 | 131 | 223 | 73 | 229 |
| same, fully decontracted | 20 | 20 | 30 | 22 | 33 |
| `get_hcore` nuclear part (`Int3c2eOpt.build`) | 111 | 151 | 245 | 80 | 252 |
| `get_ecp` | 30 | 34 | 46 | 30 | 56 |
| `get_init_guess` | 70–230 | 78–130 | 68 | 68 | 83 |
| `int1e_kin` | 16 | 22 | 26 | 14 | 26 |
| B3LYP SCF wall, shipped | 6.8 s | 12.7 s | 31.3 s | 8.4 s | 76 s |
| B3LYP SCF wall, decontracted K (expt.) | **1.6 s** | **1.75 s** | **6.9 s** | **6.1 s** | **46 s** |
| CPU B3LYP SCF (16 thr, from ladder) | 0.24 s | 0.39 s | 1.0 s | 12.8 s | 118 s |

The "decontracted" rows are the same `_VHFOpt` with `diffuse_cutoff=1e200`
(what `j_engine` already uses), monkey-patched in the profiler; K matrices agree
with the shipped path to ≤5e-11 and SCF energies to 1e-10.

## 3. Root causes, with code locations

### RC1 — exchange kernel runs an nprim⁴ serial loop per shell quartet
`scf/jk.py:421` builds the sorted basis with `decontract=True,
diffuse_cutoff=0.3`. `gto/mole.py:1184-1205` (`_optimize_contraction`, nctr==1
branch) therefore keeps every primitive with exponent > 0.3 inside one
segment. For def2 on 4d/5d metals and on Cl that is almost all of them
(AgCl/TZVP groups: s nprim 7,3,2,1; p 5,4,1; d 4,2,1). The Rys K kernel then
iterates `for klp < kprim*lprim { for ijp < iprim*jprim {...}}`
(`lib/gvhf-rys/rys_contract_k.cu:195-221`) **inside one thread block with a
`__syncthreads()` per iteration**. Measured on `(ss|ss)` 1×1 pairs: 0.021 ms at
nprim⁴=1, 9.5 ms at nprim⁴=2401. `j_engine.py:100` uses `diffuse_cutoff=1e200`
(full decontraction), which is why J is 5–20× cheaper than K here.

Side effects of the same choice: 11–12 (l,nprim) groups instead of 4–5, hence
2431 (TZ) / 3367 (QZ) kernel launches per K build vs 65 / 140, mean 5.6 of 108
SMs active, and `_VHFOpt.build` cost scaling as n_groups².

### RC2 — screening setup is launch-latency bound
`_cache_q_cond_and_non0pairs` (`jk.py:984-1101`) calls
`_group_by_split_points` (`pbc/scf/rsjk.py:2268`) once per (i,j) group pair;
each does 8 `cp.where` on arrays of a few dozen elements → 528 tiny kernels and
1.6 s on a contended GPU, ~100 ms on a quiet one. Also 83 host `np.arange`
calls for `shl_pair_offsets`.

### RC3 — one-electron setup rebuilt from scratch every SCF
`scf/hf.py:149-151` computes the nuclear attraction through
`contract_int3c2e_auxvec`, which constructs a fresh `Int3c2eOpt` (3-center
screening tables, pair cache) each call: 110–250 ms of which the actual
contraction is 5 ms. `get_hcore` is not cached on the SCF object, `get_ecp`
re-sorts the basis and rebuilds its task lists each call (30–60 ms), and
`get_init_guess` costs 70–230 ms. Together ≈0.4–0.6 s of the 1.2–2.3 s
"SCF initialization"; the rest is the first J/K build (RC1).

### RC4 — J and K built in separate engines for hybrids
`dft/rks.py:119,125` calls `get_j` (MD J-engine) then `get_k` (Rys). Each pays
its own `dm_cond`, pool allocation, and launch sequence. The fused
`RYS_build_jk` path exists (`jk.py:_VHFOpt.get_jk`) but RKS never uses it.

### RC5 — f/g quartets have no unrolled kernels and run one quartet per block
Unrolled K kernels (`unrolled_rys_k.cu`) cover 25 quartets, all `li≤3,
max(lj,lk,ll)≤2`. Anything with a g shell, or two f shells, goes to the general
kernel with `nsq_per_block=1`, and (gggg)-class launches split into 2–3
sequential launches with a blocking `cudaMemcpyToSymbol`. At QZ this is
15–25 % of K time; after RC1 is fixed it becomes the dominant kernel cost.

### RC6 — assorted per-call waste in the JK path
- `pool = cp.empty(workers*QUEUE_DEPTH+3)` per call: 108 MiB per K build
  (`jk.py:538,844`); `QUEUE_DEPTH` is 262144 in Python but 65536 in
  `vhf.cuh:31` (4× over-allocation, latent inconsistency).
- `cudaGetDeviceProperties` + `cudaMemset(head)` on every launch
  (`rys_contract_k.cu:656-660`).
- `vj` allocated and zeroed even when only K is requested (`jk.py:531`).
- Host round trips: `vj_xyz.get()` in `j_engine.py:246`,
  `transform_cart_to_xyz` / `transform_xyz_to_cart` through host in the Rys
  `get_j` (`jk.py:664-766`).
- `extract_pgto_params(mol,'diffuse')` recomputed per call (`jk.py:508,813`).

### RC7 — ECP is correct but still 2–4× slower than 16 CPU threads below 150 AO
`get_ecp` is 30–60 ms; CPU `ECPscalar` is 9–40 ms at the same sizes and only
loses above ~200 AO (MoF6/TZ: 30 vs 130 ms). The driver re-runs
`group_basis`, `sort_ecp_basis`, `_build_screen_data` and `make_tasks` on the
host per call, launches one kernel per (li,lj,lk) group, and falls to the
dynamic-shared-memory general kernel for l≥3 pairs and all g functions.
Not on the critical path once RC1–RC4 are fixed, but it is the ECP-specific
item in this project's scope and gates the `l>4` rung (`systems_tm.py` notes
cc-pV5Z-PP/ h,i functions abort).

## 4. Revision plan, ranked by expected payoff on the ladder

Each item: change → files → validation → expected effect.

### P1. Adaptive contraction policy for the exchange engine  *(days; ~5–18× on hybrids ≤300 AO)*
- Make `diffuse_cutoff` a `_VHFOpt` attribute and choose it per molecule:
  fully decontract (`1e200`) when `max(nprim)^4 * n_shell_pairs` is below a
  threshold or `nbas` is small; keep the 0.3 segment policy only when full
  decontraction would blow up the pair count (large organic systems, where the
  current policy was tuned). Expose `mf._opt_gpu` override and a
  `__config__` knob.
- Better long-term fix inside the kernel: move the primitive loop out of the
  block-serial path — either (a) give each `nsq_per_block` slot a different
  primitive quartet so primitives run in parallel across threads and reduce at
  the end, or (b) pre-contract the Rys-root-weighted primitive pairs
  (`Kab·ci·cj` tables per shell pair) so the inner loop is over pair-primitives
  with no barrier. (b) is the standard "contracted pair" trick and also
  shrinks `env` traffic.
- Files: `scf/jk.py` (`_VHFOpt.build`, `get_k`, `get_jk`), `gto/mole.py`
  (`_optimize_contraction`), later `lib/gvhf-rys/rys_contract_k.cu`.
- Validate: K matrix vs shipped path ≤1e-10 on the 9 profiled systems; full
  ladder re-run; organic set (`benchmarks/organic_benchmark`) must not regress
  — that is the case the 0.3 cutoff was built for.
- Expected: ladder B3LYP medians move from 0.15/0.23/0.67× to roughly
  0.5/1.0/1.5× by this change alone (AgCl/TZ 12.7→1.75 s, IrCl6/QZ 76→46 s).

### P2. Cache and slim the one-electron and screening setup  *(days; −0.3 to −0.6 s per SCF, matters most ≤150 AO)*
- Cache `Int3c2eOpt` for the nuclear attraction on the SCF object (or compute
  `int1e_nuc` with a dedicated 1e kernel; the 3c2e machinery is overkill for
  natm point charges). Cache `h1e` on `mf` keyed by `mol` identity, invalidated
  by `reset()`.
- `get_ecp`: cache `(sorted_mol, coeff, ecpbas, tasks_all)` on the `Mole`
  (as `_VHFOpt` does), so the gradient/Hessian callers also benefit; merge the
  k-loop into one launch per (li,lj) with the ECP group as a block index.
- Replace `_group_by_split_points` with a single `cp.argsort` on
  `bin_indices` (one kernel instead of 8 per group pair); build
  `shl_pair_offsets` vectorised.
- Files: `scf/hf.py:get_hcore`, `df/int3c2e_bdiv.py`, `gto/ecp.py`,
  `scf/jk.py:_cache_q_cond_and_non0pairs`, `pbc/scf/rsjk.py`.
- Validate: bitwise-equal `h1e`, `q_cond`; existing `gto/tests/test_ecp*.py`.

### P3. Single fused J+K call for hybrid RKS/UKS  *(days; −20–30 % per hybrid cycle after P1)*
- In `dft/rks.py:get_veff` (and `uks.py`), when `is_hybrid_xc` and no
  range separation, call `ks.get_jk(mol, dm, hermi)` once instead of
  `get_j` + `get_k`; keep the J-engine path for pure functionals and for the
  `omega != 0` K. Benchmark whether MD-J + Rys-K or fused Rys-JK wins at each
  size; pick by `nao`/`n_pairs` heuristic.
- Hoist `pool`, `vj`/`vk` buffers, `dm_cond` scratch and
  `extract_pgto_params` into `_VHFOpt` (allocated once, reused per cycle);
  reconcile `QUEUE_DEPTH`; drop per-launch `cudaGetDeviceProperties`; keep the
  cart↔xyz transforms on device.
- Files: `dft/rks.py`, `dft/uks.py`, `scf/jk.py`, `scf/j_engine.py`,
  `lib/gvhf-rys/rys_contract_k.cu`, `rys_contract_jk.cu`, `vhf.cuh`.

### P4. Launch fusion / SM filling for small systems  *(1–2 weeks; targets the remaining <1× region at DZ)*
- After P1, a 2–7 atom molecule still issues 65–140 launches each covering
  tens to a few thousand quartets on 108 SMs. Options, in order of effort:
  (i) group l-quartets with identical kernel template/scheme into one launch
  with a task-table (the pool mechanism already supports a `head` cursor);
  (ii) run independent l-quartet launches on 2–4 CUDA streams and reduce;
  (iii) CUDA Graph capture of the per-cycle launch sequence (the sequence is
  identical every cycle once `_VHFOpt` is built — only `dm` changes).
- Files: `scf/jk.py` task loop, `rys_jk_driver.cu`.
- Validate: identical J/K; measure SM occupancy with `nsys` on AgCl/TZ.

### P5. Unrolled / tiled kernels for f and g quartets  *(2–4 weeks; QZ and ECP-metal bases)*
- Extend the generator behind `unrolled_rys_k.cu` to emit `(f f | x x)` and
  `(g x | x x)` for x ≤ p, and change `threads_scheme_for_k` so
  `nsq_per_block ≥ 4` for n_tiles > 256 by splitting `gout` across registers
  + shared memory rather than across all 256 threads. Remove the 2–3 sequential
  launches for `n_tiles > 512`.
- This is also the prerequisite for `LMAX=5/6` (h, i functions) in both the
  JK and ECP kernels; today `h_shls` falls back to CPU `direct_mapdm`.
- Validate: `(gggg)` block vs CPU `int2e` to 1e-12; QZ rung timings.

### P6. ECP kernel efficiency  *(1–2 weeks; ECP-specific deliverable)*
- Add templated `type2_cart<LI,LJ,LC>` instantiations for `LI,LJ ≤ 4`
  (currently only up to (1,2,0)/(0,3,0)), so g-function metals never hit the
  dynamic-smem general kernel; batch all ECP groups of one (li,lj) into a
  single launch; precompute the radial Gauss–Chebyshev `rad_all` per
  (ECP atom, shell pair exponent) once and reuse across l.
- Raise `AO_LMAX`/`ECP_LMAX` to 6 with a shared-memory budget check, enabling
  the 5Z rung that `systems_tm.py` documents as blocked.
- Validate against `gto/tests/test_ecp_sweep.py` (1526 subtests) and add the
  h/i cases.

### P7. Benchmark & regression harness updates  *(alongside P1–P3)*
- Add `benchmarks/inorganic_benchmark/profiling/prof_components.py` output
  (per-component ms) to the report so regressions in setup cost are visible,
  not just total wall time.
- Add a "phase timing" field to `run_bench.py` records: `init`, mean
  `cycle`, `n_launches` (from `DEBUG1`), so the ladder plots can separate
  fixed from per-cycle cost.
- Store v2 baselines in `gpu4pyscf/tests/benchmark_results/` and re-run the
  organic set to guard against regressions from P1.

## 5. What this does not fix

- Below ~50 AO the CPU will stay ahead: a single `eigh`, grid build and
  Python overhead are ~50–100 ms per cycle regardless. The goal for v2 is
  crossover at ~80–100 AO for hybrids and ≥3× at 250 AO, not parity at YH3.
- Pure functionals at TZ/QZ are already J-engine bound and reasonably
  efficient (MoF6/QZ PBE: 2.6 s/cycle, 414 AO); P3/P4 help modestly, the
  main lever there is the MD J-engine's own f/g schemes (P5-style work).
- Nothing here touches the coupled-cluster roadmap in `STRATEGY.md`; the CC
  work depends on the DF 3c2e pipeline, not on the direct-SCF JK engine.

## 6. Suggested order

1. P1 (adaptive decontraction, Python-side) + P2 (caches, argsort) — one
   week, re-run the TM ladder, publish before/after heat maps.
2. P3 (fused JK, buffer hoisting) — second week.
3. P4 (launch fusion / graphs) and P6 (ECP templates) in parallel — weeks 3–5.
4. P5 (f/g unrolled kernels, LMAX>4) — weeks 5–9; unblocks the 5Z rung.
5. P1 kernel-side contracted-pair rewrite last, once P5 has settled the
   kernel structure.
