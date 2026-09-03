# ECP task-list screening

`gpu4pyscf.gto.ecp` builds a list of `(shell i, shell j, ECP group k)` triples
and runs one CUDA block per triple. Without screening the list is
`N_bas*(N_bas+1)/2 * N_ecp_groups` (scalar) / `N_bas^2 * N_ecp_groups`
(derivatives) regardless of geometry.

A two-level screen (`_screen_block`) mirrors `check_3c_overlap` in
`pyscf/lib/gto/nr_ecp.c`:

1. prune shells to the neighbourhood of each ECP centre via the best-case bound
   `ai*ak*|C-Ri|^2 < EXPCUTOFF*(ai+ak)`;
2. keep a pair `(i,j)` for centre `k` when
   `(ai*aj|Ri-Rj|^2 + ai*ak|C-Ri|^2 + aj*ak|C-Rj|^2)/(ai+aj+ak) < EXPCUTOFF`
   (`EXPCUTOFF = 39`, `~1e-17`; `ai/aj/ak` = smallest primitive exponents).

Toggle with `gpu4pyscf.gto.ecp.SCREEN_ECP` or `__config__.gto_ecp_screen`.

## Benchmark

`benchmarks/gto/benchmark_ecp_screening.py` — a loosely spaced chain of Cu atoms
(4.0 A spacing, cartesian, `crenbl` basis + ECP), screened vs unscreened.

Hardware: 1x NVIDIA A100 40 GB, NERSC Perlmutter. gpu4pyscf @ `ecp-screening`,
CUDA 12.4, CuPy 13.4.1.

```
python benchmarks/gto/benchmark_ecp_screening.py --n 20 40 80 160 --ip
```

| N (atoms) | nao | tasks (full) | tasks (screened) | kept | `get_ecp` full / s | `get_ecp` screened / s | speed-up | max \|Δ\| |
|----:|----:|------------:|-----------------:|-----:|-------:|-------:|-------:|-------:|
|  20 | 1220 |   4 343 400 |   33 402 | 0.77 % |  0.718 | 0.051 |  **14x** | 2e-15 |
|  40 | 2440 |  34 701 600 |   68 882 | 0.20 % |  5.897 | 0.108 |  **55x** | 2e-15 |
|  80 | 4880 | 277 430 400 |  139 842 | 0.05 % | 46.953 | 0.258 | **182x** | 2e-15 |

`get_ecp_ip` (gradient integrals): 35x at N=20, 55x at N=40. N=80 not measured
here — the dense `[n_ecp_atm, 3, nao, nao]` output buffer needs ~43 GiB
(addressed separately by ECP-atom slicing).

## Reading the numbers

- `tasks (full)` ~ O(N^3), `tasks (screened)` ~ O(N): each centre only interacts
  with a bounded neighbourhood, so the speed-up grows ~O(N^2) with chain length.
- `max |Δ|` is the largest elementwise difference between the screened and
  unscreened `get_ecp` matrices — at machine precision for every size; the
  benchmark asserts `< 1e-9`.
- Small / dense molecules: survivors == everything, so screening is a cheap
  no-op (see the regression suite, unchanged to 1e-10).
