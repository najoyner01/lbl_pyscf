# ECP single-point benchmark across the periodic table

GPU4PySCF vs CPU PySCF for one closed-shell molecule per element carrying a
**def2-ECP** (36 elements, Rb–Rn including La), each at several popular
functionals in a single basis.

"ptable" = periodic *table*. Nothing here uses periodic boundary conditions;
these are molecular AO calculations via `gpu4pyscf.dft.rks`.

## Files

| file | role |
|---|---|
| `systems.py` | the 36-element molecule table + VSEPR geometry generator. Run it directly for a geometry/size dump (no SCF). |
| `run_bench.py` | the driver. Writes one JSON record per (system, functional, device) to `results.jsonl`. |
| `make_report.py` | reads the JSONL, writes `report.md`, `report.csv`, `report.png`. |
| `run_bench.sbatch` | Perlmutter batch job: full sweep, then the report. |

## Running

```sh
source env.sh                                  # repo root
mkdir -p logs
sbatch benchmarks/ecp_ptable/run_bench.sbatch
```

Interactively, or for a subset:

```sh
python benchmarks/ecp_ptable/run_bench.py --only W Mo Pt      # elements or formulas
python benchmarks/ecp_ptable/run_bench.py --devices gpu       # GPU only
python benchmarks/ecp_ptable/run_bench.py --dry-run           # list the work
python benchmarks/ecp_ptable/make_report.py
```

`results.jsonl` is appended and fsync'd after every SCF, and `--resume` skips
triples already recorded — a job killed at the wall clock loses nothing and can
be resubmitted as-is.

`make_report.py` needs `matplotlib` for the figures (`pip install matplotlib`);
without it the tables and CSV are still written.

## Settings

| setting | value | why |
|---|---|---|
| basis + ECP | `def2-tzvp` | the def2 workhorse; the same name supplies the orbital basis and the def2-ECP |
| functionals | `svwn pbe tpss b3lyp pbe0 m06-2x` | one per rung: LDA, GGA, mGGA, global hybrids, mGGA hybrid |
| grid | `(75, 302)` | identical spec on both devices |
| `conv_tol` | `1e-9` | tighter than the 1e-8 energy-agreement threshold used in the report |
| `max_cycle` | 100 | charged systems need the headroom |
| CPU threads | 16 | via `lib.num_threads(16)` and `OMP_NUM_THREADS=16` |
| SCF | direct (no density fitting) | default on both sides |

`wb97x-d` is deliberately excluded: gpu4pyscf raises
`NotImplementedError('wb97x-d is not supported yet')`. `wb97m-v` and `r2scan`
do work if you want to extend the list.

## Methodology notes

**The CPU reference is built fresh from the `Mole`.** `run_bench.py` calls
`pyscf.dft.RKS(mol, xc=...)` for the CPU side. It must not use
`gpu_mf.to_cpu()`: that copies `mo_coeff` from the converged GPU calculation, so
the CPU SCF restarts from an already-converged density and converges in a couple
of cycles. Measured on WF6/def2-TZVP/B3LYP at 16 threads:

| CPU object | wall time |
|---|---|
| `to_cpu()` of a converged GPU run | 3.05 s |
| fresh `pyscf.dft.RKS(mol)` | 19.59 s |

A 6.4× bias, large enough to invert the benchmark's conclusion. The SCF cycle
counts are recorded for both devices so this class of error stays visible in the
data.

**Grids.** Both paths default to the same pruning and radial scheme
(`nwchem_prune`, `treutler_ahlrichs`, `original_becke`), but realized grid point
counts differ by ~0.1% from implementation-level screening (WF6: 99584 GPU vs
99440 CPU). `ngrids` is recorded per run.

**Warm-up.** The first GPU call for a given functional pays one-time NVRTC and
libxc compilation. The driver burns that on a tiny HI molecule per functional
before timing anything; `--no-warmup` disables it.

**Geometries are idealized**, VSEPR shapes with tabulated bond lengths, not
optimized. Both devices see identical coordinates, so GPU/CPU comparisons are
exact — but the absolute energies are not thermochemical reference values and
should not be cited as such. Optimizing every system first is a separate,
much longer job.

**Charged systems.** Six entries are anions, used where the neutral compound
would be open-shell: TcO4⁻, ReO4⁻, [RhCl6]³⁻, [IrCl6]³⁻, [PdCl4]²⁻, [PtCl4]²⁻.
Highly charged anions in vacuum can converge slowly or not at all; the driver
records `converged: false` rather than hiding it, and the report counts
unconverged runs per element. If the trianions prove unreliable, swap them for
neutral closed-shell alternatives in `systems.py` — the driver needs no change.

**Speedup** is `t_CPU / t_GPU` on total SCF wall time, one GPU against 16 CPU
threads. Comparing against a different thread count changes the number; the
count is recorded in the metadata block and printed in the report header.
