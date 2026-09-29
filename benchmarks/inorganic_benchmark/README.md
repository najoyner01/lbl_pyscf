# Inorganic ECP single-point benchmark across the periodic table

GPU4PySCF vs CPU PySCF for one molecule per element carrying a **def2-ECP**
(36 elements, Rb–Rn including La), each at several popular functionals in a
single basis. Two variants of the set are available — an all-closed-shell one
and an all-neutral one that pulls in six open-shell complexes — plus a third,
much larger set of real f-element coordination complexes (see
[Three molecule sets](#three-molecule-sets)).

Nothing here uses periodic boundary conditions; despite the periodic-*table*
coverage these are molecular AO calculations via `gpu4pyscf.dft.rks` / `uks`.

## Files

| file | role |
|---|---|
| `systems.py` | the 36-element molecule table + VSEPR geometry generator (closed-shell set, six anions). Run it directly for a geometry/size dump (no SCF). |
| `systems_neutral.py` | the all-neutral variant of the same table, derived from `systems.py`. Six species become open-shell. Run it directly for a dump of formulas, spins and electron counts. |
| `systems_lanl.py` | the 39 f-element complexes, read from `lanl_benchmark.json`. Run it directly for a composition/size dump (no SCF); pass a basis name as `argv[1]`. |
| `lanl_benchmark.json` | source data for the above — experimental geometries plus charge, spin, oxidation state, ligand class and per-row convergence notes, exported from `systems_benchmarking.xlsx`. |
| `run_bench.py` | the driver, for any of the three sets. Writes one JSON record per (system, functional, device) to `results.jsonl`, `results_neutral.jsonl` or `results_lanl.jsonl`. |
| `make_report.py` | reads a JSONL, writes `<prefix>.md`, `<prefix>.csv`, `<prefix>.png`/`.pdf`. |
| `run_bench.sbatch` | Perlmutter batch job: full closed-shell sweep, then the report. |
| `run_bench_neutral.sbatch` | same, for the neutral/open-shell set. |
| `run_bench_lanl.sbatch` | same, for the f-element set. |

## Three molecule sets

| `--systems` | what it is | SCF | results file |
|---|---|---|---|
| `singlet` (default) | `systems.py`: 30 neutral closed shells + 6 anions | all RKS, direct | `results.jsonl` |
| `neutral` | `systems_neutral.py`: all 36 neutral, 6 of them open shell | 30 RKS + 6 UKS, direct | `results_neutral.jsonl` |
| `lanl` | `systems_lanl.py`: 39 real f-element complexes | 6 RKS + 33 UKS, density fitted | `results_lanl.jsonl` |

The first two are the same 36 elements at the same geometries and bond lengths,
differing only in charge and therefore in SCF flavour. The neutral set exists to
exercise the **unrestricted** path — UKS Fock builds, UKS + ECP `get_hcore`, and
the spin-density XC kernels — against the CPU reference, which the
all-restricted set never touches.

The `lanl` set is a different kind of test entirely: real coordination
chemistry at experimental geometries, two orders of magnitude larger. See
[The f-element set](#the-f-element-set---systems-lanl).

## Running

```sh
source env.sh                                  # repo root
mkdir -p logs
sbatch benchmarks/inorganic_benchmark/run_bench.sbatch           # closed shell
sbatch benchmarks/inorganic_benchmark/run_bench_neutral.sbatch   # neutral / open shell
sbatch benchmarks/inorganic_benchmark/run_bench_lanl.sbatch      # f-element set
```

Interactively, or for a subset:

```sh
cd benchmarks/inorganic_benchmark
python run_bench.py --only W Mo Pt                 # elements or formulas
python run_bench.py --devices gpu                  # GPU only
python run_bench.py --dry-run                      # list the work
python make_report.py

python run_bench.py --systems neutral --dry-run    # the neutral set
python run_bench.py --systems neutral --only RhCl6 PtCl4
python make_report.py --results results_neutral.jsonl --prefix report_neutral

python run_bench.py --systems lanl --dry-run       # the f-element set
python run_bench.py --systems lanl --only Am       # all six americium rows
python run_bench.py --systems lanl --only r10      # one row, by index
python make_report.py --results results_lanl.jsonl --prefix report_lanl
```

`--only` on the neutral set also accepts the anion formulas (`--only TcO4-`
matches the neutral `TcO4`), so the same selection works against both sets.
`--method uks` forces the unrestricted solver even on the closed shells, which
is the way to check that UKS reproduces RKS on a singlet.

The results file is appended and fsync'd after every SCF, and `--resume` skips
(system, functional, basis, device, method) tuples already recorded — a job
killed at the wall clock loses nothing and can be resubmitted as-is. The two
sets write to different files and never collide.

`make_report.py` needs `matplotlib` for the figures (`pip install matplotlib`);
without it the tables and CSV are still written.

## Settings

Defaults for the two def2 sets. The f-element set overrides several of these —
see [its own table](#settings--what-differs-from-the-other-two-sets).

| setting | value | why |
|---|---|---|
| basis + ECP | `def2-tzvp` | the def2 workhorse; the same name supplies the orbital basis and the def2-ECP |
| functionals | `svwn pbe tpss b3lyp pbe0 m06-2x` | one per rung: LDA, GGA, mGGA, global hybrids, mGGA hybrid |
| grid | `(75, 302)` | identical spec on both devices |
| `conv_tol` | `1e-9` | tighter than the 1e-8 energy-agreement threshold used in the report |
| `max_cycle` | 100 | charged systems need the headroom |
| CPU threads | 16 | via `lib.num_threads(16)` and `OMP_NUM_THREADS=16` |
| SCF | direct (no density fitting) | default on both sides; `--systems lanl` density fits instead |
| SCF flavour | RKS if `mol.spin == 0`, else UKS | chosen from the `Mole` alone, so GPU and CPU always run the same solver |

`wb97x-d` is deliberately excluded: gpu4pyscf raises
`NotImplementedError('wb97x-d is not supported yet')`. `wb97m-v` and `r2scan`
do work if you want to extend the list.

## Methodology notes

**The CPU reference is built fresh from the `Mole`.** `run_bench.py` calls
`pyscf.dft.RKS(mol, xc=...)` (or `UKS`) for the CPU side. It must not use
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
before timing anything; `--no-warmup` disables it. When the run contains
unrestricted systems it warms up twice per functional — RKS on HI and UKS on a
doublet I atom — because the two drive different numint kernels. The f-element
set adds a third pass on a Stuttgart-RSC CeO, density fitted when DF is on, so
neither the f-block ECP kernels nor the DF path is charged to a timed run.

**Geometries.** The two def2 sets use **idealized** VSEPR shapes with tabulated
bond lengths, not optimized structures; the `lanl` set uses **experimental**
coordinates from its source spreadsheet. In both cases the two devices see
identical coordinates, so GPU/CPU comparisons are exact — but the absolute
energies are not thermochemical reference values and should not be cited as
such. Optimizing every system first is a separate, much longer job.

**Charged systems.** In the default (`singlet`) set, six entries are anions,
used precisely where the neutral compound would be open-shell: TcO4⁻, ReO4⁻,
[RhCl6]³⁻, [IrCl6]³⁻, [PdCl4]²⁻, [PtCl4]²⁻. Highly charged anions in vacuum can
converge slowly or not at all; the driver records `converged: false` rather than
hiding it, and the report counts unconverged runs per element. The `neutral` set
drops the charge on all six and takes the open shell instead.

## The f-element set (`--systems lanl`)

39 lanthanide and actinide coordination complexes from `lanl_benchmark.json`,
at **experimental geometries** rather than the VSEPR idealizations of the other
two sets. 16–211 atoms, 570–3747 AO at def2-TZVP (80336 AO over the set), 33 of
the 39 open shell with 2S up to 7, charges −2…+1. Ligand classes span soft
donors (dithiocarbamates, chalcogen ethers), hard donors (nitrates, malonamides,
crown ethers, phenanthrolines) and coordinated radicals (semiquinone, bipyridine,
pyrazine). Several rows are spin ladders over one fixed geometry — four
multiplicities of the same Am semiquinone complex, four of a Eu bipyridine, three
of a U pyrazine dimer — so the set also probes spin-state energetics at constant
nuclear coordinates.

### Settings — what differs from the other two sets

| setting | value | why |
|---|---|---|
| basis + ECP, ligands | `def2-tzvp` | **unchanged** from the rest of the directory |
| basis + ECP, f-block metal | `stuttgart-rsc` | forced: PySCF's def2 sets stop before the f block |
| SCF | **density fitted**, aux basis auto-generated | direct SCF at 3747 AO is not tractable on the CPU |
| functionals | `pbe b3lyp` | one GGA, one global hybrid — the pair the source notes are about |
| `conv_tol` | `1e-8` | still tighter than the report's agreement threshold |
| `max_cycle` | 200 | large open-shell f systems; 31 of 39 rows are also marked hard to converge |
| CPU leg | skipped above 1800 AO | keeps 21 of 39 systems paired inside the wall clock |

**The basis substitution is forced, not a preference.** None of the eight
lanthanides (Ce Pr Nd Sm Eu Tb Ho Er) or six actinides (Th U Np Pu Am Cm) in
this set has a def2 basis *or* a def2 ECP in PySCF. `stuttgart-rsc` covers all
14 — small-core Stuttgart–Köln RSC, 28 core electrons for the Ln and 60 for the
An, with scalar relativity folded into the pseudopotential. Every other element
stays on `--basis`; iodine is the only ligand element carrying an ECP and it
keeps the def2 one. The GPU and CPU ECP integrals for this combination agree to
`3.9e-13`, so the substitution does not itself introduce a device difference.

`aug-cc-pVDZ-PP` and its relatives do not help here. Peterson's `-PP` series was
built against these same Stuttgart pseudopotentials but was never extended to
the f block (PySCF's `aug-cc-pVDZ-PP.dat` holds only Cu Zn Ag Cd Au Hg), it has
no ligand coverage at all, and it targets correlated wavefunction methods rather
than single-point DFT.

**Density fitting** is on by default for this set only, and applied identically
on both devices. The def2 auxiliary sets omit the actinides exactly as the
orbital sets do, so `--auxbasis auto` generates a fit set per molecule via
`pyscf.df.addons.make_auxbasis`. `--no-density-fit` restores direct SCF if you
want to measure the DF error itself on the small systems.

**The CPU AO cap** (`--max-nao-cpu`, 1800 by default) skips the CPU leg of the
18 largest systems and writes a record carrying `skipped` instead of running
them, so the GPU still sweeps all 39 while the paired comparison stays
affordable. A skip is **not** an error and **not** "done": raise the cap and
resubmit with `--resume` and those legs will be picked up rather than stepped
over.

**Convergence aids.** The spreadsheet's `Notes` column flags 31 rows as hard to
converge; `systems_lanl.py:scf_aids` turns that into `level_shift=0.2`,
`damp=0.3`, `max_cycle=200`. These are applied to **both** devices alike — they
change how hard the SCF is driven, never what is being compared — and are
recorded per run. `--no-conv-aids` disables them; the 200-cycle cap is the
set-wide `--max-cycle` default and survives that flag.

**Recorded but inert.** `oxidation_state`, `ligand_type`, `ligand_class`,
`calc_type`, `multiplicity_target` and `notes` are written into every record so
the report can group speedups by ligand class or oxidation state. None of them
touches the calculation. Note that `Calc_Type` says `Optimization` for 34 rows —
this harness runs single points regardless; the field is metadata only.

### Reading the results

> [!WARNING]
> **A large `dE` on this set usually means the two devices converged to
> different SCF solutions, not that a GPU kernel is wrong.** The two devices
> build their own initial guesses, and gpu4pyscf's `minao` is not bit-identical
> to PySCF's — measured `max|dm0_gpu − dm0_cpu| = 57.8` on PuN3O9Cl3, and 1.4
> even on the def2 WF6 of the other sets. On an open-shell f element that is
> easily enough to reach a different determinant. Check `converged`, `cycles`
> and `<S^2>` before reading anything into `dE`. Handing both devices one
> shared `dm0` collapses the same comparison to 2e-8 Ha, which is what the
> difference actually is once the starting points match.

Fully independent devices is the deliberate choice here: each builds its own
`Mole`, its own aux basis, its own guess, and runs its own SCF. Nothing is
copied from one device's result into the other's.

**Where to look when a run fails.** `converged`, `cycles` and `<S^2>` are
recorded per device for every run, and the report puts them in three places so
a failure does not have to be hunted for:

- a **Failures** section, first after Settings, listing every errored or
  unconverged device-run with its cycle count, `<S^2>`, spin contamination and
  what the *other* device did on the same system;
- the console summary, one line per failure with the same numbers, so
  `make_report.py` tells you without opening the markdown;
- the **Full results** table, where `conv`, `cyc` and `<S^2>` are `GPU/CPU`
  columns on every row.

Read them together. `cyc` at the cap with a large `contam` means the SCF drifted
off the target spin state; `cyc` at the cap with `contam` near zero is an
oscillation that more cycles or a stronger level shift may fix; `cyc` well below
the cap on an unconverged run means it stopped early for another reason. A
**NO** in `conv` also means that device's energy is not a converged value, so
any `dE` computed against it is not an accuracy measurement.

Formulas are **not** unique in this file — `AmC84N3H114O9` appears four times as
a spin ladder and `HoC51H54N9O15` twice as two distinct conformers with
identical charge and spin. Every system therefore gets a row-indexed uid,
`r<NN>_<Formula>_s<2S>`, which is what lands in the `formula` field and what
`--only` and `--resume` key on.

## Open shell (`--systems neutral`)

Stripping the charge off those six leaves an odd or high-spin electron count.
The spins assigned in `systems_neutral.py`:

| neutral | 2S | multiplicity | e⁻ (with ECP cores) | rationale |
|---|---|---|---|---|
| TcO4 | 1 | doublet | 47 | Tc(VII) d⁰ + one O-centred hole |
| ReO4 | 1 | doublet | 47 | Re(VII) d⁰ + one O-centred hole |
| RhCl6 | 3 | quartet | 119 | Rh(VI) d³, Oh t2g³ (cf. RhF₆) |
| IrCl6 | 3 | quartet | 119 | Ir(VI) d³, Oh t2g³ (cf. IrF₆) |
| PdCl4 | 2 | triplet | 86 | Pd(IV) d⁶, 4-coordinate |
| PtCl4 | 2 | triplet | 86 | Pt(IV) d⁶, 4-coordinate |

> [!WARNING]
> **These spin states are formal assignments, not verified ground states.** No
> spin-state scan was run. They are chosen to have the right electron-count
> parity and a defensible ligand-field justification, which is all the benchmark
> needs — both devices solve the *same* problem. Do not quote these energies as
> thermochemistry.

The six also keep the **anion's** bond length, so the neutral and closed-shell
benchmarks run at byte-identical coordinates and their timings are directly
comparable. That is deliberate: the neutral species would have shorter bonds.

**What can go wrong, and how the report shows it.** A UKS solution is not
unique — GPU and CPU can converge to different symmetry-broken states from the
same guess, which shows up as a large `dE` that is *not* a GPU integral or XC
error. `run_bench.py` therefore records `<S^2>` and the multiplicity for every
unrestricted run, and `make_report.py` reports `d<S^2>` and the spin
contamination `<S^2> - S(S+1)` per pair in an **Open-shell / spin** section. A
pair with `d<S^2> > 1e-4` is flagged as *a different SCF solution* rather than
an accuracy failure. To compare those systems numerically, restart both devices
from the same converged density, or exclude them from the `dE` statistics.

Closed-shell records written before this change have no `spin`/`method` field;
both tools default them to `0` / `rks`, so old `results.jsonl` files still
report and still `--resume` correctly.

**Speedup** is `t_CPU / t_GPU` on total SCF wall time, one GPU against 16 CPU
threads. Comparing against a different thread count changes the number; the
count is recorded in the metadata block and printed in the report header.
