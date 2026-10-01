#!/usr/bin/env python
# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-point DFT benchmark over every element with a def2-ECP, GPU vs CPU.

For each (molecule, functional, device) this records the total energy, wall
time, SCF cycle count and grid size to a JSONL file. `make_report.py` turns
that into tables and figures.

    python run_bench.py                      # full sweep, both devices
    python run_bench.py --only W Mo Pt        # subset, by element or formula
    python run_bench.py --devices gpu         # GPU only
    python run_bench.py --resume              # skip records already present

FOUR MOLECULE SETS -- `--systems`.

`--systems tm` (`systems_tm.py`) is the 19 closed-shell transition-metal
complexes of `systems.py` (Y-Cd, Hf-Hg) run up a BASIS LADDER: def2-SVP,
def2-TZVP and def2-QZVP (`--bases`, labelled DZ/TZ/QZ in the report) with a
trimmed functional list (PBE, B3LYP, PBE0), so the speedup can be read as a
function of AO count from ~50 to ~500 AO. 5Z is not offered: gpu4pyscf's ECP
kernels stop at g functions (see `systems_tm.py`). All RKS, direct SCF.

    python run_bench.py --systems tm                       # full ladder
    python run_bench.py --systems tm --bases def2-tzvp     # one rung
    python run_bench.py --systems tm --only W Pt --devices gpu

`--systems singlet` (default, `systems.py`) keeps every compound closed shell by
using anions where the neutral species is not a singlet, so the whole sweep runs
restricted RKS.

`--systems neutral` (`systems_neutral.py`) is the same 36 compounds at the same
coordinates with those six anions replaced by their OPEN-SHELL neutral forms
(TcO4, ReO4, RhCl6, IrCl6, PdCl4, PtCl4). `--method auto` then runs RKS for the
30 singlets and UKS for the six open-shell cases; `--method uks` forces the
unrestricted path for all 36, which is the way to check that UKS reproduces RKS
at spin 0 on both devices.

    python run_bench.py --systems neutral                  # RKS + UKS as needed
    python run_bench.py --systems neutral --method uks     # unrestricted throughout

`--systems lanl` (`systems_lanl.py`, read from `lanl_benchmark.json`) is a
different kind of set: 39 REAL lanthanide/actinide coordination complexes at
experimental geometries, 16-211 atoms and 570-3747 AO, 33 of them open shell.
It differs from the other two in four ways, all forced by the chemistry:

  * The f-block metal takes the Stuttgart-Koeln RSC ECP and valence basis --
    PySCF's def2 sets stop before the f block. Every other element stays on
    `--basis` (def2-TZVP), so this is the ONLY basis difference from the rest
    of the directory.
  * Density fitting is on by default. Direct SCF at 3747 AO is not tractable
    on the CPU side, and the def2 aux sets also lack the actinides, so the aux
    basis is auto-generated per molecule.
  * The CPU leg is skipped above `--max-nao-cpu` (1800 AO by default), which
    keeps 21 of the 39 systems paired while the GPU still runs all of them.
  * Rows the source spreadsheet flags as hard to converge get a level shift,
    damping and a raised cycle cap -- applied identically on both devices.

    python run_bench.py --systems lanl                 # the f-element set
    python run_bench.py --systems lanl --only Am       # all six americium rows
    python run_bench.py --systems lanl --no-density-fit

Each set writes to its own results file by default (`results.jsonl` /
`results_neutral.jsonl` / `results_lanl.jsonl`).

OPEN-SHELL CAVEAT. An unrestricted SCF can settle on different symmetry-broken
solutions on the two devices from the same initial guess. When that happens the
energy difference is not a numerical-accuracy failure -- the runs converged to
different states. <S^2> and the realized multiplicity are recorded for every
unrestricted run so the report can tell the two cases apart.

This bites hardest on `--systems lanl`. The two devices build their own initial
guesses, and gpu4pyscf's `minao` is not bit-identical to PySCF's (measured:
max|dm0_gpu - dm0_cpu| = 57.8 on PuN3O9Cl3, and 1.4 even on the def2 WF6 of the
other sets). On an open-shell f element that is easily enough to land the two
runs in different SCF solutions. A large dE on that set is therefore usually a
DIFFERENT SOLUTION, not a broken GPU kernel -- read `converged`, `cycles` and
<S^2> before concluding anything from it. That is deliberate: independent
devices is what this benchmark measures.

FAIRNESS NOTE -- read before changing the CPU path.

The CPU reference is built FRESH from the Mole via `pyscf.dft.RKS`. Do not
"optimize" this by using `gpu_mf.to_cpu()`: that copies mo_coeff from the
converged GPU calculation, so the CPU SCF restarts from an already-converged
density and finishes in a couple of cycles. Measured on WF6/def2-TZVP/B3LYP at
16 threads, that shortcut reports 3.05 s instead of the honest 19.59 s -- a 6.4x
bias that inverts the conclusion of the whole benchmark.

Both devices get identical geometry, basis, ECP, grid spec, convergence
threshold and initial guess. Their default grid pruning also agrees
(nwchem_prune / treutler_ahlrichs / original_becke), but the realized grid
point count differs by ~0.1% because of implementation-level screening, so
`ngrids` is recorded per run and surfaced in the report.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DEFAULT_FUNCTIONALS = ['svwn', 'pbe', 'tpss', 'b3lyp', 'pbe0', 'm06-2x']
# wb97x-d is deliberately absent: gpu4pyscf raises
# NotImplementedError('wb97x-d is not supported yet').

# The f-element set is far more expensive per system, so it runs a narrower
# ladder: one GGA and one global hybrid. That pair is also what the source
# spreadsheet's notes are about -- it claims GGA lands on the wrong metal
# oxidation state for the Eu rows where hybrids get it right.
LANL_FUNCTIONALS = ['pbe', 'b3lyp']

# The TM ladder multiplies every system by three bases, so it runs a narrower
# ladder too: one GGA and the two most-used global hybrids.
TM_FUNCTIONALS = ['pbe', 'b3lyp', 'pbe0']

# SCF cycle caps. The LANL systems are large, unrestricted and f-open-shell;
# 100 cycles is not enough headroom even on the rows the spreadsheet does not
# flag. Applied identically on both devices.
DEFAULT_MAX_CYCLE = 100
LANL_MAX_CYCLE = 200

DEFAULT_OUT = {'singlet': 'results.jsonl', 'neutral': 'results_neutral.jsonl',
               'lanl': 'results_lanl.jsonl', 'tm': 'results_tm.jsonl'}

# Above this many AO the CPU leg of the LANL set is skipped rather than run;
# see --max-nao-cpu. 1800 keeps 21 of the 39 systems paired.
LANL_MAX_NAO_CPU = 1800


def load_systems(which):
    """The molecule-table module for `--systems`. All three expose build_mol
    and iter_systems with the same signature. `systems_neutral` entries carry
    one extra trailing field, the spin; `systems_lanl` entries are dicts rather
    than tuples (see the entry_* accessors)."""
    if which == 'neutral':
        import systems_neutral as mod
    elif which == 'lanl':
        import systems_lanl as mod
    elif which == 'tm':
        import systems_tm as mod
    else:
        import systems as mod
    return mod


# --------------------------------------------------------------------------- #
# entry accessors
#
# `systems.py` / `systems_neutral.py` rows are tuples; `systems_lanl.py` rows
# are dicts carrying far more metadata than the tuple format holds. Everything
# downstream goes through these four so the driver never branches on the set.

def entry_spin(entry):
    """2S for a table row. `systems.py` rows are 8-tuples with no spin field."""
    if isinstance(entry, dict):
        return entry['spin']
    return entry[8] if len(entry) > 8 else 0


def entry_key(entry):
    """The unique id recorded as `formula` and used for --resume. For the LANL
    set this is the row-indexed uid, because Formula alone collides there."""
    return entry['uid'] if isinstance(entry, dict) else entry[1]


def entry_element(entry):
    """The label the report groups by: the metal centre."""
    return entry['metal'] if isinstance(entry, dict) else entry[0]


def entry_extra(entry):
    """Set-specific metadata to fold into the JSONL record. Recorded only --
    none of it changes the calculation."""
    if not isinstance(entry, dict):
        return {'geometry': entry[5]}
    return {
        'geometry': 'experimental',
        'compound': entry['formula'],
        'row': entry['row'],
        'metal': entry['metal'],
        'oxidation_state': entry['oxidation_state'],
        # `multiplicity` is claimed by run_one() for the REALIZED multiplicity
        # of the converged determinant. This is the target from the table, so
        # it needs its own name -- the whole point is to compare the two.
        'multiplicity_target': entry['multiplicity'],
        'ligand_type': entry['ligand_type'],
        'ligand_class': entry['ligand_class'],
        'calc_type': entry['calc_type'],
        'notes': entry['notes'],
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--systems', default='singlet',
                   choices=['singlet', 'neutral', 'lanl', 'tm'],
                   help='molecule table: "singlet" = systems.py (anions where '
                        'the neutral is open shell, all RKS); "neutral" = '
                        'systems_neutral.py (same 36 compounds and coordinates, '
                        'no anions, six of them open shell); "lanl" = '
                        'systems_lanl.py (39 f-element complexes at '
                        'experimental geometries, Stuttgart RSC on the metal, '
                        'density fitted); "tm" = systems_tm.py (the 19 '
                        'transition metals of systems.py up the def2 '
                        'DZ/TZ/QZ ladder, see --bases)')
    p.add_argument('--method', default='auto', choices=['auto', 'uks'],
                   help='"auto" = RKS at spin 0, UKS otherwise; "uks" = '
                        'unrestricted for everything, including the singlets')
    p.add_argument('--basis', default='def2-tzvp',
                   help='orbital basis AND ECP name (def2 sets carry both). '
                        'On --systems lanl this sets everything EXCEPT the '
                        'f-block metal, which always takes stuttgart-rsc '
                        'because def2 does not reach the f block')
    p.add_argument('--bases', nargs='+', default=None,
                   help='run every system at each of these bases in turn; '
                        'default is the def2 DZ/TZ/QZ ladder on --systems tm '
                        'and [--basis] otherwise. Records carry `basis` and '
                        '`zeta`, and --resume keys on the basis')
    p.add_argument('--functionals', nargs='+', default=None,
                   help=f'default {" ".join(DEFAULT_FUNCTIONALS)}; on '
                        f'--systems lanl {" ".join(LANL_FUNCTIONALS)}; on '
                        f'--systems tm {" ".join(TM_FUNCTIONALS)}')
    p.add_argument('--devices', nargs='+', default=['gpu', 'cpu'],
                   choices=['gpu', 'cpu'])
    p.add_argument('--only', nargs='*', default=None,
                   help='restrict to these elements or formulas')
    p.add_argument('--cpu-threads', type=int, default=16)
    p.add_argument('--grid', nargs=2, type=int, default=[75, 302],
                   metavar=('NRAD', 'NANG'))
    p.add_argument('--conv-tol', type=float, default=1e-9)
    p.add_argument('--max-cycle', type=int, default=None,
                   help=f'SCF cycle cap; default {LANL_MAX_CYCLE} for '
                        f'--systems lanl (open-shell f elements at 3747 AO '
                        f'need the headroom), {DEFAULT_MAX_CYCLE} otherwise')
    p.add_argument('--density-fit', dest='density_fit', action='store_true',
                   default=None,
                   help='density fit the J/K build on BOTH devices. Default '
                        'on for --systems lanl (direct SCF at 3747 AO is not '
                        'tractable on the CPU), off for the other two sets')
    p.add_argument('--no-density-fit', dest='density_fit', action='store_false')
    p.add_argument('--auxbasis', default='auto',
                   help='DF auxiliary basis; "auto" builds one per molecule '
                        'with pyscf.df.addons.make_auxbasis, which is required '
                        'for the actinides (the def2 aux sets omit them)')
    p.add_argument('--max-nao-cpu', type=int, default=None,
                   help='skip the CPU leg above this many AO, recording a '
                        'skip rather than an error. Default '
                        f'{LANL_MAX_NAO_CPU} for --systems lanl, unlimited '
                        'otherwise')
    p.add_argument('--no-conv-aids', action='store_true',
                   help='ignore the per-row convergence settings (level '
                        'shift / damping / raised cycle cap) that the LANL '
                        'table derives from its Notes column')
    p.add_argument('--repeat', type=int, default=1,
                   help='timed repeats per run; the minimum wall time is kept')
    p.add_argument('--out', default=None,
                   help='JSONL output; defaults to results.jsonl for --systems '
                        'singlet and results_neutral.jsonl for neutral, so the '
                        'two sets never land in the same file')
    p.add_argument('--resume', action='store_true',
                   help='skip (system, functional, device) already recorded')
    p.add_argument('--no-warmup', action='store_true',
                   help='skip the per-functional GPU warm-up (inflates the '
                        'first timing of each functional by the JIT cost)')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)

    # Set-dependent defaults, applied only where the user was silent. --basis
    # is deliberately NOT one of these: the LANL set runs the same def2-TZVP
    # as the rest of the directory, and substitutes the metal internally.
    lanl = args.systems == 'lanl'
    tm = args.systems == 'tm'
    if args.functionals is None:
        if lanl:
            args.functionals = LANL_FUNCTIONALS
        elif tm:
            args.functionals = TM_FUNCTIONALS
        else:
            args.functionals = DEFAULT_FUNCTIONALS
    if args.bases is None:
        if tm:
            import systems_tm
            args.bases = list(systems_tm.LADDER.values())
        else:
            args.bases = [args.basis]
    args.bases = [b.lower() for b in args.bases]
    if args.density_fit is None:
        args.density_fit = lanl
    if args.max_cycle is None:
        args.max_cycle = LANL_MAX_CYCLE if lanl else DEFAULT_MAX_CYCLE
    if args.max_nao_cpu is None and lanl:
        args.max_nao_cpu = LANL_MAX_NAO_CPU

    if args.out is None:
        args.out = os.path.join(HERE, DEFAULT_OUT[args.systems])
    return args


# --------------------------------------------------------------------------- #
# bookkeeping

def load_done(path):
    """(formula, xc, basis, device, method) tuples already recorded without error.

    `method` is part of the key so an RKS and a UKS run of the same singlet are
    distinct work items. Records written before `--method` existed were all
    restricted, hence the 'rks' default.

    A record carrying `skipped` does NOT count as done: it means the CPU leg
    was over the AO cap, and a resubmission with a higher --max-nao-cpu should
    pick it up rather than skip it forever."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue                      # tolerate a torn final line
            if (rec.get('kind') == 'run' and not rec.get('error')
                    and not rec.get('skipped')):
                done.add((rec['formula'], rec['xc'], rec['basis'],
                          rec['device'], rec.get('method', 'rks')))
    return done


def append(path, rec):
    """Append one record and flush to disk, so a job killed at the wall clock
    keeps everything finished up to that point."""
    with open(path, 'a') as fh:
        fh.write(json.dumps(rec) + '\n')
        fh.flush()
        os.fsync(fh.fileno())


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=HERE,
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def metadata(args):
    import numpy
    import pyscf
    meta = {
        'kind': 'meta',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'hostname': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'git_commit': git_commit(),
        'systems': args.systems,
        'method': args.method,
        'basis': args.basis,
        'bases': args.bases,
        'functionals': args.functionals,
        'devices': args.devices,
        'grid': list(args.grid),
        'conv_tol': args.conv_tol,
        'max_cycle': args.max_cycle,
        'density_fit': args.density_fit,
        'auxbasis': args.auxbasis if args.density_fit else None,
        'max_nao_cpu': args.max_nao_cpu,
        'conv_aids': not args.no_conv_aids,
        'repeat': args.repeat,
        'cpu_threads': args.cpu_threads,
        'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
        'python': sys.version.split()[0],
        'numpy': numpy.__version__,
        'pyscf': pyscf.__version__,
    }
    try:
        import gpu4pyscf
        meta['gpu4pyscf'] = gpu4pyscf.__version__
        meta['gpu4pyscf_path'] = gpu4pyscf.__file__
    except Exception as exc:
        meta['gpu4pyscf'] = f'unavailable: {exc}'
    try:
        import cupy
        meta['cupy'] = cupy.__version__
        props = cupy.cuda.runtime.getDeviceProperties(0)
        meta['gpu'] = props['name'].decode()
        meta['cuda_runtime'] = cupy.cuda.runtime.runtimeGetVersion()
        # gpu4pyscf falls back to cupy.einsum when the cuTENSOR binding is
        # missing (the cupy-cuda13x wheels ship without it); record which.
        from gpu4pyscf.lib import cutensor
        meta['contract_engine'] = 'cupy' if cutensor.cutensor is None else 'cutensor'
    except Exception as exc:
        meta['cupy'] = f'unavailable: {exc}'
    return meta


# --------------------------------------------------------------------------- #
# the runs

def is_unrestricted(mol, args):
    """UKS or RKS for this Mole. Must depend only on (mol, args) so both devices
    always agree -- comparing an RKS energy against a UKS one is meaningless."""
    return args.method == 'uks' or mol.spin != 0


def make_auxbasis(mol, args):
    """The DF auxiliary basis for one Mole. Built per molecule and per device;
    nothing is shared between the two runs.

    'auto' goes through pyscf.df.addons.make_auxbasis, which generates a fit
    set on the fly for elements that have no tabulated one. That is not a
    nicety for this benchmark -- the def2 aux sets stop before the actinides,
    exactly as the orbital sets do."""
    if args.auxbasis == 'auto':
        from pyscf import df
        return df.addons.make_auxbasis(mol, mp2fit=False)
    return args.auxbasis


def build_mf(device, mol, xc, args, entry=None):
    """The SCF object for one (device, molecule, functional).

    Everything set here is a function of (mol, xc, args, entry) only, so the
    two devices are configured identically and neither can inherit state from
    the other."""
    unrestricted = is_unrestricted(mol, args)
    if device == 'gpu':
        from gpu4pyscf.dft import rks, uks
        mf = (uks.UKS if unrestricted else rks.RKS)(mol, xc=xc)
    else:
        # FRESH -- never to_cpu(); see module docstring
        from pyscf import dft
        mf = (dft.UKS if unrestricted else dft.RKS)(mol, xc=xc)

    if args.density_fit:
        mf = mf.density_fit(auxbasis=make_auxbasis(mol, args))

    mf.grids.atom_grid = tuple(args.grid)
    mf.conv_tol = args.conv_tol
    mf.max_cycle = args.max_cycle

    # Per-row convergence aids, where the table supplies them. Applied after
    # the shared settings so max_cycle can be raised, and applied to both
    # devices alike -- they change how hard the SCF is driven, never what is
    # being compared.
    for key, value in scf_aids(entry, args).items():
        setattr(mf, key, value)

    mf.verbose = 0
    return mf


def cpu_skipped(device, nao, args):
    """Whether the CPU leg of this system is over the --max-nao-cpu cap.

    A cap hit is not a failure: the GPU still runs the system, and the record
    says why the CPU side is missing so the report can tell "too big to pair"
    apart from "the CPU run broke"."""
    return (device == 'cpu' and args.max_nao_cpu is not None
            and nao > args.max_nao_cpu)


def scf_aids(entry, args):
    """Convergence settings for one entry, or {} if the table has none."""
    if entry is None or args.no_conv_aids:
        return {}
    mod = load_systems(args.systems)
    if not hasattr(mod, 'scf_aids'):
        return {}
    return mod.scf_aids(entry)


def free_gpu():
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def sync_gpu():
    import cupy
    cupy.cuda.Stream.null.synchronize()


def run_one(device, entry, xc, args, basis):
    """Time one SCF at `basis`. Returns a dict of results; raises on failure.

    The Mole is rebuilt here, per device and per repeat, rather than shared:
    a Mole carries no SCF state, but the integral layers on both sides cache
    derived data on it (sorted/grouped basis, ao_loc, opt handles). Sharing one
    object would let whichever device runs second inherit the first one's warm
    caches. Building a Mole costs milliseconds against multi-second SCFs.
    """
    build_mol = load_systems(args.systems).build_mol
    best = None
    for _ in range(max(1, args.repeat)):
        mol = build_mol(entry, basis)
        mf = build_mf(device, mol, xc, args, entry)
        if device == 'gpu':
            sync_gpu()
        t0 = time.perf_counter()
        e_tot = mf.kernel()
        if device == 'gpu':
            sync_gpu()          # kernel() is synchronous, but be explicit
        wall = time.perf_counter() - t0
        if best is None or wall < best['wall']:
            best = {
                'e_tot': float(e_tot),
                'wall': wall,
                'converged': bool(mf.converged),
                'cycles': int(getattr(mf, 'cycles', -1)),
                'ngrids': int(mf.grids.coords.shape[0]),
            }
            if is_unrestricted(mol, args):
                # <S^2> and the realized multiplicity say whether the two
                # devices found the SAME unrestricted solution. Without them a
                # symmetry-broken GPU/CPU mismatch is indistinguishable from a
                # numerical-accuracy failure.
                ss, mult = mf.spin_square()
                best['s_squared'] = float(ss)
                best['multiplicity'] = float(mult)
        del mf, mol
        if device == 'gpu':
            free_gpu()
    return best


def warmup(functionals, args, need_uks=False):
    """Force the one-time NVRTC / libxc compilation for each functional on a
    tiny ECP molecule, so it is not charged to the first real timing.

    RKS and UKS drive different numint kernels, so an open-shell sweep has to
    warm both -- otherwise the first UKS system absorbs the compile cost. The
    LANL set additionally warms a Stuttgart-RSC f-element and, when DF is on,
    the density-fitted path, for the same reason.

    The RKS pass runs once per basis in the ladder: a higher rung brings in
    higher angular momentum (def2-QZVP puts g functions on the metal), and the
    integral and numint kernels for a new l are compiled on first use."""
    from pyscf import gto
    from gpu4pyscf.dft import rks, uks

    passes = [(f'RKS/{basis}', rks.RKS,
               gto.M(atom='Ag 0 0 0; Cl 0 0 2.28', basis=basis, ecp=basis,
                     verbose=0))
              for basis in args.bases]
    if need_uks:
        passes.append(('UKS', uks.UKS, gto.M(atom='I 0 0 0', basis='def2-svp',
                                             ecp='def2-svp', spin=1, verbose=0)))
    if args.systems == 'lanl':
        # CeO: an open-shell f element on the same ECP family the real runs
        # use, small enough to be free.
        passes.append(('UKS/f-block', uks.UKS,
                       gto.M(atom='Ce 0 0 0; O 0 0 1.82',
                             basis={'Ce': 'stuttgart-rsc', 'O': 'def2-svp'},
                             ecp={'Ce': 'stuttgart-rsc'}, spin=2, verbose=0)))

    for label, cls, mol in passes:
        print(f'warm-up {label} (tiny, results discarded):', end=' ', flush=True)
        for xc in functionals:
            try:
                mf = cls(mol, xc=xc)
                if args.density_fit:
                    mf = mf.density_fit(auxbasis=make_auxbasis(mol, args))
                mf.grids.atom_grid = (29, 50)
                mf.conv_tol = 1e-6
                mf.verbose = 0
                mf.kernel()
                print(xc, end=' ', flush=True)
            except Exception as exc:
                print(f'{xc}(FAILED: {type(exc).__name__})', end=' ', flush=True)
            finally:
                free_gpu()
        print()


def main(argv=None):
    args = parse_args(argv)

    from pyscf import lib
    lib.num_threads(args.cpu_threads)

    sysmod = load_systems(args.systems)
    build_mol = sysmod.build_mol
    # `zeta_label` exists only on the ladder set; elsewhere the label is the
    # basis name itself.
    zeta_label = getattr(sysmod, 'zeta_label', lambda basis: basis)

    entries = list(sysmod.iter_systems(args.only))
    if not entries:
        sys.exit(f'no systems matched --only {args.only}')

    # UKS for anything with spin != 0, or for everything under --method uks.
    unrestricted = {entry_key(e): (args.method == 'uks' or entry_spin(e) != 0)
                    for e in entries}
    n_open = sum(1 for e in entries if entry_spin(e) != 0)

    # Basis is the OUTER loop: the whole set finishes at one rung before the
    # next starts, so a job killed at the wall clock leaves complete rungs
    # rather than a ragged ladder.
    todo = [(basis, e, xc, dev) for basis in args.bases for e in entries
            for xc in args.functionals for dev in args.devices]
    done = load_done(args.out) if args.resume else set()

    print(f'systems={len(entries)} ({n_open} open shell) '
          f'bases={args.bases} functionals={len(args.functionals)} '
          f'devices={args.devices} -> {len(todo)} runs, '
          f'{len(done)} already recorded')
    if args.density_fit:
        print(f'density fitting ON, auxbasis={args.auxbasis}')
    if args.max_nao_cpu:
        print(f'CPU leg skipped above {args.max_nao_cpu} AO')

    if args.dry_run:
        # nao needs a Mole, which is cheap (milliseconds) next to an SCF.
        for basis, entry, xc, dev in todo:
            key = entry_key(entry)
            meth = 'uks' if unrestricted[key] else 'rks'
            nao = build_mol(entry, basis).nao
            if cpu_skipped(dev, nao, args):
                flag = 'cap '
            elif (key, xc, basis, dev, meth) in done:
                flag = 'skip'
            else:
                flag = 'run '
            aids = 'aids' if scf_aids(entry, args) else ''
            print(f'  {flag} {zeta_label(basis):9s} {entry_element(entry):3s} '
                  f'{key:26s} 2S={entry_spin(entry)} nao={nao:5d} '
                  f'{xc:8s} {meth} {dev:3s} {aids}')
        return 0

    append(args.out, metadata(args))

    if 'gpu' in args.devices and not args.no_warmup:
        warmup(args.functionals, args, need_uks=any(unrestricted.values()))

    t_start = time.perf_counter()
    n_ok = n_skip = n_fail = n_cap = 0

    for basis in args.bases:
      zeta = zeta_label(basis)
      for entry in entries:
        element, formula = entry_element(entry), entry_key(entry)
        # Metadata only -- never used for an SCF; run_one() builds its own.
        mol_meta = build_mol(entry, basis)
        method = 'uks' if unrestricted[formula] else 'rks'
        aids = scf_aids(entry, args)
        base = {
            'kind': 'run',
            'element': element,
            'formula': formula,
            'charge': mol_meta.charge,
            'spin': entry_spin(entry),
            'method': method,
            'basis': basis,
            'zeta': zeta,
            'natm': int(mol_meta.natm),
            'nao': int(mol_meta.nao),
            'nelectron': int(mol_meta.nelectron),
            'nelec_core': int(mol_meta.atom_nelec_core(0)),
            'cpu_threads': args.cpu_threads,
            'density_fit': args.density_fit,
            'auxbasis': args.auxbasis if args.density_fit else None,
            **aids,
            **entry_extra(entry),
        }
        tag = f'{zeta:4s} {element:3s} {formula:12s}'
        for xc in args.functionals:
            for device in args.devices:
                if (formula, xc, basis, device, method) in done:
                    n_skip += 1
                    continue
                if cpu_skipped(device, mol_meta.nao, args):
                    n_cap += 1
                    reason = (f'nao {mol_meta.nao} > max-nao-cpu '
                              f'{args.max_nao_cpu}')
                    append(args.out, dict(base, xc=xc, device=device,
                                          skipped=reason))
                    print(f'{tag} nao={mol_meta.nao:4d} '
                          f'{xc:8s} {device:3s}  SKIPPED {reason}', flush=True)
                    continue
                rec = dict(base, xc=xc, device=device)
                try:
                    rec.update(run_one(device, entry, xc, args, basis))
                    n_ok += 1
                    status = (f"E={rec['e_tot']:.8f} {rec['wall']:7.2f}s "
                              f"cyc={rec['cycles']:>3d}"
                              f"{'' if rec['converged'] else '  NOT CONVERGED'}")
                    if 's_squared' in rec:
                        status += f"  <S^2>={rec['s_squared']:.3f}"
                except Exception as exc:
                    rec['error'] = f'{type(exc).__name__}: {exc}'
                    n_fail += 1
                    status = f'FAILED {rec["error"][:70]}'
                    free_gpu()
                append(args.out, rec)
                print(f'{tag} nao={rec["nao"]:4d} '
                      f'{xc:8s} {device:3s}  {status}', flush=True)

    dt = time.perf_counter() - t_start
    print(f'\ndone: {n_ok} ok, {n_fail} failed, {n_skip} already recorded, '
          f'{n_cap} over the CPU AO cap in {dt/60:.1f} min -> {args.out}')
    print(f'next: python {os.path.join(HERE, "make_report.py")} --results {args.out}')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
