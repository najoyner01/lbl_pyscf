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

"""Single-point RKS benchmark over every element with a def2-ECP, GPU vs CPU.

For each (molecule, functional, device) this records the total energy, wall
time, SCF cycle count and grid size to a JSONL file. `make_report.py` turns
that into tables and figures.

    python run_bench.py                      # full sweep, both devices
    python run_bench.py --only W Mo Pt        # subset, by element or formula
    python run_bench.py --devices gpu         # GPU only
    python run_bench.py --resume              # skip records already present

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

from systems import build_mol, iter_systems  # noqa: E402

DEFAULT_FUNCTIONALS = ['svwn', 'pbe', 'tpss', 'b3lyp', 'pbe0', 'm06-2x']
# wb97x-d is deliberately absent: gpu4pyscf raises
# NotImplementedError('wb97x-d is not supported yet').


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--basis', default='def2-tzvp',
                   help='orbital basis AND ECP name (def2 sets carry both)')
    p.add_argument('--functionals', nargs='+', default=DEFAULT_FUNCTIONALS)
    p.add_argument('--devices', nargs='+', default=['gpu', 'cpu'],
                   choices=['gpu', 'cpu'])
    p.add_argument('--only', nargs='*', default=None,
                   help='restrict to these elements or formulas')
    p.add_argument('--cpu-threads', type=int, default=16)
    p.add_argument('--grid', nargs=2, type=int, default=[75, 302],
                   metavar=('NRAD', 'NANG'))
    p.add_argument('--conv-tol', type=float, default=1e-9)
    p.add_argument('--max-cycle', type=int, default=100)
    p.add_argument('--repeat', type=int, default=1,
                   help='timed repeats per run; the minimum wall time is kept')
    p.add_argument('--out', default=os.path.join(HERE, 'results.jsonl'))
    p.add_argument('--resume', action='store_true',
                   help='skip (system, functional, device) already recorded')
    p.add_argument('--no-warmup', action='store_true',
                   help='skip the per-functional GPU warm-up (inflates the '
                        'first timing of each functional by the JIT cost)')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# bookkeeping

def load_done(path):
    """(formula, xc, basis, device) tuples already recorded without error."""
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
            if rec.get('kind') == 'run' and not rec.get('error'):
                done.add((rec['formula'], rec['xc'], rec['basis'], rec['device']))
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
        'basis': args.basis,
        'functionals': args.functionals,
        'devices': args.devices,
        'grid': list(args.grid),
        'conv_tol': args.conv_tol,
        'max_cycle': args.max_cycle,
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

def build_mf(device, mol, xc, args):
    if device == 'gpu':
        from gpu4pyscf.dft import rks
        mf = rks.RKS(mol, xc=xc)
    else:
        from pyscf import dft
        mf = dft.RKS(mol, xc=xc)      # FRESH -- never to_cpu(); see module docstring
    mf.grids.atom_grid = tuple(args.grid)
    mf.conv_tol = args.conv_tol
    mf.max_cycle = args.max_cycle
    mf.verbose = 0
    return mf


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


def run_one(device, entry, xc, args):
    """Time one SCF. Returns a dict of results; raises on failure.

    The Mole is rebuilt here, per device and per repeat, rather than shared:
    a Mole carries no SCF state, but the integral layers on both sides cache
    derived data on it (sorted/grouped basis, ao_loc, opt handles). Sharing one
    object would let whichever device runs second inherit the first one's warm
    caches. Building a Mole costs milliseconds against multi-second SCFs.
    """
    best = None
    for _ in range(max(1, args.repeat)):
        mol = build_mol(entry, args.basis)
        mf = build_mf(device, mol, xc, args)
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
        del mf, mol
        if device == 'gpu':
            free_gpu()
    return best


def warmup(functionals, args):
    """Force the one-time NVRTC / libxc compilation for each functional on a
    tiny ECP molecule, so it is not charged to the first real timing."""
    from pyscf import gto
    print('warm-up (tiny HI, results discarded):', end=' ', flush=True)
    mol = gto.M(atom='I 0 0 0; H 0 0 1.61', basis='def2-svp', ecp='def2-svp',
                verbose=0)
    for xc in functionals:
        try:
            from gpu4pyscf.dft import rks
            mf = rks.RKS(mol, xc=xc)
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

    entries = list(iter_systems(args.only))
    if not entries:
        sys.exit(f'no systems matched --only {args.only}')

    todo = [(e, xc, dev) for e in entries for xc in args.functionals
            for dev in args.devices]
    done = load_done(args.out) if args.resume else set()

    print(f'systems={len(entries)} functionals={len(args.functionals)} '
          f'devices={args.devices} -> {len(todo)} runs, '
          f'{len(done)} already recorded')

    if args.dry_run:
        for entry, xc, dev in todo:
            key = (entry[1], xc, args.basis, dev)
            flag = 'skip' if key in done else 'run '
            print(f'  {flag} {entry[0]:3s} {entry[1]:10s} {xc:8s} {dev}')
        return 0

    append(args.out, metadata(args))

    if 'gpu' in args.devices and not args.no_warmup:
        warmup(args.functionals, args)

    t_start = time.perf_counter()
    n_ok = n_skip = n_fail = 0

    for entry in entries:
        element, formula = entry[0], entry[1]
        # Metadata only -- never used for an SCF; run_one() builds its own.
        mol_meta = build_mol(entry, args.basis)
        base = {
            'kind': 'run',
            'element': element,
            'formula': formula,
            'geometry': entry[5],
            'charge': entry[7],
            'basis': args.basis,
            'natm': int(mol_meta.natm),
            'nao': int(mol_meta.nao),
            'nelectron': int(mol_meta.nelectron),
            'nelec_core': int(mol_meta.atom_nelec_core(0)),
            'cpu_threads': args.cpu_threads,
        }
        for xc in args.functionals:
            for device in args.devices:
                if (formula, xc, args.basis, device) in done:
                    n_skip += 1
                    continue
                rec = dict(base, xc=xc, device=device)
                try:
                    rec.update(run_one(device, entry, xc, args))
                    n_ok += 1
                    status = (f"E={rec['e_tot']:.8f} {rec['wall']:7.2f}s "
                              f"cyc={rec['cycles']:>3d}"
                              f"{'' if rec['converged'] else '  NOT CONVERGED'}")
                except Exception as exc:
                    rec['error'] = f'{type(exc).__name__}: {exc}'
                    n_fail += 1
                    status = f'FAILED {rec["error"][:70]}'
                    free_gpu()
                append(args.out, rec)
                print(f'{element:3s} {formula:10s} nao={rec["nao"]:4d} '
                      f'{xc:8s} {device:3s}  {status}', flush=True)

    dt = time.perf_counter() - t_start
    print(f'\ndone: {n_ok} ok, {n_fail} failed, {n_skip} skipped '
          f'in {dt/60:.1f} min -> {args.out}')
    print(f'next: python {os.path.join(HERE, "make_report.py")} --results {args.out}')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
