# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

'''
Effect of the (shell, shell, ECP-center) task-list screening in
gpu4pyscf.gto.ecp on a spatially extended system.

Without screening the task list is N_bas^2 * N_ecp_groups; with the two-level
screen each centre only pairs with a bounded neighbourhood, so both the host
task-list construction and the GPU kernel work fall toward O(N).  A loosely
spaced homogeneous chain of heavy (large-core ECP) atoms makes the gap visible.

Usage:
    python benchmark_ecp_screening.py                       # Cu chain, N = 10..80
    python benchmark_ecp_screening.py --n 20 40 80 160 --ip
    python benchmark_ecp_screening.py --element I --basis def2-svp --ecp def2-svp
'''

import argparse
import numpy as np
import cupy as cp
from pyscf import gto

from gpu4pyscf.gto import ecp as gecp
from gpu4pyscf.gto.ecp import get_ecp, get_ecp_ip_sum
from gpu4pyscf.gto.mole import group_basis


def build_chain(n, element, spacing, basis, ecp):
    atoms = [(element, (i * spacing, 0.0, 0.0)) for i in range(n)]
    return gto.M(atom=atoms, basis=basis, ecp=ecp, cart=True, verbose=0)


def task_counts(mol):
    '''(n_full, n_screened): size of the get_ecp task list without / with the
    screen.  n_full is counted analytically (materialising it is the thing we
    are trying to avoid); n_screened is built for real.'''
    _sorted_mol, _, _uniq_l_ctr, l_ctr_counts = group_basis(mol)
    _ecpbas, _uniq_lecp, lecp_counts, ecp_loc = gecp.sort_ecp_basis(
        _sorted_mol._ecpbas)
    l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
    lecp_offsets = np.append(0, np.cumsum(lecp_counts))

    nbas = int(l_ctr_offsets[-1])
    n_ecp_groups = len(ecp_loc) - 1
    n_full = nbas * (nbas + 1) // 2 * n_ecp_groups          # scalar, triangular

    screen_data = gecp._build_screen_data(_sorted_mol, _ecpbas, ecp_loc)
    scr = gecp.make_tasks(l_ctr_offsets, lecp_offsets, screen_data,
                          gecp._ecp_expcutoff(mol))
    n_scr = sum(len(v) for v in scr.values())
    return n_full, n_scr


def free_gpu():
    cp.get_default_memory_pool().free_all_blocks()


def gpu_time(fn, mol, runs=12, warmup=3):
    '''Mean wall time of fn(mol) in seconds, or None on out-of-memory.'''
    ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
    ts = []
    try:
        for _ in range(runs):
            free_gpu()
            ev0.record()
            out = fn(mol)
            ev1.record()
            ev1.synchronize()
            ts.append(cp.cuda.get_elapsed_time(ev0, ev1) / 1000)
            del out
    except cp.cuda.memory.OutOfMemoryError:
        free_gpu()
        return None
    return float(np.mean(ts[warmup:] or ts))


def run(ns, element, spacing, basis, ecp, do_ip):
    saved = gecp.SCREEN_ECP
    print('# ECP task-list screening benchmark')
    print(f'# {element} chain, spacing {spacing} A, cartesian, '
          f'basis={basis} ecp={ecp}, EXPCUTOFF={gecp.EXPCUTOFF}')
    print(f'{"N":>4} {"nao":>6} {"tasks_full":>13} {"tasks_scr":>11} '
          f'{"kept%":>7} {"t_full/s":>9} {"t_scr/s":>9} {"speedup":>8} '
          f'{"maxdiff":>9}')
    try:
        for n in ns:
            mol = build_chain(n, element, spacing, basis, ecp)
            nao = mol.nao
            n_full, n_scr = task_counts(mol)
            kept = 100.0 * n_scr / max(n_full, 1)

            gecp.SCREEN_ECP = False
            t_full = gpu_time(get_ecp, mol)
            gecp.SCREEN_ECP = True
            t_scr = gpu_time(get_ecp, mol)

            if t_full is not None and t_scr is not None:
                gecp.SCREEN_ECP = False
                h_full = get_ecp(mol)
                gecp.SCREEN_ECP = True
                h_scr = get_ecp(mol)
                maxdiff = float(cp.abs(h_full - h_scr).max())
                del h_full, h_scr
                free_gpu()
                speedup = t_full / max(t_scr, 1e-9)
                assert maxdiff < 1e-9, \
                    f'screening changed get_ecp at N={n}: |diff|={maxdiff:.2e}'
            else:
                maxdiff, speedup = float('nan'), float('nan')

            sf = 'OOM' if t_full is None else f'{t_full:9.4f}'
            ss = 'OOM' if t_scr is None else f'{t_scr:9.4f}'
            print(f'{n:>4} {nao:>6} {n_full:>13} {n_scr:>11} {kept:>7.2f} '
                  f'{sf:>9} {ss:>9} {speedup:>8.2f} {maxdiff:>9.1e}')

            if do_ip:
                # get_ecp_ip_sum is the memory-bounded reduction a real
                # gradient uses (grad/rhf.get_hcore); it batches over ECP atoms
                # so it runs at any N, unlike the dense get_ecp_ip tensor.
                gecp.SCREEN_ECP = False
                tip_full = gpu_time(get_ecp_ip_sum, mol)
                gecp.SCREEN_ECP = True
                tip_scr = gpu_time(get_ecp_ip_sum, mol)
                if tip_full and tip_scr:
                    gecp.SCREEN_ECP = False
                    a = get_ecp_ip_sum(mol)
                    gecp.SCREEN_ECP = True
                    b = get_ecp_ip_sum(mol)
                    dip = float(cp.abs(a - b).max())
                    del a, b
                    free_gpu()
                    print(f'     get_ecp_ip_sum {tip_full:9.4f} {tip_scr:9.4f} '
                          f'{tip_full / max(tip_scr, 1e-9):8.2f}  '
                          f'maxdiff {dip:.1e}')
                    assert dip < 1e-9
                else:
                    print('     get_ecp_ip_sum: OOM')
            del mol
            free_gpu()
    finally:
        gecp.SCREEN_ECP = saved


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, nargs='+', default=[10, 20, 40, 80],
                    help='chain lengths to sweep')
    ap.add_argument('--element', default='Cu', help='chain element')
    ap.add_argument('--spacing', type=float, default=4.0,
                    help='nearest-neighbour spacing / Angstrom')
    ap.add_argument('--basis', default='crenbl')
    ap.add_argument('--ecp', default='crenbl')
    ap.add_argument('--ip', action='store_true',
                    help='also benchmark get_ecp_ip (gradient integrals)')
    args = ap.parse_args()
    run(args.n, args.element, args.spacing, args.basis, args.ecp, args.ip)
