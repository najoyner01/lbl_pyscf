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

Without screening the task list grows as O(N_bas^2 * N_ecp); with screening each
centre only interacts with a bounded neighbourhood, so the ECP work drops toward
O(N).  A loosely spaced homogeneous chain of heavy (large-core ECP) atoms makes
the gap visible: the speed-up should grow with chain length.

Usage:
    python benchmark_ecp_screening.py                       # Cu chain, N = 10..80
    python benchmark_ecp_screening.py --n 20 40 80 160 --ip
    python benchmark_ecp_screening.py --element I --spacing 4.5
'''

import argparse
import numpy as np
import cupy as cp
from pyscf import gto

from gpu4pyscf.gto import ecp as gecp
from gpu4pyscf.gto.ecp import get_ecp, get_ecp_ip
from gpu4pyscf.gto.mole import group_basis


def build_chain(n, element='Cu', spacing=4.0, basis='crenbl', ecp='crenbl'):
    atoms = [(element, (i * spacing, 0.0, 0.0)) for i in range(n)]
    return gto.M(atom=atoms, basis=basis, ecp=ecp, cart=True, verbose=0)


def count_tasks(mol):
    '''(n_full, n_screened) task-list sizes along get_ecp's build path.

    Note: the unscreened list is materialised in host memory and grows fast
    (~N_bas^2 * N_ecp_groups); keep N modest unless you have the RAM.
    '''
    _sorted_mol, _, _uniq_l_ctr, l_ctr_counts = group_basis(mol)
    _ecpbas, _uniq_lecp, lecp_counts, ecp_loc = gecp.sort_ecp_basis(
        _sorted_mol._ecpbas)
    l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
    lecp_offsets = np.append(0, np.cumsum(lecp_counts))
    screen_data = gecp._build_screen_data(_sorted_mol, _ecpbas, ecp_loc)

    full = gecp.make_tasks(l_ctr_offsets, lecp_offsets)
    scr = gecp.make_tasks(l_ctr_offsets, lecp_offsets, screen_data,
                          gecp._ecp_expcutoff(mol))
    n_full = sum(len(v) for v in full.values())
    n_scr = sum(len(v) for v in scr.values())
    return n_full, n_scr


def gpu_time(fn, mol, runs=12, warmup=3):
    ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
    out = None
    ts = []
    for _ in range(runs):
        ev0.record()
        out = fn(mol)
        ev1.record()
        ev1.synchronize()
        ts.append(cp.cuda.get_elapsed_time(ev0, ev1) / 1000)
    return out, float(np.mean(ts[warmup:]))


def run(ns, element, spacing, do_ip):
    saved = gecp.SCREEN_ECP
    print('# ECP task-list screening benchmark')
    print(f'# {element} chain, spacing {spacing} A, cartesian, '
          f'basis=ecp=crenbl, EXPCUTOFF={gecp.EXPCUTOFF}')
    print(f'{"N":>4} {"nao":>6} {"tasks_full":>12} {"tasks_scr":>11} '
          f'{"kept%":>6} {"t_full/s":>10} {"t_scr/s":>10} {"speedup":>8} '
          f'{"maxdiff":>9}')
    try:
        for n in ns:
            mol = build_chain(n, element, spacing)
            nao = mol.nao
            n_full, n_scr = count_tasks(mol)

            gecp.SCREEN_ECP = False
            h_full, t_full = gpu_time(get_ecp, mol)
            gecp.SCREEN_ECP = True
            h_scr, t_scr = gpu_time(get_ecp, mol)

            maxdiff = float(cp.abs(h_full - h_scr).max())
            speedup = t_full / max(t_scr, 1e-9)
            kept = 100.0 * n_scr / max(n_full, 1)
            print(f'{n:>4} {nao:>6} {n_full:>12} {n_scr:>11} {kept:>6.1f} '
                  f'{t_full:>10.4f} {t_scr:>10.4f} {speedup:>8.2f} '
                  f'{maxdiff:>9.1e}')
            assert maxdiff < 1e-9, \
                f'screening changed get_ecp at N={n}: max |diff| = {maxdiff:.2e}'

            if do_ip:
                gecp.SCREEN_ECP = False
                _, tip_full = gpu_time(get_ecp_ip, mol)
                gecp.SCREEN_ECP = True
                _, tip_scr = gpu_time(get_ecp_ip, mol)
                print(f'     get_ecp_ip           '
                      f'{tip_full:>10.4f} {tip_scr:>10.4f} '
                      f'{tip_full / max(tip_scr, 1e-9):>8.2f}')
    finally:
        gecp.SCREEN_ECP = saved


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, nargs='+', default=[10, 20, 40, 80],
                    help='chain lengths to sweep')
    ap.add_argument('--element', default='Cu', help='chain element (needs a crenbl ECP)')
    ap.add_argument('--spacing', type=float, default=4.0, help='nearest-neighbour spacing / Angstrom')
    ap.add_argument('--ip', action='store_true', help='also benchmark get_ecp_ip (gradient integrals)')
    args = ap.parse_args()
    run(args.n, args.element, args.spacing, args.ip)
