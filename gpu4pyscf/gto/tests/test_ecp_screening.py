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
The task-list screening in gpu4pyscf.gto.ecp must not change any ECP integral
(or its derivatives) beyond round-off.  It also has to actually remove work
for a spatially extended system.
'''

import unittest
import numpy as np
import cupy as cp
from pyscf import gto

from gpu4pyscf.gto import ecp


def _chain(n=6, spacing=5.0):
    '''A line of ``n`` Na atoms, each carrying an explicit 10-electron ECP,
    spaced far enough apart that distant (shell, shell, ECP) triples fall
    below the screening cutoff.'''
    basis = gto.basis.parse('''
     Na    S
           2.8000000              0.0210870             -0.0045400              0.0000000
           1.3190000              0.3461290             -0.1703520              0.0000000
           0.9059000              0.0393780              0.1403820              1.0000000
     Na    P
           2.1330000              0.0868660              0.0000000
           1.2000000              0.0000000              0.5000000
           0.3827000              0.5010080              1.0000000
     Na    D
           0.3827000              1.0000000
    ''')
    ecp_na = gto.basis.parse_ecp('''
Na nelec 10
Na ul
2       1.0                   0.5
Na S
2      13.652203             732.2692
2       6.826101              26.484721
Na P
2      10.279868             299.489474
2       5.139934              26.466234
Na D
2       7.349859             124.457595
2       3.674929              14.035995
    ''')
    atoms = [('Na', (i * spacing, 0.0, 0.0)) for i in range(n)]
    return gto.M(atom=atoms, basis={'Na': basis}, ecp={'Na': ecp_na},
                 cart=True, output='/dev/null')


def setUpModule():
    global mol
    mol = _chain()


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):
    def setUp(self):
        self._saved = ecp.SCREEN_ECP

    def tearDown(self):
        ecp.SCREEN_ECP = self._saved

    def _ref_and_screened(self, fn, *args, **kwargs):
        ecp.SCREEN_ECP = False
        ref = fn(mol, *args, **kwargs)
        ecp.SCREEN_ECP = True
        got = fn(mol, *args, **kwargs)
        return cp.asnumpy(ref), cp.asnumpy(got)

    def test_get_ecp_unchanged(self):
        ref, got = self._ref_and_screened(ecp.get_ecp)
        self.assertAlmostEqual(abs(ref - got).max(), 0, 10)

    def test_get_ecp_ip_unchanged(self):
        ref, got = self._ref_and_screened(ecp.get_ecp_ip)
        self.assertAlmostEqual(abs(ref - got).max(), 0, 10)

    def test_get_ecp_ipip_unchanged(self):
        for ip_type in ('ipipv', 'ipvip'):
            ref, got = self._ref_and_screened(ecp.get_ecp_ipip, ip_type=ip_type)
            self.assertAlmostEqual(abs(ref - got).max(), 0, 10,
                                   msg=f'ip_type={ip_type}')

    def test_screening_removes_tasks(self):
        # Reproduce the task-list build path of get_ecp and check that
        # screening actually drops triples for this geometry.
        from gpu4pyscf.gto.mole import group_basis
        _sorted_mol, _, uniq_l_ctr, l_ctr_counts = group_basis(mol)
        _ecpbas, uniq_lecp, lecp_counts, ecp_loc = ecp.sort_ecp_basis(
            _sorted_mol._ecpbas)
        l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        lecp_offsets = np.append(0, np.cumsum(lecp_counts))
        screen_data = ecp._build_screen_data(_sorted_mol, _ecpbas, ecp_loc)

        full = ecp.make_tasks(l_ctr_offsets, lecp_offsets)
        screened = ecp.make_tasks(l_ctr_offsets, lecp_offsets, screen_data,
                                  ecp._ecp_expcutoff(mol))
        n_full = sum(len(v) for v in full.values())
        n_screened = sum(len(v) for v in screened.values())
        self.assertGreater(n_full, 0)
        self.assertLess(n_screened, n_full)
        self.assertGreater(n_screened, 0)


class UnitScreenGrid(unittest.TestCase):
    '''_screen_grid math, independent of any GPU kernel.'''

    def test_matches_reference_formula(self):
        rng = np.random.default_rng(0)
        bas_min_exp = np.array([0.4, 1.1, 2.5])
        bas_coords = rng.normal(size=(3, 3))
        ecp_min_exp = np.array([0.8, 3.0])
        ecp_coords = rng.normal(size=(2, 3))
        screen_data = (bas_min_exp, bas_coords, ecp_min_exp, ecp_coords)

        grid = np.array([[i, j, k]
                         for i in range(3) for j in range(3) for k in range(2)])
        cutoff = 5.0
        out = ecp._screen_grid(grid, screen_data, cutoff)

        kept = set(map(tuple, out.tolist()))
        for i, j, k in map(tuple, grid.tolist()):
            ai, aj, ak = bas_min_exp[i], bas_min_exp[j], ecp_min_exp[k]
            rrab = np.sum((bas_coords[i] - bas_coords[j]) ** 2)
            rrca = np.sum((ecp_coords[k] - bas_coords[i]) ** 2)
            rrcb = np.sum((ecp_coords[k] - bas_coords[j]) ** 2)
            eijk = (ai*aj*rrab + ai*ak*rrca + aj*ak*rrcb) / (ai + aj + ak)
            self.assertEqual((i, j, k) in kept, eijk < cutoff)

    def test_empty_grid(self):
        screen_data = (np.ones(1), np.zeros((1, 3)), np.ones(1), np.zeros((1, 3)))
        out = ecp._screen_grid(np.zeros((0, 3), dtype=int), screen_data, 1.0)
        self.assertEqual(out.shape, (0, 3))

    def test_screen_block_matches_full_grid(self):
        '''The two-level _screen_block must select exactly the same task rows as
        _screen_grid applied to the full meshgrid, incl. padding (inf-exponent)
        shells and zero-exponent ECP groups.'''
        rng = np.random.default_rng(7)
        for _ in range(50):
            nb = int(rng.integers(2, 8))
            nk = int(rng.integers(1, 5))
            scale = float(rng.uniform(0.5, 4.0))
            bas_min_exp = rng.uniform(0.2, 5.0, nb)
            if rng.random() < 0.2:
                bas_min_exp[rng.integers(nb)] = np.inf
            ecp_min_exp = rng.uniform(0.1, 3.0, nk)
            if rng.random() < 0.2:
                ecp_min_exp[rng.integers(nk)] = 0.0
            screen_data = (bas_min_exp, rng.normal(scale=scale, size=(nb, 3)),
                           ecp_min_exp, rng.normal(scale=scale, size=(nk, 3)))
            cutoff = float(rng.uniform(2.0, 40.0))
            ish = np.arange(nb)
            jsh = np.arange(nb)
            ksh = np.arange(nk)
            for triangular in (False, True):
                full = np.stack(np.meshgrid(ish, jsh, ksh),
                                axis=-1).reshape(-1, 3)
                if triangular:
                    full = full[full[:, 0] <= full[:, 1]]
                ref = sorted(map(tuple,
                                 ecp._screen_grid(full, screen_data, cutoff).tolist()))
                blk = sorted(map(tuple,
                                 ecp._screen_block(ish, jsh, ksh, screen_data,
                                                   cutoff, triangular).tolist()))
                self.assertEqual(ref, blk, msg=f'triangular={triangular}')


if __name__ == '__main__':
    print('Tests for ECP task-list screening')
    unittest.main()
