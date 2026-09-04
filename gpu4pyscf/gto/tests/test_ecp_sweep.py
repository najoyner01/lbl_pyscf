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
Breadth sweep of the GPU ECP integrals (scalar, 1st/2nd derivative, spin-orbit)
against PySCF's CPU `mol.intor(...)` across many ECP sets and elements, in both
cartesian and spherical bases, with an s..g orbital basis so every (li, lj)
combination up to g is exercised.

`get_ecp*` return the matrix in the mol's own AO basis, so cartesian mols are
compared against `..._cart` intors and spherical mols against the plain intors.

Each (element, ECP set) case that PySCF cannot build (element absent from the
set) is skipped, so the matrix is self-pruning -- the run report shows what was
actually covered.
'''

import unittest
import itertools
import numpy as np
import pyscf
from pyscf import gto, lib

from gpu4pyscf.gto.ecp import (get_ecp, get_ecp_ip, get_ecp_ipip, get_ecp_so,
                               get_soc_1e, sort_ecp_basis_so)
from gpu4pyscf.__config__ import shm_size

# s..g uncontracted probe basis -- decouples the AO l-sweep from the ECP core.
PROBE_BASIS = gto.basis.parse('''
X    S
      4.0   1.0
      0.9   1.0
X    P
      3.0   1.0
      0.7   1.0
X    D
      1.6   1.0
X    F
      1.3   1.0
X    G
      1.1   1.0
''')

SCALAR_SETS = ['crenbl', 'crenbs', 'lanl2dz', 'lanl08', 'lanl2tz', 'sbkjc',
               'stuttgartrsc', 'stuttgartrlc', 'def2-svp', 'def2-tzvp',
               'ccecpccpvdz', 'ccecp']
SO_SETS = ['ecpds10mdfso', 'ecpds28mdfso', 'ecpds28mwbso', 'ecpds46mdfso',
           'ecpds60mdfso', 'ecpds78mdfso']
ELEMENTS = ['Na', 'K', 'Ca', 'Sc', 'Cu', 'Zn', 'Ga', 'Br', 'Ag', 'Sn', 'I',
            'Cs', 'Ce', 'Yb', 'Pt', 'Au', 'Hg', 'Pb', 'Bi']

SCALAR_TOL = 1e-9
IP_TOL = 1e-8
# 2nd-derivative (Hessian) integrals lose a little more precision on the hardest
# f-in-core / high-l cases (Ce, Yb with g AOs sit at ~5e-8); test_ecp.py keeps
# the tight 1e-8 on canonical systems.
IPIP_TOL = 1e-7


def _build(elem, ecp_name, cart, dist=2.4):
    try:
        mol = gto.M(atom=f'{elem} 0 0 0; {elem} 0 0 {dist}',
                    basis={elem: PROBE_BASIS}, ecp={elem: ecp_name},
                    cart=cart, spin=None, verbose=0, output='/dev/null')
    except (KeyError, RuntimeError, ValueError):
        return None
    if len(mol._ecpbas) == 0:
        mol.stdout.close()
        return None
    return mol


def _suffix(cart):
    return '_cart' if cart else '_sph'


class ScalarSweep(unittest.TestCase):
    def test_scalar_and_derivs(self):
        import cupy as cp
        covered = 0
        for elem, ecp_name, cart in itertools.product(
                ELEMENTS, SCALAR_SETS, (False, True)):
            mol = _build(elem, ecp_name, cart)
            if mol is None:
                continue
            covered += 1
            tag = f'{elem}/{ecp_name}/{"cart" if cart else "sph"}'
            sfx = _suffix(cart)
            try:
                with self.subTest(case=tag, intor='ECPscalar'):
                    ref = mol.intor('ECPscalar' + sfx)
                    got = cp.asnumpy(get_ecp(mol))
                    self.assertEqual(got.shape, ref.shape, tag)
                    self.assertLess(abs(ref - got).max(), SCALAR_TOL, tag)

                if shm_size >= 64 * 1024:
                    with self.subTest(case=tag, intor='ECPscalar_ipnuc'):
                        ref = mol.intor('ECPscalar_ipnuc' + sfx)
                        got = cp.asnumpy(get_ecp_ip(mol).sum(axis=0))
                        self.assertLess(abs(ref - got).max(), IP_TOL, tag)

                    with self.subTest(case=tag, intor='ECPscalar_ipipnuc'):
                        ref = mol.intor('ECPscalar_ipipnuc' + sfx, comp=9)
                        got = cp.asnumpy(
                            get_ecp_ipip(mol, 'ipipv').sum(axis=0))
                        self.assertLess(abs(ref - got).max(), IPIP_TOL, tag)

                    with self.subTest(case=tag, intor='ECPscalar_ipnucip'):
                        ref = mol.intor('ECPscalar_ipnucip' + sfx, comp=9)
                        got = cp.asnumpy(
                            get_ecp_ipip(mol, 'ipvip').sum(axis=0))
                        self.assertLess(abs(ref - got).max(), IPIP_TOL, tag)

                    with self.subTest(case=tag, intor='ECPscalar_iprinv'):
                        h_gpu = get_ecp_ip(mol)
                        atoms = sorted(set(mol._ecpbas[:, gto.ATOM_OF]))
                        for k, atm_id in enumerate(atoms):
                            with mol.with_rinv_at_nucleus(atm_id):
                                ref = mol.intor('ECPscalar_iprinv' + sfx)
                            self.assertLess(
                                abs(ref - cp.asnumpy(h_gpu[k])).max(),
                                IP_TOL, f'{tag} atom {atm_id}')
            finally:
                mol.stdout.close()
        self.assertGreater(covered, 10, 'sweep covered too few cases')


class HeteronuclearSweep(unittest.TestCase):
    def test_two_ecp_atoms(self):
        import cupy as cp
        pairs = [('Cu', 'I'), ('Br', 'I'), ('Na', 'Au'), ('K', 'Pb')]
        for (a, b), ecp_name, cart in itertools.product(
                pairs, ('crenbl', 'lanl2dz'), (False, True)):
            try:
                mol = gto.M(atom=f'{a} 0 0 0; {b} 0 0 2.6',
                            basis={a: PROBE_BASIS, b: PROBE_BASIS},
                            ecp={a: ecp_name, b: ecp_name},
                            cart=cart, verbose=0, output='/dev/null')
            except (KeyError, RuntimeError, ValueError):
                continue
            if len(mol._ecpbas) == 0:
                mol.stdout.close()
                continue
            tag = f'{a}-{b}/{ecp_name}/{"cart" if cart else "sph"}'
            sfx = _suffix(cart)
            try:
                with self.subTest(case=tag):
                    ref = mol.intor('ECPscalar' + sfx)
                    got = cp.asnumpy(get_ecp(mol))
                    self.assertLess(abs(ref - got).max(), SCALAR_TOL, tag)
                if shm_size >= 64 * 1024:
                    with self.subTest(case=tag, intor='iprinv'):
                        h_gpu = get_ecp_ip(mol)
                        for k, atm_id in enumerate(
                                sorted(set(mol._ecpbas[:, gto.ATOM_OF]))):
                            with mol.with_rinv_at_nucleus(atm_id):
                                ref = mol.intor('ECPscalar_iprinv' + sfx)
                            self.assertLess(
                                abs(ref - cp.asnumpy(h_gpu[k])).max(),
                                IP_TOL, f'{tag} atom {atm_id}')
            finally:
                mol.stdout.close()


class SOSweep(unittest.TestCase):
    def test_so_sweep(self):
        import cupy as cp
        covered = 0
        for elem, ecp_name in itertools.product(ELEMENTS, SO_SETS):
            mol = _build(elem, ecp_name, cart=False)
            if mol is None:
                continue
            if not np.any(mol._ecpbas[:, gto.SO_TYPE_OF] == 1):
                mol.stdout.close()
                continue
            covered += 1
            tag = f'{elem}/{ecp_name}'
            try:
                with self.subTest(case=tag, intor='ECPso'):
                    ref = mol.intor('ECPso')
                    got = cp.asnumpy(get_ecp_so(mol))
                    self.assertEqual(got.shape, ref.shape, tag)
                    self.assertLess(abs(ref - got).max(), SCALAR_TOL, tag)

                with self.subTest(case=tag, intor='soc_1e'):
                    s = .5 * lib.PauliMatrices
                    nao = mol.nao
                    ref = np.einsum('sxy,spq->xpyq', -1j * s,
                                    mol.intor('ECPso')).reshape(2*nao, 2*nao)
                    got = cp.asnumpy(get_soc_1e(mol))
                    self.assertLess(abs(ref - got).max(), SCALAR_TOL, tag)
            finally:
                mol.stdout.close()
        self.assertGreater(covered, 0, 'no SO-ECP (element, set) case covered')


class UnitSO(unittest.TestCase):
    def test_sort_ecp_basis_so_shape(self):
        so = gto.basis.parse_ecp('''
Na nelec 10
Na ul
2   1.0  0.0
Na S
2   6.0  90.0  0.0
Na P
2   4.0  10.0  -20.0
''')
        mol = gto.M(atom='Na 0 0 0', basis={'Na': PROBE_BASIS},
                    ecp={'Na': so}, spin=1, verbose=0, output='/dev/null')
        try:
            has_so = np.any(mol._ecpbas[:, gto.SO_TYPE_OF] == 1)
            out, uniq_l, l_counts, ecp_loc = sort_ecp_basis_so(mol._ecpbas)
            if has_so:
                self.assertTrue(np.all(out[:, gto.SO_TYPE_OF] == 1))
                self.assertTrue(np.all(out[:, gto.ANG_OF] >= 0))
                self.assertEqual(int(ecp_loc[-1]), len(out))
        finally:
            mol.stdout.close()


if __name__ == '__main__':
    print('ECP validation sweep')
    unittest.main()
