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
Spin-orbit ECP integrals: gpu4pyscf.gto.ecp.get_ecp_so must reproduce
mol.intor('ECPso') (real [3, nao, nao]); get_soc_1e must reproduce the SOC
block that pyscf.scf.ghf assembles.

The GPU kernel (ECP_so_cart in libgecp) is not built yet -- the value tests
skip until it is; sort_ecp_basis_so is pure Python and is tested unconditionally.
'''

import unittest
import numpy as np
import pyscf
from pyscf import gto, lib

from gpu4pyscf.gto import ecp


# A small SO-ECP: Pb with the CRENBL relativistic ECP (has SO projectors).
def _mol():
    return gto.M(atom='Pb 0 0 0; O 0 0 2.0', basis='crenbl', ecp='crenbl',
                 spin=0, charge=0, cart=False, verbose=0, output='/dev/null')


def setUpModule():
    global mol, has_so
    mol = _mol()
    has_so = bool(np.any(mol._ecpbas[:, gto.SO_TYPE_OF] == 1))


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class UnitSort(unittest.TestCase):
    def test_sort_keeps_only_so_projectors(self):
        ecpbas = mol._ecpbas
        out, uniq_l, l_counts, ecp_loc = ecp.sort_ecp_basis_so(ecpbas)
        if not has_so:
            self.assertEqual(len(out), 0)
            return
        self.assertTrue(np.all(out[:, gto.SO_TYPE_OF] == 1))
        self.assertTrue(np.all(out[:, gto.ANG_OF] >= 0))          # ul rewritten
        self.assertEqual(int(ecp_loc[-1]), len(out))
        self.assertEqual(l_counts.sum(), len(out))

    def test_ul_becomes_lmax_plus_one(self):
        # synthetic: one atom, non-ul l=1 projector + a ul term
        fake = np.array([
            [0, 1, 1, 1, 1, 0, 0, 0],   # ANG_OF=1, SO_TYPE_OF=1
            [0, -1, 1, 1, 1, 0, 0, 0],  # ANG_OF=-1 (ul), SO_TYPE_OF=1
            [0, 2, 1, 1, 0, 0, 0, 0],   # SO_TYPE_OF=0 -> dropped
        ], dtype=np.int32)
        out, uniq_l, _, _ = ecp.sort_ecp_basis_so(fake)
        self.assertEqual(sorted(out[:, gto.ANG_OF].tolist()), [1, 2])  # ul -> 1+1


@unittest.skipUnless(ecp._HAS_SO, 'libgecp built without ECP_so_cart')
class KnownValues(unittest.TestCase):
    def setUp(self):
        if not has_so:
            self.skipTest('test molecule has no SO-ECP projectors')

    def test_get_ecp_so_vs_cpu(self):
        import cupy as cp
        ref = mol.intor('ECPso')                 # [3, nao, nao] real
        got = cp.asnumpy(ecp.get_ecp_so(mol))
        self.assertEqual(got.shape, ref.shape)
        self.assertAlmostEqual(abs(ref - got).max(), 0, 10)

    def test_get_soc_1e_vs_ghf_block(self):
        import cupy as cp
        s = .5 * lib.PauliMatrices
        ref = np.einsum('sxy,spq->xpyq', -1j * s, mol.intor('ECPso'))
        nao = mol.nao
        ref = ref.reshape(2 * nao, 2 * nao)
        got = cp.asnumpy(ecp.get_soc_1e(mol))
        self.assertAlmostEqual(abs(ref - got).max(), 0, 10)

    def test_screened_equals_unscreened(self):
        import cupy as cp
        saved = ecp.SCREEN_ECP
        try:
            ecp.SCREEN_ECP = False
            a = cp.asnumpy(ecp.get_ecp_so(mol))
            ecp.SCREEN_ECP = True
            b = cp.asnumpy(ecp.get_ecp_so(mol))
        finally:
            ecp.SCREEN_ECP = saved
        self.assertAlmostEqual(abs(a - b).max(), 0, 10)


if __name__ == '__main__':
    print('Tests for spin-orbit ECP')
    unittest.main()
