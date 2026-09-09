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

"""Actinide spin-orbit ECP: the ECP60MWB-SO set (pyscf name ``ecpds60mwbso``,
exposed through :mod:`gpu4pyscf.gto.actinide_ecp`) must

  1. flip ``mol.has_ecp_soc()`` to True, with SO projectors l = 1..4;
  2. stay inside the gpu4pyscf angular table (``_l_op`` l = 0..4, ``ECP_LMAX`` 4);
  3. reproduce ``mol.intor('ECPso')`` (sph & cart) to < 1e-9 through
     ``get_ecp_so``, and the pyscf spinor assembly through ``get_soc_1e``  --
     the Na-Bi sweep never exercised an actinide (60-core, f/g) SO projector;
  4. leave the scalar (AREP) integrals byte-identical to ``stuttgart_rsc``;
  5. give a U 5f spin-orbit splitting with the right L.S structure
     (6-fold below 8-fold, eigenvalue ratio -2 : +3/2) and physical magnitude.

The GPU kernel (``ECP_so_cart`` in libgecp) may not be built -- the value tests
skip if so.
"""

import unittest
import numpy as np
from pyscf import gto, lib

from gpu4pyscf.gto import ecp
from gpu4pyscf.gto.actinide_ecp import (
    get_actinide_so_ecp, actinide_basis, has_actinide_so_ecp, ACTINIDES)

H2CM = 219474.6313632
TEST_ELEMENTS = ('U', 'Np', 'Pu')
_HAS_SO = getattr(ecp, '_HAS_SO', False)


def _diatomic(elem, cart, r=2.10):
    # a light partner + no SCF: spin=None, we only touch integrals
    return gto.M(atom=f'{elem} 0 0 0; F 0 0 {r}',
                 basis={elem: actinide_basis(elem), 'F': 'def2-svp'},
                 ecp={elem: get_actinide_so_ecp(elem)},
                 cart=cart, spin=None, verbose=0, output='/dev/null')


class ActinideSOECPMeta(unittest.TestCase):
    def test_accessor(self):
        self.assertEqual(get_actinide_so_ecp('U'), 'ecpds60mwbso')
        self.assertEqual(get_actinide_so_ecp('u'), 'ecpds60mwbso')
        self.assertEqual(get_actinide_so_ecp('U238'), 'ecpds60mwbso')
        self.assertTrue(has_actinide_so_ecp('Pu'))
        self.assertFalse(has_actinide_so_ecp('Ce'))
        self.assertRaises(KeyError, get_actinide_so_ecp, 'Ce')
        self.assertEqual(set(ACTINIDES) & {'U', 'Np', 'Pu', 'Th', 'Cm'},
                         {'U', 'Np', 'Pu', 'Th', 'Cm'})

    def test_has_ecp_soc_and_projector_l(self):
        # check 1
        for elem in TEST_ELEMENTS:
            m = _diatomic(elem, cart=False)
            self.assertTrue(m.has_ecp_soc(), elem)
            so_l = sorted(set(m._ecpbas[m._ecpbas[:, gto.SO_TYPE_OF] == 1,
                                        gto.ANG_OF].tolist()))
            self.assertEqual(so_l, [1, 2, 3, 4], elem)
            m.stdout.close()

    def test_angular_table_covers_projectors(self):
        # check 2 -- _l_op in lib/ecp/so_ang_matrix.cu is generated for l = 0..4
        # from pyscf's _angular_moment_matrix_*; ECP_LMAX in lib/ecp/ecp.h is 4.
        import os
        here = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        cu = open(os.path.join(here, 'lib', 'ecp', 'so_ang_matrix.cu')).read()
        self.assertIn('_l_op_f', cu)          # l = 3
        self.assertIn('_l_op_g', cu)          # l = 4
        self.assertIn('*_l_op[5]', cu)
        h = open(os.path.join(here, 'lib', 'ecp', 'ecp.h')).read()
        self.assertIn('ECP_LMAX        4', h)


@unittest.skipUnless(_HAS_SO, 'libgecp built without ECP_so_cart')
class ActinideSOECPValues(unittest.TestCase):
    def test_get_ecp_so_vs_intor(self):
        # check 3
        import cupy as cp
        for elem in TEST_ELEMENTS:
            for cart in (False, True):
                m = _diatomic(elem, cart)
                ref = np.asarray(m.intor('ECPso_cart' if cart else 'ECPso_sph'))
                got = cp.asnumpy(ecp.get_ecp_so(m))
                self.assertEqual(got.shape, ref.shape)
                self.assertLess(abs(ref - got).max(), 1e-9,
                                f'{elem} {"cart" if cart else "sph"}')
                m.stdout.close()

    def test_get_soc_1e_vs_pyscf_assembly(self):
        # check 3 (spinor operator)
        import cupy as cp
        for elem in TEST_ELEMENTS:
            m = _diatomic(elem, cart=False)
            nao = m.nao_nr()
            ref = lib.einsum('sxy,spq->xpyq', -1j * .5 * lib.PauliMatrices,
                             m.intor('ECPso')).reshape(2 * nao, 2 * nao)
            got = cp.asnumpy(ecp.get_soc_1e(m))
            self.assertLess(abs(ref - got).max(), 1e-9, elem)
            m.stdout.close()

    def test_scalar_arep_unchanged(self):
        # check 4 -- ecpds60mwbso AREP == stuttgart_rsc (same ECP60MWB)
        import cupy as cp
        for elem in TEST_ELEMENTS:
            for cart in (False, True):
                m_so = _diatomic(elem, cart)
                m_sc = gto.M(atom=m_so.atom, basis={elem: actinide_basis(elem),
                                                    'F': 'def2-svp'},
                             ecp={elem: 'stuttgart_rsc'}, cart=cart, spin=None,
                             verbose=0, output='/dev/null')
                d = abs(cp.asnumpy(ecp.get_ecp(m_so)) -
                        cp.asnumpy(ecp.get_ecp(m_sc))).max()
                self.assertLess(d, 1e-12, f'{elem} {"cart" if cart else "sph"}')
                m_so.stdout.close()
                m_sc.stdout.close()

    def test_u_5f_spin_orbit_structure(self):
        # check 5 -- project get_ecp_so onto a single 5f radial function; the
        # 14x14 spinor SO operator must split 6 (j=5/2) below 8 (j=7/2) with
        # L.S eigenvalue ratio -2 : +3/2, and a physical splitting.
        import cupy as cp
        fbasis = {'U': gto.basis.parse('''
U    S
   3.0  1.0
U    P
   2.0  1.0
U    D
   1.5  1.0
U    F
   1.2  1.0
''')}
        m = gto.M(atom='U 0 0 0', basis=fbasis,
                  ecp={'U': get_actinide_so_ecp('U')}, spin=None, verbose=0)
        f_ao = []
        p0 = 0
        for ib in range(m.nbas):
            l = m.bas_angular(ib)
            for _ in range(m.bas_nctr(ib)):
                if l == 3:
                    f_ao += list(range(p0, p0 + 2 * l + 1))
                p0 += 2 * l + 1
        f_ao = np.array(f_ao)
        self.assertEqual(len(f_ao), 7)

        W = cp.asnumpy(ecp.get_ecp_so(m))[:, f_ao][:, :, f_ao]     # [3,7,7]
        H = np.einsum('sxy,sij->xiyj', -1j * .5 * lib.PauliMatrices,
                      W).reshape(14, 14)
        w = np.linalg.eigvalsh(H)
        lo, hi = w[:6], w[6:]
        # 6-fold and 8-fold levels
        self.assertLess(lo.std() * H2CM, 1.0)
        self.assertLess(hi.std() * H2CM, 1.0)
        e56, e78 = lo.mean(), hi.mean()
        self.assertLess(e56, e78)                        # 5f5/2 below 5f7/2
        # L.S: j=5/2 -> -2 zeta ; j=7/2 -> +3/2 zeta ; ratio -4/3
        self.assertAlmostEqual(e56 / e78, -4.0 / 3.0, places=4)
        split_cm = (e78 - e56) * H2CM
        self.assertTrue(50.0 < split_cm < 5.0e4, f'5f SO split {split_cm:.1f} cm^-1')


if __name__ == '__main__':
    unittest.main()
