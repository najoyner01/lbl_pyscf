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


# Pb/CRENBL: SO projectors are the `ul` term (lc == -1 -> lmax+1 on the host).
def _mol_ul():
    return gto.M(atom='Pb 0 0 0; O 0 0 2.0', basis='crenbl', ecp='crenbl',
                 spin=0, charge=0, cart=False, verbose=0, output='/dev/null')


# K/ecpds10mdfso: explicit S,P,D,F SO projectors -> exercises lc = 0..3.
def _mol_explicit():
    kbas = gto.basis.parse('''
K    S
      3.0        1.0
      0.8        1.0
K    P
      2.5        1.0
      0.5        1.0
K    D
      1.0        1.0
K    F
      1.2        1.0
K    G
      1.1        1.0
''')
    return gto.M(atom='K 0 0 0; F 0 0 2.2', basis={'K': kbas, 'F': kbas},
                 ecp={'K': 'ecpds10mdfso'}, spin=0, charge=0,
                 cart=False, verbose=0, output='/dev/null')


def setUpModule():
    global mol, mol_x, has_so
    mol = _mol_ul()
    mol_x = _mol_explicit()
    has_so = bool(np.any(mol._ecpbas[:, gto.SO_TYPE_OF] == 1))


def tearDownModule():
    global mol, mol_x
    mol.stdout.close()
    mol_x.stdout.close()
    del mol, mol_x


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
        # l_counts is (l,atom)-groups per unique l, matching sort_ecp_basis
        self.assertEqual(int(l_counts.sum()), len(ecp_loc) - 1)
        self.assertEqual(len(uniq_l), len(l_counts))

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

    def test_get_ecp_so_vs_cpu_ul(self):
        self._compare(mol)

    def test_get_ecp_so_vs_cpu_explicit_projectors(self):
        # K/ecpds10mdfso: SO projectors with explicit lc = 0,1,2,3
        self._compare(mol_x)

    def _compare(self, m):
        import cupy as cp
        ref = m.intor('ECPso')                    # [3, nao, nao] real
        got = cp.asnumpy(ecp.get_ecp_so(m))
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


@unittest.skipUnless(getattr(ecp, '_HAS_SO_IP', False),
                     'libgecp built without ECP_so_ip_cart')
class SoIpFiniteDifference(unittest.TestCase):
    '''V1 of docs/ghf-gradient-design.md step 3: the bra-derivative SO-ECP
    integral (get_ecp_so_ip / loop_ecp_so_ip) assembled per atom must match a
    central finite difference of the energy integral get_ecp_so itself.

    There is no CPU reference (pyscf registers ECPso but no ECPso_ip), so this
    self-contained integral-level check is the primary correctness gate for the
    kernel -- it isolates sign / factor / layout errors with no SCF involved.
    '''

    def _assemble_analytic(self, m):
        '''d/dR_A get_ecp_so(m)[a]  ->  [natm, 3(a), 3(xyz), nao, nao].

        For a matrix element <i| l_a U_SO |j> (i = bra, j = ket):
            d/dR_A = -Gsum[a,x,i,j]           (bra AO i centred on A)
                     +Gsum[a,x,j,i]           (ket AO j centred on A)
                     +(G^A - G^A.T)[a,x,i,j]  (A is the SO-ECP centre;
                                               translational invariance)
        with G[a,x,r,s] = <d/dr_x r | l_a U_SO | s>, Gsum summed over all
        SO-ECP centres and G^A restricted to centre A.
        '''
        import cupy as cp
        natm, nao = m.natm, m.nao
        so_atoms = ecp._default_ecp_so_atoms(m)
        Gc = {}
        Gsum = cp.zeros([3, 3, nao, nao])
        for batch, G in ecp.loop_ecp_so_ip(m, ecp_atoms=so_atoms):
            for r, C in enumerate(batch):
                Gc[int(C)] = cp.asnumpy(G[r])
                Gsum += G[r]
        Gsum = cp.asnumpy(Gsum)
        GsumT = np.swapaxes(Gsum, -1, -2)

        aoslices = m.aoslice_by_atom()
        ao_atom = np.empty(nao, dtype=int)
        for A in range(natm):
            ao_atom[aoslices[A, 2]:aoslices[A, 3]] = A

        dso = np.zeros([natm, 3, 3, nao, nao])
        for A in range(natm):
            mask = ao_atom == A
            dso[A][:, :, mask, :] += -Gsum[:, :, mask, :]
            dso[A][:, :, :, mask] += GsumT[:, :, :, mask]
            if A in Gc:
                g = Gc[A]
                dso[A] += g - np.swapaxes(g, -1, -2)
        return dso

    def _fd(self, build, h=1e-4):
        import cupy as cp
        m = build()
        natm, nao = m.natm, m.nao
        coords0 = m.atom_coords()
        dso = np.zeros([natm, 3, 3, nao, nao])
        for A in range(natm):
            for x in range(3):
                cp_, cm_ = coords0.copy(), coords0.copy()
                cp_[A, x] += h
                cm_[A, x] -= h
                mp = build().set_geom_(cp_, unit='Bohr', inplace=False)
                mp.build(False, False)
                mm = build().set_geom_(cm_, unit='Bohr', inplace=False)
                mm.build(False, False)
                dso[A, :, x] = (cp.asnumpy(ecp.get_ecp_so(mp))
                                - cp.asnumpy(ecp.get_ecp_so(mm))) / (2 * h)
        return dso

    def _check(self, build):
        an = self._assemble_analytic(build())
        fd = self._fd(build)
        err = float(np.linalg.norm((an - fd).ravel()))
        self.assertLess(err, 1e-6)

    def test_fd_ul_projector(self):
        # Pb/CRENBL: SO term is the `ul` (lc == -1 -> lmax+1) projector
        self._check(lambda: gto.M(
            atom='Pb 0 0 0; O 0 0 2.1', basis='crenbl', ecp='crenbl',
            spin=0, unit='Bohr', verbose=0, output='/dev/null'))

    def test_fd_explicit_projectors(self):
        # K/ecpds10mdfso: explicit SO projectors, lc = 0..3
        kbas = gto.basis.parse('''
K    S
      3.0        1.0
      0.8        1.0
K    P
      2.5        1.0
      0.5        1.0
K    D
      1.0        1.0
K    F
      1.2        1.0
K    G
      1.1        1.0
''')
        self._check(lambda: gto.M(
            atom='K 0 0 0; F 0 0 4.0', basis={'K': kbas, 'F': kbas},
            ecp={'K': 'ecpds10mdfso'}, spin=0, unit='Bohr', verbose=0,
            output='/dev/null'))


if __name__ == '__main__':
    print('Tests for spin-orbit ECP')
    unittest.main()
