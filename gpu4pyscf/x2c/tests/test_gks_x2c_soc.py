# Copyright 2026 The PySCF Developers. All Rights Reserved.
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
Spin-orbit X2C wired into GKS (DFT), not just GHF.

gpu4pyscf.dft.gks.GKS(rks.KohnShamDFT, GHF) -- GKS is-a GHF -- and
GHF.x2c1e()/.x2c() only requires isinstance(mf, ghf.GHF); the X2C mixin
(X2C1E_GSCF) only overrides get_hcore, leaving GKS's own get_veff/XC
machinery untouched. So GKS(mol, xc=...).x2c1e() is expected to already work
via inheritance, with no GKS-specific code. This test is the first check of
that claim -- see docs/adf-parity-strategy.md Sec 1b.
'''

import unittest
import pyscf
from pyscf import gto, lib
from gpu4pyscf.dft import gks


def setUpModule():
    global mol, mol_heavy
    mol = gto.M(
        verbose=0,
        atom='''
            O     0    0        0
            H     0    -0.757   0.587
            H     0    0.757    0.587''',
        basis='cc-pvdz',
    )
    # A heavier atom so the spin-orbit splitting is not numerically trivial.
    mol_heavy = gto.M(verbose=0, atom='I 0 0 0', basis='sto-3g', spin=1)


def tearDownModule():
    global mol, mol_heavy
    mol.stdout.close()
    mol_heavy.stdout.close()
    del mol, mol_heavy


class KnownValues(unittest.TestCase):
    def test_gks_x2c_soc_matches_cpu(self):
        ref = mol.GKS(xc='pbe0').x2c1e().run()
        mf = gks.GKS(mol, xc='pbe0').x2c1e().run()
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 8)
        self.assertAlmostEqual(
            abs(mf.mo_energy.get() - ref.mo_energy).max(), 0, 5)

    def test_gks_x2c_soc_is_ghf_subclass_path(self):
        # the whole claim rests on this holding; make it explicit
        mf = gks.GKS(mol, xc='pbe0')
        from gpu4pyscf.scf.ghf import GHF
        self.assertIsInstance(mf, GHF)
        x2c_mf = mf.x2c1e()
        self.assertTrue(hasattr(x2c_mf, 'with_x2c'))

    def test_gks_x2c_soc_heavy_atom(self):
        # exercise real (non-epsilon) spin-orbit splitting on a heavy atom
        ref = mol_heavy.GKS(xc='pbe0').x2c1e().run()
        mf = gks.GKS(mol_heavy, xc='pbe0').x2c1e().run()
        self.assertAlmostEqual(mf.e_tot, ref.e_tot, 7)
        self.assertAlmostEqual(
            abs(mf.mo_energy.get() - ref.mo_energy).max(), 0, 5)

    def test_gks_no_ecp_plus_x2c(self):
        # X2C is the all-electron alternative to ECP -- combining them must
        # raise, not silently do the wrong thing.
        ecp_mol = gto.M(verbose=0, atom='I 0 0 0', basis='crenbl',
                        ecp='crenbl', spin=1)
        with self.assertRaises(NotImplementedError):
            gks.GKS(ecp_mol, xc='pbe0').x2c1e().run()


if __name__ == '__main__':
    print('Tests for GKS + spin-orbit X2C')
    unittest.main()
