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

'''
Hirshfeld + CM5 charges.  There is no CPU PySCF implementation to diff
against, so the checks are:

  * Sigma q  ==  total molecular charge   (exact stockholder sum rule, to grid
    accuracy) -- neutral, cation, open-shell (UKS), and GKS+SO-ECP densities.
  * CM5 preserves that sum exactly (the pairwise term is antisymmetric).
  * water Hirshfeld / CM5 charges land on their well-known literature ranges
    (Hirshfeld famously small, ~ -0.3 on O; CM5 ~ 2x larger, designed to sit
    near "chemical" values).
  * ECP: the effective nuclear charge (mol.atom_charge) is used and is
    consistent with the ECP-built free-atom reference, so a small-core ECP
    molecule still satisfies the sum rule and gives a chemically sane polarity.
'''

import unittest
import numpy as np
from pyscf import gto
from gpu4pyscf import scf, dft
from gpu4pyscf.pop import hirshfeld


def setUpModule():
    global mol_h2o
    mol_h2o = gto.M(
        atom='''O  0.0000  0.0000  0.1173
                H  0.0000  0.7572 -0.4692
                H  0.0000 -0.7572 -0.4692''',
        basis='def2-svp', verbose=0, output='/dev/null')


def tearDownModule():
    global mol_h2o
    mol_h2o.stdout.close()
    del mol_h2o


class KnownValues(unittest.TestCase):
    def test_hirshfeld_sum_rule_neutral(self):
        mf = dft.RKS(mol_h2o, xc='pbe0').run()
        q = hirshfeld.hirshfeld_charges(mol_h2o, mf.make_rdm1()).get()
        self.assertAlmostEqual(q.sum(), 0.0, 3)

    def test_hirshfeld_water_literature_range(self):
        mf = dft.RKS(mol_h2o, xc='pbe0').run()
        q = hirshfeld.hirshfeld_charges(mol_h2o, mf.make_rdm1()).get()
        # Hirshfeld on water: O ~ -0.3, H ~ +0.15 (small charges by construction)
        self.assertTrue(-0.42 < q[0] < -0.20, q)
        self.assertTrue(0.10 < q[1] < 0.21, q)
        self.assertAlmostEqual(q[1], q[2], 4)                 # C2v symmetry
        self.assertAlmostEqual(q[0], -(q[1] + q[2]), 3)       # sum rule

    def test_hirshfeld_cation_sum_rule(self):
        m = gto.M(atom='''N 0 0 0; H 0 0 1.02; H 0.962 0 -0.34;
                          H -0.481 -0.833 -0.34; H -0.481 0.833 -0.34''',
                  basis='def2-svp', charge=1, verbose=0, output='/dev/null')
        mf = dft.RKS(m, xc='pbe0').run()
        q = hirshfeld.hirshfeld_charges(m, mf.make_rdm1()).get()
        self.assertAlmostEqual(q.sum(), 1.0, 3)
        m.stdout.close()

    def test_hirshfeld_uks_open_shell(self):
        m = gto.M(atom='N 0 0 0; O 0 0 1.15', basis='def2-svp', spin=1,
                  verbose=0, output='/dev/null')
        mf = dft.UKS(m, xc='pbe0').run()
        q = hirshfeld.hirshfeld_charges(m, mf.make_rdm1()).get()
        self.assertAlmostEqual(q.sum(), 0.0, 3)
        m.stdout.close()

    def test_hirshfeld_ecp_sum_rule_and_polarity(self):
        # small-core def2 ECP on iodine (28e core)
        m = gto.M(atom='I 0 0 0; H 0 0 1.609', basis='def2-svp',
                  ecp='def2-svp', verbose=0, output='/dev/null')
        mf = dft.RKS(m, xc='pbe0').run()
        q = hirshfeld.hirshfeld_charges(m, mf.make_rdm1()).get()
        self.assertAlmostEqual(q.sum(), 0.0, 3)
        # I (chi 2.66) slightly more electronegative than H (2.20): q_I <~ 0
        self.assertTrue(-0.30 < q[0] < 0.05, q)
        self.assertAlmostEqual(q[0], -q[1], 3)
        m.stdout.close()

    def test_hirshfeld_gks_soecp_sum_rule(self):
        # ties into the actinide SOC workflow: charges from a GKS + SO-ECP density
        m = gto.M(atom='I 0 0 0', basis='crenbl', ecp='crenbl', spin=1,
                  verbose=0, output='/dev/null')
        mf = dft.GKS(m, xc='pbe0')
        mf.with_soc = True
        mf.run()
        q = hirshfeld.hirshfeld_charges(m, mf.make_rdm1()).get()
        self.assertAlmostEqual(q.sum(), 0.0, 3)
        self.assertAlmostEqual(float(q[0]), 0.0, 3)           # lone atom
        m.stdout.close()

    def test_cm5_preserves_sum(self):
        mf = dft.RKS(mol_h2o, xc='pbe0').run()
        dm = mf.make_rdm1()
        q_hd = hirshfeld.hirshfeld_charges(mol_h2o, dm)
        q_cm5 = hirshfeld.cm5_charges(mol_h2o, dm, hirshfeld=q_hd).get()
        self.assertAlmostEqual(q_cm5.sum(), float(q_hd.sum().get()), 10)

    def test_cm5_water_literature_range(self):
        mf = dft.RKS(mol_h2o, xc='pbe0').run()
        q = hirshfeld.cm5_charges(mol_h2o, mf.make_rdm1()).get()
        # CM5 on water: O ~ -0.6, H ~ +0.3 (roughly 2x Hirshfeld, by design)
        self.assertTrue(-0.72 < q[0] < -0.50, q)
        self.assertTrue(0.25 < q[1] < 0.36, q)
        self.assertAlmostEqual(q[1], q[2], 4)

    def test_cm5_pair_T_special_and_general(self):
        # special pairs: antisymmetric, tabulated
        self.assertAlmostEqual(hirshfeld._cm5_pair_T(1, 6), 0.0502, 12)
        self.assertAlmostEqual(hirshfeld._cm5_pair_T(6, 1), -0.0502, 12)
        self.assertAlmostEqual(hirshfeld._cm5_pair_T(8, 1), -0.1671, 12)
        self.assertAlmostEqual(hirshfeld._cm5_pair_T(7, 8), -0.0346, 12)
        # general pair: D_Z difference
        self.assertAlmostEqual(hirshfeld._cm5_pair_T(6, 8),
                               hirshfeld._CM5_D[6] - hirshfeld._CM5_D[8], 12)
        self.assertAlmostEqual(hirshfeld._cm5_pair_T(9, 9), 0.0, 12)

    def test_grid_level_converged(self):
        mf = dft.RKS(mol_h2o, xc='pbe0').run()
        dm = mf.make_rdm1()
        q3 = hirshfeld.hirshfeld_charges(mol_h2o, dm, grid_level=3).get()
        q5 = hirshfeld.hirshfeld_charges(mol_h2o, dm, grid_level=5).get()
        self.assertLess(abs(q3 - q5).max(), 5e-3)


if __name__ == '__main__':
    print('Tests for Hirshfeld / CM5 population analysis')
    unittest.main()
