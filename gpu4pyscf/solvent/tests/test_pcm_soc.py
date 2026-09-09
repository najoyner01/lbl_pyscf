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

"""Implicit solvent (PCM / SMD) for GHF and GKS, including with_soc=True.

The reaction field is spin-independent: it couples only to the number density
Re(D_aa + D_bb) and the nuclear charges.  gpu4pyscf reduces the 2-component
spinor DM to that form at the solvent boundary (solvent/_attach_solvent.py:
_spin_sum_dm) and adds the returned potential block-diagonally to the
2-component Fock.

No CPU reference for GHF+PCM exists; the gates are (a) reduction to
scf.UHF(mol).PCM() / dft.uks.UKS(mol).PCM() for a real block-diagonal
solution, and (b) finite difference of e_tot.
"""

import unittest
import numpy as np
import pytest

try:
    import cupy as cp
    _GPU = True
except ImportError:
    _GPU = False

pytestmark = pytest.mark.skipif(not _GPU, reason='no GPU')

SPIN_SAMPLES = 50
GRID_LEVEL = 3


def _hi(r=1.61):
    from pyscf import gto
    return gto.M(atom=f'H 0 0 0; I 0 0 {r}', basis='crenbl', ecp='crenbl',
                 spin=0, verbose=0, output='/dev/null')


def _hf(r=1.75):
    from pyscf import gto
    return gto.M(atom=f'H 0 0 0; F 0 0 {r}', basis='def2-svp', unit='Bohr',
                 verbose=0, output='/dev/null')


def _fd_bond_grad(make_mf, mol, dm0, h=1e-4):
    c0 = mol.atom_coords().copy()
    g = np.zeros((mol.natm, 3))
    for A in range(mol.natm):
        for x in range(3):
            def e(d):
                c = c0.copy(); c[A, x] += d
                m = mol.set_geom_(c, unit='Bohr', inplace=False)
                mf = make_mf(m); mf.kernel(dm0)
                assert mf.converged
                return mf.e_tot
            g[A, x] = (e(h) - e(-h)) / (2 * h)
    return g


def _embed_uhf_into_2c(mu, mf2c):
    """Copy a converged UHF/UKS solution into a block-diagonal 2-component
    (GHF/GKS) mean-field object without re-running SCF."""
    nao = mu.mol.nao
    ca = cp.asarray(mu.mo_coeff[0]).astype(cp.complex128)
    cb = cp.asarray(mu.mo_coeff[1]).astype(cp.complex128)
    nmo = ca.shape[1]
    C = cp.zeros((2 * nao, 2 * nmo), dtype=cp.complex128)
    C[:nao, :nmo] = ca
    C[nao:, nmo:] = cb
    mf2c.mo_coeff = C
    mf2c.mo_energy = cp.concatenate([cp.asarray(mu.mo_energy[0]),
                                     cp.asarray(mu.mo_energy[1])])
    mf2c.mo_occ = cp.concatenate([cp.asarray(mu.mo_occ[0]),
                                  cp.asarray(mu.mo_occ[1])])
    mf2c.converged = True
    mf2c._opt_gpu = {}
    mf2c.with_solvent.eps = mu.with_solvent.eps
    return mf2c


class TestSolventReduction(unittest.TestCase):
    """V1: real block-diagonal GHF/GKS + PCM reduces to UHF/UKS + PCM."""

    def test_ghf_pcm_reduces_to_uhf(self):
        from pyscf import gto
        from gpu4pyscf import scf
        mol = gto.M(atom='O 0 0 0; O 0 0 2.3', basis='def2-svp', spin=2,
                    unit='Bohr', verbose=0, output='/dev/null')
        mu = scf.UHF(mol).PCM()
        mu.conv_tol = 1e-12
        mu.max_cycle = 100
        mu.kernel()
        self.assertTrue(mu.converged)
        g_u = mu.nuc_grad_method().kernel()

        mg = _embed_uhf_into_2c(mu, scf.GHF(mol).PCM())
        dE = abs(mg.energy_tot() - mu.e_tot)
        dG = np.linalg.norm(mg.nuc_grad_method().kernel() - g_u)
        print(f'\n[V1 GHF+PCM] dE={dE:.2e}  ||dGrad||={dG:.2e}')
        self.assertLess(dE, 1e-8)
        self.assertLess(dG, 1e-7)

    def test_gks_pcm_reduces_to_uks(self):
        from pyscf import gto
        from gpu4pyscf import dft
        mol = gto.M(atom='O 0 0 0; O 0 0 2.3', basis='def2-svp', spin=2,
                    unit='Bohr', verbose=0, output='/dev/null')
        mu = dft.uks.UKS(mol, xc='pbe0').PCM()
        mu.conv_tol = 1e-12
        mu.max_cycle = 100
        mu.grids.level = GRID_LEVEL
        mu.kernel()
        self.assertTrue(mu.converged)
        gu = mu.nuc_grad_method()
        gu.grid_response = True
        g_u = gu.kernel()

        mg = dft.GKS(mol, xc='pbe0').PCM()
        mg.spin_samples = SPIN_SAMPLES
        mg._numint.collinear = 'mcol'
        mg.grids.level = GRID_LEVEL
        mg = _embed_uhf_into_2c(mu, mg)
        dE = abs(mg.energy_tot() - mu.e_tot)
        gg = mg.nuc_grad_method()
        gg.grid_response = True
        dG = np.linalg.norm(gg.kernel() - g_u)
        print(f'\n[V1 GKS+PCM] dE={dE:.2e}  ||dGrad||={dG:.2e}')
        self.assertLess(dE, 1e-8)
        self.assertLess(dG, 1e-7)


class TestSolventFiniteDiff(unittest.TestCase):
    """V2 / V3 / V4: analytic solvated gradient vs central finite difference."""

    @pytest.mark.slow
    def test_v2_ghf_pcm_fd(self):
        from gpu4pyscf import scf
        mol = _hf()

        def make(m):
            mf = scf.GHF(m).PCM()
            mf.conv_tol = 1e-12
            return mf
        mf = make(mol); mf.kernel()
        self.assertTrue(mf.converged)
        ga = mf.nuc_grad_method().kernel()
        dm0 = mf.make_rdm1()
        dm0 = dm0.get() if hasattr(dm0, 'get') else dm0
        gn = _fd_bond_grad(make, mol, dm0)
        err = np.linalg.norm(ga - gn)
        print(f'\n[V2 GHF+PCM] ||ana-FD|| = {err:.2e}')
        self.assertLess(err, 1e-5)

    @pytest.mark.slow
    def test_v2_gks_pcm_fd(self):
        from gpu4pyscf import dft
        mol = _hf()

        def make(m):
            mf = dft.GKS(m, xc='pbe0').PCM()
            mf.spin_samples = SPIN_SAMPLES
            mf.conv_tol = 1e-12
            return mf
        mf = make(mol); mf.kernel()
        self.assertTrue(mf.converged)
        g = mf.nuc_grad_method(); g.grid_response = True
        ga = g.kernel()
        dm0 = mf.make_rdm1()
        dm0 = dm0.get() if hasattr(dm0, 'get') else dm0
        gn = _fd_bond_grad(make, mol, dm0)
        err = np.linalg.norm(ga - gn)
        print(f'\n[V2 GKS+PCM] ||ana-FD|| = {err:.2e}')
        self.assertLess(err, 1e-5)

    @pytest.mark.slow
    def test_v3_gks_pcm_soc_fd(self):
        """Compose target: SOC hcore + XC + scaled-K + PCM, all in the grad."""
        from gpu4pyscf import dft
        mol = _hi()
        self.assertTrue(mol.has_ecp_soc())

        def make(m):
            mf = dft.GKS(m, xc='pbe0').PCM()
            mf.with_soc = True
            mf.spin_samples = SPIN_SAMPLES
            mf.conv_tol = 1e-12
            return mf
        mf = make(mol); mf.kernel()
        self.assertTrue(mf.converged)
        nao = mol.nao
        self.assertGreater(float(cp.abs(mf.make_rdm1()[:nao, nao:]).max()), 1e-3)
        g = mf.nuc_grad_method(); g.grid_response = True
        ga = g.kernel()
        dm0 = mf.make_rdm1()
        dm0 = dm0.get() if hasattr(dm0, 'get') else dm0
        gn = _fd_bond_grad(make, mol, dm0)
        err = np.linalg.norm(ga - gn)
        print(f'\n[V3 GKS+PCM+SOC] ||ana-FD|| = {err:.2e}')
        self.assertLess(err, 1e-5)

    @pytest.mark.slow
    def test_v4_ghf_smd_fd(self):
        from gpu4pyscf import scf
        mol = _hf()

        def make(m):
            mf = scf.GHF(m).SMD()
            mf.with_solvent.solvent = 'water'
            mf.conv_tol = 1e-12
            return mf
        mf = make(mol); mf.kernel()
        self.assertTrue(mf.converged)
        ga = mf.nuc_grad_method().kernel()
        dm0 = mf.make_rdm1()
        dm0 = dm0.get() if hasattr(dm0, 'get') else dm0
        gn = _fd_bond_grad(make, mol, dm0)
        err = np.linalg.norm(ga - gn)
        print(f'\n[V4 GHF+SMD] ||ana-FD|| = {err:.2e}')
        self.assertLess(err, 1e-5)

    @pytest.mark.slow
    def test_v4_gks_smd_soc_fd(self):
        from gpu4pyscf import dft
        mol = _hi()

        def make(m):
            mf = dft.GKS(m, xc='pbe0').SMD()
            mf.with_solvent.solvent = 'water'
            mf.with_soc = True
            mf.spin_samples = SPIN_SAMPLES
            mf.conv_tol = 1e-12
            return mf
        mf = make(mol); mf.kernel()
        self.assertTrue(mf.converged)
        g = mf.nuc_grad_method(); g.grid_response = True
        ga = g.kernel()
        dm0 = mf.make_rdm1()
        dm0 = dm0.get() if hasattr(dm0, 'get') else dm0
        gn = _fd_bond_grad(make, mol, dm0)
        err = np.linalg.norm(ga - gn)
        print(f'\n[V4 GKS+SMD+SOC] ||ana-FD|| = {err:.2e}')
        self.assertLess(err, 1e-5)


class TestSolventSOCGeomopt(unittest.TestCase):
    """V5: end-to-end solvated spin-orbit geometry optimization."""

    @pytest.mark.slow
    def test_v5_gks_pcm_soc_optimize(self):
        from gpu4pyscf import dft
        from pyscf.geomopt.geometric_solver import optimize
        mf = dft.GKS(_hi(1.70), xc='pbe0').PCM()
        mf.with_soc = True
        mf.spin_samples = SPIN_SAMPLES
        mf.conv_tol = 1e-11
        mf.kernel()
        self.assertTrue(mf.converged)
        g = mf.nuc_grad_method(); g.grid_response = True
        mol_eq = optimize(g.as_scanner(), maxsteps=20,
                          convergence_grms=1e-4, convergence_gmax=1.5e-4,
                          convergence_energy=1e-7)
        c = mol_eq.atom_coords(unit='Ang')
        r = float(np.linalg.norm(c[0] - c[1]))
        print(f'\n[V5] GKS+PCM+SOC r(H-I) = {r:.5f} Ang')
        self.assertTrue(mol_eq.has_ecp_soc())
        self.assertTrue(1.4 < r < 1.9)

        mf2 = dft.GKS(mol_eq, xc='pbe0').PCM()
        mf2.with_soc = True
        mf2.spin_samples = SPIN_SAMPLES
        mf2.conv_tol = 1e-11
        mf2.kernel()
        g2 = mf2.nuc_grad_method(); g2.grid_response = True
        self.assertLess(np.linalg.norm(g2.kernel()), 3e-4)


class TestSolventSanity(unittest.TestCase):
    """V6: dG_solv sign / magnitude."""

    def test_v6_dg_solv_negative(self):
        from gpu4pyscf import scf
        mol = _hi()
        gas = scf.GHF(mol); gas.with_soc = True; gas.conv_tol = 1e-12
        gas.kernel()
        sol = scf.GHF(mol).PCM(); sol.with_soc = True; sol.conv_tol = 1e-12
        sol.kernel()
        dG = sol.e_tot - gas.e_tot
        print(f'\n[V6] dG_solv(HI) = {dG:.6f} Eh = {dG * 627.509:.3f} kcal/mol')
        self.assertLess(dG, 0.0)
        self.assertGreater(dG, -0.05)   # physically sized for a small molecule


if __name__ == '__main__':
    unittest.main()
