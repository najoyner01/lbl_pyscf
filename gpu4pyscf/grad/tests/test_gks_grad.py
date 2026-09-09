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

"""GKS (2-component / spin-orbit DFT) nuclear-gradient validation.

No CPU reference exists (pyscf has grad/rks.py and grad/uks.py but no
grad/gks.py, and the multi-collinear XC gradient exists nowhere here), so the
gates are:

V1 - collinear reduction.  A real, spin-block-diagonal GKS solution (UKS MOs
     embedded, collinear functional) -> grad/gks.py must match grad/uks.py.
     Isolates the (rho, m) -> spin-block XC-gradient bookkeeping and the
     hybrid-K scaling.
V2 - finite difference, no SOC, X in {svwn (LDA), pbe (GGA), pbe0 (hybrid)}.
     Analytic vs central FD of e_tot.  grid_response=True is the strict gate
     (grid-quadrature error cancels); grid_response=False is checked loosely.
V3 - finite difference, with SOC (dft.GKS(xc='pbe0'); with_soc=True) on
     HI/CRENBL (spin=0, genuinely non-collinear).  XC gradient + SOC-hcore
     gradient + scaled-K gradient together.

FD is kept cheap: small basis, spin_samples=50 set identically on the analytic
reference and every FD point, grids.level=3, 2-atom molecules.
"""

import numpy as np
import unittest
import pytest
from pyscf import gto

try:
    import cupy as cp
    NO_GPU = False
except ImportError:
    NO_GPU = True

pytestmark = pytest.mark.skipif(NO_GPU, reason='no GPU available')

SPIN_SAMPLES = 50
GRID_LEVEL = 3


def _gks(mol, xc, with_soc=False):
    from gpu4pyscf.dft.gks import GKS
    mf = GKS(mol, xc=xc)
    mf.spin_samples = SPIN_SAMPLES
    mf._numint.collinear = 'mcol'
    mf.conv_tol = 1e-12
    mf.grids.level = GRID_LEVEL
    mf.with_soc = with_soc
    return mf


def _fd_grad(mol, xc, dm0, with_soc=False, h=1e-4):
    coords0 = mol.atom_coords().copy()
    g = np.zeros((mol.natm, 3))
    for iatm in range(mol.natm):
        for ix in range(3):
            def _e(delta):
                c = coords0.copy()
                c[iatm, ix] += delta
                m = mol.set_geom_(c, unit='Bohr', inplace=False)
                mf = _gks(m, xc, with_soc)
                mf.kernel(dm0)
                assert mf.converged
                return mf.e_tot
            g[iatm, ix] = (_e(+h) - _e(-h)) / (2 * h)
    return g


class TestGKSGradReduction(unittest.TestCase):
    """V1: collinear GKS gradient == UKS gradient."""

    def _reduce(self, atom, basis, xc, spin):
        from gpu4pyscf import dft
        from gpu4pyscf.dft.gks import GKS
        mol = gto.M(atom=atom, basis=basis, spin=spin, verbose=0,
                    output='/dev/null')

        muks = dft.uks.UKS(mol, xc=xc)
        muks.conv_tol = 1e-11
        muks.max_cycle = 150
        muks.grids.level = GRID_LEVEL
        muks.kernel()
        self.assertTrue(muks.converged)
        g_uks = muks.nuc_grad_method()
        g_uks.grid_response = True
        guks = g_uks.kernel()

        nao = mol.nao
        ca = cp.asarray(muks.mo_coeff[0]).astype(cp.complex128)
        cb = cp.asarray(muks.mo_coeff[1]).astype(cp.complex128)
        nmo = ca.shape[1]
        C = cp.zeros((2 * nao, 2 * nmo), dtype=cp.complex128)
        C[:nao, :nmo] = ca
        C[nao:, nmo:] = cb
        E = cp.concatenate([cp.asarray(muks.mo_energy[0]),
                            cp.asarray(muks.mo_energy[1])])
        O = cp.concatenate([cp.asarray(muks.mo_occ[0]),
                            cp.asarray(muks.mo_occ[1])])

        mg = GKS(mol, xc=xc)
        mg.spin_samples = SPIN_SAMPLES
        mg._numint.collinear = 'mcol'
        mg.grids.level = GRID_LEVEL
        mg.mo_coeff = C
        mg.mo_energy = E
        mg.mo_occ = O
        mg.converged = True
        mg._opt_gpu = {}
        gg = mg.nuc_grad_method()
        gg.grid_response = True
        ggks = gg.kernel()

        err = np.linalg.norm(ggks - guks)
        print(f'\n[V1 {xc}] ||GKS - UKS|| = {err:.3e}')
        self.assertLess(err, 1e-7, f'{xc}: GKS grad != UKS grad ({err:.3e})')

    def test_reduce_lda(self):
        self._reduce('O 0 0 0; O 0 0 2.3', 'def2-svp', 'svwn', spin=2)

    def test_reduce_gga(self):
        self._reduce('O 0 0 0; O 0 0 2.3', 'def2-svp', 'pbe', spin=2)

    def test_reduce_hybrid(self):
        self._reduce('O 0 0 0; O 0 0 2.3', 'def2-svp', 'pbe0', spin=2)


class TestGKSGradFiniteDiff(unittest.TestCase):
    """V2: analytic GKS gradient vs finite difference, no SOC."""

    ATOM = 'O 0 0 0.117; H 0 0.755 -0.469; H 0 -0.755 -0.469'

    def _fd_check(self, xc):
        mol = gto.M(atom=self.ATOM, basis='def2-svp', unit='Bohr', verbose=0,
                    output='/dev/null')
        mf = _gks(mol, xc)
        mf.kernel()
        self.assertTrue(mf.converged)
        dm0 = mf.make_rdm1()
        dm0 = dm0.get() if hasattr(dm0, 'get') else dm0

        g = mf.nuc_grad_method()
        g.grid_response = True
        g_gr = g.kernel()
        g.grid_response = False
        g_nogr = g.kernel()

        gn = _fd_grad(mol, xc, dm0)
        e_gr = np.linalg.norm(g_gr - gn)
        e_nogr = np.linalg.norm(g_nogr - gn)
        print(f'\n[V2 {xc}] ||ana(grid_response) - FD|| = {e_gr:.3e}   '
              f'||ana(no grid_response) - FD|| = {e_nogr:.3e}')
        self.assertLess(e_gr, 1e-5,
                        f'{xc}: analytic (grid_response) != FD ({e_gr:.3e})')
        # grid_response=False is quadrature-limited (the weight-derivative term
        # is omitted); assert only that it is in the expected ~1e-4 ballpark.
        self.assertLess(e_nogr, 5e-4,
                        f'{xc}: analytic (no grid_response) wildly off '
                        f'({e_nogr:.3e})')

    @pytest.mark.slow
    def test_fd_lda(self):
        self._fd_check('svwn')

    @pytest.mark.slow
    def test_fd_gga(self):
        self._fd_check('pbe')

    @pytest.mark.slow
    def test_fd_hybrid(self):
        self._fd_check('pbe0')


class TestGKSGradSOC(unittest.TestCase):
    """V3: analytic GKS+SOC gradient vs finite difference."""

    @pytest.mark.slow
    def test_hi_soc_pbe0_finite_diff(self):
        mol = gto.M(atom='H 0 0 0; I 0 0 1.61', basis='crenbl', ecp='crenbl',
                    spin=0, verbose=0, output='/dev/null')
        self.assertTrue(mol.has_ecp_soc())
        mf = _gks(mol, 'pbe0', with_soc=True)
        mf.kernel()
        self.assertTrue(mf.converged)

        nao = mol.nao
        dmab = mf.make_rdm1()[:nao, nao:]
        self.assertGreater(float(cp.abs(dmab).max()), 1e-3)  # non-collinear

        g = mf.nuc_grad_method()
        g.grid_response = True
        g_analytic = g.kernel()

        dm0 = mf.make_rdm1()
        dm0 = dm0.get() if hasattr(dm0, 'get') else dm0
        g_numerical = _fd_grad(mol, 'pbe0', dm0, with_soc=True)

        err = np.linalg.norm(g_analytic - g_numerical)
        print(f'\n[V3 pbe0+SOC] ||analytic - numerical|| = {err:.3e}')
        print(f'  analytic:\n{g_analytic}')
        print(f'  numerical:\n{g_numerical}')
        self.assertLess(err, 1e-5,
                        f'GKS+SOC analytic gradient != FD ({err:.3e})')


class TestGKSGradWiring(unittest.TestCase):

    def test_nuc_grad_method(self):
        from gpu4pyscf.dft.gks import GKS
        from gpu4pyscf.grad.gks import Gradients
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0,
                    output='/dev/null')
        mf = GKS(mol, xc='pbe')
        self.assertIsInstance(mf.nuc_grad_method(), Gradients)


if __name__ == '__main__':
    print('GKS gradient tests')
    unittest.main()
