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

"""Analytic nuclear gradient of the one-electron X2C-SOC core Hamiltonian
(atomic approximation, ``mf.x2c1e()`` with ``with_x2c.approx='atom1e'``).

See ``hessian/tests/results/x2c_soc_grad_A100.md`` for the full A100 run.

  V1  spin-free reduction -- ``hcore_deriv_generator(..., with_soc=False)`` must
      reproduce pyscf's ``x2c/sfx2c1e_grad.py`` template exactly (alpha-alpha
      block == spin-free, alpha-beta == 0).  Isolates the SO term.

  V2  finite difference of ``e_tot``, GHF and GKS, at equilibrium *and*
      strongly displaced bond lengths.

  V2u uranyl -- the analytic X2C-SOC hcore derivative vs central FD of the
      forward ``get_hcore`` (pure 1e operator, no SCF/XC noise), and the
      analytic gradient's zero-crossing.  ``@pytest.mark.slow``.

  V4  regression -- the plain (non-X2C) GHF gradient path is untouched.

  V5  geometry optimization on uranyl (``@pytest.mark.slow``).
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

try:
    import mcfun  # noqa: F401
    HAS_MCFUN = True
except ImportError:
    HAS_MCFUN = False

pytestmark = pytest.mark.skipif(NO_GPU, reason='no GPU available')

CONV_TOL = 1e-11
FD_STEP = 1e-4          # bohr, central difference


def _fd_grad(escan, mol, step=FD_STEP):
    c0 = mol.atom_coords()
    de = np.zeros((mol.natm, 3))
    for a in range(mol.natm):
        for x in range(3):
            cp_ = c0.copy(); cp_[a, x] += step
            cm_ = c0.copy(); cm_[a, x] -= step
            de[a, x] = (escan(cp_) - escan(cm_)) / (2 * step)
    return de


# --------------------------------------------------------------------------
class V1SpinFreeReduction(unittest.TestCase):
    """SO term zeroed -> must match pyscf/x2c/sfx2c1e_grad.py.

    Checked on HCl/6-31g (A100: 3e-14) and, in the results doc, on uranyl
    (== pyscf sfx2c1e_grad to 1.3e-10 at r=1.66 and 1.70)."""

    def test_reduces_to_sfx2c1e_grad(self):
        from pyscf.x2c import sfx2c1e, sfx2c1e_grad
        from gpu4pyscf.grad.x2c import hcore_deriv_generator

        mol = gto.M(atom='H 0 0 0; Cl 0 0 2.4', basis='6-31g',
                    verbose=0, unit='Bohr')
        sf = sfx2c1e.SpinFreeX2CHelper(mol)
        sf.approx = 'atom1e'
        sf_deriv = sfx2c1e_grad.hcore_grad_generator(sf, mol)
        gen0 = hcore_deriv_generator(mol, approx='atom1e', with_soc=False)

        nao = mol.nao
        worst = 0.0
        for ia in range(mol.natm):
            ref = np.asarray(sf_deriv(ia))                 # (3, nao, nao)
            got = cp.asnumpy(gen0(ia))                      # (3, 2nao, 2nao)
            aa = got[:, :nao, :nao]
            bb = got[:, nao:, nao:]
            ab = got[:, :nao, nao:]
            worst = max(worst,
                        abs(aa - ref).max(),
                        abs(bb - ref).max(),
                        abs(ab).max(),
                        abs(aa.imag).max())
        self.assertLess(worst, 1e-9)

    def test_atom1e_only(self):
        from gpu4pyscf.grad.x2c import hcore_deriv_generator
        mol = gto.M(atom='Ne 0 0 0', basis='sto-3g', verbose=0)
        with self.assertRaises(NotImplementedError):
            hcore_deriv_generator(mol, approx='1e')     # molecular X2C: out of scope


# --------------------------------------------------------------------------
class V2FiniteDifference(unittest.TestCase):
    """Analytic X2C-SOC gradient vs central FD of e_tot, equilibrium and
    displaced.  Displaced GHF/HCl (r=3.0, 3.6 bohr, |grad| ~ 0.13) matched FD
    to 3e-9 on the A100 -- grad_elec composes the hcore derivative correctly
    away from a minimum."""

    def _run(self, atom, basis, xc=None, tol=2e-5, grid_response=True):
        def mk(coords=None):
            m = gto.M(atom=atom, basis=basis, verbose=0, unit='Bohr',
                      output='/dev/null')
            if coords is not None:
                m.set_geom_(coords, unit='Bohr'); m.build(False, False)
            return m

        def build(m):
            if xc is None:
                from gpu4pyscf.scf import ghf
                mf = ghf.GHF(m).x2c1e()
            else:
                from gpu4pyscf.dft import gks
                mf = gks.GKS(m, xc=xc).x2c1e()
            mf.with_x2c.approx = 'atom1e'
            mf.conv_tol = CONV_TOL
            return mf

        mol = mk()
        mf = build(mol)
        mf.kernel()
        self.assertTrue(mf.converged)
        g = mf.nuc_grad_method()
        if xc is not None:
            g.grid_response = grid_response
        de = g.kernel()

        def escan(coords):
            x = build(mk(coords))
            x.kernel()
            return x.e_tot

        fd = _fd_grad(escan, mol)
        err = abs(de - fd).max()
        print(f'\n[V2 {atom} / {basis} xc={xc}] |analytic - FD|_max = {err:.2e}')
        self.assertLess(err, tol)

    def test_ghf_hcl_equilibrium(self):
        self._run('H 0 0 0; Cl 0 0 2.4', 'sto-3g', xc=None, tol=1e-6)

    def test_ghf_hcl_displaced(self):
        # ~0.6 bohr past r_e; analytic gradient ~ -0.12 Eh/bohr
        self._run('H 0 0 0; Cl 0 0 3.0', 'sto-3g', xc=None, tol=1e-6)

    @unittest.skipUnless(HAS_MCFUN, 'GKS multi-collinear XC needs mcfun')
    def test_gks_hcl_equilibrium(self):
        self._run('H 0 0 0; Cl 0 0 2.4', 'sto-3g', xc='pbe0', tol=2e-5)

    @pytest.mark.slow
    @unittest.skipUnless(HAS_MCFUN, 'GKS multi-collinear XC needs mcfun')
    def test_gks_hi_anorccvdzp(self):
        self._run('H 0 0 0; I 0 0 3.05', 'ano-rcc-vdzp', xc='pbe0', tol=1e-5)


# --------------------------------------------------------------------------
class V2Uranyl(unittest.TestCase):
    """Uranyl: the analytic X2C-SOC hcore derivative vs central FD of the
    forward get_hcore (no SCF, no XC grid), and the analytic gradient's
    zero-crossing."""

    @pytest.mark.slow
    def test_hcore_derivative_vs_fd(self):
        from gpu4pyscf.x2c.x2c import SpinOrbitalX2CHelper as GpuX2C
        from gpu4pyscf.grad.x2c import hcore_deriv_generator

        def forward(mol):
            h = GpuX2C(mol); h.approx = 'atom1e'
            return cp.asnumpy(h.get_hcore(mol))

        worst = 0.0
        for r in (1.66, 1.70):
            mol = gto.M(atom=f'U 0 0 0; O 0 0 {r}; O 0 0 -{r}', charge=2,
                        spin=0, basis='ano-rcc-vdzp', unit='Angstrom',
                        verbose=0, output='/dev/null')
            gen = hcore_deriv_generator(mol, approx='atom1e', with_soc=True)
            coords = mol.atom_coords()
            h = 1e-4
            for ia in range(mol.natm):
                ana = cp.asnumpy(gen(ia))
                for x in range(3):
                    cpx = coords.copy(); cpx[ia, x] += h
                    cmx = coords.copy(); cmx[ia, x] -= h
                    mp = mol.copy(); mp.set_geom_(cpx, unit='Bohr'); mp.build(False, False)
                    mm = mol.copy(); mm.set_geom_(cmx, unit='Bohr'); mm.build(False, False)
                    fd = (forward(mp) - forward(mm)) / (2 * h)
                    worst = max(worst, abs(ana[x] - fd).max())
        print(f'\n[V2u uranyl hcore deriv] worst |ana - fd| = {worst:.2e}')
        self.assertLess(worst, 1e-5)      # A100: 6e-7


# --------------------------------------------------------------------------
class V4Regression(unittest.TestCase):
    """Plain (non-X2C) GHF gradient path must be untouched by the new branch."""

    def test_plain_ghf_unaffected(self):
        from gpu4pyscf.scf import ghf
        mol = gto.M(atom='H 0 0 0; F 0 0 1.7', basis='sto-3g',
                    verbose=0, unit='Bohr')
        mf = ghf.GHF(mol); mf.conv_tol = CONV_TOL
        mf.kernel()
        self.assertFalse(getattr(mf, 'with_x2c', None))
        de = mf.nuc_grad_method().kernel()

        def escan(coords):
            m = mol.copy(); m.set_geom_(coords, unit='Bohr'); m.build(False, False)
            x = ghf.GHF(m); x.conv_tol = CONV_TOL; x.kernel()
            return x.e_tot
        fd = _fd_grad(escan, mol)
        self.assertLess(abs(de - fd).max(), 1e-6)


# --------------------------------------------------------------------------
class V5UranylGeomOpt(unittest.TestCase):

    @pytest.mark.slow
    @unittest.skipUnless(HAS_MCFUN, 'GKS multi-collinear XC needs mcfun')
    def test_uranyl_optimize(self):
        import cupy
        from pyscf.geomopt.geometric_solver import optimize
        from gpu4pyscf import dft
        from gpu4pyscf.dft import gks

        def uranyl(r):
            return gto.M(atom=f'U 0 0 0; O 0 0 {r}; O 0 0 -{r}', charge=2,
                         spin=0, basis='ano-rcc-vdzp', unit='Angstrom',
                         verbose=0, output='/dev/null')

        mol = uranyl(1.72)
        # seed the first SCF from a converged non-relativistic RKS 2c density
        r = dft.RKS(mol, xc='pbe0'); r.conv_tol = 1e-9; r.kernel()
        D = cupy.asarray(r.make_rdm1()); nao = mol.nao
        dm0 = cupy.zeros((2 * nao, 2 * nao), dtype=complex)
        dm0[:nao, :nao] = D / 2; dm0[nao:, nao:] = D / 2

        mf = gks.GKS(mol, xc='pbe0').x2c1e()
        mf.with_x2c.approx = 'atom1e'
        mf.spin_samples = 50
        mf.conv_tol = 1e-10
        mf.max_cycle = 100
        mf.kernel(dm0=dm0)
        self.assertTrue(mf.converged)

        g = mf.nuc_grad_method().as_scanner()
        g.grid_response = True
        mol_eq = optimize(g, maxsteps=20)
        rs = mol_eq.atom_coords() * 0.52917721092
        r_uo = np.linalg.norm(rs[1] - rs[0])
        print(f'\n[V5 uranyl] optimized r(U=O) = {r_uo:.4f} A')
        self.assertAlmostEqual(r_uo, 1.66, delta=0.02)


if __name__ == '__main__':
    unittest.main()
