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

"""Spin-orbit geometry optimization end-to-end.

`pyscf.geomopt.geometric_solver.optimize` driven by the analytic SO-ECP
gradients (grad/ghf.py, grad/gks.py).  No new gpu4pyscf API -- the recommended
call patterns are:

    # GHF + SO-ECP (Hartree-Fock level)
    mf = scf.GHF(mol); mf.with_soc = True; mf.kernel()
    mol_eq = optimize(mf, maxsteps=20)

    # GKS + SO-ECP (DFT level) -- grid_response=True is strongly recommended
    # (without it the omitted grid-weight-derivative term leaves a ~5e-5
    #  gradient floor that can stall convergence near the minimum)
    mf = dft.GKS(mol, xc='pbe0'); mf.with_soc = True
    mf.spin_samples = 50            # cheaper mcfun spin-angular quadrature
    mf.kernel()
    g = mf.nuc_grad_method(); g.grid_response = True
    mol_eq = optimize(g.as_scanner(), maxsteps=20)

`optimize()` accepts an SCF object (uses its default gradient) or a
`lib.GradScanner` (from `g.as_scanner()`, carrying a pre-configured `g`); a bare
gpu4pyscf `Gradients` object is NOT accepted by pyscf's `optimize`.
"""

import numpy as np
import unittest
import pytest

try:
    import cupy  # noqa: F401
    import geometric  # noqa: F401
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

pytestmark = pytest.mark.skipif(not _DEPS_OK,
                                reason='needs GPU + geometric')

CONV = {'convergence_grms': 1e-4, 'convergence_gmax': 1.5e-4,
        'convergence_energy': 1e-7}
SPIN_SAMPLES = 50


def _hi(r=1.70):
    from pyscf import gto
    return gto.M(atom=f'H 0 0 0; I 0 0 {r}', basis='crenbl', ecp='crenbl',
                 spin=0, verbose=0, output='/dev/null')


def _bond(mol):
    c = mol.atom_coords(unit='Ang')
    return float(np.linalg.norm(c[0] - c[1]))


def _opt_ghf(with_soc):
    from gpu4pyscf import scf
    from pyscf.geomopt.geometric_solver import optimize
    mf = scf.GHF(_hi())
    mf.with_soc = with_soc
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    mol_eq = optimize(mf, maxsteps=20, **CONV)
    return mol_eq


class TestSOCGeomopt(unittest.TestCase):

    @pytest.mark.slow
    def test_t1_ghf_soc_optimize(self):
        """GHF + SO-ECP: optimize converges, sane r(H-I), tight fresh grad."""
        from gpu4pyscf import scf
        mol_eq = _opt_ghf(with_soc=True)
        r = _bond(mol_eq)
        print(f'\n[T1] GHF+SOC r(H-I) = {r:.5f} Ang')
        self.assertTrue(mol_eq.has_ecp_soc())      # SOC not lost mid-opt
        self.assertTrue(1.4 < r < 1.9)

        # T5: fresh analytic gradient at the converged geometry
        mf = scf.GHF(mol_eq)
        mf.with_soc = True
        mf.conv_tol = 1e-11
        mf.kernel()
        g = np.linalg.norm(mf.nuc_grad_method().kernel())
        print(f'[T5-GHF] ||grad(mol_eq)|| = {g:.2e}')
        self.assertLess(g, 1e-4)

    @pytest.mark.slow
    def test_t2_gks_soc_optimize(self):
        """GKS(pbe0) + SO-ECP with grid_response=True: converges."""
        from gpu4pyscf import dft
        from pyscf.geomopt.geometric_solver import optimize
        mf = dft.GKS(_hi(), xc='pbe0')
        mf.with_soc = True
        mf.spin_samples = SPIN_SAMPLES
        mf.conv_tol = 1e-11
        mf.kernel()
        self.assertTrue(mf.converged)
        g = mf.nuc_grad_method()
        g.grid_response = True
        mol_eq = optimize(g.as_scanner(), maxsteps=20, **CONV)
        r = _bond(mol_eq)
        print(f'\n[T2] GKS(pbe0)+SOC r(H-I) = {r:.5f} Ang')
        self.assertTrue(mol_eq.has_ecp_soc())
        self.assertTrue(1.4 < r < 1.9)

        # T5: fresh gradient at the converged geometry
        mf2 = dft.GKS(mol_eq, xc='pbe0')
        mf2.with_soc = True
        mf2.spin_samples = SPIN_SAMPLES
        mf2.conv_tol = 1e-11
        mf2.kernel()
        gg = mf2.nuc_grad_method()
        gg.grid_response = True
        gnorm = np.linalg.norm(gg.kernel())
        print(f'[T5-GKS] ||grad(mol_eq)|| = {gnorm:.2e}')
        self.assertLess(gnorm, 3e-4)

    @pytest.mark.slow
    def test_t3_soc_moves_the_minimum(self):
        """with_soc must survive the scanner: SOC/no-SOC minima must differ."""
        r_soc = _bond(_opt_ghf(with_soc=True))
        r_nosoc = _bond(_opt_ghf(with_soc=False))
        delta = abs(r_soc - r_nosoc)
        print(f'\n[T3] r_soc={r_soc:.5f}  r_nosoc={r_nosoc:.5f}  '
              f'delta={delta:.2e} Ang')
        self.assertGreater(delta, 1e-3)

    @pytest.mark.slow
    def test_t4_nonsoc_gks_optimize_regression(self):
        """A plain (no SOC) mcol-GKS optimize still converges."""
        from pyscf import gto
        from gpu4pyscf import dft
        from pyscf.geomopt.geometric_solver import optimize
        mol = gto.M(atom='H 0 0 0; F 0 0 0.95', basis='def2-svp', verbose=0,
                    output='/dev/null')
        mf = dft.GKS(mol, xc='pbe0')
        mf.spin_samples = SPIN_SAMPLES
        mf.conv_tol = 1e-11
        mf.kernel()
        self.assertTrue(mf.converged)
        g = mf.nuc_grad_method()
        g.grid_response = True
        mol_eq = optimize(g.as_scanner(), maxsteps=20,
                          convergence_grms=2e-4, convergence_gmax=3e-4,
                          convergence_energy=1e-7)
        r = _bond(mol_eq)
        print(f'\n[T4] non-SOC GKS(pbe0) r(H-F) = {r:.5f} Ang')
        self.assertTrue(0.8 < r < 1.05)

        mf2 = dft.GKS(mol_eq, xc='pbe0')
        mf2.spin_samples = SPIN_SAMPLES
        mf2.conv_tol = 1e-11
        mf2.kernel()
        gg = mf2.nuc_grad_method()
        gg.grid_response = True
        self.assertLess(np.linalg.norm(gg.kernel()), 1e-4)


if __name__ == '__main__':
    print('SOC geometry optimization tests')
    unittest.main()
