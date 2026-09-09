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

"""Finite-difference nuclear Hessian for GHF / GKS (gpu4pyscf/hessian/fd.py).

An analytic 2-component / SOC Hessian is future work; this is the FD interim
path, so vibrational frequencies + thermochemistry work for spin-orbit DFT
(incl. with_soc=True and .PCM()/.SMD()).  Gates:

  V1/V2  reduce to the analytic UHF / UKS Hessian for a real block-diagonal
         GHF / GKS solution (no SOC).
  V3     HI/CRENBL frequencies with with_soc=True (GHF and GKS): at the
         optimized geometry, 5 near-zero (trans+rot) modes + 1 stretch in
         2000-2600 cm^-1.
  V4     SOC shifts the HI stretch (small, nonzero).
  V5     thermochemistry end-to-end (also with .PCM()).
  V6     step-size convergence.
"""

import unittest
import numpy as np
import pytest

try:
    import cupy as cp
    from cupyx.scipy.linalg import block_diag
    _GPU = True
except ImportError:
    _GPU = False

pytestmark = pytest.mark.skipif(not _GPU, reason='no GPU')

SPIN_SAMPLES = 50
GRID_LEVEL = 3
H2O = 'O 0 0 0.12; H 0 0.75 -0.47; H 0 -0.75 -0.47'   # Bohr


def _relerr(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _to_np(x):
    return x.get() if hasattr(x, 'get') else np.asarray(x)


def _embed(mu):
    """block_diag(UHF/UKS spin blocks) -> a (2nao,2nao) GHF/GKS init DM."""
    dm = mu.make_rdm1()
    return block_diag(cp.asarray(dm[0]), cp.asarray(dm[1]))


class TestReduction(unittest.TestCase):
    """V1 / V2: FD GHF/GKS Hessian == analytic UHF/UKS Hessian (no SOC)."""

    @pytest.mark.slow
    def test_v1_ghf_fd_vs_uhf_analytic(self):
        from pyscf import gto
        from gpu4pyscf import scf
        from gpu4pyscf.hessian.fd import finite_diff_hessian
        mol = gto.M(atom=H2O, basis='sto-3g', unit='Bohr', verbose=0,
                    output='/dev/null')
        mu = scf.UHF(mol); mu.conv_tol = 1e-13; mu.kernel()
        self.assertTrue(mu.converged)
        H_ref = _to_np(mu.Hessian().kernel())

        mg = scf.GHF(mol); mg.conv_tol = 1e-12
        mg.kernel(_embed(mu))
        self.assertTrue(mg.converged)
        H_fd = finite_diff_hessian(mg, disp=1e-3)

        r = _relerr(H_fd, H_ref)
        print(f'\n[V1] ||H_fd - H_uhf|| / ||H_uhf|| = {r:.2e}')
        self.assertLess(r, 5e-4)

    @pytest.mark.slow
    def test_v2_gks_fd_vs_uks_analytic(self):
        from pyscf import gto
        from gpu4pyscf import dft
        from gpu4pyscf.hessian.fd import finite_diff_hessian
        mol = gto.M(atom=H2O, basis='sto-3g', unit='Bohr', verbose=0,
                    output='/dev/null')
        mu = dft.uks.UKS(mol, xc='pbe0'); mu.conv_tol = 1e-13
        mu.grids.level = GRID_LEVEL
        mu.kernel()
        self.assertTrue(mu.converged)
        h = mu.Hessian(); h.grid_response = True
        H_ref = _to_np(h.kernel())

        mg = dft.GKS(mol, xc='pbe0'); mg.conv_tol = 1e-12
        mg.spin_samples = SPIN_SAMPLES
        mg.grids.level = GRID_LEVEL
        mg.kernel(_embed(mu))
        self.assertTrue(mg.converged)
        H_fd = finite_diff_hessian(mg, disp=1e-3)

        r = _relerr(H_fd, H_ref)
        print(f'\n[V2] ||H_fd - H_uks|| / ||H_uks|| = {r:.2e}')
        self.assertLess(r, 5e-4)


def _hi(r=1.61):
    from pyscf import gto
    return gto.M(atom=f'H 0 0 0; I 0 0 {r}', basis='crenbl', ecp='crenbl',
                 spin=0, verbose=0, output='/dev/null')


def _opt(mf):
    from pyscf.geomopt.geometric_solver import optimize
    from gpu4pyscf.dft.gks import GKS
    if isinstance(mf, GKS):
        g = mf.nuc_grad_method(); g.grid_response = True
        return optimize(g.as_scanner(), maxsteps=20,
                        convergence_grms=5e-5, convergence_gmax=1e-4)
    return optimize(mf, maxsteps=20,
                    convergence_grms=5e-5, convergence_gmax=1e-4)


def _all_freq(mol, H):
    from pyscf.hessian import thermo
    fi = thermo.harmonic_analysis(mol, H, exclude_trans=False, exclude_rot=False)
    return np.sort(fi['freq_wavenumber'].real)


class TestSOCFrequencies(unittest.TestCase):
    """V3 / V4."""

    @pytest.mark.slow
    def test_v3_ghf_soc_hi_frequencies(self):
        from gpu4pyscf import scf
        from gpu4pyscf.hessian.fd import finite_diff_hessian
        mf = scf.GHF(_hi()); mf.with_soc = True; mf.conv_tol = 1e-12
        mf.kernel()
        mol_eq = _opt(mf)
        mf = scf.GHF(mol_eq); mf.with_soc = True; mf.conv_tol = 1e-12
        mf.kernel()
        H = finite_diff_hessian(mf, disp=1e-3)
        w = _all_freq(mol_eq, H)
        print(f'\n[V3 GHF+SOC] all 6 freqs = {np.round(w, 2)}')
        self.assertEqual(int(np.sum(np.abs(w) < 50.0)), 5)
        stretch = w[-1]
        self.assertTrue(2000.0 < stretch < 2600.0, f'stretch={stretch}')

    @pytest.mark.slow
    def test_v3_gks_soc_hi_frequencies(self):
        from gpu4pyscf import dft
        from gpu4pyscf.hessian.fd import finite_diff_hessian
        mf = dft.GKS(_hi(), xc='pbe0'); mf.with_soc = True
        mf.spin_samples = SPIN_SAMPLES; mf.conv_tol = 1e-12
        mf.kernel()
        mol_eq = _opt(mf)
        mf = dft.GKS(mol_eq, xc='pbe0'); mf.with_soc = True
        mf.spin_samples = SPIN_SAMPLES; mf.conv_tol = 1e-12
        mf.kernel()
        H = finite_diff_hessian(mf, disp=1e-3)
        w = _all_freq(mol_eq, H)
        print(f'\n[V3 GKS+SOC] all 6 freqs = {np.round(w, 2)}')
        self.assertEqual(int(np.sum(np.abs(w) < 50.0)), 5)
        stretch = w[-1]
        self.assertTrue(2000.0 < stretch < 2600.0, f'stretch={stretch}')

    @pytest.mark.slow
    def test_v4_soc_shifts_the_stretch(self):
        from gpu4pyscf import scf
        from gpu4pyscf.hessian.fd import finite_diff_hessian
        mol = _hi()   # common fixed geometry for both

        def stretch(with_soc):
            mf = scf.GHF(mol); mf.with_soc = with_soc; mf.conv_tol = 1e-12
            mf.kernel()
            H = finite_diff_hessian(mf, disp=1e-3)
            return _all_freq(mol, H)[-1]

        w_soc = stretch(True)
        w_nosoc = stretch(False)
        d = abs(w_soc - w_nosoc)
        print(f'\n[V4] HI GHF stretch  SOC={w_soc:.2f}  noSOC={w_nosoc:.2f}  '
              f'shift={d:.3f} cm^-1')
        self.assertGreater(d, 1e-2)     # nonzero
        self.assertLess(d, 100.0)       # physically small


class TestThermo(unittest.TestCase):
    """V5."""

    @pytest.mark.slow
    def test_v5_thermo_gks_soc_and_pcm(self):
        from gpu4pyscf import dft
        from gpu4pyscf.hessian.fd import harmonic_and_thermo

        mf = dft.GKS(_hi(), xc='pbe0'); mf.with_soc = True
        mf.spin_samples = SPIN_SAMPLES; mf.conv_tol = 1e-12
        mf.kernel()
        _, ti = harmonic_and_thermo(mf, T=298.15, P=101325, disp=1e-3)
        for k in ('G_tot', 'H_tot', 'S_tot', 'ZPE'):
            v = ti[k][0]
            print(f'[V5 GKS+SOC] {k} = {v}')
            self.assertTrue(np.isfinite(v))

        mfp = dft.GKS(_hi(), xc='pbe0').PCM()
        mfp.with_soc = True
        mfp.spin_samples = SPIN_SAMPLES; mfp.conv_tol = 1e-12
        mfp.kernel()
        _, tip = harmonic_and_thermo(mfp, T=298.15, P=101325, disp=1e-3)
        print(f'[V5 GKS+SOC+PCM] G_tot = {tip["G_tot"][0]}')
        self.assertTrue(np.isfinite(tip['G_tot'][0]))


class TestStepSize(unittest.TestCase):
    """V6."""

    @pytest.mark.slow
    def test_v6_step_size_convergence(self):
        from gpu4pyscf import scf
        from gpu4pyscf.hessian.fd import finite_diff_hessian
        mf = scf.GHF(_hi()); mf.with_soc = True; mf.conv_tol = 1e-12
        mf.kernel()
        H1 = finite_diff_hessian(mf, disp=1e-3)
        H2 = finite_diff_hessian(mf, disp=2e-3)
        H5 = finite_diff_hessian(mf, disp=5e-3)
        d12 = np.linalg.norm(H1 - H2)
        d15 = np.linalg.norm(H1 - H5)
        print(f'\n[V6] ||H(1e-3)-H(2e-3)|| = {d12:.2e}   '
              f'||H(1e-3)-H(5e-3)|| = {d15:.2e}')
        self.assertLess(d12, 1e-3)


class TestWiring(unittest.TestCase):

    def test_hessian_method_and_conv_tol_guard(self):
        from gpu4pyscf import scf, dft
        from gpu4pyscf.hessian.fd import Hessian
        mol = _hi()
        self.assertIsInstance(scf.GHF(mol).Hessian(), Hessian)
        self.assertIsInstance(dft.GKS(mol, xc='pbe0').Hessian(), Hessian)

        from gpu4pyscf.hessian.fd import finite_diff_hessian
        mf = scf.GHF(mol); mf.conv_tol = 1e-9
        with self.assertRaises(ValueError):
            finite_diff_hessian(mf)


if __name__ == '__main__':
    unittest.main()
