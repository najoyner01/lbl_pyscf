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

"""GHF gradient validation.

Test (a):  §4.1 of docs/ghf-gradient-design.md
    Closed-shell H₂O, no SOC, block-diagonal real GHF must reduce to
    grad/uhf.py gradient to ≤ 1e-9.

Test (b):  §4.2 of docs/ghf-gradient-design.md
    Open-shell HI/crenbl (spin=1, no SOC) GHF where D_αα ≠ D_ββ.
    Analytic gradient must match finite difference (±1e-4 bohr central)
    to ≤ 1e-6.  This exercises K_αα and K_ββ diagonal blocks.

Test (c):  §4.3 of docs/ghf-gradient-design.md
    Spin-rotation invariance.  A global SU(2) rotation of the converged
    solution activates the imaginary-diagonal (k_factor=-2) and D_αβ cross
    (k_factor=+4) gradient paths that (a)/(b) leave at zero, and the analytic
    gradient must stay put.  This closes the non-collinear gap without needing
    SO-ECP gradient integrals.

    Still pending once SO-ECP IP integrals (step 3) land: a true with_soc=True
    finite-difference check.
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


H2O_ATOM = '''
O  0.000000  0.000000  0.117400
H -0.757000  0.000000 -0.469600
H  0.757000  0.000000 -0.469600
'''


def _ghf_mol(atom, basis, ecp=None, spin=0, charge=0):
    kw = dict(atom=atom, basis=basis, spin=spin, charge=charge,
              verbose=0, output='/dev/null', max_memory=32000)
    if ecp is not None:
        kw['ecp'] = ecp
    return gto.M(**kw)


def _run_ghf(mol, with_soc=False):
    from gpu4pyscf.scf import ghf as gpu_ghf
    mf = gpu_ghf.GHF(mol)
    mf.direct_scf_tol = 1e-14
    mf.conv_tol = 1e-12
    mf.with_soc = with_soc
    mf.kernel()
    assert mf.converged, 'GHF did not converge'
    return mf


def _run_uhf(mol):
    from gpu4pyscf import scf
    mf = scf.uhf.UHF(mol)
    mf.direct_scf_tol = 1e-14
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged, 'UHF did not converge'
    return mf


def _numerical_grad(mf_factory, mol, h=1e-4):
    """Central-difference nuclear gradient.

    mf_factory(mol) → converged SCF object.
    Displaces each atom ±h bohr along x, y, z.
    """
    coords0 = mol.atom_coords().copy()
    natm = mol.natm
    grad = np.zeros((natm, 3))
    for iatm in range(natm):
        for ix in range(3):
            def _energy(delta):
                new_coords = coords0.copy()
                new_coords[iatm, ix] += delta
                mol_new = mol.set_geom_(new_coords * 0.529177249,  # bohr→Å
                                        unit='Ang', inplace=False)
                mf = mf_factory(mol_new)
                mf.kernel()
                return mf.e_tot

            ep = _energy(+h)
            em = _energy(-h)
            grad[iatm, ix] = (ep - em) / (2 * h)
    return grad


class TestGHFGrad(unittest.TestCase):

    # -----------------------------------------------------------------------
    # Test (a): §4.1 — block-diagonal real GHF ≡ UHF for closed-shell H₂O
    # -----------------------------------------------------------------------

    def test_ghf_closed_shell_equals_uhf(self):
        """GHF (block-diagonal, same DM as UHF) gradient == UHF gradient to 1e-9.

        GHF and UHF SCF can converge to slightly different DMs (~1e-7) because
        GHF has a larger parameter space and a different DIIS trajectory.  To
        test the gradient *formula* cleanly, we embed the converged UHF MOs
        into a block-diagonal GHF MO tensor (alpha block upper-left, beta
        lower-right) and compute both gradients from the same underlying DM.
        With the same DM the agreement is ~1e-14 (machine precision); the test
        threshold is 1e-9 with generous margin.
        """
        mol = _ghf_mol(H2O_ATOM, 'cc-pvdz', spin=0)

        # Run UHF to get a reference converged solution
        mf_uhf = _run_uhf(mol)
        g_uhf = mf_uhf.nuc_grad_method().kernel()

        nao = mol.nao_nr()
        ca = mf_uhf.mo_coeff[0]   # (nao, nmo) cupy
        cb = mf_uhf.mo_coeff[1]
        ea = mf_uhf.mo_energy[0]  # (nmo,) cupy
        eb = mf_uhf.mo_energy[1]
        occ_a = mf_uhf.mo_occ[0]  # (nmo,) cupy
        occ_b = mf_uhf.mo_occ[1]

        # Embed UHF MOs into block-diagonal GHF spinor format:
        #   mo_coeff_ghf[:nao, :nmo] = ca,  mo_coeff_ghf[nao:, :nmo]  = 0
        #   mo_coeff_ghf[:nao, nmo:] = 0,   mo_coeff_ghf[nao:, nmo:]  = cb
        import cupy as cp
        nmo = ca.shape[1]
        mo_coeff_ghf = cp.zeros((2*nao, 2*nmo), dtype=cp.complex128)
        mo_coeff_ghf[:nao, :nmo] = ca.astype(cp.complex128)
        mo_coeff_ghf[nao:, nmo:] = cb.astype(cp.complex128)
        mo_energy_ghf = cp.concatenate([ea, eb])
        mo_occ_ghf    = cp.concatenate([occ_a, occ_b])

        # Build a GHF object that uses these MOs without running SCF
        from gpu4pyscf.scf import ghf as gpu_ghf
        mf_ghf = gpu_ghf.GHF(mol)
        mf_ghf.direct_scf_tol = 1e-14
        mf_ghf.mo_coeff    = mo_coeff_ghf
        mf_ghf.mo_energy   = mo_energy_ghf
        mf_ghf.mo_occ      = mo_occ_ghf
        mf_ghf.converged   = True
        mf_ghf._opt_gpu    = {}

        g_ghf = mf_ghf.nuc_grad_method().kernel()

        err = np.linalg.norm(g_ghf - g_uhf)
        print(f'\n[test_a] ||GHF_grad − UHF_grad|| = {err:.3e}')
        self.assertLess(err, 1e-9,
            f'GHF grad does not reduce to UHF grad: err={err:.3e}')

    def test_ghf_nuc_grad_method_wiring(self):
        """GHF.nuc_grad_method() returns a Gradients object."""
        mol = _ghf_mol(H2O_ATOM, 'cc-pvdz', spin=0)
        mf = _run_ghf(mol)
        from gpu4pyscf.grad.ghf import Gradients
        grad_obj = mf.nuc_grad_method()
        self.assertIsInstance(grad_obj, Gradients)

    # -----------------------------------------------------------------------
    # Test (c): §4.3 — spin-rotation invariance exercises the imaginary
    # diagonal (k_factor=-2) and D_αβ cross (k_factor=+4) code paths WITHOUT
    # needing SO-ECP gradient integrals.
    #
    # A global SU(2) spin rotation is an exact symmetry of the spin-free
    # Hamiltonian, so it leaves the nuclear gradient unchanged.  Rotating the
    # converged block-diagonal solution about the x-axis moves density weight
    # from X_aa/X_bb into Y_aa/Y_bb/A_ab/B_ab, activating exactly the paths
    # that tests (a)/(b) leave at zero.  If k_factor=-2 or k_factor=+4 were
    # wrong, the rotated analytic gradient would drift from the unrotated one.
    # -----------------------------------------------------------------------

    def test_ghf_spin_rotation_invariance(self):
        """Analytic gradient is invariant under a global spin rotation."""
        import cupy as cp
        mol = _ghf_mol('O 0 0 0; H 0 0 1.8', 'sto-3g', spin=1)
        mf = _run_ghf(mol, with_soc=False)
        g0 = mf.nuc_grad_method().kernel()

        nso = mf.mo_coeff.shape[0]
        nao = nso // 2
        C = cp.asarray(mf.mo_coeff).astype(cp.complex128)

        for theta in (0.3, 1.0, 2.0):
            c, s = np.cos(theta / 2), np.sin(theta / 2)
            Crot = cp.empty_like(C)
            Crot[:nao] = c * C[:nao] - 1j * s * C[nao:]      # R_x(theta)
            Crot[nao:] = -1j * s * C[:nao] + c * C[nao:]

            from gpu4pyscf.scf import ghf as gpu_ghf
            mfr = gpu_ghf.GHF(mol)
            mfr.direct_scf_tol = 1e-14
            mfr.mo_coeff  = Crot
            mfr.mo_energy = cp.asarray(mf.mo_energy)
            mfr.mo_occ    = cp.asarray(mf.mo_occ)
            mfr.converged = True
            mfr._opt_gpu  = {}

            gr = mfr.nuc_grad_method().kernel()
            err = np.linalg.norm(gr - g0)
            print(f'\n[test_c] theta={theta:.2f}  ||g_rot − g0|| = {err:.3e}')
            self.assertLess(err, 1e-8,
                f'spin-rotation broke the gradient at theta={theta}: '
                f'err={err:.3e} -- k_factor=-2 or k_factor=+4 path is wrong')

    # -----------------------------------------------------------------------
    # Test (b): §4.2 — open-shell GHF finite difference (D_αα ≠ D_ββ)
    # -----------------------------------------------------------------------
    # OH radical / sto-3g (spin=1, all-electron).  GHF without SOC converges
    # to a block-diagonal real solution with D_αβ = 0, but D_αα ≠ D_ββ, so
    # K_αα and K_ββ are exercised independently.  Error is dominated by the
    # O(h²)=1e-8 finite-difference truncation at h=1e-4.
    #
    # NOTE: the D_αβ ≠ 0 cross-term is tested once SO-ECP IP integrals land.

    @pytest.mark.slow
    def test_ghf_open_shell_finite_diff(self):
        """GHF analytic gradient vs. finite difference for OH (spin=1)."""
        mol = _ghf_mol('O 0 0 0; H 0 0 1.8', 'sto-3g', spin=1)

        mf_ref = _run_ghf(mol, with_soc=False)

        # Analytic gradient
        g_analytic = mf_ref.nuc_grad_method().kernel()

        # Numerical gradient — factory re-runs GHF with same settings
        def factory(mol_new):
            from gpu4pyscf.scf import ghf as gpu_ghf
            mf = gpu_ghf.GHF(mol_new)
            mf.direct_scf_tol = 1e-14
            mf.conv_tol = 1e-12
            mf.with_soc = False
            dm0 = mf_ref.make_rdm1()
            mf.kernel(dm0.real.get() if hasattr(dm0, 'get') else dm0.real)
            return mf

        g_numerical = _numerical_grad(factory, mol, h=1e-4)

        err = np.linalg.norm(g_analytic - g_numerical)
        print(f'\n[test_b] ||analytic − numerical||  = {err:.3e}')
        print(f'         analytic gradient:\n{g_analytic}')
        print(f'         numerical gradient:\n{g_numerical}')
        self.assertLess(err, 1e-6,
            f'GHF analytic gradient does not match finite diff: err={err:.3e}')


class TestGHFGradH2O(unittest.TestCase):
    """Finite-difference check on closed-shell H₂O (fast, no ECP)."""

    @pytest.mark.slow
    def test_ghf_closed_shell_finite_diff(self):
        """GHF analytic grad vs finite diff for H₂O — tests the J term and
        diagonal K blocks (D_αβ = 0 for a singlet block-diagonal solution)."""
        mol = _ghf_mol(H2O_ATOM, 'cc-pvdz', spin=0)
        mf_ref = _run_ghf(mol, with_soc=False)
        g_analytic = mf_ref.nuc_grad_method().kernel()

        def factory(mol_new):
            from gpu4pyscf.scf import ghf as gpu_ghf
            mf = gpu_ghf.GHF(mol_new)
            mf.direct_scf_tol = 1e-14
            mf.conv_tol = 1e-12
            mf.with_soc = False
            mf.kernel()
            return mf

        g_numerical = _numerical_grad(factory, mol, h=1e-4)

        err = np.linalg.norm(g_analytic - g_numerical)
        print(f'\n[test_h2o_fd] ||analytic − numerical|| = {err:.3e}')
        self.assertLess(err, 1e-6,
            f'H₂O GHF analytic gradient vs finite diff: err={err:.3e}')


if __name__ == '__main__':
    unittest.main()
