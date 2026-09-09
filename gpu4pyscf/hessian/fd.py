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

"""Finite-difference nuclear Hessian for GHF / GKS.

An analytic 2-component / spin-orbit Hessian is a much larger job and is future
work; this is the interim path so that vibrational frequencies and
thermochemistry (ZPE, H, S, G) work for spin-orbit DFT, including with
``mf.with_soc = True`` and ``mf.PCM()`` / ``mf.SMD()``.

Cost: central differences over the 3N nuclear coordinates, i.e. ``6N``
re-converged SCF + analytic-gradient evaluations.  **Small systems only.**

``with_soc`` and the implicit-solvent model ride along automatically because
the driver reuses the method's own gradient scanner
(``mf.nuc_grad_method().as_scanner()``); for GKS the scanner is built from a
``grid_response=True`` gradient (otherwise the omitted grid-weight-derivative
term leaves a ~5e-5 gradient floor that shows up as noise in the Hessian).
"""

import numpy as np
from pyscf import lib
from gpu4pyscf.lib import logger

__all__ = ['finite_diff_hessian', 'harmonic_and_thermo', 'Hessian']

_CONV_TOL_MAX = 1e-11


def _is_gks(mf):
    from gpu4pyscf.dft.gks import GKS
    return isinstance(mf, GKS)


def _make_grad_scanner(mf):
    g = mf.nuc_grad_method()
    if _is_gks(mf):
        g.grid_response = True
    return g.as_scanner()


def finite_diff_hessian(mf, disp=1e-3, verbose=None, grad_scanner=None):
    """Central-difference nuclear Hessian.

    Returns
        numpy array ``(natm, natm, 3, 3)`` -- ``H[A, B, x, y] =
        d^2 E / dR_{A,x} dR_{B,y}`` -- matching the analytic-Hessian layout in
        ``gpu4pyscf/hessian/{rhf,uhf,rks,uks}.py`` (consumed directly by
        ``pyscf.hessian.thermo``).

    Args:
        mf : a converged GHF / GKS mean-field object (``with_soc`` and
            ``.PCM()`` / ``.SMD()`` are supported transparently).
        disp : displacement in Bohr (default 1e-3).
        grad_scanner : optional pre-built gradient scanner overriding the
            default ``mf.nuc_grad_method().as_scanner()`` (with
            ``grid_response=True`` for GKS).

    The FD gradient difference is ~ ``conv_tol / disp``, so a tight reference is
    required: ``mf.conv_tol <= 1e-11`` (raises otherwise).
    """
    log = logger.new_logger(mf, verbose)
    mol = mf.mol

    if getattr(mf, 'conv_tol', 1.0) > _CONV_TOL_MAX:
        raise ValueError(
            f'finite_diff_hessian requires mf.conv_tol <= {_CONV_TOL_MAX:g} '
            f'(got {mf.conv_tol:g}); the FD gradient difference is '
            f'~ conv_tol / disp.')
    if not getattr(mf, 'converged', False):
        log.warn('Reference SCF not converged; running mf.kernel() first.')
        mf.kernel()

    want_soc = bool(getattr(mol, 'has_ecp_soc', lambda: False)())

    scan = grad_scanner if grad_scanner is not None else _make_grad_scanner(mf)

    natm = mol.natm
    n3 = natm * 3
    coords0 = mol.atom_coords()   # Bohr
    H = np.empty((n3, n3))

    t0 = log.init_timer()
    for A in range(natm):
        for x in range(3):
            cpl = coords0.copy(); cpl[A, x] += disp
            cmi = coords0.copy(); cmi[A, x] -= disp
            _, g_p = scan(mol.set_geom_(cpl, unit='Bohr', inplace=False))
            _, g_m = scan(mol.set_geom_(cmi, unit='Bohr', inplace=False))
            col = (np.asarray(g_p) - np.asarray(g_m)).ravel() / (2.0 * disp)
            H[:, A * 3 + x] = col
            log.timer_debug1(f'FD Hessian column: atom {A}, xyz {x}', *t0)

    # d(grad_{B,y})/dR_{A,x} is stored at H[B*3+y, A*3+x]; symmetrize.
    H = 0.5 * (H + H.T)

    if want_soc:
        assert mol.has_ecp_soc(), \
            'SO-ECP was lost from mol during the FD Hessian loop'

    # (3N, 3N) -> (natm, natm, 3, 3) == (A, B, dR_A, dR_B).  Post-symmetrization
    # the tensor obeys H[A,B,x,y] == H[B,A,y,x], so the choice of transpose is
    # immaterial; use the layout the analytic Hessians emit.
    H4 = H.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)
    return np.ascontiguousarray(H4)


def harmonic_and_thermo(mf, T=298.15, P=101325, disp=1e-3, verbose=None,
                        project=True):
    """FD Hessian -> harmonic analysis -> thermochemistry (thin wrapper).

    Returns ``(freq_info, thermo_info)`` where ``freq_info`` is the dict from
    ``pyscf.hessian.thermo.harmonic_analysis`` and ``thermo_info`` the dict from
    ``pyscf.hessian.thermo.thermo`` (``G_tot``, ``H_tot``, ``S_tot``, ``ZPE``,
    ... as ``(value, unit)`` pairs).

    ``project=False`` keeps the translational/rotational modes in the harmonic
    analysis (useful as an FD sanity check: they should come out near zero).
    """
    from pyscf.hessian import thermo
    hess = finite_diff_hessian(mf, disp=disp, verbose=verbose)
    freq_info = thermo.harmonic_analysis(
        mf.mol, hess, exclude_trans=project, exclude_rot=project)
    thermo_info = thermo.thermo(mf, freq_info['freq_au'], T, P)
    return freq_info, thermo_info


class Hessian(lib.StreamObject):
    """Finite-difference nuclear Hessian for GHF / GKS.

    Drop-in for ``pyscf.hessian.thermo``: ``kernel()`` returns a
    ``(natm, natm, 3, 3)`` array.  This is **finite difference** -- O(6N)
    re-converged SCF + gradient evaluations -- and is only practical for small
    systems.  An analytic 2-component / SOC Hessian is future work.
    """

    from gpu4pyscf.lib.utils import to_gpu, device

    disp = 1e-3

    def __init__(self, scf_method):
        self.base = scf_method
        self.mol = scf_method.mol
        self.verbose = scf_method.verbose
        self.stdout = scf_method.stdout
        self.max_memory = self.mol.max_memory
        self.atmlst = None
        self.de = np.zeros((0, 0, 3, 3))   # (A, B, dR_A, dR_B)
        self._keys = set(self.__dict__.keys())

    def kernel(self, disp=None, verbose=None):
        if disp is None:
            disp = self.disp
        self.de = finite_diff_hessian(self.base, disp=disp,
                                      verbose=verbose or self.verbose)
        return self.de

    hess = kernel

    def harmonic_analysis(self, *args, **kwargs):
        from pyscf.hessian import thermo
        if self.de.shape[0] == 0:
            self.kernel()
        return thermo.harmonic_analysis(self.mol, self.de, *args, **kwargs)

    def thermo(self, T=298.15, P=101325):
        return harmonic_and_thermo(self.base, T=T, P=P, disp=self.disp)

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self.base.reset(mol)
        return self


# Inject to GHF (and thus GKS, which is-a GHF).  Mirrors the injections in
# gpu4pyscf/hessian/{rhf,uhf,rks,uks}.py.
from gpu4pyscf import scf  # noqa: E402
scf.ghf.GHF.Hessian = lib.class_as_method(Hessian)
