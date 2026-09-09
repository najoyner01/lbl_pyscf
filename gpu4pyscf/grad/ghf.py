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

"""GHF nuclear gradient.

No CPU pyscf reference exists; derived from first principles in
docs/ghf-gradient-design.md.  No new CUDA kernel: all 2e terms route through
_jk_energies_per_atom (grad/tdrhf.py, kernel RYS_per_atom_jk_ip1_multidm).

SOC hcore gradient (d ECPso / dR) requires step-3 SO-ECP gradient integrals
which are not yet implemented.  The flag _with_soc_hcore_grad is checked at
run time; if True the function raises NotImplementedError.
"""

import cupy as cp
from gpu4pyscf.lib import utils, logger
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad.tdrhf import _jk_energies_per_atom
from gpu4pyscf.scf.jk import _VHFOpt

__all__ = ['Gradients', 'Grad']


def make_rdm1e(mo_energy, mo_coeff, mo_occ):
    """Energy-weighted density matrix in the 2*nao spin-orbital basis.

    Returns the real part of (Dme_αα + Dme_ββ), a (nao, nao) array, which is
    what the standard 1e gradient machinery needs (the imaginary part integrates
    to zero against real 1e integral derivatives at a variational minimum).
    """
    mo_energy = cp.asarray(mo_energy)
    mo_coeff = cp.asarray(mo_coeff)
    mo_occ = cp.asarray(mo_occ)

    nso = mo_coeff.shape[0]
    nao = nso // 2
    orbo = mo_coeff[:, mo_occ > 0]          # (2*nao, n_occ)
    eps_occ = mo_energy[mo_occ > 0]          # (n_occ,)

    # Dme = sum_i eps_i * C_i C_i†  (full 2nao x 2nao)
    # Real part of alpha+beta block only needed for 1e gradient
    weighted = orbo * eps_occ               # (2*nao, n_occ)
    dme_aa = (weighted[:nao] @ orbo[:nao].conj().T).real
    dme_bb = (weighted[nao:] @ orbo[nao:].conj().T).real
    return dme_aa + dme_bb


def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None,
              atmlst=None):
    """Electronic part of GHF nuclear gradients.

    Formula (design doc §1):
        dE/dR_A = Re[Tr(dh1e/dR · D)] - Re[Tr(dS/dR · Dme)] + dE_2e/dR_A
    """
    mf = mf_grad.base
    mol = mf_grad.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ    is None: mo_occ    = mf.mo_occ
    if mo_coeff  is None: mo_coeff  = mf.mo_coeff

    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)
    t0 = log.init_timer()

    mo_energy = cp.asarray(mo_energy)
    mo_occ    = cp.asarray(mo_occ)
    mo_coeff  = cp.asarray(mo_coeff)

    nso = mo_coeff.shape[0]
    nao = nso // 2

    # --- density matrix and energy-weighted DM (nao spin-free real) ---
    dm_ghf = mf.make_rdm1(mo_coeff, mo_occ)       # (2*nao, 2*nao), complex
    dme_sf = make_rdm1e(mo_energy, mo_coeff, mo_occ)  # (nao, nao), real

    dmaa = dm_ghf[:nao, :nao]   # complex
    dmbb = dm_ghf[nao:, nao:]   # complex
    dmab = dm_ghf[:nao, nao:]   # complex
    dmba = dm_ghf[nao:, :nao]   # complex = dmab†

    # Spin-free real DM for the 1e gradient
    dm_sf = (dmaa + dmbb).real              # (nao, nao)

    # --- 1e + overlap Pulay (reuse existing scalar machinery) ---
    if getattr(mf, 'with_soc', None):
        raise NotImplementedError(
            'GHF gradient with SOC hcore (d ECPso/dR) requires step-3 '
            'SO-ECP gradient integrals, not yet implemented.  '
            'Run without with_soc=True, or wait for the SO-ECP IP kernel.')

    e1_grad = mf_grad._hcore_energy(dm_sf, dme_sf)
    log.timer_debug1('GHF gradients of h1e', *t0)

    # --- 2e JK gradient (design doc §2.3) ---
    e2_grad = mf_grad.energy_ee(mol, dm_ghf)
    log.timer_debug1('GHF gradients of 2e part', *t0)

    de = e1_grad + e2_grad
    de += cp.asnumpy(mf_grad.extra_force())
    log.timer_debug1('GHF gradients of electronic part', *t0)
    return de


def _ghf_jk_energy(mf_grad, dm_ghf, verbose=None):
    """Assemble and call _jk_energies_per_atom for GHF density matrix.

    Implements the 7-pair table from design doc §2.3.  Kernel normalization
    (verified against the CUDA source rys_contract_jk_ip1.cu):

        _jk_energies_per_atom([[dm1, dm2]], j_factor=[f_j], k_factor=[f_k])
        → f_j * 0.5 * d/dR [Tr(dm1·J(dm2)) + Tr(dm2·J(dm1))]
          - f_k * 0.25 * d/dR [Tr(dm1·K(dm2)) + Tr(dm2·K(dm1))]

    For self-pairs (dm1=dm2=D), the symmetry means:
        → f_j * 0.5 * d/dR Tr(D·J(D))   (for J)
        → -f_k * 0.5 * d/dR Tr(D·K(D))  (for K, counts bra+ket)

    For non-self-pairs (dm1≠dm2), the K part is:
        → -f_k * 0.25 * d/dR [Tr(dm1·K(dm2)) + Tr(dm2·K(dm1))]
        and since the K bilinear form is symmetric: Tr[X·K(Y)] = Tr[Y·K(X)],
        → -f_k * 0.5 * d/dR Tr(dm1·K(dm2))

    GHF two-electron energy (design doc §2.1):
        E_2e = ½ Tr[D_tot·J(D_tot)]
             − ½ Tr[D_αα·K(D_αα)]  −  ½ Tr[D_ββ·K(D_ββ)]
             −    Tr[D_αβ·K(D_βα)]  (real part)

    D_αα Hermitian = X_aa + i Y_aa (X_aa sym, Y_aa antisym).
    Tr[D_αα·K(D_αα)] = Tr[X_aa·K(X_aa)] − Tr[Y_aa·K(Y_aa)]  (real, since
    Tr[X_aa·K(Y_aa)] = 0 for Hermitian D_αα).
    Needed K_αα gradient: −½ d/dR[Tr[X_aa·K(X_aa)] − Tr[Y_aa·K(Y_aa)]].

    Cross term: Re[Tr[D_αβ·K(D_βα)]] = Tr[A·K(Aᵀ)] + Tr[B·K(Bᵀ)]
    where A = Re(D_αβ), B = Im(D_αβ).  Needed: −d/dR[Tr[A·K(Aᵀ)] + Tr[B·K(Bᵀ)]].
    """
    mf = mf_grad.base
    mol = mf.mol
    nso = dm_ghf.shape[-1]
    nao = nso // 2

    dm_ghf = cp.asarray(dm_ghf)
    dmaa = dm_ghf[:nao, :nao]
    dmbb = dm_ghf[nao:, nao:]
    dmab = dm_ghf[:nao, nao:]
    dmba = dm_ghf[nao:, :nao]   # = dmab†

    # Spin-summed density for J
    D_tot = (dmaa + dmbb).real          # real, symmetric (real part only needed)

    # Blocks for K
    X_aa = dmaa.real    # sym
    Y_aa = dmaa.imag    # antisym
    X_bb = dmbb.real
    Y_bb = dmbb.imag
    A_ab = dmab.real    # Re(D_αβ)
    B_ab = dmab.imag    # Im(D_αβ)
    A_ba = dmba.real    # = A_ab.T
    # For the cross-term imag part: use [B_ab, B_ab.T] (equiv. to [B_ab, -Im(D_βα)])
    # since Im(D_βα) = −B_ab.T, so [B_ab, Im(D_βα)] = [B_ab, −B_ab.T]
    # and k_factor sign would flip; using [B_ab, B_ab.T] with k_factor=2 is correct:
    # kernel gives −2 · 0.5 · d/dR Tr[B_ab·K(B_ab.T)] = −d/dR Tr[B_ab·K(B_ab.T)] ✓
    B_ba_neg = B_ab.T   # = −Im(D_βα); pass as dm2 so that the kernel sees B.T

    # Empirically verified normalization of multidm kernel (sto-3g test):
    #   self-pair [D,D] with k_factor=f → −(f/2)·d/dR Tr[D·K(D)]
    #   cross-pair [D1,D2] with k_factor=f → −(f/4)·d/dR Tr[D1·K(D2)]
    # (multidm kernel gives half of single-DM kernel per pair for the same DM)
    #
    # Needed contributions and required k_factors:
    #   J:      +½ d/dR Tr[D_tot·J(D_tot)]                j_fac=1, k_fac=0
    #   K_aa_R: −½ d/dR Tr[X_aa·K(X_aa)]                 j_fac=0, k_fac=2
    #   K_aa_I: +½ d/dR Tr[Y_aa·K(Y_aa)]                 j_fac=0, k_fac=−2
    #   K_bb_R: −½ d/dR Tr[X_bb·K(X_bb)]                 j_fac=0, k_fac=2
    #   K_bb_I: +½ d/dR Tr[Y_bb·K(Y_bb)]                 j_fac=0, k_fac=−2
    #   cross_R:  −d/dR Tr[A_ab·K(A_ba)]                 j_fac=0, k_fac=4
    #   cross_I:  −d/dR Tr[B_ab·K(B_ba_neg)]             j_fac=0, k_fac=4

    dm_pairs = [
        [D_tot,  D_tot ],   # J term
        [X_aa,   X_aa  ],   # K_αα real
        [Y_aa,   Y_aa  ],   # K_αα imag (antisym, contributes with − sign via k_fac=−1)
        [X_bb,   X_bb  ],   # K_ββ real
        [Y_bb,   Y_bb  ],   # K_ββ imag
        [A_ab,   A_ba  ],   # cross real  (bra≠ket)
        [B_ab,   B_ba_neg], # cross imag  (bra≠ket; dm2 = B_ab.T = −Im(D_βα))
    ]
    j_factor = [1.,  0.,  0.,  0.,  0.,  0.,  0.]
    k_factor = [0.,  2., -2.,  2., -2.,  4.,  4.]

    # VALIDATION STATUS (see docs/ghf-gradient-design.md §4):
    #   tested to FD/machine precision:  J term, and the real diagonal K
    #     blocks X_aa / X_bb (k_factor=+2) -- covers every real, collinear
    #     GHF/GKS solution.
    #   NOT yet FD-validated:  the imaginary diagonal blocks Y_aa / Y_bb
    #     (k_factor=-2) and the D_alpha,beta cross terms (k_factor=+4).
    #     These are zero for a real block-diagonal solution, so no existing
    #     test exercises them.  They switch on for any genuinely complex or
    #     non-collinear density -- i.e. every SOC calculation.  Warn loudly
    #     rather than return a silently-unvalidated number.
    _imag_scale = float(cp.abs(Y_aa).max() + cp.abs(Y_bb).max())
    _cross_scale = float(cp.abs(A_ab).max() + cp.abs(B_ab).max())
    if _imag_scale > 1e-8 or _cross_scale > 1e-8:
        logger.warn(mf,
            'GHF gradient: density has significant imaginary-diagonal '
            '(|Y|~%.1e) and/or spin-off-diagonal (|D_ab|~%.1e) blocks. '
            'The k_factor=-2 / k_factor=+4 gradient paths for these are '
            'NOT finite-difference validated yet (docs/ghf-gradient-design.md '
            '§4). Do not trust this gradient for production SOC geometry '
            'optimization until the non-collinear FD test passes.',
            _imag_scale, _cross_scale)

    vhfopt = mf._opt_gpu.get(None)
    if vhfopt is None:
        vhfopt = mf._opt_gpu[None] = _VHFOpt(mol, mf.direct_scf_tol).build()

    return _jk_energies_per_atom(
        vhfopt, dm_pairs, j_factor, k_factor,
        omega=None, lr_factor=None, sr_factor=None,
        sum_results=True, verbose=verbose)


class Gradients(rhf_grad.GradientsBase):
    """GHF nuclear gradients.

    No CPU counterpart; validated via (a) RHF/UHF reduction and (b) finite
    difference (see docs/ghf-gradient-design.md §4).
    """

    to_cpu = utils.to_cpu
    to_gpu = utils.to_gpu
    device = utils.device

    grad_elec = grad_elec

    def make_rdm1e(self, mo_energy=None, mo_coeff=None, mo_occ=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff  is None: mo_coeff  = self.base.mo_coeff
        if mo_occ    is None: mo_occ    = self.base.mo_occ
        return make_rdm1e(mo_energy, mo_coeff, mo_occ)

    def energy_ee(self, mol, dm):
        return _ghf_jk_energy(self, dm)

    def jk_energy_per_atom(self, dm=None, j_factor=1, k_factor=1,
                           omega=None, hermi=0, verbose=None):
        if dm is None:
            dm = self.base.make_rdm1()
        return _ghf_jk_energy(self, dm, verbose=verbose)


Grad = Gradients
