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

"""GKS (2-component / spin-orbit DFT) analytical nuclear gradients.

No CPU reference exists: pyscf ships grad/rks.py and grad/uks.py but no
grad/gks.py, and the multi-collinear XC-gradient exists nowhere in this tree.
Validation is (a) reduction to grad/uks.py in the collinear limit and
(b) finite difference of e_tot.  See grad/tests/test_gks_grad.py and
docs/ghf-gradient-design.md.

Structure:
  * hcore + overlap Pulay + SOC-hcore terms  -> inherited unchanged from
    grad/ghf.py (grad_elec -> _hcore_energy + _soc_hcore_grad).
  * 2e Coulomb (J)                            -> unchanged (grad/ghf.py).
  * exact exchange (K)                        -> grad/ghf.py:_ghf_jk_energy
    scaled by the functional hybrid coefficient (k_scale).
  * XC gradient (_gks_xc_grad)                -> mirrors grad/uks.py, with the
    2-component (rho, m_x, m_y, m_z) -> spin-block channel mapping taken from
    dft/numint2c.py.

Only global hybrids (omega == 0) are supported; range-separated hybrids and
MGGA raise NotImplementedError.  Multi-GPU is not used here (single device).
"""

import numpy as np
import cupy as cp
from gpu4pyscf.lib import logger
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.grad import rks as rks_grad
from gpu4pyscf.grad import uks as uks_grad
from gpu4pyscf.grad import ghf as ghf_grad
from gpu4pyscf.dft import numint, numint2c

__all__ = ['Gradients', 'Grad']


# --- (rho, m) -> nao x nao real symmetric channel density matrices -----------
# Derived from the multi-collinear density mapping in dft/numint2c.py
# (_contract_rho_m):
#     rho  <->  D_aa + D_bb
#     m_x  <->  D_ab + D_ba          (= 2 Re D_ab, symmetrised)
#     m_y  <->  i (D_ab - D_ba)      (= -2 Im D_ab, symmetrised)
#     m_z  <->  D_aa - D_bb
# The nuclear-gradient orbital-response term is
#     dE_xc/dR_A|_orb = - sum_c sum_{i in A, j} Q_c[i,j]  int wv_c  d_x chi_i  chi_j
# with Q_c = P_c + P_c^T = 2 Re(P_c) (each P_c is Hermitian).  Passing
# dm_channel_c = Q_c / 2 reproduces uks' "negate + *2" pipeline exactly, and in
# the collinear limit (D_ab = 0) this reduces term-by-term to grad/uks.py.
def _channel_dms(dm2c_sorted):
    nao = dm2c_sorted.shape[-1] // 2
    dmaa = dm2c_sorted[:nao, :nao]
    dmbb = dm2c_sorted[nao:, nao:]
    dmab = dm2c_sorted[:nao, nao:]
    a_ab = dmab.real
    b_ab = dmab.imag
    dm_r = (dmaa + dmbb).real                 # rho channel
    dm_mx = a_ab + a_ab.T                      # m_x channel
    dm_my = -(b_ab + b_ab.T)                   # m_y channel
    dm_mz = (dmaa - dmbb).real                 # m_z channel
    # order matches mcfun vxc = [wr, wmx, wmy, wmz]
    return cp.stack([dm_r, dm_mx, dm_my, dm_mz])


def _xc_grad_orbital(ni, opt, grids, xc_code, dm2c_sorted, dm_ch, xctype):
    """AO-derivative (orbital-response) part of dE_xc/dR, no grid response.

    Returns per-AO array vmat[nao, 3] (sorted AO order); the caller applies the
    "-2 * reduce_to_atom" that grad/uks.py uses.
    """
    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao
    ngrids = grids.coords.shape[0]

    if xctype == 'LDA':
        ao_deriv = 1
        rho_tm = cp.empty([4, ngrids])
    elif xctype == 'GGA':
        ao_deriv = 2
        rho_tm = cp.empty([4, 4, ngrids])
    else:
        raise NotImplementedError(f'GKS XC gradient for xctype={xctype}')

    # pass 1: full (rho, m) density on the grid
    p1 = 0
    for ao, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None):
        p0, p1 = p1, p1 + weight.size
        mask_2c = cp.concatenate([mask, mask + nao])
        dm_mask = dm2c_sorted[mask_2c[:, None], mask_2c]
        ao_val = ao[0] if xctype == 'LDA' else ao[:4]
        rho_tm[..., p0:p1] = numint2c.eval_rho(
            _sorted_mol, ao_val, dm_mask, xctype=xctype, hermi=1,
            with_lapl=False)

    eval_xc = ni.mcfun_eval_xc_adapter(xc_code)
    vxc = eval_xc(xc_code, rho_tm, deriv=1, xctype=xctype)[1]  # [4, nvar, ngrids]
    if xctype == 'LDA':
        vxc = vxc[:, 0]                        # -> [4, ngrids]
    weights = cp.asarray(grids.weights)

    # pass 2: contract d_x chi with wv_c chi against the channel DMs
    vmat = cp.zeros([nao, 3])
    p1 = 0
    for ao, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv, None):
        p0, p1 = p1, p1 + weight.size
        wv = weights[p0:p1] * vxc[..., p0:p1]
        dm_ch_mask = [numint.take_last2d(dm_ch[c], mask) for c in range(4)]
        if xctype == 'LDA':
            for c in range(4):
                aow = numint._scale_ao(ao[0], wv[c])
                vtmp = rks_grad._d1_dot_(ao[1:4], aow.T)
                vmat[mask] += cp.einsum('nij,ij->ni', vtmp, dm_ch_mask[c]).T
        else:  # GGA
            wv[:, 0] *= .5
            for c in range(4):
                vtmp = rks_grad._gga_grad_sum_(ao, wv[c])
                vmat[mask] += cp.einsum('nij,ij->ni', vtmp, dm_ch_mask[c]).T
    return vmat


def _xc_grad_full_response(mf, mol, grids0, xc_code, dm2c, xctype, verbose=None):
    """dE_xc/dR including the response of the integration grid.

    Mirrors grad/uks.py:get_exc_full_response with the 2c channel mapping.
    Returns a numpy [natm, 3] array with all sign/factor conventions applied.
    """
    from gpu4pyscf.hessian.rks import get_dweight_dA
    log = logger.new_logger(mol, verbose)
    natm = mol.natm

    grids = grids0.copy()
    grids.build(sort_grids_of_each_atom=True)
    ngrids = grids.coords.shape[0]

    ni = numint2c.NumInt2C()
    ni.collinear = 'mcol'
    ni.spin_samples = mf._numint.spin_samples
    ni.collinear_thrd = mf._numint.collinear_thrd
    ni.collinear_samples = mf._numint.collinear_samples
    ni.build(mol, grids.coords)
    opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao

    dm2c_sorted = opt.sort_orbitals(cp.asarray(dm2c), axis=[0, 1])
    dm_ch = _channel_dms(dm2c_sorted)

    if xctype == 'LDA':
        ao_deriv = 0
        rho_tm = cp.empty([4, ngrids])
    elif xctype == 'GGA':
        ao_deriv = 1
        rho_tm = cp.empty([4, 4, ngrids])
    else:
        raise NotImplementedError(f'GKS XC grid response for xctype={xctype}')

    g1 = 0
    for ao, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao, ao_deriv,
                                             None, strict_grid_order=True):
        g0, g1 = g1, g1 + weight.size
        mask_2c = cp.concatenate([mask, mask + nao])
        dm_mask = dm2c_sorted[mask_2c[:, None], mask_2c]
        ao_val = ao if xctype == 'LDA' else ao[:4]
        rho_tm[..., g0:g1] = numint2c.eval_rho(
            _sorted_mol, ao_val, dm_mask, xctype=xctype, hermi=1,
            with_lapl=False)
    assert g1 == ngrids

    eval_xc = ni.mcfun_eval_xc_adapter(xc_code)
    exc, vxc = eval_xc(xc_code, rho_tm, deriv=1, xctype=xctype)[:2]
    if xctype == 'LDA':
        vxc = vxc[:, 0]                        # -> [4, ngrids]
    weights = cp.asarray(grids.weights)
    wv = weights * vxc
    nonzero_mask = cp.abs(weights) > 1e-14

    rho0 = rho_tm[0] if xctype == 'LDA' else rho_tm[0, 0]
    dweightdA_right = rho0 * exc
    del rho_tm

    de_weight = cp.zeros((natm, 3))
    for g0 in range(0, ngrids, 4096):
        g1 = min(g0 + 4096, ngrids)
        dweight_dA = get_dweight_dA(_sorted_mol, grids, (g0, g1))
        de_weight += cp.einsum('Adg->Ad', dweight_dA * dweightdA_right[g0:g1])
    del dweight_dA, dweightdA_right, exc

    de_rho = cp.zeros((natm, 3))
    dvmat = cp.zeros((4, 3, nao, nao))
    g0 = 0
    for ao, mask, weight, _ in ni.block_loop(_sorted_mol, grids, nao,
                                             ao_deriv + 1,
                                             strict_grid_order=True):
        g1 = g0 + weight.shape[0]
        blk_mask = nonzero_mask[g0:g1]
        ao = ao[:, :, blk_mask]
        if ao.size == 0:
            g0 = g1
            continue
        i_atom = int(grids.atm_idx[g0])
        dm_ch_mask = [numint.take_last2d(dm_ch[c], mask) for c in range(4)]

        if xctype == 'LDA':
            split_wv = cp.ascontiguousarray(wv[:, g0:g1][:, blk_mask])
            for c in range(4):
                aow = numint._scale_ao(ao[0], split_wv[c])
                vtmp = rks_grad._d1_dot_(ao[1:4], aow.T)
                dvmat[c][:, mask[:, None], mask] += vtmp
                de_rho[i_atom] += cp.einsum('xij,ji->x', vtmp, dm_ch_mask[c]) * 2
        else:  # GGA
            split_wv = cp.ascontiguousarray(wv[:, :, g0:g1][:, :, blk_mask])
            split_wv[:, 0] *= .5
            for c in range(4):
                vtmp = rks_grad._gga_grad_sum_(ao, split_wv[c])
                dvmat[c][:, mask[:, None], mask] += vtmp
                de_rho[i_atom] += cp.einsum('xij,ji->x', vtmp, dm_ch_mask[c]) * 2
        g0 = g1
    assert g1 == ngrids

    excsum = (de_weight + de_rho).get()
    # - sign because nabla_X = -nabla_x ; contract_h1e_dm sums the 4 channels
    excsum -= rhf_grad.contract_h1e_dm(_sorted_mol, dvmat, dm_ch, hermi=1)
    log.timer_debug1('gks grad vxc full response')
    return excsum


def _gks_xc_grad(ks_grad, mol, dm2c, verbose=None):
    """XC contribution to the GKS nuclear gradient -> numpy [natm, 3]."""
    mf = ks_grad.base
    ni = mf._numint
    if ni.collinear[0].lower() != 'm':
        raise NotImplementedError('Only multi-collinear GKS gradient')
    if mf.do_nlc():
        raise NotImplementedError('NLC gradient for GKS is not implemented')

    xctype = ni._xc_type(mf.xc)
    if xctype == 'HF':
        return np.zeros((mol.natm, 3))
    if xctype not in ('LDA', 'GGA'):
        raise NotImplementedError(
            f'GKS nuclear gradient for xctype={xctype} '
            '(MGGA/NLC not implemented)')

    grids = ks_grad.grids if ks_grad.grids is not None else mf.grids
    if grids.coords is None:
        grids.build(sort_grids=True)

    if ks_grad.grid_response:
        return _xc_grad_full_response(mf, mol, grids, mf.xc, dm2c, xctype,
                                     verbose=verbose)

    if ni.gdftopt is None:
        ni.build(mol, grids.coords)
    opt = ni.gdftopt
    dm2c_sorted = opt.sort_orbitals(cp.asarray(dm2c), axis=[0, 1])
    dm_ch = _channel_dms(dm2c_sorted)
    vmat = _xc_grad_orbital(ni, opt, grids, mf.xc, dm2c_sorted, dm_ch, xctype)
    # grad/uks.py: get_exc returns -reduce_to_atom(vmat); energy_ee does *= 2
    return -2.0 * rks_grad._reduce_to_atom(opt._sorted_mol, vmat)


def energy_ee(ks_grad, mol=None, dm=None, verbose=None):
    """2e (J + scaled-K) + XC contributions to the GKS gradient."""
    if mol is None:
        mol = ks_grad.mol
    mf = ks_grad.base
    if dm is None:
        dm = mf.make_rdm1()
    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    if omega != 0:
        raise NotImplementedError(
            'Range-separated hybrid GKS nuclear gradient is not implemented')
    with_k = ni.libxc.is_hybrid_xc(mf.xc)
    k_scale = float(hyb) if with_k else 0.0

    e2 = ghf_grad._ghf_jk_energy(ks_grad, dm, k_scale=k_scale, verbose=verbose)
    e2 = cp.asnumpy(e2) + _gks_xc_grad(ks_grad, mol, dm, verbose=verbose)
    return e2


class Gradients(ghf_grad.Gradients):
    """GKS nuclear gradients (2-component / multi-collinear spin-orbit DFT)."""

    from gpu4pyscf.lib.utils import to_gpu, device  # noqa: F401

    grid_response = uks_grad.Gradients.grid_response
    _keys = ghf_grad.Gradients._keys | {'grids', 'nlcgrids', 'grid_response'}

    def __init__(self, mf):
        ghf_grad.Gradients.__init__(self, mf)
        self.grids = None
        self.nlcgrids = None

    energy_ee = energy_ee


Grad = Gradients
