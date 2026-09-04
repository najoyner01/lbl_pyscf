# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
Short-range ECP under periodic boundary conditions (phase 1: scalar).

GPU port of ``pyscf.pbc.gto.ecp.ecp_int``.  Design + phasing:
``docs/ecp-pbc-design.md``.

Approach: a periodic ECP matrix is a lattice sum of *molecular* ECP integrals,
so we build a super-molecule (reference-cell bra shells + image ket shells for
every lattice vector L + image ECP projectors for every L_U) and reuse the
validated molecular kernel ``libgecp.ECP_cart`` over the cross-image task list,
then fold the ket-image axis with per-k phase factors ``exp(i k . L)``.

No new CUDA required for scalar Γ/k-points.
'''

import numpy as np
import cupy as cp
from pyscf import gto
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.gto.ecp import (
    libecp, sort_ecp_basis, make_full_tasks,
    _build_screen_data, _screen_block, _ecp_expcutoff,
    SCREEN_ECP, ECP_ATOM_ID,
)
from gpu4pyscf.gto.mole import group_basis

__all__ = ['ecp_int']


def _ecp_rcut(cell):
    '''Real-space cutoff for the ECP operator.

    Never shorter than ``cell.rcut`` (tuned for AO overlap); extended so the
    slowest-decaying ECP Gaussian is captured to ``cell.precision``.
    '''
    rcut = float(cell.rcut)
    if len(cell._ecpbas) == 0:
        return rcut
    env = cell._env
    a_min = np.inf
    for row in cell._ecpbas:
        p, n = row[gto.PTR_EXP], row[gto.NPRIM_OF]
        exps = env[p:p + n]
        exps = exps[exps > 0]
        if exps.size:
            a_min = min(a_min, float(exps.min()))
    if not np.isfinite(a_min) or a_min <= 0:
        return rcut
    prec = getattr(cell, 'precision', 1e-8) or 1e-8
    ecp_rcut = np.sqrt(max(-np.log(prec) / a_min, 0.0))
    return max(rcut, float(ecp_rcut))


def _ecp_supmol(Ls, sorted_mol, sorted_ecpbas, ref_ecp_loc):
    '''Build the supmol for the ECP lattice sum.

    Layout of ``_bas``:
        [bra shells (ref-cell, L=0)] | [image 0 ket shells] | ... | [image nL-1 ket shells]

    Layout of ``_atm``:
        [ref-cell atoms] | [image 0 atoms] | ... | [image nL-1 atoms]

    Layout of ``_ecpbas``:
        [image 0 ECP projectors] | ... | [image nL-1 ECP projectors]

    With this image-major ordering the ket AO block ``mat1[0:nao_ref, nao_ref:]``
    reshaped to ``[nao_ref, nL, nao_ref]`` maps directly onto lattice image m.

    Returns
        supmol       : ``gto.Mole`` with ``_atm/_bas/_ecpbas/_env`` set
        supmol_ao_loc: Cartesian AO offsets, length ``(1+nL)*nbas_ref + 1``
        nao_ref_cart : number of Cartesian AOs in the ref-cell basis
    '''
    natm = sorted_mol.natm
    nbas_ref = sorted_mol.nbas
    nL = len(Ls)

    ref_ao_loc = sorted_mol.ao_loc_nr(cart=True)
    nao_ref_cart = int(ref_ao_loc[-1])

    ref_atom_coords = sorted_mol.atom_coords()  # [natm, 3] Bohr

    # Image atom coordinates: L_m + ref coords for each image m
    img_coords = (Ls[:, None, :] + ref_atom_coords[None, :, :]).reshape(-1, 3)

    # Extend _env: keep original, append image atom coords
    n_orig_env = len(sorted_mol._env)
    _env = np.append(sorted_mol._env, img_coords.ravel())

    # _atm: [ref-cell] + [image 0..nL-1]
    _atm_ref = sorted_mol._atm.copy()
    _atm_imgs = np.repeat(sorted_mol._atm[None, :, :], nL, axis=0).reshape(-1, gto.ATM_SLOTS)
    # Fix PTR_COORD to point into the extended _env
    _atm_imgs[:, gto.PTR_COORD] = n_orig_env + np.arange(nL * natm) * 3
    _atm = np.vstack([_atm_ref, _atm_imgs])

    # _bas: [bra shells] + [image 0 shells] + ... + [image nL-1 shells]
    _bas_bra = sorted_mol._bas.copy()
    _bas_ket_list = []
    for m in range(nL):
        _bas_ket_m = sorted_mol._bas.copy()
        _bas_ket_m[:, gto.ATOM_OF] += natm + m * natm
        _bas_ket_list.append(_bas_ket_m)
    _bas = np.vstack([_bas_bra] + _bas_ket_list)

    # _ecpbas: [image 0 ECP projectors] + ... + [image nL-1 ECP projectors]
    # sorted_ecpbas has ATOM_OF ∈ [0, natm); shift to image-specific atom indices.
    _ecpbas_list = []
    for m in range(nL):
        _ecpbas_m = sorted_ecpbas.copy()
        _ecpbas_m[:, gto.ATOM_OF] = natm + m * natm + sorted_ecpbas[:, gto.ATOM_OF]
        _ecpbas_m[:, ECP_ATOM_ID] = _ecpbas_m[:, gto.ATOM_OF]
        _ecpbas_list.append(_ecpbas_m)
    _ecpbas_sup = np.concatenate(_ecpbas_list, axis=0)

    supmol = gto.Mole()
    supmol._atm = np.asarray(_atm, dtype=np.int32)
    supmol._bas = np.asarray(_bas, dtype=np.int32)
    supmol._ecpbas = _ecpbas_sup
    supmol._env = _env
    supmol.verbose = 0
    supmol.output = None

    # AO offsets for the supmol (Cartesian, image-major)
    #   bra:     ao_loc[k]              = ref_ao_loc[k]
    #   image m: ao_loc[nbas_ref*(m+1)+k] = nao_ref_cart*(m+1) + ref_ao_loc[k]
    supmol_ao_loc = np.empty((1 + nL) * nbas_ref + 1, dtype=np.int32)
    supmol_ao_loc[:nbas_ref + 1] = ref_ao_loc
    for m in range(nL):
        base = (m + 1) * nao_ref_cart
        supmol_ao_loc[nbas_ref * (m + 1):nbas_ref * (m + 2)] = base + ref_ao_loc[:-1]
    supmol_ao_loc[-1] = (1 + nL) * nao_ref_cart

    return supmol, supmol_ao_loc, nao_ref_cart


def ecp_int(cell, kpts=None):
    '''Periodic short-range ECP integrals (scalar).

    Args:
        cell : :class:`pyscf.pbc.gto.Cell` with ``cell._ecpbas``.
        kpts : (nkpts, 3) array or None (Gamma).

    Returns:
        CuPy array ``[nao, nao]`` (Gamma, real) or ``[nkpts, nao, nao]``
        (complex).  Matches ``pyscf.pbc.gto.ecp.ecp_int(cell, kpts)``.
    '''
    if len(cell._ecpbas) == 0:
        raise ValueError('cell has no ECP basis')
    if np.any(cell._ecpbas[:, gto.SO_TYPE_OF] == 1):
        raise NotImplementedError(
            'PBC SO-ECP not implemented (docs/ecp-pbc-design.md phase 3)')

    # --- lattice image list ---
    rcut = _ecp_rcut(cell)
    Ls = cell.get_lattice_Ls(rcut=rcut)
    nL = len(Ls)

    # --- sort ref-cell basis (same as molecular get_ecp) ---
    sorted_mol, coeff, uniq_l_ctr, l_ctr_counts = group_basis(cell)
    l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
    n_groups = len(uniq_l_ctr)
    nbas_ref = sorted_mol.nbas

    # --- sort ref-cell ECP basis ---
    # sort_ecp_basis filters out SO entries; handles -1 (ul) angular momentum.
    sorted_ecpbas, uniq_lecp, lecp_counts, ref_ecp_loc = sort_ecp_basis(
        sorted_mol._ecpbas)
    n_ref_ecp_groups = len(ref_ecp_loc) - 1  # number of (l, atom) groups
    n_ref_ecpbas = len(sorted_ecpbas)
    lecp_offsets = np.append(0, np.cumsum(lecp_counts))  # group offsets by l-type
    n_ecp_l_types = len(uniq_lecp)

    # --- build supmol ---
    supmol, supmol_ao_loc, nao_ref_cart = _ecp_supmol(
        Ls, sorted_mol, sorted_ecpbas, ref_ecp_loc)

    # supmol ECP groups: image m, ref group g → supmol group m*n_ref_ecp_groups + g
    # supmol_ecp_loc[m*n_ref_ecp_groups + g] = m*n_ref_ecpbas + ref_ecp_loc[g]
    supmol_ecp_loc = np.concatenate([
        m * n_ref_ecpbas + ref_ecp_loc[:-1] for m in range(nL)
    ] + [np.array([nL * n_ref_ecpbas])]).astype(np.int32)

    # --- screening data ---
    expcutoff = _ecp_expcutoff(cell)
    screen_data = _build_screen_data(supmol, supmol._ecpbas, supmol_ecp_loc)

    # --- build task list ---
    # tasks_pbc[i, j, k]: (ish, jsh, ksh) triples where
    #   ish ∈ bra group i (ref-cell shells)
    #   jsh ∈ ALL ket shells of type j across ALL nL images (independent L sum)
    #   ksh ∈ ALL ECP groups of l-type k across ALL nL images (independent L_U sum)
    tasks_pbc = {}
    for i in range(n_groups):
        for j in range(n_groups):
            # All ket shells of type j: for each image m, shells at
            #   [nbas_ref + m*nbas_ref + l_ctr_offsets[j], ...[j+1])
            jsh_range_full = np.concatenate([
                np.arange(nbas_ref + m * nbas_ref + l_ctr_offsets[j],
                           nbas_ref + m * nbas_ref + l_ctr_offsets[j + 1])
                for m in range(nL)])
            ish_range = np.arange(l_ctr_offsets[i], l_ctr_offsets[i + 1])
            for k in range(n_ecp_l_types):
                # All supmol ECP groups of l-type k: for each image m,
                #   ref groups in [lecp_offsets[k], lecp_offsets[k+1])
                #   → supmol groups m*n_ref_ecp_groups + ref_g
                ksh_ref = np.arange(lecp_offsets[k], lecp_offsets[k + 1])
                ksh_sup = np.concatenate([
                    m * n_ref_ecp_groups + ksh_ref for m in range(nL)])
                if SCREEN_ECP:
                    grid = _screen_block(ish_range, jsh_range_full, ksh_sup,
                                         screen_data, expcutoff, triangular=False)
                else:
                    iii, jjj, kkk = np.meshgrid(
                        ish_range, jsh_range_full, ksh_sup, indexing='ij')
                    grid = np.stack([iii.ravel().astype(np.int64),
                                      jjj.ravel().astype(np.int64),
                                      kkk.ravel().astype(np.int64)], axis=1)
                tasks_pbc[i, j, k] = grid

    # --- GPU arrays ---
    atm_gpu = cp.asarray(supmol._atm, dtype=np.int32)
    bas_gpu = cp.asarray(supmol._bas, dtype=np.int32)
    env_gpu = cp.asarray(supmol._env, dtype=np.float64)
    ecpbas_gpu = cp.asarray(supmol._ecpbas, dtype=np.int32)
    ecploc_gpu = cp.asarray(supmol_ecp_loc, dtype=np.int32)
    ao_loc_gpu = cp.asarray(supmol_ao_loc, dtype=np.int32)
    nao_sup = int(supmol_ao_loc[-1])

    mat1 = cp.zeros([nao_sup, nao_sup])

    for i in range(n_groups):
        li = int(uniq_l_ctr[i, 0])
        for j in range(n_groups):
            lj = int(uniq_l_ctr[j, 0])
            for k in range(n_ecp_l_types):
                lk = int(uniq_lecp[k])
                task = tasks_pbc[i, j, k]
                if len(task) == 0:
                    continue
                task_gpu = cp.asarray(task, dtype=np.int32, order='F')
                err = libecp.ECP_cart(
                    mat1.data.ptr, ao_loc_gpu.data.ptr, nao_sup,
                    task_gpu.data.ptr, len(task),
                    ecpbas_gpu.data.ptr, ecploc_gpu.data.ptr,
                    atm_gpu.data.ptr, bas_gpu.data.ptr, env_gpu.data.ptr,
                    li, lj, lk)
                if err != 0:
                    raise RuntimeError('PBC ECP CUDA kernel failed.')

    # --- extract bra×ket block and fold ---
    # mat1[0:nao_ref_cart, nao_ref_cart:] shape [nao_ref_cart, nL*nao_ref_cart]
    # image-major ordering → reshape to [nao_ref_cart, nL, nao_ref_cart]
    blk = mat1[0:nao_ref_cart, nao_ref_cart:].reshape(nao_ref_cart, nL, nao_ref_cart)

    # Cart→sph (or original AO basis) transform on bra and ket
    coeff_gpu = cp.asarray(coeff)  # [nao_cart_sorted, nao_orig]
    blk = contract('ip,iLj->pLj', coeff_gpu, blk)
    blk = contract('pLi,iq->pLq', blk, coeff_gpu)

    # Fold over lattice images
    if kpts is None:
        return blk.sum(axis=1).real

    kpts = np.asarray(kpts)
    single_kpt = (kpts.ndim == 1)
    kpts_2d = kpts.reshape(-1, 3)
    phase = cp.asarray(np.exp(1j * kpts_2d @ Ls.T))  # [nkpts, nL]
    result = contract('kL,pLq->kpq', phase, blk.astype(cp.complex128))
    return result[0] if single_kpt else result
