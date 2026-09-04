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
Short-range ECP under periodic boundary conditions.

GPU port of ``pyscf.pbc.gto.ecp.ecp_int``.  Design + phasing:
``docs/ecp-pbc-design.md``.

Approach (phase 1, scalar, Gamma + k-points): a periodic ECP matrix is a
lattice sum of the *molecular* ECP integrals, so we build a super-molecule
(reference-cell bra shells + image ket shells + image ECP projectors) and reuse
the validated molecular kernel ``libgecp.ECP_cart`` over the cross-image task
list, then fold the ket-image axis back with per-k phase factors
``exp(i k . L)``.

STATUS: not yet implemented -- the super-molecule construction and the
lattice-sum fold need a working PBC PySCF + GPU to develop and validate
against.  ``test_ecp_pbc.py`` codifies the acceptance criteria.
'''

import numpy as np

__all__ = ['ecp_int']


def _ecp_rcut(cell):
    '''Real-space cutoff for the ECP operator: never shorter than ``cell.rcut``
    (tuned for AO overlap), extended for the slowest-decaying ECP primitive so
    the lattice sum captures the potential's tail to ``cell.precision``.
    '''
    from pyscf import gto
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


def _ecp_supmol(cell, Ls):
    '''Build the super-molecule for the ECP lattice sum.

    Returns a ``gto.Mole`` whose basis is
        [ reference-cell shells | shells translated by each L in Ls ]
    and whose ``_ecpbas`` is
        [ reference-cell projectors translated by each L in Ls ]
    with ``_env`` coordinate/exponent pointers fixed up accordingly.

    TODO(ecp-pbc phase 1): implement.  Mirror ``_build_supcell_`` in
    ``pyscf/pbc/tools/pbc.py`` for the atom/basis replication (it does *not*
    translate ``_ecpbas`` -- that part is new here), keeping the reference-cell
    shells as rows ``0:nbas_ref`` so the caller can slice the ``[i_bra, j_ket]``
    block.
    '''
    raise NotImplementedError


def ecp_int(cell, kpts=None):
    '''Periodic short-range ECP integrals (scalar).

    Args:
        cell : :class:`pyscf.pbc.gto.Cell` with ``cell._ecpbas``.
        kpts : (nkpts, 3) array or None (Gamma).

    Returns:
        CuPy array ``[nao, nao]`` (Gamma, real) or ``[nkpts, nao, nao]``
        (complex).  Matches ``pyscf.pbc.gto.ecp.ecp_int(cell, kpts, 'ECPscalar')``.
    '''
    if len(cell._ecpbas) == 0:
        raise ValueError('cell has no ECP basis')
    if np.any(cell._ecpbas[:, 4] == 1):   # SO_TYPE_OF
        raise NotImplementedError(
            'PBC SO-ECP not implemented (docs/ecp-pbc-design.md phase 3)')

    # Ls = cell.get_lattice_Ls(rcut=_ecp_rcut(cell))
    # supmol, nL, nao_ref = _ecp_supmol(cell, Ls)
    # blk = <i(0)| U(images) | j(ket images)>  -> [nao_ref, nL, nao_ref]  (cart->sph)
    # if kpts is None:  return blk.sum(axis=1).real
    # phase = exp(1j * kpts @ Ls.T)                    # [nkpts, nL]
    # return contract('kL,iLj->kij', phase, blk)
    raise NotImplementedError(
        'PBC ECP not yet implemented -- see docs/ecp-pbc-design.md. '
        f'({len(cell._ecpbas)} ECP shells, '
        f'{"Gamma" if kpts is None else np.shape(kpts)} kpts)')
