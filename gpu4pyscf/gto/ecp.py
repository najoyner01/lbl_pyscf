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

import ctypes
import numpy as np
import cupy as cp
from pyscf import gto
from pyscf import __config__
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import load_library, contract, get_avail_mem
from gpu4pyscf.gto.mole import group_basis

libecp = load_library('libgecp')

ecp_cart_argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

libecp.ECP_cart.argtypes = ecp_cart_argtypes
libecp.ECP_ip_cart.argtypes = ecp_cart_argtypes
libecp.ECP_ipipv_cart.argtypes = ecp_cart_argtypes
libecp.ECP_ipvip_cart.argtypes = ecp_cart_argtypes

# Spin-orbit ECP kernel (may be absent in an older libgecp build).
_HAS_SO = hasattr(libecp, 'ECP_so_cart')
if _HAS_SO:
    libecp.ECP_so_cart.argtypes = ecp_cart_argtypes

ECP_ATOM_ID = 7

# Enable/disable the shell-pair x ECP-center screening of the task list.
SCREEN_ECP = getattr(__config__, 'gto_ecp_screen', True)

# Exponential-argument cutoff for the 3-center overlap estimate used to skip
# negligible (shell i, shell j, ECP center C) triples.  exp(-EXPCUTOFF) ~ 1e-17,
# matching the value hard-coded in pyscf/lib/gto/nr_ecp.h.
EXPCUTOFF = 39.0


def _ecp_expcutoff(mol):
    '''Screening cutoff on the exponential argument.  Never looser than the
    pyscf default (EXPCUTOFF); tightened further when mol.precision is tighter.
    '''
    prec = getattr(mol, 'precision', 1e-13)
    if not prec or prec <= 0:
        prec = 1e-13
    return max(EXPCUTOFF, -float(np.log(prec)))


def _build_screen_data(_sorted_mol, ecpbas, ecp_loc):
    '''Per-shell and per-ECP-group data needed by _screen_grid.

    Returns
        (bas_min_exp[nbas], bas_coords[nbas,3],
         ecp_min_exp[ngrp], ecp_coords[ngrp,3])
    where a group is one (l, atom) block delimited by ecp_loc, matching the
    third column (ksh) of the task grids.  All exponents are the smallest
    primitive exponent (slowest decay = most conservative bound).
    '''
    bas = _sorted_mol._bas
    env = _sorted_mol._env
    atom_coords = _sorted_mol.atom_coords()  # Bohr

    ptr = bas[:, gto.PTR_EXP]
    nprim = bas[:, gto.NPRIM_OF]
    # np.inf for a zero-primitive padding shell -> screened out (its
    # contraction coefficients are zero anyway).
    bas_min_exp = np.array(
        [env[p:p+n].min() if n > 0 else np.inf for p, n in zip(ptr, nprim)],
        dtype=np.float64)
    bas_coords = atom_coords[bas[:, gto.ATOM_OF]]

    ecpbas = np.asarray(ecpbas)
    ngrp = len(ecp_loc) - 1
    ecp_min_exp = np.empty(ngrp, dtype=np.float64)
    ecp_coords = np.empty((ngrp, 3), dtype=np.float64)
    for g in range(ngrp):
        rows = ecpbas[ecp_loc[g]:ecp_loc[g+1]]
        exps = [env[r[gto.PTR_EXP]:r[gto.PTR_EXP]+r[gto.NPRIM_OF]] for r in rows]
        exps = np.concatenate(exps) if exps else np.empty(0)
        ecp_min_exp[g] = exps.min() if exps.size else 0.0
        ecp_coords[g] = atom_coords[rows[0, gto.ATOM_OF]]
    return bas_min_exp, bas_coords, ecp_min_exp, ecp_coords


def _screen_grid(grid, screen_data, expcutoff):
    '''Drop (ish, jsh, ksh) rows whose 3-center overlap estimate is negligible.

    Mirrors check_3c_overlap() in pyscf/lib/gto/nr_ecp.c:
        eijk = (ai*aj*|Ri-Rj|^2 + ai*ak*|C-Ri|^2 + aj*ak*|C-Rj|^2) / (ai+aj+ak)
    keep the triple when eijk < expcutoff.
    '''
    if grid.shape[0] == 0:
        return grid
    bas_min_exp, bas_coords, ecp_min_exp, ecp_coords = screen_data
    ish = grid[:, 0]
    jsh = grid[:, 1]
    ksh = grid[:, 2]
    ai = bas_min_exp[ish]
    aj = bas_min_exp[jsh]
    ak = ecp_min_exp[ksh]
    rab = bas_coords[ish] - bas_coords[jsh]
    rca = ecp_coords[ksh] - bas_coords[ish]
    rcb = ecp_coords[ksh] - bas_coords[jsh]
    rrab = np.sum(rab*rab, axis=1)
    rrca = np.sum(rca*rca, axis=1)
    rrcb = np.sum(rcb*rcb, axis=1)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        eijk = (ai*aj*rrab + ai*ak*rrca + aj*ak*rrcb) / (ai + aj + ak)
    return grid[eijk < expcutoff]


def sort_ecp_basis(_ecpbas, cart=True, log=None):
    '''
    # Sort ECP basis based on angular momentum
    # Remove SO Type basis functions
    '''
    not_so_type = _ecpbas[:, gto.SO_TYPE_OF] == 0
    _ecpbas = _ecpbas[not_so_type]

    # Sort ECP basis based on angular momentum and atom_id
    l_atm = _ecpbas[:,[gto.ANG_OF, gto.ATOM_OF]]

    uniq_l_atm, inv_idx, l_atm_counts = np.unique(
        l_atm, return_inverse=True, return_counts=True, axis=0)
    sorted_idx = np.argsort(inv_idx.ravel(), kind='stable').astype(np.int32)

    # Sort basis inplace
    _ecpbas = _ecpbas[sorted_idx]

    # Group ECP basis based on angular momentum and atom id
    # Each group contains basis with multiple power order
    ecp_loc = np.append(0, np.cumsum(l_atm_counts))

    # Further group based on angular momentum for counting
    uniq_l, l_counts = np.unique(uniq_l_atm[:,0], return_counts=True, axis=0)

    return _ecpbas, uniq_l, l_counts, ecp_loc

def _screen_block(ish_range, jsh_range, ksh_range, screen_data, expcutoff,
                  triangular):
    '''(ish, jsh, ksh) task rows surviving the 3-center overlap screen, built
    without ever materializing the full ish x jsh x ksh grid.

    Two levels, per ECP group k:
      1. keep shell i (resp. j) only if the best-case bound (partner
         exponent -> 0) is above threshold:
             ai*ak*|C-Ri|^2  <  expcutoff * (ai + ak)
      2. run the full check_3c_overlap estimate on the surviving (i, j) pairs.
    When nothing screens out (small/dense systems) level 2 costs the same as the
    old full meshgrid; when most triples are negligible the work collapses to
    the small surviving neighborhood.
    '''
    bas_min_exp, bas_coords, ecp_min_exp, ecp_coords = screen_data
    ish_range = np.asarray(ish_range)
    jsh_range = np.asarray(jsh_range)
    ai_all = bas_min_exp[ish_range]
    aj_all = bas_min_exp[jsh_range]
    ri_all = bas_coords[ish_range]
    rj_all = bas_coords[jsh_range]

    out = []
    # inf exponents (zero-primitive padding shells) produce inf/nan below; those
    # compare False and are dropped, which is the intended outcome.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        for k in ksh_range:
            ak = ecp_min_exp[k]
            ck = ecp_coords[k]
            rrca = np.sum((ri_all - ck)**2, axis=1)
            rrcb = np.sum((rj_all - ck)**2, axis=1)
            i_keep = ish_range[ai_all*ak*rrca < expcutoff * (ai_all + ak)]
            j_keep = jsh_range[aj_all*ak*rrcb < expcutoff * (aj_all + ak)]
            if i_keep.size == 0 or j_keep.size == 0:
                continue
            ii, jj = np.meshgrid(i_keep, j_keep, indexing='ij')
            ii = ii.ravel()
            jj = jj.ravel()
            if triangular:
                m = ii <= jj
                ii = ii[m]
                jj = jj[m]
                if ii.size == 0:
                    continue
            aii = bas_min_exp[ii]
            ajj = bas_min_exp[jj]
            rab = bas_coords[ii] - bas_coords[jj]
            rca = ck - bas_coords[ii]
            rcb = ck - bas_coords[jj]
            eijk = (aii*ajj*np.sum(rab*rab, axis=1)
                    + aii*ak*np.sum(rca*rca, axis=1)
                    + ajj*ak*np.sum(rcb*rcb, axis=1)) / (aii + ajj + ak)
            m = eijk < expcutoff
            if np.any(m):
                kk = np.full(int(np.count_nonzero(m)), k, dtype=np.int64)
                out.append(np.stack(
                    [ii[m].astype(np.int64), jj[m].astype(np.int64), kk],
                    axis=1))
    if out:
        return np.concatenate(out, axis=0)
    return np.zeros((0, 3), dtype=np.int64)


def _build_tasks(l_ctr_offsets, lecp_ctr_offsets, screen_data, expcutoff,
                 triangular):
    tasks = {}
    n_groups = len(l_ctr_offsets) - 1
    n_ecp_groups = len(lecp_ctr_offsets) - 1
    do_screen = screen_data is not None and SCREEN_ECP
    for i in range(n_groups):
        j_start = i if triangular else 0
        for j in range(j_start, n_groups):
            for k in range(n_ecp_groups):
                ish_range = np.arange(l_ctr_offsets[i], l_ctr_offsets[i+1])
                jsh_range = np.arange(l_ctr_offsets[j], l_ctr_offsets[j+1])
                ksh_range = np.arange(lecp_ctr_offsets[k], lecp_ctr_offsets[k+1])
                if do_screen:
                    grid = _screen_block(ish_range, jsh_range, ksh_range,
                                         screen_data, expcutoff, triangular)
                else:
                    grid = np.stack(
                        np.meshgrid(ish_range, jsh_range, ksh_range),
                        axis=-1).reshape(-1, 3)
                    if triangular:
                        grid = grid[grid[:, 0] <= grid[:, 1]]
                tasks[i, j, k] = grid
    return tasks


def make_tasks(l_ctr_offsets, lecp_ctr_offsets, screen_data=None,
               expcutoff=EXPCUTOFF):
    return _build_tasks(l_ctr_offsets, lecp_ctr_offsets, screen_data,
                        expcutoff, triangular=True)


def make_full_tasks(l_ctr_offsets, lecp_ctr_offsets, screen_data=None,
                    expcutoff=EXPCUTOFF):
    return _build_tasks(l_ctr_offsets, lecp_ctr_offsets, screen_data,
                        expcutoff, triangular=False)

def select_basis(ecpbas, ecp_atoms):
    """
    Select ECP basis for given ECP atoms, and reindexing atoms
    """
    atom_map = {}
    for idx, ecp_atom in enumerate(ecp_atoms):
        atom_map[ecp_atom] = idx

    selected_ecpbas = []
    for idx, bas in enumerate(ecpbas):
        atm_id = bas[gto.ATOM_OF]
        if atm_id in ecp_atoms:
            bas_copy = bas.copy()
            bas_copy[ECP_ATOM_ID] = atom_map[atm_id]
            selected_ecpbas.append(bas_copy)
    return np.array(selected_ecpbas)

def get_ecp(mol):
    """
    Calculate sum of ECP integrals

    Returns:
        CuPy array: [nao, nao]
            sum of ECP integrals over all ecp atoms
    """
    assert len(mol._ecpbas) > 0

    _sorted_mol, coeff, uniq_l_ctr, l_ctr_counts = group_basis(mol)

    _ecpbas = _sorted_mol._ecpbas
    _ecpbas, uniq_lecp, lecp_counts, ecp_loc= sort_ecp_basis(_ecpbas)

    l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
    lecp_offsets = np.append(0, np.cumsum(lecp_counts))

    screen_data = _build_screen_data(_sorted_mol, _ecpbas, ecp_loc)
    tasks_all = make_tasks(l_ctr_offsets, lecp_offsets, screen_data,
                           _ecp_expcutoff(mol))

    atm = cp.asarray(_sorted_mol._atm, dtype=np.int32)
    bas = cp.asarray(_sorted_mol._bas, dtype=np.int32)
    env = cp.asarray(_sorted_mol._env, dtype=np.float64)

    ecpbas = cp.asarray(_ecpbas, dtype=np.int32)
    ecploc = cp.asarray(ecp_loc, dtype=np.int32)
    n_groups = len(uniq_l_ctr)
    n_ecp_groups = len(uniq_lecp)
    ao_loc = _sorted_mol.ao_loc_nr(cart=True)
    nao = ao_loc[-1]
    ao_loc = cp.asarray(ao_loc, dtype=np.int32)

    mat1 = cp.zeros([nao, nao])
    for i in range(n_groups):
        for j in range(i,n_groups):
            for k in range(n_ecp_groups):
                tasks = cp.asarray(tasks_all[i,j,k], dtype=np.int32, order='F')
                ntasks = len(tasks)
                li = uniq_l_ctr[i,0]
                lj = uniq_l_ctr[j,0]
                lk = uniq_lecp[k]
                err = libecp.ECP_cart(
                    mat1.data.ptr, ao_loc.data.ptr, nao,
                    tasks.data.ptr, ntasks,
                    ecpbas.data.ptr, ecploc.data.ptr,
                    atm.data.ptr, bas.data.ptr, env.data.ptr,
                    li, lj, lk)
                if err != 0:
                    raise RuntimeError('ECP CUDA kernel failed.')
    coeff = cp.asarray(coeff)
    return coeff.T @ mat1 @ coeff


def sort_ecp_basis_so(_ecpbas):
    '''Like :func:`sort_ecp_basis` but keeps only the spin-orbit projectors
    (``SO_TYPE_OF == 1``) and rewrites the ``ul`` term (``ANG_OF == -1``) to
    ``max_l(atom) + 1``, matching ECPtype_so_cart in pyscf/lib/gto/nr_ecp.c.
    '''
    _ecpbas = _ecpbas[_ecpbas[:, gto.SO_TYPE_OF] == 1].copy()
    if len(_ecpbas) == 0:
        return _ecpbas, np.zeros(0, dtype=int), np.zeros(0, dtype=int), \
            np.zeros(1, dtype=int)

    # ul term (lc == -1) is treated as L_max + 1 for that atom
    # CPU (ECPtype_so_cart): ecp_lmax[atom] starts at 0 and is raised by the
    # non-ul projectors; the ul term then becomes ecp_lmax[atom] + 1.
    ul = _ecpbas[:, gto.ANG_OF] == -1
    if np.any(ul):
        for atm_id in np.unique(_ecpbas[ul, gto.ATOM_OF]):
            rows = _ecpbas[:, gto.ATOM_OF] == atm_id
            lmax = _ecpbas[rows & ~ul, gto.ANG_OF].max(initial=0)
            _ecpbas[rows & ul, gto.ANG_OF] = lmax + 1

    l_atm = _ecpbas[:, [gto.ANG_OF, gto.ATOM_OF]]
    uniq_l_atm, inv_idx, l_atm_counts = np.unique(
        l_atm, return_inverse=True, return_counts=True, axis=0)
    sorted_idx = np.argsort(inv_idx.ravel(), kind='stable').astype(np.int32)
    _ecpbas = _ecpbas[sorted_idx]
    ecp_loc = np.append(0, np.cumsum(l_atm_counts))
    uniq_l, l_counts = np.unique(uniq_l_atm[:, 0], return_counts=True, axis=0)
    return _ecpbas, uniq_l, l_counts, ecp_loc


def get_ecp_so(mol):
    '''Spin-orbit ECP integrals, real cartesian-component form.

    Returns
        CuPy array ``[3, nao, nao]`` (spherical AO basis, real): the three
        components ``<i| l_a dU^SO |j>`` for ``a in {x, y, z}``.  Equivalent to
        ``mol.intor('ECPso')``.  Assemble the spinor operator with
        :func:`get_soc_1e`.
    '''
    assert len(mol._ecpbas) > 0
    if not _HAS_SO:
        raise NotImplementedError(
            'libgecp was built without ECP_so_cart; rebuild gpu4pyscf/lib.')
    if not np.any(mol._ecpbas[:, gto.SO_TYPE_OF] == 1):
        raise ValueError('mol has no spin-orbit ECP projectors')

    _sorted_mol, coeff, uniq_l_ctr, l_ctr_counts = group_basis(mol)
    _ecpbas, uniq_lecp, lecp_counts, ecp_loc = sort_ecp_basis_so(
        _sorted_mol._ecpbas)

    l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
    lecp_offsets = np.append(0, np.cumsum(lecp_counts))

    screen_data = _build_screen_data(_sorted_mol, _ecpbas, ecp_loc)
    tasks_all = make_tasks(l_ctr_offsets, lecp_offsets, screen_data,
                           _ecp_expcutoff(mol))

    atm = cp.asarray(_sorted_mol._atm, dtype=np.int32)
    bas = cp.asarray(_sorted_mol._bas, dtype=np.int32)
    env = cp.asarray(_sorted_mol._env, dtype=np.float64)
    ecpbas = cp.asarray(_ecpbas, dtype=np.int32)
    ecploc = cp.asarray(ecp_loc, dtype=np.int32)
    n_groups = len(uniq_l_ctr)
    n_ecp_groups = len(uniq_lecp)
    ao_loc = _sorted_mol.ao_loc_nr(cart=True)
    nao = int(ao_loc[-1])
    ao_loc = cp.asarray(ao_loc, dtype=np.int32)

    mat1 = cp.zeros([3, nao, nao])
    for i in range(n_groups):
        for j in range(i, n_groups):
            for k in range(n_ecp_groups):
                task = tasks_all[i, j, k]
                if len(task) == 0:
                    continue
                task = cp.asarray(task, dtype=np.int32, order='F')
                err = libecp.ECP_so_cart(
                    mat1.data.ptr, ao_loc.data.ptr, nao,
                    task.data.ptr, len(task),
                    ecpbas.data.ptr, ecploc.data.ptr,
                    atm.data.ptr, bas.data.ptr, env.data.ptr,
                    int(uniq_l_ctr[i, 0]), int(uniq_l_ctr[j, 0]),
                    int(uniq_lecp[k]))
                if err != 0:
                    raise RuntimeError('SO-ECP CUDA kernel failed.')

    coeff = cp.asarray(coeff)
    return contract('aij,ip,jq->apq', mat1, coeff, coeff)


def get_soc_1e(mol):
    '''One-electron spin-orbit ECP operator in the 2-component (spinor) basis.

    Returns
        CuPy array ``[2*nao, 2*nao]`` complex, matching the SOC block that
        ``pyscf.scf.ghf.GHF.get_hcore`` adds:
        ``einsum('sxy,spq->xpyq', -1j * 0.5 * PauliMatrices, get_ecp_so(mol))``.
    '''
    from pyscf import lib as pyscf_lib
    so = get_ecp_so(mol)                       # [3, nao, nao] real
    nao = so.shape[-1]
    pauli = cp.asarray(pyscf_lib.PauliMatrices)   # [3, 2, 2] complex
    hso = contract('sxy,spq->xpyq', -1j * 0.5 * pauli, so.astype(cp.complex128))
    return hso.reshape(2 * nao, 2 * nao)


_IP_FN = {
    'ip':    (libecp.ECP_ip_cart,    3),
    'ipipv': (libecp.ECP_ipipv_cart, 9),
    'ipvip': (libecp.ECP_ipvip_cart, 9),
}


class _EcpDerivContext:
    '''Batch-invariant data for the ECP-derivative kernels, built once so that
    slicing over ECP atoms only re-does the cheap per-batch work (basis
    selection, screening, task list, kernel launch, AO transform).
    '''
    def __init__(self, mol):
        self._sorted_mol, coeff, self.uniq_l_ctr, l_ctr_counts = group_basis(mol)
        self.l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        self.expcutoff = _ecp_expcutoff(mol)
        self.ecpbas_src = mol._ecpbas
        self.atm = cp.asarray(self._sorted_mol._atm, dtype=np.int32)
        self.bas = cp.asarray(self._sorted_mol._bas, dtype=np.int32)
        self.env = cp.asarray(self._sorted_mol._env, dtype=np.float64)
        ao_loc = self._sorted_mol.ao_loc_nr(cart=True)
        self.nao = int(ao_loc[-1])
        self.ao_loc = cp.asarray(ao_loc, dtype=np.int32)
        self.coeff = cp.asarray(coeff)
        self.n_groups = len(self.uniq_l_ctr)

    def run_batch(self, fn, comp, ecp_atoms_batch):
        '''[len(ecp_atoms_batch), comp, nao, nao] in the AO (transformed) basis.
        Row r corresponds to ecp_atoms_batch[r].'''
        ecpbas = select_basis(self.ecpbas_src, list(ecp_atoms_batch))
        ecpbas, uniq_lecp, lecp_counts, ecp_loc = sort_ecp_basis(ecpbas)
        lecp_offsets = np.append(0, np.cumsum(lecp_counts))
        screen_data = _build_screen_data(self._sorted_mol, ecpbas, ecp_loc)
        tasks_all = make_full_tasks(self.l_ctr_offsets, lecp_offsets,
                                    screen_data, self.expcutoff)

        ecpbas_gpu = cp.asarray(ecpbas, dtype=np.int32)
        ecploc_gpu = cp.asarray(ecp_loc, dtype=np.int32)
        n_ecp_groups = len(uniq_lecp)
        nbatch = len(ecp_atoms_batch)
        mat1 = cp.zeros([nbatch, comp, self.nao, self.nao])
        for i in range(self.n_groups):
            for j in range(self.n_groups):
                for k in range(n_ecp_groups):
                    task = tasks_all[i, j, k]
                    if len(task) == 0:
                        continue
                    task = cp.asarray(task, dtype=np.int32, order='F')
                    err = fn(
                        mat1.data.ptr, self.ao_loc.data.ptr, self.nao,
                        task.data.ptr, len(task),
                        ecpbas_gpu.data.ptr, ecploc_gpu.data.ptr,
                        self.atm.data.ptr, self.bas.data.ptr, self.env.data.ptr,
                        int(self.uniq_l_ctr[i, 0]), int(self.uniq_l_ctr[j, 0]),
                        int(uniq_lecp[k]))
                    if err != 0:
                        raise RuntimeError('ECP CUDA kernel failed.')
        mat1 = contract('axij,jq->axiq', mat1, self.coeff)
        mat1 = contract('axiq,ip->axpq', mat1, self.coeff)
        return mat1


def _default_ecp_atoms(mol):
    return sorted(set(int(a) for a in mol._ecpbas[:, gto.ATOM_OF]))


def _ecp_atom_batch_size(comp, nao, n_ecp_atm, batch_size=None):
    '''How many ECP atoms to process at once so the transient
    [batch, comp, nao, nao] tensor (plus AO-transform scratch) stays well
    within free GPU memory.'''
    if batch_size is not None:
        return max(1, min(int(batch_size), n_ecp_atm))
    # mat1 + the two contract() temporaries ~ 3 buffers of this size
    per_atom = comp * nao * nao * 8 * 3
    avail = get_avail_mem()
    b = int(avail * 0.2 / max(per_atom, 1))
    return max(1, min(b, n_ecp_atm))


def loop_ecp_ip(mol, ip_type='ip', ecp_atoms=None, batch_size=None):
    '''Generator over ECP atoms in memory-bounded chunks.

    Yields ``(atom_ids, mat)`` where ``atom_ids`` is a list of atom indices and
    ``mat`` is a CuPy array ``[len(atom_ids), comp, nao, nao]`` (comp = 3 for
    ``ip``, 9 for ``ipipv``/``ipvip``); ``mat[r]`` is the derivative w.r.t.
    ``atom_ids[r]``.  Avoids ever allocating the full
    ``[n_ecp_atoms, comp, nao, nao]`` tensor.
    '''
    assert len(mol._ecpbas) > 0
    if ip_type not in _IP_FN:
        raise ValueError(f'Invalid IP type: {ip_type}')
    fn, comp = _IP_FN[ip_type]

    if ecp_atoms is None:
        ecp_atoms = _default_ecp_atoms(mol)
    else:
        ecp_atoms = [int(a) for a in ecp_atoms]
    if len(ecp_atoms) == 0:
        return

    ctx = _EcpDerivContext(mol)
    bs = _ecp_atom_batch_size(comp, ctx.nao, len(ecp_atoms), batch_size)
    for p0 in range(0, len(ecp_atoms), bs):
        batch = ecp_atoms[p0:p0+bs]
        yield batch, ctx.run_batch(fn, comp, batch)


def loop_ecp_ipip(mol, ip_type='ipipv', ecp_atoms=None, batch_size=None):
    '''Same as :func:`loop_ecp_ip` for the second derivatives
    (``ipipv``/``ipvip``, comp = 9).'''
    yield from loop_ecp_ip(mol, ip_type=ip_type, ecp_atoms=ecp_atoms,
                           batch_size=batch_size)


def get_ecp_ip(mol, ip_type='ip', ecp_atoms=None, batch_size=None):
    """
    First derivative of ECP integrals

    Returns:
        CuPy array: [n_ecp_atoms, 3, nao, nao],
            first dimension reindexed according to sorted(ecp_atoms)

    Note: materializes the full tensor.  For large systems prefer
    :func:`loop_ecp_ip` or :func:`get_ecp_ip_sum`.
    """
    mats = [mat for _batch, mat in
            loop_ecp_ip(mol, ip_type, ecp_atoms, batch_size)]
    if not mats:
        _, comp = _IP_FN[ip_type]
        return cp.zeros([0, comp, mol.nao, mol.nao])
    return cp.concatenate(mats, axis=0)


def get_ecp_ipip(mol, ip_type='ipipv', ecp_atoms=None, batch_size=None):
    """
    Second derivatives of ECP integrals
    Args:
        ip_type:
            ipipv -> (i''|ecp|j)
            ipvip -> (i'|ecp|j')

    Returns:
        CuPy array: [n_ecp_atoms, 9, nao, nao],
            first dimension reindexed according to sorted(ecp_atoms)

    Note: materializes the full tensor.  For large systems prefer
    :func:`loop_ecp_ipip` or :func:`get_ecp_ipip_sum`.
    """
    return get_ecp_ip(mol, ip_type, ecp_atoms, batch_size)


def get_ecp_ip_sum(mol, ip_type='ip', ecp_atoms=None, batch_size=None):
    '''sum over ECP atoms of the derivative integrals -> [comp, nao, nao].

    Equivalent to ``get_ecp_ip(mol, ...).sum(axis=0)`` but memory-bounded.
    '''
    out = None
    for _batch, mat in loop_ecp_ip(mol, ip_type, ecp_atoms, batch_size):
        s = mat.sum(axis=0)
        out = s if out is None else out + s
    if out is None:
        _, comp = _IP_FN[ip_type]
        return cp.zeros([comp, mol.nao, mol.nao])
    return out


def get_ecp_ipip_sum(mol, ip_type='ipipv', ecp_atoms=None, batch_size=None):
    '''sum over ECP atoms of the 2nd-derivative integrals -> [9, nao, nao].'''
    return get_ecp_ip_sum(mol, ip_type, ecp_atoms, batch_size)
