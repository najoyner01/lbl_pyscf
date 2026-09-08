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

'''
Hirshfeld ("stockholder") population analysis and CM5 charges.

Hirshfeld charge:

    q_A = Z_A - integral[ w_A(r) rho_mol(r) d3r ]

    w_A(r) = rho_A^0(r) / sum_B rho_B^0(r)          (stockholder weight)

where rho_A^0 is the spherically averaged neutral free-atom density of atom A,
placed at that atom's nuclear position; the sum over B (the "promolecule") runs
over every atom.  Z_A is the *effective* nuclear charge (``mol.atom_charge``),
so this is consistent with ECP calculations -- the free-atom reference is built
with the same ECP (``pyscf.scf.atom_hf`` handles the core removal), and the
integral then counts only the valence density on both sides.

CM5 (Charge Model 5, Marenich, Cramer, Truhlar, J. Chem. Theory Comput. 2012,
8, 527) is a parametrized, geometry-only correction on top of the Hirshfeld
charge:

    q_A^CM5 = q_A^HD + sum_{B != A} T_AB B_AB
    B_AB    = exp(-alpha (r_AB - R_A - R_B))          r_AB, R in Angstrom
    T_AB    = D_{Z_A Z_B}   for the six special pairs (H-C, H-N, H-O,
                             C-N, C-O, N-O), antisymmetric
            = D_{Z_A} - D_{Z_B}   otherwise

No CPU PySCF reference exists for either quantity; validate with the Sigma q =
total-charge sum rule and against published CM5 values (see tests).

Free-atom reference densities come from ``pyscf.scf.atom_hf`` (spherically
averaged, fractionally occupied atomic RHF) -- the same machinery PySCF uses
for its atomic SCF initial guess.  Only the small 1-D radial reference is done
on the CPU; the promolecule sum and the real-space integral run on the GPU.
'''

import numpy as np
import cupy as cp
from pyscf import gto
from pyscf.scf import atom_hf
from pyscf.data import radii
from gpu4pyscf.dft import numint
from gpu4pyscf.dft.gen_grid import Grids
from gpu4pyscf.lib import logger

BOHR = radii.BOHR  # Angstrom per Bohr

# --- radial free-atom reference grid -----------------------------------------
_RAD_N = 2000
_RAD_RMIN = 1e-4      # Bohr
_RAD_RMAX = 30.0      # Bohr; density is numerically zero well before this
_RHO_FLOOR = 1e-300


def _free_atom_radial_density(symb, mol, nelec_core=0):
    '''Spherically averaged neutral free-atom density rho_A^0(r) on a 1-D log
    radial grid (Bohr).  Returns (r_grid, rho_grid), both numpy, using the
    basis (and ECP, if any) that ``mol`` assigns to element ``symb``.
    ``nelec_core`` is the number of electrons the ECP removes for this element
    (0 for all-electron), needed to pick a consistent spin before build.'''
    pure = gto.mole._std_symbol(symb)
    try:
        bas = mol._basis[symb]
    except KeyError:
        bas = mol._basis[pure]
    atm = gto.Mole()
    atm.atom = [[pure, (0.0, 0.0, 0.0)]]
    atm.basis = {pure: bas}
    ecp = mol._ecp.get(symb, mol._ecp.get(pure)) if mol._ecp else None
    if ecp is not None:
        atm.ecp = {pure: ecp}
    atm.charge = 0
    # spin must be consistent with the (ECP-reduced) electron count *before*
    # build(); the spherical-average SCF only uses it to pick fractional occ.
    atm.spin = (gto.charge(pure) - nelec_core) % 2
    atm.cart = False          # atom_hf.AtomSphAverageRHF is spherical only
    atm.verbose = 0
    atm.build()

    res = atom_hf.get_atm_nrhf(atm)          # {pure: (e_tot, mo_e, mo_coeff, mo_occ)}
    _, _, mo_coeff, mo_occ = res[pure]
    dm = (mo_coeff * mo_occ).dot(mo_coeff.conj().T)

    r_grid = np.exp(np.linspace(np.log(_RAD_RMIN), np.log(_RAD_RMAX), _RAD_N))
    if dm.size == 0:                         # bare ghost / no basis
        return r_grid, np.full(_RAD_N, _RHO_FLOOR)
    # spherical average: sampling along +x is exact for a spherically averaged dm
    coords = np.zeros((_RAD_N, 3))
    coords[:, 0] = r_grid
    ao = atm.eval_gto('GTOval_sph', coords)          # (nr, nao)
    rho = np.einsum('pi,ij,pj->p', ao, dm, ao)
    rho = np.clip(rho.real, _RHO_FLOOR, None)
    return r_grid, rho


def _promol_weights(mol, coords, log=None):
    '''Stockholder weights w_A(r) for every atom on the grid ``coords`` (cupy,
    Bohr, shape (ngrid,3)).  Returns (w, rho0) each shape (natm, ngrid) cupy,
    where rho0 are the individual free-atom densities and w = rho0 / sum.'''
    natm = mol.natm
    ngrid = coords.shape[0]
    coords = cp.asarray(coords)

    ref_cache = {}
    rho0 = cp.zeros((natm, ngrid))
    for ia in range(natm):
        symb = mol.atom_symbol(ia)
        if symb not in ref_cache:
            r_np, rho_np = _free_atom_radial_density(
                symb, mol, nelec_core=mol.atom_nelec_core(ia))
            ref_cache[symb] = (cp.asarray(np.log(r_np)),
                               cp.asarray(np.log(rho_np)))
            if log is not None:
                log.debug('Hirshfeld free-atom reference built for %s '
                          '(rho(0+)=%.3e)', symb, rho_np[0])
        logr, logrho = ref_cache[symb]
        d = cp.linalg.norm(coords - cp.asarray(mol.atom_coord(ia)), axis=1)
        d = cp.maximum(d, _RAD_RMIN)
        logd = cp.log(d)
        # log-log linear interpolation; extrapolate to ~0 outside the tail
        lr = cp.interp(logd, logr, logrho, left=logrho[0], right=-700.0)
        rho0[ia] = cp.exp(lr)

    denom = rho0.sum(axis=0)
    # where the promolecule vanishes, no atom has any claim; assign 0 weight
    good = denom > 1e-300
    w = cp.where(good[None, :], rho0 / cp.where(good, denom, 1.0)[None, :], 0.0)
    return w, rho0


def hirshfeld_charges(mol, dm, grids=None, grid_level=3, verbose=None):
    '''Hirshfeld (stockholder) atomic charges.

    Args:
        mol : Mole
        dm  : SCF density matrix.  (nao,nao) for RHF/RKS, or (2,nao,nao) /
              stacked spin blocks for UHF/UKS (spin-summed internally).
        grids : optional pre-built ``gpu4pyscf.dft.gen_grid.Grids``.
        grid_level : Becke-grid level if ``grids`` is not supplied.

    Returns:
        cupy array of length ``mol.natm`` (charges, electrons).  Sum equals the
        total molecular charge to grid accuracy.
    '''
    log = logger.new_logger(mol, verbose)
    nao = mol.nao
    dm = cp.asarray(dm)
    if dm.ndim == 3:                                  # UHF/UKS spin blocks
        dm = dm.sum(axis=0)
    elif dm.ndim == 2 and dm.shape[0] == 2 * nao:     # GHF/GKS (2nao,2nao)
        dm = (dm[:nao, :nao] + dm[nao:, nao:]).real
    dm = dm.real if cp.iscomplexobj(dm) else dm

    if grids is None:
        grids = Grids(mol)
        grids.level = grid_level
        grids.build(sort_grids=False)

    ni = numint.NumInt()
    rho_mol = numint.get_rho(ni, mol, dm, grids)          # (ngrid,) cupy
    wgt = cp.asarray(grids.weights)

    w, _ = _promol_weights(mol, cp.asarray(grids.coords), log=log)
    nelec_A = (w * (rho_mol * wgt)[None, :]).sum(axis=1)   # (natm,)

    z_eff = cp.asarray([mol.atom_charge(ia) for ia in range(mol.natm)],
                       dtype=nelec_A.dtype)
    q = z_eff - nelec_A
    log.debug('Hirshfeld charges: %s  (sum=%.6f, target=%.6f)',
              q.get(), float(q.sum().get()), mol.charge)
    return q


# ---------------------------------------------------------------------------
# CM5  (Charge Model 5)
# ---------------------------------------------------------------------------
# Marenich, Cramer, Truhlar, J. Chem. Theory Comput. 2012, 8, 527-541.
#
# Atomic D_Z parameters, index = atomic number (D_Z[0] unused).  Transcribed
# from Table 1 of the paper.  Confident for H-Ar (the elements that matter for
# organic extractant ligands); heavier p-block values should be checked against
# the paper before publication-quality use.  All d/f-block and every element
# with Z >= 87 that is not Fr have D_Z = 0 in the model (Fr = 0.0046, Ra = 0),
# so for actinides (Z = 89-103) CM5 reduces to Hirshfeld plus the pairwise
# B_AB terms with T_{An,L} = -D_L.
_CM5_ALPHA = 2.474            # 1 / Angstrom
_CM5_D = np.zeros(119)
_CM5_D[1:37] = [
    0.0056, -0.1543, 0.0000, 0.0333, -0.1030, -0.0446, -0.1072, -0.0802,   # H  - O
    -0.0629, -0.1088, 0.0184, 0.0000, -0.0726, -0.0790, -0.0756, -0.0565,  # F  - S
    -0.0444, -0.0767, 0.0130, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,      # Cl - Cr
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0512, -0.0557,      # Mn - Ge
    -0.0533, -0.0399, -0.0313, -0.0541,                                    # As - Kr
]
_CM5_D[37:55] = [
    0.0092, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,        # Rb - Ru
    0.0000, 0.0000, 0.0000, 0.0000, -0.0361, -0.0393, -0.0376, -0.0281,    # Rh - Te
    -0.0220, -0.0381,                                                      # I  - Xe
]
_CM5_D[55:87] = [
    0.0065, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,        # Cs - Sm
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,        # Eu - Yb
    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,        # Lu - Pt
    0.0000, 0.0000, -0.0255, -0.0277, -0.0265, -0.0198, -0.0155, -0.0269,  # Au - Rn
]
_CM5_D[87] = 0.0046          # Fr;  Ra and everything heavier stay 0

# Six special atom pairs: T_AB = D_{Z_A Z_B}, antisymmetric under swap.
_CM5_TZZ = {
    (1, 6): 0.0502, (1, 7): 0.1747, (1, 8): 0.1671,
    (6, 7): 0.0556, (6, 8): 0.0234, (7, 8): -0.0346,
}

# Cordero et al. 2008 covalent radii (Angstrom) -- exactly the set the CM5
# paper uses.  pyscf stores them in Bohr.
_COV_RADII_ANG = radii.COVALENT * BOHR


def _cm5_pair_T(za, zb):
    if za == zb:
        return 0.0
    key = (za, zb) if za < zb else (zb, za)
    if key in _CM5_TZZ:
        t = _CM5_TZZ[key]
        return t if (za, zb) == key else -t
    return _CM5_D[za] - _CM5_D[zb]


def cm5_charges(mol, dm, grids=None, grid_level=3, hirshfeld=None, verbose=None):
    '''CM5 atomic charges.

    ``hirshfeld`` : optionally pass a precomputed Hirshfeld-charge array to skip
    recomputing it.  Returns a cupy array of length ``mol.natm``; the sum equals
    the total molecular charge exactly (the pairwise correction is
    antisymmetric).
    '''
    log = logger.new_logger(mol, verbose)
    if hirshfeld is None:
        q_hd = hirshfeld_charges(mol, dm, grids=grids, grid_level=grid_level,
                                 verbose=verbose)
    else:
        q_hd = cp.asarray(hirshfeld)

    natm = mol.natm
    zs = [gto.charge(mol.atom_symbol(ia)) for ia in range(natm)]
    coords_ang = mol.atom_coords() * BOHR       # (natm,3) numpy, Angstrom

    corr = np.zeros(natm)
    for a in range(natm):
        for b in range(natm):
            if a == b:
                continue
            r_ab = np.linalg.norm(coords_ang[a] - coords_ang[b])
            r_a = _COV_RADII_ANG[zs[a]]
            r_b = _COV_RADII_ANG[zs[b]]
            b_ab = np.exp(-_CM5_ALPHA * (r_ab - r_a - r_b))
            corr[a] += _cm5_pair_T(zs[a], zs[b]) * b_ab

    q = q_hd + cp.asarray(corr)
    log.debug('CM5 charges: %s  (sum=%.6f)', q.get(), float(q.sum().get()))
    return q


def _print_charges(mol, q, label):
    print(f' ** {label} charges **')
    for ia in range(mol.natm):
        print(f'{ia:>4d} {mol.atom_symbol(ia):>3s}  {float(q[ia]):>12.6f}')
    print(f' sum = {float(sum(float(q[i]) for i in range(mol.natm))):.6f}')
