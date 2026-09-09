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

"""Stage 2a actinide validation -- uranyl UO2(2+), scalar vs spin-orbit.

    python uranyl_soc_2a.py phase0     # basis/ECP enumeration + integral checks
    python uranyl_soc_2a.py phase1     # scalar RKS geometry
    python uranyl_soc_2a.py phase3     # FD Hessian / frequencies at the min
    python uranyl_soc_2a.py phase4     # + PCM, re-optimize (scalar)

Finding (phase0): this pyscf checkout has NO spin-orbit ECP for any actinide
(has_ecp_soc() is False for U crenbl / stuttgart_rsc / stuttgart_dz / lanl2dz --
all are 2-column scalar-relativistic RECPs).  So Phase 2 (SOC geometry) and the
SOC half of Phase 3 CANNOT be run; this driver does the scalar validation and
documents the block.
"""

import sys
import time
import json
import numpy as np

from pyscf import gto

HERE = __file__.rsplit('/', 1)[0]

U_BASIS = 'stuttgart_rsc'     # ECP60MWB small-core relativistic (32 valence e-)
O_BASIS = 'def2-tzvp'
XC = 'pbe0'
HARTREE2CM = 219474.6313632


def uranyl(r=1.75, basis_u=U_BASIS, basis_o=O_BASIS, ecp_u=U_BASIS):
    return gto.M(atom=f'U 0 0 0; O 0 0 {r}; O 0 0 -{r}', charge=2, spin=0,
                 basis={'U': basis_u, 'O': basis_o}, ecp={'U': ecp_u},
                 unit='Angstrom', verbose=0, output='/dev/null')


def rks(mol, level=3):
    from gpu4pyscf import dft
    mf = dft.RKS(mol, xc=XC)
    mf.grids.level = level
    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-6
    return mf


def _geom(mol):
    c = mol.atom_coords() * 0.52917721092
    rUO1 = np.linalg.norm(c[1] - c[0])
    rUO2 = np.linalg.norm(c[2] - c[0])
    v1 = (c[1] - c[0]) / rUO1
    v2 = (c[2] - c[0]) / rUO2
    ang = np.degrees(np.arccos(np.clip(v1 @ v2, -1, 1)))
    return rUO1, rUO2, ang


# --------------------------------------------------------------------------
# PHASE 0 -- prerequisites
# --------------------------------------------------------------------------
def phase0():
    print('=== PHASE 0: uranyl prerequisites ===', flush=True)
    import cupy as cp
    from gpu4pyscf.gto.ecp import get_ecp

    print('-- 0a: enumerate U ECPs and has_ecp_soc --')
    rows = []
    for ecp in ['crenbl', 'stuttgart_rsc', 'stuttgart_dz', 'lanl2dz']:
        try:
            m = uranyl(ecp_u=ecp, basis_u=ecp)
            ub = m._ecp['U']
            lmax = max(b[0] for b in ub[1])
            r = dict(ecp=ecp, ncore=ub[0], ecp_lmax=int(lmax), nao=m.nao,
                     has_ecp_soc=bool(m.has_ecp_soc()))
        except Exception as e:
            r = dict(ecp=ecp, error=f'{type(e).__name__}: {e}')
        rows.append(r)
        print('   ', r, flush=True)

    print('-- 0b: ECPso integral machinery on the actinide basis --')
    m = uranyl()
    try:
        get_ecp_so_out = 'raised'
        from gpu4pyscf.gto.ecp import get_ecp_so
        g = get_ecp_so(m)
        get_ecp_so_out = float(abs(g).max())
    except Exception as e:
        get_ecp_so_out = f'{type(e).__name__}: {e}'
    ref = np.asarray(m.intor('ECPso'))
    print(f'    get_ecp_so -> {get_ecp_so_out}')
    print(f"    mol.intor('ECPso') max|.| = {abs(ref).max():.1e}  shape {ref.shape}")

    print('-- 0b(scalar ECP GPU vs CPU on small-core uranyl; regression on large-core) --')
    ecp_check = {}
    for ecp in ['stuttgart_rsc', 'crenbl', 'stuttgart_dz', 'lanl2dz']:
        m = uranyl(ecp_u=ecp, basis_u=ecp, basis_o='def2-svp')
        d = float(abs(cp.asarray(get_ecp(m)).get() - m.intor('ECPscalar')).max())
        ecp_check[ecp] = d
        print(f'    {ecp}: |ECPscalar gpu-cpu|_max = {d:.2e}', flush=True)

    print('-- 0c: scalar RKS(pbe0) SCF --')
    m = uranyl()
    t0 = time.time()
    mf = rks(m)
    e_scalar = mf.kernel()
    dt = time.time() - t0
    print(f'    RKS(pbe0) {U_BASIS}/{O_BASIS} nao={m.nao} converged={mf.converged} '
          f'e_tot={e_scalar:.8f}  ({dt:.0f}s)', flush=True)

    json.dump(dict(ecp_rows=rows, get_ecp_so=str(get_ecp_so_out),
                   ECPso_ref_max=float(abs(ref).max()), ecp_gpu_cpu=ecp_check,
                   e_scalar=float(e_scalar), scf_wall_s=dt,
                   nao=m.nao, converged=bool(mf.converged)),
              open(f'{HERE}/_uranyl_phase0.json', 'w'), indent=1)


# --------------------------------------------------------------------------
# PHASE 1 -- scalar geometry
# --------------------------------------------------------------------------
def phase1():
    print('=== PHASE 1: scalar RKS(pbe0) geometry ===', flush=True)
    from pyscf.geomopt.geometric_solver import optimize
    mol = uranyl(r=1.75)
    mf = rks(mol)
    t0 = time.time()
    mol_eq = optimize(mf, maxsteps=40, convergence_grms=1e-4,
                      convergence_gmax=2e-4)
    dt = time.time() - t0
    rUO1, rUO2, ang = _geom(mol_eq)
    # fresh SCF + gradient at the minimum
    mf2 = rks(mol_eq)
    mf2.conv_tol = 1e-10
    mf2.kernel()
    g = mf2.nuc_grad_method().kernel()
    gmax = float(np.abs(np.asarray(g)).max())
    print(f'[P1] r(U=O) = {rUO1:.5f} / {rUO2:.5f} Ang   O=U=O = {ang:.3f} deg')
    print(f'[P1] e_tot(min) = {mf2.e_tot:.8f}   |grad|_max = {gmax:.2e}   ({dt:.0f}s)',
          flush=True)
    coords = mol_eq.atom_coords().tolist()
    json.dump(dict(r_scalar=rUO1, r_scalar_2=rUO2, angle=ang,
                   e_tot=float(mf2.e_tot), grad_max=gmax, wall_s=dt,
                   coords_bohr=coords, u_basis=U_BASIS, o_basis=O_BASIS,
                   u_ecp=U_BASIS),
              open(f'{HERE}/_uranyl_phase1.json', 'w'), indent=1)


# --------------------------------------------------------------------------
# PHASE 3 -- frequencies (scalar)
# --------------------------------------------------------------------------
def _freqs(mol, mf):
    from gpu4pyscf.hessian.fd import finite_diff_hessian
    from pyscf.hessian import thermo
    if getattr(mf, 'conv_tol', 1.0) > 1e-11:   # fd.py requires a tight ref
        mf.conv_tol = 1e-12
        mf.kernel()
    H = finite_diff_hessian(mf, disp=1e-3)
    fi = thermo.harmonic_analysis(mol, H, exclude_trans=False, exclude_rot=False)
    w = np.sort(fi['freq_wavenumber'].real)
    im = np.sort(fi['freq_wavenumber'].imag)
    return w, im, H


def phase3():
    print('=== PHASE 3: scalar frequencies at the optimized minimum ===', flush=True)
    d1 = json.load(open(f'{HERE}/_uranyl_phase1.json'))
    r = d1['r_scalar']
    mol = uranyl(r=r)
    mf = rks(mol)
    mf.conv_tol = 1e-10
    t0 = time.time()
    mf.kernel()
    w, im, H = _freqs(mol, mf)
    dt = time.time() - t0
    # linear XY2: 3N-5 = 4 real modes; identify nu1 (sym str), nu2 (bend, x2),
    # nu3 (asym str) as the 4 highest-|w|
    real4 = w[np.abs(w) > 30.0]
    print(f'[P3] all 9 wavenumbers (cm^-1): {np.round(w, 1)}')
    print(f'[P3] imaginary parts (cm^-1):   {np.round(im, 1)}')
    print(f'[P3] the 4 vibrational modes:   {np.round(np.sort(real4), 1)}   ({dt:.0f}s)',
          flush=True)
    json.dump(dict(r=r, wavenumbers=w.tolist(), imag=im.tolist(),
                   vib4=np.sort(real4).tolist(), wall_s=dt,
                   e_tot=float(mf.e_tot)),
              open(f'{HERE}/_uranyl_phase3.json', 'w'), indent=1)


# --------------------------------------------------------------------------
# PHASE 4 -- PCM (scalar), re-optimize
# --------------------------------------------------------------------------
def phase4():
    print('=== PHASE 4: scalar RKS(pbe0) + C-PCM, re-optimize ===', flush=True)
    from pyscf.geomopt.geometric_solver import optimize
    d1 = json.load(open(f'{HERE}/_uranyl_phase1.json'))
    mol = uranyl(r=d1['r_scalar'])
    mf = rks(mol).PCM()
    mf.with_solvent.eps = 78.3553
    mf.with_solvent.method = 'C-PCM'
    t0 = time.time()
    mol_eq = optimize(mf, maxsteps=40, convergence_grms=1e-4,
                      convergence_gmax=2e-4)
    dt = time.time() - t0
    rUO1, rUO2, ang = _geom(mol_eq)
    mf2 = rks(mol_eq).PCM()
    mf2.with_solvent.eps = 78.3553
    mf2.with_solvent.method = 'C-PCM'
    mf2.conv_tol = 1e-10
    mf2.kernel()
    w, im, _ = _freqs(mol_eq, mf2)
    real4 = np.sort(w[np.abs(w) > 30.0])
    print(f'[P4] PCM r(U=O) = {rUO1:.5f} Ang   O=U=O = {ang:.3f} deg')
    print(f'[P4] PCM vib modes (cm^-1): {np.round(real4, 1)}   ({dt:.0f}s)', flush=True)
    json.dump(dict(r_pcm=rUO1, angle=ang, vib4=real4.tolist(),
                   e_tot=float(mf2.e_tot), wall_s=dt),
              open(f'{HERE}/_uranyl_phase4.json', 'w'), indent=1)


if __name__ == '__main__':
    ph = sys.argv[1] if len(sys.argv) > 1 else 'phase0'
    t0 = time.time()
    {'phase0': phase0, 'phase1': phase1, 'phase3': phase3, 'phase4': phase4}[ph]()
    print(f'[{ph}] done in {time.time()-t0:.0f}s', flush=True)
