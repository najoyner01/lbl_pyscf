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

"""Numerical study: does mcfun ``spin_samples`` matter for GKS + SO-ECP on a
GENUINELY NON-COLLINEAR open-shell density?

Companion to the closed-shell study (``spin_samples_convergence.md`` /
``docs/ghf-gradient-design.md`` sec 5.6.1), which found ``spin_samples``
irrelevant for HI GKS(pbe0)+SOC because the converged density is collinear
(``int|m| = 3e-8``).  Here the test system is an open-shell heavy atom whose
converged density has ``int|m| ~ O(1)``.

    python spin_samples_noncollinear.py phaseA   # I atom: int|m|, ratio, e_tot vs spin_samples
    python spin_samples_noncollinear.py phaseB   # I atom: collinear_thrd on/off stress test
    python spin_samples_noncollinear.py phaseC   # I atom: 2P3/2-2P1/2 dSCF splitting
    python spin_samples_noncollinear.py phaseD   # IO radical (secondary)
    python spin_samples_noncollinear.py phaseE   # rotational-invariance probe
"""

import sys
import time
import json
import numpy as np
import cupy as cp

from pyscf import gto
from gpu4pyscf import dft
from gpu4pyscf.hessian.fd import finite_diff_hessian

HERE = __file__.rsplit('/', 1)[0]
SS_LIST = [50, 110, 194, 302, 434, 590, 770, 974, 1202]
XC = 'pbe0'
CONV_TOL = 1e-12
GRID_LEVEL = 3
HARTREE2CM = 219474.6313632


def i_atom(spin=1):
    return gto.M(atom='I 0 0 0', basis='crenbl', ecp='crenbl', spin=spin,
                 verbose=0, output='/dev/null')


def io_mol(r=1.87, spin=1):
    return gto.M(atom=f'I 0 0 0; O 0 0 {r}', basis='crenbl',
                 ecp={'I': 'crenbl'}, spin=spin, verbose=0, output='/dev/null')


def make_gks_soc(mol, spin_samples, grid_level=GRID_LEVEL, collinear_thrd=0.99):
    mf = dft.GKS(mol, xc=XC)
    mf.with_soc = True
    mf.spin_samples = int(spin_samples)
    mf._numint.collinear_thrd = collinear_thrd
    mf.grids.level = grid_level
    mf.conv_tol = CONV_TOL
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 200
    return mf


# --------------------------------------------------------------------------
# magnetization diagnostics on the DFT grid
# --------------------------------------------------------------------------
def m_diagnostics(mf, dm=None):
    """Return a dict:
        nelec, int_absm = int |m(r)|,
        net_m = |int m(r)| , ratio = net_m / int_absm  (~1 collinear, <1 not),
        w_noncol = fraction of int rho at points with |m|/rho < collinear_thrd
                   (i.e. the fraction actually sent through the Lebedev sphere),
        m_noncol = fraction of int |m| at those points.
    """
    mol = mf.mol
    if dm is None:
        dm = mf.make_rdm1()
    dm = cp.asarray(dm)
    nao = dm.shape[-1] // 2
    daa, dab = dm[:nao, :nao], dm[:nao, nao:]
    dbb = dm[nao:, nao:]
    ni = mf._numint
    g = mf.grids
    if g.coords is None:
        g.build()
    ao = cp.asarray(ni.eval_ao(mol, g.coords, deriv=0))
    w = cp.asarray(g.weights)

    def rr(d):
        return cp.einsum('gi,ij,gj->g', ao, cp.asarray(d), ao.conj())
    r_aa, r_bb = rr(daa).real, rr(dbb).real
    r_ab = rr(dab)
    mx = 2.0 * r_ab.real
    my = -2.0 * r_ab.imag
    mz = r_aa - r_bb
    rho = r_aa + r_bb
    s = cp.sqrt(mx**2 + my**2 + mz**2)
    thr = getattr(ni, 'collinear_thrd', 0.99)
    nelec = float((rho * w).sum().get())
    int_absm = float((s * w).sum().get())
    net = np.array([float((mx * w).sum().get()),
                    float((my * w).sum().get()),
                    float((mz * w).sum().get())])
    net_m = float(np.linalg.norm(net))
    # genuinely non-collinear points: |m| < thr*rho and rho not ~0
    mask = (s < thr * rho) & (rho > 1e-10)
    w_noncol = float(((rho * w)[mask].sum() / (rho * w).sum()).get())
    m_noncol = (float(((s * w)[mask].sum() / (s * w).sum()).get())
                if int_absm > 1e-12 else 0.0)
    return dict(nelec=nelec, int_absm=int_absm, net_m=net_m,
               ratio=net_m / int_absm if int_absm > 1e-12 else float('nan'),
               w_noncol=w_noncol, m_noncol=m_noncol, thr=thr)


# --------------------------------------------------------------------------
# PHASE A -- I atom: int|m|, non-collinearity, e_tot vs spin_samples
# --------------------------------------------------------------------------
def phaseA(ss_list=None):
    print('=== PHASE A: I atom 2P doublet, GKS(pbe0)+SOC ===', flush=True)
    if ss_list is None:
        ss_list = SS_LIST
    rows = []
    dm0 = None
    for ss in ss_list:
        mol = i_atom()
        mf = make_gks_soc(mol, ss)
        t0 = time.time()
        mf.kernel(dm0=dm0)
        dt = time.time() - t0
        assert mf.converged, f'ss={ss} not converged'
        d = m_diagnostics(mf)
        row = dict(spin_samples=ss, e_tot=float(mf.e_tot), walltime_s=dt, **d)
        rows.append(row)
        print(f'[A] ss={ss:5d}  e_tot={mf.e_tot:.10f}  int|m|={d["int_absm"]:.5f}  '
              f'ratio={d["ratio"]:.4f}  w_noncol={d["w_noncol"]:.3e}  '
              f'm_noncol={d["m_noncol"]:.3e}  ({dt:.0f}s)', flush=True)
        dm0 = mf.make_rdm1()   # warm start the next (converges the same state)

    e = np.array([r['e_tot'] for r in rows])
    print(f'[A] e_tot spread over spin_samples: max-min = {e.max()-e.min():.2e} Eh '
          f'= {(e.max()-e.min())*HARTREE2CM:.3f} cm^-1')
    ref = rows[-1]['e_tot']
    for r in rows:
        print(f'[A] ss={r["spin_samples"]:5d}  e_tot - e_tot(1202) = '
              f'{(r["e_tot"]-ref)*HARTREE2CM:+.4f} cm^-1')

    # spot-check grids.level 5 at ss=770
    mol = i_atom()
    mf = make_gks_soc(mol, 770, grid_level=5)
    mf.kernel()
    d5 = m_diagnostics(mf)
    e770_l3 = [r['e_tot'] for r in rows if r['spin_samples'] == 770]
    print(f'[A] grids.level=5, ss=770: e_tot={mf.e_tot:.10f}  int|m|={d5["int_absm"]:.5f}'
          + (f'  vs level3 {e770_l3[0]:.10f} (d={(mf.e_tot-e770_l3[0])*HARTREE2CM:+.3f} cm^-1)'
             if e770_l3 else ''), flush=True)

    json.dump(dict(rows=rows, level5_ss770=dict(e_tot=float(mf.e_tot), **d5)),
              open(f'{HERE}/_nc_phaseA.json', 'w'), indent=1)


# --------------------------------------------------------------------------
# PHASE B -- collinear_thrd on/off stress test
# --------------------------------------------------------------------------
def phaseB(ss_list=(50, 194, 434, 770, 1202)):
    print('=== PHASE B: I atom, collinear_thrd OFF (all points -> Lebedev) ===',
          flush=True)
    out = {}
    for thr_name, thr in (('default_0.99', 0.99), ('off_None', None)):
        rows = []
        dm0 = None
        for ss in ss_list:
            mol = i_atom()
            mf = make_gks_soc(mol, ss, collinear_thrd=thr)
            t0 = time.time()
            mf.kernel(dm0=dm0)
            dt = time.time() - t0
            assert mf.converged, f'{thr_name} ss={ss} not converged'
            rows.append(dict(spin_samples=ss, e_tot=float(mf.e_tot), walltime_s=dt))
            print(f'[B/{thr_name}] ss={ss:5d}  e_tot={mf.e_tot:.10f}  ({dt:.0f}s)',
                  flush=True)
            dm0 = mf.make_rdm1()
        e = np.array([r['e_tot'] for r in rows])
        print(f'[B/{thr_name}] e_tot spread = {(e.max()-e.min())*HARTREE2CM:.3f} cm^-1',
              flush=True)
        out[thr_name] = rows
    json.dump(out, open(f'{HERE}/_nc_phaseB.json', 'w'), indent=1)


# --------------------------------------------------------------------------
# PHASE C -- 2P3/2 - 2P1/2 splitting via non-aufbau dSCF
# --------------------------------------------------------------------------
def _run_excited(mol, ss, swap):
    """dSCF to the excited spinor occupation: from the converged ground state,
    move the hole from occupied spinor swap[0] to virtual spinor swap[1], then
    re-converge with a GPU-native maximum-overlap (MOM) occupation locked to
    that reference.  Returns mf or None."""
    mfg = make_gks_soc(mol, ss)
    mfg.kernel()
    if not mfg.converged:
        return None
    occ0 = cp.asarray(mfg.mo_occ).copy()
    i_occ, a_vir = swap
    if float(occ0[i_occ]) < 0.5 or float(occ0[a_vir]) > 0.5:
        return None
    occ_ref = occ0.copy()
    occ_ref[i_occ] = 0.0
    occ_ref[a_vir] = 1.0
    C = cp.asarray(mfg.mo_coeff)
    nocc = int(occ_ref.sum().get())
    Cref_occ = C[:, occ_ref > 0]                 # (2nao, nocc) reference occ block
    S = cp.asarray(mfg.get_ovlp())

    mf2 = make_gks_soc(mol, ss)

    def _mom_get_occ(mo_energy=None, mo_coeff=None):
        Cm = C if mo_coeff is None else cp.asarray(mo_coeff)
        proj = cp.abs(Cref_occ.conj().T @ S @ Cm) ** 2     # (nocc, nmo)
        score = proj.sum(axis=0)
        occ = cp.zeros(Cm.shape[1])
        idx = cp.argsort(score)[::-1][:nocc]
        occ[idx] = 1.0
        return occ

    mf2.get_occ = _mom_get_occ
    dm_exc = mfg.make_rdm1(mfg.mo_coeff, occ_ref)
    mf2.kernel(dm0=dm_exc)
    return mf2 if mf2.converged else None


def phaseC(ss_list=(50, 194, 434, 770, 1202)):
    print('=== PHASE C: I atom 2P3/2 - 2P1/2 dSCF splitting vs spin_samples ===',
          flush=True)
    rows = []
    for ss in ss_list:
        mol = i_atom()
        mfg = make_gks_soc(mol, ss)
        mfg.kernel()
        assert mfg.converged
        e_gs = float(mfg.e_tot)
        moe = np.sort(cp.asarray(mfg.mo_energy).get().real)
        nocc = int(cp.asarray(mfg.mo_occ).get().sum())
        # p-hole: 2P3/2 below 2P1/2.  The Kramers-degenerate spinors pair up;
        # the SOC gap is between the top occupied (j=1/2-ish) and the LUMO hole
        # (j=3/2-ish) pair.  Frozen-orbital (Koopmans) estimate:
        gap_orb = (moe[nocc] - moe[nocc-1]) * HARTREE2CM
        # move the hole from the LUMO (j=3/2 partner) to the HOMO (j=1/2): the
        # ground state has the p-hole in j=3/2; putting it in j=1/2 is 2P1/2
        mfe = _run_excited(mol, ss, swap=(nocc-1, nocc))
        if mfe is None:
            print(f'[C] ss={ss:5d}  dSCF to 2P1/2 did NOT converge; '
                  f'orbital gap near hole = {gap_orb:.1f} cm^-1', flush=True)
            rows.append(dict(spin_samples=ss, e_gs=e_gs, split_dscf=None,
                             gap_orb=gap_orb))
            continue
        split = (float(mfe.e_tot) - e_gs) * HARTREE2CM
        print(f'[C] ss={ss:5d}  E(2P1/2)-E(2P3/2) = {split:.1f} cm^-1  '
              f'(orbital gap {gap_orb:.1f}; exp 7603)', flush=True)
        rows.append(dict(spin_samples=ss, e_gs=e_gs, split_dscf=split,
                         gap_orb=gap_orb))
    json.dump(rows, open(f'{HERE}/_nc_phaseC.json', 'w'), indent=1)


# --------------------------------------------------------------------------
# PHASE D -- IO radical (secondary): FD stretch vs spin_samples at fixed geom
# --------------------------------------------------------------------------
def phaseD(ss_list=(50, 194, 434, 770), r=1.87):
    print('=== PHASE D: IO radical 2Pi, GKS(pbe0)+SOC ===', flush=True)
    from pyscf.hessian import thermo
    # try to converge the ground state once, robustly
    mol = io_mol(r)
    base = make_gks_soc(mol, 194)
    base.level_shift = 0.3
    try:
        base = base.newton()
    except Exception:
        pass
    base.kernel()
    print(f'[D] base SCF converged={base.converged}  e_tot={base.e_tot:.8f}  '
          f'int|m|={m_diagnostics(base)["int_absm"]:.4f}  '
          f'ratio={m_diagnostics(base)["ratio"]:.4f}', flush=True)
    if not base.converged:
        print('[D] IO ground-state SCF did not converge cleanly -- ABANDONED. '
              'The I atom is the load-bearing result.', flush=True)
        json.dump(dict(converged=False), open(f'{HERE}/_nc_phaseD.json', 'w'))
        return
    dm0 = base.make_rdm1()
    rows = []
    for ss in ss_list:
        mol = io_mol(r)
        mf = make_gks_soc(mol, ss)
        try:
            mf = mf.newton()
        except Exception:
            pass
        mf.level_shift = 0.3
        t0 = time.time()
        mf.kernel(dm0=dm0)
        dt = time.time() - t0
        if not mf.converged:
            print(f'[D] ss={ss} not converged; skipping', flush=True)
            continue
        H = finite_diff_hessian(mf, disp=1e-3)
        w = np.sort(thermo.harmonic_analysis(
            mol, H, exclude_trans=False, exclude_rot=False)['freq_wavenumber'].real)
        rows.append(dict(spin_samples=ss, e_tot=float(mf.e_tot),
                         stretch_cm=float(w[-1]), walltime_s=dt))
        print(f'[D] ss={ss:5d}  e_tot={mf.e_tot:.8f}  stretch={w[-1]:.2f} cm^-1  '
              f'({dt:.0f}s)', flush=True)
    json.dump(dict(converged=True, r=r, rows=rows),
              open(f'{HERE}/_nc_phaseD.json', 'w'), indent=1)


# --------------------------------------------------------------------------
# PHASE E -- simultaneous spatial+spin rotation invariance probe (I atom)
# --------------------------------------------------------------------------
def _wigner_d_l1(mol, R):
    """Block-diagonal l=1 rotation matrix on the AO basis for rotation `R`
    (3x3, real orthogonal); s/d/... blocks identity, p blocks rotated.
    p AO order in pyscf is (x, y, z)."""
    nao = mol.nao
    U = np.eye(nao)
    ao_l = []
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        nc = mol.bas_nctr(ib)
        for _ in range(nc):
            ao_l += [l] * (2 * l + 1)
    ao_l = np.array(ao_l)
    i = 0
    while i < nao:
        l = ao_l[i]
        if l == 1:
            U[i:i+3, i:i+3] = R          # (x,y,z) transform by R directly
        i += (2 * l + 1) if l != 1 else 3
        if l != 1:
            i += 0
    return U


def phaseE(thetas=None):
    print('=== PHASE E: simultaneous spatial+spin rotation invariance (I atom) ===',
          flush=True)
    from scipy.spatial.transform import Rotation
    if thetas is None:
        thetas = np.linspace(0, np.pi, 7)
    rows = []
    for ss in (50, 194, 770):
        mol = i_atom()
        mf = make_gks_soc(mol, ss)
        mf.kernel()
        assert mf.converged
        C = cp.asarray(mf.mo_coeff)
        occ = cp.asarray(mf.mo_occ)
        nao = C.shape[0] // 2
        e0 = float(mf.e_tot)

        def e_rot(theta):
            axis = np.array([0.0, 0.0, 1.0])
            R = Rotation.from_rotvec(theta * axis).as_matrix()
            U_sp = cp.asarray(_wigner_d_l1(mol, R))          # spatial, per spin block
            c, s = np.cos(theta / 2), np.sin(theta / 2)      # SU(2) about z
            Ca = U_sp @ C[:nao]
            Cb = U_sp @ C[nao:]
            Cr = cp.empty_like(C)
            Cr[:nao] = np.exp(-1j * theta / 2) * Ca
            Cr[nao:] = np.exp(1j * theta / 2) * Cb
            Cocc = Cr[:, occ > 0]
            dm = (Cocc * occ[occ > 0]) @ Cocc.conj().T
            h1e = cp.asarray(mf.get_hcore())
            vhf = mf.get_veff(mf.mol, dm)
            e_elec, _ = mf.energy_elec(dm, h1e, vhf)
            return float(e_elec + mf.energy_nuc())

        de = np.array([e_rot(t) - e0 for t in thetas])
        print(f'[E] ss={ss:5d}  max|dE(theta)| = {np.abs(de).max():.3e} Eh  '
              f'= {np.abs(de).max()*HARTREE2CM:.4f} cm^-1', flush=True)
        rows.append(dict(spin_samples=ss, max_dE=float(np.abs(de).max()),
                         de=de.tolist()))
    json.dump(dict(thetas=list(thetas), rows=rows),
              open(f'{HERE}/_nc_phaseE.json', 'w'), indent=1)


if __name__ == '__main__':
    ph = sys.argv[1] if len(sys.argv) > 1 else 'phaseA'
    t0 = time.time()
    {'phaseA': phaseA, 'phaseB': phaseB, 'phaseC': phaseC,
     'phaseD': phaseD, 'phaseE': phaseE}[ph]()
    print(f'[{ph}] done in {time.time()-t0:.0f}s', flush=True)
