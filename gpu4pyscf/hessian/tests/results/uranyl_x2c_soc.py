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

"""Go/no-go for the all-electron X2C-SOC route on actinides.

Fixed-geometry GKS + X2C-SOC single points on uranyl (UO2 2+) and ThO.

    python uranyl_x2c_soc.py phase0     # basis / build / dimensions
    python uranyl_x2c_soc.py phase1     # X2C-SOC SCF: converge? aids? wall profile
    python uranyl_x2c_soc.py phase2     # GPU vs CPU; SOC energy contribution; reduction
    python uranyl_x2c_soc.py phase3     # ThO corroboration
    python uranyl_x2c_soc.py phase4     # X2C-SOC PES scan r(U=O) = 1.66/1.68/1.70

Contrast: 2c-GKS + the actinide SO-ECP `ecpds60mwbso` collapses ~5 Eh
(`uranyl_soc_2a.md`).  X2C's 2c operator is variationally bounded, so this
should converge to a physical energy *below* the scalar-relativistic value by
the SOC stabilisation (order ~1 Eh for all-electron U).
"""

import sys
import time
import json
import numpy as np

from pyscf import gto

HERE = __file__.rsplit('/', 1)[0]

U_BASIS = 'ano-rcc-vdzp'      # Roos relativistic AE; the standard actinide AE set
O_BASIS = 'ano-rcc-vdzp'
XC = 'pbe0'
R_UO = 1.6791                 # Stage 2a scalar-ECP uranyl geometry, fixed
R_THO = 1.840
HARTREE2EV = 27.211386245988


def uranyl(r=R_UO, basis_u=U_BASIS, basis_o=O_BASIS):
    return gto.M(atom=f'U 0 0 0; O 0 0 {r}; O 0 0 -{r}', charge=2, spin=0,
                 basis={'U': basis_u, 'O': basis_o}, unit='Angstrom',
                 verbose=0, output='/dev/null')


def tho(r=R_THO):
    return gto.M(atom=f'Th 0 0 0; O 0 0 {r}', charge=0, spin=0,
                 basis={'Th': 'ano-rcc-vdzp', 'O': O_BASIS}, unit='Angstrom',
                 verbose=0, output='/dev/null')


def _nprim(mol):
    """Uncontracted (primitive) AO count -- the dimension the X2C decoupling
    eigenproblem actually runs at."""
    b = {}
    for a in {mol.atom_symbol(i) for i in range(mol.natm)}:
        b[a] = [[sh[0]] + [[p[0], 1.0] for p in sh[1:]]
                for sh in gto.uncontract(gto.basis.load(
                    U_BASIS if a in ('U', 'Th') else O_BASIS, a))]
    m = gto.M(atom=mol.atom, charge=mol.charge, spin=mol.spin, unit='Bohr'
              if False else 'Angstrom', basis=b, verbose=0)
    return m.nao


# --------------------------------------------------------------------------
def _gks_x2c(mol, seed_dm=None):
    from gpu4pyscf import dft
    mf = dft.GKS(mol, xc=XC).x2c1e()
    mf.spin_samples = 50
    mf.conv_tol = 1e-9
    return mf


def _rks_seed_dm2c(mol):
    """Converged non-relativistic RKS density -> block-diagonal 2c DM."""
    import cupy as cp
    from gpu4pyscf import dft
    r = dft.RKS(mol, xc=XC); r.conv_tol = 1e-9
    r.kernel()
    D = cp.asarray(r.make_rdm1())
    nao = mol.nao
    dm2 = cp.zeros((2 * nao, 2 * nao), dtype=complex)
    dm2[:nao, :nao] = D / 2
    dm2[nao:, nao:] = D / 2
    return float(r.e_tot), bool(r.converged), dm2


# --------------------------------------------------------------------------
def phase0():
    print('=== PHASE 0: basis / build / dimensions ===', flush=True)
    rows = {}
    for name, m in [('uranyl', uranyl()), ('ThO', tho())]:
        rows[name] = dict(has_ecp=bool(m.has_ecp()), nelec=int(m.nelectron),
                          nao=int(m.nao), nprim=int(_nprim(m)))
        print(f'  {name}: has_ecp={bool(m.has_ecp())} nelec={m.nelectron} '
              f'nao(contracted)={m.nao}  nao(primitive)={rows[name]["nprim"]}',
              flush=True)
    # what AE relativistic bases cover U / Th / O here
    avail = {}
    for b in ['ano-rcc-vdzp', 'ano-rcc', 'x2c-svpall', 'x2c-tzvpall',
              'sarc-dkh2', 'cc-pvdz-dk', 'dyall-v2z', 'dzp-dkh', 'tzp-dkh']:
        got = []
        for el in ('U', 'Th', 'O'):
            try:
                gto.basis.load(b, el); got.append(el)
            except Exception:
                pass
        avail[b] = got
    print('  AE relativistic basis coverage:', flush=True)
    for b, g in avail.items():
        print(f'    {b:14s}: {g}', flush=True)
    json.dump(dict(systems=rows, basis_coverage=avail,
                   basis_used=dict(U=U_BASIS, O=O_BASIS)),
              open(f'{HERE}/_x2c_phase0.json', 'w'), indent=1)


# --------------------------------------------------------------------------
def phase1():
    print('=== PHASE 1: uranyl GKS + X2C-SOC SCF ===', flush=True)
    import cupy as cp
    mol = uranyl()

    # non-relativistic RKS bracket + seed
    t0 = time.time()
    e_rks, rks_conv, dm2 = _rks_seed_dm2c(mol)
    t_seed = time.time() - t0
    print(f'[P1] nonrel RKS   e_tot = {e_rks:.8f}  conv={rks_conv}  ({t_seed:.0f}s)',
          flush=True)

    mf = _gks_x2c(mol)

    # time the one-time X2C get_hcore build in isolation
    cp.cuda.Stream.null.synchronize()
    t0 = time.time()
    h = mf.get_hcore()
    cp.cuda.Stream.null.synchronize()
    t_hcore = time.time() - t0
    print(f'[P1] X2C get_hcore build: {t_hcore:.1f}s   shape {tuple(h.shape)} '
          f'dtype {h.dtype}   finite={bool(cp.isfinite(h).all())}', flush=True)

    # SCF, seeded
    t0 = time.time()
    mf.kernel(dm0=dm2)
    t_scf = time.time() - t0
    n_it = len(mf.scf_summary.get('e_tot_hist', [])) or getattr(mf, 'cycles', -1)
    print(f'[P1] X2C-SOC SCF  e_tot = {mf.e_tot:.8f}  conv={mf.converged}  '
          f'({t_scf:.0f}s, ~{mf.max_cycle if not mf.converged else "?"} cyc)',
          flush=True)

    # try plain minao guess too (does it need the seed?)
    mf_plain = _gks_x2c(mol); mf_plain.max_cycle = 60
    t0 = time.time()
    try:
        mf_plain.kernel()
        plain_conv, plain_e = bool(mf_plain.converged), float(mf_plain.e_tot)
    except Exception as e:
        plain_conv, plain_e = False, repr(e)
    t_plain = time.time() - t0
    print(f'[P1] X2C-SOC SCF (minao, no seed): conv={plain_conv} e={plain_e} '
          f'({t_plain:.0f}s)', flush=True)

    # final tight number
    mf.conv_tol = 1e-11
    t0 = time.time()
    mf.kernel(dm0=mf.make_rdm1())
    t_tight = time.time() - t0
    e_x2c = float(mf.e_tot)

    d = dict(basis=U_BASIS, nao=int(mol.nao), nprim=int(_nprim(mol)),
             e_rks_nonrel=e_rks, e_x2c_soc=e_x2c, converged=bool(mf.converged),
             plain_minao_converged=plain_conv,
             t_rks_seed_s=t_seed, t_get_hcore_s=t_hcore, t_scf_s=t_scf,
             t_scf_tight_s=t_tight,
             mo_energy=cp.asnumpy(mf.mo_energy).real[:8].tolist())
    json.dump(d, open(f'{HERE}/_x2c_phase1.json', 'w'), indent=1)
    print(f'[P1] FINAL X2C-SOC e_tot = {e_x2c:.10f}  (tight, {t_tight:.0f}s)',
          flush=True)


# --------------------------------------------------------------------------
def _x2c_hcore_cpu(mol):
    """pyscf CPU spin-orbital X2C get_hcore (the 2c relativistic 1e operator)."""
    from pyscf.x2c import x2c as x2c_cpu
    h = x2c_cpu.SpinOrbitalX2CHelper(mol)
    return np.asarray(h.get_hcore(mol))


def phase2():
    """GPU-native cross-checks: X2C get_hcore GPU vs CPU (the actinide-specific
    new path), then the spin-free comparisons for the SOC energy contribution
    and the reduction check.  The full CPU X2C-SOC *SCF* cross-check is `phase2cpu`
    (separate process -- pyscf's OpenMP SCF segfaults under login-node thread
    exhaustion and must not take the rest of the phase down with it)."""
    print('=== PHASE 2: GPU-native cross-checks ===', flush=True)
    import cupy as cp
    from gpu4pyscf import dft
    from gpu4pyscf.x2c.x2c import SpinOrbitalX2CHelper as GpuX2C
    mol = uranyl()
    _, _, dm2 = _rks_seed_dm2c(mol)

    # ---- 2.0  GPU X2C-SOC SCF ----
    mf = _gks_x2c(mol); mf.conv_tol = 1e-10
    t0 = time.time(); mf.kernel(dm0=dm2); t_gpu = time.time() - t0
    e_gpu = float(mf.e_tot)
    print(f'[P2.0] GPU X2C-SOC e_tot = {e_gpu:.10f}  conv={mf.converged}  ({t_gpu:.0f}s)',
          flush=True)
    res = dict(e_gpu_x2c_soc=e_gpu, t_gpu_scf_s=t_gpu)

    # ---- 2.1  X2C get_hcore: GPU vs CPU pyscf (element-wise) ----
    t0 = time.time(); h_gpu = cp.asnumpy(GpuX2C(mol).get_hcore(mol)); t_hg = time.time() - t0
    t0 = time.time(); h_cpu = _x2c_hcore_cpu(mol); t_hc = time.time() - t0
    dh = float(abs(h_gpu - h_cpu).max())
    rel = dh / float(abs(h_cpu).max())
    print(f'[P2.1] X2C get_hcore  GPU {t_hg:.1f}s / CPU {t_hc:.1f}s   shape {h_gpu.shape}',
          flush=True)
    print(f'[P2.1] |h_x2c GPU - CPU|_max = {dh:.2e}   (rel to |h|_max = {rel:.1e})',
          flush=True)
    res.update(d_x2c_hcore_gpu_cpu=dh, d_x2c_hcore_rel=rel,
               t_hcore_gpu_s=t_hg, t_hcore_cpu_s=t_hc)

    # ---- 2.2  spin-free X2C-1e reference (1-component scalar relativistic) ----
    from gpu4pyscf.x2c.sfx2c1e import SpinFreeX2CHelper
    rks_sf = dft.RKS(mol, xc=XC).sfx2c1e()
    rks_sf.conv_tol = 1e-10
    t0 = time.time(); rks_sf.kernel(); t_sf = time.time() - t0
    e_sf_rks = float(rks_sf.e_tot)
    print(f'[P2.2] spin-free X2C-1e RKS (scalar rel) e_tot = {e_sf_rks:.10f}  ({t_sf:.0f}s)',
          flush=True)

    # ---- 2.3  reduction: 2c GKS fed the spin-free X2C hcore == scalar RKS ----
    # GHF.sfx2c1e is NotImplemented; build the block-diagonal spin-free hcore
    # by hand and hand it to a plain 2c GKS.
    h_sf = cp.asarray(SpinFreeX2CHelper(mol).get_hcore(mol))
    from gpu4pyscf.lib.cupy_helper import block_diag as _bd
    h_sf_2c = _bd([h_sf, h_sf])
    gks_sf = dft.GKS(mol, xc=XC)
    gks_sf.spin_samples = 50
    gks_sf.conv_tol = 1e-10
    gks_sf.get_hcore = lambda *a, **k: h_sf_2c
    gks_sf.kernel(dm0=dm2)
    e_sf_gks = float(gks_sf.e_tot)
    red = abs(e_sf_gks - e_sf_rks)
    print(f'[P2.3] 2c GKS + spin-free X2C hcore e_tot = {e_sf_gks:.10f}  '
          f'(reduction to scalar RKS |Δ| = {red:.2e})', flush=True)

    # ---- 2.2 (clean)  SOC energy contribution, within one 2c framework ----
    e_soc_clean = e_gpu - e_sf_gks          # SO w-term on vs off, same GKS/decoupling
    e_soc_naive = e_gpu - e_sf_rks          # vs sfX2C-1e (mixes the decoupling-scheme diff)
    print(f'[P2.2] SOC energy contribution (2c, w-term on-off) = {e_soc_clean:+.6f} Eh '
          f'({e_soc_clean*HARTREE2EV:+.2f} eV)', flush=True)
    print(f'[P2.2] E(X2C-2c) - E(sfX2C-1e)                     = {e_soc_naive:+.6f} Eh '
          f'(includes sfX2C<->X2C decoupling-scheme diff)', flush=True)
    res.update(e_sf_rks=e_sf_rks, e_sf_gks=e_sf_gks, reduction_dE=red,
               e_soc_contribution_clean=e_soc_clean, e_soc_naive=e_soc_naive)
    json.dump(res, open(f'{HERE}/_x2c_phase2.json', 'w'), indent=1)


def phase2soc():
    """The physically meaningful SOC energy: X2C-1e 2c with the spin-orbit
    sigma.(pV x p) term ON vs OFF, in the *same* 2c decoupling scheme.  (P2.2's
    'E(X2C-2c) - E(sfX2C-1e)' also folds in the sfX2C<->X2C decoupling-scheme
    difference, which for U is tens of Eh.)"""
    print('=== PHASE 2-SOC: within-scheme SOC energy (w spin term on/off) ===',
          flush=True)
    from gpu4pyscf import dft
    from gpu4pyscf.x2c.x2c import SpinOrbitalX2CHelper as GpuX2C
    import gpu4pyscf.x2c.x2c as _xm
    mol = uranyl()
    _, _, dm2 = _rks_seed_dm2c(mol)
    d = json.load(open(f'{HERE}/_x2c_phase2.json'))
    e_soc_on = d['e_gpu_x2c_soc']

    # Same 2c decoupling path, but zero the spin-orbit part of the pVp 'w' term
    # (int1e_spnucsp is (4,n,n) = [x,y,z,scalar]; keep only index 3).
    real_sigma = _xm._sigma_dot

    def fake_sigma(mat):
        if getattr(mat, 'ndim', 0) == 3 and mat.shape[0] == 4:
            z = np.zeros_like(mat)
            z[3] = mat[3]
            return real_sigma(z)
        return real_sigma(mat)

    _xm._sigma_dot = fake_sigma
    try:
        h_sf = GpuX2C(mol).get_hcore(mol)
    finally:
        _xm._sigma_dot = real_sigma

    mf = dft.GKS(mol, xc=XC)
    mf.spin_samples = 50
    mf.conv_tol = 1e-10
    mf.get_hcore = lambda *a, **k: h_sf
    mf.kernel(dm0=dm2)
    e_soc_off = float(mf.e_tot)
    e_soc = e_soc_on - e_soc_off
    print(f'[P2soc] E(X2C-1e 2c, SO w ON)  = {e_soc_on:.10f}', flush=True)
    print(f'[P2soc] E(X2C-1e 2c, SO w OFF) = {e_soc_off:.10f}  conv={mf.converged}',
          flush=True)
    print(f'[P2soc] within-scheme SOC energy = {e_soc:+.6f} Eh ({e_soc*HARTREE2EV:+.2f} eV)',
          flush=True)
    d.update(e_x2c_so_off_inscheme=e_soc_off, e_soc_within_scheme=e_soc)
    json.dump(d, open(f'{HERE}/_x2c_phase2.json', 'w'), indent=1)


def phase2cpu():
    """Full CPU pyscf X2C-SOC SCF cross-check (own process; OMP-capped).
    Merges its result into _x2c_phase2.json."""
    print('=== PHASE 2-CPU: full pyscf X2C-SOC SCF cross-check ===', flush=True)
    import mcfun  # noqa: F401
    from pyscf import dft as cdft, scf as cscf
    d = json.load(open(f'{HERE}/_x2c_phase2.json'))
    e_gpu = d['e_gpu_x2c_soc']
    ref = cdft.GKS(uranyl(), xc=XC).x2c1e()
    ref.collinear = 'mcol'
    try:
        ref.spin_samples = 50
    except Exception:
        pass
    ref.conv_tol = 1e-10
    t0 = time.time(); ref.kernel(); t_cpu = time.time() - t0
    e_cpu = float(ref.e_tot)
    de = abs(e_gpu - e_cpu)
    print(f'[P2cpu] CPU X2C-SOC  e_tot = {e_cpu:.10f}  conv={ref.converged}  ({t_cpu:.0f}s)',
          flush=True)
    print(f'[P2cpu] |e_gpu - e_cpu| = {de:.2e}', flush=True)

    # CPU spin-free X2C-1e for the same naive comparison
    sf = cdft.RKS(uranyl(), xc=XC).sfx2c1e(); sf.conv_tol = 1e-10
    sf.kernel()
    e_cpu_sf = float(sf.e_tot)
    print(f'[P2cpu] CPU sfX2C-1e RKS e_tot = {e_cpu_sf:.10f}   '
          f'E(X2C-2c)-E(sfX2C-1e) = {e_cpu - e_cpu_sf:+.4f} Eh', flush=True)

    d.update(e_cpu_x2c_soc=e_cpu, t_cpu_scf_s=t_cpu, d_etot_gpu_cpu=de,
             cpu_scf_converged=bool(ref.converged), e_cpu_sf_rks=e_cpu_sf)
    json.dump(d, open(f'{HERE}/_x2c_phase2.json', 'w'), indent=1)


# --------------------------------------------------------------------------
def phase3():
    print('=== PHASE 3: ThO corroboration ===', flush=True)
    import cupy as cp
    from gpu4pyscf import dft
    mol = tho()
    print(f'[P3] ThO nelec={mol.nelectron} nao={mol.nao} nprim={_nprim(mol)}',
          flush=True)
    _, _, dm2 = _rks_seed_dm2c(mol)

    mf = _gks_x2c(mol); mf.conv_tol = 1e-10
    t0 = time.time(); h = mf.get_hcore(); t_h = time.time() - t0
    t0 = time.time(); mf.kernel(dm0=dm2); t_scf = time.time() - t0
    e_gpu = float(mf.e_tot)
    mo_gpu = cp.asnumpy(mf.mo_energy).real
    print(f'[P3] GPU X2C-SOC e_tot = {e_gpu:.10f}  conv={mf.converged}  '
          f'(hcore {t_h:.0f}s, scf {t_scf:.0f}s)', flush=True)

    rks_sf = dft.RKS(mol, xc=XC).sfx2c1e(); rks_sf.conv_tol = 1e-10
    rks_sf.kernel()
    e_sf = float(rks_sf.e_tot)
    e_soc = e_gpu - e_sf
    print(f'[P3] scalar sf-X2C RKS e_tot = {e_sf:.10f}   SOC contribution = '
          f'{e_soc:.6f} Eh ({e_soc*HARTREE2EV:.2f} eV)', flush=True)

    res = dict(nao=int(mol.nao), nprim=int(_nprim(mol)), e_gpu_x2c_soc=e_gpu,
               e_sf_rks=e_sf, e_soc_contribution=e_soc, converged=bool(mf.converged),
               t_get_hcore_s=t_h, t_scf_s=t_scf)
    try:
        import mcfun  # noqa: F401
        from pyscf import dft as cdft
        ref = cdft.GKS(tho(), xc=XC).x2c1e(); ref.collinear = 'mcol'
        try:
            ref.spin_samples = 50
        except Exception:
            pass
        ref.conv_tol = 1e-10
        t0 = time.time(); ref.kernel(); t_cpu = time.time() - t0
        de = abs(e_gpu - float(ref.e_tot))
        dmo = float(abs(np.sort(mo_gpu) - np.sort(np.asarray(ref.mo_energy).real)).max())
        print(f'[P3] CPU X2C-SOC e_tot = {ref.e_tot:.10f} ({t_cpu:.0f}s)  '
              f'|Δe|={de:.2e} |Δmo|={dmo:.2e}', flush=True)
        res.update(e_cpu_x2c_soc=float(ref.e_tot), d_etot_gpu_cpu=de,
                   d_mo_energy_gpu_cpu=dmo, t_cpu_scf_s=t_cpu)
    except ImportError:
        res['cpu_cross_check'] = 'skipped (no mcfun)'
    json.dump(res, open(f'{HERE}/_x2c_phase3.json', 'w'), indent=1)


# --------------------------------------------------------------------------
def _parab_min(rr, ee):
    c = np.polyfit(np.asarray(rr, float), np.asarray(ee, float), 2)
    return -c[1] / (2 * c[0]), 2 * c[0]


def phase4():
    """X2C-SOC and spin-free-X2C single points at r(U=O) = 1.66/1.68/1.70:
    is the X2C-SOC PES smooth, and does the geometry-independent core SOC
    (+29 Eh) cancel in the bond-coordinate derivative, leaving a physical
    SOC force / bond-length shift?"""
    print('=== PHASE 4: X2C-SOC + spin-free PES scan r(U=O) ===', flush=True)
    import cupy as cp
    from gpu4pyscf import dft
    RS = (1.66, 1.68, 1.70)
    so, sf = {}, {}
    dm2 = None
    for r in RS:
        mol = uranyl(r=r)
        if dm2 is None:
            _, _, dm2 = _rks_seed_dm2c(mol)
        mf = _gks_x2c(mol); mf.conv_tol = 1e-10
        t0 = time.time(); mf.kernel(dm0=dm2)
        so[r] = dict(e_tot=float(mf.e_tot), conv=bool(mf.converged),
                     wall_s=time.time() - t0)
        dm2 = cp.asarray(mf.make_rdm1())
        msf = dft.RKS(mol, xc=XC).sfx2c1e(); msf.conv_tol = 1e-10
        msf.kernel()
        sf[r] = float(msf.e_tot)
        print(f'[P4] r={r:.2f}  X2C-SOC={so[r]["e_tot"]:.10f}  '
              f'sfX2C={sf[r]:.10f}  E_SOC={so[r]["e_tot"]-sf[r]:+.6f}', flush=True)
    e_soc = {r: so[r]['e_tot'] - sf[r] for r in RS}
    rmin_so, k_so = _parab_min(RS, [so[r]['e_tot'] for r in RS])
    rmin_sf, k_sf = _parab_min(RS, [sf[r] for r in RS])
    slope_soc = (e_soc[1.70] - e_soc[1.66]) / 0.04
    print(f'[P4] min(sfX2C) ~ {rmin_sf:.4f} A   min(X2C-SOC) ~ {rmin_so:.4f} A   '
          f'SOC shift {(rmin_so-rmin_sf)*1e3:+.1f} mA', flush=True)
    print(f'[P4] dE_SOC/dr ~ {slope_soc:+.4f} Eh/A   (k_sf ~ {k_sf:.2f} Eh/A^2)',
          flush=True)
    json.dump(dict(x2c_soc=so, sfx2c=sf, e_soc=e_soc,
                   r_min_sfx2c=float(rmin_sf), r_min_x2c_soc=float(rmin_so),
                   soc_bond_shift_ang=float(rmin_so - rmin_sf),
                   dE_soc_dr=float(slope_soc)),
              open(f'{HERE}/_x2c_phase4.json', 'w'), indent=1)


if __name__ == '__main__':
    ph = sys.argv[1] if len(sys.argv) > 1 else 'phase0'
    t0 = time.time()
    {'phase0': phase0, 'phase1': phase1, 'phase2': phase2, 'phase2soc': phase2soc,
     'phase2cpu': phase2cpu, 'phase3': phase3, 'phase4': phase4}[ph]()
    print(f'[{ph}] done in {time.time() - t0:.0f}s', flush=True)
