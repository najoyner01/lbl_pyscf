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

r"""Analytic nuclear gradient of the one-electron X2C spin-orbit core
Hamiltonian, under the **atomic (local) X2C approximation** (``approx='atom1e'``).

Scope (fixed -- do not widen here)
----------------------------------
* **Atomic X2C only.**  Each atomic decoupling block ``X_A`` / ``R_A`` is solved
  from atom ``A``'s own basis at its own origin, so ``X_A`` and ``R_A`` are
  geometry independent (``dX_A/dR_B = 0``).  The geometry dependence of
  ``h_X2C`` then comes entirely from the underlying one-electron integral
  derivatives (V, T, S, W = sigma.pVp) plus the fixed linear-algebra recipe
  that places and combines the atomic blocks -- whose derivative is closed
  form, with no response (Sylvester / CPHF) solve.  The full *molecular*
  decoupling response ``dX/dR`` is a separate later project and is **not**
  attempted here.
* **One-electron X2C-SOC only.**  No two-electron spin-orbit, no
  SNSO/Boettger, no AMFI screening (a parallel workstream).  Bare X2C-1e
  over-estimates absolute SOC *energies* (e.g. +29 Eh core term on uranyl,
  ``hessian/tests/results/uranyl_x2c_soc.md``) but that term is
  geometry-independent to ~99.6 % and cancels in ``dE/dR``; the analytic
  gradient computes the ~0.1 Eh/A physically meaningful part cleanly.
* Molecular (non-PBC).  Combines with the existing ``grad/gks.py`` XC +
  hybrid-K gradient and PCM unchanged -- the only new term is this hcore
  derivative.
* The ``pVp`` / ``pVxp`` derivative integrals (``int1e_ipspnucsp`` etc.) are
  evaluated with CPU PySCF.  A GPU kernel for them is a later scale-up item.

Method
------
Structural port of ``pyscf/x2c/sfx2c1e_grad.py`` (JCP 135, 084114 (2011))
from the spin-free scalar case to the spin-orbital (complex, ``sigma.pVp``)
case.  In the "half-transformed" formulation the atomic-approximation X2C core
Hamiltonian is

    h_X2C = c_fw0^dag @ h0 @ c_fw0 ,
    c_fw0 = [ R0 ; X0 @ R0 ] ,
    h0    = [[ V,  T          ],
            [ T,  W/4c^2 - T  ]] ,

with X0 the geometry-independent block-atomic decoupling matrix and
R0 = _get_r(S, S + X0^dag T X0 / 2c^2) the renormalisation.  Its derivative
w.r.t. the position of atom ``ia`` is

    dh = dc_fw0^dag @ h0 @ c_fw0  +  h.c.  +  c_fw0^dag @ dh0 @ c_fw0 ,

where ``dh0`` is assembled from the derivative integral blocks (dV, dT, dS,
dW with the per-atom rinv Hellmann-Feynman pieces) and ``dc_fw0`` needs only
``dR0`` (``dX0 = 0`` in the atomic approximation), obtained in closed form
from ``_get_r1`` (the spectral divided-difference derivative of ``_get_r``).

Validation: ``grad/tests/test_x2c_soc_grad.py`` and
``hessian/tests/results/x2c_soc_grad_A100.md``.
"""

from functools import reduce
import numpy as np
import scipy.linalg
import cupy as cp
from pyscf import lib
from pyscf.x2c import x2c as x2c_cpu
from pyscf.x2c.x2c import (_block_diag, _sigma_dot, _get_r,
                           _spin_orbital_atomic_1e_x)
from gpu4pyscf.lib import logger
from gpu4pyscf.grad.rhf import contract_h1e_dm

__all__ = ['hcore_deriv_generator', 'hcore_grad_energy', 'Gradients', 'Grad']

LIGHT_SPEED = lib.param.LIGHT_SPEED


# ---------------------------------------------------------------------------
# zeroth-order block operators
# ---------------------------------------------------------------------------
def _get_h0_s0(xmol, with_soc=True):
    """h0, s0 : (4n, 4n) complex, n = spin-orbital nao of the (decontracted) xmol.

    h0 = [[V, T], [T, W/4c^2 - T]] ,  s0 = [[S, 0], [0, T/2c^2]] .
    ``with_soc=False`` keeps only the scalar (spin-free) part of W.
    """
    c = LIGHT_SPEED
    s = _block_diag(xmol.intor_symmetric('int1e_ovlp'))
    t = _block_diag(xmol.intor_symmetric('int1e_kin'))
    v = _block_diag(xmol.intor_symmetric('int1e_nuc'))
    w = _sigma_dot(_maybe_drop_soc(xmol.intor('int1e_spnucsp'), with_soc))
    n = s.shape[0]
    n2 = n * 2
    h = np.zeros((n2, n2), dtype=np.complex128)
    m = np.zeros((n2, n2), dtype=np.complex128)
    h[:n, :n] = v
    h[:n, n:] = t
    h[n:, :n] = t
    h[n:, n:] = w * (.25 / c**2) - t
    m[:n, :n] = s
    m[n:, n:] = t * (.5 / c**2)
    return h, m


def _maybe_drop_soc(spnucsp, with_soc):
    """int1e_spnucsp is (4, nao, nao) = [sigma_x, sigma_y, sigma_z, scalar];
    with_soc=False zeros the three spin-dependent components, leaving the
    spin-free pVp operator that reproduces sfx2c1e."""
    if with_soc:
        return spnucsp
    out = np.zeros_like(spnucsp)
    out[3] = spnucsp[3]
    return out


def _block_diag_stack(a):
    """(k, nsp, nsp) -> (k, 2nsp, 2nsp), block-diagonal (alpha, beta) per k."""
    return np.stack([_block_diag(x) for x in a])


def _sigma_dot_stack(a, with_soc=True):
    """(3*4, nsp, nsp) [deriv-component x quaternion] -> (3, 2nsp, 2nsp) complex."""
    nsp = a.shape[-1]
    a = a.reshape(3, 4, nsp, nsp)
    return np.stack([_sigma_dot(_maybe_drop_soc(a[i], with_soc)) for i in range(3)])


# ---------------------------------------------------------------------------
# first-order block operators (derivative integrals for atom ``ia``)
# ---------------------------------------------------------------------------
def _gen_h1_s1(xmol, with_soc=True):
    c = LIGHT_SPEED
    s1 = _block_diag_stack(xmol.intor('int1e_ipovlp', comp=3))
    t1 = _block_diag_stack(xmol.intor('int1e_ipkin', comp=3))
    v1 = _block_diag_stack(xmol.intor('int1e_ipnuc', comp=3))
    w1 = _sigma_dot_stack(xmol.intor('int1e_ipspnucsp', comp=12), with_soc)

    aoslices = xmol.aoslice_by_atom()
    nsp = s1.shape[1] // 2
    n = s1.shape[1]
    n2 = n * 2

    def get_h1_s1(ia):
        h1 = np.zeros((3, n2, n2), dtype=np.complex128)
        m1 = np.zeros((3, n2, n2), dtype=np.complex128)
        p0, p1 = aoslices[ia, 2:]
        idx = np.hstack([np.arange(p0, p1), np.arange(p0, p1) + nsp])
        with xmol.with_rinv_origin(xmol.atom_coord(ia)):
            z = xmol.atom_charge(ia)
            rinv1 = -z * _block_diag_stack(xmol.intor('int1e_iprinv', comp=3))
            prinvp1 = -z * _sigma_dot_stack(
                xmol.intor('int1e_ipsprinvsp', comp=12), with_soc)
        # translational invariance: bra derivative of the total V minus this
        # atom's own contribution leaves the (rows on ia) piece; the Hellmann-
        # Feynman derivative of the operator centred on ia is the rinv term.
        rinv1[:, idx, :] -= v1[:, idx]
        prinvp1[:, idx, :] -= w1[:, idx]

        for i in range(3):
            s1cc = np.zeros((n, n), dtype=np.complex128)
            t1cc = np.zeros((n, n), dtype=np.complex128)
            s1cc[idx, :] = -s1[i, idx]
            s1cc[:, idx] -= s1[i, idx].conj().T
            t1cc[idx, :] = -t1[i, idx]
            t1cc[:, idx] -= t1[i, idx].conj().T
            v1cc = rinv1[i] + rinv1[i].conj().T
            w1cc = prinvp1[i] + prinvp1[i].conj().T

            h1[i, :n, :n] = v1cc
            h1[i, :n, n:] = t1cc
            h1[i, n:, :n] = t1cc
            h1[i, n:, n:] = w1cc * (.25 / c**2) - t1cc
            m1[i, :n, :n] = s1cc
            m1[i, n:, n:] = t1cc * (.5 / c**2)
        return h1, m1
    return get_h1_s1


# ---------------------------------------------------------------------------
# derivative of R = _get_r(S, S_nesc)  (spectral divided differences)
# ---------------------------------------------------------------------------
def _get_r1(s0_roots, s_nesc0, s1, s_nesc1, r0_roots):
    """Complex (Hermitian) port of pyscf/x2c/sfx2c1e_grad.py:_get_r1
    (JCP 135, 084114 (2011), Eq. 34).  Eigenvalues are real; every
    eigenvector transpose becomes a conjugate transpose.
    """
    w_sqrt, v_s = s0_roots
    w_invsqrt = 1. / w_sqrt
    wr0_sqrt, vr0 = r0_roots
    wr0_invsqrt = 1. / wr0_sqrt

    s1 = reduce(np.dot, (v_s.conj().T, s1, v_s))
    s_nesc1 = reduce(np.dot, (v_s.conj().T, s_nesc1, v_s))

    s1_sqrt = s1 / (w_sqrt[:, None] + w_sqrt)
    s1_invsqrt = (np.einsum('i,ij,j->ij', w_invsqrt**2, s1, w_invsqrt**2)
                  / -(w_invsqrt[:, None] + w_invsqrt))
    R1_mid = np.dot(s1_invsqrt, s_nesc0) * w_invsqrt
    R1_mid = R1_mid + R1_mid.conj().T
    R1_mid += np.einsum('i,ij,j->ij', w_invsqrt, s_nesc1, w_invsqrt)

    R1_mid = reduce(np.dot, (vr0.conj().T, R1_mid, vr0))
    R1_mid /= -(wr0_invsqrt[:, None] + wr0_invsqrt)
    R1_mid = np.einsum('i,ij,j->ij', wr0_invsqrt**2, R1_mid, wr0_invsqrt**2)
    vr0_wr0_sqrt = vr0 * wr0_invsqrt
    vr0_s0_sqrt = vr0.conj().T * w_sqrt
    vr0_s0_invsqrt = vr0.conj().T * w_invsqrt

    R1 = reduce(np.dot, (vr0_s0_invsqrt.conj().T, R1_mid, vr0_s0_sqrt))
    R1 += reduce(np.dot, (s1_invsqrt, vr0_wr0_sqrt, vr0_s0_sqrt))
    R1 += reduce(np.dot, (vr0_s0_invsqrt.conj().T, vr0_wr0_sqrt.conj().T, s1_sqrt))
    R1 = reduce(np.dot, (v_s, R1, v_s.conj().T))
    return R1


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------
def _gen_soc_hfw(xmol, contr_coeff, with_soc=True):
    """Returns ``hcore_deriv(ia) -> (3, 2*nao, 2*nao)`` complex numpy, the
    derivative of the contracted spin-orbital X2C-SOC core Hamiltonian
    w.r.t. the Cartesian coordinates of atom ``ia`` (atomic approximation)."""
    c = LIGHT_SPEED
    h0, s0 = _get_h0_s0(xmol, with_soc)
    n = h0.shape[0] // 2
    x0 = _spin_orbital_atomic_1e_x(xmol)
    if not with_soc:
        # drop the SO content of the atomic decoupling blocks too, so the
        # whole build reduces to spin-free X2C (V1 reduction check).
        x0 = _spin_free_x0(xmol)

    s0_ll = s0[:n, :n]
    t0 = _block_diag(xmol.intor_symmetric('int1e_kin'))
    t0x0 = np.dot(t0, x0) * (.5 / c**2)
    s_nesc0 = s0_ll + np.dot(x0.conj().T, t0x0)
    R0 = _get_r(s0_ll, s_nesc0)
    c_fw0 = np.vstack((R0, np.dot(x0, R0)))
    h0_fw_half = np.dot(h0, c_fw0)

    # spectral roots reused by every _get_r1 call
    w_s, v_s = scipy.linalg.eigh(s0_ll)
    w_sqrt = np.sqrt(w_s)
    s_nesc0_vbas = reduce(np.dot, (v_s.conj().T, s_nesc0, v_s))
    R0_mid = np.einsum('i,ij,j->ij', 1. / w_sqrt, s_nesc0_vbas, 1. / w_sqrt)
    wr0, vr0 = scipy.linalg.eigh(R0_mid)
    wr0_sqrt = np.sqrt(wr0)

    get_h1_s1 = _gen_h1_s1(xmol, with_soc)
    cc = _block_diag(contr_coeff) if contr_coeff is not None else None

    x0_dag = x0.conj().T
    c_fw0_dag = c_fw0.conj().T                       # (n, 2n)

    def hcore_deriv(ia):
        h1ao, s1ao = get_h1_s1(ia)
        # A @ M[x] @ B for a stack M[x]: broadcast the BLAS matmul over x.
        # (explicit @ -> O(n^3) BLAS; np.einsum with 3 operands and no
        # optimize= falls back to an O(n^4) python path -- ~40 min on HI.)
        s_nesc1 = (x0_dag @ s1ao[:, n:, n:]) @ x0    # (3, n, n)
        s_nesc1 = s_nesc1 + s1ao[:, :n, :n]

        c_fw1 = np.empty((3, 2 * n, n), dtype=np.complex128)
        for i in range(3):
            R1 = _get_r1((w_sqrt, v_s), s_nesc0_vbas,
                         s1ao[i, :n, :n], s_nesc1[i], (wr0_sqrt, vr0))
            c_fw1[i, :n] = R1
            c_fw1[i, n:] = np.dot(x0, R1)

        hfw1 = c_fw1.conj().transpose(0, 2, 1) @ h0_fw_half   # (3, n, n)
        hfw1 = hfw1 + hfw1.conj().transpose(0, 2, 1)
        hfw1 += (c_fw0_dag @ h1ao) @ c_fw0

        if cc is not None:
            hfw1 = (cc.conj().T @ hfw1) @ cc
        return hfw1
    return hcore_deriv


def _spin_free_x0(xmol):
    """Block-atomic X with the spin-orbit part of W removed -- the spin-free
    X2C decoupling matrix, embedded in the (alpha, beta) spin-orbital layout."""
    c = LIGHT_SPEED
    atoms = x2c_cpu._atoms_in_mole(xmol)
    x_conf = {}
    for elem, atom in atoms.items():
        t1 = _block_diag(atom.intor_symmetric('int1e_kin'))
        s1 = _block_diag(atom.intor_symmetric('int1e_ovlp'))
        v1 = _block_diag(atom.intor_symmetric('int1e_nuc'))
        w1 = _sigma_dot(_maybe_drop_soc(atom.intor('int1e_spnucsp'), False))
        x_conf[elem] = x2c_cpu._x2c1e_xmatrix(t1, v1, w1, s1, c)
    atom_slices = xmol.offset_nr_by_atom()
    nao = xmol.nao_nr()
    x = np.zeros((2, nao, 2, nao), dtype=np.complex128)
    for ia in range(xmol.natm):
        p0, p1 = atom_slices[ia, 2:]
        elem = xmol.atom_symbol(ia)
        x[:, p0:p1, :, p0:p1] = x_conf[elem].reshape(2, p1 - p0, 2, p1 - p0)
    return x.reshape(nao * 2, nao * 2)


def hcore_deriv_generator(mol, approx='atom1e', with_soc=True):
    """Public entry.  ``hcore_deriv(ia) -> cupy (3, 2*nao, 2*nao) complex128``.

    Only the atomic approximation is supported (see module docstring).
    """
    if 'ATOM' not in approx.upper():
        raise NotImplementedError(
            "grad/x2c.py implements only the atomic X2C approximation "
            f"(approx='atom1e'); got approx={approx!r}. The molecular "
            "decoupling response dX/dR is out of scope.")
    helper = x2c_cpu.SpinOrbitalX2CHelper(mol)
    helper.approx = approx
    xmol, contr_coeff = helper.get_xmol(mol)
    gen = _gen_soc_hfw(xmol, contr_coeff, with_soc)

    def hcore_deriv(ia):
        return cp.asarray(gen(ia))
    return hcore_deriv


# ---------------------------------------------------------------------------
# energy contribution -- slots into grad/ghf.py:grad_elec
# ---------------------------------------------------------------------------
def hcore_grad_energy(mf_grad, dm_ghf, dme_sf):
    """dE/dR from the X2C-SOC core Hamiltonian + the (unchanged, block-diagonal
    real) overlap Pulay term.

        de[A] = Re Tr[ dh_X2C/dR_A . D_spinor ]  -  Re Tr[ dS/dR_A . Dme ]

    Returns a numpy ``(natm, 3)`` array.  ``dm_ghf`` is the full complex
    (2*nao, 2*nao) GHF/GKS density matrix; ``dme_sf`` is the real (nao, nao)
    alpha+beta energy-weighted density (X2C leaves the SCF metric = the plain
    non-relativistic overlap, so the Pulay term is identical to plain GHF).
    """
    mf = mf_grad.base
    mol = mf_grad.mol
    log = logger.new_logger(mf_grad)
    t0 = log.init_timer()

    x2cobj = mf.with_x2c
    approx = getattr(x2cobj, 'approx', 'atom1e')
    with_soc = getattr(mf_grad, 'x2c_with_soc', True)
    hcore_deriv = hcore_deriv_generator(mol, approx=approx, with_soc=with_soc)

    dm = cp.asnumpy(cp.asarray(dm_ghf))
    de = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        dH = cp.asnumpy(hcore_deriv(ia))          # (3, 2nao, 2nao) complex
        de[ia] = np.einsum('xij,ji->x', dH, dm).real
    log.timer_debug1('X2C-SOC hcore gradient', *t0)

    # overlap Pulay term: metric is the plain non-relativistic overlap
    s1 = cp.asarray(mf_grad.get_ovlp(mol))
    de -= contract_h1e_dm(mol, s1, cp.asarray(dme_sf), hermi=1)
    return de


# ---------------------------------------------------------------------------
# Gradients wiring
# ---------------------------------------------------------------------------
def Gradients(mf):
    """Return the nuclear-gradient object for an ``x2c1e()`` GHF/GKS SCF.

    The X2C-SOC hcore-derivative term is dispatched inside
    ``grad/ghf.py:grad_elec`` on ``mf.with_x2c``; the returned object is the
    ordinary GKS (KS base) or GHF (HF base) ``Gradients`` otherwise unchanged
    -- J/K, XC + grid response, PCM all compose as-is.
    """
    from gpu4pyscf.dft.rks import KohnShamDFT
    if isinstance(mf, KohnShamDFT):
        from gpu4pyscf.grad import gks as _g
    else:
        from gpu4pyscf.grad import ghf as _g
    return _g.Gradients(mf)


Grad = Gradients
