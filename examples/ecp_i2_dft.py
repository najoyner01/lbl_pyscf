#!/usr/bin/env python
"""I2 / PBE0 / aug-cc-pVDZ-PP + ECP28MDF — a first ECP job on the GPU.

Iodine has no all-electron aug-cc-pVDZ in the correlation-consistent family: the
-PP sets come paired with the Stuttgart-Koeln ECP28MDF pseudopotential, which
replaces the 28 [Ar]3d core electrons. So I2 here is a 50-electron, 64-AO
system whose one-electron Hamiltonian *requires* the ECP integrals — exactly the
code path in gpu4pyscf/gto/ecp.py + lib/ecp/ (libgecp).

Usage:
    source env.sh
    python examples/ecp_i2_dft.py                 # SCF + gradient + ECP integral checks
    python examples/ecp_i2_dft.py --cpu           # ... and a full CPU SCF/gradient reference
    python examples/ecp_i2_dft.py --bond 2.70     # different bond length
"""

import argparse
import time

import numpy as np
import pyscf
from pyscf import gto

from gpu4pyscf.dft import rks
from gpu4pyscf.gto.ecp import get_ecp, get_ecp_ip, get_ecp_ipip


def iodine_aug_cc_pvdz_pp():
    """Return (orbital_basis, ecp) for I.

    PySCF ships aug-cc-pVDZ-PP.dat but it only covers Zn/Cu/Ag/Cd/Au/Hg, so
    `basis='aug-cc-pvdz-pp'` raises BasisNotFoundError for iodine. The name *is*
    in pyscf's ALIAS table, which short-circuits pyscf's own Basis Set Exchange
    fallback -- so fetch it from BSE explicitly. Same call pyscf makes
    internally (pyscf/gto/basis/__init__.py); requires `basis-set-exchange`.
    """
    import basis_set_exchange
    from pyscf.gto.basis import bse

    obj = basis_set_exchange.api.get_basis('aug-cc-pvdz-pp', elements=['I'])
    return bse._orbital_basis(obj)[0]['I'], bse._ecp_basis(obj)['I']


def build_mol(bond, verbose):
    orb, ecp = iodine_aug_cc_pvdz_pp()
    return gto.M(
        atom=f'I 0.0 0.0 0.0; I 0.0 0.0 {bond}',   # r_e(exp) = 2.666 A
        basis={'I': orb},
        ecp={'I': ecp},
        unit='Angstrom',
        verbose=verbose,
    )


def check_ecp_integrals(mol):
    """Compare the libgecp kernels against PySCF's CPU ECP integrals.

    Pairings and the 1e-8 Frobenius-norm threshold follow
    gpu4pyscf/gto/tests/test_ecp.py. The ip/ipip kernels need >=64 KB of shared
    memory per block (fine on A100's 164 KB).
    """
    checks = [
        ('<i|V_ecp|j>          ', mol.intor('ECPscalar_sph'),
         get_ecp(mol)),
        ('d/dR <i|V_ecp|j>     ', mol.intor('ECPscalar_ipnuc_sph'),
         get_ecp_ip(mol).sum(axis=0)),
        ('d2/dR2 <i|V_ecp|j>   ', mol.intor('ECPscalar_ipipnuc', comp=9),
         get_ecp_ipip(mol, 'ipipv').sum(axis=0)),
    ]
    print('\n--- ECP integrals: GPU vs CPU ---')
    ok = True
    for label, cpu, gpu in checks:
        err = np.linalg.norm(cpu - gpu.get())
        ok &= err < 1e-8
        print(f'{label} |GPU - CPU|_F = {err:.3e}  {"OK" if err < 1e-8 else "FAIL"}')
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bond', type=float, default=2.666, help='I-I distance in Angstrom')
    p.add_argument('--xc', default='pbe0')
    p.add_argument('--cpu', action='store_true', help='also run the CPU reference SCF')
    p.add_argument('--verbose', type=int, default=4)
    args = p.parse_args()

    mol = build_mol(args.bond, args.verbose)
    print(f'pyscf {pyscf.__version__}   nao = {mol.nao}   nelec = {mol.nelectron} '
          f'(core removed per I: {mol.atom_nelec_core(0)})')

    mf = rks.RKS(mol, xc=args.xc)
    mf.grids.atom_grid = (99, 590)
    mf.conv_tol = 1e-10

    t0 = time.perf_counter()
    e_tot = mf.kernel()
    t_scf = time.perf_counter() - t0
    assert mf.converged, 'SCF did not converge'
    print(f'\nE({args.xc}) = {e_tot:.9f} Ha   [SCF {t_scf:.2f} s]')

    t0 = time.perf_counter()
    grad = mf.Gradients().kernel()
    t_grad = time.perf_counter() - t0
    # Only the z component survives along the bond; the two atoms are equal and
    # opposite, so |grad| is also a convenient residual-force check.
    print(f'dE/dz on I(1)  = {grad[0, 2]:+.9f} Ha/Bohr   [grad {t_grad:.2f} s]')
    print(f'max |grad|     = {abs(grad).max():.3e}')

    check_ecp_integrals(mol)

    if args.cpu:
        print('\n--- full CPU reference ---')
        mf_cpu = mf.to_cpu()
        mf_cpu.conv_tol = 1e-10
        e_cpu = mf_cpu.kernel()
        g_cpu = mf_cpu.Gradients().kernel()
        print(f'E(CPU) = {e_cpu:.9f} Ha')
        print(f'dE          = {abs(e_tot - e_cpu):.3e} Ha    (target < 1e-8)')
        print(f'dGrad (max) = {abs(grad - g_cpu).max():.3e}  (target < 1e-6)')


if __name__ == '__main__':
    main()
