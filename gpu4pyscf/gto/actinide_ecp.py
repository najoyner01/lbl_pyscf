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

"""Spin-orbit ECPs for the actinides (Ac-Lr), for GKS / GHF + ``with_soc``.

Background
----------
Stage 2a of the actinide validation campaign
(``gpu4pyscf/hessian/tests/results/uranyl_soc_2a.md``) reported that "this pyscf
checkout ships no spin-orbit ECP for any actinide".  That conclusion was drawn
from trying only the *scalar* Stuttgart name ``stuttgart_rsc`` (ECP60MWB), whose
data file carries no ``SO`` block, so ``mol.has_ecp_soc()`` is ``False``.

It is **incorrect**.  The vendored pyscf *does* ship the matching spin-orbit
set, under the name ``ecpds60mwbso``
(``pyscf/gto/basis/soecp/ECPDS60MWBSO.dat``), covering Ac-Lr.  Its averaged
relativistic (AREP / scalar) part is byte-identical to ``stuttgart_rsc``'s
ECP60MWB; it simply adds the spin-orbit projector columns.  So

    ecp = {'U': 'ecpds60mwbso'}

is a drop-in replacement for ``ecp = {'U': 'stuttgart_rsc'}`` that additionally
flips ``mol.has_ecp_soc()`` to ``True`` and makes ``mol.intor('ECPso')`` /
``gpu4pyscf.gto.ecp.get_ecp_so`` non-trivial.

This module is a thin, documented accessor over that vendored data.  It does
**not** re-transcribe the parameters: duplicating ~250 published numbers into a
Python literal only creates a drift risk against pyscf's own file, and pyscf's
``gto.basis.parse_ecp`` already produces the exact ``_ecpbas`` layout the
gpu4pyscf SO-ECP kernel consumes.  The value added here is discoverability (so
the next actinide study does not repeat the Stage 2a mistake) and a single call
that pairs the SO-ECP with its matched valence basis.

.. warning::

   **Integral-level only.**  ``get_actinide_so_ecp`` gives you SO-ECP
   *integrals* that gpu4pyscf reproduces to ~1e-13 vs ``mol.intor('ECPso')``
   (validated Th-Cm, f and g projectors).  It does **not** make a
   *self-consistent* 2-component ``dft.GKS(...; with_soc=True)`` calculation
   physically meaningful for these 60-core sets: the U/Np/Pu 6s6p5d semicore is
   explicit, and the ECP60MWB-SO P/D spin-orbit projectors were fit for a
   restricted-active-space spin-orbit CI, not for a variational 2c-SCF one-
   electron operator.  On uranyl, variational GKS+SOC collapses ~5 Eh below the
   scalar reference (``hessian/tests/results/uranyl_soc_2a.md``).  Use these
   integrals perturbatively / in a restricted valence space, or use a
   Dirac-fitted 2c ECP or all-electron X2C-SOC for variational 2-component work.

ECP family and provenance
-------------------------
ECP60MWB-SO -- 60-electron ([Xe]4f14 + ... i.e. through 5d) small core,
energy-consistent, quasi-relativistic (Wood-Boring) adjusted, spin-orbit
potential.  Same core and AREP as the ``stuttgart_rsc`` set Stage 2a validated.

  * W. Kuechle, M. Dolg, H. Stoll, H. Preuss,
    J. Chem. Phys. 100, 7535 (1994).                     [ECP + SO potential]
  * X. Cao, M. Dolg, H. Stoll,
    J. Chem. Phys. 118, 487 (2003);
    X. Cao, M. Dolg, J. Molec. Struct. (THEOCHEM) 673, 203 (2004).  [valence basis]

Both references are reproduced verbatim in the header of
``pyscf/gto/basis/soecp/ECPDS60MWBSO.dat``.  The SO projectors run l = 1..4
(P, D, F, G); the gpu4pyscf angular table ``lib/ecp/so_ang_matrix.cu`` covers
l = 0..4, and ``ECP_LMAX`` in ``lib/ecp/ecp.h`` is 4, so no kernel change is
needed for f/g actinide SO projectors.

Usage
-----
    from pyscf import gto
    from gpu4pyscf.gto.actinide_ecp import get_actinide_so_ecp, actinide_basis
    from gpu4pyscf.gto.ecp import get_ecp_so, get_soc_1e

    mol = gto.M(atom='U 0 0 0; O 0 0 1.705; O 0 0 -1.705', charge=2, spin=0,
                basis={'U': actinide_basis('U'), 'O': 'def2-tzvp'},
                ecp={'U': get_actinide_so_ecp('U')})
    assert mol.has_ecp_soc()

    hso = get_ecp_so(mol)          # [3, nao, nao] real, == mol.intor('ECPso')
    hspinor = get_soc_1e(mol)      # [2nao, 2nao] complex

    # These are correct SO-ECP integrals.  Feed them to a *perturbative* /
    # restricted-space SOC treatment.  A bare self-consistent
    # dft.GKS(mol, xc=...; with_soc=True).kernel() is NOT recommended for these
    # 60-core sets -- see the warning above.
"""

from pyscf import gto as _gto

__all__ = ['ACTINIDES', 'SO_ECP_NAME', 'VALENCE_BASIS_NAME',
           'get_actinide_so_ecp', 'actinide_basis', 'source',
           'has_actinide_so_ecp']

# Ac-Lr, the span of ECPDS60MWBSO.dat.
ACTINIDES = ('Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm',
             'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr')

# pyscf basis-library names (vendored data; see module docstring).
SO_ECP_NAME = 'ecpds60mwbso'          # -> soecp/ECPDS60MWBSO.dat, AREP == ECP60MWB
VALENCE_BASIS_NAME = 'stuttgart_rsc'  # ECP60MWB-adjusted valence basis (Cao 2003)

# Per-element provenance.  All one family, one pair of papers -- but keep the
# map explicit so a future mixed-source addition has an obvious slot.
_SOURCE = {el: 'ECP60MWB-SO; Kuechle 1994 (JCP 100, 7535) + Cao 2003 (JCP 118, 487); '
               'pyscf soecp/ECPDS60MWBSO.dat'
           for el in ACTINIDES}


def _norm(symbol):
    sym = _gto.mole._std_symbol(symbol)
    if sym not in ACTINIDES:
        raise KeyError(
            f'{symbol!r} -> {sym!r} is not an actinide covered by '
            f'{SO_ECP_NAME} (Ac-Lr). Covered: {", ".join(ACTINIDES)}')
    return sym


def get_actinide_so_ecp(symbol):
    """Return the pyscf ECP name carrying the ECP60MWB **spin-orbit** potential
    for ``symbol`` (Ac-Lr).

    Pass it straight to ``gto.M(..., ecp={symbol: get_actinide_so_ecp(symbol)})``.
    The returned entry's scalar part equals ``stuttgart_rsc``; it additionally
    provides the SO projectors, so ``mol.has_ecp_soc()`` is ``True``.
    """
    _norm(symbol)
    return SO_ECP_NAME


def actinide_basis(symbol):
    """Return the matched valence-basis name for ``symbol`` (the ECP60MWB-adjusted
    ``stuttgart_rsc`` set; same one Stage 2a used for the scalar validation)."""
    _norm(symbol)
    return VALENCE_BASIS_NAME


def source(symbol):
    """Literature provenance string for ``symbol``'s SO-ECP."""
    return _SOURCE[_norm(symbol)]


def has_actinide_so_ecp(symbol):
    """``True`` if ``symbol`` is an actinide for which the SO-ECP is available."""
    try:
        _norm(symbol)
    except KeyError:
        return False
    return True
