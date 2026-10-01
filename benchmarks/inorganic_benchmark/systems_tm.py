#!/usr/bin/env python
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

"""The transition-metal subset of `systems.py`, run up a basis-set ladder.

19 closed-shell 4d/5d transition-metal complexes (Y-Cd and Hf-Hg), every one
carrying a def2-ECP on the metal, at the same idealized geometries as
`systems.py`. The point of this set is the BASIS LADDER: each complex is run at
double, triple and quadruple zeta so the GPU/CPU speedup can be read as a
function of system size across ~50-500 AO, rather than at one basis only.

    zeta   basis       ECP        metal lmax   ligand lmax   AO range (this set)
    DZ     def2-SVP    def2-ECP   d/f          d             46-140
    TZ     def2-TZVP   def2-ECP   f            d/f           58-262
    QZ     def2-QZVP   def2-ECP   g            f/g           142-492

QUINTUPLE ZETA IS DELIBERATELY ABSENT. def2 has no 5Z member, and the only
5Z ECP family covering these metals is cc-pV5Z-PP, which carries i functions
(l = 6) on the metal. gpu4pyscf's ECP kernels stop at g (l = 4); the h and i
blocks either abort (`ECP_cart ... invalid argument`, the general kernel asks
for more than 48 KB of shared memory without raising the limit) or, for the
nuclear-attraction path, silently return wrong integrals. cc-pVQZ-PP has the
same problem (h functions), which is why the ladder is def2 rather than
cc-pVnZ-PP throughout. 5Z is rarely used for DFT in any case; it can be added
once the l > 4 kernels land.

The six anions of `systems.py` that fall in this subset (TcO4-, ReO4-,
[RhCl6]3-, [IrCl6]3-, [PdCl4]2-, [PtCl4]2-) are kept as anions so the whole
ladder runs restricted RKS. As in `systems.py`, geometries are idealized VSEPR
shapes; both devices see identical coordinates and the energies are timing /
agreement fixtures, not thermochemical reference values.
"""

from collections import OrderedDict

from systems import SYSTEMS, atom_string, _ligand_positions  # noqa: F401

# zeta label -> def2 basis. The label is what the report and figures show; the
# basis name is what goes to gto.M (def2 sets carry their ECP under the same
# name). Ordered smallest to largest.
LADDER = OrderedDict([
    ('DZ', 'def2-svp'),
    ('TZ', 'def2-tzvp'),
    ('QZ', 'def2-qzvp'),
])
ZETA_OF = {basis: zeta for zeta, basis in LADDER.items()}

# 4d and 5d blocks, groups 3-12. La is excluded (it is not a d-block metal in
# the def2 sense either: def2 treats it with the lanthanide ECP, 46 e- core).
TRANSITION_METALS = ('Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                     'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg')

# Same 8-tuple layout as systems.SYSTEMS, filtered. Derived, not retyped, so
# the geometry and bond length of each complex live in exactly one place.
TM_SYSTEMS = [e for e in SYSTEMS if e[0] in TRANSITION_METALS]
assert len(TM_SYSTEMS) == len(TRANSITION_METALS), \
    'systems.SYSTEMS is missing a transition metal'


def zeta_label(basis):
    """`def2-tzvp` -> `TZ`; an unknown basis is returned unchanged so a custom
    --basis still labels sensibly."""
    return ZETA_OF.get(basis.lower(), basis)


def build_mol(entry, basis, verbose=0):
    """pyscf Mole for one TM_SYSTEMS entry at the given def2 basis. Same name
    for orbital basis and ECP -- def2 sets carry both."""
    from pyscf import gto
    element, formula, centre, ligand, n, geometry, r, charge = entry
    return gto.M(
        atom=atom_string(centre, ligand, n, geometry, r),
        basis=basis,
        ecp=basis,
        charge=charge,
        spin=0,
        unit='Angstrom',
        verbose=verbose,
    )


def iter_systems(only=None):
    """Yield TM_SYSTEMS entries, optionally filtered by element or formula."""
    if not only:
        yield from TM_SYSTEMS
        return
    wanted = {s.lower() for s in only}
    for entry in TM_SYSTEMS:
        if entry[0].lower() in wanted or entry[1].lower() in wanted:
            yield entry


if __name__ == '__main__':
    # Sanity dump: AO count and max angular momentum at every rung. No SCF.
    print(f'{"el":3s} {"formula":10s} {"chg":>4s} '
          + ' '.join(f'{z:>9s}' for z in LADDER) + '   lmax per rung')
    for entry in TM_SYSTEMS:
        naos, lmax = [], []
        for basis in LADDER.values():
            mol = build_mol(entry, basis)
            naos.append(mol.nao)
            lmax.append(max(mol.bas_angular(i) for i in range(mol.nbas)))
        print(f'{entry[0]:3s} {entry[1]:10s} {entry[7]:4d} '
              + ' '.join(f'{n:9d}' for n in naos) + f'   {lmax}')
    print(f'\n{len(TM_SYSTEMS)} systems x {len(LADDER)} bases')
