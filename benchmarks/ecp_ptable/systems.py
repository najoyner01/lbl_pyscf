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

"""One closed-shell test molecule for every element with a def2-ECP.

The def2 basis sets carry the Stuttgart-Koeln pseudopotentials for 36 elements
(Rb-Rn, including La). This module pairs each of them with a small, chemically
sensible, closed-shell singlet compound so the whole set can be run with plain
RKS -- no spin-state guessing, no UKS.

Geometries are IDEALIZED (VSEPR shapes with tabulated bond lengths), not
optimized. That is deliberate: this benchmark measures GPU/CPU timing and
GPU/CPU agreement, so both devices see identical coordinates. The energies are
NOT thermochemical reference values -- do not cite them as such.

Anions appear where the neutral compound would be open-shell (TcO4-, ReO4-,
[RhCl6]3-, [IrCl6]3-, [PdCl4]2-, [PtCl4]2-): a d0 or low-spin d6/d8
configuration keeps everything a singlet.
"""

import numpy as np

# (element, formula, centre, ligand, n_ligands, geometry, bond length /A, charge)
#
# Bond lengths are typical experimental or standard-reference values rounded to
# 0.01 A. Where a species is only known in the solid state or in solution
# (the halide anions), a representative crystallographic distance is used.
SYSTEMS = [
    ('Rb', 'RbH',      'Rb', 'H',  1, 'diatomic',      2.37,  0),
    ('Sr', 'SrCl2',    'Sr', 'Cl', 2, 'linear',        2.63,  0),
    ('Y',  'YH3',       'Y', 'H',  3, 'trigonal',      2.05,  0),
    ('Zr', 'ZrCl4',    'Zr', 'Cl', 4, 'tetrahedral',   2.32,  0),
    ('Nb', 'NbCl5',    'Nb', 'Cl', 5, 'bipyramidal',   2.29,  0),
    ('Mo', 'MoF6',     'Mo', 'F',  6, 'octahedral',    1.82,  0),
    ('Tc', 'TcO4-',    'Tc', 'O',  4, 'tetrahedral',   1.71, -1),
    ('Ru', 'RuO4',     'Ru', 'O',  4, 'tetrahedral',   1.71,  0),
    ('Rh', 'RhCl6_3-', 'Rh', 'Cl', 6, 'octahedral',    2.35, -3),
    ('Pd', 'PdCl4_2-', 'Pd', 'Cl', 4, 'square_planar', 2.31, -2),
    ('Ag', 'AgCl',     'Ag', 'Cl', 1, 'diatomic',      2.28,  0),
    ('Cd', 'CdCl2',    'Cd', 'Cl', 2, 'linear',        2.25,  0),
    ('In', 'InCl3',    'In', 'Cl', 3, 'trigonal',      2.29,  0),
    ('Sn', 'SnH4',     'Sn', 'H',  4, 'tetrahedral',   1.71,  0),
    ('Sb', 'SbH3',     'Sb', 'H',  3, 'pyramidal',     1.70,  0),
    ('Te', 'H2Te',     'Te', 'H',  2, 'bent',          1.66,  0),
    ('I',  'HI',        'I', 'H',  1, 'diatomic',      1.61,  0),
    ('Xe', 'XeF2',     'Xe', 'F',  2, 'linear',        1.98,  0),
    ('Cs', 'CsH',      'Cs', 'H',  1, 'diatomic',      2.49,  0),
    ('Ba', 'BaCl2',    'Ba', 'Cl', 2, 'linear',        2.78,  0),
    ('La', 'LaCl3',    'La', 'Cl', 3, 'trigonal',      2.59,  0),
    ('Hf', 'HfCl4',    'Hf', 'Cl', 4, 'tetrahedral',   2.32,  0),
    ('Ta', 'TaCl5',    'Ta', 'Cl', 5, 'bipyramidal',   2.30,  0),
    ('W',  'WF6',       'W', 'F',  6, 'octahedral',    1.83,  0),
    ('Re', 'ReO4-',    'Re', 'O',  4, 'tetrahedral',   1.72, -1),
    ('Os', 'OsO4',     'Os', 'O',  4, 'tetrahedral',   1.71,  0),
    ('Ir', 'IrCl6_3-', 'Ir', 'Cl', 6, 'octahedral',    2.35, -3),
    ('Pt', 'PtCl4_2-', 'Pt', 'Cl', 4, 'square_planar', 2.32, -2),
    ('Au', 'AuCl',     'Au', 'Cl', 1, 'diatomic',      2.20,  0),
    ('Hg', 'HgCl2',    'Hg', 'Cl', 2, 'linear',        2.25,  0),
    ('Tl', 'TlCl',     'Tl', 'Cl', 1, 'diatomic',      2.48,  0),
    ('Pb', 'PbCl2',    'Pb', 'Cl', 2, 'bent',          2.44,  0),
    ('Bi', 'BiH3',     'Bi', 'H',  3, 'pyramidal',     1.78,  0),
    ('Po', 'H2Po',     'Po', 'H',  2, 'bent',          1.75,  0),
    ('At', 'HAt',      'At', 'H',  1, 'diatomic',      1.72,  0),
    ('Rn', 'RnF2',     'Rn', 'F',  2, 'linear',        2.08,  0),
]

# Bond angle used for the bent / pyramidal shapes. Heavy p-block hydrides and
# dihalides sit near 90-100 deg (the central atom's lone pair dominates).
BENT_ANGLE = 95.0
PYRAMIDAL_ANGLE = 92.0


def _ligand_positions(geometry, r, n):
    """Unit-geometry ligand coordinates, scaled to bond length r (Angstrom)."""
    if geometry == 'diatomic':
        return [(0.0, 0.0, r)]

    if geometry == 'linear':
        return [(0.0, 0.0, r), (0.0, 0.0, -r)]

    if geometry == 'bent':
        h = np.deg2rad(BENT_ANGLE) / 2
        return [(r * np.sin(h), 0.0, r * np.cos(h)),
                (-r * np.sin(h), 0.0, r * np.cos(h))]

    if geometry == 'trigonal':                      # trigonal planar, D3h
        return [(r * np.cos(a), r * np.sin(a), 0.0)
                for a in np.deg2rad([0, 120, 240])]

    if geometry == 'pyramidal':                     # C3v, e.g. SbH3
        # Polar angle from the C3 axis that reproduces the X-A-X bond angle.
        theta = np.arccos(
            np.sqrt((1 + 2 * np.cos(np.deg2rad(PYRAMIDAL_ANGLE))) / 3))
        return [(r * np.sin(theta) * np.cos(a), r * np.sin(theta) * np.sin(a),
                 r * np.cos(theta)) for a in np.deg2rad([0, 120, 240])]

    if geometry == 'tetrahedral':
        d = r / np.sqrt(3)
        return [(d, d, d), (-d, -d, d), (-d, d, -d), (d, -d, -d)]

    if geometry == 'square_planar':
        return [(r, 0.0, 0.0), (-r, 0.0, 0.0), (0.0, r, 0.0), (0.0, -r, 0.0)]

    if geometry == 'bipyramidal':                   # trigonal bipyramidal, D3h
        eq = [(r * np.cos(a), r * np.sin(a), 0.0) for a in np.deg2rad([0, 120, 240])]
        return eq + [(0.0, 0.0, r), (0.0, 0.0, -r)]

    if geometry == 'octahedral':
        return [(r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0), (0, 0, r), (0, 0, -r)]

    raise ValueError(f'unknown geometry {geometry!r}')


def atom_string(centre, ligand, n, geometry, r):
    lines = [f'{centre} 0.0 0.0 0.0']
    pos = _ligand_positions(geometry, r, n)
    assert len(pos) == n, f'{geometry} gives {len(pos)} ligands, expected {n}'
    for x, y, z in pos:
        lines.append(f'{ligand} {x:.10f} {y:.10f} {z:.10f}')
    return '; '.join(lines)


def build_mol(entry, basis, verbose=0):
    """Build a pyscf Mole for one SYSTEMS entry. Same basis name for orbital
    basis and ECP -- def2 sets carry their ECP under the same name."""
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
    """Yield SYSTEMS entries, optionally filtered by element or formula."""
    if not only:
        yield from SYSTEMS
        return
    wanted = {s.lower() for s in only}
    for entry in SYSTEMS:
        if entry[0].lower() in wanted or entry[1].lower() in wanted:
            yield entry


if __name__ == '__main__':
    # Sanity dump: geometry + size of every system, no SCF. Cheap, CPU only.
    from pyscf import gto
    print(f'{"el":3s} {"formula":10s} {"geometry":14s} {"chg":>4s} '
          f'{"natm":>5s} {"nao":>5s} {"nelec":>6s} core/centre')
    for entry in SYSTEMS:
        mol = build_mol(entry, 'def2-tzvp')
        print(f'{entry[0]:3s} {entry[1]:10s} {entry[5]:14s} {entry[7]:4d} '
              f'{mol.natm:5d} {mol.nao:5d} {mol.nelectron:6d} '
              f'{mol.atom_nelec_core(0):d}')
    print(f'\n{len(SYSTEMS)} systems')
