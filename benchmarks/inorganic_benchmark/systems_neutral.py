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

"""The `systems.py` molecule set with every anion replaced by its neutral form.

`systems.py` keeps the whole def2-ECP sweep closed-shell by using anions for the
six cases where the neutral compound is not a singlet (TcO4-, ReO4-, [RhCl6]3-,
[IrCl6]3-, [PdCl4]2-, [PtCl4]2-). This module drops the charges instead, which
makes those six OPEN SHELL and exercises the unrestricted (UKS) path on GPU and
CPU. The other 30 compounds are already neutral singlets and pass through
unchanged, so the two sets share 30 of 36 entries.

The table is DERIVED from `systems.SYSTEMS` rather than retyped: the element
list, geometries and bond lengths stay in one place. In particular the six
formerly-anionic species keep the bond length that was tabulated for the anion.
That is deliberate -- it is the wrong bond length for the neutral molecule, but
it means the neutral and singlet benchmarks run at *identical coordinates*, so
their timings are directly comparable. As in `systems.py`, the energies are
timing/agreement fixtures, not thermochemical reference values.

SPIN ASSIGNMENTS ARE FORMAL, NOT VERIFIED GROUND STATES.

Each of the six is given the multiplicity implied by its formal d-count in the
idealized ligand field. Electron parity fixes the parity of `spin`; the choice
within that parity is the textbook one:

| species | formal | d-count | spin (2S) | basis for the choice |
|---|---|---|---|---|
| TcO4  | Tc(VII) | d0 | 1 | 47 e-, odd. Metal is d0; the hole is O-centred (the known TcO4 radical). |
| ReO4  | Re(VII) | d0 | 1 | 47 e-, same as TcO4. |
| RhCl6 | Rh(VI)  | d3 | 3 | 119 e-, odd. Octahedral t2g^3, S=3/2 -- as in the known d3 RhF6. |
| IrCl6 | Ir(VI)  | d3 | 3 | 119 e-, odd. Octahedral t2g^3, S=3/2 -- as in the known d3 IrF6. |
| PdCl4 | Pd(IV)  | d6 | 2 | 86 e-, even. Four-coordinate d6 is not the low-spin octahedral case; triplet. |
| PtCl4 | Pt(IV)  | d6 | 2 | 86 e-, even. Same as PdCl4. |

M(VI) hexachlorides and four-coordinate M(IV) chlorides are strongly oxidizing,
chemically marginal species -- at an unrelaxed anion geometry some of them may
converge slowly, land on a symmetry-broken solution, or not converge at all.
That is expected and is recorded rather than hidden: `run_bench.py` writes
`converged` and, for unrestricted runs, <S^2> and the realized multiplicity, so
spin contamination and CPU/GPU solution mismatches are visible in the report.

Override a multiplicity by editing NEUTRAL_OVERRIDES below; nothing else needs
to change.
"""

from systems import SYSTEMS, atom_string  # noqa: F401  (atom_string re-exported)

# anion formula in systems.SYSTEMS -> (neutral formula, spin = 2S, rationale)
NEUTRAL_OVERRIDES = {
    'TcO4-':    ('TcO4',  1, 'Tc(VII) d0, O-centred hole; 47 e- doublet'),
    'ReO4-':    ('ReO4',  1, 'Re(VII) d0, O-centred hole; 47 e- doublet'),
    'RhCl6_3-': ('RhCl6', 3, 'Rh(VI) d3, Oh t2g^3; 119 e- quartet'),
    'IrCl6_3-': ('IrCl6', 3, 'Ir(VI) d3, Oh t2g^3; 119 e- quartet'),
    'PdCl4_2-': ('PdCl4', 2, 'Pd(IV) d6, 4-coordinate; 86 e- triplet'),
    'PtCl4_2-': ('PtCl4', 2, 'Pt(IV) d6, 4-coordinate; 86 e- triplet'),
}


def _build_table():
    """(element, formula, centre, ligand, n, geometry, r, charge, spin) rows.

    One more field than `systems.SYSTEMS`: the trailing `spin` (= 2S = nalpha -
    nbeta). `charge` is always 0 and is kept only so the tuple stays a superset
    of the singlet table's layout.
    """
    table, used = [], set()
    for element, formula, centre, ligand, n, geometry, r, charge in SYSTEMS:
        if charge == 0:
            spin = 0
        else:
            if formula not in NEUTRAL_OVERRIDES:
                raise KeyError(
                    f'{formula} is charged in systems.SYSTEMS but has no entry '
                    f'in NEUTRAL_OVERRIDES -- add one giving its neutral '
                    f'formula and multiplicity.')
            formula, spin, _ = NEUTRAL_OVERRIDES[formula]
            used.add(formula)
        table.append((element, formula, centre, ligand, n, geometry, r, 0, spin))

    stale = {f for f, _, _ in NEUTRAL_OVERRIDES.values()} - used
    if stale:
        raise KeyError(f'NEUTRAL_OVERRIDES entries never applied: {sorted(stale)} '
                       f'-- systems.SYSTEMS no longer has the matching anion.')
    return table


NEUTRAL_SYSTEMS = _build_table()

# Convenience views used by the driver and the report.
OPEN_SHELL = [e[1] for e in NEUTRAL_SYSTEMS if e[8] != 0]
SPIN_OF = {e[1]: e[8] for e in NEUTRAL_SYSTEMS}


def build_mol(entry, basis, verbose=0):
    """Build a pyscf Mole for one NEUTRAL_SYSTEMS entry.

    Same basis name for orbital basis and ECP -- def2 sets carry their ECP under
    the same name. Mirrors `systems.build_mol` but honours the entry's spin.
    """
    from pyscf import gto
    element, formula, centre, ligand, n, geometry, r, charge, spin = entry
    return gto.M(
        atom=atom_string(centre, ligand, n, geometry, r),
        basis=basis,
        ecp=basis,
        charge=charge,
        spin=spin,
        unit='Angstrom',
        verbose=verbose,
    )


def iter_systems(only=None):
    """Yield NEUTRAL_SYSTEMS entries, optionally filtered by element or formula.

    The filter also accepts the anion formula from `systems.py` (`--only TcO4-`
    finds neutral `TcO4`), so the same `--only` list works against both sets.
    """
    if not only:
        yield from NEUTRAL_SYSTEMS
        return
    wanted = {s.lower() for s in only}
    # map anion formula -> neutral formula so either name matches
    alias = {k.lower(): v[0].lower() for k, v in NEUTRAL_OVERRIDES.items()}
    wanted |= {alias[w] for w in wanted if w in alias}
    for entry in NEUTRAL_SYSTEMS:
        if entry[0].lower() in wanted or entry[1].lower() in wanted:
            yield entry


def check_spins(basis='def2-tzvp'):
    """Assert every entry is constructible: (nelectron - spin) must be even.

    gto.M raises on a parity mismatch anyway, but doing it up front turns a
    mid-sweep crash into an import-time error. Returns the built Moles.
    """
    mols = []
    for entry in NEUTRAL_SYSTEMS:
        mol = build_mol(entry, basis)
        assert (mol.nelectron - mol.spin) % 2 == 0, entry
        mols.append((entry, mol))
    return mols


if __name__ == '__main__':
    # Sanity dump: geometry, spin and size of every system, no SCF. CPU only.
    print(f'{"el":3s} {"formula":8s} {"geometry":14s} {"chg":>4s} {"2S":>3s} '
          f'{"mult":>5s} {"natm":>5s} {"nao":>5s} {"nelec":>6s} {"core":>5s}')
    for entry, mol in check_spins():
        print(f'{entry[0]:3s} {entry[1]:8s} {entry[5]:14s} {entry[7]:4d} '
              f'{entry[8]:3d} {entry[8] + 1:5d} {mol.natm:5d} {mol.nao:5d} '
              f'{mol.nelectron:6d} {mol.atom_nelec_core(0):5d}')
    print(f'\n{len(NEUTRAL_SYSTEMS)} systems, '
          f'{len(OPEN_SHELL)} open shell: {" ".join(OPEN_SHELL)}')
