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

"""The LANL f-element benchmark set, read from `lanl_benchmark.json`.

39 lanthanide and actinide coordination complexes at EXPERIMENTAL geometries --
16 to 211 atoms, 570 to 3747 AO at def2-TZVP, charges -2..+1, open shells up to
2S = 7. Unlike `systems.py` these are real molecules, not VSEPR idealizations.

BASIS. Everything is def2 (whatever `--basis` says, def2-TZVP by default), the
same as the rest of this benchmark directory, with ONE substitution: the
f-block metal. PySCF's def2 sets stop before the f block -- none of the eight
lanthanides or six actinides in this set has a def2 basis OR a def2 ECP -- so
those atoms take the Stuttgart-Koeln RSC small-core pseudopotential and its
matching valence basis instead (28e core for the Ln, 60e for the An). Iodine is
the only ligand element carrying an ECP, and it keeps the def2 one.

The substitution is forced, not a preference. It is also the pseudopotential
family the cc-pVnZ-PP basis sets are built against; those sets were never
extended to the f block, which is why they are not used here.

ENTRIES ARE DICTS, not the 8-tuples of `systems.py`: these rows carry oxidation
state, ligand type/class, multiplicity and free-text notes that the tuple
format has no room for. `run_bench.py` handles both shapes.

UNIQUE IDs. `Formula` is not unique in this file -- AmC84N3H114O9 appears four
times as a spin ladder on one geometry, and HoC51H54N9O15 appears twice as two
distinct conformers with identical labels, charge AND spin. Every entry
therefore gets a row-indexed uid, `r<NN>_<Formula>_s<2S>`, which is what the
driver records as `formula` so --resume and the report key correctly.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'lanl_benchmark.json')

# The f-block metals in this set, and the ECP + valence basis they take in
# place of def2. Small-core Stuttgart-Koeln RSC: 28 core electrons for the
# lanthanides ([Ar]3d10 replaced), 60 for the actinides ([Kr]4d10 4f14).
# Scalar relativity is folded into the pseudopotential.
STUTTGART = 'stuttgart-rsc'

LANTHANIDES = ('Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Tb', 'Ho', 'Er')
ACTINIDES = ('Th', 'U', 'Np', 'Pu', 'Am', 'Cm')
F_BLOCK = frozenset(LANTHANIDES + ACTINIDES)

# Notes text -> convergence aids. 31 of the 39 rows are flagged in the
# spreadsheet as hard to converge; plain DIIS from a minao guess will not get
# an open-shell f element there. These are applied IDENTICALLY on both devices
# -- they change how hard the SCF is driven, never what is being compared.
CONV_AIDS = {'level_shift': 0.2, 'damp': 0.3, 'max_cycle': 200}

_HARD_SCF_MARKERS = ('struggle', 'struggles')


def load_rows(path=DATA):
    """The raw spreadsheet rows, in file order."""
    with open(path) as fh:
        doc = json.load(fh)
    sheets = doc['sheets']
    if len(sheets) != 1:
        raise ValueError(f'expected 1 sheet in {path}, found {len(sheets)}')
    return sheets[0]['rows']


def _uid(index, row):
    """Collision-free id. Formula alone is not unique (see module docstring)."""
    return f'r{index:02d}_{row["Formula"]}_s{row["SpinPolarization"]}'


def _entry(index, row):
    parsed = row['Exp_xyz_parsed']
    atoms = [(a[0], (a[1], a[2], a[3])) for a in parsed['atoms']]

    declared = row.get('NumberAtoms')
    if declared is not None and len(atoms) != declared:
        raise ValueError(f'{row["Formula"]}: {len(atoms)} atoms parsed, '
                         f'NumberAtoms says {declared}')

    return {
        'uid': _uid(index, row),
        'row': index,
        'formula': row['Formula'],
        'metal': row['Metal'],
        'atoms': atoms,
        'charge': row['Charge'],
        # SpinPolarization is 2S (n_alpha - n_beta), which is what gto.M wants.
        'spin': row['SpinPolarization'],
        'multiplicity': row['Multiplicity'],
        # Recorded for grouping in the report; none of these touch the SCF.
        'oxidation_state': row['Oxidation State'],
        'ligand_type': row['LigandType'],
        'ligand_class': row['LigandClass'],
        'calc_type': row['Calc_Type'],
        'notes': row.get('Notes') or '',
    }


def load_entries(path=DATA):
    return [_entry(i, row) for i, row in enumerate(load_rows(path))]


SYSTEMS = load_entries()


# --------------------------------------------------------------------------- #
# basis / ECP assignment

def elements(entry):
    """Sorted unique element symbols in one entry."""
    return sorted({sym for sym, _ in entry['atoms']})


def basis_for(entry, basis):
    """Per-element orbital basis: Stuttgart RSC on the f-block metal, the
    requested def2 set on everything else."""
    return {el: (STUTTGART if el in F_BLOCK else basis)
            for el in elements(entry)}


def ecp_for(entry, basis):
    """Per-element ECP. The f-block metal always carries one; a ligand carries
    the def2 ECP only if the requested set actually defines one for it (of the
    ligand elements here that is iodine alone). Probing beats hard-coding: it
    stays correct if --basis changes."""
    from pyscf import gto

    ecp = {}
    for el in elements(entry):
        if el in F_BLOCK:
            ecp[el] = STUTTGART
            continue
        try:
            if gto.basis.load_ecp(basis, el):
                ecp[el] = basis
        except Exception:
            pass                        # no ECP for this element in this set
    return ecp


def build_mol(entry, basis, verbose=0):
    """Build a pyscf Mole for one entry at its experimental geometry."""
    from pyscf import gto
    return gto.M(
        atom=entry['atoms'],
        basis=basis_for(entry, basis),
        ecp=ecp_for(entry, basis),
        charge=entry['charge'],
        spin=entry['spin'],
        unit='Angstrom',
        verbose=verbose,
    )


def scf_aids(entry):
    """Convergence settings for one entry, from its Notes text. Empty dict when
    the row is not flagged. The caller applies these to both devices."""
    if any(m in entry['notes'].lower() for m in _HARD_SCF_MARKERS):
        return dict(CONV_AIDS)
    return {}


def iter_systems(only=None):
    """Yield entries, optionally filtered by uid, formula, metal or row index.

    Formula and metal match loosely on purpose: `--only Am` takes all six
    americium rows, `--only AmC84N3H114O9` takes the whole four-member spin
    ladder, and a uid takes exactly one. A row index matches either bare or
    in the `rNN` form the uid uses, so `--only 10` and `--only r10` both work."""
    if not only:
        yield from SYSTEMS
        return
    wanted = {str(s).lower() for s in only}
    for entry in SYSTEMS:
        row = entry['row']
        if (entry['uid'].lower() in wanted
                or entry['formula'].lower() in wanted
                or entry['metal'].lower() in wanted
                or str(row) in wanted
                or f'r{row:02d}' in wanted):
            yield entry


if __name__ == '__main__':
    # Sanity dump: composition and size of every system, no SCF. Cheap, CPU
    # only. Verifies that every row builds and that the electron count is
    # consistent with the requested spin.
    import sys

    basis = sys.argv[1] if len(sys.argv) > 1 else 'def2-tzvp'
    print(f'basis: {basis} ligands, {STUTTGART} on the f-block metal\n')
    print(f'{"uid":26s} {"M":3s} {"ox":>3s} {"chg":>4s} {"2S":>3s} '
          f'{"natm":>5s} {"nao":>6s} {"nelec":>6s} {"core":>5s} aids')

    total_nao = 0
    for entry in SYSTEMS:
        mol = build_mol(entry, basis)
        # gto.M would already have raised on an impossible (charge, spin)
        # combination; assert the parity explicitly so the failure is legible.
        assert (mol.nelectron - entry['spin']) % 2 == 0, entry['uid']
        core = sum(mol.atom_nelec_core(i) for i in range(mol.natm))
        total_nao += mol.nao
        print(f'{entry["uid"]:26s} {entry["metal"]:3s} '
              f'{entry["oxidation_state"]:3d} {entry["charge"]:4d} '
              f'{entry["spin"]:3d} {mol.natm:5d} {mol.nao:6d} '
              f'{mol.nelectron:6d} {core:5d} '
              f'{"yes" if scf_aids(entry) else "-"}')

    n_open = sum(1 for e in SYSTEMS if e['spin'] != 0)
    print(f'\n{len(SYSTEMS)} systems ({n_open} open shell), '
          f'{total_nao} AO total at {basis}')
