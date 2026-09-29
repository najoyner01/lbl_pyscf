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

"""Turn run_bench.py's JSONL into markdown tables, a CSV, and figures.

    python make_report.py                       # reads ./results.jsonl
    python make_report.py --results other.jsonl --prefix run2

    # the neutral / open-shell set
    python make_report.py --results results_neutral.jsonl --prefix report_neutral

Outputs (next to the results file): <prefix>.md, <prefix>.csv, <prefix>.png.
matplotlib is optional -- without it the tables and CSV are still written.

Open-shell runs (`--systems neutral`) carry a `spin`, a `method` of `uks`, and
<S^2>. The report adds a spin section for those: an unrestricted SCF can reach
different symmetry-broken solutions on the two devices, which shows up as a huge
`dE` that is NOT a numerical-accuracy failure. Comparing <S^2> between devices
is what separates the two cases. Records without these fields are treated as
restricted singlets, so old results files still render.
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))

# Energies are quoted to 1e-8 Ha; anything above this counts as a real
# GPU/CPU disagreement rather than accumulated rounding.
DE_TOL = 1e-8

# <S^2> agreement between devices. Above this the two unrestricted SCFs are on
# different solutions, so their energy difference says nothing about accuracy.
DSS_TOL = 1e-4


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', default=os.path.join(HERE, 'results.jsonl'))
    p.add_argument('--prefix', default='report',
                   help='basename for the .md / .csv / .png outputs')
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--plot-subset', choices=('auto', 'anion', 'neutral', 'all'),
                   default='auto',
                   help="which charge states the figures show. 'auto' (default) "
                        "picks anion-only for the singlet set (its 6 charged "
                        "species are the thing unique to it) and neutral-only "
                        "for the neutral/lanl sets; the markdown tables and CSV "
                        "always cover every row regardless of this choice.")
    return p.parse_args(argv)


def load(path):
    meta, runs = {}, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get('kind') == 'meta':
                meta = rec            # last metadata block wins
            elif rec.get('kind') == 'run':
                runs.append(rec)
    return meta, runs


def pair_runs(runs):
    """Collapse per-device records into one row per (formula, functional)."""
    rows = OrderedDict()
    for rec in runs:
        key = (rec['formula'], rec['xc'])
        row = rows.setdefault(key, {
            # `element` is the metal centre on the LANL set, where there is no
            # one-compound-per-element mapping.
            'element': rec.get('element') or rec.get('metal'),
            'formula': rec['formula'],
            'xc': rec['xc'], 'nao': rec['nao'], 'natm': rec['natm'],
            'charge': rec['charge'], 'geometry': rec.get('geometry'),
            # absent in results written before --systems/--method existed
            'spin': rec.get('spin', 0), 'method': rec.get('method', 'rks'),
        })
        dev = rec['device']
        if rec.get('skipped'):
            # Not an error: the system was over --max-nao-cpu, so the CPU leg
            # was never attempted. Kept distinct so the report does not read
            # "too big to pair" as "the run broke".
            row[f'{dev}_skipped'] = rec['skipped']
            continue
        if rec.get('error'):
            row[f'{dev}_error'] = rec['error']
            continue
        row[f'{dev}_e'] = rec['e_tot']
        row[f'{dev}_t'] = rec['wall']
        row[f'{dev}_conv'] = rec['converged']
        row[f'{dev}_cycles'] = rec['cycles']
        row[f'{dev}_ngrids'] = rec['ngrids']
        if 's_squared' in rec:
            row[f'{dev}_ss'] = rec['s_squared']
            row[f'{dev}_mult'] = rec['multiplicity']

    for row in rows.values():
        if 'gpu_e' in row and 'cpu_e' in row:
            row['dE'] = abs(row['gpu_e'] - row['cpu_e'])
        if 'gpu_t' in row and 'cpu_t' in row:
            row['speedup'] = row['cpu_t'] / row['gpu_t']
        if 'gpu_ss' in row and 'cpu_ss' in row:
            row['dSS'] = abs(row['gpu_ss'] - row['cpu_ss'])
        # <S^2> - S(S+1) for the requested spin: how far the converged
        # unrestricted determinant is from the target spin state.
        s = row['spin'] / 2.0
        for dev in ('gpu', 'cpu'):
            if f'{dev}_ss' in row:
                row[f'{dev}_contam'] = row[f'{dev}_ss'] - s * (s + 1)
    return list(rows.values())


def fmt(value, spec='', dash='--'):
    return dash if value is None else format(value, spec)


def pair(row, key, spec='', dash='--'):
    """`gpu/cpu` for one per-device quantity, e.g. `56/44` or `6.0750/--`."""
    return (f'{fmt(row.get("gpu_" + key), spec, dash)}/'
            f'{fmt(row.get("cpu_" + key), spec, dash)}')


def conv_cell(row):
    """Per-device convergence as `yes/yes`, with any failure bolded so it is
    visible when skimming. A device that never ran shows `--`."""
    out = []
    for dev in ('gpu', 'cpu'):
        if row.get(f'{dev}_error'):
            out.append('**err**')
        elif row.get(f'{dev}_skipped'):
            out.append('skip')
        elif row.get(f'{dev}_conv') is True:
            out.append('yes')
        elif row.get(f'{dev}_conv') is False:
            out.append('**NO**')
        else:
            out.append('--')
    return '/'.join(out)


def failures(rows):
    """Every run where a device errored or hit the cycle cap without
    converging, newest concern first. Used for the report's failure table and
    the console summary -- the two places you look before anything else."""
    out = []
    for row in rows:
        for dev in ('gpu', 'cpu'):
            if row.get(f'{dev}_error') or row.get(f'{dev}_conv') is False:
                out.append((row, dev))
    return out


def write_csv(path, rows):
    import csv
    cols = ['element', 'formula', 'charge', 'spin', 'method', 'geometry',
            'natm', 'nao', 'xc',
            'gpu_e', 'cpu_e', 'dE', 'gpu_t', 'cpu_t', 'speedup',
            'gpu_cycles', 'cpu_cycles', 'gpu_conv', 'cpu_conv',
            'gpu_ngrids', 'cpu_ngrids',
            'gpu_ss', 'cpu_ss', 'dSS', 'gpu_contam', 'cpu_contam',
            'gpu_error', 'cpu_error', 'gpu_skipped', 'cpu_skipped']
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow(row)


def markdown(meta, rows, functionals, elements, png_name):
    out = []
    A = out.append
    open_rows = [r for r in rows if r.get('spin', 0) != 0]
    uks_rows = [r for r in rows if r.get('method') == 'uks']
    set_name = meta.get('systems', 'singlet')

    lanl = set_name == 'lanl'

    A('# ECP single-point benchmark: GPU4PySCF vs PySCF\n')
    if lanl:
        n_open = len({r['formula'] for r in open_rows})
        A(f'{len({r["formula"] for r in rows})} lanthanide and actinide '
          f'coordination complexes at experimental geometries '
          f'({len(elements)} metals, {n_open} of them open shell), each run '
          f'with {len(functionals)} functionals on both devices.\n')
    elif open_rows:
        n_open = len({r['formula'] for r in open_rows})
        A(f'One neutral compound per element carrying a def2-ECP '
          f'({len(elements)} elements, {n_open} of them open shell), each run '
          f'with {len(functionals)} functionals on both devices.\n')
    else:
        A('One closed-shell compound per element carrying a def2-ECP '
          f'({len(elements)} elements), each run with {len(functionals)} '
          'functionals on both devices.\n')

    A('> [!NOTE]')
    if lanl:
        A('> Geometries are experimental, not optimized. The basis is def2 '
          'throughout, as in the rest of this directory, with one forced '
          'substitution: PySCF\'s def2 sets stop before the f block, so each '
          'f-block metal takes the Stuttgart-Koeln RSC ECP and valence basis. '
          'Density fitting is on for both devices. Both devices see identical '
          'coordinates, so GPU/CPU comparisons are exact; the absolute '
          'energies are not thermochemical reference values.\n')
    else:
        A('> Geometries are idealized VSEPR shapes with tabulated bond '
          'lengths, not optimized. Both devices see identical coordinates, so '
          'GPU/CPU comparisons are exact; the absolute energies are not '
          'thermochemical reference values.\n')

    if lanl:
        A('> [!WARNING]')
        A('> The two devices build their own initial guesses, and on an '
          'open-shell f element that is often enough to land them in '
          'different SCF solutions -- `minao` alone differs by '
          '`max|dm0_gpu - dm0_cpu| = 57.8` on PuN3O9Cl3. A large `dE` on this '
          'set usually means "different solution", not "broken GPU kernel". '
          'Check `converged`, `cycles` and `<S^2>` in the **Open-shell / '
          'spin** section before reading any `dE`.\n')
    elif open_rows:
        A('> [!WARNING]')
        A('> The open-shell species keep the bond length tabulated for the '
          'corresponding anion in `systems.py`, and their multiplicities are '
          'formal d-count assignments, not verified ground states. An '
          'unrestricted SCF can also settle on different symmetry-broken '
          'solutions on the two devices; where that happens `dE` is large for '
          'reasons that have nothing to do with GPU accuracy. Check the '
          '**Open-shell / spin** section below before reading any `dE` on a '
          'UKS row.\n')

    A('## Settings\n')
    A('| key | value |')
    A('|---|---|')
    for key in ('systems', 'method', 'basis', 'grid', 'conv_tol', 'max_cycle',
                'density_fit', 'auxbasis', 'max_nao_cpu', 'conv_aids',
                'cpu_threads', 'repeat', 'gpu', 'contract_engine', 'cupy',
                'pyscf', 'gpu4pyscf', 'git_commit', 'hostname', 'slurm_job_id',
                'timestamp'):
        if meta.get(key) is not None:
            A(f'| {key} | `{meta[key]}` |')
    A(f'| functionals | `{" ".join(functionals)}` |')
    if uks_rows:
        A(f'| SCF flavour | `{len(rows) - len(uks_rows)} RKS, '
          f'{len(uks_rows)} UKS` |')
    A('')
    A('CPU references are built fresh from the Mole (`pyscf.dft.RKS` / '
      '`pyscf.dft.UKS`), never via `to_cpu()` of a converged GPU object -- '
      'that inherits the GPU density and biases CPU timings low by ~6x. The '
      'restricted/unrestricted choice depends only on the molecule and the '
      '`--method` flag, so both devices always run the same flavour.\n')

    # ---- failures ----------------------------------------------------------
    # Deliberately the first section after Settings: when a run breaks, the
    # cycle count and <S^2> are what say whether it stalled, ran away from the
    # target spin state, or died outright -- without paging through the full
    # table to find it.
    fails = failures(rows)
    A('## Failures\n')
    if not fails:
        A('None: every run converged on both devices.\n')
    else:
        A(f'{len(fails)} device-run{"s" if len(fails) > 1 else ""} errored or '
          'hit the cycle cap. `cyc` at the cap means the SCF was still moving '
          'when it ran out; `contam` is `<S^2> - S(S+1)` against the '
          'requested spin, so a large value means the determinant drifted off '
          'the target state. The paired device is shown alongside for '
          'contrast.\n')
        A('| device | element | compound | 2S | functional | nao | converged '
          '| cyc | <S^2> | contam | other device | note |')
        A('|---|---|---|---:|---|---:|---|---:|---:|---:|---|---|')
        for row, dev in sorted(
                fails, key=lambda rd: (rd[1], rd[0]['element'], rd[0]['xc'])):
            other = 'cpu' if dev == 'gpu' else 'gpu'
            if row.get(f'{other}_error'):
                peer = 'error'
            elif row.get(f'{other}_skipped'):
                peer = 'skipped'
            elif row.get(f'{other}_conv') is True:
                peer = f'ok, {fmt(row.get(f"{other}_cycles"), "d")} cyc'
            elif row.get(f'{other}_conv') is False:
                peer = f'also NO, {fmt(row.get(f"{other}_cycles"), "d")} cyc'
            else:
                peer = '--'
            err = row.get(f'{dev}_error')
            note = f'`{err[:50]}`' if err else 'hit cycle cap'
            A(f'| **{dev.upper()}** | {row["element"]} | {row["formula"]} '
              f'| {row.get("spin", 0)} | {row["xc"]} | {row["nao"]} '
              f'| {"error" if err else "**NO**"} '
              f'| {fmt(row.get(f"{dev}_cycles"), "d")} '
              f'| {fmt(row.get(f"{dev}_ss"), ".4f")} '
              f'| {fmt(row.get(f"{dev}_contam"), "+.4f")} '
              f'| {peer} | {note} |')
        A('')

    # ---- correctness -------------------------------------------------------
    A('## GPU vs CPU energy agreement\n')
    A(f'Worst |E_GPU - E_CPU| across functionals, per system. '
      f'Threshold {DE_TOL:.0e} Ha.\n')
    A('| element | compound | chg | 2S | SCF | natm | nao | max abs dE (Ha) | worst functional | status |')
    A('|---|---|---:|---:|---|---:|---:|---:|---|---|')
    for el in elements:
        sub = [r for r in rows if r['element'] == el and 'dE' in r]
        if not sub:
            A(f'| {el} | -- | | | | | | -- | -- | no paired runs |')
            continue
        worst = max(sub, key=lambda r: r['dE'])
        bad = [r for r in rows if r['element'] == el
               and (r.get('gpu_conv') is False or r.get('cpu_conv') is False)]
        errs = [r for r in rows if r['element'] == el
                and (r.get('gpu_error') or r.get('cpu_error'))]
        # A UKS pair on different symmetry-broken solutions fails dE for a
        # reason that is not about GPU accuracy -- say which it is.
        split = [r for r in rows if r['element'] == el
                 and r.get('dSS', 0.0) > DSS_TOL]
        status = 'ok' if worst['dE'] < DE_TOL else f'**dE > {DE_TOL:.0e}**'
        if split:
            status += (f', {len(split)} run{"s" if len(split) > 1 else ""} on a '
                       f'different SCF solution')
        if bad:
            status += f', {len(bad)} unconverged'
        if errs:
            status += f', {len(errs)} error'
        A(f'| {el} | {worst["formula"]} | {worst["charge"]:+d} '
          f'| {worst.get("spin", 0)} | {worst.get("method", "rks").upper()} '
          f'| {worst["natm"]} | {worst["nao"]} | {worst["dE"]:.2e} '
          f'| {worst["xc"]} | {status} |')
    A('')

    # ---- open shell --------------------------------------------------------
    if open_rows:
        A('## Open-shell / spin\n')
        A('Unrestricted runs only. `<S^2>` is the converged value on each '
          'device; `contam` is `<S^2> - S(S+1)` against the requested spin '
          '(large positive = the determinant is not the target spin state, '
          'expected for these formal high-oxidation-state species). '
          f'`d<S^2>` above {DSS_TOL:.0e} means the two devices converged to '
          'DIFFERENT solutions, and the corresponding `dE` is meaningless as '
          'an accuracy measure.\n')
        A('| element | compound | 2S | target S(S+1) | functional | '
          '<S^2> GPU | <S^2> CPU | d<S^2> | contam GPU | dE (Ha) | same solution? |')
        A('|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|')
        for row in sorted(open_rows, key=lambda r: (r['element'], r['xc'])):
            s = row['spin'] / 2.0
            same = '--'
            if 'dSS' in row:
                same = 'yes' if row['dSS'] <= DSS_TOL else '**no**'
            A(f'| {row["element"]} | {row["formula"]} | {row["spin"]} '
              f'| {s * (s + 1):.2f} | {row["xc"]} '
              f'| {fmt(row.get("gpu_ss"), ".4f")} '
              f'| {fmt(row.get("cpu_ss"), ".4f")} '
              f'| {fmt(row.get("dSS"), ".1e")} '
              f'| {fmt(row.get("gpu_contam"), "+.4f")} '
              f'| {fmt(row.get("dE"), ".2e")} | {same} |')
        A('')

        paired = [r for r in open_rows if 'dSS' in r]
        agree = [r for r in paired if r['dSS'] <= DSS_TOL]
        A(f'{len(agree)}/{len(paired)} unrestricted pairs reached the same '
          'solution on both devices.')
        if paired and len(agree) < len(paired):
            A('')
            A('> [!IMPORTANT]')
            A('> Rows marked **no** are a UKS initial-guess / convergence '
              'difference, not a GPU integral or XC error. To compare those '
              'systems numerically, restart both devices from the same '
              'converged density, or exclude them from the dE statistics.')
        A('')

    # ---- speedup matrix ---------------------------------------------------
    A('## Speedup (CPU wall time / GPU wall time)\n')
    A(f'CPU at {meta.get("cpu_threads", "?")} threads, single GPU. '
      'Values below 1.00 mean the CPU was faster.\n')
    A('| element | compound | nao | ' + ' | '.join(functionals) + ' | mean |')
    A('|---|---|---:|' + '---:|' * (len(functionals) + 1))
    for el in elements:
        sub = {r['xc']: r for r in rows if r['element'] == el}
        if not sub:
            continue
        any_row = next(iter(sub.values()))
        cells, vals = [], []
        for xc in functionals:
            sp = sub.get(xc, {}).get('speedup')
            cells.append(fmt(sp, '.2f'))
            if sp is not None:
                vals.append(sp)
        mean = sum(vals) / len(vals) if vals else None
        A(f'| {el} | {any_row["formula"]} | {any_row["nao"]} | '
          + ' | '.join(cells) + f' | {fmt(mean, ".2f")} |')
    A('')

    # ---- per-functional summary ------------------------------------------
    A('## Per-functional summary\n')
    A('| functional | systems | mean speedup | median speedup | '
      'total GPU (s) | total CPU (s) | max abs dE (Ha) |')
    A('|---|---:|---:|---:|---:|---:|---:|')
    for xc in functionals:
        sub = [r for r in rows if r['xc'] == xc and 'speedup' in r]
        if not sub:
            A(f'| {xc} | 0 | -- | -- | -- | -- | -- |')
            continue
        sp = sorted(r['speedup'] for r in sub)
        mid = sp[len(sp) // 2] if len(sp) % 2 else (sp[len(sp)//2 - 1] + sp[len(sp)//2]) / 2
        A(f'| {xc} | {len(sub)} | {sum(sp)/len(sp):.2f} | {mid:.2f} | '
          f'{sum(r["gpu_t"] for r in sub):.1f} | '
          f'{sum(r["cpu_t"] for r in sub):.1f} | '
          f'{max(r["dE"] for r in sub if "dE" in r):.2e} |')
    A('')

    # ---- full table -------------------------------------------------------
    A('## Full results\n')
    A('Every per-device column is `GPU/CPU`. `conv` is the SCF convergence '
      'flag, `cyc` the cycle count and `<S^2>` the converged spin '
      'expectation -- the three numbers to read together when a run looks '
      'wrong. A **NO** in `conv` means that device hit the cycle cap, so its '
      'energy and any `dE` built from it are not converged values.\n')
    A('| element | compound | 2S | SCF | nao | functional | E_GPU (Ha) | '
      'E_CPU (Ha) | dE (Ha) | t_GPU (s) | t_CPU (s) | speedup | conv | '
      'cyc | <S^2> |')
    A('|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|')
    for row in sorted(rows, key=lambda r: (r['nao'], r['element'], r['xc'])):
        head = (f'| {row["element"]} | {row["formula"]} | {row.get("spin", 0)} '
                f'| {row.get("method", "rks").upper()} | {row["nao"]} ')
        tail = (f'| {conv_cell(row)} | {pair(row, "cycles", "d")} '
                f'| {pair(row, "ss", ".4f")} |')
        err = row.get('gpu_error') or row.get('cpu_error')
        if err:
            # Still emit conv/cyc/<S^2>: on a partial failure the surviving
            # device's numbers are exactly what diagnoses the dead one.
            A(head + f'| {row["xc"]} | `{err[:60]}` | '
              + ' | '.join(['--'] * 5) + ' ' + tail)
            continue
        A(head + f'| {row["xc"]} '
          f'| {fmt(row.get("gpu_e"), ".8f")} | {fmt(row.get("cpu_e"), ".8f")} '
          f'| {fmt(row.get("dE"), ".2e")} | {fmt(row.get("gpu_t"), ".2f")} '
          f'| {fmt(row.get("cpu_t"), ".2f")} | {fmt(row.get("speedup"), ".2f")} '
          + tail)
    A('')

    if png_name:
        A('## Figures\n')
        A(f'![benchmark figures]({png_name})\n')
    return '\n'.join(out)


# Publication style: no gridlines anywhere, heavy type, thick axes. Every
# number a reader has to read off the figure is bold and >= 9 pt at the
# default 300 dpi raster size.
PUB_RC = {
    'font.family': 'DejaVu Sans',
    'font.weight': 'bold',
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.labelweight': 'bold',
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.linewidth': 2.0,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'xtick.minor.width': 1.2,
    'ytick.minor.width': 1.2,
    'xtick.major.size': 6.5,
    'ytick.major.size': 6.5,
    'xtick.minor.size': 3.5,
    'ytick.minor.size': 3.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'legend.fontsize': 11,
    'legend.frameon': False,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'figure.dpi': 100,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}


def select_plot_subset(rows, meta, choice):
    """Pick which charge states the figures show.

    'auto' keeps the two def2 sets internally homogeneous: the singlet set
    (`--systems singlet`, the default) mixes 30 neutrals with 6 anions used
    precisely to dodge open-shell neutrals, so its figures show the anions --
    the thing unique to that set. The neutral set is charge-0 throughout
    except for those same six species (now open-shell neutrals), so its
    figures show the neutral rows. Falls back to every row if the requested
    subset is empty (e.g. an all-neutral lanl run), so a mismatched --systems
    guess never produces a blank figure.
    """
    systems = meta.get('systems') or 'singlet'
    if choice == 'all':
        return rows, None
    if choice == 'auto':
        want = 'anion' if systems == 'singlet' else 'neutral'
    else:
        want = choice
    if want == 'anion':
        sub = [r for r in rows if r.get('charge', 0) < 0]
        label = 'anions only'
    else:
        sub = [r for r in rows if r.get('charge', 0) == 0]
        label = 'neutrals only'
    if not sub:
        return rows, None
    return sub, label


def _bold_ticks(ax):
    """Bold every tick label and kill any inherited gridlines."""
    ax.grid(False, which='both')
    for lab in list(ax.get_xticklabels(which='both')) + \
            list(ax.get_yticklabels(which='both')):
        lab.set_fontweight('bold')


def plot(rows, functionals, path, meta, subset_label=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print('matplotlib not installed -- skipping figures.\n'
              '  pip install matplotlib', file=sys.stderr)
        return None

    written = []
    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(17, 14))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.32)
        title = (f'ECP single-point benchmark: {meta.get("basis", "?")}, '
                 f'{meta.get("gpu", "GPU")} vs CPU @ {meta.get("cpu_threads", "?")} threads')
        if subset_label:
            title += f'  --  {subset_label}'
        fig.suptitle(title, fontsize=19, fontweight='bold')
        # Colour-blind-safe qualitative set (Okabe-Ito), one colour per rung.
        palette = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
                   '#E69F00', '#56B4E9', '#000000', '#7F7F7F']
        colors = {xc: palette[i % len(palette)]
                  for i, xc in enumerate(functionals)}

        # (a) wall time vs system size -- B3LYP only, the workhorse hybrid
        ax = fig.add_subplot(gs[0, 0])
        b3lyp = next((xc for xc in functionals if xc.lower() == 'b3lyp'), None)
        if b3lyp is not None:
            sub = sorted((r for r in rows if r['xc'] == b3lyp and 'gpu_t' in r),
                         key=lambda r: r['nao'])
            if sub:
                ax.plot([r['nao'] for r in sub], [r['gpu_t'] for r in sub],
                        'o-', color=colors[b3lyp], ms=7, lw=2.4, label='GPU')
            sub = sorted((r for r in rows if r['xc'] == b3lyp and 'cpu_t' in r),
                         key=lambda r: r['nao'])
            if sub:
                ax.plot([r['nao'] for r in sub], [r['cpu_t'] for r in sub],
                        's--', color=colors[b3lyp], ms=6, lw=2.0, alpha=0.7,
                        label='CPU')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Number of AOs')
        ax.set_ylabel('SCF wall time (s)')
        ax.set_title('(a)  B3LYP wall time vs system size\nsolid = GPU,  dashed = CPU',
                     loc='left')
        leg = ax.legend(fontsize=11, handlelength=2.2, labelspacing=0.3)
        for txt in leg.get_texts():
            txt.set_fontweight('bold')
        _bold_ticks(ax)

        # (b) speedup vs system size, all functionals -- log y, since the
        # ratio spans two decades
        ax = fig.add_subplot(gs[0, 1])
        for xc in functionals:
            sub = sorted((r for r in rows if r['xc'] == xc and 'speedup' in r),
                         key=lambda r: r['nao'])
            if sub:
                ax.plot([r['nao'] for r in sub], [r['speedup'] for r in sub],
                        'o-', color=colors[xc], ms=6, lw=2.2, label=xc)
        ax.axhline(1.0, color='k', ls='-', lw=2.5, zorder=0)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Number of AOs')
        ax.set_ylabel('Speedup   t(CPU) / t(GPU)')
        ax.set_title('(b)  Speedup vs system size', loc='left')
        ax.text(0.03, 0.95, 'GPU faster', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', color='#333333')
        ax.text(0.03, 0.05, 'CPU faster', transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='bottom', color='#333333')
        leg = ax.legend(ncol=2, fontsize=11, handlelength=2.2,
                        columnspacing=1.0, labelspacing=0.3, loc='lower right')
        for txt in leg.get_texts():
            txt.set_fontweight('bold')
        _bold_ticks(ax)

        # (c) speedup heat map, systems ordered by size -- spans the full
        # bottom row now that panel (d) is gone
        ax = fig.add_subplot(gs[1, :])
        order = sorted({(r['nao'], r['element']) for r in rows})
        els = [el for _, el in order]
        grid = np.full((len(els), len(functionals)), np.nan)
        for i, el in enumerate(els):
            for j, xc in enumerate(functionals):
                hit = [r for r in rows if r['element'] == el and r['xc'] == xc
                       and 'speedup' in r]
                if hit:
                    grid[i, j] = hit[0]['speedup']
        if np.isfinite(grid).any():
            lg = np.log2(grid)
            vmax = np.nanmax(np.abs(lg[np.isfinite(lg)]))
            im = ax.imshow(lg, aspect='auto', cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax)
            cb = fig.colorbar(im, ax=ax, pad=0.02)
            cb.set_label('log₂(speedup)      > 0: GPU faster',
                         fontsize=13, fontweight='bold')
            cb.outline.set_linewidth(2.0)
            cb.ax.tick_params(width=2.0, length=6, labelsize=12)
            for lab in cb.ax.get_yticklabels():
                lab.set_fontweight('bold')
            for i in range(len(els)):
                for j in range(len(functionals)):
                    if np.isfinite(grid[i, j]):
                        # white on the saturated ends, black in the middle
                        shade = abs(lg[i, j]) / vmax if vmax else 0.0
                        ax.text(j, i, f'{grid[i, j]:.1f}', ha='center',
                                va='center', fontsize=8.5, fontweight='bold',
                                color='white' if shade > 0.55 else 'black')
        ax.set_xticks(range(len(functionals)))
        ax.set_xticklabels(functionals, rotation=40, ha='right', fontsize=13)
        ax.set_yticks(range(len(els)))
        ax.set_yticklabels([f'{el} ({nao})' for nao, el in order], fontsize=8.5)
        ax.tick_params(length=0)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(True)
        ax.set_title('(c)  Speedup by element (AO count) and functional',
                     loc='left')
        _bold_ticks(ax)

        fig.tight_layout(rect=(0, 0, 1, 0.965))
        fig.savefig(path, dpi=300)
        written.append(path)
        # vector companion for typesetting
        pdf_path = os.path.splitext(path)[0] + '.pdf'
        fig.savefig(pdf_path)
        written.append(pdf_path)
        plt.close(fig)

    for p in written:
        print(f'wrote {p}')
    return path


def main(argv=None):
    args = parse_args(argv)
    if not os.path.exists(args.results):
        sys.exit(f'no results file at {args.results} -- run run_bench.py first')

    meta, runs = load(args.results)
    if not runs:
        sys.exit(f'{args.results} contains no run records')
    rows = pair_runs(runs)

    # Preserve the order the driver used, not alphabetical.
    functionals = meta.get('functionals') or list(
        OrderedDict.fromkeys(r['xc'] for r in rows))
    elements = list(OrderedDict.fromkeys(r['element'] for r in rows))

    outdir = os.path.dirname(os.path.abspath(args.results))
    csv_path = os.path.join(outdir, f'{args.prefix}.csv')
    md_path = os.path.join(outdir, f'{args.prefix}.md')
    png_path = os.path.join(outdir, f'{args.prefix}.png')

    write_csv(csv_path, rows)
    print(f'wrote {csv_path}')

    plot_rows, subset_label = select_plot_subset(rows, meta, args.plot_subset)
    png = None if args.no_plot else plot(plot_rows, functionals, png_path, meta,
                                         subset_label=subset_label)
    with open(md_path, 'w') as fh:
        fh.write(markdown(meta, rows, functionals, elements,
                          os.path.basename(png) if png else None))
    print(f'wrote {md_path}')

    paired = [r for r in rows if 'speedup' in r]
    if paired:
        sp = [r['speedup'] for r in paired]
        worst = max((r for r in rows if 'dE' in r), key=lambda r: r['dE'])
        print(f'\n{len(paired)} paired runs | speedup mean {sum(sp)/len(sp):.2f}x, '
              f'min {min(sp):.2f}x, max {max(sp):.2f}x')
        print(f'worst energy deviation: {worst["dE"]:.2e} Ha '
              f'({worst["formula"]}, {worst["xc"]})')
    # Failures with the two numbers that diagnose them, rather than a bare
    # list of formulas you then have to go look up.
    fails = failures(rows)
    if fails:
        print(f'\n{len(fails)} failed device-run(s) '
              '[device system xc: cycles, <S^2>, contam]')
        for row, dev in sorted(
                fails, key=lambda rd: (rd[1], rd[0]['element'], rd[0]['xc'])):
            err = row.get(f'{dev}_error')
            detail = (f'cyc={fmt(row.get(f"{dev}_cycles"), "d")} '
                      f'<S^2>={fmt(row.get(f"{dev}_ss"), ".4f")} '
                      f'contam={fmt(row.get(f"{dev}_contam"), "+.4f")}')
            reason = f'ERROR {err[:50]}' if err else 'not converged'
            print(f'  {dev:3s} {row["formula"]:26s} {row["xc"]:8s} '
                  f'{reason:20s} {detail}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
