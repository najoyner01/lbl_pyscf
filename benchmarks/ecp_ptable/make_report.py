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

Outputs (next to the results file): <prefix>.md, <prefix>.csv, <prefix>.png.
matplotlib is optional -- without it the tables and CSV are still written.
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results', default=os.path.join(HERE, 'results.jsonl'))
    p.add_argument('--prefix', default='report',
                   help='basename for the .md / .csv / .png outputs')
    p.add_argument('--no-plot', action='store_true')
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
            'element': rec['element'], 'formula': rec['formula'],
            'xc': rec['xc'], 'nao': rec['nao'], 'natm': rec['natm'],
            'charge': rec['charge'], 'geometry': rec.get('geometry'),
        })
        dev = rec['device']
        if rec.get('error'):
            row[f'{dev}_error'] = rec['error']
            continue
        row[f'{dev}_e'] = rec['e_tot']
        row[f'{dev}_t'] = rec['wall']
        row[f'{dev}_conv'] = rec['converged']
        row[f'{dev}_cycles'] = rec['cycles']
        row[f'{dev}_ngrids'] = rec['ngrids']

    for row in rows.values():
        if 'gpu_e' in row and 'cpu_e' in row:
            row['dE'] = abs(row['gpu_e'] - row['cpu_e'])
        if 'gpu_t' in row and 'cpu_t' in row:
            row['speedup'] = row['cpu_t'] / row['gpu_t']
    return list(rows.values())


def fmt(value, spec='', dash='--'):
    return dash if value is None else format(value, spec)


def write_csv(path, rows):
    import csv
    cols = ['element', 'formula', 'charge', 'geometry', 'natm', 'nao', 'xc',
            'gpu_e', 'cpu_e', 'dE', 'gpu_t', 'cpu_t', 'speedup',
            'gpu_cycles', 'cpu_cycles', 'gpu_conv', 'cpu_conv',
            'gpu_ngrids', 'cpu_ngrids', 'gpu_error', 'cpu_error']
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow(row)


def markdown(meta, rows, functionals, elements, png_name):
    out = []
    A = out.append
    A('# ECP single-point benchmark: GPU4PySCF vs PySCF\n')
    A('One closed-shell compound per element carrying a def2-ECP '
      f'({len(elements)} elements), each run with {len(functionals)} '
      'functionals on both devices.\n')

    A('> [!NOTE]')
    A('> Geometries are idealized VSEPR shapes with tabulated bond lengths, '
      'not optimized. Both devices see identical coordinates, so GPU/CPU '
      'comparisons are exact; the absolute energies are not thermochemical '
      'reference values.\n')

    A('## Settings\n')
    A('| key | value |')
    A('|---|---|')
    for key in ('basis', 'grid', 'conv_tol', 'max_cycle', 'cpu_threads',
                'repeat', 'gpu', 'contract_engine', 'cupy', 'pyscf',
                'gpu4pyscf', 'git_commit', 'hostname', 'slurm_job_id',
                'timestamp'):
        if meta.get(key) is not None:
            A(f'| {key} | `{meta[key]}` |')
    A(f'| functionals | `{" ".join(functionals)}` |')
    A('')
    A('CPU references are built fresh from the Mole (`pyscf.dft.RKS`), never '
      'via `to_cpu()` of a converged GPU object -- that inherits the GPU '
      'density and biases CPU timings low by ~6x.\n')

    # ---- correctness -------------------------------------------------------
    A('## GPU vs CPU energy agreement\n')
    A(f'Worst |E_GPU - E_CPU| across functionals, per system. '
      f'Threshold {DE_TOL:.0e} Ha.\n')
    A('| element | compound | chg | natm | nao | max abs dE (Ha) | worst functional | status |')
    A('|---|---|---:|---:|---:|---:|---|---|')
    for el in elements:
        sub = [r for r in rows if r['element'] == el and 'dE' in r]
        if not sub:
            A(f'| {el} | -- | | | | -- | -- | no paired runs |')
            continue
        worst = max(sub, key=lambda r: r['dE'])
        bad = [r for r in rows if r['element'] == el
               and (r.get('gpu_conv') is False or r.get('cpu_conv') is False)]
        errs = [r for r in rows if r['element'] == el
                and (r.get('gpu_error') or r.get('cpu_error'))]
        status = 'ok' if worst['dE'] < DE_TOL else f'**dE > {DE_TOL:.0e}**'
        if bad:
            status += f', {len(bad)} unconverged'
        if errs:
            status += f', {len(errs)} error'
        A(f'| {el} | {worst["formula"]} | {worst["charge"]:+d} | {worst["natm"]} '
          f'| {worst["nao"]} | {worst["dE"]:.2e} | {worst["xc"]} | {status} |')
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
    A('| element | compound | nao | functional | E_GPU (Ha) | E_CPU (Ha) | '
      'dE (Ha) | t_GPU (s) | t_CPU (s) | speedup | cyc GPU/CPU |')
    A('|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|')
    for row in sorted(rows, key=lambda r: (r['nao'], r['element'], r['xc'])):
        err = row.get('gpu_error') or row.get('cpu_error')
        if err:
            A(f'| {row["element"]} | {row["formula"]} | {row["nao"]} | '
              f'{row["xc"]} | ' + ' | '.join(['--'] * 6) + f' | `{err[:60]}` |')
            continue
        A(f'| {row["element"]} | {row["formula"]} | {row["nao"]} | {row["xc"]} '
          f'| {fmt(row.get("gpu_e"), ".8f")} | {fmt(row.get("cpu_e"), ".8f")} '
          f'| {fmt(row.get("dE"), ".2e")} | {fmt(row.get("gpu_t"), ".2f")} '
          f'| {fmt(row.get("cpu_t"), ".2f")} | {fmt(row.get("speedup"), ".2f")} '
          f'| {fmt(row.get("gpu_cycles"), "d")}/{fmt(row.get("cpu_cycles"), "d")} |')
    A('')

    if png_name:
        A('## Figures\n')
        A(f'![benchmark figures]({png_name})\n')
    return '\n'.join(out)


def plot(rows, functionals, path, meta):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print('matplotlib not installed -- skipping figures.\n'
              '  pip install matplotlib', file=sys.stderr)
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    threads = meta.get('cpu_threads', '?')
    fig.suptitle(f'ECP single-point benchmark: {meta.get("basis", "?")}, '
                 f'{meta.get("gpu", "GPU")} vs CPU @ {threads} threads',
                 fontsize=13)
    cmap = plt.get_cmap('tab10')
    colors = {xc: cmap(i % 10) for i, xc in enumerate(functionals)}

    # (a) wall time vs system size
    ax = axes[0][0]
    for xc in functionals:
        sub = sorted((r for r in rows if r['xc'] == xc and 'gpu_t' in r),
                     key=lambda r: r['nao'])
        if sub:
            ax.plot([r['nao'] for r in sub], [r['gpu_t'] for r in sub], 'o-',
                    color=colors[xc], ms=4, lw=1, label=f'{xc} GPU')
        sub = sorted((r for r in rows if r['xc'] == xc and 'cpu_t' in r),
                     key=lambda r: r['nao'])
        if sub:
            ax.plot([r['nao'] for r in sub], [r['cpu_t'] for r in sub], 's--',
                    color=colors[xc], ms=4, lw=1, alpha=0.55, label=f'{xc} CPU')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('number of AOs'); ax.set_ylabel('SCF wall time (s)')
    ax.set_title('(a) wall time vs system size\nsolid = GPU, dashed = CPU')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=6, ncol=2)

    # (b) speedup vs system size
    ax = axes[0][1]
    for xc in functionals:
        sub = sorted((r for r in rows if r['xc'] == xc and 'speedup' in r),
                     key=lambda r: r['nao'])
        if sub:
            ax.plot([r['nao'] for r in sub], [r['speedup'] for r in sub], 'o-',
                    color=colors[xc], ms=4, lw=1, label=xc)
    ax.axhline(1.0, color='k', ls=':', lw=1.5)
    ax.text(0.02, 0.94, 'above 1: GPU faster', transform=ax.transAxes, fontsize=8)
    ax.set_xscale('log')
    ax.set_xlabel('number of AOs'); ax.set_ylabel('speedup (t_CPU / t_GPU)')
    ax.set_title('(b) speedup vs system size')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7)

    # (c) speedup heat map, systems ordered by size
    ax = axes[1][0]
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
        vmax = np.nanmax(np.abs(np.log2(grid[np.isfinite(grid)])))
        im = ax.imshow(np.log2(grid), aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label('log2(speedup):  >0 GPU faster')
        for i in range(len(els)):
            for j in range(len(functionals)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f'{grid[i, j]:.1f}', ha='center', va='center',
                            fontsize=5.5)
    ax.set_xticks(range(len(functionals)))
    ax.set_xticklabels(functionals, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(els)))
    ax.set_yticklabels([f'{el} ({nao})' for nao, el in order], fontsize=6)
    ax.set_title('(c) speedup by element (AO count) and functional')

    # (d) GPU/CPU energy agreement
    ax = axes[1][1]
    pts = [(r['nao'], max(r['dE'], 1e-16), r['xc'])
           for r in rows if 'dE' in r]
    for xc in functionals:
        sub = [(n, d) for n, d, x in pts if x == xc]
        if sub:
            ax.scatter([n for n, _ in sub], [d for _, d in sub], s=18,
                       color=colors[xc], label=xc, alpha=0.8)
    ax.axhline(DE_TOL, color='r', ls='--', lw=1.2,
               label=f'{DE_TOL:.0e} Ha threshold')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('number of AOs'); ax.set_ylabel('|E_GPU - E_CPU| (Ha)')
    ax.set_title('(d) GPU/CPU energy agreement')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    print(f'wrote {path}')
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

    png = None if args.no_plot else plot(rows, functionals, png_path, meta)
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
    bad = [r for r in rows if r.get('gpu_conv') is False or r.get('cpu_conv') is False]
    err = [r for r in rows if r.get('gpu_error') or r.get('cpu_error')]
    if bad:
        print(f'unconverged: {", ".join(sorted({r["formula"] for r in bad}))}')
    if err:
        print(f'errors:      {", ".join(sorted({r["formula"] for r in err}))}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
