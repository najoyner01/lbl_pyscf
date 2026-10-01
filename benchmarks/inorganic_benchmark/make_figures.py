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

"""Publication figures: GPU vs CPU resources for the two def2-ECP sets.

    python make_figures.py            # both sets, all 36 metals each

Two sets, three figures each, written as separate PNG (300 dpi) + PDF files
plus a combined three-panel sheet:

    figures/closed_shell_{a,b,c}.{png,pdf}   results.jsonl          (36 closed-shell
                                             complexes: 30 neutrals + 6 anions)
    figures/open_shell_{a,b,c}.{png,pdf}     results_neutral.jsonl  (36 neutrals:
                                             30 closed shell + 6 open shell)
    figures/closed_shell.{png,pdf}, figures/open_shell.{png,pdf}   (a)+(b)+(c)

  (a) SCF wall time (s) vs number of AOs, GPU and CPU, B3LYP.
  (b) Speedup t(CPU)/t(GPU) vs number of AOs, one line per functional.
  (c) Speedup heat map, metal (ordered by AO count) x functional.

A third set, the transition-metal BASIS LADDER (`--sets tm_ladder`,
results_tm.jsonl), has its own layout because each metal appears at three
bases:

    figures/tm_ladder_a.{png,pdf}   Speedup vs AO count, one panel per
                                    functional, one colour per rung (DZ/TZ/QZ).
                                    This is the speed-up curve the ladder exists
                                    to measure: 19 metals x 3 bases = 57 points
                                    per functional across ~50-500 AO.
    figures/tm_ladder_b.{png,pdf}   Wall time vs AO count, GPU and CPU, per
                                    functional; rung by colour, device by marker.
    figures/tm_ladder_c.{png,pdf}   Speedup heat map, metal x (rung, functional).
    figures/tm_ladder.{png,pdf}     (a)+(b)+(c) on one sheet.

Unlike make_report.py, every metal in the set is plotted. Axis ticks are plain
numbers (no scientific notation); the time and speedup axes are logarithmic
because both span more than two decades.
"""

import argparse
import os
import sys
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from make_report import load, pair_runs, PUB_RC, _bold_ticks  # noqa: E402

# Okabe-Ito, one colour per functional, fixed order (never cycled). Adjacent
# pairs clear the colour-vision-deficiency floor (worst tpss/b3lyp dE 7.6,
# OKLab x100) with the per-functional marker shape as secondary encoding.
PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
LABEL = {'svwn': 'SVWN', 'pbe': 'PBE', 'tpss': 'TPSS', 'b3lyp': 'B3LYP',
         'pbe0': 'PBE0', 'm06-2x': 'M06-2X'}

# The ladder set colours by RUNG, not by functional: functionals get their own
# panel there. Three slots, fixed order smallest to largest basis, from the
# same Okabe-Ito set (blue / vermillion / green -- the three that clear the
# all-pairs CVD floor together). Marker shape is the secondary encoding.
RUNG_COLORS = OrderedDict([('DZ', '#0072B2'), ('TZ', '#D55E00'),
                           ('QZ', '#009E73')])
RUNG_MARKERS = {'DZ': 'o', 'TZ': 's', 'QZ': '^'}

SETS = OrderedDict([
    ('closed_shell', dict(
        results='results.jsonl',
        title='Closed-shell set: 30 neutrals + 6 anions, all RKS')),
    ('open_shell', dict(
        results='results_neutral.jsonl',
        title='Open-shell set: 36 neutrals, 6 open shell (UKS)')),
    ('tm_ladder', dict(
        results='results_tm.jsonl',
        title='Transition-metal basis ladder: 19 closed-shell complexes at '
              'def2-SVP / TZVP / QZVP',
        ladder=True)),
])


def plain_ticks(ax, axis='both'):
    """Plain decimal tick labels on log axes: 10, 100, 1000 rather than 10^2,
    and 0.1 / 0.5 / 2 rather than 5 x 10^-1."""
    import matplotlib.ticker as mt
    fmt = mt.FuncFormatter(lambda v, _: (f'{v:g}' if v >= 1 else f'{v:.2g}'))
    axes = [ax.xaxis, ax.yaxis] if axis == 'both' else [getattr(ax, axis + 'axis')]
    for a in axes:
        a.set_major_formatter(fmt)
        a.set_minor_formatter(mt.NullFormatter())


def log_yticks(ax, candidates):
    """Round-number ticks inside the current log y-range, printed plainly."""
    lo, hi = ax.get_ylim()
    ax.set_yticks([t for t in candidates if lo <= t <= hi])
    plain_ticks(ax, 'y')


def style_x(ax, naos):
    """Log x-axis with plain-number ticks at sensible round values."""
    ax.set_xscale('log')
    lo, hi = min(naos), max(naos)
    ticks = [t for t in (30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000)
             if lo * 0.9 <= t <= hi * 1.1]
    ax.set_xticks(ticks)
    ax.set_xlim(lo * 0.92, hi * 1.08)
    ax.set_xlabel('Number of atomic orbitals')
    plain_ticks(ax, 'x')


SUP = {'2': '\u00b2', '3': '\u00b3', '-': '\u207b'}


def pretty_formula(formula, charge):
    """`RhCl6_3-` -> `[RhCl6]\u00b3\u207b`, `TcO4-` -> `TcO4\u207b`, neutrals unchanged."""
    if charge == 0:
        return formula
    base = formula.split('_')[0].rstrip('-')
    if charge == -1:
        return base + SUP['-']
    return f'[{base}]' + SUP[str(abs(charge))] + SUP['-']


def label_for(row):
    """Metal tag for the heat map: element, formula and spin if open shell."""
    tag = f'{row["element"]}  {pretty_formula(row["formula"], row.get("charge", 0))}'
    if row.get('spin', 0):
        tag += f'  (2S={row["spin"]})'
    return tag


def hollow_unconverged(ax, pts, marker, color):
    """Overdraw points whose SCF hit max_cycle as hollow markers: their wall
    time measures the cycle cap, not a converged SCF."""
    bad = [(x, y) for x, y, ok in pts if not ok]
    if bad:
        ax.plot([b[0] for b in bad], [b[1] for b in bad], marker=marker, ls='none',
                ms=9, mfc='white', mec=color, mew=2.0, zorder=5)


def panel_a(ax, rows, colors, xc='b3lyp'):
    sub = [r for r in rows if r['xc'] == xc]
    naos = [r['nao'] for r in sub]
    for dev, mk, ls, lab, alpha in (('gpu', 'o', '-', 'GPU (A100)', 1.0),
                                    ('cpu', 's', '--', 'CPU (16 threads)', 0.75)):
        pts = sorted((r['nao'], r[f'{dev}_t'], r.get(f'{dev}_conv'))
                     for r in sub if f'{dev}_t' in r)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, ls=ls,
                color=colors[xc], ms=7, lw=2.2, alpha=alpha, label=lab)
        hollow_unconverged(ax, pts, mk, colors[xc])
    ax.set_yscale('log')
    style_x(ax, naos)
    log_yticks(ax, (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500))
    ax.set_ylabel('SCF wall time (s)')
    ax.set_title(f'(a)  {LABEL.get(xc, xc)} wall time vs system size', loc='left')
    leg = ax.legend(loc='upper left', handlelength=2.4, labelspacing=0.3)
    for t in leg.get_texts():
        t.set_fontweight('bold')
    _bold_ticks(ax)


def panel_b(ax, rows, functionals, colors):
    naos = [r['nao'] for r in rows]
    for i, xc in enumerate(functionals):
        pts = sorted((r['nao'], r['speedup'],
                      bool(r.get('gpu_conv') and r.get('cpu_conv')))
                     for r in rows if r['xc'] == xc and 'speedup' in r)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=MARKERS[i],
                ls='-', color=colors[xc], ms=6.5, lw=2.0, label=LABEL.get(xc, xc))
        hollow_unconverged(ax, pts, MARKERS[i], colors[xc])
    ax.axhline(1.0, color='k', lw=2.2, zorder=0)
    ax.set_yscale('log')
    style_x(ax, naos)
    log_yticks(ax, (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20))
    ax.set_ylabel('Speedup   t(CPU) / t(GPU)')
    ax.set_title('(b)  GPU speedup vs system size', loc='left')
    ax.text(0.02, 0.97, 'GPU faster', transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top', color='#333333')
    ax.text(0.02, 0.03, 'CPU faster', transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='bottom', color='#333333')
    leg = ax.legend(ncol=2, loc='lower right', handlelength=2.2,
                    columnspacing=1.0, labelspacing=0.3)
    for t in leg.get_texts():
        t.set_fontweight('bold')
    _bold_ticks(ax)


def panel_c(ax, fig, rows, functionals, unconverged_marker=True):
    """Heat map of log2(speedup), metals ordered by AO count (largest at top).
    Cells where either device hit the cycle cap are hatched: the wall time
    then measures max_cycle, not an SCF."""
    by_el = {}
    for r in rows:
        by_el.setdefault((r['nao'], r['element']), r)
    order = sorted(by_el)[::-1]
    grid = np.full((len(order), len(functionals)), np.nan)
    unconv = np.zeros_like(grid, dtype=bool)
    for i, key in enumerate(order):
        for j, xc in enumerate(functionals):
            hit = [r for r in rows if (r['nao'], r['element']) == key
                   and r['xc'] == xc and 'speedup' in r]
            if hit:
                grid[i, j] = hit[0]['speedup']
                unconv[i, j] = not (hit[0].get('gpu_conv') and hit[0].get('cpu_conv'))
    lg = np.log2(grid)
    vmax = np.nanmax(np.abs(lg))
    im = ax.imshow(lg, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.04)
    ticks = [t for t in range(-4, 5) if abs(t) <= vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{2.0**t:g}' + 'x' for t in ticks])
    cb.set_label('Speedup  t(CPU) / t(GPU)', fontsize=13, fontweight='bold')
    cb.outline.set_linewidth(2.0)
    cb.ax.tick_params(width=2.0, length=6, labelsize=12)
    for lab in cb.ax.get_yticklabels():
        lab.set_fontweight('bold')
    for i in range(len(order)):
        for j in range(len(functionals)):
            if not np.isfinite(grid[i, j]):
                continue
            shade = abs(lg[i, j]) / vmax if vmax else 0.0
            txt = f'{grid[i, j]:.1f}'
            if unconverged_marker and unconv[i, j]:
                txt += '*'
                ax.add_patch(plt_rect(j, i))
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if shade > 0.55 else 'black')
    ax.set_xticks(range(len(functionals)))
    ax.set_xticklabels([LABEL.get(x, x) for x in functionals], fontsize=12)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label_for(by_el[k]) + f'  [{k[0]}]' for k in order],
                       fontsize=9)
    ax.tick_params(length=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(True)
    ax.set_title('(c)  Speedup by metal (ordered by AO count) and functional',
                 loc='left')
    _bold_ticks(ax)
    if unconverged_marker and unconv.any():
        ax.text(1.0, -0.06, '* outlined: a device hit max_cycle, so its wall time is the cycle cap',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                fontweight='bold', color='#333333')


def plt_rect(j, i):
    from matplotlib.patches import Rectangle
    return Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                     edgecolor='#111111', lw=2.2, zorder=4)


# --------------------------------------------------------------------------- #
# basis-ladder figures

def rung_style(zeta):
    """(colour, marker) for a rung; unknown labels fall back to grey/circle
    so a custom --bases list still plots."""
    return RUNG_COLORS.get(zeta, '#7F7F7F'), RUNG_MARKERS.get(zeta, 'o')


def _rung_legend(ax, rungs, loc, extra=()):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=rung_style(z)[0], marker=rung_style(z)[1],
                      ls='none', ms=8, label=z) for z in rungs]
    handles += list(extra)
    leg = ax.legend(handles=handles, loc=loc, handlelength=1.6,
                    labelspacing=0.3, ncol=1)
    for t in leg.get_texts():
        t.set_fontweight('bold')


def ladder_speedup_panels(axes, rows, functionals, rungs):
    """One panel per functional: speedup vs nao, points coloured by rung.
    A thin grey line joins each metal's DZ->TZ->QZ points so the reader can
    follow one compound up the ladder; the coloured markers carry the rung."""
    naos = [r['nao'] for r in rows if 'speedup' in r]
    for ax, xc in zip(axes, functionals):
        sub = [r for r in rows if r['xc'] == xc and 'speedup' in r]
        # per-metal ladder paths, drawn first so markers sit on top
        for formula in sorted({r['formula'] for r in sub}):
            path = sorted((r['nao'], r['speedup']) for r in sub
                          if r['formula'] == formula)
            if len(path) > 1:
                ax.plot([p[0] for p in path], [p[1] for p in path], '-',
                        color='#BBBBBB', lw=1.2, zorder=1)
        for z in rungs:
            pts = sorted((r['nao'], r['speedup'],
                          bool(r.get('gpu_conv') and r.get('cpu_conv')))
                         for r in sub if r['zeta'] == z)
            c, mk = rung_style(z)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk,
                    ls='none', color=c, ms=8, mec='white', mew=1.0, zorder=3,
                    label=z)
            hollow_unconverged(ax, pts, mk, c)
        ax.axhline(1.0, color='k', lw=2.0, zorder=0)
        ax.set_yscale('log')
        style_x(ax, naos)
        log_yticks(ax, (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20))
        ax.set_title(LABEL.get(xc, xc), loc='left')
        _bold_ticks(ax)
    axes[0].set_ylabel('Speedup   t(CPU) / t(GPU)')
    axes[0].text(0.03, 0.97, 'GPU faster', transform=axes[0].transAxes,
                 fontsize=11, fontweight='bold', va='top', color='#333333')
    axes[0].text(0.03, 0.03, 'CPU faster', transform=axes[0].transAxes,
                 fontsize=11, fontweight='bold', va='bottom', color='#333333')
    _rung_legend(axes[-1], rungs, 'lower right')
    # share the y-range so the three panels read on one scale
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)


def ladder_walltime_panels(axes, rows, functionals, rungs):
    """One panel per functional: wall time vs nao for both devices. Rung by
    colour, device by fill: GPU filled markers, CPU hollow."""
    from matplotlib.lines import Line2D
    naos = [r['nao'] for r in rows]
    for ax, xc in zip(axes, functionals):
        sub = [r for r in rows if r['xc'] == xc]
        for z in rungs:
            c, mk = rung_style(z)
            for dev, mfc in (('gpu', c), ('cpu', 'white')):
                pts = sorted((r['nao'], r[f'{dev}_t']) for r in sub
                             if r['zeta'] == z and f'{dev}_t' in r)
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            marker=mk, ls='none', color=c, mfc=mfc, mew=2.0,
                            ms=8, zorder=3)
        ax.set_yscale('log')
        style_x(ax, naos)
        log_yticks(ax, (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000))
        ax.set_title(LABEL.get(xc, xc), loc='left')
        _bold_ticks(ax)
    axes[0].set_ylabel('SCF wall time (s)')
    dev_handles = [Line2D([], [], color='#333333', marker='o', ls='none',
                          ms=8, label='GPU (filled)'),
                   Line2D([], [], color='#333333', marker='o', ls='none',
                          ms=8, mfc='white', mew=2.0, label='CPU (hollow)')]
    _rung_legend(axes[-1], rungs, 'upper left', extra=dev_handles)
    lo = min(ax.get_ylim()[0] for ax in axes)
    hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lo, hi)


def ladder_heatmap(ax, fig, rows, functionals, rungs):
    """Metals (rows, ordered by TZ AO count) x (rung, functional) columns,
    grouped rung-major so the three functionals of one rung sit together."""
    tz = rungs[min(1, len(rungs) - 1)]
    size = {}
    for r in rows:
        if r['zeta'] == tz:
            size[r['formula']] = r['nao']
    for r in rows:
        size.setdefault(r['formula'], r['nao'])
    order = sorted(size, key=lambda f: -size[f])
    cols = [(z, xc) for z in rungs for xc in functionals]
    grid = np.full((len(order), len(cols)), np.nan)
    unconv = np.zeros_like(grid, dtype=bool)
    by_formula = {}
    for i, f in enumerate(order):
        for j, (z, xc) in enumerate(cols):
            hit = [r for r in rows if r['formula'] == f and r['zeta'] == z
                   and r['xc'] == xc and 'speedup' in r]
            if hit:
                grid[i, j] = hit[0]['speedup']
                unconv[i, j] = not (hit[0].get('gpu_conv') and hit[0].get('cpu_conv'))
                by_formula.setdefault(f, hit[0])
    lg = np.log2(grid)
    vmax = np.nanmax(np.abs(lg)) if np.isfinite(lg).any() else 1.0
    im = ax.imshow(lg, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.04)
    ticks = [t for t in range(-5, 6) if abs(t) <= vmax]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{2.0**t:g}' + 'x' for t in ticks])
    cb.set_label('Speedup  t(CPU) / t(GPU)', fontsize=13, fontweight='bold')
    cb.outline.set_linewidth(2.0)
    cb.ax.tick_params(width=2.0, length=6, labelsize=12)
    for lab in cb.ax.get_yticklabels():
        lab.set_fontweight('bold')
    for i in range(len(order)):
        for j in range(len(cols)):
            if not np.isfinite(grid[i, j]):
                continue
            shade = abs(lg[i, j]) / vmax if vmax else 0.0
            txt = f'{grid[i, j]:.1f}'
            if unconv[i, j]:
                txt += '*'
                ax.add_patch(plt_rect(j, i))
            ax.text(j, i, txt, ha='center', va='center', fontsize=9,
                    fontweight='bold', color='white' if shade > 0.55 else 'black')
    # rung separators
    for k in range(1, len(rungs)):
        ax.axvline(k * len(functionals) - 0.5, color='white', lw=3.0)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f'{z}\n{LABEL.get(xc, xc)}' for z, xc in cols],
                       fontsize=11)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label_for(by_formula[f]) + f'  [{size[f]} @ {tz}]'
                        for f in order], fontsize=9)
    ax.tick_params(length=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(True)
    ax.set_title(f'(c)  Speedup by metal (ordered by {tz} AO count), rung and '
                 'functional', loc='left')
    _bold_ticks(ax)
    if unconv.any():
        ax.text(1.0, -0.08, '* outlined: a device hit max_cycle, so its wall '
                'time is the cycle cap', transform=ax.transAxes, ha='right',
                va='top', fontsize=10, fontweight='bold', color='#333333')


def make_ladder(name, spec, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    meta, rows, functionals = load_set(os.path.join(HERE, spec['results']))
    for r in rows:
        r.setdefault('zeta', r.get('basis'))
    rungs = [z for z in RUNG_COLORS if any(r['zeta'] == z for r in rows)]
    rungs += sorted({r['zeta'] for r in rows} - set(rungs))
    n_el = len({r['element'] for r in rows})
    head = (f'{spec["title"]}  --  {n_el} metals, '
            f'{meta.get("gpu", "GPU")} vs CPU @ {meta.get("cpu_threads", "?")} '
            'threads')
    nf = len(functionals)

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, nf, figsize=(5.2 * nf, 5.2), sharey=True)
        axes = np.atleast_1d(axes)
        ladder_speedup_panels(axes, rows, functionals, rungs)
        fig.suptitle('(a)  GPU speedup vs system size, by basis rung',
                     x=0.01, ha='left', fontsize=15, fontweight='bold')
        fig.tight_layout()
        save(fig, os.path.join(outdir, f'{name}_a'))
        plt.close(fig)

        fig, axes = plt.subplots(1, nf, figsize=(5.2 * nf, 5.2), sharey=True)
        axes = np.atleast_1d(axes)
        ladder_walltime_panels(axes, rows, functionals, rungs)
        fig.suptitle('(b)  SCF wall time vs system size, GPU and CPU',
                     x=0.01, ha='left', fontsize=15, fontweight='bold')
        fig.tight_layout()
        save(fig, os.path.join(outdir, f'{name}_b'))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4 + 1.1 * nf * len(rungs), 9))
        ladder_heatmap(ax, fig, rows, functionals, rungs)
        fig.tight_layout()
        save(fig, os.path.join(outdir, f'{name}_c'))
        plt.close(fig)

        fig = plt.figure(figsize=(17, 22), layout='constrained')
        gs = fig.add_gridspec(3, nf, height_ratios=[1, 1, 2.0])
        fig.suptitle(head + '\n', fontsize=17, fontweight='bold')
        top = [fig.add_subplot(gs[0, j]) for j in range(nf)]
        ladder_speedup_panels(top, rows, functionals, rungs)
        top[0].set_title('(a)  ' + top[0].get_title(loc='left'), loc='left')
        mid = [fig.add_subplot(gs[1, j]) for j in range(nf)]
        ladder_walltime_panels(mid, rows, functionals, rungs)
        mid[0].set_title('(b)  ' + mid[0].get_title(loc='left'), loc='left')
        ax = fig.add_subplot(gs[2, :])
        ladder_heatmap(ax, fig, rows, functionals, rungs)
        save(fig, os.path.join(outdir, name))
        plt.close(fig)

    paired = [r for r in rows if 'speedup' in r]
    sp = np.array([r['speedup'] for r in paired])
    print(f'  {name}: {len(paired)} paired runs, speedup min {sp.min():.2f}x '
          f'median {np.median(sp):.2f}x max {sp.max():.2f}x; '
          f'GPU faster in {(sp > 1).sum()} of {len(sp)}')
    for z in rungs:
        zr = [r for r in paired if r['zeta'] == z]
        naos = [r['nao'] for r in zr]
        line = f'    {z}: {min(naos)}-{max(naos)} AO;'
        for xc in functionals:
            s_ = np.array([r['speedup'] for r in zr if r['xc'] == xc])
            if len(s_):
                line += f' {xc} median {np.median(s_):.2f}x'
        print(line)


def load_set(path):
    meta, runs = load(path)
    rows = pair_runs(runs)
    functionals = meta.get('functionals') or list(
        OrderedDict.fromkeys(r['xc'] for r in rows))
    return meta, rows, functionals


def save(fig, base):
    fig.savefig(base + '.png', dpi=300)
    fig.savefig(base + '.pdf')
    print(f'wrote {base}.png / .pdf')


def make_set(name, spec, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    meta, rows, functionals = load_set(os.path.join(HERE, spec['results']))
    colors = {xc: PALETTE[i % len(PALETTE)] for i, xc in enumerate(functionals)}
    n_el = len({r['element'] for r in rows})
    head = (f'{spec["title"]}  --  {n_el} metals, {meta.get("basis", "?")}, '
            f'{meta.get("gpu", "GPU")} vs CPU @ {meta.get("cpu_threads", "?")} threads')

    with plt.rc_context(PUB_RC):
        # standalone panels
        fig, ax = plt.subplots(figsize=(8, 6))
        panel_a(ax, rows, colors)
        fig.tight_layout()
        save(fig, os.path.join(outdir, f'{name}_a'))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        panel_b(ax, rows, functionals, colors)
        fig.tight_layout()
        save(fig, os.path.join(outdir, f'{name}_b'))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 12))
        panel_c(ax, fig, rows, functionals)
        fig.tight_layout()
        save(fig, os.path.join(outdir, f'{name}_c'))
        plt.close(fig)

        # combined sheet
        fig = plt.figure(figsize=(17, 20), layout='constrained')
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.9])
        fig.suptitle(head + '\n', fontsize=17, fontweight='bold')
        panel_a(fig.add_subplot(gs[0, 0]), rows, colors)
        panel_b(fig.add_subplot(gs[0, 1]), rows, functionals, colors)
        ax = fig.add_subplot(gs[1, :])
        panel_c(ax, fig, rows, functionals)
        save(fig, os.path.join(outdir, name))
        plt.close(fig)

    # console summary the caption can quote
    paired = [r for r in rows if 'speedup' in r]
    sp = np.array([r['speedup'] for r in paired])
    print(f'  {name}: {len(paired)} paired runs, speedup min {sp.min():.2f}x '
          f'median {np.median(sp):.2f}x max {sp.max():.2f}x; '
          f'GPU faster in {(sp > 1).sum()} of {len(sp)}')
    for xc in functionals:
        s = np.array([r['speedup'] for r in paired if r['xc'] == xc])
        big = [r for r in paired if r['xc'] == xc and r['nao'] >= 200]
        sb = np.array([r['speedup'] for r in big])
        print(f'    {xc:7s} median {np.median(s):.2f}x, >=200 AO median '
              f'{np.median(sb):.2f}x (n={len(sb)})')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sets', nargs='+', choices=list(SETS), default=list(SETS))
    p.add_argument('--outdir', default=os.path.join(HERE, 'figures'))
    args = p.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    for name in args.sets:
        if SETS[name].get('ladder'):
            make_ladder(name, SETS[name], args.outdir)
        else:
            make_set(name, SETS[name], args.outdir)


if __name__ == '__main__':
    main()
