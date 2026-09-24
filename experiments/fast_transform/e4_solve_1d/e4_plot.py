"""Plots and summary tables for E4. Run after e4_bench.py:
    uv run --no-sync python experiments/fast_transform/e4_solve_1d/e4_plot.py
"""
import os
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = list(csv.DictReader(open(os.path.join(HERE, 'e4_results.csv'))))
for r in R:
    for k in ['approx_degree', 'n_expected', 'n_found', 'n_missing', 'n_spurious', 'n_calls_deg_ge_256']:
        r[k] = int(r[k])
    for k in ['solve_median_s', 'max_rel_err_vs_ref', 'transform_share']:
        r[k] = float(r[k])
PROBS = []
for r in R:
    if (r['problem'], r['approx_degree']) not in PROBS: PROBS.append((r['problem'], r['approx_degree']))
T = {(r['problem'], r['method']): r for r in R}
ALLM = []
for r in R:
    if r['method'] not in ALLM: ALLM.append(r['method'])
PROBS = sorted([pd for pd in PROBS if all((pd[0], m) in T for m in ALLM)], key=lambda t: t[1])  # complete problems only
degs = np.array([d for _, d in PROBS])
def col(m, key='solve_median_s'):
    return np.array([T[(p, m)][key] if (p, m) in T else np.nan for p, _ in PROBS])

INK, INK2, GRID, SURF = '#1f1f1e', '#5f5e58', '#e4e3dc', '#ffffff'
SERIES = [('dense', '#2a78d6', 'o'), ('fast', '#eb6834', 's'), ('auto256', '#1baf7a', '^'),
          ('auto1024', '#eda100', 'D'), ('auto2048', '#e87ba4', 'v'), ('autoT1024', '#4a3aa7', 'P')]
LABEL = {'dense': 'dense (default)', 'fast': 'fast (always NUFFT)', 'auto256': 'auto, min degree 256',
         'auto1024': 'auto, min degree 1024', 'auto2048': 'auto, min degree 2048',
         'autoT1024': 'auto 1024 + tail chop (prototype)'}

def style(ax):
    ax.set_facecolor(SURF)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, which='major', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# Figure 1: absolute solve time vs degree
fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
style(ax)
for m, c, mk in SERIES:
    if m not in ALLM: continue
    ax.plot(degs, col(m), ls='none', marker=mk, ms=7, color=c, mec=SURF, mew=1.2,
            label=LABEL[m], alpha=0.95)
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Chebyshev approximation degree of f on [a,b]', color=INK)
ax.set_ylabel('median solve time (s), 1 thread', color=INK)
ax.set_title('E4: 1D yroots.solve time vs degree, by transform method', color=INK, loc='left', fontsize=11)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'e4_solve_time_vs_degree.png'))

# Figure 2: time relative to dense
fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
style(ax)
ax.axhline(1.0, color=INK2, lw=1.2)
for m, c, mk in SERIES[1:]:
    if m not in ALLM: continue
    ax.plot(degs, col(m) / col('dense'), ls='none', marker=mk, ms=7, color=c, mec=SURF,
            mew=1.2, label=LABEL[m])
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Chebyshev approximation degree of f on [a,b]', color=INK)
ax.set_ylabel('solve time / dense solve time (below 1 = faster)', color=INK)
ax.set_title('E4: slowdown relative to the dense transform', color=INK, loc='left', fontsize=11)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'e4_speedup_vs_degree.png'))

# Summary tables (markdown) for results.md
methods = [m for m in ['dense', 'fast', 'auto64', 'auto128', 'auto256', 'auto512', 'auto1024', 'auto2048',
                       'auto4096', 'fastT', 'autoT512', 'autoT1024', 'autoT2048'] if m in ALLM]
def fmt(c, v):
    if isinstance(v, float) and c in methods: return f'{v:.3g}'
    if c in ('err_dense', 'err_worst'): return f'{v:.1e}'
    if c == 'dense_trans_share': return f'{100 * v:.1f}%'
    return str(v)
rows = []
for p, d in PROBS:
    g = {m: T[(p, m)] for m in methods if (p, m) in T}
    bad = [m for m in g if g[m]['n_missing'] or g[m]['n_spurious']]
    rows.append(dict(problem=p, deg=d, roots=f"{g['dense']['n_found']}/{g['dense']['n_expected']}",
                     **{m: g[m]['solve_median_s'] for m in methods},
                     dense_trans_share=g['dense']['transform_share'],
                     root_sets_ok='yes' if not bad else 'NO: ' + ','.join(bad),
                     err_dense=g['dense']['max_rel_err_vs_ref'],
                     err_worst=max(g[m]['max_rel_err_vs_ref'] for m in g)))
def md(rows, cols):
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(fmt(c, r[c]) for c in cols) + ' |')
    return '\n'.join(out) + '\n'
cols = ['problem', 'deg', 'roots'] + [m for m in ['dense', 'fast', 'auto256', 'auto512', 'auto1024', 'auto2048',
                                                   'auto4096', 'autoT1024'] if m in methods] + \
       ['dense_trans_share', 'root_sets_ok', 'err_dense', 'err_worst']
open(os.path.join(HERE, 'e4_summary_table.md'), 'w').write(md(rows, cols))
open(os.path.join(HERE, 'e4_summary_table_all_methods.md'), 'w').write(md(rows, ['problem', 'deg'] + methods))
crow = []
for p, d in PROBS:
    if d < 256: continue
    crow.append(dict(problem=p, deg=d, **{m: T[(p, m)]['n_calls_deg_ge_256'] for m in methods}))
open(os.path.join(HERE, 'e4_calls_ge256_table.md'), 'w').write(md(crow, ['problem', 'deg'] + methods))
fcrow = [dict(problem=p, deg=d, **{m: int(T[(p, m)]['n_fast_calls']) for m in methods}) for p, d in PROBS if d >= 256]
open(os.path.join(HERE, 'e4_fast_calls_table.md'), 'w').write(md(fcrow, ['problem', 'deg'] + methods))
for fn in ['e4_summary_table.md', 'e4_calls_ge256_table.md', 'e4_fast_calls_table.md']:
    print(fn); print(open(os.path.join(HERE, fn)).read())
