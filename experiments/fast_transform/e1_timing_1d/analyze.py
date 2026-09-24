"""Plots and summary numbers for E1. Run after bench_1d.py (from repo root, uv run --no-sync python ...)."""
import os
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

def load(name):
    with open(os.path.join(HERE, name)) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows

T = load('timings.csv')
cases = list(dict.fromkeys(r['case'] for r in T))
kinds = ['decay', 'unit']
methods = ['dense', 'fast', 'nd_dense', 'nd_fast']
# series[(kind, case, method)] -> arrays sorted by degree
series = defaultdict(list)
ab = {}
for r in T:
    series[(r['kind'], r['case'], r['method'])].append(r)
    ab[r['case']] = (r['alpha'], r['beta'])
for k in series:
    s = sorted(series[k], key=lambda r: r['degree'])
    series[k] = {f: np.array([r[f] for r in s]) for f in ['degree', 'median_s', 'p10_s', 'p90_s', 'out_len']}

# ------------------------------------------------ plot
fig, axes = plt.subplots(2, len(cases), figsize=(4 * len(cases), 7.5), sharex=True, sharey=True)
style = {'dense': ('#1f77b4', '-', 'o'), 'fast': ('#d62728', '-', 's'),
         'nd_dense': ('#1f77b4', ':', None), 'nd_fast': ('#d62728', ':', None)}
for i, kind in enumerate(kinds):
    for j, case in enumerate(cases):
        ax = axes[i, j]
        for m, (col, ls, mk) in style.items():
            s = series[(kind, case, m)]
            ax.plot(s['degree'], s['median_s'] * 1e6, color=col, ls=ls, marker=mk, ms=3, lw=1.3, label=m)
            if m in ('dense', 'fast'):
                ax.fill_between(s['degree'], s['p10_s'] * 1e6, s['p90_s'] * 1e6, color=col, alpha=0.15, lw=0)
        ax.set_xscale('log'); ax.set_yscale('log')
        a, b = ab[case]
        ax.set_title(f'{kind}: {case}\n(alpha={a:g}, beta={b:g})', fontsize=9)
        ax.grid(True, which='major', alpha=0.3)
        if i == 1: ax.set_xlabel('degree n')
        if j == 0: ax.set_ylabel('median time [us]')
axes[0, 0].legend(fontsize=8)
fig.suptitle('E1: 1D affine Chebyshev transform, dense (numba O(n^2)) vs fast (FINUFFT t3 + DCT-I), 1 thread; bands = p10-p90')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'time_vs_degree.png'), dpi=130)

# ------------------------------------------------ crossover + speedups
def ratio(kind, case, dm, fm):
    d, f = series[(kind, case, dm)], series[(kind, case, fm)]
    return d['degree'], d['median_s'] / f['median_s']

def crossover(deg, r):
    """Smallest degree beyond which fast stays faster (log-interpolated)."""
    for k in range(len(deg)):
        if np.all(r[k:] > 1):
            if k == 0:
                return deg[0]
            x0, x1 = np.log(deg[k - 1]), np.log(deg[k])
            y0, y1 = np.log(r[k - 1]), np.log(r[k])
            return float(np.exp(x0 - y0 * (x1 - x0) / (y1 - y0)))
    return np.inf

def at(deg, r, n):
    return float(np.exp(np.interp(np.log(n), np.log(deg), np.log(r))))

fmt = lambda v: 'never' if not np.isfinite(v) else f'{v:.0f}'
lines = ['| coeffs | case | alpha | beta | crossover n (direct) | crossover n (via ND) | speedup n=1e3 | n=1e4 | n=65535 | dense out len @65535 |',
         '|---|---|---|---|---|---|---|---|---|---|']
for kind in kinds:
    for case in cases:
        deg, r = ratio(kind, case, 'dense', 'fast')
        degn, rn = ratio(kind, case, 'nd_dense', 'nd_fast')
        a, b = ab[case]
        lines.append(f'| {kind} | {case} | {a:g} | {b:g} | {fmt(crossover(deg, r))} | {fmt(crossover(degn, rn))} | '
                     f'{at(deg, r, 1e3):.3g}x | {at(deg, r, 1e4):.3g}x | {r[-1]:.3g}x | {int(series[(kind, case, "dense")]["out_len"][-1])} |')
tbl = '\n'.join(lines)
print(tbl)

# absolute times at selected degrees
sel = [1, 7, 31, 127, 511, 1023, 4095, 16383, 65535]
abs_lines = ['| coeffs | case | degree | dense us | fast us | nd_dense us | nd_fast us |', '|---|---|---|---|---|---|---|']
for kind in kinds:
    for case in cases:
        for n in sel:
            vals = []
            for m in methods:
                s = series[(kind, case, m)]
                idx = np.where(s['degree'] == n)[0]
                vals.append(f"{s['median_s'][idx[0]] * 1e6:.3g}" if len(idx) else '-')
            abs_lines.append(f'| {kind} | {case} | {n} | ' + ' | '.join(vals) + ' |')
with open(os.path.join(HERE, 'summary_tables.md'), 'w') as fh:
    fh.write('## Crossover and speedups (dense time / fast time)\n\n' + tbl + '\n\n## Median times (us)\n\n' + '\n'.join(abs_lines) + '\n')

# ND dispatch overhead: nd_x - x at small n
print('\nND dispatch overhead (median over cases/kinds, degree<=31), us:')
for dm, m in [('dense', 'nd_dense'), ('fast', 'nd_fast')]:
    diffs = []
    for kind in kinds:
        for case in cases:
            a, b = series[(kind, case, dm)], series[(kind, case, m)]
            mask = a['degree'] <= 31
            diffs.extend((b['median_s'][mask] - a['median_s'][mask]) * 1e6)
    print(f'  {m} - {dm}: median {np.median(diffs):.2f} us (p10 {np.percentile(diffs, 10):.2f}, p90 {np.percentile(diffs, 90):.2f})')

print('\nSpread (p90/p10) of fast/dense direct, median over all points:')
for m in ['dense', 'fast']:
    sp = np.concatenate([series[(k, c, m)]['p90_s'] / series[(k, c, m)]['p10_s'] for k in kinds for c in cases])
    print(f'  {m}: median p90/p10 = {np.median(sp):.3f}, max = {sp.max():.2f}')

br = load('breakdown.csv')
print('\nFast-path breakdown (us):')
keys = [k for k in br[0].keys() if k != 'N']
print('N ' + ' '.join(keys))
for r in br:
    print(int(r['N']), ' '.join(f'{r[k] * 1e6:.1f}' for k in keys))

et = load('eps_threads.csv')
print('\nEPS / threads (ms):')
for r in et:
    print(f"N={int(r['N'])} eps={r['eps']:g} nthreads={int(r['nthreads'])}: {r['median_s'] * 1e3:.3f} ms "
          f"[p10 {r['p10_s'] * 1e3:.3f}, p90 {r['p90_s'] * 1e3:.3f}] err_vs_dense={r['maxabs_err_vs_dense']:.2e}")

acc = load('accuracy.csv')
print('\nMax |dense - fast| (over all N) and max |fast tail beyond dense truncation|:')
for kind in kinds:
    for case in cases:
        rs = [r for r in acc if r['kind'] == kind and r['case'] == case]
        print(f"  {kind} {case}: diff {max(r['maxabs_diff'] for r in rs):.2e}, tail {max(r['fast_tail_max'] for r in rs):.2e}")
