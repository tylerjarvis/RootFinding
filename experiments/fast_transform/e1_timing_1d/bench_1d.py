"""E1: 1D affine Chebyshev transform timing, dense (numba O(n^2)) vs fast (FINUFFT type-3 + DCT-I).

Run from repo root:
    NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python experiments/fast_transform/e1_timing_1d/bench_1d.py

Writes timings.csv, breakdown.csv, eps_threads.csv, accuracy.csv next to this file.
"""
import os
import sys
import time
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..', '..')))

import finufft
from scipy.fft import dct
import yroots.ChebyshevSubdivisionSolver as CSS
import yroots.FastTransform as FT

QUICK = '--quick' in sys.argv

# ---------------------------------------------------------------- timing helper
def bench(fn, target=0.25, min_rep=5, max_rep=2000, max_single=None):
    """Returns (median, p10, p90, nrep) of per-call time in seconds. One warmup call first."""
    t0 = time.perf_counter(); fn(); t1 = time.perf_counter()
    first = t1 - t0
    nrep = int(np.clip(target / max(first, 1e-7), min_rep, max_rep))
    if first > 1.0:
        nrep = 3
    ts = np.empty(nrep)
    for i in range(nrep):
        s = time.perf_counter(); fn(); ts[i] = time.perf_counter() - s
    return float(np.median(ts)), float(np.percentile(ts, 10)), float(np.percentile(ts, 90)), nrep

# ---------------------------------------------------------------- setup
sizes = []
k = 1
while 2 ** k <= 65536:
    sizes.append(2 ** k)
    if 3 * 2 ** (k - 1) < 65536 and k >= 1:
        sizes.append(3 * 2 ** (k - 1))
    k += 1
sizes = sorted(set(s for s in sizes if s >= 2))  # N = n+1 = number of coefficients
if QUICK:
    sizes = [s for s in sizes if s <= 4096]

params = [
    ('half_left', 0.5, -0.5),
    ('half_right', 0.5, 0.5),
    ('offcenter', 0.5 + 1e-3, -0.5 + 1e-3),
    ('zoom_0.1', 0.1, 0.3),
    ('zoom_1e-3', 1e-3, -0.2),
    ('zoom_1e-6', 1e-6, 0.7),
]
for name, a, b in params:
    assert abs(a) + abs(b) <= 1.0, name

rng = np.random.default_rng(12345)
def make_coeffs(N, kind):
    r = rng.uniform(-1, 1, N)
    if kind == 'unit':
        return r
    n = max(N - 1, 1)
    return r * 10.0 ** (-16.0 * np.arange(N) / n)

def dense_direct(c, a, b):
    return CSS.TransformChebInPlace1D(c, a, b)

def fast_direct(c, a, b):
    return FT.cheb_affine_fast_axis0(c, a, b, nthreads=1)

def nd_dense(c, a, b):
    CSS.TRANSFORM_METHOD = 'dense'
    return CSS.TransformChebInPlaceND(c, 0, a, b, False)

def nd_fast(c, a, b):
    CSS.TRANSFORM_METHOD = 'fast'
    return CSS.TransformChebInPlaceND(c, 0, a, b, False)

methods = [('dense', dense_direct), ('fast', fast_direct), ('nd_dense', nd_dense), ('nd_fast', nd_fast)]

FT.NUFFT_NTHREADS = 1
FT.NUFFT_EPS = 1e-15

# warm up numba compile
dense_direct(np.random.rand(10), 0.5, -0.5)

# ---------------------------------------------------------------- main timing sweep
rows = []
acc_rows = []
for kind in ['decay', 'unit']:
    for N in sizes:
        c = make_coeffs(N, kind)
        for pname, a, b in params:
            outs = {}
            for mname, fn in methods:
                med, p10, p90, nrep = bench(lambda: fn(c, a, b))
                out = fn(c, a, b)
                outs[mname] = out
                rows.append(dict(kind=kind, N=N, degree=N - 1, case=pname, alpha=a, beta=b,
                                 method=mname, median_s=med, p10_s=p10, p90_s=p90, nrep=nrep,
                                 out_len=len(out)))
            d, f = outs['dense'], outs['fast']
            m = len(d)
            err_head = np.max(np.abs(d - f[:m]))
            err_tail = np.max(np.abs(f[m:])) if len(f) > m else 0.0
            acc_rows.append(dict(kind=kind, N=N, case=pname, dense_len=m, maxabs_diff=err_head,
                                 fast_tail_max=err_tail, coef_scale=np.max(np.abs(c))))
        print(f'{kind} N={N} done', flush=True)
CSS.TRANSFORM_METHOD = 'dense'

def write(fname, rs):
    with open(os.path.join(HERE, fname), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rs[0].keys()))
        w.writeheader(); w.writerows(rs)
write('timings.csv', rows)
write('accuracy.csv', acc_rows)

# ---------------------------------------------------------------- fast path overhead breakdown
def fast_stages(c, a, b, eps=1e-15, nthreads=1):
    """Replica of cheb_affine_fast_axis0 with stage timers; returns dict of stage -> seconds."""
    T = {}
    t0 = time.perf_counter()
    shape = c.shape
    n = shape[0] - 1
    A = np.ascontiguousarray(c.reshape(n + 1, -1).T)
    ntrans = A.shape[0]
    t1 = time.perf_counter(); T['reshape'] = t1 - t0
    theta = np.pi * np.arange(n + 1) / n
    phi = np.arccos(np.clip(a * np.cos(theta) + b, -1.0, 1.0))
    t2 = time.perf_counter(); T['nodes'] = t2 - t1
    cc = np.empty((ntrans, 2 * n + 1), dtype=np.complex128)
    cc[:, n] = A[:, 0]
    cc[:, n + 1:] = 0.5 * A[:, 1:]
    cc[:, :n] = 0.5 * A[:, :0:-1]
    modes = np.arange(-n, n + 1, dtype=np.float64)
    if ntrans == 1:
        cc = cc[0]
    t3 = time.perf_counter(); T['coef_setup'] = t3 - t2
    values = finufft.nufft1d3(modes, cc, phi, eps=eps, isign=1, nthreads=nthreads).real
    t4 = time.perf_counter(); T['nufft'] = t4 - t3
    values = values.reshape(ntrans, n + 1)
    bb = dct(values, type=1, axis=1) / n
    bb[:, 0] *= 0.5
    bb[:, -1] *= 0.5
    t5 = time.perf_counter(); T['dct'] = t5 - t4
    out = np.ascontiguousarray(bb.T).reshape(shape)
    t6 = time.perf_counter(); T['output'] = t6 - t5
    T['total_staged'] = t6 - t0
    return T

br_rows = []
for N in [2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384, 65536]:
    if QUICK and N > 4096:
        continue
    c = make_coeffs(N, 'unit')
    a, b = 0.5, -0.5
    for _ in range(20):
        fast_stages(c, a, b)
    reps = 400 if N <= 1024 else (60 if N <= 16384 else 15)
    acc = {}
    for _ in range(reps):
        for k2, v in fast_stages(c, a, b).items():
            acc.setdefault(k2, []).append(v)
    whole = bench(lambda: FT.cheb_affine_fast_axis0(c, a, b, nthreads=1))[0]
    # the NUFFT call alone, pre-built inputs, and also finufft plan/setpts/execute split
    n = N - 1
    theta = np.pi * np.arange(n + 1) / n
    phi = np.arccos(np.clip(a * np.cos(theta) + b, -1.0, 1.0))
    cc = np.zeros(2 * n + 1, dtype=np.complex128); cc[n:] = c; cc[:n] = c[:0:-1]
    modes = np.arange(-n, n + 1, dtype=np.float64)
    def plan_only():
        p = finufft.Plan(3, 1, eps=1e-15, isign=1, nthreads=1)
        return p
    def plan_setpts():
        p = finufft.Plan(3, 1, eps=1e-15, isign=1, nthreads=1)
        p.setpts(modes, s=phi)
        return p
    def plan_setpts_exec():
        p = finufft.Plan(3, 1, eps=1e-15, isign=1, nthreads=1)
        p.setpts(modes, s=phi)
        return p.execute(cc)
    t_plan = bench(plan_only)[0]
    t_setpts = bench(plan_setpts)[0]
    t_exec = bench(plan_setpts_exec)[0]
    t_dct_only = bench(lambda: dct(phi, type=1))[0]
    row = dict(N=N, whole_fast_call=whole)
    for k2, v in acc.items():
        row[k2] = float(np.median(v))
    row.update(finufft_plan_create=t_plan, finufft_plan_plus_setpts=t_setpts,
               finufft_plan_setpts_execute=t_exec, scipy_dct1_only=t_dct_only)
    br_rows.append(row)
    print('breakdown', N, flush=True)
write('breakdown.csv', br_rows)

# ---------------------------------------------------------------- eps / threads at large N
et_rows = []
eps_list = [1e-15, 1e-12, 1e-9]
thread_list = [1, 0]  # 0 = FINUFFT auto (all cores)
Ns = [1024, 4096, 16384, 65536] if not QUICK else [1024, 4096]
for N in Ns:
    c = make_coeffs(N, 'unit')
    a, b = 0.5, -0.5
    ref = CSS.TransformChebInPlace1D(c, a, b) if N <= 16384 else None
    for eps in eps_list:
        for nt in thread_list:
            med, p10, p90, nrep = bench(lambda: FT.cheb_affine_fast_axis0(c, a, b, eps=eps, nthreads=nt))
            out = FT.cheb_affine_fast_axis0(c, a, b, eps=eps, nthreads=nt)
            err = float(np.max(np.abs(out[:len(ref)] - ref))) if ref is not None else np.nan
            et_rows.append(dict(N=N, eps=eps, nthreads=nt, median_s=med, p10_s=p10, p90_s=p90,
                                nrep=nrep, maxabs_err_vs_dense=err))
    print('eps/threads', N, flush=True)
write('eps_threads.csv', et_rows)
print('done')
