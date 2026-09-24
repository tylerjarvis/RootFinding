"""E4: end-to-end 1D rootfinding with dense vs fast (NUFFT) vs auto affine Chebyshev transforms.

Run from repo root:
    uv run --no-sync python experiments/fast_transform/e4_solve_1d/e4_bench.py [--quick] [--only NAME,...]

Outputs (in this directory):
    e4_results.csv      one row per (problem, method): timings, transform stats, root agreement
    e4_transform_calls.csv  per (problem, method, degree bin): #calls, #fast calls, total/mean time
    e4_refs.npz         reference roots per problem
"""
import os, sys, time, csv, argparse, math
import numpy as np
import scipy.special as sps
from scipy.optimize import brentq
import mpmath

import yroots as yr
import yroots.ChebyshevSubdivisionSolver as C
import yroots.Combined_Solver as CS
import yroots.ChebyshevApproximator as CA
from yroots import FastTransform as F

HERE = os.path.dirname(os.path.abspath(__file__))
F.NUFFT_NTHREADS = 1
mpmath.mp.dps = 40

# ----------------------------------------------------------------------------------------
# Reference-root helpers
# ----------------------------------------------------------------------------------------
def inside(r, a, b):
    r = np.asarray(r, float)
    return np.sort(r[(r >= a) & (r <= b)])

def grid_refine_roots(f, fm, a, b, npts):
    """Independent reference: sign changes on a fine grid, brentq, then mpmath.findroot polish."""
    x = np.linspace(a, b, npts)
    y = f(x)
    roots = list(x[y == 0])
    idx = np.nonzero(np.sign(y[:-1]) * np.sign(y[1:]) < 0)[0]
    for i in idx:
        roots.append(brentq(f, x[i], x[i + 1], xtol=1e-15, rtol=1e-15))
    out = []
    for r in roots:
        try:
            out.append(float(mpmath.findroot(fm, mpmath.mpf(r))))
        except Exception:
            out.append(r)
    return np.sort(np.array(out))

def cheb_T(n):
    c = np.zeros(n + 1); c[n] = 1.0
    return c

# ----------------------------------------------------------------------------------------
# Problem suite
# ----------------------------------------------------------------------------------------
def make_problems():
    P = []
    def add(name, f, a, b, ref, note=''):
        P.append(dict(name=name, f=f, a=a, b=b, ref=ref, note=note))

    # low-degree polynomials
    add('poly3_callable', lambda x: (x - 0.1) * (x + 0.5) * (x - 0.7), -1, 1, np.array([-0.5, 0.1, 0.7]))
    pr = np.array([-0.95, -0.4, 0.2, 0.9])
    add('poly4_MultiPower', yr.MultiPower(np.poly(pr)[::-1].copy()), -1, 1, pr)

    # sin(omega x), roots k*pi/omega
    for w in [10, 100, 1000, 5000, 20000]:
        k = np.arange(-int(w / np.pi) - 1, int(w / np.pi) + 2)
        add(f'sin_{w}x', (lambda w: lambda x: np.sin(w * x))(w), -1, 1, inside(k * np.pi / w, -1, 1))

    # cos(omega x^2), roots x = +-sqrt((k+1/2) pi / omega)
    for w in [100, 1000, 5000]:
        k = np.arange(0, int(w / np.pi) + 2)
        s = np.sqrt((k + 0.5) * np.pi / w)
        add(f'cos_{w}x2', (lambda w: lambda x: np.cos(w * x * x))(w), -1, 1, inside(np.r_[-s, s], -1, 1))

    # Chebyshev T_n: roots cos((2k-1) pi / (2n))
    for n in [100, 1000, 4000]:
        k = np.arange(1, n + 1)
        add(f'T{n}_MultiCheb', yr.MultiCheb(cheb_T(n)), -1, 1, np.sort(np.cos((2 * k - 1) * np.pi / (2 * n))))
    n = 300
    k = np.arange(1, n + 1)
    add(f'T{n}_callable', lambda x, n=n: np.polynomial.chebyshev.chebval(x, cheb_T(n)), -1, 1,
        np.sort(np.cos((2 * k - 1) * np.pi / (2 * n))))

    # product sin(100x)(x-0.3)
    k = np.arange(-40, 41)
    add('sin100x_times_x-0.3', lambda x: np.sin(100 * x) * (x - 0.3), -1, 1, inside(np.r_[k * np.pi / 100, 0.3], -1, 1))

    # Bessel J0(omega x) on [-1,1], roots +- j_{0,k}/omega
    for w in [300, 3000]:
        z = sps.jn_zeros(0, int(w / np.pi) + 5) / w
        add(f'J0_{w}x', (lambda w: lambda x: sps.j0(w * x))(w), -1, 1, inside(np.r_[-z, z], -1, 1))

    # exp(x) - cos(omega x): reference by grid+brentq+mpmath polish
    for w in [30, 300]:
        f = (lambda w: lambda x: np.exp(x) - np.cos(w * x))(w)
        fm = (lambda w: lambda x: mpmath.exp(x) - mpmath.cos(w * x))(w)
        add(f'exp_minus_cos{w}x', f, -1, 1, ('grid', f, fm), note='ref: grid sign changes + brentq + mpmath.findroot')

    # other intervals
    add('sin_x_[1,3000]', lambda x: np.sin(x), 1.0, 3000.0, inside(np.arange(0, 1000) * np.pi, 1, 3000))
    add('sin_x_[-2000,6000]', lambda x: np.sin(x), -2000.0, 6000.0,
        inside(np.arange(-700, 2000) * np.pi, -2000, 6000))
    z = sps.jn_zeros(0, 700)
    add('J0_x_[0,2000]', lambda x: sps.j0(x), 0.0, 2000.0, inside(z, 0, 2000))
    add('exp_minus_cos30x_[-3,0.5]', lambda x: np.exp(x) - np.cos(30 * x), -3.0, 0.5,
        ('grid', lambda x: np.exp(x) - np.cos(30 * x), lambda x: mpmath.exp(x) - mpmath.cos(30 * x)),
        note='ref: grid+brentq+mpmath')
    return P

def resolve_ref(p):
    if isinstance(p['ref'], tuple) and p['ref'][0] == 'grid':
        _, f, fm = p['ref']
        p['ref'] = grid_refine_roots(f, fm, p['a'], p['b'], 8_000_001)
    return p['ref']

# ----------------------------------------------------------------------------------------
# Instrumentation
# ----------------------------------------------------------------------------------------
_orig_T = C.TransformChebInPlaceND
MACHEPS = 2.0 ** -52
CHOP_FRAC = float(os.environ.get('E4_CHOP_FRAC', '1.0'))  # fraction of the transform-error budget the chop may use
TRUNCATE_FAST = False   # prototype variant ('fastT', 'autoT<k>'): chop the NUFFT output's noise tail

def _truncT(coeffs, dim, alpha, beta, exact):
    """Prototype (experiment-only, yroots untouched): like TransformChebInPlaceND, but after a NUFFT
    transform drop trailing rows along `dim` while their cumulative abs-sum stays below the error the
    solver already budgets for this transform (getTransformationError: n * macheps * absSum(M)).
    Mimics the degree reduction the dense TransformChebInPlace1D gets from its |entry|<=1e-16 cutoff."""
    if (alpha == 1.0 and beta == 0.0) or coeffs.shape[dim] == 1:
        return coeffs
    if not C.useFastTransform(coeffs, dim, alpha, beta, exact):
        return _orig_T(coeffs, dim, alpha, beta, exact)
    out = _orig_T(coeffs, dim, alpha, beta, exact)
    budget = CHOP_FRAC * coeffs.shape[dim] * MACHEPS * np.sum(np.abs(coeffs))
    rows = np.moveaxis(np.abs(out), dim, 0).reshape(out.shape[dim], -1).sum(axis=1)
    tail = np.cumsum(rows[::-1])[::-1]          # tail[k] = sum_{j>=k} rows[j]
    keep = int(np.argmax(tail <= budget)) if tail[-1] <= budget else len(rows)
    keep = max(keep, 3)
    if keep >= out.shape[dim]:
        return out
    sl = [slice(None)] * out.ndim; sl[dim] = slice(0, keep)
    return out[tuple(sl)]

def base_T(*a):
    return (_truncT if TRUNCATE_FAST else _orig_T)(*a)
_orig_approx = CS.ChebyshevApproximator.chebApproximate
CALLS = []    # (degree, ndim, fast?, seconds)
APPROX = []   # (shape, seconds)

def _wrapped_T(coeffs, dim, alpha, beta, exact):
    use_fast = C.useFastTransform(coeffs, dim, alpha, beta, exact)
    trivial = (alpha == 1.0 and beta == 0.0) or coeffs.shape[dim] == 1
    t = time.perf_counter()
    out = base_T(coeffs, dim, alpha, beta, exact)
    dt = time.perf_counter() - t
    CALLS.append((coeffs.shape[dim] - 1, coeffs.ndim, bool(use_fast and not trivial), dt,
                  abs(alpha) + abs(beta) > 1.0))
    return out

def _wrapped_approx(f, a, b, *args, **kw):
    t = time.perf_counter()
    out = _orig_approx(f, a, b, *args, **kw)
    APPROX.append((out[0].shape, time.perf_counter() - t))
    return out

def instrument(on):
    C.TransformChebInPlaceND = _wrapped_T if on else (_truncT if TRUNCATE_FAST else _orig_T)
    CS.ChebyshevApproximator.chebApproximate = _wrapped_approx if on else _orig_approx

def set_method(m):
    global TRUNCATE_FAST
    TRUNCATE_FAST = m.startswith('fastT') or m.startswith('autoT')
    m = m.replace('fastT', 'fast').replace('autoT', 'auto')
    if m == 'dense':
        C.TRANSFORM_METHOD = 'dense'
    elif m == 'fast':
        C.TRANSFORM_METHOD = 'fast'
    else:
        C.TRANSFORM_METHOD = 'auto'
        C.FAST_TRANSFORM_MIN_DEGREE = int(m.split('auto')[1])
        C.FAST_TRANSFORM_NDIMS = (1,)

def run_solve(p):
    return np.asarray(yr.solve(p['f'], p['a'], p['b'], max_cpu=1)).reshape(-1)

# ----------------------------------------------------------------------------------------
# Root comparison
# ----------------------------------------------------------------------------------------
def compare(found, ref, scale, tol_rel=1e-7):
    found = np.sort(np.asarray(found, float)); ref = np.sort(np.asarray(ref, float))
    tol = tol_rel * scale
    if len(ref) == 0:
        return dict(n_missing=0, n_spurious=len(found), max_err=np.nan)
    if len(found) == 0:
        return dict(n_missing=len(ref), n_spurious=0, max_err=np.nan)
    def nearest(xs, ys):
        i = np.clip(np.searchsorted(ys, xs), 1, len(ys) - 1)
        left, right = ys[i - 1], ys[i]
        j = np.where(np.abs(xs - left) <= np.abs(xs - right), i - 1, i)
        return j, np.abs(xs - ys[j])
    jf, df = nearest(found, ref) if len(ref) > 1 else (np.zeros(len(found), int), np.abs(found - ref[0]))
    jr, dr = nearest(ref, found) if len(found) > 1 else (np.zeros(len(ref), int), np.abs(ref - found[0]))
    matched = df <= tol
    # duplicates: several found roots matched to the same ref root
    dup = len(jf[matched]) - len(np.unique(jf[matched]))
    return dict(n_missing=int(np.sum(dr > tol)), n_spurious=int(np.sum(~matched) + dup),
                max_err=float(np.max(df[matched]) / scale) if matched.any() else np.nan)

# ----------------------------------------------------------------------------------------
BINS = [0, 16, 64, 128, 256, 512, 1024, 4096, 10 ** 9]
def bin_label(d):
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        if lo <= d < hi:
            return f'[{lo},{hi})' if hi < 10 ** 9 else f'>={lo}'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--only', default='')
    ap.add_argument('--methods', default='dense,fast,auto64,auto128,auto256,auto512,auto1024,auto2048,auto4096,fastT,autoT512,autoT1024,autoT2048')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    methods = args.methods.split(',')
    probs = make_problems()
    if args.only:
        keep = set(args.only.split(';'))
        probs = [p for p in probs if p['name'] in keep]

    res_path = os.path.join(HERE, f'e4_results{args.tag}.csv')
    call_path = os.path.join(HERE, f'e4_transform_calls{args.tag}.csv')
    res_f = open(res_path, 'w', newline=''); call_f = open(call_path, 'w', newline='')
    rw = csv.writer(res_f); cw = csv.writer(call_f)
    rw.writerow(['problem', 'a', 'b', 'approx_degree', 'n_approx_calls', 'method', 'n_repeats',
                 'solve_median_s', 'solve_min_s', 'solve_all_s', 'approx_time_s_instr', 'transform_time_s_instr',
                 'instr_total_s', 'transform_share', 'n_transform_calls', 'n_fast_calls', 'n_nonunit_map_calls',
                 'max_transform_degree', 'median_transform_degree', 'n_calls_deg_ge_256',
                 'n_expected', 'n_found', 'n_missing', 'n_spurious', 'max_rel_err_vs_ref',
                 'max_diff_vs_dense', 'same_count_as_dense', 'note'])
    cw.writerow(['problem', 'method', 'degree_bin', 'n_calls', 'n_fast', 'total_s', 'mean_us_dense_calls',
                 'mean_us_fast_calls'])
    refs = {}
    for p in probs:
        ref = resolve_ref(p)
        refs[p['name']] = ref
        scale = max(1.0, abs(p['a']), abs(p['b']))
        dense_roots = None
        for m in methods:
            set_method(m)
            instrument(False)
            t0 = time.perf_counter(); r = run_solve(p); tw = time.perf_counter() - t0  # warm-up (numba JIT / caches)
            times = []
            nrep = 5
            t0 = time.perf_counter(); r = run_solve(p); t1 = time.perf_counter() - t0
            times.append(t1)
            if tw > 15: times.append(tw); nrep = 1   # JIT cost negligible vs. run; reuse warm-up as a sample
            elif t1 > 3: nrep = 3
            if args.quick: nrep = 1
            for _ in range(nrep - 1):
                t0 = time.perf_counter(); r = run_solve(p); times.append(time.perf_counter() - t0)
            # instrumented run
            CALLS.clear(); APPROX.clear()
            instrument(True)
            t0 = time.perf_counter(); r_i = run_solve(p); t_instr = time.perf_counter() - t0
            instrument(False)
            calls = list(CALLS); approx = list(APPROX)
            degs = np.array([c[0] for c in calls]) if calls else np.zeros(0, int)
            ttrans = sum(c[3] for c in calls)
            tapprox = sum(a[1] for a in approx)
            cmp = compare(r, ref, scale)
            if m == 'dense':
                dense_roots = np.sort(r)
            rs = np.sort(r)
            if dense_roots is not None and len(rs) == len(dense_roots):
                mdiff = float(np.max(np.abs(rs - dense_roots)) / scale) if len(rs) else 0.0
                same = True
            else:
                mdiff, same = np.nan, False
            top_deg = approx[0][0][0] - 1 if approx else (p['f'].coeff.shape[0] - 1)
            rw.writerow([p['name'], p['a'], p['b'], top_deg, len(approx), m, len(times),
                         f'{np.median(times):.6g}', f'{np.min(times):.6g}', ';'.join(f'{t:.4g}' for t in times),
                         f'{tapprox:.6g}', f'{ttrans:.6g}', f'{t_instr:.6g}', f'{ttrans / t_instr:.4f}',
                         len(calls), sum(c[2] for c in calls), sum(c[4] for c in calls),
                         int(degs.max()) if len(degs) else 0, float(np.median(degs)) if len(degs) else 0,
                         int(np.sum(degs >= 256)),
                         len(ref), len(r), cmp['n_missing'], cmp['n_spurious'], f"{cmp['max_err']:.3e}",
                         f'{mdiff:.3e}', same, p['note']])
            res_f.flush()
            by = {}
            for c in calls:
                lab = bin_label(c[0])
                d = by.setdefault(lab, [0, 0, 0.0, 0.0, 0.0])
                d[0] += 1; d[1] += c[2]; d[2] += c[3]
                if c[2]: d[4] += c[3]
                else: d[3] += c[3]
            for lab in sorted(by, key=lambda s: int(s.strip('[>=').split(',')[0])):
                n, nf, tt, td, tf = by[lab]
                cw.writerow([p['name'], m, lab, n, nf, f'{tt:.6g}',
                             f'{1e6 * td / (n - nf):.4g}' if n - nf else '',
                             f'{1e6 * tf / nf:.4g}' if nf else ''])
            call_f.flush()
            print(f"{p['name']:28s} deg={top_deg:6d} {m:9s} med={np.median(times):9.4f}s "
                  f"trans={ttrans / t_instr:5.1%} calls={len(calls):6d} fast={sum(c[2] for c in calls):6d} "
                  f"found={len(r)}/{len(ref)} miss={cmp['n_missing']} spur={cmp['n_spurious']} "
                  f"err={cmp['max_err']:.1e} dDense={mdiff:.1e}", flush=True)
    np.savez(os.path.join(HERE, f'e4_refs{args.tag}.npz'), **refs)
    res_f.close(); call_f.close()

if __name__ == '__main__':
    main()
