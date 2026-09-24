"""Validate the DD reference: (1) vs mpmath dps=60 recurrence for moderate n;
(2) for large n, independent pointwise check q(x) = p(alpha*x+beta) in DD Clenshaw."""
import sys, time
import numpy as np, mpmath as mp
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from ddref import ref_transform, dd_clenshaw, two_prod, two_sum

mp.mp.dps = 60

def mp_transform(a, alpha, beta):
    N = len(a); al = mp.mpf(alpha); be = mp.mpf(beta)
    b = [mp.mpf(0)] * N
    c0 = [mp.mpf(0)] * N; c0[0] = mp.mpf(1)
    b[0] += mp.mpf(a[0])
    c1 = [mp.mpf(0)] * N; c1[0] = be; c1[1] = al
    b[0] += mp.mpf(a[1]) * be; b[1] += mp.mpf(a[1]) * al
    for k in range(2, N):
        new = [mp.mpf(0)] * N
        for j in range(k + 1):
            left = c1[j - 1] if j >= 1 else 0
            if j == 1: left = 2 * c1[0]
            right = c1[j + 1] if j + 1 < N else 0
            new[j] = 2 * be * c1[j] + al * (left + right) - c0[j]
        ak = mp.mpf(a[k])
        for j in range(k + 1):
            b[j] += ak * new[j]
        c0, c1 = c1, new
    return b

rng = np.random.default_rng(1)
worst = 0.0
for n in [16, 64, 200]:
    for (alpha, beta) in [(0.5, 0.5), (0.5197277737990524, -0.4802722262009476), (1e-3, 0.37123), (1e-6, -0.8), (0.1, 0.9)]:
        a = rng.standard_normal(n + 1)
        bh, bl = ref_transform(a, alpha, beta)
        bm = mp_transform(a, alpha, beta)
        err = max(abs(mp.mpf(bh[j]) + mp.mpf(bl[j]) - bm[j]) for j in range(n + 1))
        rel = float(err / sum(abs(x) for x in a))
        worst = max(worst, rel)
        print(f"n={n:4d} alpha={alpha:.3g} beta={beta:.3g}: max|DD-mp|/||a||_1 = {rel:.2e}")
print("worst DD vs mpmath relative error:", worst)

# Pointwise check at large n
for n in [2048, 8192]:
    for (alpha, beta) in [(0.5, -0.5), (1e-3, 0.37123), (0.1, 0.9)]:
        a = rng.standard_normal(n + 1)
        t = time.time(); bh, bl = ref_transform(a, alpha, beta); tr = time.time() - t
        x = rng.uniform(-1, 1, 40)
        qh, ql = dd_clenshaw(bh, bl, x, np.zeros_like(x))
        ph_, pl_ = two_prod(alpha, x)
        yh, yl = two_sum(ph_, beta); yl = yl + pl_
        ph, pl = dd_clenshaw(a, np.zeros_like(a), yh, yl)
        d = np.max(np.abs((qh - ph) + (ql - pl))) / np.sum(np.abs(a))
        print(f"n={n} alpha={alpha} beta={beta}: pointwise |q(x)-p(ax+b)|/||a||_1 = {d:.2e} (ref time {tr:.1f}s)")
