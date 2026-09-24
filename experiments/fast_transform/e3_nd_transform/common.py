"""Shared helpers for experiment E3 (N-d dense vs fast Chebyshev affine transforms).

Reference transform: the transformation matrix C (C[:, k] = Chebyshev coefficients of
T_k(alpha*x + beta)) is built with exact integer fixed-point arithmetic (scale 2^-P, P=300),
rounded to double-double (Chi + Clo), and applied with a compensated dot product (Ogita-Rump-Oishi
Dot2 with TwoProd via Veltkamp splitting). The result is accurate to ~eps*|result| + O(n eps^2),
far below the n*eps*sum|M| bound being tested.
"""
import os
import time
from fractions import Fraction

import numpy as np
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
EPS = 2.0 ** -52

PARAMS = [  # (label, alpha, beta)
    ("sub_lo", 0.5, -0.5),
    ("sub_hi", 0.5, 0.5),
    ("zoom0.1", 0.1, 0.3),
    ("zoom1e-3", 1e-3, 0.2),
]


def make_coeffs(shape, kind, rng):
    """Random coefficient tensor. 'decay': N(0,1) * prod_d rho_d^{i_d}, each axis decaying to 1e-16
    at its last index (a converged approximation). 'unit': N(0,1) with no decay."""
    M = rng.standard_normal(shape)
    if kind == "decay":
        for d, n in enumerate(shape):
            if n > 1:
                rho = 1e-16 ** (1.0 / (n - 1))
                s = [1] * len(shape)
                s[d] = n
                M = M * (rho ** np.arange(n)).reshape(s)
    return M


# ---------------------------------------------------------------- exact transformation matrix
_P = 300


def _exact_matrix_int(n, alpha, beta):
    """Fixed-point (scale 2^P) integer n x n matrix C, exact up to floor rounding at 2^-P."""
    fa, fb = Fraction(alpha), Fraction(beta)
    an, ad = fa.numerator, fa.denominator  # ad is a power of 2
    bn, bd = fb.numerator, fb.denominator
    ea, eb = ad.bit_length() - 1, bd.bit_length() - 1
    one = 1 << _P
    C = np.zeros((n, n), dtype=object)
    C[:, :] = 0
    c0 = np.array([0] * n, dtype=object)
    c0[0] = one
    C[:, 0] = c0
    if n == 1:
        return C
    c1 = np.array([0] * n, dtype=object)
    c1[0] = (bn * one) >> eb
    c1[1] = (an * one) >> ea
    C[:, 1] = c1
    prev2, prev = c0, c1
    for k in range(2, n):
        S = np.array([0] * n, dtype=object)
        S[0] = prev[1]
        if n > 2:
            S[1] = 2 * prev[0] + prev[2]
        if n > 3:
            S[2:n - 1] = prev[1:n - 2] + prev[3:n]
        if n > 2:
            S[n - 1] = prev[n - 2]
        cur = np.array([((an * s) >> ea) for s in S], dtype=object) \
            + np.array([((2 * bn * p) >> eb) for p in prev], dtype=object) - prev2
        C[:, k] = cur
        prev2, prev = prev, cur
    return C


def exact_matrix_dd(n, alpha, beta):
    """Double-double transformation matrix (Chi, Clo), cached on disk."""
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, f"C_{n}_{alpha!r}_{beta!r}.npz")
    if os.path.exists(fn):
        z = np.load(fn)
        return z["hi"], z["lo"]
    Ci = _exact_matrix_int(n, alpha, beta)
    flat = Ci.ravel()
    scale = 2.0 ** -_P
    hi = np.empty(flat.size)
    lo = np.empty(flat.size)
    for i, v in enumerate(flat):
        h = float(v)  # correctly rounded
        hi[i] = h * scale
        lo[i] = float(v - int(h)) * scale
    hi = hi.reshape(n, n)
    lo = lo.reshape(n, n)
    np.savez(fn, hi=hi, lo=lo)
    return hi, lo


@njit(cache=True)
def _dot2_matmul(Chi, Clo, A):
    """out = (Chi + Clo) @ A with compensated (Dot2) accumulation. A is (n, m) C-contiguous."""
    n = Chi.shape[0]
    m = A.shape[1]
    out = np.zeros((n, m))
    splitter = 134217729.0  # 2^27 + 1
    for i in range(n):
        for j in range(m):
            s = 0.0
            e = 0.0
            for k in range(n):
                c = Chi[i, k]
                if c == 0.0 and Clo[i, k] == 0.0:
                    continue
                a = A[k, j]
                # TwoProd(c, a)
                p = c * a
                t = splitter * c
                c1 = t - (t - c)
                c2 = c - c1
                t = splitter * a
                a1 = t - (t - a)
                a2 = a - a1
                pe = c2 * a2 - (((p - c1 * a1) - c2 * a1) - c1 * a2)
                # TwoSum(s, p)
                x = s + p
                z = x - s
                se = (s - (x - z)) + (p - z)
                s = x
                e += pe + se + Clo[i, k] * a
            out[i, j] = s + e
    return out


def reference_axis(M, dim, alpha, beta):
    """Near-exact transform of axis dim of M (full size, no truncation)."""
    n = M.shape[dim]
    if n == 1:
        return M.copy()
    Chi, Clo = exact_matrix_dd(n, alpha, beta)
    A = np.ascontiguousarray(np.moveaxis(M, dim, 0).reshape(n, -1))
    out = _dot2_matmul(Chi, Clo, A)
    return np.moveaxis(out.reshape((n,) + tuple(np.delete(M.shape, dim))), 0, dim)


def pad_to(X, shape):
    """Zero-pad a (possibly row-truncated) dense result to the full shape."""
    if X.shape == tuple(shape):
        return X
    out = np.zeros(shape)
    out[tuple(slice(0, s) for s in X.shape)] = X
    return out


def time_call(fn, min_reps=5, max_reps=400, budget=0.4):
    """Median/quantiles of per-call wall time. One warm-up call first."""
    fn()
    ts = []
    t_start = time.perf_counter()
    while len(ts) < max_reps:
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
        if len(ts) >= min_reps and time.perf_counter() - t_start > budget:
            break
    ts = np.array(ts)
    return np.median(ts), np.quantile(ts, 0.1), np.quantile(ts, 0.9), len(ts)
