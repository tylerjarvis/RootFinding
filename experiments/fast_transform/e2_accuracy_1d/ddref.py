"""Double-double (~106-bit) reference for the affine Chebyshev transform, plus helpers.

The exact transform of the double input coefficients a (or a DD pair a_hi+a_lo) under
x -> alpha*x + beta (alpha, beta doubles) is b = sum_k a_k C_k, where the column
C_k = coefficients of T_k(alpha*x+beta) satisfies
    C_0 = e_0,  C_1 = beta e_0 + alpha e_1,
    C_k = 2*beta*C_{k-1} + alpha*S(C_{k-1}) - C_{k-2},
    S(c)_0 = c_1, S(c)_1 = 2 c_0 + c_2, S(c)_j = c_{j-1} + c_{j+1}.
We run this recurrence in vectorized double-double arithmetic (Dekker/Knuth error-free
transforms, no truncation of small entries). Validated against mpmath (dps=60) in validate_ref.py.
"""
import numpy as np

_SPLIT = 134217729.0  # 2^27 + 1

def two_sum(a, b):
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e

def quick_two_sum(a, b):
    s = a + b
    e = b - (s - a)
    return s, e

def split(a):
    t = _SPLIT * a
    hi = t - (t - a)
    return hi, a - hi

def two_prod(a, b):
    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e

def dd_add(ah, al, bh, bl):
    s, e = two_sum(ah, bh)
    t, f = two_sum(al, bl)
    e = e + t
    s, e = quick_two_sum(s, e)
    e = e + f
    return quick_two_sum(s, e)

def dd_mul_d(ah, al, b):
    p, e = two_prod(ah, b)
    e = e + al * b
    return quick_two_sum(p, e)

def dd_mul_dd(ah, al, bh, bl):
    p, e = two_prod(ah, bh)
    e = e + (ah * bl + al * bh)
    return quick_two_sum(p, e)

def ref_transform(a_hi, alpha, beta, a_lo=None):
    """DD exact-ish transform. Returns (b_hi, b_lo), both length len(a_hi)."""
    a_hi = np.asarray(a_hi, dtype=np.float64)
    N = len(a_hi)
    if a_lo is None:
        a_lo = np.zeros(N)
    bh = np.zeros(N); bl = np.zeros(N)
    # C0
    c0h = np.zeros(N); c0l = np.zeros(N); c0h[0] = 1.0
    bh[0], bl[0] = a_hi[0], a_lo[0]
    if N == 1:
        return bh, bl
    c1h = np.zeros(N); c1l = np.zeros(N); c1h[0] = beta; c1h[1] = alpha
    th, tl = dd_mul_dd(np.full(2, a_hi[1]), np.full(2, a_lo[1]), c1h[:2], c1l[:2])
    bh[:2], bl[:2] = dd_add(bh[:2], bl[:2], th, tl)
    twobeta = 2.0 * beta
    for k in range(2, N):
        m = k + 1
        # S(C_{k-1}) on rows 0..k (C_{k-1} is supported on 0..k-1)
        uh = np.zeros(m + 1); ul = np.zeros(m + 1)
        uh[:k] = c1h[:k]; ul[:k] = c1l[:k]
        # down-shift part: rows j>=1 get c_{j-1}; up-shift: rows j get c_{j+1}
        dh = np.zeros(m); dl = np.zeros(m)
        dh[1:] = uh[:m - 1]; dl[1:] = ul[:m - 1]
        dh[1] *= 2.0; dl[1] *= 2.0  # row 1 gets 2*c_0 (exact scaling)
        sh, sl = dd_add(dh, dl, uh[1:m + 1], ul[1:m + 1])
        xh, xl = dd_mul_d(sh, sl, alpha)
        yh, yl = dd_mul_d(uh[:m], ul[:m], twobeta)
        zh, zl = dd_add(xh, xl, yh, yl)
        nh, nl = dd_add(zh, zl, -c0h[:m], -c0l[:m])
        # accumulate a_k * C_k
        ph, pl = dd_mul_dd(np.full(m, a_hi[k]), np.full(m, a_lo[k]), nh, nl)
        bh[:m], bl[:m] = dd_add(bh[:m], bl[:m], ph, pl)
        # rotate
        c0h, c0l = c1h, c1l
        c1h = np.zeros(N); c1l = np.zeros(N)
        c1h[:m] = nh; c1l[:m] = nl
    return bh, bl

def cheb_coeffs_from_values(f, n):
    """Chebyshev interpolation coefficients (degree n) of f at Chebyshev extreme points (double)."""
    from scipy.fft import dct
    x = np.cos(np.pi * np.arange(n + 1) / n)
    v = f(x)
    c = dct(v, type=1) / n
    c[0] *= 0.5; c[-1] *= 0.5
    return c

def eval_on_grid(e, factor=8):
    """Values of the Chebyshev series e at the factor*n+1 Chebyshev extreme points (double)."""
    from scipy.fft import dct
    n = len(e) - 1
    N = max(factor * n, 16)
    y = np.zeros(N + 1)
    y[:n + 1] = e
    y[0] *= 2.0
    y[N] *= 2.0
    return dct(y, type=1) / 2.0

def dd_clenshaw(ch, cl, xh, xl):
    """Evaluate sum c_k T_k(x) in DD at DD points x (vectors). Returns (hi, lo)."""
    xh = np.asarray(xh, float); xl = np.asarray(xl, float)
    x2h, x2l = 2 * xh, 2 * xl
    b1h = np.zeros_like(xh); b1l = np.zeros_like(xh)
    b2h = np.zeros_like(xh); b2l = np.zeros_like(xh)
    for k in range(len(ch) - 1, 0, -1):
        th, tl = dd_mul_dd(x2h, x2l, b1h, b1l)
        th, tl = dd_add(th, tl, -b2h, -b2l)
        th, tl = dd_add(th, tl, np.full_like(xh, ch[k]), np.full_like(xh, cl[k]))
        b2h, b2l = b1h, b1l
        b1h, b1l = th, tl
    th, tl = dd_mul_dd(xh, xl, b1h, b1l)
    th, tl = dd_add(th, tl, -b2h, -b2l)
    return dd_add(th, tl, np.full_like(xh, ch[0]), np.full_like(xh, cl[0]))
