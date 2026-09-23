"""Experimental O(n log n) affine Chebyshev transform via a type-3 NUFFT.

Given coefficients a_k of p(x) = sum_k a_k T_k(x) on [-1, 1], the coefficients of
q(x) = p(alpha*x + beta) are obtained by evaluating p at the images of the n+1
Chebyshev extreme points x_j = cos(j*pi/n) under x -> alpha*x + beta, and then applying
a DCT-I. Writing alpha*x_j + beta = cos(phi_j) turns the evaluation into the cosine sum
sum_k a_k cos(k*phi_j), which is a nonuniform FFT (FINUFFT type 3). Since q has degree n,
n+1 samples determine it exactly; the only error is the NUFFT tolerance plus rounding.

Requires |alpha| + |beta| <= 1 (the new interval lies inside [-1, 1]); otherwise the
nodes phi_j are not real and callers must fall back to the dense transform.

The algorithm is the one in fast_cheb_affine.py; this module vectorizes it over the
trailing axes so that one NUFFT call transforms a whole tensor along axis 0.
"""
import numpy as np
from scipy.fft import dct

try:
    import finufft
    HAVE_FINUFFT = True
except ImportError:  # pragma: no cover
    finufft = None
    HAVE_FINUFFT = False

#Requested NUFFT tolerance and FINUFFT thread count. Module level so experiments can tune them.
NUFFT_EPS = 1e-15
NUFFT_NTHREADS = 1

def canUseFast(alpha, beta):
    """Whether x -> alpha*x + beta maps [-1, 1] into [-1, 1], so the fast transform applies."""
    return HAVE_FINUFFT and abs(alpha) + abs(beta) <= 1.0

def cheb_affine_fast_axis0(coeffs, alpha, beta, eps=None, nthreads=None):
    """Applies x -> alpha*x + beta along axis 0 of a Chebyshev coefficient tensor.

    Parameters
    ----------
    coeffs : numpy array
        Coefficient tensor; axis 0 is the dimension being transformed. Must have shape[0] >= 2.
    alpha : double
        The scaler of the transformation
    beta : double
        The shifting of the transformation
    eps : float
        Requested NUFFT tolerance (defaults to NUFFT_EPS).
    nthreads : int
        FINUFFT thread count (defaults to NUFFT_NTHREADS).

    Returns
    -------
    transformedCoeffs : numpy array
        Same shape as coeffs.
    """
    eps = NUFFT_EPS if eps is None else eps
    nthreads = NUFFT_NTHREADS if nthreads is None else nthreads
    shape = coeffs.shape
    n = shape[0] - 1
    A = np.ascontiguousarray(coeffs.reshape(n + 1, -1).T)  # (ntrans, n+1)
    ntrans = A.shape[0]

    theta = np.pi * np.arange(n + 1) / n
    phi = np.arccos(np.clip(alpha * np.cos(theta) + beta, -1.0, 1.0))

    # Two-sided complex Fourier coefficients: c_0 = a_0, c_{+-k} = a_k / 2.
    c = np.empty((ntrans, 2 * n + 1), dtype=np.complex128)
    c[:, n] = A[:, 0]
    c[:, n + 1:] = 0.5 * A[:, 1:]
    c[:, :n] = 0.5 * A[:, :0:-1]
    modes = np.arange(-n, n + 1, dtype=np.float64)

    if ntrans == 1:
        c = c[0]
    values = finufft.nufft1d3(modes, c, phi, eps=eps, isign=1, nthreads=nthreads).real
    values = values.reshape(ntrans, n + 1)

    b = dct(values, type=1, axis=1) / n
    b[:, 0] *= 0.5
    b[:, -1] *= 0.5
    return np.ascontiguousarray(b.T).reshape(shape)
