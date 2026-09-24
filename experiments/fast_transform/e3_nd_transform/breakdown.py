"""Where the fast path spends its time: FINUFFT type-3 call vs. the rest (copies, DCT-I)."""
import os, sys, time
import numpy as np
import finufft
from scipy.fft import dct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import time_call
from yroots import FastTransform
import yroots.ChebyshevSubdivisionSolver as CSS

for shape in [(8, 8), (64, 64), (256, 256), (1024, 1024), (3,) * 5, (6,) * 5, (10,) * 5, (20,) * 5]:
    M = np.random.default_rng(0).standard_normal(shape)
    n = shape[0] - 1
    A = np.ascontiguousarray(M.reshape(n + 1, -1).T)
    c = np.zeros((A.shape[0], 2 * n + 1), complex); c[:, n:] = A
    modes = np.arange(-n, n + 1, dtype=float)
    phi = np.arccos(0.5 * np.cos(np.pi * np.arange(n + 1) / n) + 0.5)
    cc = c[0] if c.shape[0] == 1 else c
    t_nufft = time_call(lambda: finufft.nufft1d3(modes, cc, phi, eps=1e-15, isign=1, nthreads=1))[0]
    v = np.random.rand(A.shape[0], n + 1)
    t_dct = time_call(lambda: dct(v, type=1, axis=1))[0]
    t_fast = time_call(lambda: FastTransform.cheb_affine_fast_axis0(M, 0.5, 0.5, nthreads=1))[0]
    t_dense = time_call(lambda: CSS.TransformChebInPlace1D(M, 0.5, 0.5))[0]
    print(f"{'x'.join(map(str, shape)):>14}: ntrans={A.shape[0]:>6} fast axis0 {t_fast*1e3:8.3f} ms = nufft {t_nufft*1e3:8.3f}"
          f" + dct {t_dct*1e3:7.3f} + other {max(t_fast-t_nufft-t_dct,0)*1e3:7.3f};  dense axis0 {t_dense*1e3:8.3f} ms;"
          f" nufft us per transform {t_nufft/A.shape[0]*1e6:.2f}")
