"""Diagnostic: why 'fast' changes the subdivision tree. Compare the high-order tail of the
coefficients after mapping a degree-n approximation onto a small subinterval, dense vs NUFFT.
Also time single transforms (dense numba vs NUFFT) by degree, 1 thread.
Run: uv run --no-sync python experiments/fast_transform/e4_solve_1d/e4_tail_diag.py
"""
import time, numpy as np
import yroots.ChebyshevSubdivisionSolver as C
import yroots.ChebyshevApproximator as CA
from yroots import FastTransform as F
F.NUFFT_NTHREADS = 1
rows = []
print("tail diagnostics: f=sin(1000x) approximated on [-1,1], mapped to [c-h, c+h], c=0.123*(1-h)")
M, err = CA.chebApproximate(lambda x: np.sin(1000 * x), np.array([-1.]), np.array([1.]))
n = len(M) - 1
print(f"degree {n}, approx err {err:.2e}")
print("dense TransformChebInPlace1D stops growing its output once the newest diagonal entry of the transform matrix is <= 1e-16 (≈ rows k with |alpha|^k > 1e-16); the NUFFT path returns all n+1 rows.")
print(f"{'h':>8} {'dense_len':>9} {'fast_len':>9} {'fast: last k>1e-3*err':>22} {'fast |c_k| max beyond dense_len':>32} {'max|diff| common':>17}")
for h in [0.999, 0.9, 0.5, 0.1, 0.02, 0.005, 0.001]:
    alpha, beta = h, 0.123 * (1 - h)
    Cd = C.TransformChebInPlace1D(M.copy(), alpha, beta)
    Cf = F.cheb_affine_fast_axis0(M.copy(), alpha, beta)
    thr = 1e-3 * err
    kf = np.max(np.nonzero(np.abs(Cf) > thr)[0])
    L = len(Cd)
    tail = np.max(np.abs(Cf[L:])) if L < len(Cf) else 0.0
    print(f"{h:8.3f} {L:9d} {len(Cf):9d} {kf:22d} {tail:32.2e} {np.max(np.abs(Cd - Cf[:L])):17.2e}")

print("\nsingle-transform timing (median of reps, us). dense output length shown; alpha=0.5 (subdivision-like) and alpha=0.95 (zoom-like):")
print(f"{'deg':>6} {'alpha':>6} {'dense_len':>9} {'dense_us':>10} {'fast_us':>10} {'dense/fast':>10}")
rng = np.random.default_rng(0)
for d in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
    c = rng.standard_normal(d + 1)
    for al, be in [(0.5, 0.5), (0.95, 0.03)]:
        C.TransformChebInPlace1D(c, al, be); F.cheb_affine_fast_axis0(c, al, be)
        reps = max(3, min(200, int(2e6 / (d * d + 1000))))
        td = []; tf = []
        for _ in range(reps):
            t = time.perf_counter(); out = C.TransformChebInPlace1D(c, al, be); td.append(time.perf_counter() - t)
            t = time.perf_counter(); F.cheb_affine_fast_axis0(c, al, be); tf.append(time.perf_counter() - t)
        a, b = 1e6 * np.median(td), 1e6 * np.median(tf)
        print(f"{d:6d} {al:6.2f} {len(out):9d} {a:10.1f} {b:10.1f} {a/b:10.2f}", flush=True)
