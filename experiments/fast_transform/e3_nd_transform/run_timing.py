"""E3 timing: TransformChebInPlaceND per axis and full transformCheb, dense vs fast, in 2D and 5D.

Usage: uv run --no-sync python experiments/fast_transform/e3_nd_transform/run_timing.py [2|5 ...]
"""
import os
# OMP_NUM_THREADS left unset so FINUFFT can use 4 threads in the fast_mt4 variant (nthreads=1 otherwise)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PARAMS, make_coeffs, time_call, HERE
import yroots.ChebyshevSubdivisionSolver as CSS
from yroots import FastTransform

SHAPES = {
    2: [(n, n) for n in (4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)]
       + [(1024, 8), (8, 1024), (1024, 32), (32, 1024), (256, 16), (16, 256), (128, 8), (8, 128)],
    5: [(n,) * 5 for n in (3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20)]
       + [(20, 3, 3, 3, 3), (3, 3, 3, 3, 20), (40, 4, 4, 4, 4), (12, 8, 5, 3, 3), (20, 20, 5, 5, 5),
          (64, 3, 3, 3, 3)],
}
METHODS = [("dense", "dense", 1), ("fast", "fast", 1), ("fast_mt4", "fast", 4)]
KINDS = ["decay", "unit"]


def run(dims):
    out = os.path.join(HERE, "timing.csv")
    new = not os.path.exists(out)
    f = open(out, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["ndim", "shape", "param", "alpha", "beta", "coeffs", "op", "n_axis", "method",
                    "median_s", "p10_s", "p90_s", "reps", "out_shape"])
    # Warm-up / JIT
    for M in (np.random.rand(5, 5), np.random.rand(3, 3, 3, 3, 3)):
        for meth in ("dense", "fast"):
            CSS.TRANSFORM_METHOD = meth
            CSS.transformCheb(M, [0.5] * M.ndim, [0.5] * M.ndim, 0.0, False)
    rng = np.random.default_rng(12345)
    done = set()
    if not new:  # resume: skip (ndim, shape, coeffs) blocks already in the CSV
        with open(out) as fr:
            for r in csv.DictReader(fr):
                done.add((int(r["ndim"]), r["shape"], r["coeffs"]))
    for nd in dims:
        for shape in SHAPES[nd]:
            for kind in KINDS:
                M = make_coeffs(shape, kind, rng)  # always drawn so the RNG stream is reproducible
                if (nd, "x".join(map(str, shape)), kind) in done:
                    continue
                for plabel, a, b in PARAMS:
                    ops = [(d, shape[d]) for d in range(nd)] + [("full", max(shape))]
                    for op, naxis in ops:
                        for mlabel, meth, nth in METHODS:
                            # FINUFFT with 4 threads was measured 5-8x slower than 1 thread for 5D
                            # tensors >= 12^5 (many tiny transforms); skip it for large 5D to save time.
                            if nth > 1 and nd == 5 and M.size > 2e5:
                                continue
                            CSS.TRANSFORM_METHOD = meth
                            FastTransform.NUFFT_NTHREADS = nth
                            if op == "full":
                                fn = lambda: CSS.transformCheb(M, [a] * nd, [b] * nd, 0.0, False)
                                res = fn()[0]
                            else:
                                fn = lambda: CSS.TransformChebInPlaceND(M, op, a, b, False)
                                res = fn()
                            big = M.size * max(shape) > 5e7
                            med, p10, p90, reps = time_call(fn, min_reps=3 if big else 7,
                                                            budget=0.3)
                            w.writerow([nd, "x".join(map(str, shape)), plabel, a, b, kind, op, naxis,
                                        mlabel, f"{med:.4e}", f"{p10:.4e}", f"{p90:.4e}", reps,
                                        "x".join(map(str, res.shape))])
                        f.flush()
                print(nd, shape, kind, flush=True)
    CSS.TRANSFORM_METHOD = "dense"
    FastTransform.NUFFT_NTHREADS = 1
    f.close()


if __name__ == "__main__":
    dims = [int(x) for x in sys.argv[1:]] or [2, 5]
    run(dims)
