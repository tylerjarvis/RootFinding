"""E3 accuracy: per-axis and full-tensor transform errors vs a double-double/compensated reference,
compared with the solver bound getTransformationError(M, dim) = n*2^-52*sum|M|.

Usage: uv run --no-sync python experiments/fast_transform/e3_nd_transform/run_accuracy.py [2|5 ...]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PARAMS, make_coeffs, reference_axis, pad_to, HERE
from run_timing import SHAPES, KINDS
import yroots.ChebyshevSubdivisionSolver as CSS
from yroots import FastTransform

FastTransform.NUFFT_NTHREADS = 1


def run(dims):
    out = os.path.join(HERE, "accuracy.csv")
    new = not os.path.exists(out)
    f = open(out, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["ndim", "shape", "param", "alpha", "beta", "coeffs", "op", "n_axis", "method",
                    "sumabs_M", "max_abs_err", "err_over_sumabs", "bound", "err_over_bound",
                    "violation"])
    rng = np.random.default_rng(777)
    for nd in dims:
        for shape in SHAPES[nd]:
            for kind in KINDS:
                M = make_coeffs(shape, kind, rng)
                S = np.abs(M).sum()
                for plabel, a, b in PARAMS:
                    refs = {}
                    for d in range(nd):
                        refs[d] = reference_axis(M, d, a, b)
                    # full reference: axes in order 0..nd-1
                    R = refs[0]
                    for d in range(1, nd):
                        R = reference_axis(R, d, a, b)
                    for mlabel, meth, exact in (("dense", "dense", False), ("fast", "fast", False),
                                                ("exact", "dense", True)):
                        CSS.TRANSFORM_METHOD = meth
                        for d in range(nd):
                            X = pad_to(CSS.TransformChebInPlaceND(M, d, a, b, exact), shape)
                            err = np.abs(X - refs[d]).max()
                            bound = CSS.getTransformationError(M, d)
                            w.writerow([nd, "x".join(map(str, shape)), plabel, a, b, kind, d, shape[d],
                                        mlabel, f"{S:.6e}", f"{err:.4e}", f"{err / S:.4e}",
                                        f"{bound:.4e}", f"{err / bound:.4e}", int(err > bound)])
                        X, bound = CSS.transformCheb(M, [a] * nd, [b] * nd, 0.0, exact)
                        X = pad_to(X, shape)
                        err = np.abs(X - R).max()
                        w.writerow([nd, "x".join(map(str, shape)), plabel, a, b, kind, "full",
                                    max(shape), mlabel, f"{S:.6e}", f"{err:.4e}", f"{err / S:.4e}",
                                    f"{bound:.4e}", f"{err / bound:.4e}", int(err > bound)])
                    f.flush()
                print(nd, shape, kind, flush=True)
    CSS.TRANSFORM_METHOD = "dense"
    f.close()


if __name__ == "__main__":
    dims = [int(x) for x in sys.argv[1:]] or [2, 5]
    run(dims)
