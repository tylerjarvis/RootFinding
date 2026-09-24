import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, HERE)
import yroots.ChebyshevSubdivisionSolver as css
from yroots.FastTransform import cheb_affine_fast_axis0
from ddref import ref_transform, cheb_coeffs_from_values, eval_on_grid

MACH = 2.0 ** -52
M_SPLIT = 0.0394555475981047  # yroots' default nextTransformPoints
EPS_LIST = [1e-15, 1e-14, 1e-12]
METHODS = ["dense", "dense_ef"] + [f"fast_{e:.0e}" for e in EPS_LIST]

def apply_method(method, a, alpha, beta):
    a = np.ascontiguousarray(a, dtype=np.float64)
    if method == "dense":
        return css.TransformChebInPlace1D(a, alpha, beta)
    if method == "dense_ef":
        return css.TransformChebInPlace1DErrorFree(a, alpha, beta)
    eps = float(method.split("_")[1])
    return cheb_affine_fast_axis0(a, alpha, beta, eps=eps, nthreads=1)

def pad(b, N):
    out = np.zeros(N); out[:len(b)] = b; return out

def bound(a):
    """yroots getTransformationError for a 1D array."""
    return css.getTransformationError(np.asarray(a, dtype=np.float64), 0)

def trimmed_length(b, E, rel=1e-3):
    """Length left by yroots trimMs (1D) applied to b with current error bound E."""
    allowed = rel * E
    L = len(b)
    while L > 3 and abs(b[L - 1]) < allowed:
        allowed -= abs(b[L - 1]); L -= 1
    return L

def error_metrics(b, rh, rl, l1a, E):
    """b computed (padded), (rh, rl) DD reference. Returns dict of metrics."""
    N = len(rh)
    e = (b - rh) - rl
    ae = np.abs(e)
    l1e = ae.sum()
    sup = np.max(np.abs(eval_on_grid(e, 8)))
    n = N - 1
    ref_abs = np.abs(rh + rl)
    neg = ref_abs < 1e-20 * l1a
    m = dict(
        maxabs_err=ae.max(), maxabs_rel=ae.max() / l1a,
        l1_err=l1e, l1_rel=l1e / l1a, sup_err=sup, sup_rel=sup / l1a,
        bound=E, l1_over_bound=l1e / E, sup_over_bound=sup / E,
        viol_l1=bool(l1e > E), viol_sup=bool(sup > E),
        frac_err_tophalf=ae[n // 2 + 1:].sum() / l1e if l1e > 0 else np.nan,
        frac_err_top10=ae[int(0.9 * n) + 1:].sum() / l1e if l1e > 0 else np.nan,
        n_negligible=int(neg.sum()),
        noise_max_rel=(np.abs(b[neg]).max() / l1a) if neg.any() else np.nan,
        noise_l1_rel=(np.abs(b[neg]).sum() / l1a) if neg.any() else np.nan,
    )
    return m, e
