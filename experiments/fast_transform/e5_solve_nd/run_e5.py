"""E5: end-to-end yroots.solve timing and accuracy, dense vs fast vs auto transforms.

Usage (from repo root):
    uv run --no-sync python experiments/fast_transform/e5_solve_nd/run_e5.py GROUP [GROUP ...] [--only name,...]
GROUP in {cf, c2, 5d}. Writes results_<groups>.csv and calls_<groups>.csv next to this file.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import sys
import csv
import json
import signal
import time
import warnings
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from problems import all_problems  # noqa: E402  (also puts repo on sys.path)

import yroots.ChebyshevSubdivisionSolver as C  # noqa: E402
import yroots.FastTransform as F  # noqa: E402
from yroots.Combined_Solver import solve  # noqa: E402
from yroots import ChebyshevApproximator as CA  # noqa: E402

warnings.simplefilter("ignore")
F.NUFFT_NTHREADS = 1

METHODS = {
    "dense":           ("dense", (1,), 256),
    "fast":            ("fast", (1,), 256),
    "auto_nd12_d32":   ("auto", (1, 2), 32),
    "auto_nd12_d64":   ("auto", (1, 2), 64),
    "auto_nd12_d256":  ("auto", (1, 2), 256),
    "auto_ndall_d16":  ("auto", (1, 2, 3, 4, 5), 16),
    # Experimental variants (not in yroots): NUFFT output truncated to the degree the dense
    # recurrence would keep (maxRow, which depends only on n, alpha, beta). Emulates a fast
    # transform that also trims negligible high-degree output like the dense one does.
    "fasttrunc":           ("fast", (1,), 256, True),
    "autotrunc_nd12_d64":  ("auto", (1, 2), 64, True),
    "autotrunc_ndall_d16": ("auto", (1, 2, 3, 4, 5), 16, True),
}
DEFAULT_METHODS = ["dense", "fast", "auto_nd12_d32", "auto_nd12_d64", "auto_nd12_d256",
                   "auto_ndall_d16"]
TRUNC = [False]
_orig_fast_axis0 = F.cheb_affine_fast_axis0


def _fast_axis0_maybe_trunc(coeffs, alpha, beta, eps=None, nthreads=None):
    out = _orig_fast_axis0(coeffs, alpha, beta, eps, nthreads)
    if TRUNC[0]:
        m = C.TransformChebInPlace1D(np.ones((coeffs.shape[0], 1)), alpha, beta).shape[0]
        out = out[:m]
    return out


F.cheb_affine_fast_axis0 = _fast_axis0_maybe_trunc


def set_method(m):
    meth, nd, md, *rest = METHODS[m]
    TRUNC[0] = bool(rest and rest[0])
    C.TRANSFORM_METHOD = meth
    C.FAST_TRANSFORM_NDIMS = nd
    C.FAST_TRANSFORM_MIN_DEGREE = md


# ---------------------------------------------------------------- instrumentation
_orig_transform = C.TransformChebInPlaceND
_orig_subdiv = C.solveChebyshevSubdivision
REC = None  # dict when instrumenting


def _wrapped_transform(coeffs, dim, alpha, beta, exact):
    if REC is None:
        return _orig_transform(coeffs, dim, alpha, beta, exact)
    trivial = (alpha == 1.0 and beta == 0.0) or coeffs.shape[dim] == 1
    fast = (not trivial) and C.useFastTransform(coeffs, dim, alpha, beta, exact)
    t0 = time.perf_counter()
    out = _orig_transform(coeffs, dim, alpha, beta, exact)
    dt = time.perf_counter() - t0
    key = (coeffs.ndim, coeffs.shape[dim] - 1, bool(fast), bool(trivial))
    agg = REC["calls"][key]
    agg[0] += 1
    agg[1] += dt
    agg[2] += coeffs.size // coeffs.shape[dim]
    REC["t_transform"] += dt
    return out


def _wrapped_subdiv(*args, **kw):
    if REC is None:
        return _orig_subdiv(*args, **kw)
    t0 = time.perf_counter()
    out = _orig_subdiv(*args, **kw)
    REC["t_subdiv"] += time.perf_counter() - t0
    return out


C.TransformChebInPlaceND = _wrapped_transform
C.solveChebyshevSubdivision = _wrapped_subdiv


class SolveTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise SolveTimeout()


signal.signal(signal.SIGALRM, _alarm)
TIME_LIMIT = [None]


def run_solve(p):
    if TIME_LIMIT[0]:
        signal.setitimer(signal.ITIMER_REAL, TIME_LIMIT[0])
    try:
        r = solve(p["funcs"], p["a"], p["b"], max_cpu=1)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return np.asarray(r, dtype=float).reshape(-1, p["dim"])


def instrumented(p):
    global REC
    REC = dict(calls=defaultdict(lambda: [0, 0.0, 0]), t_transform=0.0, t_subdiv=0.0)
    t0 = time.perf_counter()
    status = "ok"
    try:
        roots = run_solve(p)
    except SolveTimeout:
        status, roots = f"timeout>{TIME_LIMIT[0]:.0f}s", None
    except Exception as e:  # noqa: BLE001  (RecursionError etc. = solver failure)
        status, roots = type(e).__name__, None
    total = time.perf_counter() - t0
    rec = REC
    REC = None
    rec["status"] = status
    return roots, total, rec


def timed(p):
    t0 = time.perf_counter()
    run_solve(p)
    return time.perf_counter() - t0


# ---------------------------------------------------------------- accuracy
def match(found, ref, tol):
    """One-to-one matching; pairs farther than tol count as missing + spurious."""
    if len(found) == 0 or len(ref) == 0:
        return 0, len(ref), len(found), np.nan
    D = np.linalg.norm(found[:, None, :] - ref[None, :, :], axis=2)
    r, c = linear_sum_assignment(D)
    d = D[r, c]
    ok = d <= tol
    nmatch = int(ok.sum())
    maxerr = float(d[ok].max()) if nmatch else np.nan
    return nmatch, len(ref) - nmatch, len(found) - nmatch, maxerr


def max_residual(p, roots):
    if len(roots) == 0:
        return 0.0
    return max(float(np.max(np.abs(f(*roots.T)))) for f in p["funcs"])


def newton_polish(p, roots, iters=4):
    P = roots.copy()
    for _ in range(iters):
        P = P - np.linalg.solve(p["J"](P), p["F"](P)[..., None])[..., 0]
    return P


def nreps_for(t):
    if t < 0.5:
        return 7
    if t < 5:
        return 5
    if t < 30:
        return 3
    return 1


def main():
    args = sys.argv[1:]
    only = None
    if "--only" in args:
        i = args.index("--only")
        only = set(args[i + 1].split(","))
        args = args[:i] + args[i + 2:]
    methods = list(DEFAULT_METHODS)
    tagx = ""
    if "--tag" in args:
        i = args.index("--tag")
        tagx = "_" + args[i + 1]
        args = args[:i] + args[i + 2:]
    if "--methods" in args:
        i = args.index("--methods")
        methods = args[i + 1].split(",")
        args = args[:i] + args[i + 2:]
    groups = args
    tag = "_".join(groups) + tagx + ("" if only is None else "_" + "_".join(sorted(only)))
    res_path = os.path.join(HERE, f"results_{tag}.csv")
    calls_path = os.path.join(HERE, f"calls_{tag}.csv")
    probs = [p for p in all_problems(groups) if only is None or p["name"] in only]

    res_fields = ["problem", "group", "dim", "shapes", "max_degree", "ref_kind", "n_ref", "method",
                  "t_median", "t_min", "t_max", "n_reps", "t_instrumented", "t_subdiv_instr",
                  "t_transform_instr", "transform_share", "n_calls", "n_calls_nontrivial",
                  "n_calls_fast", "t_calls_fast", "n_found", "n_matched", "n_missing",
                  "n_spurious", "max_err_vs_ref", "max_err_vs_dense", "max_residual", "root_tol",
                  "pass", "status"]
    call_fields = ["problem", "method", "ndim", "degree", "fast", "trivial", "n_calls", "t_total",
                   "mean_ntrans"]
    with open(res_path, "w", newline="") as fr, open(calls_path, "w", newline="") as fc:
        wr = csv.DictWriter(fr, res_fields); wr.writeheader()
        wc = csv.DictWriter(fc, call_fields); wc.writeheader()

        # global NUFFT warm-up
        F.cheb_affine_fast_axis0(np.random.rand(40, 3), 0.5, 0.1)

        for p in probs:
            shapes = [CA.chebApproximate(f, p["a"], p["b"])[0].shape for f in p["funcs"]]
            maxdeg = max(max(s) for s in shapes) - 1
            print(f"== {p['name']} dim={p['dim']} shapes={shapes}", flush=True)
            set_method("dense")
            TIME_LIMIT[0] = None
            t0 = time.perf_counter()
            run_solve(p)  # warm-up (numba JIT, caches)
            twarm = time.perf_counter() - t0
            TIME_LIMIT[0] = min(max(120.0, 25 * twarm), 1500.0)

            inst = {}
            for m in methods:
                set_method(m)
                roots, total, rec = instrumented(p)
                inst[m] = (roots, total, rec)
                print(f"   instr {m:16s} {total:8.3f}s status={rec['status']} roots="
                      f"{None if roots is None else len(roots)} share="
                      f"{rec['t_transform']/total:.2f}", flush=True)

            if p["ref"] is None:  # reference from Newton-polished dense roots
                ref = newton_polish(p, inst["dense"][0])
                p["ref"] = ref
            ref = p["ref"]
            width = float(np.max(p["b"] - p["a"]))
            mtol = 1e-6 * width

            times = {m: [] for m in methods}
            reps = {m: (nreps_for(inst[m][1]) if inst[m][2]["status"] == "ok" else 0)
                    for m in methods}
            for r in range(max(reps.values())):
                for m in methods:
                    if len(times[m]) < reps[m]:
                        set_method(m)
                        times[m].append(timed(p))
            set_method("dense")

            dense_roots = inst["dense"][0] if "dense" in inst else ref
            for m in methods:
                roots, total, rec = inst[m]
                if roots is None:
                    wr.writerow(dict(problem=p["name"], group=p["group"], dim=p["dim"],
                                     shapes=json.dumps(shapes), max_degree=maxdeg,
                                     ref_kind=p["ref_kind"], n_ref=len(ref), method=m,
                                     t_instrumented=total, status=rec["status"],
                                     root_tol=p["root_tol"], **{"pass": False}))
                    print(f"   {m:16s} FAILED: {rec['status']} after {total:.1f}s", flush=True)
                    continue
                nm, nmiss, nsp, maxerr = match(roots, ref, mtol)
                _, _, _, errd = match(roots, dense_roots, mtol)
                calls = rec["calls"]
                ncalls = sum(v[0] for v in calls.values())
                nnt = sum(v[0] for k, v in calls.items() if not k[3])
                nfast = sum(v[0] for k, v in calls.items() if k[2])
                tfast = sum(v[1] for k, v in calls.items() if k[2])
                resid = max_residual(p, roots)
                passed = (nmiss == 0 and nsp == 0 and len(roots) == len(ref)
                          and (np.isnan(maxerr) or maxerr <= p["root_tol"]))
                ts = np.array(times[m])
                wr.writerow(dict(problem=p["name"], group=p["group"], dim=p["dim"],
                                 shapes=json.dumps(shapes), max_degree=maxdeg,
                                 ref_kind=p["ref_kind"], n_ref=len(ref), method=m,
                                 t_median=np.median(ts), t_min=ts.min(), t_max=ts.max(),
                                 n_reps=len(ts), t_instrumented=total,
                                 t_subdiv_instr=rec["t_subdiv"],
                                 t_transform_instr=rec["t_transform"],
                                 transform_share=rec["t_transform"] / total,
                                 n_calls=ncalls, n_calls_nontrivial=nnt, n_calls_fast=nfast,
                                 t_calls_fast=tfast, n_found=len(roots), n_matched=nm,
                                 n_missing=nmiss, n_spurious=nsp, max_err_vs_ref=maxerr,
                                 max_err_vs_dense=errd, max_residual=resid,
                                 root_tol=p["root_tol"], status="ok", **{"pass": passed}))
                for (nd, deg, fast, triv), (n, t, nt) in sorted(calls.items()):
                    wc.writerow(dict(problem=p["name"], method=m, ndim=nd, degree=deg,
                                     fast=int(fast), trivial=int(triv), n_calls=n, t_total=t,
                                     mean_ntrans=nt / n))
                print(f"   {m:16s} med={np.median(ts):8.3f}s (n={len(ts)}) found={len(roots)}/"
                      f"{len(ref)} miss={nmiss} spur={nsp} err={maxerr:.1e} "
                      f"errVsDense={errd:.1e} fastcalls={nfast}/{nnt}", flush=True)
            fr.flush(); fc.flush()


if __name__ == "__main__":
    main()
