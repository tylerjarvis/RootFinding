"""Does 'fast' on chebfun2 case 2.4 terminate if the Python recursion limit is raised?"""
import sys, time, signal, warnings
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from problems import chebfun2_problems
import yroots.ChebyshevSubdivisionSolver as C
import yroots.FastTransform as F
from yroots.Combined_Solver import solve
warnings.simplefilter("ignore")
F.NUFFT_NTHREADS = 1
p = [q for q in chebfun2_problems() if q["name"] == "cf2.4"][0]
depth = [0, 0]
orig = C.solvePolyRecursive
def wrapped(*a, **k):
    depth[0] += 1; depth[1] = max(depth[1], depth[0])
    try:
        return orig(*a, **k)
    finally:
        depth[0] -= 1
C.solvePolyRecursive = wrapped
for lim in [1000, 5000, 20000]:
    sys.setrecursionlimit(lim)
    C.TRANSFORM_METHOD = "fast"
    depth[:] = [0, 0]
    signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(300)
    t = time.perf_counter()
    try:
        r = solve(p["funcs"], p["a"], p["b"], max_cpu=1)
        st = f"ok, {len(r)} roots"
    except Exception as e:
        st = type(e).__name__
    signal.alarm(0)
    print(f"recursionlimit={lim}: {st} after {time.perf_counter()-t:.1f}s, max solvePolyRecursive depth={depth[1]}", flush=True)
    if st.startswith("ok"):
        break
C.TRANSFORM_METHOD = "dense"; depth[:] = [0, 0]
r = solve(p["funcs"], p["a"], p["b"], max_cpu=1)
print(f"dense: {len(r)} roots, max depth={depth[1]}")
