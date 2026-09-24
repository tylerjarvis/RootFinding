"""E5 problem set: 2D chebfun2 suite + constructed high-degree 2D + constructed 5D systems.

Each problem is a dict with keys name, dim, funcs, a, b, ref (reference roots or None),
ref_kind ('polished' | 'exact' | 'newton-polished dense'), jac (optional, for polishing),
root_tol (per-case tolerance used for a pass flag).
"""
import os
import sys
import itertools
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def chebfun2_problems():
    from tests.test_chebfun2_suite import CASES
    out = []
    for c in CASES:
        out.append(dict(name=f"cf{c.name}", group="2D-chebfun2", dim=2, funcs=c.funcs,
                        a=np.asarray(c.a, float), b=np.asarray(c.b, float),
                        ref=np.asarray(c.reference_roots, float), ref_kind="polished",
                        root_tol=c.root_tol))
    return out


def _in_box(P, a, b, pad=0.0):
    return np.all((P >= a - pad) & (P <= b + pad), axis=1)


def diag_problem(w):
    """f = sin(w(x+y)+0.3), g = x - 2y + 0.1. Exact roots: x+y=(k pi-0.3)/w on the line."""
    funcs = [lambda x, y, w=w: np.sin(w*(x+y)+0.3), lambda x, y: x - 2*y + 0.1]
    # x = 2y - 0.1 -> x + y = 3y - 0.1 = s
    ks = np.arange(-int(3*w), int(3*w)+1)
    s = (ks*np.pi - 0.3)/w
    y = (s + 0.1)/3
    x = 2*y - 0.1
    P = np.column_stack([x, y])
    a, b = -np.ones(2), np.ones(2)
    return dict(name=f"diag_w{w}", group="2D-constructed", dim=2, funcs=funcs, a=a, b=b,
                ref=P[_in_box(P, a, b)], ref_kind="exact", root_tol=1e-10)


def diag2_problem(w):
    """f = sin(w(x+y)+0.3), g = sin(w(x-y)/8+0.2): high degree in both variables of both
    functions. Exact roots on a rotated lattice."""
    funcs = [lambda x, y, w=w: np.sin(w*(x+y)+0.3), lambda x, y, w=w: np.sin(w*(x-y)/8+0.2)]
    ks = np.arange(-int(3*w), int(3*w)+1)
    ms = np.arange(-int(w), int(w)+1)
    S = (ks*np.pi - 0.3)/w
    D = 8*(ms*np.pi - 0.2)/w
    SS, DD = np.meshgrid(S, D)
    P = np.column_stack([(SS+DD).ravel()/2, (SS-DD).ravel()/2])
    a, b = -np.ones(2), np.ones(2)
    return dict(name=f"diag2_w{w}", group="2D-constructed", dim=2, funcs=funcs, a=a, b=b,
                ref=P[_in_box(P, a, b)], ref_kind="exact", root_tol=1e-10)


def sincos_problem(w):
    """f = sin(w x) - y, g = cos(w y) - x. Reference: dense-solver roots Newton-polished."""
    funcs = [lambda x, y, w=w: np.sin(w*x) - y, lambda x, y, w=w: np.cos(w*y) - x]

    def F(P):
        x, y = P[:, 0], P[:, 1]
        return np.column_stack([np.sin(w*x) - y, np.cos(w*y) - x])

    def J(P):
        x, y = P[:, 0], P[:, 1]
        n = len(P)
        Jm = np.empty((n, 2, 2))
        Jm[:, 0, 0] = w*np.cos(w*x); Jm[:, 0, 1] = -1
        Jm[:, 1, 0] = -1;            Jm[:, 1, 1] = -w*np.sin(w*y)
        return Jm
    return dict(name=f"sincos_w{w}", group="2D-constructed", dim=2, funcs=funcs,
                a=-np.ones(2), b=np.ones(2), ref=None, ref_kind="newton-polished dense",
                F=F, J=J, root_tol=1e-10)


def tri5_problem():
    """Triangular polynomial system (extends the 4D test in test_Combined_Solver)."""
    funcs = [lambda a, b, c, d, e: a - 0.5,
             lambda a, b, c, d, e: b + a**2 - 0.75,
             lambda a, b, c, d, e: c - a*b,
             lambda a, b, c, d, e: d**2 - c - 0.5,
             lambda a, b, c, d, e: e**2 + a*d - 0.3]
    x4 = np.sqrt(0.75)
    ref = []
    for d in (-x4, x4):
        r = 0.3 - 0.5*d
        if r >= 0:
            for e in (-np.sqrt(r), np.sqrt(r)):
                ref.append([0.5, 0.5, 0.25, d, e])
    return dict(name="tri5", group="5D", dim=5, funcs=funcs, a=-np.ones(5), b=np.ones(5),
                ref=np.array(ref), ref_kind="exact", root_tol=1e-10)


def diag5_problem(w):
    """f1 = sin(w*sum x + 0.3) (full 5D tensor), f2..f5 linear chain. Roots on a line."""
    funcs = [lambda a, b, c, d, e, w=w: np.sin(w*(a+b+c+d+e)+0.3),
             lambda a, b, c, d, e: b - 0.5*a - 0.1,
             lambda a, b, c, d, e: c - 0.5*b + 0.2,
             lambda a, b, c, d, e: d - 0.5*c - 0.05,
             lambda a, b, c, d, e: e - 0.5*d + 0.15]

    def chain(a):
        b = 0.5*a + 0.1; c = 0.5*b - 0.2; d = 0.5*c + 0.05; e = 0.5*d - 0.15
        return np.array([a, b, c, d, e])
    p0 = chain(0.0); p1 = chain(1.0) - p0   # x = p0 + t p1
    s0, s1 = p0.sum(), p1.sum()
    ks = np.arange(-50, 51)
    t = ((ks*np.pi - 0.3)/w - s0)/s1
    P = p0[None, :] + t[:, None]*p1[None, :]
    a, b = -np.ones(5), np.ones(5)
    return dict(name=f"diag5_w{w}", group="5D", dim=5, funcs=funcs, a=a, b=b,
                ref=P[_in_box(P, a, b)], ref_kind="exact", root_tol=1e-10)


def trig5_problem(k, seed=0):
    """f_i = sin(k_i (A x)_i + c_i), A = I + 0.15 R: coupled trig system, exact lattice roots."""
    rng = np.random.default_rng(seed)
    A = np.eye(5) + 0.15*rng.uniform(-1, 1, (5, 5))
    kk = k*np.array([1.0, 1.1, 1.2, 0.9, 1.05])
    cc = rng.uniform(-0.5, 0.5, 5)
    funcs = []
    for i in range(5):
        funcs.append(lambda a, b, c, d, e, i=i: np.sin(kk[i]*(A[i, 0]*a + A[i, 1]*b + A[i, 2]*c
                                                            + A[i, 3]*d + A[i, 4]*e) + cc[i]))
    # Enumerate lattice y_i = (m pi - c_i)/k_i, x = A^{-1} y
    rowmax = np.abs(A).sum(axis=1)
    ranges = []
    for i in range(5):
        lo = int(np.floor((-kk[i]*rowmax[i] + cc[i])/np.pi)) - 1
        hi = int(np.ceil((kk[i]*rowmax[i] + cc[i])/np.pi)) + 1
        ranges.append(np.arange(lo, hi+1))
    Ainv = np.linalg.inv(A)
    P = []
    for m in itertools.product(*ranges):
        yv = (np.array(m)*np.pi - cc)/kk
        P.append(Ainv @ yv)
    P = np.array(P)
    a, b = -np.ones(5), np.ones(5)
    return dict(name=f"trig5_k{k}", group="5D", dim=5, funcs=funcs, a=a, b=b,
                ref=P[_in_box(P, a, b)], ref_kind="exact", root_tol=1e-10)


def constructed_2d():
    return ([diag_problem(w) for w in (50, 100, 200, 400, 800)]
            + [diag2_problem(w) for w in (100, 200)]
            + [sincos_problem(w) for w in (20, 40, 100)])


def problems_5d():
    return [tri5_problem(), trig5_problem(2), trig5_problem(3), diag5_problem(3), diag5_problem(4)]


def all_problems(groups):
    out = []
    if "cf" in groups:
        out += chebfun2_problems()
    if "c2" in groups:
        out += constructed_2d()
    if "5d" in groups:
        out += problems_5d()
    if "big" in groups:
        out += [diag_problem(800), diag_problem(1600)]
    return out
