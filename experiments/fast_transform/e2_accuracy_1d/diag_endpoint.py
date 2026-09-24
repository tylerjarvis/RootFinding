"""Diagnostic: is the fast method's excess error (splits, edge zooms) due to arccos conditioning
of the NUFFT nodes phi_j = arccos(alpha*cos(theta_j)+beta) near +-1? Recompute the fast transform
with phi_j correctly rounded (mpmath) and compare. Also: trimMs length vs the error level E."""
import mpmath as mp, finufft, csv
from scipy.fft import dct
from common import *
mp.mp.dps = 40

def fast_with_phi(a, phi, eps=1e-15):
    n = len(a) - 1
    c = np.empty(2 * n + 1, complex); c[n] = a[0]; c[n + 1:] = 0.5 * a[1:]; c[:n] = 0.5 * a[:0:-1]
    v = finufft.nufft1d3(np.arange(-n, n + 1, dtype=float), c, phi, eps=eps, isign=1, nthreads=1).real
    b = dct(v, type=1) / n; b[0] *= .5; b[-1] *= .5
    return b

def exact_phi(n, al, be):
    return np.array([float(mp.acos(max(mp.mpf(-1), min(mp.mpf(1), mp.mpf(al) * mp.cos(mp.pi * j / n) + mp.mpf(be))))) for j in range(n + 1)])

m = M_SPLIT
TR = {"split_R(0.5,0.5)": (0.5, 0.5), "offsplit_R": (-(m - 1) / 2, (m + 1) / 2),
      "zoom_1e-1_edge": (0.1, -0.9), "zoom_1e-1": (0.1, 0.37123456789)}
rows = []
for n in [512, 2048, 8192]:
    a = np.random.default_rng(200).standard_normal(n + 1)
    l1a = np.abs(a).sum(); E = bound(a)
    for tn, (al, be) in TR.items():
        rh, rl = ref_transform(a, al, be)
        theta = np.pi * np.arange(n + 1) / n
        phi_np = np.arccos(np.clip(al * np.cos(theta) + be, -1, 1))
        phi_ex = exact_phi(n, al, be)
        dphi = np.abs(phi_np - phi_ex)
        for lab, b in [("fast_numpy_phi", cheb_affine_fast_axis0(a, al, be, eps=1e-15, nthreads=1)),
                       ("fast_exact_phi", fast_with_phi(a, phi_ex))]:
            e = (b - rh) - rl
            r = dict(n=n, transform=tn, variant=lab, max_dphi=dphi.max(), argmax_dphi_j=int(dphi.argmax()),
                     l1_rel=np.abs(e).sum() / l1a, maxabs_rel=np.abs(e).max() / l1a, l1_over_bound=np.abs(e).sum() / E)
            rows.append(r); print(r, flush=True)
with open(os.path.join(HERE, "diag_endpoint.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)

# Trimming vs error level E: zoom 1e-3 (true degree effectively ~5)
rows = []
for n in [256, 1024, 8192]:
    for pn, a in [("random_s0", np.random.default_rng(200).standard_normal(n + 1)),
                  ("sin(n/2 x)", cheb_coeffs_from_values(lambda x: np.sin(0.5 * n * x), n))]:
        l1a = np.abs(a).sum(); al, be = 1e-3, -0.61234567891
        rh, rl = ref_transform(a, al, be)
        outs = {meth: pad(apply_method(meth, a, al, be), n + 1) for meth in ["dense", "fast_1e-15"]}
        outs["ref"] = rh
        for Erel in [None, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6]:
            E = bound(a) if Erel is None else Erel * l1a
            r = dict(n=n, profile=pn, E=("bound" if Erel is None else f"{Erel:.0e}*||a||1"), E_rel=E / l1a)
            for k, b in outs.items(): r[f"trimlen_{k}"] = trimmed_length(b, E)
            rows.append(r); print(r)
with open(os.path.join(HERE, "trim_vs_E.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
