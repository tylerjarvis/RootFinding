"""Per-step (local) error along the same paths: at every step, compare each method's output with the
DD-exact transform of *that method's own input*, against getTransformationError of that input."""
import time, csv
from common import *
from run_single import roots_poly
from run_paths import make_path

LOCAL_METHODS = ["dense", "fast_1e-15", "fast_1e-12"]
rows = []; t0 = time.time()
for n in [64, 512, 4096]:
    polys = []
    for s in range(2):
        rng = np.random.default_rng(100 + s)
        r = np.sort(rng.uniform(-1, 1, max(n // 2, 1)))
        polys.append((f"roots_n/2_s{s}", cheb_coeffs_from_values(roots_poly(n, 100 + s), n), r[len(r) // 3]))
    om = 0.5 * n
    polys.append(("sin(n/2 x)", cheb_coeffs_from_values(lambda x: np.sin(om * x), n), 7 * np.pi / om))
    for pname, a0, root in polys:
        for kind in ["split", "mixed", "zoomdeep"]:
            steps = make_path(kind, root, np.random.default_rng(7))
            for meth in LOCAL_METHODS:
                cur = a0.copy()
                for d, (al, be, lab) in enumerate(steps, 1):
                    if meth.startswith("fast") and d > 30 and n == 4096:
                        break  # cost cap; first 30 steps
                    E = bound(cur); l1 = np.abs(cur).sum()
                    rh, rl = ref_transform(cur, al, be)
                    new = pad(apply_method(meth, cur, al, be), len(cur))
                    e = (new - rh) - rl
                    rows.append(dict(n=n, profile=pname, path=kind, depth=d, step=lab, method=meth, len_in=len(cur),
                                     l1_in=l1, local_l1_err=np.abs(e).sum(), local_l1_rel=np.abs(e).sum() / l1,
                                     local_over_bound=np.abs(e).sum() / E))
                    cur = apply_method(meth, cur, al, be)
            print(n, pname, kind, f"{time.time()-t0:.0f}s", flush=True)
with open(os.path.join(HERE, "repeated_local_steps.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
