"""E2 part 2: accumulated error along simulated yroots subdivision/zoom paths (depth up to 60).

Each method applies its own transform step after step (dense arrays shrink via its pruning,
exactly as in yroots, without trimMs). The reference applies the same (alpha, beta) sequence
in double-double carrying a DD coefficient vector, so it is the exact composite transform of
the starting coefficients (up to ~1e-30 relative). The accumulated bound is
sum_i getTransformationError(M_i) using each method's own intermediate M_i.
"""
import time, csv
from common import *
from run_single import roots_poly

DEPTH = 60
RECORD = list(range(1, DEPTH + 1))
m = M_SPLIT

def make_path(kind, u0, rng):
    """Sequence of (alpha, beta, label) steering toward local root coordinate u0."""
    steps = []; u = u0
    for i in range(DEPTH):
        if kind == "split" or (kind == "mixed" and i % 2 == 0):
            if u <= m:
                al, be, lab = (m + 1) / 2, (m - 1) / 2, "split"
            else:
                al, be, lab = -(m - 1) / 2, (m + 1) / 2, "split"
        else:
            z = rng.uniform(0.05, 0.5) if kind == "mixed" else (1e-3 if i % 3 == 0 else 0.1)
            c = u + rng.uniform(-0.5, 0.5) * z
            c = float(np.clip(c, -1 + z, 1 - z))
            al, be, lab = z, c, "zoom"
            if abs(al) + abs(be) > 1.0:
                be = np.nextafter(be, 0.0)
        steps.append((al, be, lab))
        u = float(np.clip((u - be) / al, -1, 1))
    return steps

def main():
    rows = []
    t0 = time.time()
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
                l1a0 = np.abs(a0).sum()
                rh, rl = a0.copy(), np.zeros_like(a0)
                cur = {meth: a0.copy() for meth in METHODS}
                accb = {meth: 0.0 for meth in METHODS}
                for d, (al, be, lab) in enumerate(steps, 1):
                    rh, rl = ref_transform(rh, al, be, rl)
                    for meth in METHODS:
                        accb[meth] += bound(cur[meth])
                        cur[meth] = apply_method(meth, cur[meth], al, be)
                    if d in RECORD:
                        l1r = np.abs(rh + rl).sum()
                        for meth in METHODS:
                            b = pad(cur[meth], n + 1)
                            met, _ = error_metrics(b, rh, rl, l1r, accb[meth])
                            met.update(n=n, profile=pname, path=kind, depth=d, method=meth, step=lab,
                                       l1a0=l1a0, l1_ref=l1r, cur_len=len(cur[meth]),
                                       l1_rel0=met["l1_err"] / l1a0,
                                       trimmed_len=trimmed_length(b, accb[meth]),
                                       ref_trimmed_len=trimmed_length(rh, accb[meth]))
                            rows.append(met)
                print(f"n={n} {pname} {kind} done {time.time()-t0:.0f}s", flush=True)
    keys = ["n", "profile", "path", "depth", "step", "method", "l1a0", "l1_ref", "bound", "maxabs_err", "maxabs_rel",
            "l1_err", "l1_rel", "l1_rel0", "sup_err", "sup_rel", "l1_over_bound", "sup_over_bound", "viol_l1",
            "viol_sup", "frac_err_tophalf", "frac_err_top10", "n_negligible", "noise_max_rel", "noise_l1_rel",
            "cur_len", "trimmed_len", "ref_trimmed_len"]
    with open(os.path.join(HERE, "repeated_transform.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader()
        for r in rows: w.writerow({k: r[k] for k in keys})

if __name__ == "__main__":
    main()
