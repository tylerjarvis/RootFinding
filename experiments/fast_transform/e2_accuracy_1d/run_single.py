"""E2 part 1 + 3: single-transform accuracy vs a double-double reference."""
import time, csv
from common import *

NS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
m = M_SPLIT
TRANSFORMS = {
    "split_L(0.5,-0.5)": (0.5, -0.5),
    "split_R(0.5,0.5)": (0.5, 0.5),
    "offsplit_L": ((m + 1) / 2, (m - 1) / 2),
    "offsplit_R": (-(m - 1) / 2, (m + 1) / 2),
    "zoom_1e-1": (0.1, 0.37123456789),
    "zoom_1e-1_edge": (0.1, -0.9),
    "zoom_1e-3": (1e-3, -0.61234567891),
    "zoom_1e-6": (1e-6, 0.24681357913),
}

def roots_poly(n, seed):
    rng = np.random.default_rng(seed)
    r = np.sort(rng.uniform(-1, 1, max(n // 2, 1)))
    def f(x):
        L = np.log(np.abs(x[:, None] - r[None, :]) + 1e-300).sum(1)
        s = np.prod(np.sign(x[:, None] - r[None, :]), 1)
        return s * np.exp(L - L.max())
    return f

def profiles(n):
    out = []
    out.append(("sin(n/2 x)", cheb_coeffs_from_values(lambda x: np.sin(0.5 * n * x), n)))
    out.append(("exp(x)cos(50x)", cheb_coeffs_from_values(lambda x: np.exp(x) * np.cos(50 * x), n)))
    out.append(("runge", cheb_coeffs_from_values(lambda x: 1 / (1 + 25 * x ** 2), n)))
    for s in range(3):
        out.append((f"roots_n/2_s{s}", cheb_coeffs_from_values(roots_poly(n, 100 + s), n)))
    for s in range(3):
        rng = np.random.default_rng(200 + s)
        out.append((f"random_s{s}", rng.standard_normal(n + 1)))
    for s in range(3):
        rng = np.random.default_rng(300 + s)
        rho = 1e-16 ** (1.0 / n)
        out.append((f"geom_to_eps_s{s}", rng.standard_normal(n + 1) * rho ** np.arange(n + 1)))
    rng = np.random.default_rng(400)
    out.append(("geom_0.8", rng.standard_normal(n + 1) * 0.8 ** np.arange(n + 1)))
    return out

SAVE_IDX = {1024, 8192}
def main():
    for (al, be) in TRANSFORMS.values():
        assert abs(al) + abs(be) <= 1.0, (al, be)
    rows = []; perindex = {}
    t0 = time.time()
    for n in NS:
        for pname, a in profiles(n):
            l1a = np.abs(a).sum(); E = bound(a)
            for tname, (al, be) in TRANSFORMS.items():
                rh, rl = ref_transform(a, al, be)
                reftrim = trimmed_length(rh, E)
                for meth in METHODS:
                    braw = apply_method(meth, a, al, be)
                    b = pad(braw, n + 1)
                    met, e = error_metrics(b, rh, rl, l1a, E)
                    met.update(n=n, profile=pname, transform=tname, alpha=al, beta=be, method=meth,
                               l1a=l1a, returned_len=len(braw), trimmed_len=trimmed_length(b, E),
                               ref_trimmed_len=reftrim)
                    rows.append(met)
                    if n in SAVE_IDX and pname in ("random_s0", "sin(n/2 x)", "geom_to_eps_s0", "runge"):
                        perindex[f"{n}|{pname}|{tname}|{meth}"] = np.abs(e) / l1a
                        perindex[f"{n}|{pname}|{tname}|ref"] = np.abs(rh + rl) / l1a
                        perindex[f"{n}|{pname}|{tname}|{meth}|coef"] = np.abs(b) / l1a
        print(f"n={n} done, {time.time()-t0:.0f}s", flush=True)
    keys = ["n", "profile", "transform", "alpha", "beta", "method", "l1a", "bound", "maxabs_err", "maxabs_rel",
            "l1_err", "l1_rel", "sup_err", "sup_rel", "l1_over_bound", "sup_over_bound", "viol_l1", "viol_sup",
            "frac_err_tophalf", "frac_err_top10", "n_negligible", "noise_max_rel", "noise_l1_rel",
            "returned_len", "trimmed_len", "ref_trimmed_len"]
    with open(os.path.join(HERE, "single_transform.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader()
        for r in rows: w.writerow({k: r[k] for k in keys})
    np.savez_compressed(os.path.join(HERE, "perindex.npz"), **perindex)

if __name__ == "__main__":
    main()
