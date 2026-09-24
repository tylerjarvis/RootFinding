"""E3 analysis: plots and summary tables from timing.csv and accuracy.csv (numpy/csv only)."""
import os
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = 2.0 ** -52
COL = {"dense": "#1f77b4", "fast": "#d62728", "fast_mt4": "#ff9896", "exact": "#2ca02c"}
PARAM_ORDER = ["sub_lo", "sub_hi", "zoom0.1", "zoom1e-3"]
KINDS = ["decay", "unit"]


def read(fn):
    with open(os.path.join(HERE, fn)) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        p = r["shape"].split("x")
        r["n"] = int(p[0])
        r["square"] = len(set(p)) == 1
        r["ndim"] = int(r["ndim"])
        for k in ("median_s", "p10_s", "p90_s", "err_over_sumabs", "err_over_bound"):
            if k in r:
                r[k] = float(r[k])
        if "violation" in r:
            r["violation"] = int(r["violation"])
    return rows


def sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def tindex(T):
    """(ndim, shape, param, coeffs, op, method) -> row"""
    return {(r["ndim"], r["shape"], r["param"], r["coeffs"], r["op"], r["method"]): r for r in T}


def square_ns(T, nd):
    return sorted({r["n"] for r in T if r["ndim"] == nd and r["square"]})


def plot_timing(T, nd):
    ops = ["0", str(nd - 1), "full"]
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), sharex=True)
    for i, op in enumerate(ops):
        for j, p in enumerate(PARAM_ORDER):
            ax = axes[i, j]
            for m in ("dense", "fast", "fast_mt4"):
                for kind, ls in (("decay", "-"), ("unit", "--")):
                    d = sorted(sel(T, ndim=nd, square=True, op=op, param=p, method=m, coeffs=kind),
                               key=lambda r: r["n"])
                    if not d:
                        continue
                    n = [r["n"] for r in d]
                    ax.plot(n, [r["median_s"] for r in d], ls, color=COL[m], marker="o", ms=3,
                            label=f"{m} ({kind})")
                    ax.fill_between(n, [r["p10_s"] for r in d], [r["p90_s"] for r in d],
                                    color=COL[m], alpha=0.12)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            name = "full transformCheb" if op == "full" else f"TransformChebInPlaceND axis {op}"
            ax.set_title(f"{nd}D {name}, {p}", fontsize=9)
            if j == 0:
                ax.set_ylabel("time [s] (median; p10-p90 band)")
            if i == 2:
                ax.set_xlabel(f"per-axis size n (shape n^{nd})")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fn = os.path.join(HERE, f"timing_{nd}d.png")
    fig.savefig(fn, dpi=120)
    plt.close(fig)
    return fn


def ratio(TI, nd, shape, p, kind, op, m="fast"):
    try:
        return TI[(nd, shape, p, kind, op, "dense")]["median_s"] / TI[(nd, shape, p, kind, op, m)]["median_s"]
    except KeyError:
        return np.nan


def plot_speedup(T, TI):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, nd in enumerate((2, 5)):
        ns = square_ns(T, nd)
        if not ns:
            continue
        for row, op in enumerate(("0", "full")):
            ax = axes[row, col]
            for p, c in zip(PARAM_ORDER, ["C0", "C1", "C2", "C3"]):
                for kind, ls in (("decay", "-"), ("unit", "--")):
                    r = [ratio(TI, nd, "x".join([str(n)] * nd), p, kind, op) for n in ns]
                    ax.plot(ns, r, ls, color=c, marker="o", ms=3, label=f"{p} {kind}")
            ax.axhline(1, color="k", lw=1)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            name = "full transformCheb" if op == "full" else "axis 0"
            ax.set_title(f"{nd}D {name}: dense time / fast time (>1: fast wins)", fontsize=10)
            ax.set_xlabel("per-axis size n")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fn = os.path.join(HERE, "speedup.png")
    fig.savefig(fn, dpi=120)
    plt.close(fig)
    return fn


def plot_accuracy(A):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    for i, nd in enumerate((2, 5)):
        for j, p in enumerate(PARAM_ORDER):
            ax = axes[i, j]
            for m, mk, ms in (("exact", "o", 6), ("dense", "x", 5), ("fast", "o", 3)):
                for kind, ls in (("decay", "-"), ("unit", "--")):
                    d = sorted(sel(A, ndim=nd, square=True, op="0", param=p, method=m, coeffs=kind),
                               key=lambda r: r["n"])
                    ax.plot([r["n"] for r in d], [max(r["err_over_sumabs"], 1e-20) for r in d], ls,
                            color=COL[m], marker=mk, ms=ms, mfc="none" if m == "exact" else None,
                            label=f"{m} ({kind})")
            ns = np.array(sorted({r["n"] for r in A if r["ndim"] == nd and r["square"]}))
            ax.plot(ns, ns * EPS, "k:", label="bound n*2^-52")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            ax.set_title(f"{nd}D axis 0, {p}", fontsize=10)
            if j == 0:
                ax.set_ylabel("max abs err / sum|M|")
            ax.set_xlabel("per-axis size n")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fn = os.path.join(HERE, "accuracy.png")
    fig.savefig(fn, dpi=120)
    plt.close(fig)
    return fn


def crossover(T, TI):
    out = []
    for nd in (2, 5):
        ns = square_ns(T, nd)
        ops = [str(d) for d in range(nd)] + ["full"]
        for p in PARAM_ORDER:
            for kind in KINDS:
                for op in ops:
                    r = np.array([ratio(TI, nd, "x".join([str(n)] * nd), p, kind, op) for n in ns])
                    rmt = np.array([ratio(TI, nd, "x".join([str(n)] * nd), p, kind, op, "fast_mt4")
                                    for n in ns])
                    if np.all(np.isnan(r)):
                        continue
                    first = next((n for n, x in zip(ns, r) if x > 1), None)
                    stable = None
                    for n, x in zip(ns[::-1], r[::-1]):
                        if x > 1:
                            stable = n
                        else:
                            break
                    k = int(np.nanargmax(r))
                    out.append(dict(ndim=nd, param=p, coeffs=kind, op=op, first_win_n=first,
                                    stable_win_n=stable, max_speedup=round(float(r[k]), 3),
                                    at_n=ns[k], min_ratio=round(float(np.nanmin(r)), 3),
                                    max_speedup_mt4=round(float(np.nanmax(rmt)), 3)))
    return out


def write_csv(rows, fn):
    with open(os.path.join(HERE, fn), "w", newline="") as f:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def ratio_table(T, TI, nd, ns, op):
    print(f"\n## {nd}D op={op}: dense/fast (fast_mt4 in brackets)")
    print("param coeffs | " + " ".join(f"n={n:<11}" for n in ns))
    for p in PARAM_ORDER:
        for kind in KINDS:
            cells = []
            for n in ns:
                s = "x".join([str(n)] * nd)
                cells.append(f"{ratio(TI, nd, s, p, kind, op):5.2f}[{ratio(TI, nd, s, p, kind, op, 'fast_mt4'):5.2f}]")
            print(f"{p} {kind} | " + " ".join(cells))


def abs_table(T, TI, nd, ns, op, p, kind):
    print(f"\n## {nd}D op={op} {p} {kind}: median ms dense / fast (p10-p90 dense; fast)")
    for n in ns:
        s = "x".join([str(n)] * nd)
        d = TI.get((nd, s, p, kind, op, "dense"))
        f = TI.get((nd, s, p, kind, op, "fast"))
        if d and f:
            print(f"n={n}: {d['median_s']*1e3:.4g} ({d['p10_s']*1e3:.3g}-{d['p90_s']*1e3:.3g}) / "
                  f"{f['median_s']*1e3:.4g} ({f['p10_s']*1e3:.3g}-{f['p90_s']*1e3:.3g})  out_dense={d['out_shape']}")


def skew(T, TI):
    shapes = sorted({(r["ndim"], r["shape"]) for r in T if not r["square"]})
    rows = []
    for nd, s in shapes:
        ops = [str(d) for d in range(nd)] + ["full"]
        for p in PARAM_ORDER:
            for kind in KINDS:
                d = dict(ndim=nd, shape=s, param=p, coeffs=kind)
                for op in ops:
                    d[f"r_{op}"] = round(ratio(TI, nd, s, p, kind, op), 3)
                rows.append(d)
    return rows


if __name__ == "__main__":
    T = read("timing.csv")
    TI = tindex(T)
    for nd in sorted({r["ndim"] for r in T}):
        print(plot_timing(T, nd))
    print(plot_speedup(T, TI))
    A = read("accuracy.csv") if os.path.exists(os.path.join(HERE, "accuracy.csv")) else None
    if A:
        print(plot_accuracy(A))
    co = crossover(T, TI)
    write_csv(co, "crossover.csv")
    print("\n## crossover (per-axis n; first_win = smallest n with fast faster; stable = fast faster for all n >= this)")
    for r in co:
        print(r)
    for nd, ns in ((2, [8, 16, 32, 48, 64, 96, 128, 256, 512, 1024]), (5, [3, 4, 5, 6, 8, 10, 12, 16, 20])):
        if nd not in {r["ndim"] for r in T}:
            continue
        for op in ("0", str(nd - 1), "full"):
            ratio_table(T, TI, nd, ns, op)
        for p in ("sub_hi", "zoom1e-3"):
            abs_table(T, TI, nd, ns, "full", p, "unit")
    sk = skew(T, TI)
    write_csv(sk, "skewed_summary.csv")
    print("\n## skewed shapes: dense/fast per op")
    for r in sk:
        print(r)
    if A:
        print("\n## accuracy summary")
        groups = defaultdict(list)
        for r in A:
            groups[(r["ndim"], r["method"], r["coeffs"], r["op"] == "full")].append(r)
        for k in sorted(groups):
            g = groups[k]
            eb = np.array([r["err_over_bound"] for r in g])
            es = np.array([r["err_over_sumabs"] for r in g])
            print(f"ndim={k[0]} {k[1]:6s} {k[2]:5s} full={k[3]!s:5s}: max err/bound={eb.max():.3g} "
                  f"median={np.median(eb):.3g} max err/sum|M|={es.max():.3g} violations={int(sum(r['violation'] for r in g))}/{len(g)}")
        print("\n## violations")
        for r in A:
            if r["violation"]:
                print({k: r[k] for k in ("ndim", "shape", "param", "coeffs", "op", "method",
                                         "err_over_sumabs", "err_over_bound")})
