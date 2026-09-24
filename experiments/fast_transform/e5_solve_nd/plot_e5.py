"""Plots and summary tables for E5 (no pandas: the project venv does not ship it).

Reads results_*.csv / calls_*.csv in this directory (files with 'trunc' in the name are the
truncated-NUFFT variant run) and writes e5_*.csv, e5_*.png, e5_table*.md.
"""
import os
import csv
import glob
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load(prefix, trunc):
    rows = []
    for f in sorted(glob.glob(os.path.join(HERE, f"{prefix}_*.csv"))):
        b = os.path.basename(f)
        if b.startswith("e5_") or (("trunc" in b) != trunc):
            continue
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    for r in rows:
        if not r.get("status"):
            r["status"] = "ok"
    return rows


def write(rows, name):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(HERE, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


res, calls = load("results", False), load("calls", False)
res_tr, calls_tr = load("results", True), load("calls", True)
write(res, "e5_results_all.csv"); write(calls, "e5_calls_all.csv")
write(res_tr, "e5_results_trunc_variant.csv"); write(calls_tr, "e5_calls_trunc_variant.csv")

R = {(r["problem"], r["method"]): r for r in res}
RT = {(r["problem"], r["method"]): r for r in res_tr}

METHODS = ["dense", "fast", "auto_nd12_d32", "auto_nd12_d64", "auto_nd12_d256", "auto_ndall_d16"]
COLORS = {"dense": "#52514e", "fast": "#eb6834", "auto_nd12_d32": "#2a78d6",
          "auto_nd12_d64": "#1baf7a", "auto_nd12_d256": "#4a3aa7", "auto_ndall_d16": "#e87ba4"}
MARK = {"fast": "o", "auto_nd12_d32": "s", "auto_nd12_d64": "^", "auto_nd12_d256": "v",
        "auto_ndall_d16": "D"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.grid": True, "grid.color": "#e5e5e5",
                     "grid.linewidth": 0.6})

GORDER = {"2D-chebfun2": 0, "2D-constructed": 1, "5D": 2}
probs = sorted({(GORDER[r["group"]], int(r["max_degree"]), r["problem"], int(r["dim"]))
                for r in res if r["method"] == "dense"})
pnames = [p[2] for p in probs]
pdeg = [p[1] for p in probs]
pdim = [p[3] for p in probs]
pgrp = [p[0] for p in probs]


def tmed(p, m, table=R):
    r = table[(p, m)]
    return fl(r["t_median"]) if r["status"] == "ok" else np.nan


# ---------------- Fig 1: time ratio vs dense
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(pnames))
for i, m in enumerate(METHODS[1:]):
    ratio = np.array([tmed(p, m) / tmed(p, "dense") for p in pnames])
    failed = np.array([R[(p, m)]["status"] != "ok" for p in pnames])
    xs = x + (i - 2) * 0.13
    ax.scatter(xs[~failed], ratio[~failed], s=24, marker=MARK[m], color=COLORS[m], label=m,
               zorder=3, edgecolor="white", linewidth=0.5)
    for j in np.where(failed)[0]:
        tf = fl(R[(pnames[j], m)]["t_instrumented"]) / tmed(pnames[j], "dense")
        ax.scatter([xs[j]], [tf], s=60, marker="x", color=COLORS[m], zorder=4, linewidth=1.6)
        ax.annotate(f"{m} FAILED\n({R[(pnames[j], m)]['status']})", (xs[j], tf),
                    xytext=(6, 2), textcoords="offset points", fontsize=7, color="#333")
ax.axhline(1, color="#333", lw=1)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([f"{p} (deg {d})" for p, d in zip(pnames, pdeg)], rotation=70, ha="right",
                   fontsize=7.5)
ax.set_ylabel("median solve time / dense   (log scale)")
ax.set_title("E5: end-to-end yroots.solve time relative to dense transform "
             "(below 1 = faster; x = solver failure, plotted at time-to-failure)", loc="left")
for j in range(1, len(pgrp)):
    if pgrp[j] != pgrp[j - 1]:
        ax.axvline(j - 0.5, color="#aaa", lw=0.8, ls="--")
ax.text(0, 0.02, "2D chebfun2 suite", transform=ax.get_xaxis_transform(), fontsize=8, color="#555")
j2 = pgrp.index(1); j5 = pgrp.index(2)
ax.text(j2, 0.02, "2D constructed", transform=ax.get_xaxis_transform(), fontsize=8, color="#555")
ax.text(j5, 0.02, "5D", transform=ax.get_xaxis_transform(), fontsize=8, color="#555")
ax.legend(ncol=5, frameon=False, loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "e5_time_ratio.png"), dpi=150)
plt.close(fig)

# ---------------- Fig 2: per-column cost vs degree, dense vs NUFFT path
nt = [c for c in calls if c["trivial"] == "0"]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
for ax, nd in zip(axes, ["2", "5"]):
    for fast, lab, col in [("0", "dense (numba) path", COLORS["dense"]),
                           ("1", "NUFFT path", COLORS["fast"])]:
        agg = defaultdict(lambda: [0.0, 0.0])
        for c in nt:
            if c["ndim"] == nd and c["fast"] == fast:
                a = agg[int(c["degree"])]
                a[0] += fl(c["n_calls"]) * fl(c["mean_ntrans"])
                a[1] += fl(c["t_total"])
        if agg:
            d = np.array(sorted(agg))
            v = np.array([agg[k][1] / agg[k][0] * 1e6 for k in d])
            ax.scatter(d, v, s=10, color=col, label=lab, alpha=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("input degree along transformed axis")
    ax.set_ylabel("time per transformed 1D column (us)")
    ax.set_title(f"{nd}D tensors: cost per 1D column vs input degree\n(pooled over "
                 "problems and methods; dense cost includes truncation)", loc="left", fontsize=9)
    ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "e5_percolumn_vs_degree.png"), dpi=150)
plt.close(fig)

# ---------------- Fig 3: degree distribution of transform calls / time under dense
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
for ax, nd in zip(axes, ["2", "5"]):
    agg = defaultdict(lambda: [0.0, 0.0])
    for c in nt:
        if c["method"] == "dense" and c["ndim"] == nd:
            a = agg[int(c["degree"])]
            a[0] += fl(c["n_calls"]); a[1] += fl(c["t_total"])
    d = np.array(sorted(agg))
    n = np.array([agg[k][0] for k in d]); t = np.array([agg[k][1] for k in d])
    ax.step(d, np.cumsum(n) / n.sum(), where="post", color=COLORS["auto_nd12_d32"], lw=2,
            label="fraction of calls")
    ax.step(d, np.cumsum(t) / t.sum(), where="post", color=COLORS["fast"], lw=2,
            label="fraction of transform time")
    for dd in (16, 32, 64, 256):
        ax.axvline(dd, color="#bbb", lw=0.8, ls=":")
        ax.text(dd * 1.05, 0.93, f"deg {dd}", fontsize=7, color="#777", ha="left",
                transform=ax.get_xaxis_transform())
    ax.set_xscale("log"); ax.set_ylim(0, 1.05)
    ax.set_xlabel("input degree along transformed axis")
    ax.set_ylabel("cumulative fraction (<= degree)")
    ax.set_title(f"{nd}D problems, dense method: degree of nontrivial transform calls",
                 loc="left", fontsize=9)
    ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "e5_degree_distribution.png"), dpi=150)
plt.close(fig)


# ---------------- main table
def fmt_t(table, p, m):
    if (p, m) not in table:
        return "-"
    r = table[(p, m)]
    return f"{fl(r['t_median']):.3g}" if r["status"] == "ok" else f"FAIL ({r['status']})"


def agree(table, p, methods):
    out = []
    for m in methods:
        r = table[(p, m)]
        if r["status"] != "ok":
            out.append(f"{m}:FAIL")
            continue
        miss, sp = int(fl(r["n_missing"])), int(fl(r["n_spurious"]))
        if miss or sp:
            out.append(f"{m}:{int(fl(r['n_found']))} (-{miss}/+{sp})")
    nref = int(fl(table[(p, methods[0])]["n_ref"]))
    return (f"all {nref}/{nref}" if not out else f"ref {nref}; " + "; ".join(out))


hdr = ("| problem | dim | max deg | n_ref | " + " | ".join(f"{m} (s)" for m in METHODS)
       + " | xform share dense / fast | xform calls (dense) | roots vs ref | max err dense / worst other |")
lines = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
for p, d, dim in zip(pnames, pdeg, pdim):
    dr, fr = R[(p, "dense")], R[(p, "fast")]
    errs = [fl(R[(p, m)]["max_err_vs_ref"]) for m in METHODS[1:] if R[(p, m)]["status"] == "ok"]
    lines.append("| " + " | ".join([
        p, str(dim), str(d), dr["n_ref"], *[fmt_t(R, p, m) for m in METHODS],
        f"{fl(dr['transform_share']):.0%} / "
        + (f"{fl(fr['transform_share']):.0%}" if fr["status"] == "ok" else "-"),
        dr["n_calls_nontrivial"], agree(R, p, METHODS),
        f"{fl(dr['max_err_vs_ref']):.1e} / {np.nanmax(errs):.1e}"]) + " |")
with open(os.path.join(HERE, "e5_table.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))

print()
summary = []
for g in ["2D-chebfun2", "2D-constructed", "5D"]:
    ps = [p for p, gg in zip(pnames, pgrp) if gg == GORDER[g]]
    for m in METHODS[1:]:
        r = np.array([tmed(p, m) / tmed(p, "dense") for p in ps])
        nf = int(np.isnan(r).sum()); r = r[~np.isnan(r)]
        s = (f"{g:15s} {m:15s} time/dense geomean={np.exp(np.log(r).mean()):6.2f} "
             f"best={r.min():5.2f} worst={r.max():6.1f} failures={nf}/{len(ps)}")
        summary.append(s); print(s)
with open(os.path.join(HERE, "e5_summary.txt"), "w") as f:
    f.write("\n".join(summary) + "\n")

# ---------------- truncated-NUFFT variant table
TM = ["dense", "fasttrunc", "autotrunc_nd12_d64", "autotrunc_ndall_d16"]
hdr = ("| problem | dim | max deg | " + " | ".join(f"{m} (s)" for m in TM)
       + " | fast, untruncated (s) | auto_nd12_d64, untruncated (s) | auto_ndall_d16, untruncated (s) | roots vs ref (trunc methods) |")
lines = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
for p, d, dim in zip(pnames, pdeg, pdim):
    if (p, "dense") not in RT:
        continue
    lines.append(f"| {p} | {dim} | {d} | " + " | ".join(fmt_t(RT, p, m) for m in TM)
                 + f" | {fmt_t(R, p, 'fast')} | {fmt_t(R, p, 'auto_nd12_d64')} | "
                 f"{fmt_t(R, p, 'auto_ndall_d16')} | {agree(RT, p, TM)} |")
with open(os.path.join(HERE, "e5_table_trunc.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n" + "\n".join(lines))
