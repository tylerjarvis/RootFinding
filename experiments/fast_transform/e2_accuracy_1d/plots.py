import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tab import load
HERE = os.path.dirname(os.path.abspath(__file__))
COL = {"dense": "#2a78d6", "dense_ef": "#eb6834", "fast_1e-15": "#1baf7a", "fast_1e-14": "#eda100", "fast_1e-12": "#e87ba4"}
MK = {"dense": "o", "dense_ef": "s", "fast_1e-15": "^", "fast_1e-14": "v", "fast_1e-12": "D"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "axes.spines.top": False, "axes.spines.right": False})

d = load(os.path.join(HERE, "single_transform.csv"))
NS = sorted({r["n"] for r in d})
def agg(rows, key, fn):
    return [fn([r[key] for r in rows if r["n"] == n]) for n in NS]

# 1. error vs n per method: split-type vs zoom-type transforms; l1 error / ||a||_1 and ratio to bound
groups = {"subdivision splits (touch an endpoint)": lambda r: "split" in r["transform"],
          "interior zooms (0.1, 1e-3, 1e-6)": lambda r: r["transform"] in ("zoom_1e-1", "zoom_1e-3", "zoom_1e-6"),
          "edge zoom (0.1, -0.9)": lambda r: r["transform"] == "zoom_1e-1_edge"}
fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
for j, (gname, g) in enumerate(groups.items()):
    for meth in COL:
        rows = [r for r in d if r["method"] == meth and g(r)]
        med = agg(rows, "l1_rel", np.median); mx = agg(rows, "l1_rel", np.max)
        axs[0, j].loglog(NS, med, marker=MK[meth], color=COL[meth], lw=2, ms=5, label=meth)
        axs[0, j].loglog(NS, mx, color=COL[meth], lw=1, ls="--")
        med = agg(rows, "l1_over_bound", np.median); mx = agg(rows, "l1_over_bound", np.max)
        axs[1, j].loglog(NS, med, marker=MK[meth], color=COL[meth], lw=2, ms=5, label=meth)
        axs[1, j].loglog(NS, mx, color=COL[meth], lw=1, ls="--")
    axs[0, j].loglog(NS, [(n + 1) * 2 ** -52 for n in NS], color="#0b0b0b", lw=1.5, ls=":", label="bound (n+1)u")
    axs[1, j].axhline(1.0, color="#0b0b0b", lw=1.5, ls=":")
    axs[0, j].set_title(gname); axs[1, j].set_xlabel("degree n")
axs[0, 0].set_ylabel("||b - b_exact||_1 / ||a||_1\n(solid: median, dashed: worst)")
axs[1, 0].set_ylabel("||b - b_exact||_1 / getTransformationError\n(>1 = bound violated)")
axs[0, 0].legend(fontsize=8)
fig.suptitle("E2: single affine transform error vs degree (all 13 coefficient profiles)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "single_error_vs_n.png"), dpi=130); plt.close(fig)

# 1b. sup-norm vs l1 for fast_1e-15 (sanity) and dense
fig, ax = plt.subplots(figsize=(6, 4.5))
for meth in ["dense", "dense_ef", "fast_1e-15"]:
    rows = [r for r in d if r["method"] == meth]
    ax.loglog(NS, agg(rows, "sup_over_bound", np.max), marker=MK[meth], color=COL[meth], lw=2, label=meth + " (worst)")
    ax.loglog(NS, agg(rows, "sup_over_bound", np.median), color=COL[meth], lw=1, ls="--")
ax.axhline(1, color="#0b0b0b", ls=":"); ax.set_xlabel("degree n")
ax.set_ylabel("sup_[-1,1] |q - q_exact| / getTransformationError")
ax.set_title("Sup-norm error vs bound (solid worst, dashed median)"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(HERE, "single_sup_vs_bound.png"), dpi=130); plt.close(fig)

# 2. error along coefficient index
P = np.load(os.path.join(HERE, "perindex.npz"))
cases = [("random_s0", "offsplit_R"), ("sin(n/2 x)", "offsplit_L"), ("runge", "zoom_1e-3"), ("geom_to_eps_s0", "zoom_1e-6")]
for n in [1024, 8192]:
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, (pn, tn) in zip(axs.flat, cases):
        k = np.arange(n + 1)
        ref = P[f"{n}|{pn}|{tn}|ref"]
        ax.semilogy(k, np.maximum(ref, 1e-40), color="#8a8984", lw=1, label="|exact coeff| / ||a||_1")
        for meth in ["dense", "fast_1e-15", "fast_1e-12"]:
            e = P[f"{n}|{pn}|{tn}|{meth}"]
            ax.semilogy(k, np.maximum(e, 1e-40), color=COL[meth], lw=0.8, alpha=0.85, label=f"|err| {meth}")
        ax.axhline((n + 1) * 2 ** -52, color="#0b0b0b", ls=":", lw=1, label="bound/||a||_1")
        ax.set_ylim(1e-36, 10); ax.set_title(f"{pn}, {tn}, n={n}"); ax.set_xlabel("coefficient index k")
    axs[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Error per Chebyshev coefficient (relative to ||a||_1); values < 1e-36 clipped")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, f"error_along_index_n{n}.png"), dpi=130); plt.close(fig)

# 3. accumulated error vs depth
dp = load(os.path.join(HERE, "repeated_transform.csv"))
fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
for j, path in enumerate(["split", "mixed", "zoomdeep"]):
    for meth in COL:
        for n, ls in [(512, "-"), (4096, "--")]:
            rows = [r for r in dp if r["path"] == path and r["method"] == meth and r["n"] == n]
            D = sorted({r["depth"] for r in rows})
            y = [max(r["l1_rel0"] for r in rows if r["depth"] == dd) for dd in D]
            yb = [max(r["l1_over_bound"] for r in rows if r["depth"] == dd) for dd in D]
            lab = f"{meth} n={n}" if j == 0 else None
            axs[0, j].semilogy(D, y, color=COL[meth], ls=ls, lw=1.5, label=lab)
            axs[1, j].semilogy(D, yb, color=COL[meth], ls=ls, lw=1.5)
    axs[1, j].axhline(1, color="#0b0b0b", ls=":")
    axs[0, j].set_title(f"path: {path}"); axs[1, j].set_xlabel("depth (number of transforms)")
axs[0, 0].set_ylabel("||b_d - b_exact,d||_1 / ||a_0||_1 (worst of 3 polys)")
axs[1, 0].set_ylabel("error / accumulated getTransformationError")
axs[0, 0].legend(fontsize=7, ncol=2)
fig.suptitle("E2: accumulated error along simulated subdivision/zoom paths (solid n=512, dashed n=4096)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "accumulated_error_vs_depth.png"), dpi=130); plt.close(fig)

# 4. local per-step errors along paths
lp = os.path.join(HERE, "repeated_local_steps.csv")
if os.path.exists(lp):
    dl = load(lp)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for j, path in enumerate(["split", "mixed", "zoomdeep"]):
        for meth in ["dense", "fast_1e-15", "fast_1e-12"]:
            for n, ls in [(512, "-"), (4096, "--")]:
                rows = [r for r in dl if r["path"] == path and r["method"] == meth and r["n"] == n]
                D = sorted({r["depth"] for r in rows})
                y = [max(r["local_over_bound"] for r in rows if r["depth"] == dd) for dd in D]
                axs[j].semilogy(D, y, color=COL[meth], ls=ls, lw=1.5, label=f"{meth} n={n}" if j == 0 else None)
        axs[j].axhline(1, color="#0b0b0b", ls=":"); axs[j].set_title(f"path: {path}"); axs[j].set_xlabel("step")
    axs[0].set_ylabel("per-step error / per-step bound (worst of 3 polys)"); axs[0].legend(fontsize=7); axs[0].set_ylim(1e-6, 1e4)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "local_step_error_vs_bound.png"), dpi=130); plt.close(fig)
print("ok")
