"""Cost of an Fbr_dyn timestep against L, phase by phase, for the three layouts.

Data: app/output/bench_dyn_cost_{sym,block,ns}_L<L>_U<U>.dat  (app/bench_dyn_cost.cpp)
Each file is one run of the same number of steps; at fixed t the window and the
bond dimension do not depend on L, so the time of the run against L is the
scaling of a timestep. The slope is the least-squares fit of log(time) against
log(L) over L >= 1000, where the O(L) and O(1) parts no longer matter.
Run:  python3 app/plot/bench_dyn_cost.py     (figures in app/plot/figures/)
"""
import re

import matplotlib.pyplot as plt
import numpy as np

from common import by_length, save, style

LAYOUTS = [("sym", "spin_symmetric, SIAM U=0.1"),
           ("block", "spin_block, SIAM U=0.1"),
           ("ns", "leading (spinless), IRLM U=0.2")]

# phase -> (columns summed, colour, label)
PHASES = {
    "total":  (["total"], "#2b2b2a", "whole step"),
    "orbit":  (["buildK", "plan", "applyK", "applyfb"], "#2a78d6", "orbital update (build_K, plan, apply)"),
    "buildK": (["buildK"], "#7fb2ea", "build_K only"),
    "tdvp":   (["tdvp"], "#1baf7a", "TDVP in the window"),
}


def measure_times(d):
    """(library correlator(0,0) s, O(L^2) version s) from the '# measure:' line."""
    for line in d.header:
        m = re.search(r"correlator\(0,0\) ([-+0-9.eE]+) s, O\(L\^2\) version ([-+0-9.eE]+) s", line)
        if m:
            return float(m.group(1)), float(m.group(2))
    return np.nan, np.nan


def slope(Ls, y, Lmin=1000):
    keep = (Ls >= Lmin) & np.isfinite(y) & (y > 0)
    if keep.sum() < 2:
        return np.nan
    return np.polyfit(np.log(Ls[keep]), np.log(y[keep]), 1)[0]


def table():
    """{layout: dict of arrays over L} for every layout with data."""
    out = {}
    for lay, _ in LAYOUTS:
        runs = {L: d for L, d in by_length(f"bench_dyn_cost_{lay}_L*_U*.dat").items() if d is not None and len(d)}
        if not runs:
            continue
        nstep = min(len(d) for d in runs.values())   # compare the same steps at every L
        Ls = np.array(list(runs), dtype=float)
        row = {"L": Ls, "nstep": nstep}
        for key, (cols, _, _) in PHASES.items():
            row[key] = np.array([sum(np.sum(d[c][:nstep]) for c in cols) for d in runs.values()]) / nstep
        row["n_active"] = np.array([d["n_active"][nstep - 1] for d in runs.values()])
        row["chi"] = np.array([d["chi"][nstep - 1] for d in runs.values()])
        m = np.array([measure_times(d) for d in runs.values()])
        row["corr_lib"], row["corr_L2"] = m[:, 0], m[:, 1]
        out[lay] = row
    return out


def make():
    tab = table()
    if not tab:
        return []
    fig, axes = plt.subplots(1, len(tab), figsize=(5.2 * len(tab), 4.4), sharey=True, squeeze=False)
    notes = []
    for ax, (lay, title) in zip(axes[0], [x for x in LAYOUTS if x[0] in tab]):
        r = tab[lay]
        Ls = r["L"]
        for key, (_, color, label) in PHASES.items():
            ax.plot(Ls, r[key], "o-", color=color, lw=2, ms=6,
                    label=f"{label}  (slope {slope(Ls, r[key]):.2f})")
        ax.plot(Ls, r["corr_lib"], "s--", color="#eb6834", lw=1.5, ms=6,
                label=f"correlator(0,0), library  (slope {slope(Ls, r['corr_lib']):.2f})")
        ax.plot(Ls, r["corr_L2"], "s:", color="#eb6834", lw=1.5, ms=6,
                label=f"correlator(0,0), two rows  (slope {slope(Ls, r['corr_L2']):.2f})")
        # guides through the largest-L whole-step point
        x = np.array([Ls.min(), Ls.max()])
        y0 = r["total"][-1]
        for p, ls in ((2, (0, (4, 3))), (3, (0, (1, 2)))):
            ax.plot(x, y0 * (x / Ls.max()) ** p, color="#8a8983", lw=1, ls=ls)
            ax.text(x[0], y0 * (x[0] / Ls.max()) ** p, f" L^{p}", fontsize=8, color="#52514e", va="bottom")
        ax.set_xscale("log")
        style(ax, f"{title}, t={r['nstep'] * 0.1:g}", xlabel="L",
              ylabel="s per step (mean)" if ax is axes[0, 0] else None, log=True)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        notes.append(f"{lay}: step slope {slope(Ls, r['total']):.2f}, orbital-update slope "
                     f"{slope(Ls, r['orbit']):.2f}, n_active {int(r['n_active'].min())}-{int(r['n_active'].max())}, "
                     f"chi {int(r['chi'].min())}-{int(r['chi'].max())}")
    return [(save(fig, "bench_dyn_cost.png"),
             "Fbr_dyn wall time per step against L (mean over the run, one thread). Dashed and dotted "
             "grey lines are L^2 and L^3 through the largest-L step. Slopes fitted over L>=1000. "
             + "; ".join(notes) + ".")]


if __name__ == "__main__":
    tab = table()
    for lay, r in tab.items():
        print(f"== {lay}  (mean over the first {r['nstep']} steps)")
        keys = ["total", "orbit", "buildK", "tdvp", "corr_lib", "corr_L2"]
        print("       L " + " ".join(f"{k:>10}" for k in keys) + "  n_active chi")
        for i, L in enumerate(r["L"]):
            print(f"{int(L):8d} " + " ".join(f"{r[k][i]:10.4g}" for k in keys)
                  + f"  {int(r['n_active'][i]):8d} {int(r['chi'][i]):3d}")
        print("   slope " + " ".join(f"{slope(r['L'], r[k]):10.2f}" for k in keys))
    for name, caption in make():
        print(name, "-", caption)
