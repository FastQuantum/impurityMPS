"""SIAM Green-function excitation B = c_0up^dag|gs>, star vs few-body (FBR).

Two figures:

1. cutoff study (small L=100): how the two truncation cutoffs -- the MPS/circuit
   cutoff (mps) and the orbital-activity cutoff (act) -- trade cost against
   accuracy. The tight reference spends bond dimension on noise: loosening mps
   drops chi and N_active with almost no change to Renyi-1/2 or to the impurity
   occupation n0up(t). Accuracy is measured as running max|n0up - n0up_ref|
   against the tightest run.

2. production (L=1000, t=500): star (full MPS) against FBR for U=0.05 and 0.1 --
   bond dimension, Renyi-1/2 at the bottleneck bond, wall time per step, and the
   FBR active-window size. The star saturates the bond ceiling (1024) and stops;
   the FBR keeps a small window.

Data: app/output/excitation_cutoff_siam_<method>_L<L>_U<U>_mc<mps>_ac<act>.dat
      (app/excitation_cutoff_siam.cpp)
Run:  python3 app/plot/excitation_cutoff.py      (figures in app/plot/figures/)
"""
import re

import matplotlib.pyplot as plt
import numpy as np

from common import APP, COLOR, load, save, seq_colors, style


def running_max_dev(d, ref, col="n0up"):
    ta, tb = np.round(d["t"], 6), np.round(ref["t"], 6)
    common, ia, ib = np.intersect1d(ta, tb, return_indices=True)
    dev = np.abs(d[col][ia] - ref[col][ib])
    return common, np.maximum.accumulate(dev)


def cutoff_study(L=100, U="0.1"):
    files = sorted(APP.glob(f"excitation_cutoff_siam_fbr_L{L}_U{U}_mc*_ac*.dat"))
    runs = []
    for f in files:
        m = re.search(r"_mc([\d.eE+-]+)_ac([\d.eE+-]+)\.dat", f.name)
        d = load(f)
        if d is not None and m:
            runs.append((float(m.group(1)), float(m.group(2)), m.group(1), m.group(2), d))
    if not runs:
        return []
    # tightest run (smallest mps, then smallest act) is the reference
    runs.sort(key=lambda r: (r[0], r[1]))
    ref = runs[0][4]

    # the diagonal scan mps==act, ordered loose->tight, is the headline story
    diag = sorted([r for r in runs if abs(r[0] - r[1]) < 1e-30], key=lambda r: r[0])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    # (a) chi and Renyi vs t: loose vs tight
    ax = axes[0]
    shades = seq_colors(len(diag))
    for (mc, ac, mcs, acs, d), c in zip(diag, shades):
        ax.plot(d["t"], d["bond_dim"], color=c, lw=1.8, label=f"cutoff {mcs}")
    style(ax, f"(a) L={L} U={U}: bond dim vs cutoff", ylabel="chi (max bond)")
    ax.legend(fontsize=7, frameon=False, title="mps=act")

    ax = axes[1]
    for (mc, ac, mcs, acs, d), c in zip(diag, shades):
        ax.plot(d["t"], d["renyi_half"], color=c, lw=1.8, label=f"{mcs}")
    style(ax, "(b) Renyi-1/2 barely moves with cutoff", ylabel="S_1/2 (bottleneck bond)")
    ax.legend(fontsize=7, frameon=False, title="mps=act")

    # (c) cost vs accuracy: two scans (vary mps at act=1e-9; vary act at mps=1e-9)
    ax = axes[2]
    def scan(fixed_idx, fixed_val):
        pts = []
        for mc, ac, mcs, acs, d in runs:
            other = ac if fixed_idx == 0 else mc
            fix = mc if fixed_idx == 0 else ac
            if abs(fix - fixed_val) < fix * 1e-6:
                _, dev = running_max_dev(d, ref)
                pts.append((other, d["wall_s"].sum(), dev.max()))
        return sorted(pts)
    for fixed_idx, fixed_val, mk, lab in [(1, 1e-9, "o", "vary mps (act=1e-9)"),
                                          (0, 1e-9, "s", "vary act (mps=1e-9)")]:
        pts = scan(fixed_idx, fixed_val)
        if pts:
            x = [p[2] for p in pts]; y = [p[1] for p in pts]
            ax.plot(x, y, mk + "-", lw=1.5, ms=6, label=lab)
            for cut, w, a in pts:
                ax.annotate(f"{cut:.0e}", (a, w), fontsize=6,
                            textcoords="offset points", xytext=(4, 3))
    style(ax, "(c) cost vs accuracy (label=loosened cutoff)",
          xlabel="max|n0up - n0up_ref|", ylabel="total wall (s)")
    ax.set_xscale("log")
    ax.legend(fontsize=7, frameon=False)
    return [save(fig, "excitation_cutoff_study.png")]


def production(L=1000, Us=("0.05", "0.1"), mc="1e-8", ac="1e-8"):
    def path(meth, U):
        return APP / f"excitation_cutoff_siam_{meth}_L{L}_U{U}_mc{mc}_ac{ac}.dat"
    have = {(m, U): load(path(m, U)) for m in ("star", "fbr") for U in Us}
    if all(v is None for v in have.values()):
        return []
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    styles = {"0.05": "-", "0.1": "--"}

    def draw(ax, col, title, ylabel, log=False):
        for (m, U), d in have.items():
            if d is None or col not in d:
                continue
            ax.plot(d["t"], d[col], color=COLOR[m], ls=styles.get(U, "-"), lw=1.8,
                    label=f"{m} U={U}")
        style(ax, title, ylabel=ylabel, log=log)
        ax.legend(fontsize=8, frameon=False)

    draw(axes[0, 0], "bond_dim", f"(a) L={L}: max bond dimension", "chi")
    # (b) Renyi-1/2: prefer the summed-over-bonds value (smooth); fall back to the
    # single max-bond value for any run that predates it (the star backup).
    axb = axes[0, 1]
    for (m, U), d in have.items():
        if d is None:
            continue
        col = "renyi_sum" if "renyi_sum" in d else "renyi_half"
        tag = "sum" if col == "renyi_sum" else "max-bond"
        axb.plot(d["t"], d[col], color=COLOR[m], ls=styles.get(U, "-"), lw=1.8,
                 label=f"{m} U={U} ({tag})")
    style(axb, "(b) Renyi-1/2 entanglement", ylabel="S_1/2  (FBR: sum over bonds)")
    axb.legend(fontsize=8, frameon=False)
    # per-step wall, lightly smoothed
    for (m, U), d in have.items():
        if d is None:
            continue
        w = d["wall_s"]
        k = max(1, len(w) // 200)
        ws = np.convolve(w, np.ones(k) / k, mode="same") if k > 1 else w
        axes[1, 0].plot(d["t"], ws, color=COLOR[m], ls=styles.get(U, "-"), lw=1.5,
                        label=f"{m} U={U}")
    style(axes[1, 0], "(c) wall time per step", ylabel="s / step")
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 0].legend(fontsize=8, frameon=False)
    # FBR window (star has no window: it keeps all L)
    for (m, U), d in have.items():
        if d is None or m != "fbr":
            continue
        axes[1, 1].plot(d["t"], d["n_active"], color=COLOR["fbr"], ls=styles.get(U, "-"),
                        lw=1.8, label=f"fbr U={U}")
    style(axes[1, 1], "(d) FBR active-window size", ylabel="N_active")
    axes[1, 1].legend(fontsize=8, frameon=False)
    return [save(fig, "excitation_production.png")]


if __name__ == "__main__":
    made = []
    made += cutoff_study()
    made += production()
    print("wrote:", ", ".join(made) if made else "nothing (no data yet)")
