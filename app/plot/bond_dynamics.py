"""Bond dimension during real-time evolution: few-body (FBR) against full MPS.

1. Fig. 2 of the paper (docs/TD_NO_paper-3.pdf): IRLM quench, U=0.2, V=0.1, L=200,
   dt=0.1 -- max bond dimension for the real-space chain, the full star and the FBR,
   with the FBR active window as an inset; plus the impurity occupation difference
   to the FBR, as a check that the three runs are the same physics.
2. SIAM Green-function excitation c_0up^dag|gs>, scanned over U: full star against
   the FBR (spin_block), L=100 and 200.

Data: app/output/quench_bond_irlm_<method>_L200_U0.2.dat   (app/quench_bond_irlm.cpp)
      app/output/excitation_bond_siam_<method>_L<L>_U<U>.dat (app/excitation_bond_siam.cpp)
Run:  python3 app/plot/bond_dynamics.py     (figures in app/plot/figures/)
"""
import re

import matplotlib.pyplot as plt
import numpy as np

from common import APP, COLOR, load, save, seq_colors, style

IRLM_METHODS = [("chain", "real-space chain (full MPS)"),
                ("star", "star (full MPS)"),
                ("fbr", "few-body (FBR)")]


def fig2_irlm(L=200, U="0.2"):
    runs = {m: load(APP / f"quench_bond_irlm_{m}_L{L}_U{U}.dat") for m, _ in IRLM_METHODS}
    if runs["fbr"] is None:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    for m, label in IRLM_METHODS:
        d = runs[m]
        if d is not None:
            ax.plot(d["t"], d["bond_dim"], color=COLOR[m], lw=2, label=label)
    style(ax, f"IRLM quench, U={U}, V=0.1, L={L}: max bond dimension", ylabel="χmax")
    ax.set_ylim(0, 420)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ins = ax.inset_axes([0.58, 0.45, 0.38, 0.35])
    ins.plot(runs["fbr"]["t"], runs["fbr"]["n_active"], color=COLOR["fbr"], lw=1.5)
    ins.set_title("FBR N_active", fontsize=8)
    ins.tick_params(labelsize=7)
    ins.grid(alpha=0.25, lw=0.5)

    ax = axes[1]
    f = runs["fbr"]
    for m, label in IRLM_METHODS[:2]:
        d = runs[m]
        if d is None:
            continue
        t, ia, ib = np.intersect1d(np.round(d["t"], 6), np.round(f["t"], 6), return_indices=True)
        ax.plot(t, np.maximum(np.abs(d["n0"][ia] - f["n0"][ib]), 1e-12), color=COLOR[m], lw=1.5,
                label=f"|n0({m}) − n0(fbr)|")
    style(ax, "impurity occupation vs the FBR", ylabel="|Δn0|", log=True)
    ax.legend(fontsize=8, frameon=False)
    return [(save(fig, "fig2_irlm_bond.png"),
             f"Paper Fig. 2: IRLM quench, U={U}, L={L}, dt=0.1, truncation 1e-10. The full-MPS "
             "bond dimension grows without bound (chain and star); the FBR saturates, and so "
             "does its active window (inset). Right: the three runs agree on n0(t).")]


def siam_runs():
    """{(method, L, U-string): Data} for every excitation_bond_siam file."""
    out = {}
    for p in APP.glob("excitation_bond_siam_*_L*_U*.dat"):
        m = re.match(r"excitation_bond_siam_([a-z_]+?)_L(\d+)_U([-0-9.]+)\.dat", p.name)
        if m:
            out[(m.group(1), int(m.group(2)), m.group(3))] = load(p)
    return out


def excitation_siam():
    runs = siam_runs()
    Ls = sorted({L for _, L, _ in runs})
    Us = sorted({u for _, _, u in runs}, key=float)
    if not Ls:
        return []
    shade = dict(zip(Us, seq_colors(len(Us), "Oranges")))
    shade_fbr = dict(zip(Us, seq_colors(len(Us), "Blues")))
    fig, axes = plt.subplots(len(Ls), 4, figsize=(21, 4 * len(Ls)), squeeze=False)
    for row, L in enumerate(Ls):
        for u in Us:
            s, f = runs.get(("star", L, u)), runs.get(("fbr", L, u))
            if s is not None:
                axes[row, 0].plot(s["t"], s["bond_dim"], color=shade[u], lw=2, label=f"star U={u}")
            if f is not None:
                axes[row, 0].plot(f["t"], f["bond_dim"], "--", color=shade_fbr[u], lw=2,
                                  label=f"fbr U={u}")
                axes[row, 1].plot(f["t"], f["n_active"], color=shade_fbr[u], lw=2, label=f"U={u}")
            if s is not None and f is not None:
                t, ia, ib = np.intersect1d(np.round(s["t"], 6), np.round(f["t"], 6),
                                           return_indices=True)
                dn = np.abs(s["n0up"][ia] - f["n0up"][ib])
                axes[row, 2].plot(t, np.maximum(dn, 1e-12), color=shade_fbr[u], lw=1.5, label=f"U={u}")
            for d, c, ls in [(s, shade, "-"), (f, shade_fbr, "--")]:
                if d is not None:
                    axes[row, 3].plot(d["t"], np.cumsum(d["wall_s"]), ls, color=c[u], lw=1.5)
        style(axes[row, 0], f"excitation c₀↑†|gs>, L={L}: max bond dimension", ylabel="χmax")
        style(axes[row, 1], f"FBR active window, L={L}", ylabel="N_active")
        style(axes[row, 2], f"|n0↑(star) − n0↑(fbr)|, L={L}", ylabel="|Δn0↑|", log=True)
        style(axes[row, 3], f"wall time so far, L={L} (star solid, fbr dashed)", ylabel="s", log=True)
        axes[row, 0].legend(fontsize=7, frameon=False, ncol=2)
        axes[row, 1].legend(fontsize=7, frameon=False)
    return [(save(fig, "excitation_bond_siam.png"),
             "SIAM Green-function excitation c_0up^dag|gs> (V=0.1, dt=0.1, truncation 1e-10): "
             "full star MPS (solid, orange) vs FBR spin_block (dashed, blue), darker = larger U.")]


def make():
    return fig2_irlm() + excitation_siam()


if __name__ == "__main__":
    for name, caption in make():
        print(f"{name}: {caption}")
