"""SIAM impurity Green function G00: the few-body FBR against the full-length star
and the real-space chain baselines, for several L.

Data: app/output/fbr_green_siam_L<L>_U<U>.dat           (app/fbr_green_siam.cpp)
      test/ref/output/star_green_siam_L<L>_U<U>.dat     (test/ref/star_green_siam.cpp)
      test/ref/output/chain_green_siam_L<L>_U<U>.dat    (test/ref/chain_green_siam.cpp)
Run:  python3 app/plot/green_siam.py     (figures in app/plot/figures/)
"""
import matplotlib.pyplot as plt
import numpy as np

from common import APP, COLOR, REF, by_length, load, max_dev, save, seq_colors, style

US = ["0.1", "0.025"]
G = [("ReG00", "ImG00")]


def data(u):
    return {"fbr": by_length(f"fbr_green_siam_L*_U{u}.dat", APP),
            "star": by_length(f"star_green_siam_L*_U{u}.dat", REF),
            "chain": by_length(f"chain_green_siam_L*_U{u}.dat", REF)}


def make():
    d = {u: data(u) for u in US}
    figs = []

    # G00 at L=100, where all three methods exist.
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for col, u in enumerate(US):
        for row, name in enumerate(["ReG00", "ImG00"]):
            ax = axes[row, col]
            for key, label, lw, ls in [("chain", "chain (baseline)", 2.4, "-"),
                                       ("star", "star (whole chain active)", 1.8, "--"),
                                       ("fbr", "FBR (active window)", 1.8, ":")]:
                s = d[u][key].get(100)
                if s is not None:
                    ax.plot(s["t"], s[name], ls, color=COLOR[key], lw=lw, label=label)
            style(ax, f"{name}, L=100, U={u}", xlabel="t" if row else "")
    axes[0, 0].legend(fontsize=8, frameon=False)
    figs.append((save(fig, "green_siam_G00_L100.png"),
                 "SIAM G00(t) at L=100. The chain baseline exists for U=0.1 only."))

    # Error: FBR against the star at every L, and both against the chain at L=100.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    notes = []
    for ax, u in zip(axes, US):
        fbr, star, chain = d[u]["fbr"], d[u]["star"], d[u]["chain"]
        Ls = [L for L in fbr if L in star]
        for L, c in zip(Ls, seq_colors(len(Ls))):
            t, dev = max_dev(fbr[L], star[L], G)
            ax.plot(t, dev, color=c, lw=1.8, label=f"|FBR - star|, L={L}")
            notes.append(f"U={u} L={L}: FBR-star {dev[-1]:.1e} to t={t[-1]:g}")
        for L in chain:
            for key, ls in [("fbr", ":"), ("star", "--")]:
                if L in d[u][key]:
                    t, dev = max_dev(d[u][key][L], chain[L], G)
                    ax.plot(t, dev, ls, color=COLOR[key], lw=1.8, label=f"|{key.upper()} - chain|, L={L}")
                    notes.append(f"U={u} L={L}: {key}-chain {dev[-1]:.1e}")
        style(ax, f"max |dG00| up to t, U={u}", ylabel="error" if u == US[0] else None, log=True)
        ax.legend(fontsize=7, frameon=False)
    figs.append((save(fig, "green_siam_error.png"),
                 "Running max of the G00 difference. " + "; ".join(notes) + "."))

    # Cost: FBR window, bond dimensions and wall time, per L.
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for row, u in enumerate(US):
        fbr, star = d[u]["fbr"], d[u]["star"]
        for L, c in zip(fbr, seq_colors(len(fbr), "Blues")):
            s = fbr[L]
            axes[row, 0].plot(s["t"], s["n_active"], color=c, lw=1.8, label=f"FBR, L={L}")
            axes[row, 1].plot(s["t"], s["bond_dim"], color=c, lw=1.8, label=f"FBR, L={L}")
            axes[row, 2].plot(s["t"], np.cumsum(s["wall_s"]), color=c, lw=1.8, label=f"FBR, L={L}")
        for L, c in zip(star, seq_colors(len(star), "Oranges")):
            s = star[L]
            axes[row, 1].plot(s["t"], s["bond_dim"], "--", color=c, lw=1.8, label=f"star, L={L}")
            axes[row, 2].plot(s["t"], np.cumsum(s["wall_s"]), "--", color=c, lw=1.8, label=f"star, L={L}")
        style(axes[row, 0], f"FBR active window, U={u}", ylabel="n_active")
        style(axes[row, 1], f"max bond dimension, U={u}", ylabel="bond dim", log=True)
        style(axes[row, 2], f"wall time so far, U={u}", ylabel="s", log=True)
        axes[row, 0].legend(fontsize=7, frameon=False)
        axes[row, 2].legend(fontsize=7, frameon=False, ncol=2)
    figs.append((save(fig, "green_siam_cost.png"),
                 "Cost per L: the FBR keeps a window of a few tens of orbitals while the star "
                 "keeps all L. Runs end at t=L/2 or when a bond dimension reaches 1024."))
    return figs


if __name__ == "__main__":
    for name, caption in make():
        print(f"{name}: {caption}")
