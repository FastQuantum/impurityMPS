"""Cost and accuracy of the frame ALIGNMENT in the separate-frame SIAM Green function.

Data: app/output/green_align_gates_siam_L*_U*.dat, from app/green_align_gates_siam.cpp.
psi0 and c_0^dag psi0 are evolved in SEPARATE frames and aligned only at the
measurement (green_overlap.h). The rotated MPS the alignment builds is a throwaway
feeding one scalar G(t), so it is truncated at mps_cutoff, which the program SWEEPS
(columns chi<j>, alignS<j>, ReG<j>, ImG<j> for each value in the header's mps_cutoffs).

Four panels vs time:
  1. alignment gates -- the two-site Givens gates applied, the O(band^2) a naive
     full-band reduction would cost, and the band width. This is the residual cost
     once the throwaway MPS is cheap: ~8600 gates through a small chi.
  2. bond dimension -- chi_align (the throwaway MPS) for each mps_cutoff against the
     STATE bond dims. The states saturate; a tight mps_cutoff lets chi_align run
     away, a loose one keeps it bounded.
  3. wall time per step -- alignment at each mps_cutoff vs one TDVP sweep. Loose
     truncation brings the alignment back down near the evolution cost.
  4. accuracy -- |G(mps_cutoff) - G(finest)| vs time, with the 3-digit line. Shows
     which cutoff is loose enough to be cheap yet still good to ~3 digits.

Run:  python3 app/plot/green_align_gates.py
"""
import matplotlib.pyplot as plt
import numpy as np

from common import COLOR, save, style, by_length, seq_colors


def _cutoffs(d):
    """The swept mps_cutoff values, from the header's 'mps_cutoffs a,b,c'."""
    for h in d.header:
        if "mps_cutoffs" in h:
            tok = h.split("mps_cutoffs", 1)[1].split()[0]
            return [float(x) for x in tok.split(",")]
    return []


def _p_gates(ax, d, L, U):
    t = d["t"]
    naive = d["band"] * (d["band"] - 1) / 2
    ax.plot(t, naive, color="#b8b7b3", lw=1.4, ls="-.",
            label="naive full-band  band(band-1)/2")
    ax.plot(t, d["gates"], color=COLOR["fbr"], lw=1.8, label="gates applied")
    style(ax, title=f"Alignment circuit  (SIAM L={L:g}, U={U:g})",
          ylabel="number of Givens gates")
    aband = ax.twinx()
    aband.plot(t, d["band"], color=COLOR["star"], lw=1.1, ls="--")
    aband.set_ylabel("band width  (b-a)", color=COLOR["star"])
    aband.tick_params(axis="y", labelcolor=COLOR["star"])
    aband.set_ylim(0, L * 1.05)
    aband.spines["top"].set_visible(False)
    ax.legend(fontsize=8, loc="center right", framealpha=0.9)


def _p_bond(ax, d, cuts, shades):
    t = d["t"]
    for j, c in enumerate(cuts):
        lbl = fr"$\chi_{{align}}$  mps_cut={c:g}" if len(cuts) > 1 \
            else fr"$\chi_{{align}}$ (rotated throwaway MPS)"
        ax.plot(t, d[f"chi{j}"], color=shades[j], lw=1.8, label=lbl)
    ax.plot(t, d["bondB"], color=COLOR["fbr"], lw=1.4, ls=":", label="bond dim B (state)")
    ax.plot(t, d["bondA"], color=COLOR["shared"], lw=1.4, ls=":", label="bond dim A (state)")
    style(ax, title="Bond dimension: rotated throwaway MPS vs states",
          ylabel="max bond dimension")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)


def _p_time(ax, d, cuts, shades):
    t = d["t"]
    ev = t[:-1] if len(t) > 1 else t
    ax.plot(ev, d["evolveB_s"][:len(ev)], color="#888", lw=1.6,
            label="TDVP sweep (state B)")
    for j, c in enumerate(cuts):
        lbl = f"alignment  mps_cut={c:g}" if len(cuts) > 1 else "alignment + overlap"
        ax.plot(t, d[f"alignS{j}"], color=shades[j], lw=1.6, label=lbl)
    style(ax, title="Wall time per step", ylabel="seconds")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)


def _p_accuracy(ax, d, cuts, shades, jref):
    t = d["t"]
    gref = d[f"ReG{jref}"] + 1j * d[f"ImG{jref}"]
    for j, c in enumerate(cuts):
        if j == jref:
            continue
        g = d[f"ReG{j}"] + 1j * d[f"ImG{j}"]
        ax.plot(t, np.abs(g - gref), color=shades[j], lw=1.6, label=f"mps_cut={c:g}")
    ax.axhline(1e-3, color="#a02020", lw=1.0, ls="--", label="3-digit level (1e-3)")
    style(ax, title=f"G accuracy vs finest cutoff (mps_cut={cuts[jref]:g})",
          ylabel=r"$|G - G_{finest}|$", log=True)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)


def _panels(d, L, U):
    cuts = _cutoffs(d)
    nc = len(cuts)
    shades = seq_colors(max(nc, 2), "viridis") if nc > 1 else [COLOR["star"]]
    jref = int(np.argmin(cuts)) if nc else 0

    if nc >= 2:
        # swept ("after tuning"): cost AND accuracy across cutoffs
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        _p_gates(ax[0, 0], d, L, U)
        _p_bond(ax[0, 1], d, cuts, shades)
        _p_time(ax[1, 0], d, cuts, shades)
        _p_accuracy(ax[1, 1], d, cuts, shades, jref)
    else:
        # single cutoff ("before tuning"): the diagnostic that motivates the sweep
        fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
        _p_gates(ax[0], d, L, U)
        _p_bond(ax[1], d, cuts, shades)
        _p_time(ax[2], d, cuts, shades)

    return fig, cuts, jref


def make():
    figs = []
    runs = by_length("green_align_gates_siam_L*_U0.1.dat")
    for L, d in runs.items():
        if d is None or len(d) == 0 or "gates" not in d:
            continue
        U = d.param("U", 0.1)
        fig, cuts, jref = _panels(d, L, U)
        name = save(fig, f"green_align_gates_siam_L{L}.png")
        gmax = int(d["gates"].max())
        if len(cuts) >= 2:
            jloose = int(np.argmax(cuts))
            gL = d[f"ReG{jloose}"][-1] + 1j * d[f"ImG{jloose}"][-1]
            gF = d[f"ReG{jref}"][-1] + 1j * d[f"ImG{jref}"][-1]
            cap = (
                f"Separate-frame SIAM Green function, L={L}, U={U:g}, up to t={d['t'][-1]:g}. "
                f"The rotated throwaway MPS is truncated at mps_cutoff (swept "
                f"{', '.join(f'{c:g}' for c in cuts)}). The STATE bond dims saturate, and a "
                f"loose mps_cutoff keeps chi_align -- and so the alignment cost -- bounded too, "
                f"while G stays good to ~3 digits (mps_cut={cuts[jref]:g} vs {cuts[jloose]:g} "
                f"differ by {abs(gL - gF):.1e} at the end). The residual alignment cost is just "
                f"applying the ~{gmax} gates (band saturates at L) through a small chi.")
        else:
            amax = float(d["alignS0"].max())
            cap = (
                f"Separate-frame SIAM Green function, L={L}, U={U:g}, up to t={d['t'][-1]:g}, "
                f"overlap truncated at mps_cutoff={cuts[0]:g}. The state bond dims and the "
                f"FINAL chi_align both saturate (~{int(d['chi0'].max())}), yet the alignment "
                f"wall time hits a cost wall at intermediate t (up to {amax:.0f} s/step here): "
                f"the staircase circuit passes through highly-entangled INTERMEDIATE MPS "
                f"configurations that the small final bond hides. Loosening mps_cutoff (it sets "
                f"the final truncation) secures G accuracy and the early-time cost but does not "
                f"remove this wall.")
        figs.append((name, cap))
    if not figs:
        figs.append((None, "No green_align_gates_siam_L*_U0.1.dat found in app/output/."))
    return figs


if __name__ == "__main__":
    for name, cap in make():
        print(name, "--", cap)
