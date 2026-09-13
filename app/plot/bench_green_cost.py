"""Cost of the few-body separate-frame Green function against the full star
(the whole chain active), IRLM and SIAM.

Data: app/output/bench_green_cost_L<L>_U0.2.dat         (app/bench_green_cost.cpp)
      app/output/bench_green_cost_siam_L<L>_U0.025.dat  (app/bench_green_cost_siam.cpp)
Wall times are machine-dependent: compare methods within one file, not across
machines.
Run:  python3 app/plot/bench_green_cost.py     (figures in app/plot/figures/)
"""
import matplotlib.pyplot as plt
import numpy as np

from common import COLOR, by_length, save, style


def make():
    figs = []

    # IRLM: median time per sweep against L, and the speedup.
    irlm = {L: d for L, d in by_length("bench_green_cost_L*_U0.2.dat").items() if len(d) > 3}
    if irlm:
        Ls = np.array(list(irlm))
        fb = np.array([np.median(d["fewbody_sweep_s"][2:]) for d in irlm.values()])
        st = np.array([np.median(d["star_sweep_s"][2:]) for d in irlm.values()])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(Ls, fb, "o-", color=COLOR["fbr"], lw=2, ms=8, label="few-body, separate frames")
        axes[0].plot(Ls, st, "s-", color=COLOR["star"], lw=2, ms=8, label="star, n_active = L")
        axes[0].set_xscale("log")
        style(axes[0], "IRLM U=0.2: median wall time per sweep (3 states)", xlabel="L", ylabel="s", log=True)
        axes[0].legend(fontsize=8, frameon=False)
        axes[1].plot(Ls, st / fb, "o-", color=COLOR["chain"], lw=2, ms=8)
        axes[1].set_xscale("log")
        style(axes[1], "speedup, star / few-body", xlabel="L", ylabel="x")
        rows = ", ".join(f"L={L}: {s / f:.1f}x" for L, f, s in zip(Ls, fb, st))
        figs.append((save(fig, "bench_green_cost_irlm.png"),
                     f"IRLM Green function, time per sweep against L. Speedup {rows}."))

    # SIAM: time per step along the run, split into evolution and measurement.
    siam = {L: d for L, d in by_length("bench_green_cost_siam_L*_U0.025.dat").items()
            if "fb_measure_s" in d and len(d) > 1}
    if siam:
        fig, axes = plt.subplots(1, len(siam), figsize=(5 * len(siam), 4.2), sharey=True, squeeze=False)
        notes = []
        for ax, (L, d) in zip(axes[0], siam.items()):
            t = d["t"]
            ax.plot(t, d["star_evolve_s"] + d["star_measure_s"], color=COLOR["star"], lw=2, label="star, total")
            ax.plot(t, d["fb_evolve_s"] + d["fb_measure_s"], color=COLOR["fbr"], lw=2, label="few-body, total")
            ax.plot(t, d["fb_measure_s"], ":", color=COLOR["fbr"], lw=2, label="few-body, overlap only")
            style(ax, f"SIAM U=0.025, L={L} (to t={t[-1]:g})", ylabel="s / step" if L == min(siam) else None, log=True)
            tot_fb = np.sum(d["fb_evolve_s"] + d["fb_measure_s"])
            tot_st = np.sum(d["star_evolve_s"] + d["star_measure_s"])
            notes.append(f"L={L}: {tot_st / tot_fb:.1f}x, max|dG|={np.max(d['|dG|']):.1e}")
        axes[0, 0].legend(fontsize=8, frameon=False)
        figs.append((save(fig, "bench_green_cost_siam.png"),
                     "SIAM Green function, wall time per step. Whole-run speedup and the largest "
                     "few-body vs star difference: " + "; ".join(notes) + "."))
    return figs


if __name__ == "__main__":
    for name, caption in make():
        print(f"{name}: {caption}")
