"""What one FBR evolution costs, by protocol (SIAM): the quench of the paper and the
ground state itself (spin_symmetric), and the Green-function excitation c_0^dag|gs>
(spin_block, since it is spin-polarized);
plus the co-moving-frame negative result (Fbr_dyn_frame against Fbr_dyn on |gs>).

Data: app/output/{quench,excitation,gs}_siam_L<L>_U0.025.dat  (app/quench_vs_excitation_siam.cpp)
      app/output/gs_{ip,frame}_siam_L<L>_U0.dat               (app/gs_frame_vs_ip_siam.cpp)
Run:  python3 app/plot/quench_vs_excitation.py     (figures in app/plot/figures/)
"""
import matplotlib.pyplot as plt
import numpy as np

from common import APP, COLOR, by_length, save, style

# three protocols -> the first three categorical slots, fixed order
PROTOCOL = [("quench", COLOR["fbr"]), ("excitation", COLOR["star"]), ("gs", COLOR["shared"])]


def make():
    figs = []

    runs = {p: by_length(f"{p}_siam_L*_U0.025.dat") for p, _ in PROTOCOL}
    Ls = sorted({L for r in runs.values() for L in r})
    if Ls:
        fig, axes = plt.subplots(len(Ls), 3, figsize=(15, 3.8 * len(Ls)), squeeze=False)
        for row, L in enumerate(Ls):
            for p, color in PROTOCOL:
                d = runs[p].get(L)
                if d is None:
                    continue
                axes[row, 0].plot(d["t"], d["n_active"], color=color, lw=1.8, label=p)
                axes[row, 1].plot(d["t"], d["bond_dim"], color=color, lw=1.8, label=p)
                axes[row, 2].plot(d["t"], np.cumsum(d["wall_s"]), color=color, lw=1.8, label=p)
            style(axes[row, 0], f"active window, L={L}", ylabel="n_active")
            style(axes[row, 1], f"max bond dimension, L={L}", ylabel="bond dim")
            style(axes[row, 2], f"wall time so far, L={L}", ylabel="s", log=True)
            axes[row, 0].legend(fontsize=8, frameon=False)
        figs.append((save(fig, "quench_vs_excitation_siam.png"),
                     "One Fbr_dyn evolution per protocol, SIAM U=0.025: the paper's quench, the "
                     "excitation c_0^dag|gs> a Green function needs, and |gs> itself."))

    frame = {name: by_length(f"gs_{name}_siam_L*_U0.dat") for name in ["ip", "frame"]}
    Ls = sorted(set(frame["ip"]) | set(frame["frame"]))
    if Ls:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for name, color, label in [("ip", COLOR["fbr"], "interaction picture (Fbr_dyn)"),
                                   ("frame", COLOR["star"], "co-moving frame (Fbr_dyn_frame)")]:
            for L, ls in zip(Ls, ["-", "--", ":"]):
                d = frame[name].get(L)
                if d is None:
                    continue
                axes[0].plot(d["t"], d["n_active"], ls, color=color, lw=1.8, label=f"{label}, L={L}")
                axes[1].plot(d["t"], np.maximum(d["drift"], 1e-16), ls, color=color, lw=1.8)
        style(axes[0], "active window on |gs>, U=0", ylabel="n_active")
        style(axes[1], "max drift of the impurity block of cc", ylabel="drift", log=True)
        axes[0].legend(fontsize=7, frameon=False)
        figs.append((save(fig, "gs_frame_vs_ip_siam.png"),
                     "Negative result: rotating the window with the bath propagator grows the window "
                     "and corrupts the eigenstate (see include/fbr/fbr_dyn_frame.h)."))
    return figs


if __name__ == "__main__":
    for name, caption in make():
        print(f"{name}: {caption}")
