"""IRLM Green function, L=100: separate frames vs one shared frame, both against
the chain baseline.

Data: app/output/fbr_green_{sep,shared}_irlm_L100_U<U>.dat
      test/ref/output/chain_green_irlm_U<U>_ref.txt
Run:  python3 app/plot/green_irlm.py      (figures in app/plot/figures/)
"""
import matplotlib.pyplot as plt

from common import APP, COLOR, REF, load, max_dev, save, style

US = ["0.1", "0.2"]
METHODS = [("sep", "separate frames", COLOR["fbr"], "--"),
           ("shared", "shared frame", COLOR["shared"], ":")]
G_COLS = [("ReG00", "ImG00"), ("ReG01", "ImG01")]


def data():
    out = {}
    for u in US:
        out[u] = {"chain": load(REF / f"chain_green_irlm_U{u}_ref.txt")}
        for key, *_ in METHODS:
            out[u][key] = load(APP / f"fbr_green_{key}_irlm_L100_U{u}.dat")
    return out


def make():
    d = data()
    figs = []

    # G00 itself: the chain line underneath, the two FBR runs on top.
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex="col")
    for col, u in enumerate(US):
        chain = d[u]["chain"]
        for row, name in enumerate(["ReG00", "ImG00"]):
            ax = axes[row, col]
            ax.plot(chain["t"], chain[name], color=COLOR["chain"], lw=2.2, label="chain (baseline)")
            for key, label, color, ls in METHODS:
                if d[u][key] is not None:
                    ax.plot(d[u][key]["t"], d[u][key][name], ls, color=color, lw=1.8, label=label)
            style(ax, f"{name}, U={u}", xlabel="t" if row else "")
    axes[0, 0].legend(fontsize=8, frameon=False)
    figs.append((save(fig, "green_irlm_G00.png"),
                 "IRLM G00(t), L=100: separate and shared frames over the chain baseline."))

    # Error against the chain, running max over G00 and G01.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    lines = []
    for ax, u in zip(axes, US):
        for key, label, color, ls in METHODS:
            if d[u][key] is None:
                continue
            t, dev = max_dev(d[u][key], d[u]["chain"], G_COLS)
            ax.plot(t, dev, color=color, lw=2, label=label)
            lines.append(f"{label} U={u}: {dev[-1]:.1e} by t={t[-1]:g}")
        style(ax, f"max |G - G_chain| up to t, U={u}", ylabel="error" if u == US[0] else None, log=True)
    axes[0].legend(fontsize=8, frameon=False)
    figs.append((save(fig, "green_irlm_error.png"),
                 "Running max of |G00-G00_chain|, |G01-G01_chain|. " + "; ".join(lines) + "."))

    # Cost: active window and bond dimension.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, label, color, _ in METHODS:
        for u, ls in zip(US, ["-", "--"]):
            s = d[u][key]
            if s is None:
                continue
            axes[0].plot(s["t"], s["n_active"], ls, color=color, lw=1.8, label=f"{label}, U={u}")
            axes[1].plot(s["t"], s["maxBondDim"], ls, color=color, lw=1.8, label=f"{label}, U={u}")
    for u, ls in zip(US, ["-", "--"]):
        c = d[u]["chain"]
        axes[1].plot(c["t"], c["m"], ls, color=COLOR["chain"], lw=1.4, label=f"chain, U={u}")
    style(axes[0], "active window", ylabel="n_active")
    style(axes[1], "max bond dimension", ylabel="bond dim", log=True)
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].legend(fontsize=8, frameon=False)
    figs.append((save(fig, "green_irlm_cost.png"),
                 "Cost of the IRLM runs: the FBR window (of L=100 orbitals) and the bond dimension, "
                 "with the full real-space chain for scale."))
    return figs


if __name__ == "__main__":
    for name, caption in make():
        print(f"{name}: {caption}")
