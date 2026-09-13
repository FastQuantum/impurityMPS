"""Shared helpers for the app/plot scripts: where the data lives, one loader for
every .dat/.txt format in the repo, and one look for the figures.

Every data file is whitespace-separated numbers, with comment/header lines
before them. The column names come from the last header line that starts with
"# t", so a script reads columns by name:  d = load(...); d["ReG00"].
"""
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
REF = REPO / "test" / "ref" / "output"   # baselines (chain, star, fbr replays)
APP = REPO / "app" / "output"            # experiment data
FIG = HERE / "figures"                   # generated, not versioned

# One colour per method, the same in every figure (colour follows the entity).
# Slots 1-3 of the reference categorical palette: they stay distinguishable to
# colour-blind readers even when all three are on screen. The chain baseline is
# the ground truth, drawn as a neutral dark line underneath the rest.
COLOR = {
    "chain":  "#2b2b2a",
    "fbr":    "#2a78d6",   # few-body; separate frames for the IRLM runs
    "star":   "#eb6834",
    "shared": "#1baf7a",
}


def seq_colors(n, cmap="Blues"):
    """n shades of one hue, light to dark, for an ordered parameter such as L."""
    return [plt.get_cmap(cmap)(x) for x in np.linspace(0.45, 0.95, n)]


class Data(dict):
    """Columns by name (numpy arrays) plus the file's header lines."""

    def __init__(self, path, header, cols):
        super().__init__(cols)
        self.path = pathlib.Path(path)
        self.header = header

    def __len__(self):
        return len(self["t"]) if "t" in self else 0

    def param(self, key, default=None):
        """A 'key=value' or 'key value' parameter from the header lines."""
        text = " ".join(self.header)
        m = re.search(rf"(?:^|[\s#]){re.escape(key)}(?:=|\s+)([-+0-9.eE]+)", text)
        return float(m.group(1)) if m else default


def load(path):
    """Read a data file; returns None if it does not exist."""
    path = pathlib.Path(path)
    if not path.exists():
        return None
    header, rows, names = [], [], None
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            header.append(line)
            if parts[0] == "#" and len(parts) > 1 and parts[1] == "t":
                names = parts[1:]
    ncol = min(len(r) for r in rows) if rows else 0
    data = np.array([r[:ncol] for r in rows]).reshape(-1, ncol)
    names = (names or [])[:ncol]
    names += [f"c{i}" for i in range(len(names), ncol)]
    return Data(path, header, {n: data[:, i] for i, n in enumerate(names)})


def length_of(path):
    """L from a file name like ..._L200_U0.1.dat."""
    m = re.search(r"_L(\d+)_", pathlib.Path(path).name)
    return int(m.group(1)) if m else None


def by_length(pattern, folder=APP):
    """{L: Data} for every file matching the glob pattern, sorted by L."""
    files = sorted(folder.glob(pattern), key=length_of)
    return {length_of(f): load(f) for f in files}


def style(ax, title=None, xlabel="t", ylabel=None, log=False):
    ax.set_title(title or "", fontsize=10, loc="left")
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.25, lw=0.6, which="major")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    """Save into app/plot/figures/ and return the file name (for the report)."""
    FIG.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)
    return name


def max_dev(a, b, cols):
    """Running max over t of |a-b| on the common time grid of two Data."""
    ta = np.round(a["t"], 6)
    tb = np.round(b["t"], 6)
    common, ia, ib = np.intersect1d(ta, tb, return_indices=True)
    d = np.zeros(len(common))
    for re_, im_ in cols:
        za = a[re_][ia] + 1j * a[im_][ia]
        zb = b[re_][ib] + 1j * b[im_][ib]
        d = np.maximum(d, np.abs(za - zb))
    return common, np.maximum.accumulate(d)
