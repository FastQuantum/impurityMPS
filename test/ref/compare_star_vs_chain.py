#!/usr/bin/env python3
"""Compare the star and chain reference baselines and write a static report.

The unit tests validate the FBR solvers against the *chain* baseline only (the
MPS-friendly gold standard). This script keeps a record of how the naive star
baseline (star_dyn_siam_center) tracks the chain one, per snapshot, so the
agreement is documented without wiring the star files into the test suite.

Run from test/ref/ :  python3 compare_star_vs_chain.py
Writes: output/star_vs_chain.txt
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
LABELS = ["initial", "t=0.1", "t=5.0", "t=10.0", "t=20.0"]


def load(path):
    toks = open(path).read().split()
    i = 0
    assert toks[i] == "chain_dyn_siam_center_ref_v1"; i += 1
    assert toks[i] == "L"; i += 1
    L = int(toks[i]); i += 1
    assert toks[i] == "snapshots"; i += 1
    n = int(toks[i]); i += 1
    snaps = {}
    for _ in range(n):
        assert toks[i] == "snapshot"; i += 1
        label = toks[i]; i += 1
        assert toks[i] == "ni"; i += 1
        ni = [float(toks[i + k]) for k in range(L)]; i += L
        assert toks[i] == "cc"; i += 1
        cc = [float(toks[i + k]) for k in range(2 * L * L)]; i += 2 * L * L
        snaps[label] = (ni, cc)
    return snaps


def main():
    lines = [
        "star vs chain reference comparison",
        "==================================",
        "max_i |ni_star - ni_chain|  and  max_ij |cc_star - cc_chain|,  per snapshot.",
        "",
    ]
    for us in ["0.2", "0.1"]:
        chain = load(os.path.join(OUT, f"chain_dyn_siam_center_U{us}_ref.txt"))
        star = load(os.path.join(OUT, f"star_dyn_siam_center_U{us}_ref.txt"))
        lines.append(f"U={us}")
        lines.append(f"  {'snapshot':10s}  {'max|dni|':>10s}  {'max|dcc|':>10s}")
        for lab in LABELS:
            if lab in chain and lab in star:
                dni = max(abs(a - b) for a, b in zip(chain[lab][0], star[lab][0]))
                dcc = max(abs(a - b) for a, b in zip(chain[lab][1], star[lab][1]))
                lines.append(f"  {lab:10s}  {dni:10.2e}  {dcc:10.2e}")
        lines.append("")
    report = "\n".join(lines)
    with open(os.path.join(OUT, "star_vs_chain.txt"), "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
