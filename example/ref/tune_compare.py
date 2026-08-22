#!/usr/bin/env python3
"""Compare a tuning star run against the committed chain reference.

Usage: tune_compare.py <star_file> <U>
Prints per-snapshot max|dni| and max|dcc| for every snapshot present in both.
"""
import os, sys

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
    star_file = sys.argv[1]
    us = sys.argv[2]
    chain = load(os.path.join(OUT, f"chain_dyn_siam_center_U{us}_ref.txt"))
    star = load(star_file)
    print(f"  {'snapshot':10s}  {'max|dni|':>10s}  {'max|dcc|':>10s}")
    for lab in LABELS:
        if lab in chain and lab in star:
            dni = max(abs(a - b) for a, b in zip(chain[lab][0], star[lab][0]))
            dcc = max(abs(a - b) for a, b in zip(chain[lab][1], star[lab][1]))
            print(f"  {lab:10s}  {dni:10.2e}  {dcc:10.2e}")


if __name__ == "__main__":
    main()
