# -*- coding: utf-8 -*-
"""2x2 factorial analysis: input hop x weight hop.

Cells (Vicuna):
  input_local, input_full, both_local, both_full, mix_lw, mix_fl
Definitions per dataset:
  gap_in   = input_full - input_local            (how much info the input misses)
  sub_gain = mix_lw - input_local                (weight_full on a local input)
  recovery = sub_gain / gap_in                   (how much of the gap the adapter
                                                  restores; ~0 = useless, ~1 = full
                                                  substitute)
  red_full = mix_fl - input_full                 (weight_local on a full input)
  red_same = both_full - input_full              (same-hop control)
  red_loc  = both_local - input_local
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.environ.get("GARA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIRS = ["gara_diag", "gara_mix"]


def load():
    rows = defaultdict(list)
    for d in DIRS:
        for f in glob.glob(os.path.join(ROOT, d, "*.jsonl")):
            for line in open(f, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                rows[(r["dataset"], r["mode"])].append(r["test_acc"])
    return rows


def m(rows, ds, mode):
    v = rows.get((ds, mode), [])
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (np.nan, np.nan, 0)


def main():
    rows = load()
    datasets = sorted({k[0] for k in rows})
    out = ["# 2x2 factorial (Vicuna-7B)", "",
           "| dataset | IL | IF | mix_lw | mix_fl | BL | BF | gap_in | sub_gain | recovery | red_full |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    stats = []
    for ds in datasets:
        il, ils, n = m(rows, ds, "input_local")
        inf, _, _ = m(rows, ds, "input_full")
        lw, _, _ = m(rows, ds, "mix_lw")
        fl, _, _ = m(rows, ds, "mix_fl")
        bl, _, _ = m(rows, ds, "both_local")
        bf, _, _ = m(rows, ds, "both_full")
        gap = inf - il
        sub = lw - il
        rec = sub / gap if abs(gap) > 1e-6 else float("nan")
        red = fl - inf
        stats.append((ds, gap, sub, rec, red, bl - il, bf - inf))
        out.append(f"| {ds} | {il:.3f} | {inf:.3f} | {lw:.3f} | {fl:.3f} | "
                   f"{bl:.3f} | {bf:.3f} | {gap:+.3f} | {sub:+.3f} | "
                   f"{rec:+.2f} | {red:+.3f} |")
    out += ["", "Spearman(gap_in, sub_gain) = "
            f"{spearman([s[1] for s in stats], [s[2] for s in stats]):+.3f}"
            " (n=%d)" % len(stats), "",
            "## per-seed Roman", ""]
    for mode in ["input_local", "input_full", "mix_lw", "mix_fl",
                 "both_local", "both_full"]:
        v = rows.get(("roman_empire", mode), [])
        if v:
            out.append(f"{mode}: {[round(x, 4) for x in v]}")
    text = "\n".join(out)
    print(text)
    open(os.path.join(ROOT, "GARA_2X2.md"), "w", encoding="utf-8").write(text)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return float("nan")
    x, y = x[ok], y[ok]
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / (np.sqrt((rx ** 2).sum() * (ry ** 2).sum()) + 1e-12))


if __name__ == "__main__":
    main()
