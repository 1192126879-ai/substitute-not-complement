# -*- coding: utf-8 -*-
"""Summarize a directory of gara_diag JSONL records."""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

MODES = ["input_local", "input_full", "weight_local", "weight_full",
         "both_local", "both_full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="SUMMARY.md")
    args = ap.parse_args()
    rows = defaultdict(list)
    for f in glob.glob(os.path.join(args.dir, "*.jsonl")):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            rows[(d["dataset"], d["mode"])].append(d)
    lines = [f"# {args.dir}", "", "| dataset | mode | test | aux | peak GB | sec | n |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for (ds, m) in sorted(rows):
        v = rows[(ds, m)]
        t = np.array([x["test_acc"] for x in v])
        a = np.array([x.get("aux_val", 0) for x in v])
        mem = np.array([x.get("peak_mem_gb", 0) for x in v])
        sec = np.array([x.get("seconds", 0) for x in v])
        lines.append(f"| {ds} | {m} | {t.mean():.4f}±{t.std():.4f} | "
                     f"{a.mean():.4f} | {mem.mean():.1f} | {sec.mean():.0f} | {len(v)} |")
    out = "\n".join(lines)
    print(out)
    open(args.out, "w", encoding="utf-8").write(out)


if __name__ == "__main__":
    main()
