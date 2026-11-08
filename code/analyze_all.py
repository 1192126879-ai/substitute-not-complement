# -*- coding: utf-8 -*-
"""Consolidated analysis for the substitution paper.

Merges every run directory, keeps configs separate, and reports:
  - per-dataset 2x2 cells (input hop x weight hop)
  - recovery ratio and same-hop redundancy
  - causal control (shuffled z)
  - capacity sweeps (rank, K, tokens, depth, layer)
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.environ.get("GARA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAIN = {"gara_diag": None, "gara_mix": None, "gara_qwen": None, "gara_qmix": None,
        "gara_ctrl": None, "gara_sweep": None, "gara_cora": None}

CELL = ["input_local", "input_full", "weight_local", "weight_full",
        "both_local", "both_full", "mix_lw", "mix_fl"]


def load():
    recs = []
    for d in MAIN:
        for f in glob.glob(os.path.join(ROOT, d, "*.jsonl")):
            for line in open(f, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    r["_dir"] = d
                    r["_file"] = os.path.basename(f)
                    recs.append(r)
    return recs


def key(r):
    c = r["config"]
    return (c.get("n_tokens"), c.get("full_layers"), c.get("layer"),
            c.get("K"), c.get("rank"), bool(c.get("shuffle_z")))


def main():
    recs = load()
    base = key({"config": {"n_tokens": 8, "full_layers": 4, "layer": 20,
                           "K": 8, "rank": 16, "shuffle_z": False}})
    cell = defaultdict(list)
    other = defaultdict(list)
    for r in recs:
        if r["mode"] not in CELL:
            continue
        k = (r["dataset"], r["mode"])
        if key(r) == base and not r["config"].get("shuffle_z"):
            cell[k].append(r["test_acc"])
        else:
            other[(r["dataset"], r["mode"],) + key(r)].append(r["test_acc"])

    out = ["# Consolidated results", ""]
    dsets = sorted({k[0] for k in cell})
    out += ["| dataset | IL | IF | BL | BF | WL | WF | mix_lw | mix_fl | gap_in | recovery | n_s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def m(ds, mode):
        v = cell.get((ds, mode), [])
        return float(np.mean(v)) if v else float("nan"), len(v)

    roman_stats = []
    for ds in dsets:
        il, n = m(ds, "input_local")
        inf, _ = m(ds, "input_full")
        bl, _ = m(ds, "both_local")
        bf, _ = m(ds, "both_full")
        wl, _ = m(ds, "weight_local")
        wf, _ = m(ds, "weight_full")
        lw, _ = m(ds, "mix_lw")
        fl, _ = m(ds, "mix_fl")
        gap = inf - il
        rec = (lw - il) / gap if abs(gap) > 1e-9 and not np.isnan(lw) else float("nan")
        out.append(f"| {ds} | {il:.3f} | {inf:.3f} | {bl:.3f} | {bf:.3f} | "
                   f"{wl:.3f} | {wf:.3f} | {lw:.3f} | {fl:.3f} | {gap:+.3f} | "
                   f"{rec:+.2f} | {n} |")
        if ds in ("roman_empire", "amazon_ratings"):
            roman_stats.append((ds, gap, lw - il, rec, fl - inf, bl - il, bf - inf))

    out += ["", "## Controls and sweeps (Roman)", "",
            "| condition | test | n |", "|---|---:|---:|"]
    ctrl = defaultdict(list)
    for (ds, mode, nt, fl, layer, K, rank, shuf), v in other.items():
        if ds != "roman_empire":
            continue
        tag = []
        if shuf:
            tag.append("shuffled-z")
        if nt != 8:
            tag.append(f"tokens={nt}")
        if fl != 4:
            tag.append(f"full_layers={fl}")
        if layer != 20:
            tag.append(f"layer={layer}")
        if K != 8:
            tag.append(f"K={K}")
        if rank != 16:
            tag.append(f"rank={rank}")
        if not tag:
            continue
        ctrl[(mode, ", ".join(tag))].append(np.mean(v))
    for (mode, tag) in sorted(ctrl):
        v = ctrl[(mode, tag)]
        out.append(f"| {mode} | {tag} | {np.mean(v):.4f} | {len(v)} |")

    out += ["", "## Shuffled-z control", ""]
    for (ds, mode, nt, fl, layer, K, rank, shuf), v in sorted(other.items()):
        if shuf:
            out.append(f"- {ds} {mode}: shuffled {np.mean(v):.4f} vs "
                       f"true {np.mean(cell[(ds, mode)]):.4f} "
                       f"vs input_local {np.mean(cell[(ds,'input_local')]):.4f}")

    text = "\n".join(out)
    print(text)
    open(os.path.join(ROOT, "GARA_FINDINGS.md"), "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
