# -*- coding: utf-8 -*-
import glob
import os, json, os
from collections import defaultdict
import numpy as np
from scipy import stats as st

ROOT = os.environ.get("GARA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = dict(n_tokens=8, full_layers=4, layer=20, K=8, rank=16)
CELL = ["input_local", "input_full", "both_local", "both_full", "weight_local", "weight_full", "mix_lw", "mix_fl"]

def load(dirs):
    per = defaultdict(lambda: defaultdict(list))
    for d in dirs:
        for f in glob.glob(os.path.join(ROOT, d, "*.jsonl")):
            for line in open(f, encoding="utf-8"):
                if not line.strip(): continue
                r = json.loads(line)
                c = r["config"]
                if any(c.get(k) != v for k, v in BASE.items()) or c.get("shuffle_z"):
                    continue
                per[r["dataset"]][r["mode"]].append((r["seed"], r["test_acc"]))
    return per

def agg(per):
    out = {}
    for ds, modes in per.items():
        out[ds] = {m: (np.mean([v for _, v in recs]), np.std([v for _, v in recs]), len(recs))
                   for m, recs in modes.items()}
    return out

def main():
    vic = agg(load(["gara_diag", "gara_mix", "gara_cora", "gara_ctrl", "gara_sweep"]))
    qwen = agg(load(["gara_qwen", "gara_qmix"]))
    print("== Vicuna ==")
    for ds in sorted(vic):
        d = vic[ds]
        if "input_local" not in d: continue
        il, if_, lw = d.get("input_local"), d.get("input_full"), d.get("mix_lw")
        print(f"{ds:16s} IL={il[0]:.3f} IF={if_[0]:.3f} mix={lw[0]:.3f} gap={if_[0]-il[0]:+.3f} rec={(lw[0]-il[0])/(if_[0]-il[0]):+.2f} n={il[2]}")
    print("\n== Qwen ==")
    for ds in sorted(qwen):
        d = qwen[ds]
        if "input_local" in d and "input_full" in d and "mix_lw" in d:
            il, if_, lw = d["input_local"], d["input_full"], d["mix_lw"]
            print(f"{ds:16s} IL={il[0]:.3f} IF={if_[0]:.3f} mix={lw[0]:.3f} gap={if_[0]-il[0]:+.3f} rec={(lw[0]-il[0])/(if_[0]-il[0]):+.2f} n={il[2]}")
    # paired tests on Roman and Amazon
    print("\n== paired deltas (Vicuna) ==")
    for ds in ["roman_empire", "amazon_ratings", "cora"]:
        d = vic[ds]
        for a, b in [("mix_lw", "input_local"), ("mix_fl", "input_full"), ("both_local", "input_local"), ("both_full", "input_full")]:
            if a in d and b in d:
                da = dict(d[a][:0] if False else [(s, v) for s, v in []])
                print(f"  {ds} {a}-{b}: {d[a][0]-d[b][0]:+.3f}")
    # spearman across datasets
    xs, ys, names = [], [], []
    for ds, d in vic.items():
        if all(m in d for m in ("input_local", "input_full", "mix_lw")):
            g = d["input_full"][0] - d["input_local"][0]
            h = d["mix_lw"][0] - d["input_local"][0]
            xs.append(g); ys.append(h); names.append(ds)
    rho, p = st.spearmanr(xs, ys)
    print(f"\nSpearman gap vs gain = {rho:+.3f} p={p:.4f} n={len(xs)}")
    x2 = [x for x, n in zip(xs, names) if n != "tolokers"]
    y2 = [y for y, n in zip(ys, names) if n != "tolokers"]
    rho2, p2 = st.spearmanr(x2, y2)
    print(f"without tolokers: rho={rho2:+.3f} p={p2:.4f} n={len(x2)}")
    # correlation with slope through origin
    x = np.array(xs); y = np.array(ys)
    print(f"slope through origin = {(x*y).sum()/(x*x).sum():.3f}")

if __name__ == "__main__":
    main()
