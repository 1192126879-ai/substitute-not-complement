# -*- coding: utf-8 -*-
"""Figures for the substitution paper."""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/root/autodl-tmp/gt_paper"
FIG = os.path.join(ROOT, "figs")
os.makedirs(FIG, exist_ok=True)
DIRS = ["gara_diag", "gara_mix", "gara_ctrl", "gara_sweep", "gara_cora",
        "gara_qmix", "gara_qwen"]
CELL = ["input_local", "input_full", "both_local", "both_full",
        "weight_local", "weight_full", "mix_lw", "mix_fl"]
BASE = dict(n_tokens=8, full_layers=4, layer=20, K=8, rank=16)


def load():
    recs = []
    for d in DIRS:
        for f in glob.glob(os.path.join(ROOT, d, "*.jsonl")):
            for line in open(f, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    r["_dir"] = d
                    r["_file"] = os.path.basename(f)
                    recs.append(r)
    return recs


def is_base(r):
    c = r["config"]
    return (all(c.get(k) == v for k, v in BASE.items())
            and not c.get("shuffle_z")
            and not c.get("z_cache", "").endswith("_x"))


def mean(v):
    return float(np.mean(v)) if v else float("nan")


def main():
    recs = load()
    cell = defaultdict(list)
    model = defaultdict(str)
    ctrl = defaultdict(list)
    for r in recs:
        c = r["config"]
        if r["mode"] not in CELL:
            continue
        if "qwen" in r["_dir"] or "qwen" in c.get("model", ""):
            tag = "qwen"
        else:
            tag = "vicuna"
        if is_base(r):
            cell[(tag, r["dataset"], r["mode"])].append(r["test_acc"])
        elif tag == "vicuna" and r["dataset"] == "roman_empire":
            t = []
            if c.get("shuffle_z"):
                t.append("shuffled")
            if c.get("n_tokens") != 8:
                t.append(f"tok{c.get('n_tokens')}")
            if c.get("full_layers") != 4:
                t.append(f"D{c.get('full_layers')}")
            if c.get("layer") != 20:
                t.append(f"L{c.get('layer')}")
            if c.get("K") != 8:
                t.append(f"K{c.get('K')}")
            if c.get("rank") != 16:
                t.append(f"r{c.get('rank')}")
            if t:
                ctrl[(r["mode"], t[0])].append(r["test_acc"])

    dsets = sorted({k[1] for k in cell if k[0] == "vicuna"})
    # ---- Fig 1: 2x2 interaction on the two large-gap datasets
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, ds, title in zip(axes, ["roman_empire", "amazon_ratings"],
                             ["Roman-empire", "Amazon-ratings"]):
        il = mean(cell[("vicuna", ds, "input_local")])
        inf = mean(cell[("vicuna", ds, "input_full")])
        bl = mean(cell[("vicuna", ds, "both_local")])
        bf = mean(cell[("vicuna", ds, "both_full")])
        lw = mean(cell[("vicuna", ds, "mix_lw")])
        fl = mean(cell[("vicuna", ds, "mix_fl")])
        x = [0, 1]
        ax.plot(x, [il, inf], "o-", color="#2b6cb0", label="input only")
        ax.plot(x, [bl, bf], "s--", color="#2b6cb0", alpha=.55,
                label="input + weight (same hop)")
        ax.plot(x, [lw, fl], "^-", color="#c53030",
                label="input + weight (other hop)")
        ax.set_xticks(x)
        ax.set_xticklabels(["1-hop input", "4-hop input"])
        ax.set_ylabel("test accuracy")
        ax.set_title(title)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_interaction.png"), dpi=300)

    # ---- Fig 2: gap vs recovery
    pts = []
    for ds in dsets:
        il = mean(cell[("vicuna", ds, "input_local")])
        inf = mean(cell[("vicuna", ds, "input_full")])
        lw = mean(cell[("vicuna", ds, "mix_lw")])
        if np.isnan(lw):
            continue
        pts.append((inf - il, lw - il, ds))
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=45, color="#2b6cb0",
               zorder=3)
    off = {"roman_empire": (6, 4), "amazon_ratings": (6, 4),
           "chameleon": (-40, 8), "squirrel": (6, -10), "cora": (-28, 6),
           "actor": (6, 3), "minesweeper": (6, -10), "tolokers": (6, 4)}
    for g, s_, ds in pts:
        ax.annotate(ds.replace("_", "-"), (g, s_), fontsize=7,
                    xytext=off.get(ds, (4, 3)), textcoords="offset points")
    lim = max(abs(p[0]) for p in pts) * 1.08
    ax.plot([0, lim], [0, lim], "k--", lw=.8, label="full recovery (slope 1)")
    ax.axhline(0, color="gray", lw=.6)
    ax.set_xlabel("information gap (input_full - input_local)")
    ax.set_ylabel("weight gain (mix_lw - input_local)")
    ax.set_title("weight substitutes for missing input information", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_gap_recovery.png"), dpi=300)

    # ---- Fig 3: controls on Roman
    base_il = mean(cell[("vicuna", "roman_empire", "input_local")])
    base_lw = mean(cell[("vicuna", "roman_empire", "mix_lw")])
    groups = ["shuffled", "D1", "D2", "D4", "D6", "r4", "r16", "r32",
              "K2", "K8", "K32", "L12", "L20", "L24", "tok2", "tok8", "tok16"]
    fill = {"D4": base_lw, "r16": base_lw, "K8": base_lw,
            "L20": base_lw, "tok8": base_lw}
    vals = []
    for g in groups:
        v = ctrl.get(("mix_lw", g), [])
        vals.append(mean(v) if v else fill.get(g, np.nan))
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    colors = ["#c53030" if g == "shuffled" else "#2b6cb0" for g in groups]
    ax.bar(range(len(groups)), vals, color=colors)
    ax.axhline(base_lw, color="#2b6cb0", ls="--", lw=.8, label="mix_lw (full)")
    ax.axhline(base_il, color="gray", ls=":", lw=.9, label="input_local")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, fontsize=7)
    ax.set_ylim(0.3, 0.58)
    ax.set_ylabel("test accuracy (Roman)")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_controls.png"), dpi=300)

    # ---- Fig 4: efficiency
    modes = ["input_full", "weight_full", "mix_lw"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for ax, field, lab in zip(axes, ["peak_mem_gb", "seconds", "trainable_params"],
                              ["peak memory (GB)", "train time (s)", "trainable params"]):
        vals = []
        for m in modes:
            v = [r[field] for r in recs if r["mode"] == m and r["dataset"] == "roman_empire"
                 and is_base(r) and "qwen" not in r["_dir"]]
            vals.append(mean(v))
        ax.bar(modes, vals, color=["#2b6cb0", "#dd6b20", "#c53030"])
        ax.set_title(lab, fontsize=8)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_efficiency.png"), dpi=300)

    # ---- Fig 5: structure-starved deficit
    feat = defaultdict(list)
    for f in glob.glob(os.path.join(ROOT, "gara_feat", "*.jsonl")):
        if os.path.basename(f).startswith("shuf_"):
            continue
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                feat[(r["dataset"], r["mode"])].append(r["test_acc"])
    order = ["cora", "chameleon", "minesweeper", "squirrel",
             "roman_empire", "amazon_ratings", "actor", "tolokers"]
    labels = {"cora": "Cora", "chameleon": "Chameleon", "minesweeper": "Minesweeper",
              "squirrel": "Squirrel", "roman_empire": "Roman", "amazon_ratings": "Amazon",
              "actor": "Actor", "tolokers": "Tolokers"}
    x = np.arange(len(order)); w = 0.27
    fv = [mean(feat[(d, "input_feat")]) for d in order]
    mv = [mean(feat[(d, "mix_feat")]) for d in order]
    tv = [mean(cell[("vicuna", d, "input_full")]) for d in order]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.bar(x - w, fv, w, label="features only (no structure)", color="#a0aec0")
    ax.bar(x, mv, w, label="+ 4-hop adapter", color="#c53030")
    ax.bar(x + w, tv, w, label="4-hop soft tokens", color="#2b6cb0")
    ax.set_xticks(x); ax.set_xticklabels([labels[d] for d in order], fontsize=7)
    ax.set_ylabel("test accuracy")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_structure_deficit.png"), dpi=300)
    print("fig5 written")

    print("figs written to", FIG)


if __name__ == "__main__":
    main()
