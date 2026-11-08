# -*- coding: utf-8 -*-
"""Pre-submission audit: every claim in the draft must trace to records.

Landmines checked here:
  1. conclusion vs evidence   -> claim C1..C8 each mapped to a computed number
  2. numbers reconcile        -> one canonical table, no second source
  3. pseudo-replication       -> variance is across seeds; correlation is across
                                 datasets; leverage of each point reported
  4. hidden heterogeneity     -> leave-one-dataset-out sensitivity, per-dataset
                                 values reported next to the pooled slope
  5. small-n overreach        -> permutation p, and what survives without the
                                 dominant point
"""
import glob
import json
import os
from itertools import combinations

import numpy as np
from scipy import stats as st

ROOT = "/root/autodl-tmp/gt_paper"
DIRS = ["gara_diag", "gara_mix", "gara_cora", "gara_ctrl", "gara_sweep",
        "gara_feat"]
BASE = dict(n_tokens=8, full_layers=4, layer=20, K=8, rank=16)


def load():
    per = {}
    for d in DIRS:
        for f in glob.glob(os.path.join(ROOT, d, "*.jsonl")):
            for line in open(f, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                c = r["config"]
                if any(c.get(k) != v for k, v in BASE.items()):
                    continue
                if c.get("shuffle_z"):
                    continue
                per.setdefault((r["dataset"], r["mode"]), []).append(
                    (r["seed"], r["test_acc"]))
    return per


def agg(per, ds, mode):
    v = sorted(set(per.get((ds, mode), [])))
    return (float(np.mean([x for _, x in v])), float(np.std([x for _, x in v])),
            len(v), dict(v))


def main():
    per = load()
    dsets = sorted({k[0] for k in per})
    rows = []
    print(f"{'dataset':16s} {'IL':>6s} {'IF':>6s} {'mix':>6s} {'gap':>7s} "
          f"{'gain':>7s} {'rec':>6s} {'n':>2s}  d<0 (tokens)")
    for ds in dsets:
        il, ils, n, ilv = agg(per, ds, "input_local")
        inf, _, _, ifv = agg(per, ds, "input_full")
        lw, _, nlw, lwv = agg(per, ds, "mix_lw")
        if np.isnan(lw):
            continue
        gap, gain = inf - il, lw - il
        rec = gain / gap if abs(gap) > 1e-9 else float("nan")
        seeds = sorted(set(ilv) & set(lwv) & set(ifv))
        deltas = [lwv[s] - ilv[s] for s in seeds]
        rows.append((ds, il, inf, lw, gap, gain, rec, n, deltas))
        print(f"{ds:16s} {il:6.3f} {inf:6.3f} {lw:6.3f} {gap:+7.3f} "
              f"{gain:+7.3f} {rec:+6.2f} {n:2d}  "
              f"{np.mean(deltas):+.3f}±{np.std(deltas):.3f}")

    gaps = np.array([r[4] for r in rows])
    gains = np.array([r[5] for r in rows])
    names = [r[0] for r in rows]
    rho, p = st.spearmanr(gaps, gains)
    slope = float((gaps * gains).sum() / (gaps * gaps).sum())
    print(f"\n[A] Spearman = {rho:+.3f}, p = {p:.4f}, n = {len(rows)}")
    print(f"[A] slope through origin = {slope:.3f}")
    # permutation test on rho
    rng = np.random.default_rng(0)
    null = np.array([st.spearmanr(gaps, rng.permutation(gains))[0]
                     for _ in range(5000)])
    print(f"[A] permutation p = {(np.abs(null) >= abs(rho)).mean():.4f}")

    print("\n[B] leave-one-dataset-out (which point carries the result?)")
    for i, nm in enumerate(names):
        keep = [j for j in range(len(rows)) if j != i]
        r_, p_ = st.spearmanr(gaps[keep], gains[keep])
        print(f"  drop {nm:16s} rho={r_:+.3f} p={p_:.3f}")
    keep = [j for j in range(len(rows)) if np.abs(gaps[j]) > 0.02]
    if len(keep) >= 3:
        r_, p_ = st.spearmanr(gaps[keep], gains[keep])
        print(f"\n[B] only datasets with |gap| > 0.02 ({len(keep)}): "
              f"rho={r_:+.3f} p={p_:.3f}")

    print("\n[C] same-hop redundancy (max |both - input|)")
    worst = 0.0
    for ds in dsets:
        il, _, _, _ = agg(per, ds, "input_local")
        bl, _, _, _ = agg(per, ds, "both_local")
        inf, _, _, _ = agg(per, ds, "input_full")
        bf, _, _, _ = agg(per, ds, "both_full")
        if not np.isnan(bl):
            worst = max(worst, abs(bl - il))
            print(f"  {ds:16s} both_local-IL {bl-il:+.3f}")
        if not np.isnan(bf):
            worst = max(worst, abs(bf - inf))
            print(f"  {ds:16s} both_full-IF  {bf-inf:+.3f}")
    print(f"[C] worst |both - input| = {worst:.3f}")

    print("\n[D] effect concentrated? share of total positive gain from Roman")
    pos = gains[gains > 0]
    rom = gains[names.index("roman_empire")]
    print(f"  sum positive gain = {pos.sum():.3f}; roman = {rom:.3f} "
          f"({rom / pos.sum():.0%})")

    print("\\n[F] policy evaluation (decision rule vs always-on)")
    il_all = np.array([r[1] for r in rows])
    lw_all = np.array([r[3] for r in rows])
    print(f"  never add adapter: mean acc {il_all.mean():.4f}, gain +0.0000")
    print(f"  always add:        mean acc {lw_all.mean():.4f}, "
          f"gain {lw_all.mean() - il_all.mean():+.4f}")
    for tau in (0.01, 0.02, 0.03):
        sel = gaps > tau
        acc = np.where(sel, lw_all, il_all).mean()
        print(f"  gate at gap>{tau:.2f}: mean acc {acc:.4f}, "
              f"gain {acc - il_all.mean():+.4f}, adapters on {int(sel.sum())}/{len(rows)}")
    loo_sum = 0.0
    for i in range(len(rows)):
        tr = [j for j in range(len(rows)) if j != i]
        best_tau, best_gain = 0.0, -9.0
        for tau in np.linspace(gaps[tr].min(), gaps[tr].max(), 25):
            sel = gaps[tr] > tau
            g = np.where(sel, gains[tr], 0).mean()
            if g > best_gain:
                best_gain, best_tau = g, tau
        pred = gains[i] if gaps[i] > best_tau else 0.0
        loo_sum += pred
        print(f"  LOO {names[i]:16s} tau={best_tau:+.3f} "
              f"pred_gain={pred:+.3f} true={gains[i]:+.3f}")
    print(f"  LOO policy total gain = {loo_sum:+.3f} vs always "
          f"{gains.sum():+.3f} vs never 0")

    print("\\n[E] seeds per cell (pseudo-replication check)")
    for ds in dsets:
        n = agg(per, ds, "input_local")[2]
        print(f"  {ds:16s} n_seeds={n}")


if __name__ == "__main__":
    main()
