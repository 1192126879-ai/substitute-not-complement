# A Substitute, Not a Complement

Code and per-run records for the paper *A Substitute, Not a Complement: Weight-Space Adaptation
Carries Missing Graph Information into Large Language Models* (Neurocomputing, submitted).

The paper runs a 2x2 design that crosses the receptive field of a soft-token input channel with the
receptive field of a graph-conditioned low-rank adapter, on a frozen 7B backbone. Every number in the
paper is a mean over seeds of the JSONL records in `results/`; the analysis scripts regenerate the
tables and figures from those records without touching a GPU.

## Layout

```
code/
  gara_diag.py        one training+evaluation run for one (dataset, mode, seed)
  garalib.py          graph-normalised adjacency, GaRA-style low-rank layer, batching
  latency_bench.py    per-node inference latency of the two channels
  analyze_2x2.py      2x2 table, recovery ratio, shuffled control
  analyze_all.py      consolidated tables across every run directory
  audit_draft.py      reproduces every claim and the leave-one-dataset-out checks
  stats_final.py      means, standard deviations, Spearman, slope
  make_figs.py        Figures 1-5
  summarize_runs.py   per-directory summary tables
results/
  gara_diag/          7 datasets x 6 modes x 3 seeds (Vicuna-7B)
  gara_mix/           cross cells: local tokens + full adapter, and the reverse
  gara_ctrl/          shuffled-z control, rank/K sweeps, extra seeds
  gara_sweep/         token budget, encoder depth, injection layer
  gara_cora/          Cora, all modes
  gara_feat/          structure-starved tokens (features only, no propagation)
  gara_qwen/          Qwen2.5-7B replication
  gara_qmix/          Qwen2.5-7B cross cells
  gara_abl/           ablations
figures/              Figures 1-5 at 300 dpi
```

## Install

```
pip install torch transformers numpy scipy matplotlib
```

Tested with Python 3.8, PyTorch 2.x, transformers 4.x, on one 80 GB GPU.

## Data

Datasets are the public heterophilous benchmarks (Roman-empire, Amazon-ratings, Minesweeper,
Tolokers; https://github.com/yandex-research/heterophilous-graphs) plus Chameleon, Squirrel, Actor and
Cora (PyTorch Geometric). Each dataset is stored as an `.npz` with:

```
node_features  float32 [n, d]
node_labels    int64   [n]
edges          int64   [2, E]      (undirected, one direction stored)
train_masks    bool    [1, n]
val_masks      bool    [1, n]
test_masks     bool    [1, n]
```

Features are row-L2-normalised when all values lie in [0, 1] (binary/bag-of-words) and standardised
otherwise; the choice is made inside `gara_diag.py` from the data itself.

## Reproduce one cell

```
python code/gara_diag.py --data roman_empire.npz --mode mix_lw --seed 0     --model /path/to/vicuna-7b-v1.5 --layer 20 --K 8 --rank 16 --n-tokens 8     --epochs 3 --batch-size 8 --limit 4000 --vlimit 2000 --z-cache zcache     --out results/roman_mix_lw_s0.jsonl
```

Modes: `input_local`, `input_full`, `weight_local`, `weight_full`, `both_local`, `both_full`,
`mix_lw`, `mix_fl`, `input_feat`, `mix_feat`. Add `--shuffle-z` for the capacity control.

The graph encoder is pretrained once per (dataset, depth, seed) and cached in `--z-cache`, so every
mode of a dataset shares the identical embedding; this is what makes the comparison a paired one.

## Reproduce the tables and figures

```
python code/analyze_2x2.py     # 2x2 cells, recovery, shuffled control
python code/audit_draft.py     # every claim, leave-one-dataset-out, policy evaluation
python code/stats_final.py     # means, Spearman, slope
python code/make_figs.py       # Figures 1-5
```

## Record format

One JSON object per line:

```
dataset, mode, seed, test_acc, aux_val, aux_tok, aux_w, K, rank, layer, n_tokens,
peak_mem_gb, trainable_params, seconds, config{...}
```

`aux_val` is the validation accuracy of a linear head on the frozen embedding — the probe used to
measure how much label information a receptive field carries.

## Citation

```
@article{hu2026substitute,
  title  = {A Substitute, Not a Complement: Weight-Space Adaptation Carries Missing Graph
            Information into Large Language Models},
  author = {Hu, En},
  journal= {Neurocomputing (submitted)},
  year   = {2026}
}
```

## License

MIT (see LICENSE).
