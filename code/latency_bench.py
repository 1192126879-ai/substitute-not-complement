# -*- coding: utf-8 -*-
"""Per-node inference latency: soft-token path vs graph-adapter path.

Both paths run the same frozen 7B on the same prompt; the only difference is
what is added per node. Weights of the projectors/adapters are random: latency
does not depend on their values, only on shapes.
"""
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from k_scan_cora import GaRALinear

MODEL = os.environ.get("MODEL", "vicuna-7b-v1.5")
CACHE_L = os.environ.get("CACHE_L", "zcache/roman_empire_l1_s0.pt")
CACHE_F = os.environ.get("CACHE_F", "zcache/roman_empire_f4_s0.pt")
N_NODES = 96
N_TOKENS = 8
REPEATS = 2


def main():
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    hidden = model.config.hidden_size
    embed = model.get_input_embeddings()
    gara = GaRALinear(model.model.layers[20].self_attn.v_proj, 16, 8).to(dev)
    model.model.layers[20].self_attn.v_proj = gara

    zl = torch.load(CACHE_L, map_location=dev)["z"].float()[:N_NODES]
    zf = torch.load(CACHE_F, map_location=dev)["z"].float()[:N_NODES]
    n_cls = 18
    prompt = ("Question: Which class does this node belong to? Options: " +
              ", ".join(f"Class_{i}" for i in range(n_cls)) + ". Answer:")
    pids = tok(prompt, add_special_tokens=False,
               return_tensors="pt").input_ids[0].to(dev)
    projector = nn.Linear(128, N_TOKENS * hidden).to(dev)
    alpha_mlp = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 8)).to(dev)

    def run(tokens_from, weight_from, bs):
        zt = {"l": zl, "f": zf, None: None}[tokens_from]
        zw = {"l": zl, "f": zf, None: None}[weight_from]
        if zw is not None:
            logits = alpha_mlp(zw)
            _, idx = logits.topk(2, dim=1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(1, idx, 0.0)
            gara.set_alpha(F.softmax(logits + mask, dim=1))
        else:
            gara.set_alpha(None)
        ref = zt if zt is not None else zw
        B = ref.shape[0] if ref is not None else bs
        ids = pids.unsqueeze(0).expand(B, -1)
        txt = embed(ids)
        if zt is not None:
            soft = projector(zt).view(B, N_TOKENS, -1).to(txt.dtype)
            emb = torch.cat([soft, txt], dim=1)
        else:
            emb = txt
        out = model(inputs_embeds=emb, output_hidden_states=True)
        return out.hidden_states[-1][:, -1].float()

    configs = [("prompt only", None, None),
               ("input_local (8 tokens)", "l", None),
               ("input_full (8 tokens)", "f", None),
               ("weight_full (adapter)", None, "f"),
               ("mix_lw", "l", "f")]
    report = {"nodes": N_NODES, "device": torch.cuda.get_device_name(0),
              "configs": []}
    for bs in (1, 8):
        for name, tf, wf in configs:
            with torch.no_grad():
                for i in range(0, N_NODES, bs):
                    run(tf, wf, bs)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                times = []
                for _ in range(REPEATS):
                    t0 = time.time()
                    for i in range(0, N_NODES, bs):
                        run(tf, wf, bs)
                    torch.cuda.synchronize()
                    times.append(time.time() - t0)
                times.sort()
                med = times[len(times) // 2]
                mem = (torch.cuda.max_memory_allocated() - base) / 1e9
            rec = {"batch": bs, "config": name,
                   "ms_per_node": round(med / N_NODES * 1000, 2),
                   "nodes_per_s": round(N_NODES / med, 1),
                   "extra_gb": round(mem, 2)}
            report["configs"].append(rec)
            print(json.dumps(rec), flush=True)
    open("/root/autodl-tmp/gt_paper/latency.json", "w").write(
        json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
