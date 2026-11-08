# -*- coding: utf-8 -*-
"""GaRA-style diagnostic: input local/full vs weight local/full vs both.

Modes
-----
  input_local   soft tokens from a 1-hop GCN
  input_full    soft tokens from a 4-hop GCN
  weight_local  GaRA-style K-basis LoRA conditioned on 1-hop pooled z
  weight_full   same, conditioned on 4-hop pooled z
  both_local    1-hop soft tokens + 1-hop weight
  both_full     4-hop soft tokens + 4-hop weight

All modes share the frozen Vicuna-7B, the same prompt, and the same heads.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from garalib import GaRALinear, batches, normalize_adj


class DiagGCN(nn.Module):
    def __init__(self, in_dim, hidden, layers=1, dropout=0.2, no_prop=False):
        super().__init__()
        self.no_prop = no_prop
        self.inp = nn.Linear(in_dim, hidden)
        self.lins = nn.ModuleList([nn.Linear(hidden, hidden)
                                   for _ in range(max(0, layers - 1))])
        self.drop = dropout
        self.head = nn.Linear(hidden, 1)  # unused, keeps state shape stable

    def encode(self, x, adj):
        if self.no_prop:
            return F.relu(self.inp(x))
        h = F.relu(self.inp(torch.sparse.mm(adj, x)))
        for lin in self.lins:
            h = h + F.dropout(F.relu(lin(torch.sparse.mm(adj, h))),
                              self.drop, self.training)
        return h

    def forward(self, x, adj):
        return self.encode(x, adj)


def load_npz(path, limit, vlimit):
    d = np.load(path, allow_pickle=True)
    x = torch.tensor(d["node_features"], dtype=torch.float32)
    y = torch.tensor(d["node_labels"], dtype=torch.long)
    edges = d["edges"]
    tr_m, va_m, te_m = d["train_masks"][0], d["val_masks"][0], d["test_masks"][0]
    tr = torch.tensor(np.nonzero(tr_m)[0][:limit], dtype=torch.long)
    va = torch.tensor(np.nonzero(va_m)[0][:vlimit], dtype=torch.long)
    te = torch.tensor(np.nonzero(te_m)[0][:vlimit], dtype=torch.long)
    return x, y, edges, tr, va, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["input_local", "input_full", "weight_local",
                             "weight_full", "both_local", "both_full",
                             "mix_lw", "mix_fl", "input_feat",
                             "mix_feat"])
    ap.add_argument("--model", default="/root/autodl-tmp/vicuna-7b-v1.5")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--n-tokens", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--pretrain-epochs", type=int, default=100)
    ap.add_argument("--local-layers", type=int, default=1)
    ap.add_argument("--full-layers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--vlimit", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--z-cache", default="")
    ap.add_argument("--shuffle-z", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    x, y, edges, tr, va, te = load_npz(args.data, args.limit, args.vlimit)
    x, y = x.to(dev), y.to(dev)
    if float(x.min()) >= 0.0 and float(x.max()) <= 1.0:
        # binary / bag-of-words features: row L2 normalization
        x = x / (x.norm(dim=1, keepdim=True) + 1e-6)
    else:
        mu, sd = x[tr].mean(0, keepdim=True), x[tr].std(0, keepdim=True) + 1e-2
        x = (x - mu) / sd
    adj = normalize_adj(len(y), edges, dev)
    n_cls = int(y.max()) + 1
    print(f"{args.data} mode={args.mode} n={len(y)} classes={n_cls} "
          f"train={len(tr)} val={len(va)} test={len(te)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    embed = model.get_input_embeddings()
    hidden_size = model.config.hidden_size

    base = model.model.layers[args.layer].self_attn.v_proj
    gara = GaRALinear(base, args.rank, args.K).to(dev)
    model.model.layers[args.layer].self_attn.v_proj = gara

    hidden = 128
    z_local_net = DiagGCN(x.shape[1], hidden, layers=args.local_layers).to(dev)
    z_full_net = DiagGCN(x.shape[1], hidden, layers=args.full_layers).to(dev)
    z_feat_net = DiagGCN(x.shape[1], hidden, layers=1, no_prop=True).to(dev)
    alpha_mlp = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(),
                              nn.Linear(64, args.K)).to(dev)
    aux_local = nn.Linear(hidden, n_cls).to(dev)
    aux_full = nn.Linear(hidden, n_cls).to(dev)
    aux_feat = nn.Linear(hidden, n_cls).to(dev)
    cls_head = nn.Linear(hidden_size, n_cls).to(dev)
    cls_ln = nn.LayerNorm(hidden_size).to(dev)
    projector = nn.Sequential(
        nn.Linear(hidden, args.n_tokens * hidden_size)).to(dev)

    MODE_HOPS = {"input_local": ("local", None), "input_full": ("full", None),
                 "weight_local": (None, "local"), "weight_full": (None, "full"),
                 "both_local": ("local", "local"), "both_full": ("full", "full"),
                 "mix_lw": ("local", "full"), "mix_fl": ("full", "local"),
                 "input_feat": ("feat", None), "mix_feat": ("feat", "full")}
    tok_hop, w_hop = MODE_HOPS[args.mode]
    use_tokens = tok_hop is not None
    use_weight = w_hop is not None

    # pretrain the GCN + auxiliary head so the gap is measured on a fitted
    # encoder, not on a 3-epoch joint run. Select on val (Actor overfits the
    # 520 training nodes within a few dozen full-batch steps).
    def get_z(net, aux, hop):
        hop_tag = ("feat" if hop == "feat" else
                   (f"l{args.local_layers}" if hop == "local"
                    else f"f{args.full_layers}"))
        ds_tag = os.path.basename(args.data).replace(".npz", "")
        ck = ""
        if args.z_cache:
            os.makedirs(args.z_cache, exist_ok=True)
            ck = os.path.join(args.z_cache, f"{ds_tag}_{hop_tag}_s{args.seed}.pt")
        if ck and os.path.exists(ck):
            blob = torch.load(ck, map_location=dev)
            print(f"z-cache hit {ck} aux_val={float(blob['aux_val']):.4f}",
                  flush=True)
            return blob["z"].float().to(dev), float(blob["aux_val"])
        best_val, best_zn, best_an, bad = -1.0, None, None, 0
        opt_pre = torch.optim.AdamW(
            list(net.parameters()) + list(aux.parameters()),
            lr=5e-3, weight_decay=5e-4)
        for ep in range(args.pretrain_epochs):
            net.train(); aux.train()
            z = F.layer_norm(
                net.encode(F.dropout(x, 0.5, True), adj), (hidden,))
            loss = F.cross_entropy(aux(z)[tr], y[tr])
            opt_pre.zero_grad(); loss.backward(); opt_pre.step()
            if (ep + 1) % 10 == 0:
                net.eval()
                with torch.no_grad():
                    zv = F.layer_norm(net.encode(x, adj), (hidden,))
                    av = (aux(zv)[va].argmax(1) == y[va]).float().mean().item()
                if av > best_val + 1e-4:
                    best_val, bad = av, 0
                    best_zn = {k: v.detach().clone()
                               for k, v in net.state_dict().items()}
                    best_an = {k: v.detach().clone()
                               for k, v in aux.state_dict().items()}
                else:
                    bad += 1
                if (ep + 1) % 25 == 0:
                    print(f"pretrain {hop} {ep+1} loss={loss.item():.3f} "
                          f"aux_val={av:.4f} best={best_val:.4f}", flush=True)
                if bad >= 5:
                    print(f"early stop at epoch {ep+1} best={best_val:.4f}",
                          flush=True)
                    break
        if best_zn is not None:
            net.load_state_dict(best_zn)
            aux.load_state_dict(best_an)
        net.eval(); aux.eval()
        for p in list(net.parameters()) + list(aux.parameters()):
            p.requires_grad = False
        with torch.no_grad():
            z = F.layer_norm(net.encode(x, adj), (hidden,))
            av = (aux(z[va]).argmax(1) == y[va]).float().mean().item()
        if ck:
            tmp = ck + f".{os.getpid()}.tmp"
            torch.save({"z": z.cpu(), "aux_val": av}, tmp)
            os.replace(tmp, ck)
        print(f"pretrained {hop} aux_val={av:.4f}", flush=True)
        return z, av

    need = {h for h in (tok_hop, w_hop) if h}
    zl = al = zf = af = zft = aft = None
    if "local" in need:
        zl, al = get_z(z_local_net, aux_local, "local")
    if "full" in need:
        zf, af = get_z(z_full_net, aux_full, "full")
    if "feat" in need:
        zft, aft = get_z(z_feat_net, aux_feat, "feat")
    zmap = {"local": zl, "full": zf, "feat": zft}
    amap = {"local": al, "full": af, "feat": aft}
    z_tok = zmap.get(tok_hop)
    z_w = zmap.get(w_hop)
    if args.shuffle_z and z_w is not None:
        gp = torch.Generator().manual_seed(9000 + args.seed)
        perm = torch.randperm(len(y), generator=gp).to(z_w.device)
        z_w = z_w[perm]
        print("z_w shuffled across nodes (capacity control)", flush=True)
    aux_tok = amap.get(tok_hop)
    aux_w = amap.get(w_hop)

    groups = [
        {"params": list(alpha_mlp.parameters()) + list(cls_head.parameters()) + list(cls_ln.parameters()), "lr": 1e-3},
        {"params": [gara.L, gara.R, gara.G], "lr": args.lr},
    ]
    if use_tokens:
        groups.append({"params": list(projector.parameters()), "lr": 1e-3})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)

    prompt = ("Question: Which class does this node belong to? Options: " +
              ", ".join(f"Class_{i}" for i in range(n_cls)) + ". Answer:")
    pids = tok(prompt, add_special_tokens=False,
               return_tensors="pt").input_ids[0].to(dev)

    def alpha_for(z):
        logits = alpha_mlp(z)
        if args.topk < args.K:
            _, idx = logits.topk(args.topk, dim=1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(1, idx, 0.0)
            logits = logits + mask
        return F.softmax(logits, dim=1)

    def forward_logits(z_t, z_w_):
        ref = z_t if z_t is not None else z_w_
        B = ref.shape[0]
        if use_weight:
            gara.set_alpha(alpha_for(z_w_))
        else:
            gara.set_alpha(None)
        ids = pids.unsqueeze(0).expand(B, -1)
        txt = embed(ids)
        if use_tokens:
            soft = projector(z_t).view(B, args.n_tokens, -1).to(txt.dtype)
            emb = torch.cat([soft, txt], dim=1)
        else:
            emb = txt
        out = model(inputs_embeds=emb, output_hidden_states=True)
        h = cls_ln(out.hidden_states[-1][:, -1].float())
        return cls_head(h)

    def evaluate(nodes):
        correct = 0
        with torch.no_grad():
            for bidx in batches(nodes, args.batch_size, None):
                zt = z_tok[bidx] if use_tokens else None
                zw = z_w[bidx] if use_weight else None
                correct += (forward_logits(zt, zw).argmax(1) == y[bidx]).sum().item()
        return correct / max(len(nodes), 1)

    g = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    for ep in range(args.epochs):
        tot, nb = 0.0, 0
        for bidx in batches(tr, args.batch_size, g):
            zt = z_tok[bidx] if use_tokens else None
            zw = z_w[bidx] if use_weight else None
            logits = forward_logits(zt, zw)
            loss = F.cross_entropy(logits, y[bidx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for grp in opt.param_groups for p in grp["params"]
                 if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item(); nb += 1
        acc_ref = aux_tok if use_tokens else aux_w
        print(f"epoch {ep+1}/{args.epochs} loss={tot/max(nb,1):.4f} "
              f"aux_val={acc_ref:.4f} elapsed={time.time()-t0:.0f}s", flush=True)

    test_acc = evaluate(te)
    aux_acc = aux_tok if use_tokens else aux_w
    trainable = sum(p.numel() for grp in opt.param_groups for p in grp["params"]
                    if p.requires_grad)
    rec = {"dataset": args.data.split("/")[-1].replace(".npz", ""),
           "mode": args.mode, "seed": args.seed, "test_acc": test_acc,
           "aux_val": aux_acc, "aux_tok": aux_tok, "aux_w": aux_w,
           "K": args.K, "rank": args.rank,
           "layer": args.layer, "n_tokens": args.n_tokens,
           "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
           "trainable_params": int(trainable),
           "seconds": round(time.time() - t0, 1), "config": vars(args)}
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print("FINAL", json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
