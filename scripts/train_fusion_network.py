"""Modality-encoder fusion network, as proposed at the zeroth review.

Each kind of information has its own small encoder that turns its features
into one 32-dimensional "modality token". A multi-head attention layer, queried
by a learned summary token, combines the available tokens (Tier 1 fusion).
A kind of information that is missing for a person is replaced by a learned
"missing" token, so the attention can learn to rely on it less. Two output
heads share the fused representation:

  risk head         probability of dementia within 3 years (classification)
  progression head  change in CDR-SB over 3 years (regression), trained on the
                    participants who have a visit near the 3-year mark

During training whole kinds of information are hidden at random (modality
dropout, probability 0.35), as in train_final.py. PET is added afterwards by the
existing late-fusion model (Tier 2), because too few scanned participants have
3-year outcomes to train it inside the network.

The attention weights given to each modality token are reported as an
explanation of how much the network relied on each kind of information.

Output  reports/phase6b_fusion_network_results.json   (aggregate only)
Usage:  python scripts/train_fusion_network.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, roc_auc_score

import modeling as M
import nacc_utils as nu
from train_decline import three_year_change

TARGET = "y_dementia_3y"
SEEDS = [0, 1, 2]
GROUPS = list(M.FEATURE_GROUPS)                       # five tabular modalities
OPTIONAL = [g for g in GROUPS if g != "Demographics"]
DROP_PROB = 0.35
DIM = 32
DX_GROUPS = {"All non-demented": [1, 2, 3], "Cognitively normal": [1], "MCI": [3]}
PATTERNS = {"Everything": GROUPS,
            "No cardiovascular data": [g for g in GROUPS if g != "Cardiovascular / medical"],
            "No cognitive tests": [g for g in GROUPS if g != "Cognitive tests"],
            "No clinical stage": [g for g in GROUPS if g != "Clinical stage (CDR-SB, diagnosis)"],
            "No cognitive information at all": ["Demographics", "Genetics / APOE", "Cardiovascular / medical"],
            "Demographics only": ["Demographics"]}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg):
    print(f"[fusion net] {msg}", flush=True)


class GroupPrep:
    """Per modality: median fill, missing flags and scaling fitted on training rows; plus a 'modality present' flag."""

    def fit(self, df):
        self.preps = {g: M.Prep().fit(df[M.FEATURE_GROUPS[g]].to_numpy(np.float32)) for g in GROUPS}
        return self

    def transform(self, df):
        xs, present = [], []
        for g in GROUPS:
            raw = df[M.FEATURE_GROUPS[g]].to_numpy(np.float32)
            xs.append(torch.from_numpy(self.preps[g].transform(raw)))
            present.append(~np.isnan(raw).all(axis=1))
        return xs, torch.from_numpy(np.stack(present, axis=1))


class ModalityFusionNet(nn.Module):
    def __init__(self, in_dims, dim=DIM, heads=4, dropout=0.2):
        super().__init__()
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(d, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim)) for d in in_dims])
        self.missing = nn.Parameter(torch.randn(len(in_dims), dim) * 0.02)
        self.type_emb = nn.Parameter(torch.randn(len(in_dims), dim) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.risk_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, 1))
        self.prog_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, xs, present):
        tokens = torch.stack([enc(x) for enc, x in zip(self.encoders, xs)], dim=1)         # (B, G, D)
        tokens = torch.where(present.unsqueeze(-1), tokens, self.missing.unsqueeze(0).expand_as(tokens)) + self.type_emb
        fused, weights = self.attn(self.query.expand(len(tokens), -1, -1), tokens, tokens, need_weights=True)
        fused = self.norm(fused.squeeze(1))
        return self.risk_head(fused).squeeze(-1), self.prog_head(fused).squeeze(-1), weights.squeeze(1)   # weights (B, G)


def to_device(xs, present):
    return [x.to(DEVICE) for x in xs], present.to(DEVICE)


@torch.no_grad()
def predict(model, xs, present, batch=4096):
    model.eval()
    risk, prog, w = [], [], []
    for i in range(0, len(present), batch):
        r, p, a = model([x[i:i + batch].to(DEVICE) for x in xs], present[i:i + batch].to(DEVICE))
        risk.append(torch.sigmoid(r).cpu()); prog.append(p.cpu()); w.append(a.cpu())
    return torch.cat(risk).numpy(), torch.cat(prog).numpy(), torch.cat(w).numpy()


def mask(present, available):
    out = present.clone()
    for j, g in enumerate(GROUPS):
        if g not in available:
            out[:, j] = False
    return out


def train(seed, tr, va):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    xs, present, y, ch = tr
    model = ModalityFusionNet([x.shape[1] for x in xs]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce, huber = nn.BCEWithLogitsLoss(), nn.HuberLoss()
    opt_idx = torch.tensor([GROUPS.index(gn) for gn in OPTIONAL])
    best, best_state, bad = -1.0, None, 0
    for epoch in range(80):
        model.train()
        for idx in torch.randperm(len(y), generator=g).split(256):
            p = present[idx].clone()
            drop = torch.rand(len(idx), len(OPTIONAL), generator=g) < DROP_PROB
            p[:, opt_idx] &= ~drop
            r, prog, _ = model([x[idx].to(DEVICE) for x in xs], p.to(DEVICE))
            loss = bce(r, y[idx].to(DEVICE))
            has = ~torch.isnan(ch[idx])
            if has.any():
                loss = loss + 0.5 * huber(prog[has.to(DEVICE)], ch[idx][has].to(DEVICE))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        r_va, _, _ = predict(model, va[0], va[1])
        auc = roc_auc_score(va[2].numpy(), r_va)
        if auc > best:
            best, bad, best_state = auc, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    model.load_state_dict(best_state)
    return model


def main():
    t0 = time.time()
    visits = M.load_visits()
    df = visits[(visits.NACCVNUM == 1) & visits.eligible & visits[TARGET].notna()].reset_index(drop=True)
    df = df.join(three_year_change(visits), on="NACCID")
    prep = GroupPrep().fit(df[df.split == "train"])

    def tensors(part):
        xs, present = prep.transform(part)
        return xs, present, torch.tensor(part[TARGET].to_numpy(np.float32)), torch.tensor(part.cdrsb_change_3y.to_numpy(np.float32))

    sets = {s: tensors(df[df.split == s]) for s in M.SPLITS}
    models = [train(seed, sets["train"], sets["val"]) for seed in SEEDS]
    log(f"trained {len(models)} seeds on {DEVICE} in {time.time() - t0:.0f}s")

    results = {"design": {"modalities": GROUPS, "token_dim": DIM, "attention_heads": 4, "modality_dropout": DROP_PROB,
                          "heads": ["risk (dementia within 3 years)", "progression (3-year change in CDR-SB)"], "seeds": len(SEEDS)},
               "risk": {}, "missing_patterns": {}, "attention_share": {}, "progression": {}}
    for split in ["test", "external"]:
        part = df[df.split == split]
        xs, present, y, ch = sets[split]
        outs = [predict(m, xs, present) for m in models]
        risk = np.mean([o[0] for o in outs], axis=0)
        prog = np.mean([o[1] for o in outs], axis=0)
        w = np.mean([o[2] for o in outs], axis=0)
        yv, dx = y.numpy().astype(int), part.NACCUDSD.to_numpy()
        results["risk"][split] = {g: M.bootstrap(yv[np.isin(dx, c)], risk[np.isin(dx, c)], n_boot=500) for g, c in DX_GROUPS.items()}
        results["attention_share"][split] = {g: dict(zip(GROUPS, np.round(w[np.isin(dx, c)].mean(axis=0), 4).tolist())) for g, c in DX_GROUPS.items()}
        has = ~np.isnan(ch.numpy())
        cv = ch.numpy()
        results["progression"][split] = {g: {"n": int((has & np.isin(dx, c)).sum()),
                                             "mae_model": float(mean_absolute_error(cv[has & np.isin(dx, c)], prog[has & np.isin(dx, c)])),
                                             "mae_predict_no_change": float(np.abs(cv[has & np.isin(dx, c)]).mean())} for g, c in DX_GROUPS.items()}
        results["missing_patterns"][split] = {}
        for name, available in PATTERNS.items():
            pm = mask(present, available)
            rp = np.mean([predict(m, xs, pm)[0] for m in models], axis=0)
            wp = np.mean([predict(m, xs, pm)[2] for m in models], axis=0)
            results["missing_patterns"][split][name] = {**M.bootstrap(yv, rp, n_boot=300), "attention_share": dict(zip(GROUPS, np.round(wp.mean(axis=0), 4).tolist()))}
        r = results["risk"][split]
        log(f"{split}: " + ", ".join(f"{g} {v['auroc']:.3f}" for g, v in r.items()) +
            f" | progression MAE {results['progression'][split]['All non-demented']['mae_model']:.3f} vs no-change "
            f"{results['progression'][split]['All non-demented']['mae_predict_no_change']:.3f}")
        log(f"{split} attention share: {results['attention_share'][split]['All non-demented']}")

    text = json.dumps(results, indent=1)
    nu.assert_no_ids(text, "fusion network results")
    (nu.REPORTS / "phase6b_fusion_network_results.json").write_text(text)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
