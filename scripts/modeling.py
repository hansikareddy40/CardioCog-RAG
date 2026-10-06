"""Shared modelling code: feature groups, preprocessing, models and metrics.

Every model here is trained on the training split, tuned on the validation
split, and scored once on the test split and on the held-out centres.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

import nacc_utils as nu

PROCESSED = nu.REPO / "data" / "processed"

FEATURE_GROUPS = {
    "Demographics": ["age", "female", "educ_years", "race_white", "race_black", "race_asian", "hispanic", "lives_alone"],
    "Genetics / APOE": ["apoe_e4_count", "apoe_unknown", "family_history"],
    "Cardiovascular / medical": [
        "htn_ever", "chol_ever", "diabetes_ever", "mi_ever", "afib_ever", "chf_ever", "revasc_ever", "bypass_ever",
        "pacemaker_ever", "othercvd_ever", "stroke_ever", "tia_ever", "smoke_ever", "smoke_years", "bp_sys", "bp_dia",
        "pulse_pressure", "heart_rate", "bmi", "vascular_risk_count", "med_bp", "med_lipid", "med_diabetes",
        "med_anticoag", "med_count"],
    "Cognitive tests": ["z_memory", "z_attention", "z_executive", "z_language", "z_visuospatial", "z_global",
                        "tests_failed_cognitive", "gds", "npi_count"],
    "Clinical stage (CDR-SB, diagnosis)": ["CDRSUM", "is_mci", "is_impaired_not_mci"],
}
# Clinician judgements that sit very close to the diagnosis. Reported separately.
PROXIMAL = ["faq_total", "independence", "complaint_self", "complaint_informant", "med_ad"]
TRAJECTORY = [f"{k}_{v}" for v in ["CDRSUM", "z_memory", "z_attention", "z_executive", "z_language", "z_global"]
              for k in ("slope", "delta")] + ["n_prior_visits", "years_of_history"]
PET = ["amyloid_centiloids", "amyloid_amyloid_status", "tau_meta_temporal_suvr_z", "tau_ctx_entorhinal_suvr_z",
       "has_amyloid", "has_tau"]
CORE = [f for feats in FEATURE_GROUPS.values() for f in feats]
SPLITS = ["train", "val", "test", "external"]


def load_visits() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "visits.parquet")


def split_xy(df: pd.DataFrame, features: list[str], target: str) -> dict:
    """{'train': (X, y), ...} as float arrays; missing values stay NaN."""
    return {s: (df.loc[df.split == s, features].to_numpy(np.float32), df.loc[df.split == s, target].to_numpy(int))
            for s in SPLITS}


# ---------------------------------------------------------------------------
# Preprocessing for models that cannot take NaN
# ---------------------------------------------------------------------------

class Prep:
    """Median fill + 'was missing' flags + standardisation, fitted on training data only."""

    def fit(self, X: np.ndarray) -> "Prep":
        self.median = np.nanmedian(X, axis=0)
        self.flag = np.isnan(X).mean(axis=0) > 0.01
        Z = self._fill(X)
        self.mean, self.std = Z.mean(axis=0), Z.std(axis=0) + 1e-6
        return self

    def _fill(self, X):
        return np.hstack([np.where(np.isnan(X), self.median, X), np.isnan(X)[:, self.flag].astype(np.float32)])

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((self._fill(X) - self.mean) / self.std).astype(np.float32)


# ---------------------------------------------------------------------------
# Models. Each fit_* returns a function X -> probability.
# ---------------------------------------------------------------------------

def fit_logistic(d, seed=0):
    prep = Prep().fit(d["train"][0])
    best = None
    for C in [0.01, 0.03, 0.1, 0.3, 1.0]:
        m = LogisticRegression(C=C, max_iter=3000).fit(prep.transform(d["train"][0]), d["train"][1])
        auc = roc_auc_score(d["val"][1], m.predict_proba(prep.transform(d["val"][0]))[:, 1])
        if best is None or auc > best[0]:
            best = (auc, m)
    return lambda X: best[1].predict_proba(prep.transform(X))[:, 1]


def fit_random_forest(d, seed=0):
    prep = Prep().fit(d["train"][0])
    m = RandomForestClassifier(n_estimators=500, min_samples_leaf=10, max_features="sqrt", n_jobs=-1,
                               random_state=seed).fit(prep.transform(d["train"][0]), d["train"][1])
    return lambda X: m.predict_proba(prep.transform(X))[:, 1]


def fit_xgboost(d, seed=0, depths=(3, 4, 6)):
    """Gradient-boosted trees. Missing values are handled natively; the number
    of trees is chosen by early stopping on the validation split."""
    best = None
    for depth in depths:
        m = XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=depth, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=5, reg_lambda=2.0, eval_metric="auc", early_stopping_rounds=100,
                          tree_method="hist", random_state=seed, n_jobs=-1)
        m.fit(d["train"][0], d["train"][1], eval_set=[(d["val"][0], d["val"][1])], verbose=False)
        if best is None or m.best_score > best[0]:
            best = (m.best_score, m)
    model = best[1]
    predict = lambda X: model.predict_proba(X)[:, 1]
    predict.model = model
    return predict


class MLP(nn.Module):
    def __init__(self, n_in, hidden=(128, 64), dropout=0.2):
        super().__init__()
        layers, last = [], n_in
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        self.net = nn.Sequential(*layers, nn.Linear(last, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer (Gorishniy et al., 2021).

    Each feature becomes one token: value * weight + bias. A missing value gets
    that feature's own learned "missing" token instead of an imputed number.
    A [CLS] token attends over all feature tokens and feeds the output layer.
    """

    def __init__(self, n_features, d=64, heads=8, layers=3, dropout=0.15):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d) * 0.05)
        self.bias = nn.Parameter(torch.zeros(n_features, d))
        self.missing = nn.Parameter(torch.randn(n_features, d) * 0.05)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        block = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x):                      # x: (batch, features), NaN = missing
        miss = torch.isnan(x)
        tok = torch.nan_to_num(x).unsqueeze(-1) * self.weight + self.bias
        tok = torch.where(miss.unsqueeze(-1), self.missing.expand_as(tok), tok)
        tok = torch.cat([self.cls.expand(len(x), -1, -1), tok], dim=1)
        return self.head(self.encoder(tok)[:, 0]).squeeze(-1)


def train_torch(model, train, val, predict_fn, epochs=60, lr=1e-3, weight_decay=1e-4, batch=256, patience=8, seed=0):
    """Generic training loop with early stopping on validation AUROC.

    train / val are tuples of tensors whose last element is the label.
    predict_fn(model, *inputs) returns logits.
    """
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(train[-1])
    best_auc, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(predict_fn(model, *[t[idx] for t in train[:-1]]), train[-1][idx].float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        auc = roc_auc_score(val[-1].numpy(), predict_torch(model, val[:-1], predict_fn))
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_torch(model, inputs, predict_fn, batch=2048):
    model.eval()
    out = [torch.sigmoid(predict_fn(model, *[t[i:i + batch] for t in inputs])) for i in range(0, len(inputs[0]), batch)]
    return torch.cat(out).numpy()


_plain = lambda m, x: m(x)


def fit_mlp(d, seed=0):
    torch.manual_seed(seed)
    prep = Prep().fit(d["train"][0])
    t = lambda s: (torch.from_numpy(prep.transform(d[s][0])), torch.from_numpy(d[s][1]))
    model = train_torch(MLP(prep.transform(d["train"][0][:2]).shape[1]), t("train"), t("val"), _plain, seed=seed)
    return lambda X: predict_torch(model, (torch.from_numpy(prep.transform(X)),), _plain)


def fit_ft_transformer(d, seed=0):
    torch.manual_seed(seed)
    mean, std = np.nanmean(d["train"][0], axis=0), np.nanstd(d["train"][0], axis=0) + 1e-6
    scale = lambda X: torch.from_numpy(((X - mean) / std).astype(np.float32))      # NaN stays NaN
    t = lambda s: (scale(d[s][0]), torch.from_numpy(d[s][1]))
    model = train_torch(FTTransformer(d["train"][0].shape[1]), t("train"), t("val"), _plain, lr=5e-4, seed=seed)
    return lambda X: predict_torch(model, (scale(X),), _plain)


MODELS = {"Logistic regression": fit_logistic, "Random forest": fit_random_forest, "XGBoost": fit_xgboost,
          "MLP (neural network)": fit_mlp, "FT-Transformer": fit_ft_transformer}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def bootstrap(y, p, n_boot=1000, seed=0, p_ref=None):
    """AUROC, AUPRC and Brier score with 95% bootstrap intervals.

    If p_ref is given, also returns the paired AUROC difference p - p_ref,
    resampling the same participants for both models.
    """
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    stats = {"auroc": [], "auprc": [], "brier": [], "delta_auroc": []}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        a = roc_auc_score(y[i], p[i])
        stats["auroc"].append(a)
        stats["auprc"].append(average_precision_score(y[i], p[i]))
        stats["brier"].append(brier_score_loss(y[i], p[i]))
        if p_ref is not None:
            stats["delta_auroc"].append(a - roc_auc_score(y[i], p_ref[i]))
    out = {"n": int(len(y)), "events": int(y.sum()),
           "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
           "brier": float(brier_score_loss(y, p))}
    for k, v in stats.items():
        if v:
            out[f"{k}_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    if p_ref is not None:
        out["delta_auroc"] = out["auroc"] - float(roc_auc_score(y, p_ref))
    return out


def calibration_table(y, p, bins=10):
    """Predicted versus observed risk in equal-sized groups (aggregate only)."""
    q = pd.qcut(pd.Series(p), bins, duplicates="drop")
    t = pd.DataFrame({"y": y, "p": p}).groupby(q, observed=True).agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    return t.round(4).to_dict("list")
