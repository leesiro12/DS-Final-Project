#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Difficulty Regressor (from engineered per-map features)
-------------------------------------------------------------------
Input  : ./map_features/features_all_maps.csv  (one row per tlozD_Q)
Target : baseD (difficulty level)
Seqing : group rows into sequences (by quest, default), pad, mask, train transformer
Split  : TRAIN = odd baseD, TEST = even baseD (configurable)
Outputs:
  - runs/transformer_preds.csv           (key, quest, baseD_true, baseD_pred)
  - runs/pred_vs_true.png                (scatter)
  - runs/metrics.json                    (MAE/RMSE/R2 on train/test)
  - runs/feature_columns.json            (feature list used)
"""

from __future__ import annotations
import os, json, math, random, re, warnings
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# =========================
# CONFIG
# =========================
PARAMS = {
    "features_csv": "../Features/features_all_maps.csv",  # input
    "out_dir": "../Transformer_Output",                   # outputs

    # Grouping for sequences: "quest" (default) or "baseD"
    # - "quest": sequences are tloz1_Q -> tloz2_Q -> ... (various difficulties within the same quest)
    # - "baseD": sequences are tlozD_1 -> tlozD_2 -> ... (various quests within same base difficulty)
    "sequence_group": "quest",

    # Train/Test split rule
    # If group="quest": split by baseD parity (odd train / even test) across all rows
    # If group="baseD": split by quest parity (odd train / even test)
    "train_on_odd": True,  # odd go to train, even to test (switch False to invert)

    # Model hparams
    "d_model": 128,
    "nhead": 4,
    "nlayers": 3,
    "ff_mult": 4,
    "dropout": 0.1,
    "max_epochs": 120,
    "batch_size": 16,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "patience": 12,          # early stop patience (epochs without val improvement)
    "val_frac": 0.15,        # fraction of TRAIN to hold out as VAL (sequence-wise)
    "seed": 42,
}

# =========================
# UTILITIES
# =========================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def ensure_out(p: str|Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def standardize_train_test(Xtr_list: List[np.ndarray],
                           Xte_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """
    Fit a StandardScaler on TRAIN (concatenated across time) and apply to both splits.
    Returns standardized lists and the feature column names (provided by caller).
    """
    # concatenate over rows (timepoints)
    Xtr_all = np.concatenate(Xtr_list, axis=0) if Xtr_list else np.zeros((0,0), dtype=np.float32)
    scaler = StandardScaler(with_mean=True, with_std=True)
    if Xtr_all.shape[0] > 0:
        scaler.fit(Xtr_all)
    Xtr_std = [scaler.transform(X) if X.size else X for X in Xtr_list]
    Xte_std = [scaler.transform(X) if X.size else X for X in Xte_list]
    return Xtr_std, Xte_std

# =========================
# DATA LOADING
# =========================
META_COLS = ["key", "baseD", "quest", "H", "W"]

def load_feature_table(path: str|Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic sanity
    need = set(META_COLS)
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {missing}")
    return df

def pick_feature_columns(df: pd.DataFrame) -> List[str]:
    # Use every numeric feature excluding metadata + target
    ignore = set(META_COLS)
    feat_cols = [c for c in df.columns if c not in ignore]
    # Enforce numeric only
    feat_cols = [c for c in feat_cols if np.issubdtype(df[c].dtype, np.number)]
    if not feat_cols:
        raise SystemExit("No numeric feature columns found.")
    return feat_cols

def build_sequences(df: pd.DataFrame, feat_cols: List[str], group_by: str) -> Tuple:
    """
    Build sequences and split into TRAIN/TEST.
    Returns:
      (train_sequences, test_sequences, train_meta, test_meta)
    where each sequence is a dict with:
      - X: (T, F) features
      - y: (T,) targets (baseD)
      - mask: (T,) booleans (all True — we pad later)
      - meta: list[dict] per time step: {"key","quest","baseD"}
    Split rule:
      If group_by="quest": odd baseD -> train, even baseD -> test (per-time-step)
      If group_by="baseD": odd quest  -> train, even quest  -> test (per-time-step)
    """
    sequences = []
    if group_by == "quest":
        grouper = ["quest"]
        order_by = ["baseD", "key"]
    elif group_by == "baseD":
        grouper = ["baseD"]
        order_by = ["quest", "key"]
    else:
        raise ValueError("sequence_group must be 'quest' or 'baseD'.")

    for _, g in df.sort_values(order_by).groupby(grouper, sort=True):
        X = g[feat_cols].to_numpy(dtype=np.float32)
        y = g["baseD"].to_numpy(dtype=np.float32)  # regress difficulty
        meta = [{"key": g["key"].iloc[i],
                 "quest": int(g["quest"].iloc[i]),
                 "baseD": int(g["baseD"].iloc[i])} for i in range(len(g))]
        sequences.append({"X": X, "y": y, "meta": meta})

    # Split per time-step using parity rule; then re-pack sequences (may become ragged after filtering)
    train_seqs, test_seqs = [], []
    for seq in sequences:
        X, y, meta = seq["X"], seq["y"], seq["meta"]
        if group_by == "quest":
            mask_train = (y.astype(int) % 2 == (1 if PARAMS["train_on_odd"] else 0))
        else:
            q_arr = np.array([m["quest"] for m in meta], dtype=int)
            mask_train = (q_arr % 2 == (1 if PARAMS["train_on_odd"] else 0))
        mask_test = ~mask_train

        # TRAIN slice
        if mask_train.any():
            train_seqs.append({
                "X": X[mask_train], "y": y[mask_train],
                "meta": [meta[i] for i in np.where(mask_train)[0]]
            })
        # TEST slice
        if mask_test.any():
            test_seqs.append({
                "X": X[mask_test], "y": y[mask_test],
                "meta": [meta[i] for i in np.where(mask_test)[0]]
            })

    return train_seqs, test_seqs

def split_train_val(train_seqs: List[Dict], val_frac: float, seed: int = 42):
    if len(train_seqs) <= 1 or val_frac <= 0.0:
        return train_seqs, []
    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_seqs))
    rng.shuffle(idx)
    n_val = max(1, int(round(val_frac * len(train_seqs))))
    val_idx = set(idx[:n_val])
    tr = [train_seqs[i] for i in range(len(train_seqs)) if i not in val_idx]
    va = [train_seqs[i] for i in range(len(train_seqs)) if i in val_idx]
    if len(tr) == 0:  # ensure non-empty
        tr, va = train_seqs, []
    return tr, va

# =========================
# DATASET & COLLATE
# =========================
class SeqDataset(Dataset):
    def __init__(self, seqs: List[Dict]):
        self.seqs = seqs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        s = self.seqs[i]
        X = torch.from_numpy(s["X"])        # (T, F)
        y = torch.from_numpy(s["y"])        # (T,)
        return X, y, s["meta"]

def collate_pad(batch):
    # batch: list of (X,y,meta)
    lens = [b[0].shape[0] for b in batch]
    T_max = max(lens)
    F = batch[0][0].shape[1]
    B = len(batch)

    Xp = torch.zeros(B, T_max, F, dtype=torch.float32)
    yp = torch.zeros(B, T_max, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.bool)  # True = pad (ignored by Transformer)

    metas = []
    for i, (X, y, meta) in enumerate(batch):
        L = X.shape[0]
        Xp[i, :L, :] = X
        yp[i, :L] = y
        mask[i, L:] = True
        metas.append(meta)
    return Xp, yp, mask, metas

# =========================
# MODEL
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x):  # x: (B, T, d)
        T = x.size(1)
        return x + self.pe[:T, :]

class TransformerRegressor(nn.Module):
    def __init__(self, in_dim: int, d_model: int, nhead: int, nlayers: int,
                 ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_mult*d_model,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.pos = PositionalEncoding(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, X, src_key_padding_mask=None):
        # X: (B, T, F)
        Z = self.proj(X)            # (B, T, d)
        Z = self.pos(Z)             # (B, T, d)
        Z = self.enc(Z, src_key_padding_mask=src_key_padding_mask)  # (B, T, d)
        out = self.head(Z).squeeze(-1)   # (B, T)
        return out

# =========================
# TRAINING
# =========================
@torch.no_grad()
def evaluate(model, dl, device="cpu"):
    model.eval()
    all_true, all_pred = [], []
    for X, y, pad_mask, _ in dl:
        X = X.to(device); y = y.to(device)
        pad_mask = pad_mask.to(device)
        pred = model(X, src_key_padding_mask=pad_mask)
        # collect only non-pad positions
        keep = ~pad_mask
        all_true.append(y[keep].detach().cpu().numpy())
        all_pred.append(pred[keep].detach().cpu().numpy())
    if not all_true:
        return {"mae": None, "rmse": None, "r2": None}, (np.array([]), np.array([]))
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    mae = float(mean_absolute_error(y_true, y_pred)) if y_true.size else None
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2))) if y_true.size else None
    r2 = float(r2_score(y_true, y_pred)) if (y_true.size and len(np.unique(y_true)) > 1) else None
    return {"mae": mae, "rmse": rmse, "r2": r2}, (y_true, y_pred)

def train_model(model, dl_train, dl_val, device="cpu"):
    opt = torch.optim.AdamW(model.parameters(), lr=PARAMS["lr"], weight_decay=PARAMS["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=4
    )

    best_val = float("inf"); best_state = None; patience = PARAMS["patience"]; no_improve = 0

    for epoch in range(1, PARAMS["max_epochs"]+1):
        model.train()
        epoch_losses = []
        for X, y, pad_mask, _ in dl_train:
            X = X.to(device); y = y.to(device)
            pad_mask = pad_mask.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(X, src_key_padding_mask=pad_mask)     # (B,T)
            keep = ~pad_mask
            loss = F.mse_loss(pred[keep], y[keep])
            loss.backward(); opt.step()
            epoch_losses.append(loss.item())
        tr_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        # validation
        if dl_val is not None:
            model.eval()
            with torch.no_grad():
                v_losses = []
                for X, y, pad_mask, _ in dl_val:
                    X = X.to(device); y = y.to(device); pad_mask = pad_mask.to(device)
                    pred = model(X, src_key_padding_mask=pad_mask)
                    keep = ~pad_mask
                    v_loss = F.mse_loss(pred[keep], y[keep]).item()
                    v_losses.append(v_loss)
                va_loss = float(np.mean(v_losses)) if v_losses else tr_loss
            scheduler.step(va_loss)

            if va_loss + 1e-8 < best_val:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        else:
            # no validation; keep last
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    set_seed(PARAMS["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ensure_out(PARAMS["out_dir"])

    # ---- Load table & choose features
    df = load_feature_table(PARAMS["features_csv"])
    feat_cols = pick_feature_columns(df)
    (out_dir / "feature_columns.json").write_text(json.dumps(feat_cols, indent=2), encoding="utf-8")

    # ---- Build sequences and split
    train_seqs, test_seqs = build_sequences(df, feat_cols, group_by=PARAMS["sequence_group"])
    if len(train_seqs) == 0:
        raise SystemExit("Train split ended up empty — adjust split rule.")
    # Standardize features based on TRAIN only
    Xtr_list = [s["X"] for s in train_seqs]
    Xte_list = [s["X"] for s in test_seqs] if test_seqs else []
    Xtr_std, Xte_std = standardize_train_test(Xtr_list, Xte_list)
    for s, Xs in zip(train_seqs, Xtr_std): s["X"] = Xs
    for s, Xs in zip(test_seqs, Xte_std):  s["X"] = Xs

    # ---- Train/Val split (sequence-wise)
    tr_seqs, va_seqs = split_train_val(train_seqs, PARAMS["val_frac"], PARAMS["seed"])

    # ---- DataLoaders
    dl_tr = DataLoader(SeqDataset(tr_seqs), batch_size=PARAMS["batch_size"], shuffle=True, collate_fn=collate_pad)
    dl_va = DataLoader(SeqDataset(va_seqs), batch_size=PARAMS["batch_size"], shuffle=False, collate_fn=collate_pad) if va_seqs else None
    dl_te = DataLoader(SeqDataset(test_seqs), batch_size=PARAMS["batch_size"], shuffle=False, collate_fn=collate_pad) if test_seqs else None

    in_dim = len(feat_cols)
    model = TransformerRegressor(
        in_dim=in_dim,
        d_model=PARAMS["d_model"],
        nhead=PARAMS["nhead"],
        nlayers=PARAMS["nlayers"],
        ff_mult=PARAMS["ff_mult"],
        dropout=PARAMS["dropout"]
    ).to(device)

    # ---- Train
    train_model(model, dl_tr, dl_va, device=device)

    # ---- Evaluate
    tr_metrics, (ytr_true, ytr_pred) = evaluate(model, dl_tr, device=device)
    if dl_va is not None:
        va_metrics, _ = evaluate(model, dl_va, device=device)
    else:
        va_metrics = {"mae": None, "rmse": None, "r2": None}
    if dl_te is not None:
        te_metrics, (yte_true, yte_pred) = evaluate(model, dl_te, device=device)
    else:
        te_metrics, (yte_true, yte_pred) = ({"mae": None, "rmse": None, "r2": None}, (np.array([]), np.array([])))

    metrics = {
        "train": tr_metrics,
        "val": va_metrics,
        "test": te_metrics,
        "sequence_group": PARAMS["sequence_group"],
        "split_rule": f"{'odd' if PARAMS['train_on_odd'] else 'even'} -> train"
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

    # ---- Save per-time-step predictions (TEST only)
    if dl_te is not None:
        rows = []
        model.eval()
        with torch.no_grad():
            for X, y, pad_mask, metas in dl_te:
                X = X.to(device); pad_mask = pad_mask.to(device)
                pred = model(X, src_key_padding_mask=pad_mask).cpu().numpy()  # (B,T)
                y = y.cpu().numpy()
                pm = pad_mask.cpu().numpy()
                for b in range(X.shape[0]):
                    keep_T = (~pm[b]).astype(bool)
                    yb = y[b, keep_T]
                    pb = pred[b, keep_T]
                    for i, m in enumerate(np.array(metas[b])[np.where(keep_T)[0]]):
                        rows.append({
                            "key": m["key"],
                            "quest": int(m["quest"]),
                            "baseD_true": int(m["baseD"]),
                            "baseD_pred": float(pb[i])
                        })
        pd.DataFrame(rows).to_csv(out_dir / "transformer_preds.csv", index=False)

        # ---- Plot pred vs true
        if yte_true.size and yte_pred.size:
            plt.figure(figsize=(5,4))
            plt.scatter(yte_true, yte_pred, s=18, alpha=0.9)
            lo = min(float(np.min(yte_true)), float(np.min(yte_pred))) - 0.5
            hi = max(float(np.max(yte_true)), float(np.max(yte_pred))) + 0.5
            plt.plot([lo,hi], [lo,hi])
            plt.xlim([lo,hi]); plt.ylim([lo,hi])
            plt.xlabel("True difficulty (baseD)")
            plt.ylabel("Predicted difficulty")
            plt.title(f"TEST R² = {metrics['test']['r2']:.3f}" if metrics['test']['r2'] is not None else "Pred vs True")
            plt.tight_layout()
            plt.savefig(out_dir / "pred_vs_true.png", dpi=160)
            plt.close()

    print("Done. Artifacts in:", out_dir.resolve())
