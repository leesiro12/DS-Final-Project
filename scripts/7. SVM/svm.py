#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVR regression using engineered map features (features_all_maps.csv)
-------------------------------------------------------------------
Input  : ./map_features/features_all_maps.csv (from your feature engineering step)
Target : difficulty label = baseD + 0.5 if quest == 2 else baseD
Split  : Train = odd baseD, Test = even baseD

Artifacts (OUT_DIR):
  - svm_model.joblib
  - X_train.csv, y_train.csv, X_test.csv, y_test.csv
  - train_predictions.csv, test_predictions.csv (if any)
  - pred_vs_true_train.png, pred_vs_true_test.png (if any)
  - cv_results.csv (GridSearchCV on TRAIN)
  - metrics.json (train/test metrics & correlations)
  - summary.txt (run log)
"""

from __future__ import annotations
import json, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, median_absolute_error,
    explained_variance_score, max_error
)
from scipy.stats import spearmanr, kendalltau
import joblib

# =========================
# CONFIG
# =========================
CONFIG = {
    "FEATURES_CSV": "../Features/features_all_maps.csv",  # <-- your engineered features file
    "OUT_DIR": "../SVM_Outputs",
    "VERBOSE": True,
    "SVM_PARAMS": {
        "svr__C": [1, 10, 100],
        "svr__gamma": ["scale", 0.1, 0.01],
        "svr__epsilon": [0.01, 0.1, 1.0],
        "svr__kernel": ["rbf"],  # keep rbf; add "linear","poly" if you want
    },
    "GRID_CV": 3,   # folds on TRAIN; will auto-downgrade if too few samples
    "SEED": 42,
}

# =========================
# Logging
# =========================
_log_lines: List[str] = []
def log(msg: str) -> None:
    _log_lines.append(msg)
    if CONFIG.get("VERBOSE", True):
        print(msg)

# =========================
# Correlations & metrics
# =========================
def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float('nan')
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    X = np.abs(x[:, None] - x[None, :])
    Y = np.abs(y[:, None] - y[None, :])
    Xr = X - X.mean(axis=1, keepdims=True) - X.mean(axis=0, keepdims=True) + X.mean()
    Yr = Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + Y.mean()
    dcov2_xy = (Xr * Yr).mean()
    dcov2_xx = (Xr * Xr).mean()
    dcov2_yy = (Yr * Yr).mean()
    if dcov2_xx <= 0 or dcov2_yy <= 0:
        return 0.0
    return float(np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy)))

def concordance_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float('nan')
    mu_x = float(np.mean(y_true)); mu_y = float(np.mean(y_pred))
    vx = float(np.var(y_true, ddof=1)); vy = float(np.var(y_pred, ddof=1))
    cov_xy = float(np.cov(y_true, y_pred, ddof=1)[0, 1])
    denom = vx + vy + (mu_x - mu_y) ** 2
    return float((2 * cov_xy) / denom) if denom > 0 else float('nan')

def correlation_suite(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    sp_r, _ = spearmanr(y_true, y_pred)
    kd_t, _ = kendalltau(y_true, y_pred)
    return {
        "pearson_r": pearson_r(y_true, y_pred),
        "spearman_rho": float(sp_r),
        "kendall_tau": float(kd_t),
        "distance_correlation": distance_correlation(y_true, y_pred),
        "concordance_ccc": concordance_ccc(y_true, y_pred),
    }

def mape_safe(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom))) * 100.0

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "median_ae": float(median_absolute_error(y_true, y_pred)),
        "mape_pct": mape_safe(y_true, y_pred),
        "max_error": float(max_error(y_true, y_pred)),
    }

def save_pred_true_plot(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred)
    lo = float(min(np.min(y_true), np.min(y_pred))) - 0.25
    hi = float(max(np.max(y_true), np.max(y_pred))) + 0.25
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True Difficulty")
    plt.ylabel("Predicted Difficulty")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

# =========================
# Data loading & split
# =========================
def load_features_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Features file not found: {path}")
    df = pd.read_csv(path)
    required = {"key", "baseD", "quest"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in features CSV: {missing}")
    # build label: baseD + 0.5 if quest==2
    df["label"] = df["baseD"].astype(float) + 0.5 * (df["quest"].astype(int) == 2).astype(float)
    # index by key for traceability
    if "key" in df.columns:
        df = df.set_index("key")
    return df

def select_feature_columns(df: pd.DataFrame) -> List[str]:
    # Use all numeric columns except these:
    drop_cols = {"label", "baseD", "quest"}
    # keep H/W as features (map size), so don't drop them
    feat_cols = [c for c in df.columns if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]
    if not feat_cols:
        raise SystemExit("No numeric feature columns found after excluding label/baseD/quest.")
    return feat_cols

def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    # Train = odd baseD, Test = even baseD
    train = df[df["baseD"] % 2 == 1].copy()
    test  = df[df["baseD"] % 2 == 0].copy()
    if train.empty:
        log("WARNING: Train split (odd baseD) is empty.")
    if test.empty:
        log("WARNING: Test split (even baseD) is empty.")

    feat_cols = select_feature_columns(df)
    X_train = train[feat_cols]
    y_train = train["label"].astype(float)
    X_test  = test[feat_cols]
    y_test  = test["label"].astype(float)
    return X_train, y_train, X_test, y_test

# =========================
# Training
# =========================
def fit_svr(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV | Pipeline:
    # If too few samples for CV, fit a default pipeline
    n = len(y_train)
    if n < 2:
        raise SystemExit(f"Too few training samples (n={n}).")
    cv = max(2, min(CONFIG["GRID_CV"], n))  # at least 2, at most n
    if cv < 2:
        # fallback (shouldn't happen given guard above)
        pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
        pipe.fit(X_train.values, y_train.values)
        return pipe

    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
    gscv = GridSearchCV(
        estimator=pipe,
        param_grid=CONFIG["SVM_PARAMS"],
        scoring="r2",
        cv=cv,
        n_jobs=None,
        refit=True,
        return_train_score=True,
    )
    log(f"Running GridSearchCV on TRAIN (cv={cv}) with params: {CONFIG['SVM_PARAMS']}")
    gscv.fit(X_train.values, y_train.values)
    log(f"Best params: {getattr(gscv, 'best_params_', {})}")
    return gscv

# =========================
# Main
# =========================
def main():
    out_dir = Path(CONFIG["OUT_DIR"]); out_dir.mkdir(parents=True, exist_ok=True)
    feats_path = Path(CONFIG["FEATURES_CSV"])
    log(f"Loading engineered features from: {feats_path}")
    df = load_features_table(feats_path)

    if df.empty:
        log("Empty features table — nothing to do.")
        (out_dir / "summary.txt").write_text("\n".join(_log_lines), encoding="utf-8")
        return

    X_train, y_train, X_test, y_test = split_train_test(df)

    # Persist splits
    X_train.to_csv(out_dir / "X_train.csv")
    y_train.to_csv(out_dir / "y_train.csv")
    X_test.to_csv(out_dir / "X_test.csv")
    y_test.to_csv(out_dir / "y_test.csv")

    # Train
    model_or_cv = fit_svr(X_train, y_train)
    if isinstance(model_or_cv, GridSearchCV):
        best_model: Pipeline = model_or_cv.best_estimator_
        pd.DataFrame(model_or_cv.cv_results_).to_csv(out_dir / "cv_results.csv", index=False)
        best_params = model_or_cv.best_params_
    else:
        best_model = model_or_cv
        best_params = {"svr__kernel": "rbf"}
    joblib.dump(best_model, out_dir / "svm_model.joblib")
    log("Saved model: svm_model.joblib")

    # Predictions
    yhat_tr = best_model.predict(X_train.values)
    pd.DataFrame({"key": X_train.index, "y_true": y_train.values, "y_pred": yhat_tr}) \
        .set_index("key").to_csv(out_dir / "train_predictions.csv")
    log("Saved train_predictions.csv")

    has_test = len(y_test) > 0
    if has_test:
        yhat_te = best_model.predict(X_test.values)
        pd.DataFrame({"key": X_test.index, "y_true": y_test.values, "y_pred": yhat_te}) \
            .set_index("key").to_csv(out_dir / "test_predictions.csv")
        log("Saved test_predictions.csv")
    else:
        yhat_te = np.array([])

    # Metrics
    train_metrics = {
        **compute_metrics(y_train.values, yhat_tr),
        **correlation_suite(y_train.values, yhat_tr),
    }
    if has_test:
        test_metrics = {
            **compute_metrics(y_test.values, yhat_te),
            **correlation_suite(y_test.values, yhat_te),
        }
    else:
        test_metrics = None

    all_metrics = {
        "n_samples_total": int(len(df)),
        "n_features": int(X_train.shape[1]),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "best_params": best_params,
        "train": train_metrics,
        "test": test_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    log("Saved metrics.json")

    # Plots
    save_pred_true_plot(y_train.values, yhat_tr, out_dir / "pred_vs_true_train.png", "Pred vs True (Train)")
    log("Saved pred_vs_true_train.png")
    if has_test:
        save_pred_true_plot(y_test.values, yhat_te, out_dir / "pred_vs_true_test.png", "Pred vs True (Test)")
        log("Saved pred_vs_true_test.png")

    # Summary
    log("--- SUMMARY ---")
    log(f"Total samples: {len(df)} | Features: {int(X_train.shape[1])}")
    log(f"Train size (odd baseD): {len(y_train)} | Test size (even baseD): {len(y_test)}")
    log(f"Best SVR params: {best_params}")
    def _kv_line(d): return ", ".join([f"{k}={v:.4f}" if isinstance(v, float) and not math.isnan(v) else f"{k}={v}" for k,v in d.items()])
    log("Train metrics: " + _kv_line(train_metrics))
    if has_test:
        log("Test  metrics: " + _kv_line(test_metrics))
    else:
        log("Test  metrics: n/a (no even baseD rows)")

    (out_dir / "summary.txt").write_text("\n".join(_log_lines), encoding="utf-8")
    log(f"Saved summary.txt to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
