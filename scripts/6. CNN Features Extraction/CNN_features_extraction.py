#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Engineer per-map features (one row per tlozD_Q)
Reads:
  ../MDP_LAD_Output/prob_matrix_tloz{D}_{Q}.csv
  ../Paths/shortest_path_tloz{D}_{Q}.csv

Writes:
  ./map_features/features_all_maps.csv
  ./map_features/features_all_maps.json  (same content as CSV)
"""

from __future__ import annotations
import json, math, re, warnings
from pathlib import Path
from typing import Dict, Tuple, List
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
PARAMS = {
    # where your CSVs are
    "prob_dir": "../MDP_LAD_Output",   # prob_matrix_tlozD_Q.csv
    "path_dir": "../Paths",            # shortest_path_tlozD_Q.csv

    # output folder
    "out_root": "../Features",

    # other options
    "coerce_nan_to_zero": True,        # sanitize strange cells to 0
    "neighbor_kernel": 1,              # radius for local stats (3 -> 3x3)
}

# =========================
# CSV READERS (ROBUST)
# =========================
def _read_grid_csv(path: Path) -> np.ndarray:
    """
    Robust 2D numeric CSV reader:
    - tolerates header row/col and comments (#),
    - auto-detects delimiter,
    - coerces non-numeric to NaN then to 0 (if configured).
    """
    try:
        df = pd.read_csv(path, header=None, comment="#", sep=None, engine="python", dtype=str)
    except Exception:
        df = pd.read_csv(path, header=None, comment="#", sep=",", engine="python", dtype=str)
    df = df.applymap(lambda s: s.strip() if isinstance(s, str) else s)
    num = df.apply(pd.to_numeric, errors="coerce")

    def _mostly_nan_row(series, thresh=0.5): return series.isna().mean() >= thresh
    # Drop leading header-like row/col if they’re mostly NaN
    if num.shape[0] and _mostly_nan_row(num.iloc[0, :]): num = num.iloc[1:, :]
    if num.shape[1] and _mostly_nan_row(num.iloc[:, 0]): num = num.iloc[:, 1:]

    if PARAMS["coerce_nan_to_zero"]:
        num = num.fillna(0.0)

    arr = num.values.astype(np.float32)
    if arr.ndim != 2 or min(arr.shape) < 2:
        raise ValueError(f"{path}: expected 2D grid, got shape {arr.shape}")
    return arr

def _read_shortest_path_as_grid(path: Path, H: int, W: int) -> np.ndarray:
    """
    Reads a shortest_path CSV that may be:
      1) an HxW numeric grid (binarized >0),
      2) a 2+ column list of (row, col, ...).
    Returns (H, W) float32 mask in {0,1}.
    """
    # Try grid
    try:
        g = _read_grid_csv(path)
        if g.shape == (H, W):
            return (g > 0).astype(np.float32)
    except Exception:
        pass

    # Try coordinates list
    try:
        df = pd.read_csv(path, header=None, comment="#", sep=None, engine="python", dtype=str)
    except Exception:
        df = pd.read_csv(path, header=None, comment="#", sep=",", engine="python", dtype=str)
    df = df.applymap(lambda s: s.strip() if isinstance(s, str) else s)
    num = df.apply(pd.to_numeric, errors="coerce")
    cand = num.dropna(axis=1, how="all")
    if cand.shape[1] < 2:
        raise ValueError(f"{path.name}: cannot interpret as coordinates; need 2+ numeric cols")

    cols = list(cand.columns)
    best = None
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i == j: 
                continue
            rvals = cand[cols[i]].to_numpy()
            cvals = cand[cols[j]].to_numpy()
            for off_r in (0, -1):
                for off_c in (0, -1):
                    rr = np.floor(rvals + off_r).astype(np.int64)
                    cc = np.floor(cvals + off_c).astype(np.int64)
                    ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                    hits = int(ok.sum())
                    if best is None or hits > best[0]:
                        best = (hits, cols[i], cols[j], off_r, off_c)
    if best is None or best[0] == 0:
        raise ValueError(f"{path.name}: no coordinate mapping fits {H}x{W}")

    _, rcol, ccol, off_r, off_c = best
    rr = np.floor(cand[rcol].to_numpy() + off_r).astype(np.int64)
    cc = np.floor(cand[ccol].to_numpy() + off_c).astype(np.int64)
    ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    rr = rr[ok]; cc = cc[ok]

    mask = np.zeros((H, W), dtype=np.float32)
    if rr.size: 
        mask[rr, cc] = 1.0
    return mask

# =========================
# FEATURE HELPERS
# =========================
def entropy_from_prob(prob: np.ndarray, eps=1e-9) -> float:
    p = prob.clip(min=0.0)
    s = p.sum()
    if s <= eps: 
        return 0.0
    p = p / s
    ent = -(p * np.log(p + eps)).sum()
    return float(ent)

def gini_from_prob(prob: np.ndarray, eps=1e-9) -> float:
    """Gini of flattened prob distribution (after normalizing to sum=1)."""
    p = prob.clip(min=0.0).astype(np.float64)
    s = p.sum()
    if s <= eps: 
        return 0.0
    p = p / s
    p = np.sort(p.reshape(-1))
    n = p.size
    cum = np.cumsum(p)
    gini = 1.0 - 2.0 * np.sum(cum) / (n * np.sum(p)) + 1.0 / n
    return float(gini)

def neighbors_count(mask: np.ndarray, r: int, c: int) -> int:
    H, W = mask.shape
    cnt = 0
    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
        rr, cc = r+dr, c+dc
        if 0 <= rr < H and 0 <= cc < W and mask[rr,cc] > 0.5:
            cnt += 1
    return cnt

def path_components(mask: np.ndarray) -> int:
    """4-neighbor connected components on path cells (>0.5)."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = 0
    for r in range(H):
        for c in range(W):
            if mask[r,c] > 0.5 and not seen[r,c]:
                comps += 1
                # BFS
                q = [(r,c)]
                seen[r,c] = True
                while q:
                    rr, cc = q.pop()
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        r2, c2 = rr+dr, cc+dc
                        if 0 <= r2 < H and 0 <= c2 < W and mask[r2,c2] > 0.5 and not seen[r2,c2]:
                            seen[r2,c2] = True
                            q.append((r2,c2))
    return comps

def path_graph_stats(mask: np.ndarray):
    """Degree histogram & end-to-end Manhattan distance using detected endpoints."""
    H, W = mask.shape
    deg1 = deg2 = deg3p = 0
    ends = []
    for r in range(H):
        for c in range(W):
            if mask[r,c] > 0.5:
                d = neighbors_count(mask, r, c)
                if d == 1:
                    deg1 += 1; ends.append((r,c))
                elif d == 2:
                    deg2 += 1
                elif d >= 3:
                    deg3p += 1
    manhattan = -1
    if len(ends) >= 2:
        (r1,c1),(r2,c2) = ends[0], ends[-1]
        manhattan = abs(r1-r2) + abs(c1-c2)
    return deg1, deg2, deg3p, manhattan

def local_mean(arr: np.ndarray, k: int = 3) -> np.ndarray:
    """Box filter (k x k) with replicate padding."""
    assert k % 2 == 1 and k >= 1
    pad = k // 2
    a = np.pad(arr, ((pad,pad),(pad,pad)), mode="edge")
    out = np.zeros_like(arr, dtype=np.float32)
    H, W = arr.shape
    # separable box
    row = np.cumsum(a, axis=1, dtype=np.float64)
    row[:, k:] = row[:, k:] - row[:, :-k]
    row = row[:, k-1:]
    col = np.cumsum(row, axis=0, dtype=np.float64)
    col[k:, :] = col[k:, :] - col[:-k, :]
    col = col[k-1:, :]
    out[:] = (col / (k*k)).astype(np.float32)
    return out

def extract_ordered_path(mask: np.ndarray) -> List[Tuple[int,int]]:
    """
    Try to recover an ordered path by walking from an endpoint through degree-2 nodes.
    If multiple components/branches, returns the longest simple walk found.
    """
    H, W = mask.shape
    coords = np.argwhere(mask > 0.5)
    if coords.size == 0:
        return []

    # Find endpoints in all components
    ends = []
    for (r,c) in coords:
        if neighbors_count(mask, r, c) == 1:
            ends.append((r,c))

    # Candidate starts: endpoints (prefer), otherwise any path cell
    starts = ends if ends else [tuple(x) for x in coords.tolist()]

    best_path = []

    for s in starts:
        # simple walk that doesn't revisit nodes
        path = [s]
        seen = set([s])
        cur = s
        while True:
            nbrs = []
            r, c = cur
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                rr, cc = r+dr, c+dc
                if 0 <= rr < H and 0 <= cc < W and mask[rr,cc] > 0.5 and (rr,cc) not in seen:
                    nbrs.append((rr,cc))
            if len(nbrs) == 0:
                break
            # pick one deterministically (e.g., first)
            nxt = nbrs[0]
            path.append(nxt); seen.add(nxt); cur = nxt
        if len(path) > len(best_path):
            best_path = path

    return best_path

# =========================
# ENGINEERED FEATURES (PER MAP)
# =========================
def engineered_features(prob: np.ndarray, sp: np.ndarray, key: str, baseD: int, quest: int,
                        k_neigh: int = 3) -> Dict[str, float]:
    # Normalize prob into [0,1] if outside
    if prob.max() > 1.0 or prob.min() < 0.0:
        pmin, pmax = float(prob.min()), float(prob.max())
        prob = (prob - pmin) / (pmax - pmin) if pmax > pmin else np.zeros_like(prob, dtype=np.float32)

    H, W = prob.shape
    feat: Dict[str, float] = {}
    feat["key"] = key
    feat["baseD"] = int(baseD)
    feat["quest"] = int(quest)
    feat["H"] = int(H)
    feat["W"] = int(W)

    # Global prob stats
    flat = prob.reshape(-1)
    feat["prob_mean"] = float(flat.mean())
    feat["prob_std"]  = float(flat.std())
    feat["prob_min"]  = float(flat.min())
    feat["prob_max"]  = float(flat.max())
    for q in (5,10,25,50,75,90,95):
        feat[f"prob_p{q}"] = float(np.percentile(flat, q))
    feat["prob_entropy"] = entropy_from_prob(prob)
    feat["prob_gini"]    = gini_from_prob(prob)

    # Path stats
    plen = float(sp.sum())
    feat["path_len"] = plen
    feat["path_cov"] = float(plen / (H*W))
    deg1, deg2, deg3p, manh = path_graph_stats(sp)
    feat["path_deg1"] = float(deg1)
    feat["path_deg2"] = float(deg2)
    feat["path_deg3p"] = float(deg3p)
    feat["path_components"] = float(path_components(sp))
    feat["path_end_manhattan"] = float(manh)

    # Prob along path and neighborhood
    if plen > 0:
        on_mask = sp > 0.5
        feat["prob_on_path_mean"] = float((prob * on_mask).sum() / plen)
        feat["prob_on_path_std"]  = float(prob[on_mask].std())

        # local mean of prob (k x k) and then sample on path
        k = int(k_neigh)
        lmean = local_mean(prob, k=k)
        feat["prob_local_on_path_mean"] = float(lmean[on_mask].mean()) if on_mask.any() else 0.0

        # path ordering & curvature proxy (turn-rate)
        ordered = extract_ordered_path(sp)
        turns = 0
        if len(ordered) >= 3:
            # compare successive direction vectors
            def dir_vec(a, b):
                return (np.sign(b[0]-a[0]), np.sign(b[1]-a[1]))
            prev = dir_vec(ordered[0], ordered[1])
            for i in range(1, len(ordered)-1):
                cur = dir_vec(ordered[i], ordered[i+1])
                if cur != prev:
                    turns += 1
                prev = cur
        feat["path_turns"] = float(turns)
        feat["path_turn_rate"] = float(turns / max(1.0, len(ordered))) if ordered else 0.0

        # cumulative prob along ordered path (AUC-ish)
        if ordered:
            vals = [float(prob[r,c]) for (r,c) in ordered]
            feat["path_prob_sum"] = float(np.sum(vals))
            feat["path_prob_mean_ordered"] = float(np.mean(vals))
        else:
            feat["path_prob_sum"] = float((prob * on_mask).sum())
            feat["path_prob_mean_ordered"] = feat["prob_on_path_mean"]
    else:
        feat["prob_on_path_mean"] = 0.0
        feat["prob_on_path_std"]  = 0.0
        feat["prob_local_on_path_mean"] = 0.0
        feat["path_turns"] = 0.0
        feat["path_turn_rate"] = 0.0
        feat["path_prob_sum"] = 0.0
        feat["path_prob_mean_ordered"] = 0.0

    feat["prob_path_minus_global"] = feat["prob_on_path_mean"] - feat["prob_mean"]
    return feat

# =========================
# DISCOVERY & MAIN
# =========================
def discover_pairs(prob_dir: Path, path_dir: Path):
    """
    Find all (prob_matrix, shortest_path) pairs across two directories.
    prob_matrix_tlozD_Q.csv lives in prob_dir
    shortest_path_tlozD_Q.csv lives in path_dir
    """
    pairs: Dict[str, Tuple[int,int,Path,Path]] = {}
    for p in prob_dir.glob("prob_matrix_tloz*_*.csv"):
        m = re.search(r"prob_matrix_tloz(\d+)_(\d+)\.csv$", p.name, re.IGNORECASE)
        if not m:
            continue
        D = int(m.group(1))
        Q = int(m.group(2))
        key = f"tloz{D}_{Q}"
        sp = path_dir / f"shortest_path_{key}.csv"
        if not sp.exists():
            print(f"[warn] missing shortest_path for {key} in {path_dir} -> skip")
            continue
        pairs[key] = (D, Q, p, sp)
    return pairs

if __name__ == "__main__":
    prob_dir = Path(PARAMS["prob_dir"])
    path_dir = Path(PARAMS["path_dir"])
    out_root = Path(PARAMS["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(prob_dir, path_dir)
    if not pairs:
        raise SystemExit("No valid (prob_matrix, shortest_path) pairs found.")

    rows: List[Dict[str, float]] = []
    shapes = set()
    for key, (D, Q, pprob, psp) in sorted(pairs.items()):
        try:
            prob = _read_grid_csv(pprob)
        except Exception as e:
            print(f"[skip] {key}: prob read error: {e}")
            continue
        H, W = prob.shape
        shapes.add((H, W))
        try:
            sp = _read_shortest_path_as_grid(psp, H, W)
        except Exception as e:
            print(f"[skip] {key}: sp read error: {e}")
            continue

        feats = engineered_features(
            prob, sp, key=key, baseD=D, quest=Q,
            k_neigh=int(PARAMS["neighbor_kernel"])
        )
        rows.append(feats)

    if not rows:
        raise SystemExit("No rows produced — check [skip] messages above.")

    df = pd.DataFrame(rows)
    # Order columns: id/meta first, then features sorted
    meta = ["key", "baseD", "quest", "H", "W"]
    others = [c for c in df.columns if c not in meta]
    df = df[meta + sorted(others)]

    df.to_csv(out_root / "features_all_maps.csv", index=False)
    (out_root / "features_all_maps.json").write_text(
        df.to_json(orient="records", indent=2),
        encoding="utf-8"
    )

    print(f"Maps processed: {len(df)} | Unique shapes: {sorted(shapes)}")
    print("Saved:", (out_root / "features_all_maps.csv").resolve())
