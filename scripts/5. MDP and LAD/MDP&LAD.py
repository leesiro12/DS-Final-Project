#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LAD Pathline Visualizer (Left→Right stitched path)
--------------------------------------------------
Creates one continuous horizontal visualization:
Each room visited is appended rightwards,
so the full journey appears as a single flow.

Door→Door Incremental MDP:
- For every room-segment, we reset the probability field.
- Entry door tile = 0, Exit door tile = 1 (smooth ramp via geodesic distances).

Outputs:
  - lad_pathline_<map>.png         (one long path)
  - prob_matrix_<map>_segXX.csv    (per room-segment probability)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from collections import deque
import heapq

# ===============================================================
PARAMS = {
    "LEGEND_BASE_PATH": Path("../legend_all_tiles.csv"),
    "LEGEND_OTHER_PATH": Path("../legend_other.csv"),   # not used here but kept for consistency
    "SHORTEST_PATH_DIR": Path("../Paths"),
    "OUT_DIR": Path("../MDP_LAD_Output"),

    "TILE_W": 16,
    "TILE_H": 16,
    "UPSCALE": 2.0,
    "COLORMAP": cv2.COLORMAP_TURBO,
    "HEATMAP_STRENGTH": 0.25,
    "PATH_COLOR": (255, 255, 255),
    "PATH_THICKNESS": 3,

    # LAD blend
    "LAD_K": 1.0,
    "LAD_B": 1,
    "LAD_BLEND": 0.5,

    # Wall decay
    "WALL_DECAY_RADIUS": 5.0,
    "WALL_DECAY_STRENGTH": 2.0,

    "SAVE_SEGMENT_CSVS": True,
    "VERBOSE": True
}
# ===============================================================

def log(msg):
    if PARAMS["VERBOSE"]:
        print(msg)

def _is_solid(row):
    return str(row.get("manual_override", "")).strip().lower() in ("solid", "blocked", "wall", "lava")

def build_grid(df):
    H = int(df["row"].max()) + 1
    W = int(df["col"].max()) + 1
    walk = np.zeros((H, W), bool)
    exits = np.zeros((H, W), bool)  # tiles with weights==1 (doors/starts)
    solids = np.zeros((H, W), bool)
    for _, r in df.iterrows():
        rr, cc = int(r["row"]), int(r["col"])
        if _is_solid(r):
            solids[rr, cc] = True
            continue
        walk[rr, cc] = True
        w = float(r.get("weights", r.get("weight", 0)) or 0)
        if w == 1:
            exits[rr, cc] = True
    return H, W, walk, exits, solids

def connected_components(walk):
    H, W = walk.shape
    labels = -np.ones((H, W), int)
    cid = 0
    for r in range(H):
        for c in range(W):
            if not walk[r, c] or labels[r, c] != -1:
                continue
            q = deque([(r, c)])
            labels[r, c] = cid
            while q:
                rr, cc = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and walk[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = cid
                        q.append((nr, nc))
            cid += 1
    return labels, cid

def load_path(map_id, H, W):
    p = PARAMS["SHORTEST_PATH_DIR"] / f"shortest_path_{map_id}.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    return [(int(r), int(c)) for r, c in zip(df["row"], df["col"]) if 0 <= r < H and 0 <= c < W]

# ===================== Door→Door incremental MDP ======================
def bfs_distance(walk, mask, start):
    """4-neighbour BFS distance restricted to walk & mask."""
    H, W = walk.shape
    dist = np.full((H, W), np.inf)
    if not (0 <= start[0] < H and 0 <= start[1] < W) or not (mask[start] and walk[start]):
        return dist
    dq = deque([start])
    dist[start] = 0
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and walk[nr, nc]:
                nd = dist[r, c] + 1
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    dq.append((nr, nc))
    return dist

def door_to_door_prob(walk, mask, entry, exit_):
    """
    Reset probability inside this room (mask) and ramp from:
      prob(entry) = 0  →  prob(exit) = 1
    Using harmonic-like blend: p = d_entry / (d_entry + d_exit)
    """
    d_entry = bfs_distance(walk, mask, entry)
    d_exit  = bfs_distance(walk, mask, exit_)

    prob = np.zeros_like(d_entry, dtype=float)
    valid = np.isfinite(d_entry) & np.isfinite(d_exit) & mask
    denom = d_entry + d_exit
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(valid, d_entry / np.clip(denom, 1e-9, None), 0.0)

    # Force exact anchors
    if valid.any():
        prob[valid] = p[valid]
        prob[entry] = 0.0
        prob[exit_] = 1.0
    prob[~mask] = 0.0
    return prob
# =====================================================================

def lad_dispersion(prob, mask, path_pts, k, b, solids):
    H, W = prob.shape
    out = np.zeros_like(prob)
    seeds = [p for p in path_pts if 0 <= p[0] < H and 0 <= p[1] < W and mask[p]]
    if not seeds:
        return prob
    dist = np.full((H, W), np.inf)
    q = deque(seeds)
    for s in seeds:
        dist[s] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
                nd = dist[r, c] + 1
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    q.append((nr, nc))
    val = np.isfinite(dist) & mask
    if np.any(val):
        dmax = dist[val].max() + 1e-6
        x = dist[val] / dmax
        lad = np.exp(-(k ** (-b / np.sqrt(x + 1e-6))) * x)
        out[val] = lad
        vmax = out[val].max()
        if vmax > 0:
            out[val] /= vmax
    blend = PARAMS["LAD_BLEND"]
    res = (1 - blend) * prob + blend * out
    res[solids] = 0
    res[~mask] = 0
    return res

def wall_decay(prob, solids, mask):
    H, W = prob.shape
    dist = np.full((H, W), np.inf)
    q = deque([(r, c) for r, c in np.argwhere(solids)])
    for r, c in q:
        dist[r, c] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and mask[nr, nc]:
                nd = dist[r, c] + 1
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    q.append((nr, nc))
    finite = np.isfinite(dist) & mask
    if np.any(finite):
        decay = np.exp(
            -PARAMS["WALL_DECAY_STRENGTH"] * (1 / (np.clip(dist[finite], 1e-9, None)) / PARAMS["WALL_DECAY_RADIUS"])
        )
        prob[finite] *= decay
    prob[solids] = 0
    prob[~mask] = 0
    return prob

def prob_to_heat(prob):
    return cv2.applyColorMap((np.clip(prob, 0, 1) * 255).astype(np.uint8), PARAMS["COLORMAP"])

def upscale(img, H, W):
    s = PARAMS["UPSCALE"]
    tw, th = PARAMS["TILE_W"], PARAMS["TILE_H"]
    return cv2.resize(img, (int(W * tw * s), int(H * th * s)), interpolation=cv2.INTER_CUBIC)

def draw_path(img, path, H, W):
    if not path:
        return img
    s = PARAMS["UPSCALE"]
    tw, th = PARAMS["TILE_W"], PARAMS["TILE_H"]
    t = int(PARAMS["PATH_THICKNESS"] * s)
    for i in range(1, len(path)):
        r0, c0 = path[i - 1]
        r1, c1 = path[i]
        x0, y0 = int((c0 + 0.5) * tw * s), int((r0 + 0.5) * th * s)
        x1, y1 = int((c1 + 0.5) * tw * s), int((r1 + 0.5) * th * s)
        cv2.line(img, (x0, y0), (x1, y1), PARAMS["PATH_COLOR"], t, cv2.LINE_AA)
    return img

# ---------------------------------------------------------------
def process_map(map_id, base_df, out_dir):
    log(f"\n[MAP] {map_id}")
    H, W, walk, exits, solids = build_grid(base_df)
    labels, ncomp = connected_components(walk)
    path = load_path(map_id, H, W)
    if not path:
        log(f"[WARN] No path for {map_id}")
        return

    # Split path by room-component
    segs = []
    cur_lab = labels[path[0]]
    cur = []
    for p in path:
        l = labels[p]
        if l != cur_lab:
            segs.append((cur_lab, cur))
            cur_lab = l
            cur = [p]
        else:
            cur.append(p)
    segs.append((cur_lab, cur))

    room_imgs = []
    for seg_idx, (comp, seg) in enumerate(segs, start=1):
        if comp < 0 or len(seg) == 0:
            continue
        mask = (labels == comp)

        # -------- Door→Door incremental MDP (reset per room) ----------
        entry = seg[0]
        exit_ = seg[-1]
        p0 = door_to_door_prob(walk, mask, entry, exit_)   # 0 at entry, 1 at exit

        # Optional LAD + wall shaping (still confined to this room)
        p1 = lad_dispersion(p0, mask, seg, PARAMS["LAD_K"], PARAMS["LAD_B"], solids)
        p2 = wall_decay(p1, solids, mask)

        # Save per-segment probability CSV if desired
        if PARAMS["SAVE_SEGMENT_CSVS"]:
            seg_csv = out_dir / f"prob_matrix_{map_id}.csv"
            pd.DataFrame(p2).to_csv(seg_csv, index=False)

        heat = upscale(prob_to_heat(p2), H, W)
        panel = draw_path(heat, seg, H, W)
        room_imgs.append(panel)

    if not room_imgs:
        return

    # Stitch horizontally
    target_h = max(p.shape[0] for p in room_imgs)
    resized = [cv2.resize(p, (int(p.shape[1] * target_h / p.shape[0]), target_h)) for p in room_imgs]
    pathline = np.concatenate(resized, axis=1)
    cv2.imwrite(str(out_dir / f"lad_pathline_{map_id}.png"), pathline)
    log(f"[SAVE] lad_pathline_{map_id}.png")

# ---------------------------------------------------------------
def main():
    out = PARAMS["OUT_DIR"]
    out.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(PARAMS["LEGEND_BASE_PATH"])
    base["source_norm"] = base["source"].apply(lambda s: (s or "").split(".")[0])
    for m in sorted(base["source_norm"].unique()):
        sub = base[base["source_norm"] == m]
        try:
            process_map(m, sub, out)
        except Exception as e:
            log(f"[ERROR] {m}: {e}")
    log("[DONE]")

if __name__ == "__main__":
    main()
