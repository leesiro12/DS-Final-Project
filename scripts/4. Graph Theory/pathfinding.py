#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path planner for Zelda-like tile maps

Pipeline (high-level, per map):
  1) Collectibles first (keys/bombs), nearest-first.
  2) Per-room rule: upon entering a room, clear enemies (category=='enemy'),
     then go to a door in that SAME room. Re-applies after each teleport.
  3) Portals sequence (re-using previously visited portals if enabled).
  4) If no portals, try stairs pairs w/ backtracking; else go to nearest global goal.
  5) Draw overlay (walk path solid, teleports dotted) + optional step labels.
  6) Write CSVs + JSON summary.

Inputs:
  - legend (base) CSV: tile grid, passability & “starts” via weights==1 at edges.
  - legend_other CSV: items, portals, arrows, enemies, ends, etc.
  - tiles_base/, tiles_other/ folders for reconstruction.

Outputs (per map):
  - reconstruct_base_<map>.png
  - reconstruct_others_<map>.png
  - reconstruct_combined_<map>.png
  - overlay_shortest_path_<map>.png
  - shortest_path_<map>.csv         (tile centers along path)
  - shortest_path_summary.json      (rollup for all maps)
  - collectibles_summary.csv        (quick table across maps)
"""

from pathlib import Path
import csv, json
from collections import defaultdict, deque
from itertools import combinations
import numpy as np
import cv2

# PARAMS
LEGEND_BASE_PATH   = Path("../legend_all_tiles.csv")   # <- your base CSV
LEGEND_OTHER_PATH  = Path("../legend_other.csv")       # <- your others CSV
TILES_BASE_DIR     = Path("../Tiles_Base")
TILES_OTHER_DIR    = Path("../Tiles_Other")
OUT_DIR            = Path("../Paths")

# Walkability
TREAT_TRANSPARENT_AS_WALKABLE = True
TREAT_LIQUID_AS_WALKABLE      = True  # tags: liquid/water/pool

# Grid
TILE_W, TILE_H     = 16, 16
ALLOW_DIAGONAL     = False  # BFS/tele-BFS neighbor set

# Fix for specific mis-shifted CSVs (room column downshift)
BAD_SHIFT_FIX = {
    # rows in the last room-column shifted DOWN by +11 rows in CSV -> subtract 11
    "tloz4_2.png": {"room_columns": 4, "shift_rows": 11}
}

# Portals (alphabet indices)
REUSE_VISITED_PORTALS = True
PORTAL_OTHER_INDICES = {
    "86","88","196","420","409","426","412","419",
    "568","578","567","581","565","560","590","591","615","616","669"
}
# One-way (directed) portal pairs: src_idx -> dst_idx
DIRECTED_PORTAL_INDEX_PAIRS = {("432","479")}  # E Green -> E Red

# Goals in legend_other (grid rc inferred from row/col or x/y/cx/cy)
END_OTHER_INDICES = {"13","788","707"}

# Collectibles
COLLECT_OTHER_INDICES = {"25","235","48"}  # keys + bombs
KEY_INDICES  = {"25","235"}
BOMB_INDICES = {"48"}

# Enemies & doors
ENEMY_CATEGORY = "enemy"
DOOR_INDEX_WHITELIST = {
    "92","102","103","105","106",
    "154","196","197","198","199",
    "302","303","308","309","313",
    "410","411","414","415","416",
    "466","515","516","518","519",
    "591","636","637","639","640"
}
BASE_DOOR_INDEX_FIELDS = ("index","other_index","tile_index")

# One-way arrows
ARROW_BY_FIELD_INDICES = {"87", "201", "216", "219", "243", "569", "831", "786"}  # 87 entries should have row["direction"] (up/down/left/right)
DEFAULT_ARROW_DIR      = "left"
ARROW_SURROUND_RADIUS_TILES = 3  # strict local zone around each arrow
ARROW_APPLY_ON_WALKABLE     = True
ARROW_OVERLAP_KEEP_FIRST    = True

# Styling
WALK_COLORS = [
    (50,220,120,255),(80,170,255,255),(255,180,60,255),(200,90,200,255),
    (30,200,255,255),(90,90,255,255),(170,255,90,255),(255,120,200,255),
]
PATH_MAIN_THICKNESS = 3
WALK_THICKNESS      = max(2, PATH_MAIN_THICKNESS)
TELEPORT_COLOR         = (255,140,0,255)
TELEPORT_THICKNESS     = max(2, PATH_MAIN_THICKNESS)
TELEPORT_DOT_PERIOD_PX = 12
PIN_START_COLOR         = (0,200,0,255)
PIN_GOAL_REACHED_COLOR  = (220,0,0,255)
PIN_GOAL_PENDING_COLOR  = (0,150,255,255)
PIN_RADIUS              = 4
NODE_RADIUS             = 2

# Step-count labels (adaptive)
SHOW_TILE_COUNT   = True
TILE_COUNT_OFFSET = (4,-4)
TILE_COUNT_FONT_SCALE = 0.38
TILE_COUNT_THICKNESS  = 1
TILE_COUNT_PILL_PAD   = 1
TILE_COUNT_PILL_ALPHA = 140
LABEL_ADAPTIVE            = True
MIN_LABEL_ARC_TILES       = 4
TURN_LABEL_ANGLE_DEG      = 35
STRAIGHT_LABEL_ARC_FACTOR = 1.6
LABEL_CLEAR_RADIUS_PX     = 10
LABEL_EDGE_MARGIN_PX      = 6
LABEL_MAX_TRIES           = 16
LABEL_NUDGE_PX            = 6

DEBUG_LOGS = True

NEIGH4 = [(1,0),(-1,0),(0,1),(0,-1)]
NEIGH8 = NEIGH4 + [(1,1),(1,-1),(-1,1),(-1,-1)]

DIR_VEC = {"L":(0,-1),"LEFT":(0,-1),"R":(0,1),"RIGHT":(0,1),"U":(-1,0),"UP":(-1,0),"D":(1,0),"DOWN":(1,0)}

# helpers 
def _scan_along_dir(grid_walk, start_rc, dir_rc, max_len, stop_at_block=True):
    H, W = grid_walk.shape
    out = []
    r, c = start_rc
    for _ in range(max(0, int(max_len)) + 1):
        if not (0 <= r < H and 0 <= c < W): break
        if stop_at_block and not grid_walk[r, c]: break
        out.append((r, c))
        r += dir_rc[0]; c += dir_rc[1]
    return out

def _apply_arrow_band(edge_dir_map, grid_walk, center_rc, dir_code,
                      half_width=1, max_len=0, stop_at_block=True, keep_first=True):
    """
    Make a directed band along the arrow direction (and opposite), widening sideways by half_width.
    Use max_len=0 to affect only the row/col of the arrow; >0 to propagate along the corridor.
    """
    H, W = grid_walk.shape
    dr0, dc0 = DIR_VEC[dir_code]
    neg = (-dr0, -dc0)

    rays = []
    rays += _scan_along_dir(grid_walk, center_rc, (dr0, dc0), max_len, stop_at_block)
    rays += _scan_along_dir(grid_walk, center_rc, neg,            max_len, stop_at_block)

    def inb(r,c): return 0 <= r < H and 0 <= c < W

    for (r,c) in rays:
        # widen sideways: for UP/DOWN expand horizontally; for LEFT/RIGHT expand vertically
        band_tiles = []
        if dr0 != 0:   # vertical (UP/DOWN) -> expand left/right
            for dc in range(-half_width, half_width+1):
                rr, cc = r, c+dc
                if inb(rr,cc) and (not ARROW_APPLY_ON_WALKABLE or grid_walk[rr,cc]):
                    band_tiles.append((rr,cc))
        else:          # horizontal (LEFT/RIGHT) -> expand up/down
            for dr in range(-half_width, half_width+1):
                rr, cc = r+dr, c
                if inb(rr,cc) and (not ARROW_APPLY_ON_WALKABLE or grid_walk[rr,cc]):
                    band_tiles.append((rr,cc))

        for (rr, cc) in band_tiles:
            for vr, vc in ((1,0),(-1,0),(0,1),(0,-1)):
                nr, nc = rr+vr, cc+vc
                if not inb(nr, nc): 
                    continue
                if ARROW_APPLY_ON_WALKABLE and not grid_walk[nr, nc]:
                    continue
                _add_dir_constraint(
                    edge_dir_map, (rr,cc), (nr,nc), (dr0,dc0),
                    overwrite=not ARROW_OVERLAP_KEEP_FIRST
                )

def _is_ladder_row(row: dict) -> bool:
    """True if base CSV row represents a ladder tile (manual_override or classification)."""
    s = ((row.get("manual_override") or "") + " " + (row.get("classification") or "")).lower()
    return "ladder" in s

def _collect_ladders_from_base(rows_for_src):
    """Return [(r,c,'LADDER'), ...] for ladder tiles found in base rows."""
    ladders = []
    for r in rows_for_src:
        if r.get("row") and r.get("col") and _is_ladder_row(r):
            rr = int(float(r["row"])); cc = int(float(r["col"]))
            ladders.append((rr, cc, "LADDER"))
    return ladders

def _apply_arrow_plus(edge_dir_map, grid_walk, center_rc, dir_code, keep_first=True):
    """
    Constrain ONLY the center tile and its 4-neighbors (Manhattan distance 1)
    to allow movement strictly in dir_code (e.g., 'U' => (-1,0)).
    """
    H, W = grid_walk.shape
    dr0, dc0 = DIR_VEC[dir_code]
    r0, c0 = center_rc

    def inb(r,c): return 0 <= r < H and 0 <= c < W

    # tiles in the plus: center, up, down, left, right
    plus_tiles = [(r0, c0), (r0-1, c0), (r0+1, c0), (r0, c0-1), (r0, c0+1)]

    for (r, c) in plus_tiles:
        if not inb(r, c): 
            continue
        if ARROW_APPLY_ON_WALKABLE and not grid_walk[r, c]:
            continue
        # Constrain the 4 edges touching (r,c)
        for vr, vc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r + vr, c + vc
            if not inb(nr, nc):
                continue
            if ARROW_APPLY_ON_WALKABLE and not grid_walk[nr, nc]:
                continue
            _add_dir_constraint(
                edge_dir_map,
                (r, c), (nr, nc),
                (dr0, dc0),
                overwrite=not ARROW_OVERLAP_KEEP_FIRST if keep_first else True
            )


def strip_ext(name: str) -> str:
    return name.rsplit(".", 1)[0] if name and "." in name else (name or "")

def name_for(source_key: str) -> str:
    return strip_ext((source_key or "").strip())

def normalize_other_source(name: str) -> str:
    s = strip_ext((name or "").strip())
    return s[:-1] if s.endswith("e") else s

def normalize_base_source(name: str) -> str:
    return strip_ext((name or "").strip())

def read_csv_rows(path: Path):
    rows = []
    with path.open(newline='', encoding='utf-8-sig') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
    return rows

def ensure_rgba(im):
    if im is None: return None
    if im.ndim == 2:  return cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    if im.shape[2] == 3: return cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    if im.shape[2] == 4: return im
    # fallback
    h, w = im.shape[:2]
    out = np.zeros((h, w, 4), dtype=np.uint8)
    out[..., :min(3, im.shape[2])] = im[..., :min(3, im.shape[2])]
    out[..., 3] = 255
    return out

def _normalize_tag(classification: str, manual_override: str) -> str:
    return (manual_override or classification or "").strip().lower()

def is_liquid_tile(classification: str, manual_override: str) -> bool:
    tag = _normalize_tag(classification, manual_override)
    return tag in {"liquid","water","pool"}

def is_walkable_base(classification: str, manual_override: str) -> bool:
    tag = _normalize_tag(classification, manual_override)
    air_like    = {"air","walkable","non-solid","nonsolid","stairs"}
    solid_like  = {"solid","blocked","wall","lava"}
    liquid_like = {"liquid","water","pool"}
    if tag in air_like: return True
    if tag in solid_like: return False
    if TREAT_LIQUID_AS_WALKABLE and tag in liquid_like: return True
    return False

def collect_base_by_source(legend_rows):
    by_src = defaultdict(list)
    for r in legend_rows:
        src = (r.get("source","") or "").strip()
        if src:
            by_src[src].append(r)
    return by_src

# DATA NORMALIZATION
def _fix_legend_bad_shift_inplace(rows_for_src, room_columns=4, shift_rows=11):
    if not rows_for_src:
        return
    max_c = max(int(float(r.get("col", 0) or 0)) for r in rows_for_src)
    W_tiles = max_c + 1
    room_w_tiles = max(1, round(W_tiles / max(1, room_columns)))
    last_start_c = (room_columns - 1) * room_w_tiles
    for r in rows_for_src:
        try:
            c = int(float(r.get("col", 0) or 0))
            if c >= last_start_c:
                rr = int(float(r.get("row", 0) or 0))
                r["row"] = str(rr - shift_rows)
        except Exception:
            continue
    # keep non-negative
    min_row = min(int(float(r.get("row", 0) or 0)) for r in rows_for_src)
    if min_row < 0:
        off = -min_row
        for r in rows_for_src:
            r["row"] = str(int(float(r.get("row", 0) or 0)) + off)

def dims_from_rows(rows_for_src):
    max_r = max(int(float(r.get("row",0))) for r in rows_for_src)
    max_c = max(int(float(r.get("col",0))) for r in rows_for_src)
    return (max_r+1, max_c+1)

# image reconstruction 
def paste_rgba(dst_rgba, src_rgba, x, y):
    """Alpha composite src onto dst at (x,y)."""
    H, W = dst_rgba.shape[:2]
    h, w = src_rgba.shape[:2]
    if x >= W or y >= H or x+w <= 0 or y+h <= 0: 
        return
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x+w); y1 = min(H, y+h)
    sx0 = x0 - x; sy0 = y0 - y
    sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)

    dst_roi = dst_rgba[y0:y1, x0:x1]
    src_roi = src_rgba[sy0:sy1, sx0:sx1]  # <-- fixed slice bug

    if src_roi.ndim == 2:
        src_roi = cv2.cvtColor(src_roi, cv2.COLOR_GRAY2BGRA)
    elif src_roi.shape[2] == 3:
        src_roi = cv2.cvtColor(src_roi, cv2.COLOR_BGR2BGRA)

    src_rgb = src_roi[..., :3].astype(np.float32)
    src_a   = (src_roi[..., 3:4].astype(np.float32)) / 255.0
    dst_rgb = dst_roi[..., :3].astype(np.float32)
    dst_a   = (dst_roi[..., 3:4].astype(np.float32)) / 255.0

    out_a = src_a + dst_a * (1.0 - src_a)
    num   = src_rgb * src_a + dst_rgb * dst_a * (1.0 - src_a)
    denom = np.clip(out_a, 1e-6, 1.0)
    out_rgb = num / denom

    mask = (out_a > 1e-6)
    mask3 = np.repeat(mask, 3, axis=2)
    dst_roi[..., :3] = np.where(mask3, np.clip(out_rgb, 0, 255), dst_rgb).astype(np.uint8)
    dst_roi[..., 3]  = np.clip(out_a * 255.0, 0, 255).astype(np.uint8).squeeze(-1)

def reconstruct_base_png(rows, H, W):
    canvas = np.zeros((H*TILE_H, W*TILE_W, 4), dtype=np.uint8); canvas[:] = (0,0,0,0)
    cache = {}
    for r in rows:
        fn = r.get("filename","")
        if not fn: continue
        if fn not in cache:
            p = TILES_BASE_DIR / fn
            if not p.exists():
                if DEBUG_LOGS: print(f"[BASE] Missing: {p}")
                continue
            im = ensure_rgba(cv2.imread(str(p), cv2.IMREAD_UNCHANGED))
            if im is None:
                if DEBUG_LOGS: print(f"[BASE] Failed read: {p}")
                continue
            if im.shape[1] != TILE_W or im.shape[0] != TILE_H:
                im = cv2.resize(im, (TILE_W, TILE_H), interpolation=cv2.INTER_NEAREST)
            cache[fn] = im
        rr = int(float(r.get("row",0))); cc = int(float(r.get("col",0)))
        y0, x0 = rr*TILE_H, cc*TILE_W
        tile = cache.get(fn)
        if tile is None: continue
        paste_rgba(canvas, tile, x0, y0)
    return canvas

def reconstruct_others_png(other_rows_for_src, canvas_w, canvas_h):
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8); canvas[:] = (0,0,0,0)
    cache = {}
    for r in other_rows_for_src:
        fn = (r.get("filename") or r.get("name") or r.get("tile") or "").strip()
        if not fn: continue
        if fn not in cache:
            p = TILES_OTHER_DIR / fn
            if not p.exists():
                if DEBUG_LOGS: print(f"[OTHERS] Missing: {p}"); continue
            im = ensure_rgba(cv2.imread(str(p), cv2.IMREAD_UNCHANGED))
            if im is None:
                if DEBUG_LOGS: print(f"[OTHERS] Failed read: {p}"); continue
            cache[fn] = im
        im = cache[fn]
        # priority: (row,col) -> (x,y) -> (cx,cy)
        if r.get("row") and r.get("col"):
            rr = int(float(r["row"])); cc = int(float(r["col"]))
            x = int(cc*TILE_W); y = int(rr*TILE_H)
        elif r.get("x") and r.get("y"):
            x = int(float(r["x"])); y = int(float(r["y"]))
        elif r.get("cx") and r.get("cy"):
            cx = float(r["cx"]); cy = float(r["cy"])
            x = int(cx - im.shape[1]/2.0); y = int(cy - im.shape[0]/2.0)
        else:
            if DEBUG_LOGS: print(f"[OTHERS] No placement for {fn}")
            continue
        paste_rgba(canvas, im, x, y)
    return canvas

def compose(base_rgba, others_rgba):
    combined = ensure_rgba(base_rgba.copy())
    paste_rgba(combined, ensure_rgba(others_rgba), 0, 0)
    return combined

# passability & endpoints
def build_passability_presence_and_liquid(rows, H, W):
    grid = np.zeros((H, W), dtype=bool)
    present = np.zeros((H, W), dtype=bool)
    liquid = np.zeros((H, W), dtype=bool)
    for r in rows:
        rr = int(float(r.get("row",0))); cc = int(float(r.get("col",0)))
        cls = r.get("classification",""); mo = r.get("manual_override","")
        grid[rr, cc]   = is_walkable_base(cls, mo)
        present[rr, cc]= True
        liquid[rr, cc] = is_liquid_tile(cls, mo)
    return grid, present, liquid

def find_edge_starts(rows, H, W):
    starts = []
    for r in rows:
        try:
            wt = float(r.get("weights", r.get("weight","0")) or "0")
        except:
            wt = 0.0
        if wt != 1: 
            continue
        rr = int(float(r.get("row",0))); cc = int(float(r.get("col",0)))
        if not is_walkable_base(r.get("classification",""), r.get("manual_override","")):
            continue
        if rr==0 or cc==0 or rr==H-1 or cc==W-1:
            starts.append((rr, cc))
    return starts

def load_ends_from_legend_other(base_source_key_norm, legend_other_rows):
    ends = []
    wanted = {(_ or "").strip() for _ in END_OTHER_INDICES}
    for r in legend_other_rows:
        if normalize_other_source(r.get("source","")) != base_source_key_norm:
            continue
        idx = str(r.get("other_index","") or r.get("index","")).strip()
        if idx not in wanted:
            continue
        if r.get("row") and r.get("col"):
            ends.append((int(float(r["row"])), int(float(r["col"]))))
        elif r.get("x") and r.get("y"):
            ends.append((int(float(r["y"]) // TILE_H), int(float(r["x"]) // TILE_W)))
        elif r.get("cx") and r.get("cy"):
            ends.append((int(float(r["cy"]) // TILE_H), int(float(r["cx"]) // TILE_W)))
    return ends

DIR_VEC = {"L": (0,-1), "R": (0,1), "U": (-1,0), "D": (1,0)}

def _parse_arrow_dir_str(s: str) -> str:
    s = (s or "").strip().upper()
    if s in ("L","LEFT"):  return "L"
    if s in ("R","RIGHT"): return "R"
    if s in ("U","UP"):    return "U"
    if s in ("D","DOWN"):  return "D"
    # fallback
    return _parse_arrow_dir_str(DEFAULT_ARROW_DIR)

def _extract_arrow_dir_from_row(row: dict) -> str:
    # 1) explicit CSV field
    for k in ("direction","arrow_dir","dir"):
        if k in row and row[k]:
            return _parse_arrow_dir_str(row[k])
    # 2) filename hints (optional)
    fn = (row.get("filename") or row.get("name") or row.get("tile") or "").lower()
    if any(w in fn for w in ("left","_l","-l"," l.")):  return "L"
    if any(w in fn for w in ("right","_r","-r"," r.")): return "R"
    if any(w in fn for w in ("up","_u","-u"," u.")):    return "U"
    if any(w in fn for w in ("down","_d","-d"," d.")):  return "D"
    # 3) default
    return _parse_arrow_dir_str(DEFAULT_ARROW_DIR)


def _norm_idx(idx: str) -> str:
    s = (idx or "").strip()
    return (s.lstrip("0") or "0") if s else s

def extract_px_xy(row):
    if row.get("cx") and row.get("cy"):
        return float(row["cx"]), float(row["cy"])
    if row.get("x") and row.get("y"):
        return float(row["x"]), float(row["y"])
    if row.get("row") and row.get("col"):
        r = int(float(row["row"])); c = int(float(row["col"]))
        x = int(c * TILE_W + TILE_W/2); y = int(r * TILE_H + TILE_H/2)
        return x, y
    return None

def rowcol_from_px(x_px, y_px):
    return int(float(y_px) // TILE_H), int(float(x_px) // TILE_W)

def px_center_of_rc(r, c):
    x = int(c * TILE_W + TILE_W / 2)
    y = int(r * TILE_H + TILE_H / 2)
    return x, y

def nearest_walkable_tile_around_px(grid_walk, x_px, y_px, search_radius_tiles=10):
    H, W = grid_walk.shape
    r0, c0 = rowcol_from_px(x_px, y_px)
    best, bestd2 = None, None
    for dr in range(-search_radius_tiles, search_radius_tiles+1):
        for dc in range(-search_radius_tiles, search_radius_tiles+1):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < H and 0 <= c < W and grid_walk[r, c]:
                cx = int(c * TILE_W + TILE_W/2)
                cy = int(r * TILE_H + TILE_H/2)
                d2 = (cx - x_px)**2 + (cy - y_px)**2
                if best is None or d2 < bestd2:
                    best, bestd2 = (r, c), d2
    return best

def nearest_walkable_tile_around_rc(grid_walk, r, c, search_radius_tiles=3):
    x_px, y_px = px_center_of_rc(r, c)
    return nearest_walkable_tile_around_px(grid_walk, x_px, y_px, search_radius_tiles=search_radius_tiles)

def _add_dir_constraint(edge_dir_map, a, b, drdc, overwrite=False):
    """Constrain undirected edge {a,b} to allow ONLY vector drdc."""
    k = frozenset((a, b))
    if (not overwrite) and (k in edge_dir_map): return
    edge_dir_map[k] = drdc

def _apply_arrow_zone_strict(edge_dir_map, grid_walk, center_rc, dir_code, radius_tiles):
    """Within Chebyshev radius, constrain all touching edges to arrow direction."""
    H, W = grid_walk.shape
    dr0, dc0 = DIR_VEC[dir_code]
    def inb(r,c): return 0 <= r < H and 0 <= c < W

    for drr in range(-radius_tiles, radius_tiles + 1):
        for dcc in range(-radius_tiles, radius_tiles + 1):
            r = center_rc[0] + drr
            c = center_rc[1] + dcc
            if not inb(r, c): continue
            if ARROW_APPLY_ON_WALKABLE and not grid_walk[r, c]: continue
            for vr, vc in ((1,0),(-1,0),(0,1),(0,-1)):
                nr, nc = r + vr, c + vc
                if not inb(nr, nc): continue
                if ARROW_APPLY_ON_WALKABLE and not grid_walk[nr, nc]: continue
                _add_dir_constraint(edge_dir_map, (r,c), (nr,nc), (dr0, dc0),
                                    overwrite=not ARROW_OVERLAP_KEEP_FIRST)

def load_oneway_arrows_for_src(base_source_key_norm, legend_other_rows, grid_walk):
    """
    Reads rows whose other_index is in ARROW_BY_FIELD_INDICES and consumes
    row['direction'] (or filename hints) to produce:
      - arrow_cells: {(r,c): 'L'/'R'/'U'/'D'}
      - edge_dir_map: {frozenset({(ra,ca),(rb,cb)}): (dr,dc)}
    """
    arrows = {}
    edge_dir_map = {}
    by_field_norm = {_norm_idx(i) for i in ARROW_BY_FIELD_INDICES}

    for row in legend_other_rows:
        if normalize_other_source(row.get("source","")) != base_source_key_norm:
            continue

        raw_idx = (row.get("other_index") or row.get("index") or "").strip()
        if not raw_idx:
            continue
        if _norm_idx(raw_idx) not in by_field_norm:
            continue

        xy = extract_px_xy(row)
        if not xy:
            continue
        x, y = xy

        dir_code = _extract_arrow_dir_from_row(row)   # <-- uses CSV direction
        center_rc = nearest_walkable_tile_around_px(grid_walk, x, y, search_radius_tiles=10) or rowcol_from_px(x, y)
        arrows[center_rc] = dir_code

        # Optionally constrain the immediate opening adjacent to the arrow tile
        r, c = center_rc
        if   dir_code == "L": nb, vec = (r, c-1), (0,-1)
        elif dir_code == "R": nb, vec = (r, c+1), (0, 1)
        elif dir_code == "U": nb, vec = (r-1, c), (-1,0)
        else:                 nb, vec = (r+1, c), (1, 0)
        _add_dir_constraint(edge_dir_map, (r,c), nb, vec, overwrite=False)

        # Strict local one-way zone (keeps things simple and robust)
        _apply_arrow_plus(edge_dir_map, grid_walk, center_rc, dir_code, keep_first=True)

    return arrows, edge_dir_map

# BFS (with arrow constraints & liquid rule)
def bfs_from_sources(grid, sources, diagonal=False, liquid_mask=None, arrow_edges=None):
    H, W = grid.shape
    neigh = NEIGH8 if diagonal else NEIGH4
    dist   = np.full((H,W), -1, dtype=int)
    parent = np.full((H,W,2), -1, dtype=int)
    q = deque()

    for (r,c) in sources:
        if 0 <= r < H and 0 <= c < W and grid[r,c]:
            dist[r,c] = 0
            parent[r,c] = [-2,-2]
            q.append((r,c))

    def is_liq(r,c):
        return (liquid_mask is not None and 0 <= r < H and 0 <= c < W and liquid_mask[r,c])

    def allowed_move(r, c, nr, nc):
        if not (0 <= nr < H and 0 <= nc < W): return False
        if not grid[nr, nc]: return False
        if is_liq(r,c) and is_liq(nr,nc): return False
        if arrow_edges:
            dr, dc = nr - r, nc - c
            v = arrow_edges.get(frozenset(((r, c), (nr, nc))))
            if v is not None and (dr * v[0] + dc * v[1]) <= 0:
                return False
        return True

    while q:
        r,c = q.popleft()
        for dr,dc in neigh:
            nr, nc = r+dr, c+dc
            # NEW: bounds check first
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if dist[nr, nc] != -1:
                continue
            if not allowed_move(r,c,nr,nc):
                continue
            dist[nr,nc] = dist[r,c] + 1
            parent[nr,nc] = [r,c]
            q.append((nr,nc))

    return dist, parent


def bfs_single_source(grid, start_rc, diagonal=False, liquid_mask=None, arrow_edges=None):
    return bfs_from_sources(grid, [start_rc], diagonal=diagonal, liquid_mask=liquid_mask, arrow_edges=arrow_edges)

# portal detection & tele-BFS
def detect_portal_pairs_for_src(base_source_key_norm, legend_other_rows):
    buckets = defaultdict(list)  # idx_norm -> [(x,y,(r,c)), ...]
    for r in legend_other_rows:
        if normalize_other_source(r.get("source","")) != base_source_key_norm:
            continue
        raw_idx = (r.get("other_index","") or r.get("index","") or "").strip()
        if not raw_idx: continue
        xy = extract_px_xy(r)
        if not xy: continue
        x, y = xy
        rc = rowcol_from_px(x, y)
        buckets[_norm_idx(raw_idx)].append((int(x), int(y), rc))

    pairs = []
    # undirected
    for idx_norm in sorted(PORTAL_OTHER_INDICES, key=lambda s: int(_norm_idx(s))):
        pts = buckets.get(_norm_idx(idx_norm), [])
        if len(pts) == 2:
            a, b = pts[0], pts[1]
            pairs.append((_norm_idx(idx_norm), a, b))
            if DEBUG_LOGS: print(f"[PORTAL] Undirected idx {_norm_idx(idx_norm)}: 2 markers -> OK")
        else:
            if DEBUG_LOGS: print(f"[PORTAL] Undirected idx {_norm_idx(idx_norm)}: found {len(pts)} (need 2) -> skip")

    # directed
    for src_raw, dst_raw in sorted(DIRECTED_PORTAL_INDEX_PAIRS, key=lambda t: int(_norm_idx(t[0]))):
        src_idx = _norm_idx(src_raw); dst_idx = _norm_idx(dst_raw)
        src_pts = buckets.get(src_idx, []); dst_pts = buckets.get(dst_idx, [])
        if len(src_pts) == 1 and len(dst_pts) == 1:
            pairs.append((f"{src_idx}>{dst_idx}", src_pts[0], dst_pts[0]))
            if DEBUG_LOGS: print(f"[PORTAL] Directed {src_idx}>{dst_idx}: OK")
        else:
            if DEBUG_LOGS: print(f"[PORTAL] Directed {src_idx}>{dst_idx}: need 1+1, got {len(src_pts)}+{len(dst_pts)} -> skip")

    def _key(tag_ab):
        return int(tag_ab[0].split(">")[0]) if ">" in tag_ab[0] else int(tag_ab[0])

    pairs.sort(key=_key)
    return pairs

def build_teleport_adj(visited_tags, ordered_pairs):
    adj = defaultdict(list)
    for tag, (xA,yA,eA), (xB,yB,eB) in ordered_pairs:
        if tag not in visited_tags: 
            continue
        if ">" in tag:
            adj[eA].append((eB, (xA,yA), (xB,yB), tag))
        else:
            adj[eA].append((eB, (xA,yA), (xB,yB), tag))
            adj[eB].append((eA, (xB,yB), (xA,yA), tag))
    return adj

def bfs_with_teleports(grid, start_rc, targets, tele_adj, diagonal=False, liquid_mask=None, arrow_edges=None):
    H, W = grid.shape
    neigh = NEIGH8 if diagonal else NEIGH4
    dist   = np.full((H,W), -1, dtype=int)
    parent = np.full((H,W,2), -1, dtype=int)
    tele_used = {}

    def is_liq(r,c):
        return (liquid_mask is not None and 0 <= r < H and 0 <= c < W and liquid_mask[r,c])

    def allowed_move(r, c, nr, nc):
        if not (0 <= nr < H and 0 <= nc < W): return False
        if not grid[nr, nc]: return False
        if is_liq(r,c) and is_liq(nr,nc): return False
        if arrow_edges:
            dr, dc = nr - r, nc - c
            v = arrow_edges.get(frozenset(((r, c), (nr, nc))))
            if v is not None and (dr * v[0] + dc * v[1]) <= 0:
                return False
        return True

    q = deque()
    if 0 <= start_rc[0] < H and 0 <= start_rc[1] < W and grid[start_rc]:
        dist[start_rc] = 0
        parent[start_rc] = [-2,-2]
        q.append(start_rc)

    while q:
        r,c = q.popleft()

        # normal steps
        for dr,dc in neigh:
            nr, nc = r+dr, c+dc
            if not (0 <= nr < H and 0 <= nc < W): continue
            if dist[nr,nc] != -1: continue
            if not allowed_move(r,c,nr,nc): continue
            dist[nr,nc] = dist[r,c] + 1
            parent[nr,nc] = [r,c]
            q.append((nr,nc))

        # teleports ignore arrow/liquid
        for (to_rc, entry_px, exit_px, _tag) in tele_adj.get((r,c), []):
            if dist[to_rc] == -1:
                dist[to_rc] = dist[r,c] + 1
                parent[to_rc] = [r,c]
                tele_used[to_rc] = (entry_px, exit_px)
                q.append(to_rc)

    best_t, bestd = None, None
    for t in targets:
        if 0 <= t[0] < H and 0 <= t[1] < W and dist[t] >= 0:
            if bestd is None or dist[t] < bestd:
                best_t, bestd = t, dist[t]
    return dist, parent, tele_used, best_t

def reconstruct_path_with_tele(parent, endpoint):
    if endpoint is None: return []
    path = []
    cur = tuple(endpoint)
    while True:
        path.append(cur)
        pr, pc = parent[cur]
        if (pr,pc) == (-2,-2): break
        cur = (pr,pc)
    path.reverse()
    return path

# drawing helpers
def _blend_rect_rgba(dst, x1, y1, x2, y2, color_bgra=(0,0,0,200)):
    H, W = dst.shape[:2]
    x1 = max(0, min(W, x1)); x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1)); y2 = max(0, min(H, y2))
    if x1 >= x2 or y1 >= y2: return
    roi = dst[y1:y2, x1:x2]
    a = color_bgra[3] / 255.0
    overlay = np.zeros_like(roi, dtype=np.uint8)
    overlay[..., 0] = color_bgra[0]
    overlay[..., 1] = color_bgra[1]
    overlay[..., 2] = color_bgra[2]
    roi[..., 0] = (overlay[...,0] * a + roi[...,0] * (1-a)).astype(np.uint8)
    roi[..., 1] = (overlay[...,1] * a + roi[...,1] * (1-a)).astype(np.uint8)
    roi[..., 2] = (overlay[...,2] * a + roi[...,2] * (1-a)).astype(np.uint8)

def _stamp_text_rgba(dst, text, center_xy, font, scale, fill_bgr, outline_bgr, thickness=1):
    H, W = dst.shape[:2]
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    cx, cy = int(center_xy[0]), int(center_xy[1])
    org = (cx - tw//2, cy + th//2)
    pad = max(2, thickness + 1)
    x1 = max(0, org[0] - pad); y1 = max(0, org[1] - th - pad)
    x2 = min(W, org[0] + tw + pad); y2 = min(H, org[1] + base + pad)
    if x1 >= x2 or y1 >= y2: return
    box_w, box_h = x2 - x1, y2 - y1
    mask_fill = np.zeros((box_h, box_w), dtype=np.uint8)
    mask_outline = np.zeros((box_h, box_w), dtype=np.uint8)
    tx = org[0] - x1; ty = org[1] - y1
    for dx, dy in ((-1,0),(1,0),(0,-1),(0,1)):
        cv2.putText(mask_outline, text, (tx+dx, ty+dy), font, scale, 255, thickness+1, cv2.LINE_AA)
    cv2.putText(mask_fill, text, (tx, ty), font, scale, 255, thickness, cv2.LINE_AA)
    roi = dst[y1:y2, x1:x2]
    m_o = (mask_outline > 0)
    roi[m_o, 0] = outline_bgr[0]; roi[m_o,1] = outline_bgr[1]; roi[m_o,2] = outline_bgr[2]; roi[m_o,3] = 255
    m_f = (mask_fill > 0)
    roi[m_f, 0] = fill_bgr[0]; roi[m_f,1] = fill_bgr[1]; roi[m_f,2] = fill_bgr[2]; roi[m_f,3] = 255

def draw_dotted_line(img, p0, p1, color, thickness=2, period=12):
    x0,y0 = p0; x1,y1 = p1
    length = int(np.hypot(x1-x0, y1-y0))
    if length == 0: return
    ux = (x1-x0) / max(length,1); uy = (y1-y0) / max(length,1)
    on = max(1, period // 2)
    t = 0
    while t < length:
        t_end = min(t + on, length)
        sx, sy = int(x0 + ux*t),     int(y0 + uy*t)
        ex, ey = int(x0 + ux*t_end), int(y0 + uy*t_end)
        cv2.line(img, (sx,sy), (ex,ey), color, thickness, lineType=cv2.LINE_AA)
        t += period

def _angle_deg(p0, p1, p2):
    v1 = np.array([p0[0]-p1[0], p0[1]-p1[1]], dtype=np.float32)
    v2 = np.array([p2[0]-p1[0], p2[1]-p1[1]], dtype=np.float32)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return 0.0
    v1 /= n1; v2 /= n2
    dot = np.clip(float(v1 @ v2), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    return ang if cross >= 0 else -ang

def _in_bounds(x, y, W, H, margin):
    return (margin <= x < (W - margin)) and (margin <= y < (H - margin))

def _place_step_labels_adaptive(out, path_rc):
    if not (SHOW_TILE_COUNT and LABEL_ADAPTIVE): 
        return
    H, W = out.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = TILE_COUNT_FONT_SCALE
    dx0, dy0 = TILE_COUNT_OFFSET
    fill_bgr    = (240, 240, 240)
    outline_bgr = (0, 0, 0)
    pill_alpha  = TILE_COUNT_PILL_ALPHA
    pill_pad    = TILE_COUNT_PILL_PAD

    pts_px = [px_center_of_rc(r, c) for (r, c) in path_rc]
    if len(pts_px) < 3: return

    min_arc_px      = MIN_LABEL_ARC_TILES * min(TILE_W, TILE_H)
    straight_arc_px = int(min_arc_px * STRAIGHT_LABEL_ARC_FACTOR)
    placed = []
    arc_since_last = 0.0

    for i in range(1, len(pts_px) - 1):
        arc_since_last += np.hypot(pts_px[i][0] - pts_px[i-1][0], pts_px[i][1] - pts_px[i-1][1])
        turn = abs(_angle_deg(pts_px[i-1], pts_px[i], pts_px[i+1]))
        want_label = (arc_since_last >= min_arc_px and turn >= TURN_LABEL_ANGLE_DEG) or (arc_since_last >= straight_arc_px)
        if not want_label: 
            continue

        cx, cy = pts_px[i]
        cx += int(dx0); cy += int(dy0)

        tries = 0
        while tries < LABEL_MAX_TRIES:
            ok_dist = all(np.hypot(cx - px, cy - py) > LABEL_CLEAR_RADIUS_PX for (px, py) in placed)
            ok_edge = _in_bounds(cx, cy, W, H, LABEL_EDGE_MARGIN_PX)
            if ok_dist and ok_edge:
                break
            sx = ((tries % 3) - 1) * LABEL_NUDGE_PX
            sy = (((tries + 1) % 3) - 1) * LABEL_NUDGE_PX
            cx += sx; cy += sy
            tries += 1

        text = str(i)
        (tw, th), base = cv2.getTextSize(text, font, scale, TILE_COUNT_THICKNESS)
        x1 = cx - tw//2 - pill_pad
        y1 = cy - th//2 - pill_pad
        x2 = cx + (tw - tw//2) + pill_pad
        y2 = cy + (th - th//2) + base + pill_pad
        _blend_rect_rgba(out, x1, y1, x2, y2, color_bgra=(0,0,0,pill_alpha))
        _stamp_text_rgba(out, text, (cx, cy), font, scale, fill_bgr, outline_bgr, thickness=TILE_COUNT_THICKNESS)

        placed.append((cx, cy))
        arc_since_last = 0.0

def draw_overlay_with_markers(combined_rgba, path_rc, starts, terminal_rc, reached_goal,
                              portal_segments_px=None, color_cuts=None):
    out = combined_rgba.copy()
    color_cuts = sorted(set(color_cuts or []))

    # Start pins
    for (sr, sc) in (starts or []):
        sx, sy = px_center_of_rc(sr, sc)
        cv2.circle(out, (sx, sy), PIN_RADIUS, PIN_START_COLOR, -1, lineType=cv2.LINE_AA)

    # Goal pin
    if terminal_rc is not None:
        tr, tc = terminal_rc
        tx, ty = px_center_of_rc(tr, tc)
        gcol = PIN_GOAL_REACHED_COLOR if reached_goal else PIN_GOAL_PENDING_COLOR
        cv2.circle(out, (tx, ty), PIN_RADIUS+1, gcol, -1, lineType=cv2.LINE_AA)

    # Walk path segments with color changes after teleports
    if path_rc and len(path_rc) >= 2:
        pts = [px_center_of_rc(r, c) for (r, c) in path_rc]
        cut_indices = [i for i in color_cuts if 1 <= i < len(pts)]
        bounds = [0] + cut_indices + [len(pts)-1]
        color_idx = 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a < 1: continue
            sub = np.array(pts[a:b+1], dtype=np.int32)
            col = WALK_COLORS[color_idx % len(WALK_COLORS)]
            cv2.polylines(out, [sub], False, col, WALK_THICKNESS, lineType=cv2.LINE_AA)
            color_idx += 1

    # Teleports dotted
    if portal_segments_px:
        for (p0, p1) in portal_segments_px:
            draw_dotted_line(out, p0, p1, TELEPORT_COLOR, thickness=TELEPORT_THICKNESS, period=TELEPORT_DOT_PERIOD_PX)

    # Small node dots
    if path_rc:
        node_color = WALK_COLORS[min(len(color_cuts or []), len(WALK_COLORS)-1)]
        for (r, c) in path_rc:
            x, y = px_center_of_rc(r, c)
            cv2.circle(out, (x, y), NODE_RADIUS, node_color, -1, lineType=cv2.LINE_AA)

    # Adaptive labels
    if SHOW_TILE_COUNT and path_rc and len(path_rc) >= 2:
        _place_step_labels_adaptive(out, path_rc)

    return out

def concat_paths(a, b):
    if not a: return b or []
    if not b: return a or []
    return a + (b[1:] if a[-1] == b[0] else b)

# collectibles/enemies/doors/rooms
def _collect_collectibles_for_src(base_source_key_norm, legend_other_rows, indices_set):
    pts = []
    for r in legend_other_rows:
        if normalize_other_source(r.get("source","")) != base_source_key_norm:
            continue
        idx = (r.get("other_index","") or r.get("index","") or "").strip()
        if idx not in indices_set:
            continue
        if r.get("row") and r.get("col"):
            pts.append((int(float(r["row"])), int(float(r["col"])), idx))
        elif r.get("x") and r.get("y"):
            rr, cc = rowcol_from_px(float(r["x"]), float(r["y"]))
            pts.append((rr, cc, idx))
        elif r.get("cx") and r.get("cy"):
            rr, cc = rowcol_from_px(float(r["cx"]), float(r["cy"]))
            pts.append((rr, cc, idx))
    return pts

def _snap_collectibles_to_walkable(grid_walk, items, search_radius=8):
    out, seen = [], set()
    for t in items:
        # allow (r,c), (r,c,idx), or longer
        r, c = t[0], t[1]
        idx = t[2] if len(t) >= 3 else ""
        rc2 = nearest_walkable_tile_around_rc(grid_walk, r, c, search_radius_tiles=search_radius)
        if rc2 is None: 
            continue
        key = (rc2[0], rc2[1], idx)
        if key not in seen:
            out.append((rc2[0], rc2[1], idx))
            seen.add(key)
    return out


def _looks_like_door_row(row: dict) -> bool:
    for fld in BASE_DOOR_INDEX_FIELDS:
        v = (row.get(fld) or "").strip()
        if v and v in DOOR_INDEX_WHITELIST:
            return True
    for fld in ("manual_override", "classification", "filename"):
        t = (row.get(fld) or "").lower()
        if "door" in t:
            return True
    return False

def find_all_doors_in_base_rows(rows_for_src):
    out = []
    for r in rows_for_src:
        if _looks_like_door_row(r) and r.get("row") and r.get("col"):
            rr = int(float(r["row"])); cc = int(float(r["col"]))
            out.append((rr, cc, (r.get("index") or r.get("other_index") or r.get("tile_index") or "door").strip()))
    return out

def label_connected_components(grid_walk):
    H, W = grid_walk.shape
    labels = np.full((H, W), -1, dtype=np.int32)
    cc_id = 0
    for r in range(H):
        for c in range(W):
            if not grid_walk[r, c] or labels[r, c] != -1:
                continue
            q = deque([(r, c)])
            labels[r, c] = cc_id
            while q:
                cr, cc = q.popleft()
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr, nc = cr+dr, cc+dc
                    if not (0 <= nr < H and 0 <= nc < W): continue
                    if not grid_walk[nr, nc]: continue
                    if labels[nr, nc] != -1: continue
                    labels[nr, nc] = cc_id
                    q.append((nr, nc))
            cc_id += 1
    return labels, cc_id

def _nearest_target(dist_mat, targets):
    best, bestd = None, None
    for t in targets:
        if dist_mat[t] >= 0 and (bestd is None or dist_mat[t] < bestd):
            best, bestd = t, dist_mat[t]
    return best

def enforce_room_enemies_then_door(current_rc, grid_walk, liquid_mask, arrow_edges,
                                   room_labels, enemies_by_room, doors_by_room,
                                   final_path, enemies_visited, keyname):
    """Clear room enemies (nearest-first), then move to a door in that room."""
    if current_rc is None: return current_rc, final_path, enemies_visited
    cur_rid = room_labels[current_rc[0], current_rc[1]]
    if cur_rid < 0: return current_rc, final_path, enemies_visited

    room_enemies = enemies_by_room.get(cur_rid, [])
    room_doors   = doors_by_room.get(cur_rid, [])

    if not room_enemies or not room_doors:
        return current_rc, final_path, enemies_visited

    # (a) visit all enemies
    rem = room_enemies[:]
    while rem:
        distE, parE = bfs_single_source(grid_walk, current_rc, diagonal=ALLOW_DIAGONAL,
                                        liquid_mask=liquid_mask, arrow_edges=arrow_edges)
        best = _nearest_target(distE, rem)
        if best is None:
            if DEBUG_LOGS: print(f"[ROOM {cur_rid} :: {keyname}] unreachable enemy from {current_rc}")
            break
        seg = reconstruct_path_with_tele(parE, best)
        final_path = concat_paths(final_path, seg)
        current_rc = best
        enemies_visited += 1
        rem.remove(best)
    enemies_by_room[cur_rid] = rem

    # (b) nearest door
    if room_doors:
        distD, parD = bfs_single_source(grid_walk, current_rc, diagonal=ALLOW_DIAGONAL,
                                        liquid_mask=liquid_mask, arrow_edges=arrow_edges)
        best_door = _nearest_target(distD, room_doors)
        if best_door is not None:
            segD = reconstruct_path_with_tele(parD, best_door)
            final_path = concat_paths(final_path, segD)
            current_rc = best_door
            doors_by_room[cur_rid] = [d for d in room_doors if d != best_door]

    return current_rc, final_path, enemies_visited

# MAIN
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_rows_all = read_csv_rows(LEGEND_BASE_PATH)
    other_rows    = read_csv_rows(LEGEND_OTHER_PATH)

    by_src = collect_base_by_source(base_rows_all)
    summary = {}

    def build_teleport_adj_allow_all(ordered_pairs):
        adj = defaultdict(list)
        for tag, (xA,yA,eA), (xB,yB,eB) in ordered_pairs:
            if ">" in tag:
                adj[eA].append((eB, (xA,yA), (xB,yB), tag))
            else:
                adj[eA].append((eB, (xA,yA), (xB,yB), tag))
                adj[eB].append((eA, (xB,yB), (xA,yA), tag))
        return adj

    for base_source_key, rows in by_src.items():
        try:
            # 1) Normalize odd CSV shifts for specific sources
            fix = BAD_SHIFT_FIX.get((base_source_key or "").strip())
            if fix:
                _fix_legend_bad_shift_inplace(rows,
                                              room_columns=int(fix.get("room_columns", 4)),
                                              shift_rows=int(fix.get("shift_rows", 11)))

            # 2) Size & filenames
            H, W = dims_from_rows(rows)
            keyname = name_for(base_source_key)
            base_key_norm = normalize_base_source(base_source_key)
            canvas_h, canvas_w = H*TILE_H, W*TILE_W

            # 3) Rebuild images
            base_rgba = reconstruct_base_png(rows, H, W)
            cv2.imwrite(str(OUT_DIR / f"reconstruct_base_{keyname}.png"), base_rgba)
            others_for_src = [r for r in other_rows if normalize_other_source(r.get("source","")) == base_key_norm]
            if DEBUG_LOGS:
                print(f"[MATCH] base='{base_source_key}' -> legend_other rows: {len(others_for_src)}")
            others_rgba = reconstruct_others_png(others_for_src, canvas_w, canvas_h)
            cv2.imwrite(str(OUT_DIR / f"reconstruct_others_{keyname}.png"), others_rgba)
            combined_rgba = compose(base_rgba, others_rgba)
            cv2.imwrite(str(OUT_DIR / f"reconstruct_combined_{keyname}.png"), combined_rgba)

            # 4) Grids & starts/goals
            grid, present, liquid_mask = build_passability_presence_and_liquid(rows, H, W)
            grid_walk = grid.copy()
            if TREAT_TRANSPARENT_AS_WALKABLE:
                grid_walk[~present] = True
            starts = find_edge_starts(rows, H, W)
            if not starts:
                raise RuntimeError("No valid starts (edge tiles with weight==1 and walkable).")
            end_rc_list = load_ends_from_legend_other(base_key_norm, other_rows)

            # 5) Rooms & doors
            room_labels, n_rooms = label_connected_components(grid_walk)
            all_doors = find_all_doors_in_base_rows(rows)
            doors_by_room = defaultdict(list)
            for (dr, dc, _tag) in all_doors:
                if 0 <= dr < H and 0 <= dc < W:
                    rid = room_labels[dr, dc]
                    if rid >= 0:
                        doors_by_room[rid].append((dr, dc))

            # 6) Portals
            portal_pairs_raw = detect_portal_pairs_for_src(base_key_norm, other_rows)
            ordered_pairs = []
            for tag, a, b in portal_pairs_raw:
                (x1, y1, rc1) = a
                (x2, y2, rc2) = b
                e1 = nearest_walkable_tile_around_px(grid_walk, x1, y1, search_radius_tiles=8) or rc1
                e2 = nearest_walkable_tile_around_px(grid_walk, x2, y2, search_radius_tiles=8) or rc2
                if not (0 <= e1[0] < H and 0 <= e1[1] < W and 0 <= e2[0] < H and 0 <= e2[1] < W):
                    if DEBUG_LOGS:
                        print(f"[PORTAL] {tag} snapped OOB; skip. e1={e1}, e2={e2}, HxW={H}x{W}")
                    continue
                ordered_pairs.append((tag, (int(x1), int(y1), e1), (int(x2), int(y2), e2)))

            # 7) Arrows
            arrow_cells, arrow_edges = load_oneway_arrows_for_src(base_key_norm, other_rows, grid_walk)
            if DEBUG_LOGS and arrow_cells:
                print(f"[ARROWS] {keyname}: {len(arrow_cells)} one-way openings.")

           # 8) Collectibles & enemies
            final_path = []
            portal_segments_px = []
            color_cuts = []
            reached_goal = False
            terminal_rc = None
            goal_chosen = None
            visited_tags = []
            
            # gather collectibles: ladders (from BASE rows) + keys/bombs (from legend_other)
            raw_collect_other = _collect_collectibles_for_src(base_key_norm, other_rows, COLLECT_OTHER_INDICES)
            collect_ladders_base = _collect_ladders_from_base(rows)  # [(r,c,'LADDER'), ...]
            
            # snap everything to nearest walkable
            collectibles = _snap_collectibles_to_walkable(grid_walk, raw_collect_other, search_radius=8)
            collectibles = collectibles + _snap_collectibles_to_walkable(grid_walk, collect_ladders_base, search_radius=6)
            
            # counts
            keys_total   = sum(1 for _,_,idx in collectibles if idx in KEY_INDICES)
            bombs_total  = sum(1 for _,_,idx in collectibles if idx in BOMB_INDICES)
            ladders_total = sum(1 for _,_,idx in collectibles if idx == "LADDER")
            
            keys_collected = 0
            bombs_collected = 0
            ladders_collected = 0

            # enemies (unchanged, but we keep as TRIPLES)
            raw_enemies = [
                r for r in other_rows
                if normalize_other_source(r.get("source","")) == base_key_norm
                and (r.get("category") or "").strip().lower() == ENEMY_CATEGORY
                and (not r.get("weights") and not r.get("weight") or
                     (lambda w: (w.strip()=="" or (w.replace('.','',1).replace('-','',1).isdigit() and float(w)>0)))(r.get("weights") or r.get("weight") or ""))
            ]
            
            enemy_positions = []
            for r in raw_enemies:
                if r.get("row") and r.get("col"):
                    rr = int(float(r["row"])); cc = int(float(r["col"]))
                else:
                    rr, cc = rowcol_from_px(float(r.get("x") or r.get("cx")), float(r.get("y") or r.get("cy")))
                idx = (r.get("other_index") or r.get("index") or "").strip()
                enemy_positions.append((rr, cc, idx))
            
            enemies = _snap_collectibles_to_walkable(grid_walk, enemy_positions, search_radius=8) if enemy_positions else []
            enemies_total = len(enemies)
            enemies_visited = 0
            enemies_by_room = defaultdict(list)
            for (er, ec, _idx) in enemies:
                if 0 <= er < H and 0 <= ec < W:
                    rid = room_labels[er, ec]
                    if rid >= 0:
                        enemies_by_room[rid].append((er, ec))


            # build enemies as TRIPLES (r, c, idx)
            raw_enemies = [
                r for r in other_rows
                if normalize_other_source(r.get("source","")) == base_key_norm
                and (r.get("category") or "").strip().lower() == ENEMY_CATEGORY
                and (not r.get("weights") and not r.get("weight") or
                     (lambda w: (w.strip()=="" or (w.replace('.','',1).replace('-','',1).isdigit() and float(w)>0)))(r.get("weights") or r.get("weight") or ""))
            ]
            
            enemy_positions = []
            for r in raw_enemies:
                if r.get("row") and r.get("col"):
                    rr = int(float(r["row"])); cc = int(float(r["col"]))
                else:
                    rr, cc = rowcol_from_px(float(r.get("x") or r.get("cx")), float(r.get("y") or r.get("cy")))
                idx = (r.get("other_index") or r.get("index") or "").strip()
                enemy_positions.append((rr, cc, idx))
            
            enemies = _snap_collectibles_to_walkable(
                grid_walk,
                enemy_positions,
                search_radius=8
            ) if enemy_positions else []


            enemies_total = len(enemies)
            enemies_visited = 0
            enemies_by_room = defaultdict(list)
            for (er, ec, _idx) in enemies:
                if 0 <= er < H and 0 <= ec < W:
                    rid = room_labels[er, ec]
                    if rid >= 0:
                        enemies_by_room[rid].append((er, ec))

            # 9) Seed current position via nearest collectible from ANY start
            current_rc = None
            if collectibles:
                distS, parentS = bfs_from_sources(grid_walk, starts, diagonal=ALLOW_DIAGONAL,
                                                  liquid_mask=liquid_mask, arrow_edges=arrow_edges)
                best_item, bestd = None, None
                for (rr, cc, idx) in collectibles:
                    if distS[rr, cc] >= 0 and (bestd is None or distS[rr, cc] < bestd):
                        best_item, bestd = (rr, cc, idx), distS[rr, cc]
                if best_item is not None:
                    seg = reconstruct_path_with_tele(parentS, (best_item[0], best_item[1]))
                    final_path = concat_paths(final_path, seg)
                    current_rc = (best_item[0], best_item[1])
                    # after taking best_item
                    if best_item[2] in KEY_INDICES:      keys_collected   += 1
                    elif best_item[2] in BOMB_INDICES:   bombs_collected  += 1
                    elif best_item[2] == "LADDER":       ladders_collected += 1

                    collectibles.remove(best_item)

                # Greedy to remaining collectibles
                while collectibles and current_rc is not None:
                    distC, parentC = bfs_single_source(grid_walk, current_rc, diagonal=ALLOW_DIAGONAL,
                                                       liquid_mask=liquid_mask, arrow_edges=arrow_edges)
                    best_next, bestd = None, None
                    for (rr, cc, idx) in collectibles:
                        if distC[rr, cc] >= 0 and (bestd is None or distC[rr, cc] < bestd):
                            best_next, bestd = (rr, cc, idx), distC[rr, cc]
                    if best_next is None:
                        if DEBUG_LOGS:
                            print(f"[INFO] {keyname}: some collectibles unreachable from {current_rc}; continuing.")
                        break
                    seg = reconstruct_path_with_tele(parentC, (best_next[0], best_next[1]))
                    final_path = concat_paths(final_path, seg)
                    current_rc = (best_next[0], best_next[1])
                    # after taking best_next
                    if best_next[2] in KEY_INDICES:      keys_collected   += 1
                    elif best_next[2] in BOMB_INDICES:   bombs_collected  += 1
                    elif best_next[2] == "LADDER":       ladders_collected += 1

                    collectibles.remove(best_next)

            # 10) If still no current, walk from starts toward first reachable thing
            if current_rc is None:
                distS, parentS = bfs_from_sources(grid_walk, starts, diagonal=ALLOW_DIAGONAL,
                                                  liquid_mask=liquid_mask, arrow_edges=arrow_edges)
                seed_candidates = []
                if ordered_pairs:
                    tag0, (xA0,yA0,eA0), (xB0,yB0,eB0) = ordered_pairs[0]
                    seed_candidates.extend([eA0] if ">" in tag0 else [eA0, eB0])
                for e in enemies:
                    seed_candidates.append((e[0], e[1]))
                found_seed = None
                for t in seed_candidates:
                    if 0 <= t[0] < H and 0 <= t[1] < W and distS[t] >= 0:
                        found_seed = t; break
                if found_seed is not None:
                    seg = reconstruct_path_with_tele(parentS, found_seed)
                    final_path = concat_paths(final_path, seg)
                    current_rc = found_seed
                elif np.max(distS) >= 0:
                    far = tuple(np.argwhere(distS == np.max(distS))[0])
                    final_path = concat_paths(final_path, reconstruct_path_with_tele(parentS, far))
                    current_rc = far

            # 11) Per-room rule (entry)
            current_rc, final_path, enemies_visited = enforce_room_enemies_then_door(
                current_rc, grid_walk, liquid_mask, arrow_edges,
                room_labels, enemies_by_room, doors_by_room,
                final_path, enemies_visited, keyname
            )

            # 12) Portals plan
            def is_directed(tag: str) -> bool: return ">" in tag

            if ordered_pairs:
                if current_rc is None:
                    tag0, (xA0,yA0,eA0), (xB0,yB0,eB0) = ordered_pairs[0]
                    initial_targets = [eA0] if is_directed(tag0) else [eA0, eB0]
                    distS, parentS = bfs_from_sources(grid_walk, starts, diagonal=ALLOW_DIAGONAL,
                                                      liquid_mask=liquid_mask, arrow_edges=arrow_edges)
                    best_t = None; bestd = None
                    for t in initial_targets:
                        if distS[t] >= 0 and (bestd is None or distS[t] < bestd):
                            best_t, bestd = t, distS[t]
                    if best_t is not None:
                        seg = reconstruct_path_with_tele(parentS, best_t)
                        final_path = concat_paths(final_path, seg)
                        current_rc = best_t
                    elif np.max(distS) >= 0:
                        far = tuple(np.argwhere(distS == np.max(distS))[0])
                        final_path = concat_paths(final_path, reconstruct_path_with_tele(parentS, far))
                        current_rc = far

                if current_rc is not None:
                    for (tag, (xA, yA, eA), (xB, yB, eB)) in ordered_pairs:
                        tele_adj = build_teleport_adj(set(visited_tags) if REUSE_VISITED_PORTALS else set(), ordered_pairs)
                        targets = [eA] if is_directed(tag) else [eA, eB]
                        dist, parent, tele_used, best_t = bfs_with_teleports(
                            grid_walk, current_rc, targets, tele_adj, diagonal=ALLOW_DIAGONAL,
                            liquid_mask=liquid_mask, arrow_edges=arrow_edges
                        )
                        if best_t is None:
                            tele_adj_all = build_teleport_adj_allow_all(ordered_pairs)
                            dist, parent, tele_used, best_t = bfs_with_teleports(
                                grid_walk, current_rc, targets, tele_adj_all, diagonal=ALLOW_DIAGONAL,
                                liquid_mask=liquid_mask, arrow_edges=arrow_edges
                            )
                        if best_t is None:
                            if DEBUG_LOGS: print(f"[PLAN] Skip unreachable portal: {tag} from {current_rc}")
                            continue

                        seg_rc = reconstruct_path_with_tele(parent, best_t)
                        offset = len(final_path) - (1 if final_path else 0)
                        for k in range(1, len(seg_rc)):
                            node = seg_rc[k]
                            if node in tele_used:
                                entry_px, exit_px = tele_used[node]
                                prev = seg_rc[k-1]
                                en_cx, en_cy = px_center_of_rc(*prev)
                                ex_cx, ex_cy = px_center_of_rc(*node)
                                portal_segments_px.extend([
                                    ((en_cx, en_cy), (int(entry_px[0]), int(entry_px[1]))),
                                    ((int(entry_px[0]), int(entry_px[1])), (int(exit_px[0]), int(exit_px[1]))),
                                    ((int(exit_px[0]), int(exit_px[1])), (ex_cx, ex_cy))
                                ])
                                color_cuts.append(offset + k)

                        final_path = concat_paths(final_path, seg_rc)
                        current_rc = best_t

                        entry_is_A = (current_rc == eA)
                        exit_rc  = eB if entry_is_A else eA
                        entry_px = (xA, yA) if entry_is_A else (xB, yB)
                        exit_px  = (xB, yB) if entry_is_A else (xA, yA)

                        if DEBUG_LOGS:
                            print(f"[PORTAL-USE] {tag} entry={current_rc} -> exit={exit_rc}")

                        en_cx, en_cy = px_center_of_rc(*current_rc)
                        ex_cx, ex_cy = px_center_of_rc(*exit_rc)
                        portal_segments_px.extend([
                            ((en_cx, en_cy), (int(entry_px[0]), int(entry_px[1]))),
                            ((int(entry_px[0]), int(entry_px[1])), (int(exit_px[0]), int(exit_px[1]))),
                            ((int(exit_px[0]), int(exit_px[1])), (ex_cx, ex_cy))
                        ])
                        current_rc = exit_rc
                        color_cuts.append(len(final_path))

                        if tag not in visited_tags:
                            visited_tags.append(tag)

                        # per-room after teleport
                        current_rc, final_path, enemies_visited = enforce_room_enemies_then_door(
                            current_rc, grid_walk, liquid_mask, arrow_edges,
                            room_labels, enemies_by_room, doors_by_room,
                            final_path, enemies_visited, keyname
                        )

                    terminal_rc = current_rc

                    # Go to best goal using visited teleports
                    if terminal_rc is not None and end_rc_list and visited_tags:
                        tele_adj = build_teleport_adj(set(visited_tags) if REUSE_VISITED_PORTALS else set(), ordered_pairs)
                        distG, parG, teleG, best_goal = bfs_with_teleports(
                            grid_walk, terminal_rc, end_rc_list, tele_adj, diagonal=ALLOW_DIAGONAL,
                            liquid_mask=liquid_mask, arrow_edges=arrow_edges
                        )
                        if best_goal is not None:
                            seg_goal = reconstruct_path_with_tele(parG, best_goal)
                            offset = len(final_path) - (1 if final_path else 0)
                            for k in range(1, len(seg_goal)):
                                node = seg_goal[k]
                                if node in teleG:
                                    entry_px, exit_px = teleG[node]
                                    prev = seg_goal[k-1]
                                    en_cx, en_cy = px_center_of_rc(*prev)
                                    ex_cx, ex_cy = px_center_of_rc(*node)
                                    portal_segments_px.extend([
                                        ((en_cx, en_cy), (int(entry_px[0]), int(entry_px[1]))),
                                        ((int(entry_px[0]), int(entry_px[1])), (int(exit_px[0]), int(exit_px[1]))),
                                        ((int(exit_px[0]), int(exit_px[1])), (ex_cx, ex_cy))
                                    ])
                                    color_cuts.append(offset + k)
                            final_path = concat_paths(final_path, seg_goal)
                            terminal_rc = best_goal
                            goal_chosen = best_goal
                            reached_goal = True

            # 13) Stairs-only fallback with backtracking
            if (not ordered_pairs) and (goal_chosen is None):
                if current_rc is None:
                    distS, parentS = bfs_from_sources(grid_walk, starts, diagonal=ALLOW_DIAGONAL,
                                                      liquid_mask=liquid_mask, arrow_edges=arrow_edges)
                    current_rc = starts[0]
                    if np.max(distS) >= 0:
                        far = tuple(np.argwhere(distS == np.max(distS))[0])
                        final_path = concat_paths(final_path, reconstruct_path_with_tele(parentS, far))
                        current_rc = far

                stairs_pairs = []
                stairs_rc = []
                for r in rows:
                    mo = (r.get("manual_override","") or "").strip().lower()
                    if "stairs" in mo:
                        rr = int(float(r.get("row",0))); cc = int(float(r.get("col",0)))
                        stairs_rc.append((rr,cc))
                for idx_i, (a_rc, b_rc) in enumerate(combinations(stairs_rc, 2)):
                    (r1,c1), (r2,c2) = a_rc, b_rc
                    d = abs(r1-r2) + abs(c1-c2)
                    x1,y1 = px_center_of_rc(r1,c1); x2,y2 = px_center_of_rc(r2,c2)
                    e1 = nearest_walkable_tile_around_rc(grid_walk, r1, c1) or (r1,c1)
                    e2 = nearest_walkable_tile_around_rc(grid_walk, r2, c2) or (r2,c2)
                    stairs_pairs.append((d, f"stairs_{idx_i}", (x1,y1,e1), (x2,y2,e2)))
                stairs_pairs.sort(key=lambda t: t[0])
                stairs_pairs = [(tag, a, b) for (_d, tag, a, b) in stairs_pairs]

                if stairs_pairs and end_rc_list:
                    pre_position_rc = final_path[-1] if final_path else current_rc
                    for tag, (xA, yA, eA), (xB, yB, eB) in stairs_pairs:
                        distA, parA = bfs_single_source(grid_walk, pre_position_rc, diagonal=ALLOW_DIAGONAL,
                                                        liquid_mask=liquid_mask, arrow_edges=arrow_edges)
                        candE = _nearest_target(distA, [eA, eB])
                        if candE is None:
                            continue

                        path_to_entry = reconstruct_path_with_tele(parA, candE)
                        final_path = concat_paths(final_path, path_to_entry)

                        entry_is_A = (candE == eA)
                        exit_rc  = eB if entry_is_A else eA
                        entry_px = (xA, yA) if entry_is_A else (xB, yB)
                        exit_px  = (xB, yB) if entry_is_A else (xA, yA)

                        en_cx, en_cy = px_center_of_rc(*candE)
                        ex_cx, ex_cy = px_center_of_rc(*exit_rc)
                        portal_segments_px.extend([
                            ((en_cx, en_cy), (int(entry_px[0]), int(entry_px[1]))),
                            ((int(entry_px[0]), int(entry_px[1])), (int(exit_px[0]), int(exit_px[1]))),
                            ((int(exit_px[0]), int(exit_px[1])), (ex_cx, ex_cy))
                        ])
                        color_cuts.append(len(final_path))

                        distG, parG = bfs_single_source(grid_walk, exit_rc, diagonal=ALLOW_DIAGONAL,
                                                        liquid_mask=liquid_mask, arrow_edges=arrow_edges)
                        best_goal = _nearest_target(distG, end_rc_list)
                        if best_goal is not None:
                            seg_to_goal = reconstruct_path_with_tele(parG, best_goal)
                            final_path = concat_paths(final_path, seg_to_goal)
                            terminal_rc = best_goal
                            goal_chosen = best_goal
                            reached_goal = True
                            break
                        else:
                            backtrack = list(reversed(path_to_entry))
                            if backtrack and backtrack[0] == backtrack[-1]:
                                backtrack = backtrack[1:]
                            final_path = concat_paths(final_path, backtrack)
                            terminal_rc = pre_position_rc

                # Nearest goal if still none
                if (goal_chosen is None) and end_rc_list and final_path:
                    current_rc2 = final_path[-1]
                    distG, parG = bfs_single_source(grid_walk, current_rc2, diagonal=ALLOW_DIAGONAL,
                                                    liquid_mask=liquid_mask, arrow_edges=arrow_edges)
                    best_goal = _nearest_target(distG, end_rc_list)
                    if best_goal is not None:
                        seg_to_goal = reconstruct_path_with_tele(parG, best_goal)
                        final_path = concat_paths(final_path, seg_to_goal)
                        terminal_rc = best_goal
                        goal_chosen = best_goal
                        reached_goal = True
                    else:
                        terminal_rc = current_rc2

            # 14) Write overlay + CSVs + summary for this map
            overlay_png = OUT_DIR / f"overlay_shortest_path_{keyname}.png"
            overlay = draw_overlay_with_markers(
                combined_rgba, final_path, starts, terminal_rc, reached_goal,
                portal_segments_px=portal_segments_px, color_cuts=color_cuts
            )
            cv2.imwrite(str(overlay_png), overlay)

            csv_path = ""
            if final_path and len(final_path) >= 2:
                csv_path = OUT_DIR / f"shortest_path_{keyname}.csv"
                with Path(csv_path).open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["row","col","y_px","x_px"])
                    for (r,c) in final_path:
                        y = r*TILE_H + TILE_H/2; x = c*TILE_W + TILE_W/2
                        w.writerow([r,c,y,x])

            strategy = "keys_first+" + ("alphabet_forced_sequence_with_reuse" if ordered_pairs else "stairs_or_direct")
            summary[keyname] = {
                "source": base_source_key,
                "grid_size": [H, W],
                "start_candidates": starts,
                "goal_candidates": list(end_rc_list),
                "goal_chosen": terminal_rc,
                "reached_goal": bool(reached_goal),
                "path_length": len(final_path) if final_path else 0,
                "overlay_path": str(overlay_png),
                "path_csv": str(csv_path) if csv_path else "",
                "strategy": strategy,
                "portals_detected": [p[0] for p in ordered_pairs],
                "directed_pairs_cfg": sorted(f"{a}>{b}" for (a,b) in DIRECTED_PORTAL_INDEX_PAIRS),
                "reuse_enabled": bool(REUSE_VISITED_PORTALS),
                "keys_total": keys_total,
                "bombs_total": bombs_total,
                "keys_collected": keys_collected,
                "bombs_collected": bombs_collected,
                "ladders_total": ladders_total,
                "ladders_collected": ladders_collected,
                "collectible_types": ["ladder","key","bomb"],
                "enemies_total": enemies_total,
                "enemies_visited": enemies_visited,
                "oneway_openings": {f"{r},{c}": d for (r,c), d in (arrow_cells or {}).items()},
                "doors_found": [(int(r), int(c)) for (r,c,_) in all_doors],
                "rooms_count": int(n_rooms),
            }

        except Exception as e:
            print(f"[ERROR] {base_source_key}: {e}")
            summary[name_for(base_source_key)] = {"source": base_source_key, "error": str(e)}

    with (OUT_DIR / "shortest_path_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Done. Wrote {OUT_DIR/'shortest_path_summary.json'}")

    # Rollup CSV
    csv_report_path = OUT_DIR / "collectibles_summary.csv"
    with csv_report_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "map",
            "ladders_collected","ladders_total",
            "keys_collected","keys_total",
            "bombs_collected","bombs_total",
            "enemies_visited","enemies_total",
            "rooms_count","path_csv","overlay_png"
        ])

        for m, info in sorted(summary.items()):
            w.writerow([
                m,
                info.get("ladders_collected", 0), 
                info.get("ladders_total", 0),
                info.get("keys_collected", 0),    
                info.get("keys_total", 0),
                info.get("bombs_collected", 0),   
                info.get("bombs_total", 0),
                info.get("enemies_visited", 0),   
                info.get("enemies_total", 0),
                info.get("rooms_count", 0),
                info.get("path_csv", ""),
                info.get("overlay_path", ""),
            ])

    print(f"Wrote {csv_report_path}")

if __name__ == "__main__":
    main()
