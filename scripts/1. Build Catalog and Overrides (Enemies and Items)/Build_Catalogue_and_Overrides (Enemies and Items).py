#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract individual sprites from a map that is tiles+sprites, aligned to a base legend,
then BUILD CATALOGUES (master + per-source) of the extracted sprites,
and overlay indices on the original maps.

What this does
--------------
- Uses tiles/ (16x16) to remove tile background
- Finds connected components (sprites) and saves each to zelda_out/tiles_other/
- Writes zelda_out/legend_other.csv with ORIGINAL MAP positions (x,y,w,h) and tile spans
- Ensures the position system follows legend_base.csv so outputs can overlay
  perfectly on the base reconstructed file.
- Builds visual catalogues (PNG pages, plus PDF if Pillow is present)
- (NEW) Writes per-source maps with indices overlaid at each sprite position:
  zelda_out/overlays_indexed/<source>_indexed.png

Alignment guardrails
--------------------
- Reads legend_base.csv to learn the canonical canvas size per "source"
- Normalizes source names: e.g., "tloz1_1e.png" -> "tloz1_1"
- If the incoming map's canvas doesn't match the base:
    * If it's a clean uniform scale to the base, auto-resize (nearest).
    * Else, skip with a warning (to avoid corrupt coordinates).
"""

import csv
import cv2
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Try Pillow for optional PDF export
try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

# PARAMS
INPUT_DIR    = Path("../Data")                    # where your overlaid map PNGs live
TILES_DIR    = Path("../Tiles_Base")                # directory of 16x16 tiles
OUT_ROOT     = Path("../Tiles_Other")
OUT_OTHERS   = Path("../Tiles_Other")
OUT_DEBUG    = Path("../Tiles_Other")
OUT_CATALOG  = Path("../Tiles_Other")
OUT_OVERLAYS = Path("../Tiles_Other")
LEGEND_PATH  = OUT_ROOT / "legend_other.csv"

# Base legend (canonical coordinate system for overlay)
BASE_LEGEND  = Path("legend_all_tiles.csv")

# Grid
TILE_W, TILE_H = 16, 16
OFFSET_X, OFFSET_Y = 0, 0                   # grid offset if any (usually 0,0)

# Matching / extraction
TOLERANCE   = 0                              # 0 exact; try 1..2 if slight palette drift
BATCH_SIZE  = 512
MIN_AREA_PX = 8                              # ignore tiny specks
CONNECTIVITY = 8
MORPH_OPEN   = 0                             # >0 to denoise component mask (radius px)

# Catalogue layout
CAT_PAGE_W  = 2200
CAT_PAGE_H  = 3000
CAT_MARGIN  = 40
CELL_W      = 300
CELL_H      = 260
CELL_PAD    = 16
THUMB_MAX_W = 240
THUMB_MAX_H = 160
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FSZ_LBL     = 0.5
FSZ_HDR     = 0.7
THICK_LBL   = 1
THICK_HDR   = 2
GRID_COLOR  = (200, 200, 200, 255)
TEXT_COLOR  = (20, 20, 20, 255)
BG_COLOR    = (245, 245, 245, 255)
CELL_COLOR  = (255, 255, 255, 255)

# Index overlay options
MAKE_INDEX_OVERLAYS = True
SEARCH_DIRS         = [Path("."), Path("sources"), Path("sources_with_others")]

# Drawing config
IDX_FONT_SCALE_BASE = 0.5
IDX_THICK_OUTLINE   = 3
IDX_THICK_FILL      = 1
IDX_DRAW_BBOX       = False
IDX_BOX_COLOR       = (0, 0, 255)     # BGR
DOT_RADIUS_MIN      = 2
DOT_RADIUS_MAX      = 6
DOT_COLOR_INNER     = (255, 255, 0)   # BGR (yellow-ish)
DOT_COLOR_BASE      = (0, 0, 0)       # BGR (black)

def ensure_dirs(*paths):
    for p in paths: p.mkdir(parents=True, exist_ok=True)

def to_rgba(img):
    if img is None:
        return None
    if img.ndim == 2:  # gray
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    if img.shape[2] == 4:  # BGRA -> RGBA
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)

def md5_rgba(arr):
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()

def load_tiles(tiles_dir: Path, tw: int, th: int):
    paths = sorted(tiles_dir.glob("*.png"))
    stack = []
    hash_dict = {}
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        rgba = to_rgba(im)
        if rgba is None or rgba.shape[:2] != (th, tw):
            continue
        stack.append(rgba)
    if not stack:
        raise RuntimeError(f"No {tw}x{th} tiles found in {tiles_dir}")
    tile_stack = np.stack(stack, axis=0)  # (K,H,W,4)
    for i in range(tile_stack.shape[0]):
        hash_dict[md5_rgba(tile_stack[i])] = i
    return tile_stack, hash_dict

def best_equal_mask(patch_rgba, tile_stack, tol, batch):
    """Return equality mask vs best matching tile."""
    P = patch_rgba.astype(np.int16)
    K = tile_stack.shape[0]
    best_count, best_mask = -1, None
    for s in range(0, K, batch):
        batch_tiles = tile_stack[s:s+batch].astype(np.int16)      # (b,H,W,4)
        diff = np.abs(batch_tiles - P)
        eq = (diff <= tol).all(axis=-1).astype(np.uint8)          # (b,H,W)
        counts = eq.reshape(eq.shape[0], -1).sum(axis=1)
        j = int(np.argmax(counts))
        if counts[j] > best_count:
            best_count = int(counts[j])
            best_mask = eq[j]
    return best_mask, best_count

def sprites_only_from_map(map_rgba, tile_stack, hash_dict, tol, batch):
    """Return RGBA where only sprite pixels remain (tile background cleared)."""
    H, W = map_rgba.shape[:2]
    out = np.zeros((H, W, 4), dtype=np.uint8)
    cols = (W - OFFSET_X) // TILE_W
    rows = (H - OFFSET_Y) // TILE_H

    for r in range(rows):
        y0 = OFFSET_Y + r*TILE_H; y1 = y0 + TILE_H
        for c in range(cols):
            x0 = OFFSET_X + c*TILE_W; x1 = x0 + TILE_W
            if y1 > H or x1 > W:  # clamp (defensive)
                continue
            patch = map_rgba[y0:y1, x0:x1, :]
            # fast path: exact tile match -> everything is background
            if tol == 0 and md5_rgba(patch) in hash_dict:
                continue
            eq_mask, _ = best_equal_mask(patch, tile_stack, tol, batch)
            if eq_mask is None:
                out[y0:y1, x0:x1, :] = patch
                continue
            keep = (eq_mask == 0)
            if np.any(keep):
                cell = out[y0:y1, x0:x1, :]
                cell[keep] = patch[keep]
                out[y0:y1, x0:x1, :] = cell
    return out

def connected_sprites(sprites_rgba):
    alpha = sprites_rgba[...,3]
    mask = (alpha > 0).astype(np.uint8)
    if MORPH_OPEN > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN, MORPH_OPEN))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=CONNECTIVITY)
    return num, labels, stats

def save_sprite_cv2(rgba, out_path: Path):
    if rgba is None or rgba.size == 0:
        return False
    h, w = rgba.shape[:2]
    if h <= 0 or w <= 0:
        return False
    bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    return bool(cv2.imwrite(str(out_path), bgra))

# Base legend handling (for overlay alignment)

def norm_source_key(name: str) -> str:
    """
    Normalize a filename to match base legend 'source' names:
    - strip directory & extension
    - lower-case and replace '-' with '_'
    - drop trailing '_e'
    - drop trailing 'e' if preceded by a digit (e.g., 'tloz1_1e' -> 'tloz1_1')
    """
    stem = Path(name).stem.lower().replace("-", "_")
    if stem.endswith("_e"):
        stem = stem[:-2]
    if len(stem) >= 2 and stem[-1] == "e" and stem[-2].isdigit():
        stem = stem[:-1]
    return stem

def load_base_canvas_sizes(base_csv: Path) -> Dict[str, Tuple[int, int]]:
    sizes: Dict[str, Tuple[int,int]] = {}
    if not base_csv.exists():
        return sizes
    max_hw: Dict[str, Tuple[int,int]] = {}
    with open(base_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row.get("source") or row.get("map") or ""
            if not src:
                continue
            key = norm_source_key(src)
            try:
                x = int(float(row["x"])); y = int(float(row["y"]))
                w = int(float(row["w"])); h = int(float(row["h"]))
            except Exception:
                continue
            maxW, maxH = max_hw.get(key, (0,0))
            max_hw[key] = (max(maxW, x + w), max(maxH, y + h))
    for k, (W, H) in max_hw.items():
        sizes[k] = (W, H)
    return sizes

def align_to_base_if_needed(rgba: np.ndarray,
                            map_name: str,
                            base_sizes: Dict[str, Tuple[int,int]]) -> Tuple[Optional[np.ndarray], str]:
    key = norm_source_key(map_name)
    if key not in base_sizes:
        return rgba, ""
    H, W = rgba.shape[:2]
    baseW, baseH = base_sizes[key]
    if (W, H) == (baseW, baseH):
        return rgba, ""
    fx = baseW / float(W)
    fy = baseH / float(H)
    if abs(fx - fy) < 1e-6 and fx > 0:
        resized = cv2.resize(rgba, (baseW, baseH), interpolation=cv2.INTER_NEAREST)
        return resized, f"[align] {map_name}: resized {W}x{H} -> {baseW}x{baseH} to match base"
    else:
        return None, (f"[skip] {map_name}: size {W}x{H} does not match base {baseW}x{baseH} "
                      f"and is not a uniform scale; skipping to preserve coordinate fidelity.")

# CATALOGUE BUILDERS

def _blank_page():
    page = np.zeros((CAT_PAGE_H, CAT_PAGE_W, 4), dtype=np.uint8)
    page[...] = BG_COLOR
    return page

def _draw_header(page: np.ndarray, title: str, y: int = None):
    if y is None: y = CAT_MARGIN
    cv2.putText(page, title, (CAT_MARGIN, y+20), FONT, FSZ_HDR, TEXT_COLOR, THICK_HDR, cv2.LINE_AA)

def _grid_dims():
    inner_w = CAT_PAGE_W - 2*CAT_MARGIN
    inner_h = CAT_PAGE_H - 2*CAT_MARGIN - 40
    cols = max(1, inner_w // CELL_W)
    rows = max(1, inner_h // CELL_H)
    return int(cols), int(rows)

def _paste_rgba(dst: np.ndarray, src: np.ndarray, x: int, y: int):
    h, w = src.shape[:2]
    if x < 0 or y < 0 or x+w > dst.shape[1] or y+h > dst.shape[0]:
        w = min(w, dst.shape[1] - x)
        h = min(h, dst.shape[0] - y)
        if w <= 0 or h <= 0: return
        src = src[:h, :w]
    alpha = (src[...,3:4].astype(np.float32) / 255.0)
    inv   = 1.0 - alpha
    dst_region = dst[y:y+h, x:x+w, :3].astype(np.float32)
    src_rgb    = src[..., :3].astype(np.float32)
    out_rgb = alpha * src_rgb + inv * dst_region
    dst[y:y+h, x:x+w, :3] = out_rgb.astype(np.uint8)
    dst[y:y+h, x:x+w, 3] = np.maximum(dst[y:y+h, x:x+w, 3], src[...,3])

def _thumb(sprite_rgba: np.ndarray) -> np.ndarray:
    h, w = sprite_rgba.shape[:2]
    scale = min(THUMB_MAX_W / max(1, w), THUMB_MAX_H / max(1, h))
    scale = min(1.0, scale)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(sprite_rgba, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def _draw_cell(page: np.ndarray, cell_x: int, cell_y: int, info: dict):
    x0 = CAT_MARGIN + cell_x * CELL_W
    y0 = CAT_MARGIN + 40 + cell_y * CELL_H
    x1 = x0 + CELL_W - 1
    y1 = y0 + CELL_H - 1
    cv2.rectangle(page, (x0, y0), (x1, y1), GRID_COLOR, 1)

    ix0 = x0 + CELL_PAD
    iy0 = y0 + CELL_PAD
    ix1 = x1 - CELL_PAD
    iy1 = y1 - CELL_PAD

    cv2.rectangle(page, (ix0, iy0), (ix1, iy1), CELL_COLOR, -1)

    thumb = _thumb(info["img"])
    th, tw = thumb.shape[:2]
    cx = ix0 + (ix1 - ix0 - tw)//2
    cy = iy0
    _paste_rgba(page, thumb, cx, cy)

    lbl_y = cy + th + 16
    label1 = f"#{info['index']:04d}  {info['filename']}"
    label2 = f"{info['source']}   {info['w']}x{info['h']} px   tiles [{info['r0']},{info['c0']}]→[{info['r1']},{info['c1']}]"
    cv2.putText(page, label1[:48], (ix0, lbl_y), FONT, FSZ_LBL, TEXT_COLOR, THICK_LBL, cv2.LINE_AA)
    cv2.putText(page, label2[:52], (ix0, lbl_y+18), FONT, FSZ_LBL, TEXT_COLOR, THICK_LBL, cv2.LINE_AA)

def build_catalogue_pages(items: List[dict], title: str, out_dir: Path) -> List[Path]:
    ensure_dirs(out_dir)
    cols, rows = _grid_dims()
    per_page = cols * rows

    pages_png: List[Path] = []
    total = len(items)
    if total == 0:
        page = _blank_page()
        _draw_header(page, title)
        pth = out_dir / "page_000.png"
        cv2.imwrite(str(pth), cv2.cvtColor(page, cv2.COLOR_RGBA2BGRA))
        pages_png.append(pth)
        return pages_png

    for pg, start in enumerate(range(0, total, per_page)):
        page = _blank_page()
        _draw_header(page, f"{title}  (page {pg+1}/{(total-1)//per_page + 1})")
        subset = items[start:start+per_page]
        for i, info in enumerate(subset):
            r = i // cols
            c = i % cols
            _draw_cell(page, c, r, info)
        pth = out_dir / f"page_{pg:03d}.png"
        cv2.imwrite(str(pth), cv2.cvtColor(page, cv2.COLOR_RGBA2BGRA))
        pages_png.append(pth)

    if PIL_OK and pages_png:
        try:
            imgs = [Image.open(str(p)).convert("RGB") for p in pages_png]
            pdf_path = out_dir / "catalogue.pdf"
            imgs[0].save(str(pdf_path), save_all=True, append_images=imgs[1:])
        except Exception:
            pass
    return pages_png

# INDEX OVERLAY HELPERS

def _find_source_image(source_name: str) -> Optional[Path]:
    """
    Locate the source map image across SEARCH_DIRS.
    Tries exact name and common variants (with/without trailing 'e' or '_e').
    """
    src = Path(source_name)
    cand = []

    cand.append(src.name)
    st = src.stem
    if st.endswith("_e"):
        cand.append(st[:-2] + src.suffix)
    if len(st) >= 2 and st[-1] == "e" and st[-2].isdigit():
        cand.append(st[:-1] + src.suffix)
    if src.suffix.lower() != ".png":
        cand.append(st + ".png")

    for d in SEARCH_DIRS:
        for c in cand:
            p = d / c
            if p.exists():
                return p

    # fallback: match normalized stem
    key = norm_source_key(source_name)
    for d in SEARCH_DIRS:
        for p in d.glob("*.png"):
            if norm_source_key(p.name) == key:
                return p
    return None

def _draw_text_with_outline(img_bgr, text: str, org, scale: float):
    cv2.putText(img_bgr, text, org, FONT, scale, (0,0,0), IDX_THICK_OUTLINE, cv2.LINE_AA)
    cv2.putText(img_bgr, text, org, FONT, scale, (255,255,255), IDX_THICK_FILL, cv2.LINE_AA)

def _overlay_indices_for_source(source_name: str, rows: List[List], out_dir: Path) -> bool:
    """
    rows contain: [idx, filename, src_name, x, y, w, h, r0, c0, r1, c1, area]
    """
    p = _find_source_image(source_name)
    if p is None:
        print(f"[warn] Source image not found for '{source_name}' in {SEARCH_DIRS}")
        return False
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[warn] Cannot read image: {p}")
        return False

    # normalize to BGR for drawing
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        img_bgr = img.copy()

    H, W = img_bgr.shape[:2]
    drawn = 0
    for r in rows:
        try:
            idx = int(r[0]); x = int(float(r[3])); y = int(float(r[4]))
            w = int(float(r[5])); h = int(float(r[6]))
        except Exception:
            continue

        x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
        w = max(1, min(w, W - x)); h = max(1, min(h, H - y))
        cx = x + w // 2; cy = y + h // 2

        if IDX_DRAW_BBOX:
            cv2.rectangle(img_bgr, (x, y), (x+w-1, y+h-1), IDX_BOX_COLOR, 1)

        # dot
        rad = max(DOT_RADIUS_MIN, min(DOT_RADIUS_MAX, int(round(min(w, h) * 0.08))))
        cv2.circle(img_bgr, (cx, cy), rad, DOT_COLOR_BASE, -1)
        cv2.circle(img_bgr, (cx, cy), max(1, rad-1), DOT_COLOR_INNER, -1)

        # text
        scale = max(0.4, IDX_FONT_SCALE_BASE * (0.5 + 0.5 * min(1.5, (w*h) / (24*24))))
        tx = min(W-1, cx + rad + 3)
        ty = max(12, cy - rad - 3)
        _draw_text_with_outline(img_bgr, f"{idx:04d}", (tx, ty), scale)
        drawn += 1

    out_name = f"{norm_source_key(source_name)}_indexed.png"
    out_path = out_dir / out_name
    cv2.imwrite(str(out_path), img_bgr)
    print(f"[ok] {source_name}: {drawn} index label(s) → {out_path}")
    return True

# MAIN

def main():
    ensure_dirs(OUT_ROOT, OUT_OTHERS, OUT_DEBUG, OUT_CATALOG, OUT_CATALOG / "by_source", OUT_CATALOG / "master", OUT_OVERLAYS)

    # 0) Load base canvas sizes to enforce overlay compatibility
    base_sizes = load_base_canvas_sizes(BASE_LEGEND)
    if base_sizes:
        print(f"[info] Loaded base overlay sizes for {len(base_sizes)} source(s) from {BASE_LEGEND}")

    # 1) Load tiles + resume-safe index for others
    tile_stack, hash_dict = load_tiles(TILES_DIR, TILE_W, TILE_H)

    exist_hash2idx = {}
    for p in OUT_OTHERS.glob("other_*.png"):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        rgba = to_rgba(im)
        if rgba is None:
            continue
        exist_hash2idx[md5_rgba(rgba)] = int(p.stem.split("_")[1])
    next_idx = (max(exist_hash2idx.values()) + 1) if exist_hash2idx else 0

    legend_rows: List[List] = []

    # 2) Process maps (PNG files only)
    for mp in sorted(INPUT_DIR.glob("*.png")):
        try:
            if TILES_DIR.resolve() in mp.resolve().parents or OUT_ROOT.resolve() in mp.resolve().parents:
                continue
        except Exception:
            pass

        src = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if src is None:
            print(f"[warn] cannot read {mp.name}, skipping.")
            continue
        rgba = to_rgba(src)
        if rgba is None:
            print(f"[warn] cannot convert {mp.name} to RGBA, skipping.")
            continue

        rgba, note = align_to_base_if_needed(rgba, mp.name, base_sizes)
        if rgba is None:
            print(note); continue
        if note: print(note)

        H, W = rgba.shape[:2]

        # 3) remove tiles -> sprites-only RGBA
        sprites_rgba = sprites_only_from_map(rgba, tile_stack, hash_dict, TOLERANCE, BATCH_SIZE)

        # 4) connected components
        num, labels, stats = connected_sprites(sprites_rgba)
        if num <= 1:
            print(f"[info] {mp.name}: no sprites found")
            continue

        qc = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA).copy()
        kept = 0
        for lab in range(1, num):
            x, y, w, h, area = map(int, stats[lab])
            x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
            w = max(0, min(w, W - x)); h = max(0, min(h, H - y))
            if w == 0 or h == 0 or area < MIN_AREA_PX:
                continue

            sub_mask = (labels[y:y+h, x:x+w] == lab).astype(np.uint8)
            if sub_mask.sum() == 0:
                continue

            sub_rgb = sprites_rgba[y:y+h, x:x+w, :3]
            sprite_rgba = np.zeros((h, w, 4), dtype=np.uint8)
            sprite_rgba[..., :3] = sub_rgb
            sprite_rgba[..., 3] = sub_mask * 255

            hsh = md5_rgba(sprite_rgba)
            if hsh in exist_hash2idx:
                idx = exist_hash2idx[hsh]
            else:
                idx = next_idx
                out_path = OUT_OTHERS / f"other_{idx:04d}.png"
                ok = save_sprite_cv2(sprite_rgba, out_path)
                if not ok:
                    continue
                exist_hash2idx[hsh] = idx
                next_idx += 1

            c0 = max(0, (x - OFFSET_X) // TILE_W)
            r0 = max(0, (y - OFFSET_Y) // TILE_H)
            c1 = max(0, (x + w - 1 - OFFSET_X) // TILE_W)
            r1 = max(0, (y + h - 1 - OFFSET_Y) // TILE_H)

            legend_rows.append([
                idx, f"other_{idx:04d}.png", mp.name,
                x, y, w, h,
                int(r0), int(c0), int(r1), int(c1),
                int(w*h)
            ])
            kept += 1

            cv2.rectangle(qc, (x, y), (x+w-1, y+h-1), (0, 0, 255, 255), 1)

        cv2.imwrite(str((OUT_DEBUG / f"qc_{mp.stem}.png")), qc)
        print(f"[ok] {mp.name}: extracted {kept} sprite(s), QC → {OUT_DEBUG / f'qc_{mp.stem}.png'}")

    # 5) Write legend
    ensure_dirs(LEGEND_PATH.parent)
    legend_rows.sort(key=lambda r: (norm_source_key(r[2]), r[0]))
    with open(LEGEND_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "other_index","filename","source","x","y","w","h",
            "tile_r0","tile_c0","tile_r1","tile_c1","area_px"
        ])
        for row in legend_rows:
            w.writerow(row)

    print(f"[ok] Sprites → {OUT_OTHERS}")
    print(f"[ok] Legend  → {LEGEND_PATH}")
    print("[hint] Overlay check: use legend_base.csv + legend_other.csv to composite base + others by (x,y,w,h).")

    # 6) Build catalogues
    items_master: List[dict] = []
    by_source: Dict[str, List[dict]] = {}

    for row in legend_rows:
        idx, fname, src_name, x, y, w, h, r0, c0, r1, c1, area = row
        img_path = OUT_OTHERS / fname
        im_bgra = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if im_bgra is None:
            continue
        im_rgba = to_rgba(im_bgra)
        info = {
            "index": int(idx),
            "filename": fname,
            "source": norm_source_key(src_name),
            "w": int(w), "h": int(h),
            "r0": int(r0), "c0": int(c0), "r1": int(r1), "c1": int(c1),
            "img": im_rgba
        }
        items_master.append(info)
        by_source.setdefault(info["source"], []).append(info)

    items_master.sort(key=lambda d: (d["source"], d["index"]))
    master_dir = OUT_CATALOG / "master"
    build_catalogue_pages(items_master, "Sprite Catalogue (MASTER)", master_dir)

    bs_dir = OUT_CATALOG / "by_source"
    for src_key, lst in sorted(by_source.items()):
        lst.sort(key=lambda d: d["index"])
        safe_name = src_key.replace("/", "_")
        pages = build_catalogue_pages(lst, f"Source: {src_key}", bs_dir / safe_name)
        print(f"[ok] Catalogue for {src_key}: {len(pages)} page(s)")

    print(f"[ok] Catalogues → {OUT_CATALOG} (PNG pages{' + PDF' if PIL_OK else ''})")

    # 7) Build indexed overlays (NEW)
    if MAKE_INDEX_OVERLAYS and legend_rows:
        # regroup legend_rows by original 'source' (exact column 2)
        groups: Dict[str, List[List]] = {}
        for r in legend_rows:
            groups.setdefault(r[2], []).append(r)
        made = 0
        for src_name, rows in groups.items():
            if _overlay_indices_for_source(src_name, rows, OUT_OVERLAYS):
                made += 1
        print(f"[ok] Indexed overlays → {OUT_OVERLAYS} ({made} map(s))")

if __name__ == "__main__":
    main()
