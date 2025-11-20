#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zelda_extract_tiles_no_overlay_highdpi_with_axes.py
---------------------------------------------------
- Builds per-map catalogues at 450 DPI
- Adds visible axes: Row numbers (left) and Col numbers (top)
- No map overlaps (each map has its own catalogue)
- Deduplicates tiles across sources
- Auto classifies air/solid
- Skips black/blank tiles
- Outputs JSON + CSV legends for manual override
"""

import cv2
import csv
import json
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR          = Path("../Data/")
OUT_DIR_TILES      = Path("../Tiles_Base")
OUT_DIR_ROOT       = Path("../")
OUT_CATALOGUE      = Path("../Catalogue/")
LEGEND_PATH_JSON   = OUT_DIR_ROOT / "legend_all_tiles.json"
LEGEND_PATH_CSV    = OUT_DIR_ROOT / "legend_all_tiles.csv"

# Tile settings
TILE_SIZE          = 16
TILE_SPACING       = 2
LABEL_FONT_SIZE    = 7
DRAW_INDEX         = True

# Classification thresholds
AIR_THRESHOLD      = 0.5
BLACK_THRESHOLD    = 15

# Combined master catalogue settings
MAKE_MASTER_CATALOGUE = True
MASTER_GAP_BETWEEN_MAPS = 32

# DPI for saved images
SAVE_DPI = (450, 450)

# Optional visual aids
DRAW_GRIDLINES     = False
GRID_ALPHA         = 60

# Axes settings
SHOW_AXES          = True
AXIS_BG_ALPHA      = 140      # background bar opacity for axes
AXIS_TICK_STEP     = 1        # label every N tiles
AXIS_PAD_LEFT      = 36       # left margin for row labels
AXIS_PAD_TOP       = 24       # top margin for col labels
AXIS_FONT_SIZE     = 10       # axis label font
AXIS_TITLE_GAP     = 2        # gap between top/left bars and "Col"/"Row"
AXIS_TITLE_FONT_SZ = 10

# ============================================================
# UTILITIES
# ============================================================

def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def tile_hash_rgba(tile_rgba: np.ndarray) -> str:
    return hashlib.md5(tile_rgba.tobytes()).hexdigest()

def is_black_tile(tile_arr: np.ndarray) -> bool:
    # Treat fully transparent as "blank/black"
    if tile_arr.shape[2] >= 4:
        alpha = tile_arr[:, :, 3]
        if np.all(alpha == 0):
            return True
    rgb = tile_arr[:, :, :3]
    return float(np.mean(rgb)) < BLACK_THRESHOLD

def classify_tile(tile_arr: np.ndarray, threshold: float = AIR_THRESHOLD) -> str:
    if tile_arr.shape[2] < 4:
        return "solid"
    alpha = tile_arr[:, :, 3]
    opaque_ratio = float(np.sum(alpha > 10)) / alpha.size
    return "air" if opaque_ratio < threshold else "solid"

def load_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        # Fallback to default PIL font
        return ImageFont.load_default()

def draw_index_label(img: Image.Image, text: str, xy, font=None):
    d = ImageDraw.Draw(img)
    if font is None:
        font = load_font(LABEL_FONT_SIZE)
    d.text(xy, text, fill=(255,255,255,220), font=font)

def draw_grid(img: Image.Image, x0: int, y0: int, width: int, height: int, tile_pitch: int, alpha=GRID_ALPHA):
    d = ImageDraw.Draw(img)
    color = (255,255,255,alpha)
    # verticals
    for x in range(x0, x0 + width + 1, tile_pitch):
        d.line([(x, y0), (x, y0 + height)], fill=color, width=1)
    # horizontals
    for y in range(y0, y0 + height + 1, tile_pitch):
        d.line([(x0, y), (x0 + width, y)], fill=color, width=1)

def draw_axes(canvas: Image.Image, rows: int, cols: int, tile_size: int, tile_spacing: int,
              axis_left: int, axis_top: int, tick_step: int = 1):
    """
    Draw a top bar with column labels and a left bar with row labels.
    Tiles start at origin (axis_left, axis_top) on the canvas.
    """
    d = ImageDraw.Draw(canvas)
    font_tick  = load_font(AXIS_FONT_SIZE)
    font_title = load_font(AXIS_TITLE_FONT_SZ)

    pitch = tile_size + tile_spacing
    # Top axis background bar
    top_bar = Image.new("RGBA", (canvas.size[0], axis_top), (0, 0, 0, AXIS_BG_ALPHA))
    canvas.alpha_composite(top_bar, (0, 0))
    # Left axis background bar
    left_bar = Image.new("RGBA", (axis_left, canvas.size[1]), (0, 0, 0, AXIS_BG_ALPHA))
    canvas.alpha_composite(left_bar, (0, 0))

    # Column tick labels
    for c in range(0, cols, max(1, tick_step)):
        cx = axis_left + c * pitch + tile_size // 2
        # center text horizontally
        text = str(c)
        tw, th = d.textbbox((0,0), text, font=font_tick)[2:]
        d.text((cx - tw // 2, (axis_top - th) // 2), text, fill=(255,255,255,230), font=font_tick)

    # Row tick labels
    for r in range(0, rows, max(1, tick_step)):
        cy = axis_top + r * pitch + tile_size // 2
        text = str(r)
        tw, th = d.textbbox((0,0), text, font=font_tick)[2:]
        d.text((axis_left - tw - 6, cy - th // 2), text, fill=(255,255,255,230), font=font_tick)

    # Axis titles
    # "Col" centered in top-left bar above the first tiles
    title_col = "Col"
    ttw, tth = d.textbbox((0,0), title_col, font=font_title)[2:]
    d.text((axis_left + 4, max(2, (axis_top - tth) // 2 - AXIS_TITLE_GAP)),
           title_col, fill=(200,200,255,230), font=font_title)

    # "Row" along left bar near the top
    title_row = "Row"
    rtw, rth = d.textbbox((0,0), title_row, font=font_title)[2:]
    d.text((max(2, (axis_left - rtw)//2), axis_top + 4),
           title_row, fill=(200,255,200,230), font=font_title)

# ============================================================
# MAIN LOGIC
# ============================================================

def main():
    ensure_dirs(OUT_DIR_ROOT, OUT_DIR_TILES, OUT_CATALOGUE)

    pngs = sorted(INPUT_DIR.glob("*.png"))
    if not pngs:
        print("[!] No PNG files found.")
        return

    legend = {
        "tile_size": TILE_SIZE,
        "tile_spacing": TILE_SPACING,
        "air_threshold": AIR_THRESHOLD,
        "black_threshold": BLACK_THRESHOLD,
        "sources": [],
        "tiles": [],
        "unique_tiles": []
    }

    hash_to_index = {}
    index_meta = {}
    next_index = 0
    master_sections = []

    for png_path in pngs:
        src = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if src is None:
            print(f"[warn] Cannot read {png_path.name}, skipping.")
            continue

        legend["sources"].append(png_path.name)
        h, w = src.shape[:2]
        rows, cols = h // TILE_SIZE, w // TILE_SIZE

        # size of the tile area (without axes)
        tile_pitch = TILE_SIZE + TILE_SPACING
        cat_w = w + (cols - 1) * TILE_SPACING
        cat_h = h + (rows - 1) * TILE_SPACING

        # final canvas includes axes margins
        axis_left = AXIS_PAD_LEFT if SHOW_AXES else 0
        axis_top  = AXIS_PAD_TOP if SHOW_AXES else 0
        canvas_w  = axis_left + cat_w
        canvas_h  = axis_top + cat_h

        cat_img = Image.new("RGBA", (canvas_w, canvas_h), (0,0,0,0))
        print(f"[info] Processing {png_path.name}: {rows}x{cols} tiles")

        # Axes
        if SHOW_AXES:
            draw_axes(cat_img, rows, cols, TILE_SIZE, TILE_SPACING, axis_left, axis_top, AXIS_TICK_STEP)

        # Optional gridlines over tile area
        if DRAW_GRIDLINES:
            draw_grid(cat_img, axis_left, axis_top, cat_w, cat_h, tile_pitch, alpha=GRID_ALPHA)

        # Extract tiles and draw on catalogue
        for r in range(rows):
            for c in range(cols):
                y0, y1 = r*TILE_SIZE, (r+1)*TILE_SIZE
                x0, x1 = c*TILE_SIZE, (c+1)*TILE_SIZE
                crop = src[y0:y1, x0:x1, :]
                if crop.size == 0:
                    continue
                if is_black_tile(crop):
                    continue

                rgba = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGBA) if crop.shape[2] == 4 else cv2.cvtColor(crop, cv2.COLOR_BGR2RGBA)
                hsh = tile_hash_rgba(rgba)

                if hsh in hash_to_index:
                    idx = hash_to_index[hsh]
                    cls = index_meta[idx]["classification"]
                    fname = index_meta[idx]["filename"]
                    is_dup = True
                else:
                    idx = next_index
                    cls = classify_tile(crop)
                    tile_img = Image.fromarray(rgba)
                    fname = f"tile_{idx:04d}.png"
                    tile_img.save(OUT_DIR_TILES / fname)
                    hash_to_index[hsh] = idx
                    index_meta[idx] = {"filename": fname, "classification": cls, "hash": hsh}
                    next_index += 1
                    is_dup = False

                ox = axis_left + c * tile_pitch
                oy = axis_top  + r * tile_pitch

                tile_img = Image.fromarray(rgba)
                cat_img.alpha_composite(tile_img, (ox, oy))

                # subtle class overlay
                overlay_color = (255,0,0,40) if cls=="solid" else (0,255,0,40)
                overlay = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), overlay_color)
                cat_img.alpha_composite(overlay, (ox, oy))

                if DRAW_INDEX:
                    draw_index_label(cat_img, str(idx), (ox+1, oy+1))

                legend["tiles"].append({
                    "source": png_path.name,
                    "row": r, "col": c,
                    "index": idx,
                    "classification": cls,
                    "is_duplicate": is_dup,
                    "filename": fname
                })

        # Save per-source catalogue @ high DPI
        out_cat = OUT_CATALOGUE / f"Catalogue_Layout_{png_path.stem}.png"
        cat_img.save(out_cat, dpi=SAVE_DPI)
        master_sections.append((cat_img, cat_img.size[1], cat_img.size[0], png_path.stem))
        print(f"[ok] Catalogue with axes (450 DPI) → {out_cat.name}")

    # Save unified legend (JSON)
    for idx in range(next_index):
        legend["unique_tiles"].append({
            "index": idx,
            "filename": index_meta[idx]["filename"],
            "classification": index_meta[idx]["classification"],
            "hash": index_meta[idx]["hash"]
        })
    with open(LEGEND_PATH_JSON, "w", encoding="utf-8") as f:
        json.dump(legend, f, indent=2, ensure_ascii=False)

    # Save CSV (manual overrides blank)
    with open(LEGEND_PATH_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index","filename","source","row","col","classification","manual_override"])
        for t in legend["tiles"]:
            writer.writerow([t["index"],t["filename"],t["source"],t["row"],t["col"],t["classification"],""])

    # Optional: master stacked catalogue (includes axes)
    if MAKE_MASTER_CATALOGUE and master_sections:
        total_h = sum(h for _,h,_,_ in master_sections) + (len(master_sections)-1)*MASTER_GAP_BETWEEN_MAPS
        max_w = max(w for _,_,w,_ in master_sections)
        master = Image.new("RGBA",(max_w,total_h),(0,0,0,0))
        y = 0
        for img,h,w,name in master_sections:
            master.alpha_composite(img,(0,y))
            # Add title bar over each section (semi-opaque)
            bar = Image.new("RGBA",(w,16),(0,0,0,150))
            d = ImageDraw.Draw(bar)
            font = load_font(10)
            d.text((4,2),name,fill=(255,255,255,200),font=font)
            master.alpha_composite(bar,(0,y))
            y += h + MASTER_GAP_BETWEEN_MAPS
        master_path = OUT_CATALOGUE / "catalogue_layout_all.png"
        master.save(master_path, dpi=SAVE_DPI)
        print(f"[ok] Master stacked catalogue (450 DPI) → {master_path.name}")

    print(f"[ok] {next_index} unique tiles saved → {OUT_DIR_TILES}")
    print(f"[ok] JSON legend → {LEGEND_PATH_JSON.name}")
    print(f"[ok] CSV legend  → {LEGEND_PATH_CSV.name}")

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
