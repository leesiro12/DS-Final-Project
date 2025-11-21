#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, sys, re
from typing import Optional, Tuple, Dict
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# PARAMS
PARAMS: Dict[str, object] = {

    
    "CSV_PATH": "../legend_all_tiles.csv",
    "SOURCE_DIR": "../Data",
    "SOURCE_EXT": ".png",
    "OUT_DIR": "../Validation",

    # This column is used to group CSV rows before overlaying.
    "MAP_ID_COLUMN": "source",

    # The name of the column that indicates whether a tile is walkable.
    # If set to None, the program will automatically search for
    # common keywords such as “manual_override”, “class”, or “label”.
    # When defined, it must exist in the CSV.
    "MANUAL_OVERRIDE_COLUMN": None,

    # Manual column mapping override.
    # Use this dictionary when your CSV has unconventional column names.
    # For example, if your file uses “grid_row” instead of “row”,
    # uncomment and assign as shown below.
    "FORCED_COL_NAMES": {
        # "row": "grid_row",    # Force use of this column for row index
        # "col": "grid_col",    # Force use of this column for column index
        # "x": "x",             # Force X coordinate (pixels)
        # "y": "y",             # Force Y coordinate (pixels)
        # "tile_w": "tile_w",   # Force tile width column
        # "tile_h": "tile_h",   # Force tile height column
    },


    "DEFAULT_TILE_W": 16,
    "DEFAULT_TILE_H": 16,
    "GRID_OFFSET_X": 0,
    "GRID_OFFSET_Y": 0,

    # Radius of the dot (in pixels) drawn at the center of each
    # walkable tile. The dot visually marks passable areas.
    "DOT_RADIUS": 4,

    # Fill color for the walkable dot (in RGB).
    # Default: yellow = (255, 255, 0)
    "DOT_FILL_RGB": (255, 255, 0),
    
    "DOT_FILL_ALPHA": 230,
    "DOT_STROKE_RGB": (0, 0, 0),
    "DOT_STROKE_ALPHA": 255,

    # Stroke width (in pixels) for the dot outline.
    "DOT_STROKE_WIDTH": 1,

    # Add a soft glow around walkable dots.
    # This creates a halo effect to make the dot more visible.
    "ADD_GLOW": True,
    "GLOW_COLOR_RGB": (255, 255, 0),
    "GLOW_ALPHA": 130,
    "GLOW_RADIUS": 4,
    "GLOW_EXPAND": 4,

    # If True, non-walkable tiles are darkened by overlaying a
    # semi-transparent layer over them. Walkable tiles remain bright.
    "DIM_NON_WALKABLE": True,
    "DIM_ALPHA": 140,

    # Draw the tile grid overlay.
    # The grid helps visualize tile boundaries.
    "DRAW_GRID": True,
    "GRID_COLOR_RGB": (255, 255, 255),
    "GRID_ALPHA": 90,
    "GRID_LINE_WIDTH": 1,

    # Frequency of grid lines. Example:
    # 1 → draw every tile
    # 2 → draw every other tile
    "GRID_EVERY_TILE": 1,

    # Display tile indices (row/col numbers) along
    # the top and left axes for reference.
    "DRAW_AXES": True,
    "AXIS_INDEX_START": 0,
    "AXIS_LABEL_EVERY": 4,
    "AXIS_FONT_SIZE": 12,
    "AXIS_TEXT_RGB": (255, 255, 255),
    "AXIS_TEXT_ALPHA": 220,
    "AXIS_TEXT_STROKE_RGB": (0, 0, 0),
    "AXIS_TEXT_STROKE_ALPHA": 255,
    "AXIS_TEXT_STROKE_WIDTH": 2,

    # Optional fine-tuning offset (in pixels) for text placement.
    # Adjust when text overlaps grid lines or needs alignment tweaks.
    "AXIS_OFFSET_PX": 0,

    # If True, prints progress information, detected columns,
    # and per-map status messages to the console.
    # Set to False for silent batch runs.
    "VERBOSE": True,
}

# Column autodetection (schema-tolerant)
def _cols_map(df: pd.DataFrame) -> dict:
    return {c.lower(): c for c in df.columns}

def _find(cols_map: dict, *cands) -> Optional[str]:
    for c in cands:
        if c and c.lower() in cols_map:
            return cols_map[c.lower()]
    return None

def autodetect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Map your CSV headers to normalized names used by the pipeline."""
    cm = _cols_map(df)
    fc = PARAMS["FORCED_COL_NAMES"] or {}

    row = fc.get("row") or _find(cm, "row", "tile_row", "grid_row", "r")
    col = fc.get("col") or _find(cm, "col", "column", "tile_col", "grid_col", "c")
    x   = fc.get("x")   or _find(cm, "x", "px", "pos_x", "left")
    y   = fc.get("y")   or _find(cm, "y", "py", "pos_y", "top")
    tile_w = fc.get("tile_w") or _find(cm, "tile_w", "tile_width", "w")
    tile_h = fc.get("tile_h") or _find(cm, "tile_h", "tile_height", "h")

    manual = PARAMS["MANUAL_OVERRIDE_COLUMN"]
    if manual:
        manual = _find(cm, manual)
    else:
        manual = _find(cm, "manual_override", "override", "class_override",
                       "manual", "label", "class")

    map_id = PARAMS["MAP_ID_COLUMN"]
    map_id = _find(cm, map_id) if map_id else _find(cm, "source", "map_id",
                                                    "screen_id", "room_id", "level_id")

    if PARAMS["VERBOSE"]:
        print("Detected columns:", {
            "row": row, "col": col, "x": x, "y": y,
            "tile_w": tile_w, "tile_h": tile_h,
            "manual": manual, "map_id": map_id
        })
    return {"row": row, "col": col, "x": x, "y": y,
            "tile_w": tile_w, "tile_h": tile_h,
            "manual": manual, "map_id": map_id}

def infer_tile_size(df: pd.DataFrame, cols) -> Tuple[int, int]:
    """Pick tile size from CSV if present; fall back to defaults."""
    tw = th = None
    if cols["tile_w"] and pd.notna(df[cols["tile_w"]]).any():
        try: tw = int(float(df[cols["tile_w"]].dropna().iloc[0]))
        except Exception: pass
    if cols["tile_h"] and pd.notna(df[cols["tile_h"]]).any():
        try: th = int(float(df[cols["tile_h"]].dropna().iloc[0]))
        except Exception: pass
    return (tw or int(PARAMS["DEFAULT_TILE_W"]),
            th or int(PARAMS["DEFAULT_TILE_H"]))

# Helpers (paths, colors, sanitization, flags)
def is_walkable(v: object) -> bool:
    """Classify walkable tiles using normalized truthy labels."""
    if pd.isna(v): return False
    s = str(v).strip().lower()
    return s in {"air", "walkable", "passable", "0", "true", "t", "yes", "y"}

def build_source_path(source_val: str) -> Optional[str]:
    """Resolve base image path from map_id (string/filename) + SOURCE_* params."""
    if source_val is None or str(source_val).strip() == "":
        return None
    s = str(source_val).strip()
    # If it's already a path or a file, use as-is
    if os.path.isabs(s) or os.path.sep in s or os.path.isfile(s):
        return s
    base = s
    ext = str(PARAMS["SOURCE_EXT"] or "")
    if ext and not base.lower().endswith(ext.lower()):
        base += ext
    return os.path.join(str(PARAMS["SOURCE_DIR"]), base)

def _rgba(rgb, a): return (int(rgb[0]), int(rgb[1]), int(rgb[2]), int(a))

def _sanitize(s: str) -> str:
    """Safe filename fragment from arbitrary map_id."""
    s = re.sub(r"[^\w\-]+", "_", str(s))
    return s.strip("_") or "map"

# Layers (composable rendering units)
def dot_layers_for_map(sdf: pd.DataFrame, cols, base_size: Tuple[int, int]) -> Tuple[Image.Image, Image.Image]:
    """
    Walkable center dots (+ optional glow).
    Returns (glow_layer, dots_layer) as RGBA images sized to base_size.
    """
    W, H = base_size
    glow_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dots_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw_dots = ImageDraw.Draw(dots_layer)

    tw, th = infer_tile_size(sdf, cols)
    ox, oy = int(PARAMS["GRID_OFFSET_X"]), int(PARAMS["GRID_OFFSET_Y"])

    dot_r = int(PARAMS["DOT_RADIUS"])
    fill = _rgba(PARAMS["DOT_FILL_RGB"], PARAMS["DOT_FILL_ALPHA"])
    stroke = _rgba(PARAMS["DOT_STROKE_RGB"], PARAMS["DOT_STROKE_ALPHA"])
    sw = max(0, int(PARAMS["DOT_STROKE_WIDTH"]))

    if PARAMS["ADD_GLOW"]:
        glow_color = _rgba(PARAMS["GLOW_COLOR_RGB"], PARAMS["GLOW_ALPHA"])
        g_rad = int(PARAMS["GLOW_RADIUS"])
        expand = int(PARAMS["GLOW_EXPAND"])

    for _, rr in sdf.iterrows():
        # Determine tile top-left using (col,row) or (x,y)
        if cols["col"] is not None and pd.notna(rr[cols["col"]]):
            x = int(rr[cols["col"]]) * tw + ox
        elif cols["x"] is not None and pd.notna(rr[cols["x"]]):
            x = int(rr[cols["x"]]) + ox
        else:
            continue

        if cols["row"] is not None and pd.notna(rr[cols["row"]]):
            y = int(rr[cols["row"]]) * th + oy
        elif cols["y"] is not None and pd.notna(rr[cols["y"]]):
            y = int(rr[cols["y"]]) + oy
        else:
            continue

        if cols["manual"] and is_walkable(rr[cols["manual"]]):
            cx = x + tw // 2
            cy = y + th // 2

            if PARAMS["ADD_GLOW"]:
                gw = (dot_r + expand) * 2
                gh = (dot_r + expand) * 2
                gx = cx - gw // 2
                gy = cy - gh // 2
                gpatch = Image.new("RGBA", (gw, gh), (0, 0, 0, 0))
                gd = ImageDraw.Draw(gpatch)
                gd.ellipse((0, 0, gw-1, gh-1), fill=glow_color)
                gpatch = gpatch.filter(ImageFilter.GaussianBlur(g_rad))
                glow_layer.alpha_composite(gpatch, dest=(gx, gy))

            x0, y0, x1, y1 = cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r
            if sw > 0:
                draw_dots.ellipse((x0, y0, x1, y1), fill=fill, outline=stroke, width=sw)
            else:
                draw_dots.ellipse((x0, y0, x1, y1), fill=fill)

    return glow_layer, dots_layer

def dim_non_walkable_layer(sdf: pd.DataFrame, cols, base_size: Tuple[int, int]) -> Image.Image:
    """
    Dim everything except walkable tiles using a soft mask.
    Returns a transparent RGBA layer if DIM_NON_WALKABLE is False.
    """
    if not PARAMS["DIM_NON_WALKABLE"]:
        return Image.new("RGBA", base_size, (0, 0, 0, 0))

    W, H = base_size
    dim = Image.new("RGBA", (W, H), (0, 0, 0, int(PARAMS["DIM_ALPHA"])))
    erase = Image.new("L", (W, H), 0)  # will punch holes where tiles are walkable
    draw = ImageDraw.Draw(erase)

    tw, th = infer_tile_size(sdf, cols)
    ox, oy = int(PARAMS["GRID_OFFSET_X"]), int(PARAMS["GRID_OFFSET_Y"])
    dot_r = int(PARAMS["DOT_RADIUS"])

    for _, rr in sdf.iterrows():
        # Compute tile top-left
        if cols["col"] is not None and pd.notna(rr[cols["col"]]):
            x = int(rr[cols["col"]]) * tw + ox
        elif cols["x"] is not None and pd.notna(rr[cols["x"]]):
            x = int(rr[cols["x"]]) + ox
        else:
            continue

        if cols["row"] is not None and pd.notna(rr[cols["row"]]):
            y = int(rr[cols["row"]]) * th + oy
        elif cols["y"] is not None and pd.notna(rr[cols["y"]]):
            y = int(rr[cols["y"]]) + oy
        else:
            continue

        if cols["manual"] and is_walkable(rr[cols["manual"]]):
            cx = x + tw // 2
            cy = y + th // 2
            draw.ellipse((cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r), fill=255)

    # apply mask: keep dim everywhere except holes around walkable centers
    transparent = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dim.paste(transparent, (0, 0), mask=erase)
    return dim

def grid_layer(base_size: Tuple[int, int], tw: int, th: int, ox: int, oy: int) -> Image.Image:
    """Tile lattice lines aligned to the (tw, th) grid with optional cadence."""
    layer = Image.new("RGBA", base_size, (0, 0, 0, 0))
    if not PARAMS["DRAW_GRID"]:
        return layer

    W, H = base_size
    draw = ImageDraw.Draw(layer)
    col = _rgba(PARAMS["GRID_COLOR_RGB"], PARAMS["GRID_ALPHA"])
    lw = max(1, int(PARAMS["GRID_LINE_WIDTH"]))
    step = max(1, int(PARAMS["GRID_EVERY_TILE"]))

    # verticals
    x = ox; i = 0
    while x <= W:
        if i % step == 0:
            draw.line([(x, 0), (x, H)], fill=col, width=lw)
        x += tw; i += 1

    # horizontals
    y = oy; j = 0
    while y <= H:
        if j % step == 0:
            draw.line([(0, y), (W, y)], fill=col, width=lw)
        y += th; j += 1

    return layer

def axes_layer(base_size: Tuple[int, int], tw: int, th: int, ox: int, oy: int) -> Image.Image:
    """Row/column index labels at a cadence (top for columns, left for rows)."""
    layer = Image.new("RGBA", base_size, (0, 0, 0, 0))
    if not PARAMS["DRAW_AXES"]:
        return layer

    W, H = base_size
    draw = ImageDraw.Draw(layer)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    start = int(PARAMS["AXIS_INDEX_START"])
    every = max(1, int(PARAMS["AXIS_LABEL_EVERY"]))
    off = int(PARAMS["AXIS_OFFSET_PX"])

    txt_col = _rgba(PARAMS["AXIS_TEXT_RGB"], PARAMS["AXIS_TEXT_ALPHA"])
    stroke_col = _rgba(PARAMS["AXIS_TEXT_STROKE_RGB"], PARAMS["AXIS_TEXT_STROKE_ALPHA"])
    stroke_w = int(PARAMS["AXIS_TEXT_STROKE_WIDTH"])

    n_cols = max(0, (W - ox) // tw)
    n_rows = max(0, (H - oy) // th)

    # Column labels (top)
    for c in range(n_cols):
        if c % every != 0: continue
        cx = ox + c * tw + tw // 2
        cy = max(0, oy - 2) + off
        label = str(c + start)
        draw.text((cx, cy), label, anchor="ma", fill=txt_col,
                  stroke_width=stroke_w, stroke_fill=stroke_col, font=font)

    # Row labels (left)
    for r in range(n_rows):
        if r % every != 0: continue
        cx = max(0, ox - 2) + off
        cy = oy + r * th + th // 2
        label = str(r + start)
        draw.text((cx, cy), label, anchor="rm", fill=txt_col,
                  stroke_width=stroke_w, stroke_fill=stroke_col, font=font)

    return layer

# MAIN 
def main():
    # Load CSV
    csv_path = str(PARAMS["CSV_PATH"])
    if not os.path.isfile(csv_path):
        print(f"CSV not found: {csv_path}", file=sys.stderr); sys.exit(1)

    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV has no rows.", file=sys.stderr); sys.exit(1)

    cols = autodetect_columns(df)
    if not cols["map_id"]:
        print("Could not find the 'source' column. Set MAP_ID_COLUMN correctly.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(str(PARAMS["OUT_DIR"]), exist_ok=True)

    # Per-map processing
    for sid, sdf in df.groupby(cols["map_id"]):
        src_path = build_source_path(sid)
        if not src_path or not os.path.isfile(src_path):
            print(f"Warning: source image not found for '{sid}': {src_path}")
            continue

        if PARAMS["VERBOSE"]:
            print(f"\n=== {sid} (rows={len(sdf)}) ===")
            print(f"Base image: {src_path}")

        base = Image.open(src_path).convert("RGBA")
        W, H = base.size

        # Geometry for grid/axes even if CSV rows are sparse
        tw, th = infer_tile_size(sdf, cols)

        # Build layers
        layer_glow, layer_dots = dot_layers_for_map(sdf, cols, (W, H))

        # Compose
        comp = base.copy()
        comp.alpha_composite(layer_glow)
        comp.alpha_composite(layer_dots)

        # Persist artifacts
        tag = _sanitize(sid)
        out_dir = str(PARAMS["OUT_DIR"])
        base.copy().save(os.path.join(out_dir, f"{tag}_base.png"))
        comp.save(os.path.join(out_dir, f"{tag}_overlay_walkable.png"))

        print(f"Saved → {os.path.join(out_dir, f'{tag}_overlay_walkable.png')}")

if __name__ == "__main__":
    main()
