#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reconstruct_overlay_from_legends_pixels_normalized_sources.py

Uses ALL available positional variables to make overlays pixel-perfect:
- Per-source canvas from SOURCES_DIR (e.g., sources/tloz1_1.png).
- BASE origin fused from (priority):
    1) explicit origin_x/origin_y in BASE,
    2) BASE pixel anchors (x,y) minus (col,row)*tile,
    3) OTHER tile spans vs pixels,
    4) OTHER min(x,y) snapped to grid,
    5) fallback (0,0).
- OTHER positions reconciled from (x,y,w,h) and (tile_r0,c0,r1,c1) if present.

Result: end-to-end identical size to original, with maximum alignment reliability.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, Tuple, Iterable, Dict, List
import numpy as np
import pandas as pd
from PIL import Image

# =======================
# ======= PARAMS ========
# =======================
LEGEND_BASE_CSV   = "legend_base.csv"
LEGEND_OTHER_CSV  = "legend_other.csv"
TILES_BASE_DIR    = "tiles_base"
TILES_OTHER_DIR   = "tiles_other"
SOURCES_DIR       = "sources"
OUT_DIR           = "reconstructed"

# Optional single reference for the OVERALL canvas; per-source canvases come from SOURCES_DIR.
REFERENCE_IMAGE_PATH: Optional[str] = None  # e.g. "sources/tloz1_1.png"

# If None, infer from a referenced base tile (or optional columns in base legend)
FORCE_TILE_SIZE: Optional[Tuple[int, int]] = None  # e.g. (16, 16)

# OTHER rendering
OTHER_SCALE_FACTOR = 1
SCALE_OTHER_POSITIONS = False
OTHER_BLACK_IS_TRANSPARENT = True

# BASE rendering
ON_MISSING_BASE_TILE = "skip"  # or "error"

# Optional source-name overrides after normalization
CUSTOM_SOURCE_ALIASES: Dict[str, str] = {
    # "tloz1_1e": "tloz1_1",
}

SAVE_DPI = (300, 300)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
# =======================


# ---------- source normalization ----------

def normalize_source_name(s: str) -> str:
    """
    Normalize variants like 'tloz1_1e.png' -> 'tloz1_1' (so 'e' overlays map to base).
    """
    if s is None:
        return "overall"
    s = str(s).strip()
    name = Path(s).name
    stem = Path(name).stem.lower()

    # unify hyphens
    stem = stem.replace("-", "_")

    # map *_e or trailing digit+e to base (tloz1_1e -> tloz1_1)
    if stem.endswith("_e"):
        stem = stem[:-2]
    elif stem.endswith("e") and len(stem) >= 2 and stem[-2].isdigit():
        stem = stem[:-1]

    if stem in CUSTOM_SOURCE_ALIASES:
        stem = CUSTOM_SOURCE_ALIASES[stem]

    stem = re.sub(r"[^\w\-.]+", "_", stem)
    return stem or "overall"


def _safe_dirname(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s or "unnamed"


# ---------- CSV normalizers ----------

def _norm_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts required: index + (row,col) OR (y,x) OR (r,c).
    Optional columns if present (used for stronger alignment):
      - x,y (pixel position of tile's top-left in the original canvas)
      - w,h (pixel size of tile; if uniform across rows we can infer tile size)
      - origin_x, origin_y (explicit grid origin in pixels)
      - tile_w, tile_h (explicit tile size)
      - source
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # index
    if "index" not in df.columns:
        cand = [c for c in df.columns if "index" in c]
        if len(cand) == 1:
            df = df.rename(columns={cand[0]: "index"})
        else:
            raise ValueError("legend_base.csv must have an 'index' column.")

    # coords
    for a, b in (("row", "col"), ("y", "x"), ("r", "c")):
        if a in df.columns and b in df.columns:
            df = df.rename(columns={a: "row", b: "col"})
            break
    else:
        raise ValueError("legend_base.csv must have tile coords (row,col) or (y,x) or (r,c).")

    # source
    if "source" not in df.columns:
        df["source"] = "overall"
    df["source"] = df["source"].map(normalize_source_name)

    # cast
    df["row"] = df["row"].astype(int)
    df["col"] = df["col"].astype(int)
    df["index"] = df["index"].astype(int)

    # keep optional pixel anchors / origin / tile size if present
    for opt in ("x", "y", "w", "h", "origin_x", "origin_y", "tile_w", "tile_h"):
        if opt not in df.columns:
            df[opt] = pd.NA

    return df[["row", "col", "index", "source", "x", "y", "w", "h", "origin_x", "origin_y", "tile_w", "tile_h"]]


def _norm_other(df: pd.DataFrame) -> pd.DataFrame:
    """
    Required: filename, x, y, w, h (pixels), source.
    Optional (if present): tile_r0, tile_c0, tile_r1, tile_c1.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    if "filename" not in df.columns:
        raise ValueError("legend_other.csv must include 'filename'.")

    if "index" not in df.columns:
        cand = [c for c in df.columns if "index" in c]
        df = df.rename(columns={cand[0]: "index"}) if cand else df.assign(index=-1)

    required = ["x", "y", "w", "h"]
    if not all(r in df.columns for r in required):
        raise ValueError("legend_other.csv must include x,y,w,h.")

    if "source" not in df.columns:
        df["source"] = "overall"
    df["source"] = df["source"].map(normalize_source_name)

    for c in ("x", "y", "w", "h"):
        df[c] = df[c].astype(int)

    # Optional tile spans
    for opt in ("tile_r0", "tile_c0", "tile_r1", "tile_c1"):
        if opt in df.columns:
            df[opt] = df[opt].astype("Int64")
        else:
            df[opt] = pd.Series([pd.NA] * len(df), dtype="Int64")

    return df[["index", "filename", "source", "x", "y", "w", "h", "tile_r0", "tile_c0", "tile_r1", "tile_c1"]]


# ---------- helper: tiles & canvas sizes ----------

def _load_base_tile(tiles_dir: Path, idx: int) -> Optional[Image.Image]:
    for p in (tiles_dir / f"tile_{idx}.png", tiles_dir / f"tile_{idx:04d}.png"):
        if p.exists():
            return Image.open(p).convert("RGBA")
    return None


def _infer_tile_size_from_tiles(tiles_dir: Path, indices: Iterable[int]) -> Tuple[int, int]:
    for i in indices:
        im = _load_base_tile(tiles_dir, i)
        if im is not None:
            return im.size
    raise RuntimeError(f"Cannot infer tile size from {tiles_dir}; no referenced tiles found.")


def _rows_cols_extent(df: pd.DataFrame) -> Tuple[int, int]:
    return int(df["row"].max()) + 1, int(df["col"].max()) + 1


def _apply_black_to_transparency(png: Image.Image) -> Image.Image:
    arr = np.array(png.convert("RGBA"), dtype=np.uint8)
    rgb = arr[..., :3]
    a = arr[..., 3]
    mask = (rgb[..., 0] == 0) & (rgb[..., 1] == 0) & (rgb[..., 2] == 0)
    a[mask] = 0
    return Image.fromarray(np.dstack([rgb, a]), "RGBA")


def _all_key_variants(stem: str) -> List[str]:
    """
    Build a set of matching keys for a given stem (case-insensitive).
    Supports *_e mapping to base stem automatically.
    """
    keys = set()
    base = stem.strip().lower().replace("-", "_")
    keys.add(base)
    # normalized variant (drop _e or trailing e after digit)
    norm = base
    if norm.endswith("_e"):
        norm = norm[:-2]
    elif norm.endswith("e") and len(norm) >= 2 and norm[-2].isdigit():
        norm = norm[:-1]
    keys.add(norm)
    keys.add(base.replace("_", ""))
    keys.add(norm.replace("_", ""))
    return list(keys)


def build_source_canvas_index(sources_dir: Path) -> Dict[str, Tuple[int, int, Path]]:
    """
    Scan SOURCES_DIR and map multiple key variants to (W,H,path).
    """
    idx: Dict[str, Tuple[int, int, Path]] = {}
    if not sources_dir.exists():
        return idx

    for p in sources_dir.iterdir():
        if p.suffix.lower() not in IMG_EXTS or not p.is_file():
            continue
        try:
            im = Image.open(p)
            W, H = im.size
        except Exception:
            continue
        stem = p.stem
        for k in _all_key_variants(stem):
            if k not in idx:  # first one wins
                idx[k] = (W, H, p)
    return idx


def lookup_canvas_for_source(src_norm: str, index: Dict[str, Tuple[int, int, Path]]) -> Optional[Tuple[int, int, Path]]:
    """
    Try exact, then loose variants (without underscores), then prefix/suffix matches.
    """
    s1 = src_norm
    s2 = src_norm.replace("_", "")
    if s1 in index:
        return index[s1]
    if s2 in index:
        return index[s2]
    # loose scan
    for k, v in index.items():
        if k == s1 or k == s2 or k.startswith(s1) or s1.startswith(k):
            return v
    return None


# ---------- positional fusion ----------

def _series_has_all(df: pd.DataFrame, cols: List[str]) -> bool:
    return all(c in df.columns and df[c].notna().any() for c in cols)

def _infer_tile_size(df_base_src: pd.DataFrame, tiles_dir: Path, idxs: Iterable[int]) -> Tuple[int, int]:
    """
    Try tile size from BASE legend columns (tile_w,tile_h) or w,h if consistent;
    else infer from tile image files.
    """
    # explicit tile size columns
    if _series_has_all(df_base_src, ["tile_w", "tile_h"]):
        tws = pd.to_numeric(df_base_src["tile_w"], errors="coerce").dropna()
        ths = pd.to_numeric(df_base_src["tile_h"], errors="coerce").dropna()
        if not tws.empty and not ths.empty:
            tw, th = int(round(tws.median())), int(round(ths.median()))
            if tw > 0 and th > 0:
                return tw, th

    # consistent per-tile w,h (pixel size per tile placement)
    if _series_has_all(df_base_src, ["w", "h"]):
        ws = pd.to_numeric(df_base_src["w"], errors="coerce").dropna()
        hs = pd.to_numeric(df_base_src["h"], errors="coerce").dropna()
        if not ws.empty and not hs.empty:
            tw, th = int(round(ws.median())), int(round(hs.median()))
            if tw > 0 and th > 0:
                return tw, th

    # fallback to actual tile images
    return _infer_tile_size_from_tiles(tiles_dir, idxs)

def _fuse_origin_from_all_hints(
    df_base_src: pd.DataFrame,
    df_other_src: pd.DataFrame,
    tile_w: int,
    tile_h: int
) -> Tuple[int, int, str]:
    """
    Compute (ox,oy) using all hints; return also a human-readable reason.
    Priority:
      1) origin_x/origin_y in BASE
      2) BASE pixel anchors: median(x - col*tile_w), median(y - row*tile_h)
      3) OTHER tile spans vs pixels: median(x - tile_c0*tile_w), ...
      4) OTHER min(x,y) snapped to grid
      5) fallback (0,0)
    """
    # 1) explicit origin
    if _series_has_all(df_base_src, ["origin_x", "origin_y"]):
        oxs = pd.to_numeric(df_base_src["origin_x"], errors="coerce").dropna()
        oys = pd.to_numeric(df_base_src["origin_y"], errors="coerce").dropna()
        if not oxs.empty and not oys.empty:
            ox, oy = int(round(oxs.median())), int(round(oys.median()))
            return ox, oy, "origin from BASE.origin_x/y"

    # 2) BASE pixel anchors
    if _series_has_all(df_base_src, ["x", "y"]):
        # Use rows where x,y are valid
        bb = df_base_src.dropna(subset=["x", "y"])[["x", "y", "row", "col"]].copy()
        bb["x"] = pd.to_numeric(bb["x"], errors="coerce")
        bb["y"] = pd.to_numeric(bb["y"], errors="coerce")
        bb = bb.dropna()
        if not bb.empty:
            offs_x = (bb["x"] - bb["col"] * tile_w).values
            offs_y = (bb["y"] - bb["row"] * tile_h).values
            ox = int(round(np.median(offs_x)))
            oy = int(round(np.median(offs_y)))
            return ox, oy, "origin from BASE pixel anchors (x,y - grid)"

    # 3) OTHER tile spans vs pixels
    if _series_has_all(df_other_src, ["tile_r0", "tile_c0"]):
        oo = df_other_src.dropna(subset=["x", "y", "tile_r0", "tile_c0"])[["x", "y", "tile_r0", "tile_c0"]].copy()
        oo["tile_r0"] = pd.to_numeric(oo["tile_r0"], errors="coerce")
        oo["tile_c0"] = pd.to_numeric(oo["tile_c0"], errors="coerce")
        oo["x"] = pd.to_numeric(oo["x"], errors="coerce")
        oo["y"] = pd.to_numeric(oo["y"], errors="coerce")
        oo = oo.dropna()
        if not oo.empty:
            offs_x = (oo["x"] - oo["tile_c0"] * tile_w).values
            offs_y = (oo["y"] - oo["tile_r0"] * tile_h).values
            ox = int(round(np.median(offs_x)))
            oy = int(round(np.median(offs_y)))
            return ox, oy, "origin from OTHER tile spans vs pixels"

    # 4) min(x,y) snapped
    if not df_other_src.empty:
        min_x = int(df_other_src["x"].min())
        min_y = int(df_other_src["y"].min())
        ox = (min_x // tile_w) * tile_w
        oy = (min_y // tile_h) * tile_h
        return ox, oy, "origin from OTHER min(x,y) snapped to grid"

    # 5) fallback
    return 0, 0, "origin fallback (0,0)"

def _reconcile_other_xywh(
    row: pd.Series,
    tile_w: int,
    tile_h: int,
    ox: int,
    oy: int
) -> Tuple[int, int, int, int, Optional[str]]:
    """
    Use all available info to decide (x,y,w,h) for a sprite:
      - primary: x,y,w,h (pixels)
      - if tile spans exist, compute span-derived (xs,ys) = (c0*tw+ox, r0*th+oy) and compare.
      - if disagreement, warn and prefer pixels but clamp within span extents when reasonable.
    Returns (x,y,w,h, note)
    """
    x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
    note = None

    if pd.notna(row["tile_r0"]) and pd.notna(row["tile_c0"]):
        r0 = int(row["tile_r0"]); c0 = int(row["tile_c0"])
        xs = c0 * tile_w + ox
        ys = r0 * tile_h + oy

        dx = abs(x - xs)
        dy = abs(y - ys)
        # If huge mismatch, flag it
        if dx > tile_w // 2 or dy > tile_h // 2:
            note = f"sprite pos mismatch: pixel({x},{y}) vs span({xs},{ys}); using pixel with sanity clamp"
            # sanity clamp start within the span grid cell (optional mild clamp)
            x = max(min(x, xs + tile_w), xs - tile_w)
            y = max(min(y, ys + tile_h), ys - tile_h)

    # keep w/h as-is (pixel source of truth)
    return x, y, w, h, note


# ---------- reconstructors ----------

def reconstruct_base(
    df_base: pd.DataFrame,
    tiles_dir: Path,
    tile_w: int,
    tile_h: int,
    canvas_size: Tuple[int, int],
    origin_xy: Tuple[int, int]
) -> Image.Image:
    W, H = canvas_size
    ox, oy = origin_xy
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for r, c, idx, _src, *_rest in df_base.itertuples(index=False):
        im = _load_base_tile(tiles_dir, idx)
        if im is None:
            if ON_MISSING_BASE_TILE == "error":
                raise FileNotFoundError(f"Missing base tile index={idx}")
            continue
        if im.size != (tile_w, tile_h):
            im = im.resize((tile_w, tile_h), Image.NEAREST)
        x = ox + c * tile_w
        y = oy + r * tile_h
        if x >= W or y >= H or x + tile_w <= 0 or y + tile_h <= 0:
            continue
        canvas.alpha_composite(im, dest=(x, y))
    return canvas


def reconstruct_other(
    df_other: pd.DataFrame,
    tiles_dir: Path,
    base_size: Tuple[int, int],
    tile_w: int,
    tile_h: int,
    origin_xy: Tuple[int, int]
) -> Image.Image:
    W, H = base_size
    ox, oy = origin_xy
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for row in df_other.itertuples(index=False):
        _idx, fname, _src, x, y, w, h, r0, c0, r1, c1 = row
        p = Path(tiles_dir) / fname
        if not p.exists():
            # try numeric fallback
            m = re.search(r"(\d+)", fname)
            if m:
                i = int(m.group(1))
                for pp in (Path(tiles_dir) / f"tile_{i}.png", Path(tiles_dir) / f"tile_{i:04d}.png"):
                    if pp.exists():
                        p = pp
                        break
        if not p.exists():
            continue

        # reconcile x,y with spans if any
        xx, yy, ww, hh, _note = _reconcile_other_xywh(
            pd.Series({"x": x, "y": y, "w": w, "h": h, "tile_r0": r0, "tile_c0": c0}),
            tile_w, tile_h, ox, oy
        )

        sprite = Image.open(p).convert("RGBA")
        if OTHER_BLACK_IS_TRANSPARENT:
            sprite = _apply_black_to_transparency(sprite)
        if OTHER_SCALE_FACTOR != 1:
            sprite = sprite.resize(
                (max(1, int(sprite.width * OTHER_SCALE_FACTOR)),
                 max(1, int(sprite.height * OTHER_SCALE_FACTOR))),
                Image.NEAREST
            )
            if SCALE_OTHER_POSITIONS:
                xx = int(xx * OTHER_SCALE_FACTOR)
                yy = int(yy * OTHER_SCALE_FACTOR)

        if xx >= W or yy >= H or xx + sprite.width <= 0 or yy + sprite.height <= 0:
            continue
        canvas.alpha_composite(sprite, dest=(int(xx), int(yy)))
    return canvas


def alpha_overlay(base: Image.Image, top: Image.Image) -> Image.Image:
    # Both inputs should already match size, but keep defensive.
    W = max(base.width, top.width)
    H = max(base.height, top.height)
    b = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    t = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    b.alpha_composite(base, dest=(0, 0))
    t.alpha_composite(top, dest=(0, 0))
    return Image.alpha_composite(b, t)


def save_triplet(folder: Path, base_img: Image.Image, other_img: Image.Image, overlay_img: Image.Image):
    folder.mkdir(parents=True, exist_ok=True)
    base_img.save(folder / "reconstruct_base.png", dpi=SAVE_DPI)
    other_img.save(folder / "reconstruct_other.png", dpi=SAVE_DPI)
    overlay_img.save(folder / "reconstruct_overlay.png", dpi=SAVE_DPI)


# ---------- pipeline ----------

def main():
    tiles_base = Path(TILES_BASE_DIR)
    tiles_other = Path(TILES_OTHER_DIR)
    sources_dir = Path(SOURCES_DIR)
    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    df_base_raw  = pd.read_csv(LEGEND_BASE_CSV)
    df_other_raw = pd.read_csv(LEGEND_OTHER_CSV)
    df_base  = _norm_base(df_base_raw)
    df_other = _norm_other(df_other_raw)

    # Build index of source canvases from SOURCES_DIR
    src_index = build_source_canvas_index(sources_dir)
    if not src_index:
        print(f"[WARN] No source images found in {sources_dir.resolve()} — per-source canvases will fall back.")

    # Tile size (prefer strongest signal per source later; overall fallback here)
    if FORCE_TILE_SIZE is None:
        tile_w_fallback, tile_h_fallback = _infer_tile_size_from_tiles(tiles_base, df_base["index"].unique())
    else:
        tile_w_fallback, tile_h_fallback = FORCE_TILE_SIZE

    # ---------- OVERALL ----------
    if REFERENCE_IMAGE_PATH and Path(REFERENCE_IMAGE_PATH).exists():
        ref = Image.open(REFERENCE_IMAGE_PATH); base_W, base_H = ref.size
    else:
        rows_all, cols_all = _rows_cols_extent(df_base)
        base_W, base_H = cols_all * tile_w_fallback, rows_all * tile_h_fallback

    # Derive overall tile size and origin using all hints pooled (weakly useful, per-source is stronger)
    tile_w_overall, tile_h_overall = tile_w_fallback, tile_h_fallback
    if _series_has_all(df_base, ["tile_w", "tile_h"]):
        tw = pd.to_numeric(df_base["tile_w"], errors="coerce").dropna()
        th = pd.to_numeric(df_base["tile_h"], errors="coerce").dropna()
        if not tw.empty and not th.empty:
            tile_w_overall, tile_h_overall = int(round(tw.median())), int(round(th.median()))

    ox_all, oy_all, reason_all = _fuse_origin_from_all_hints(df_base, df_other, tile_w_overall, tile_h_overall)

    base_overall  = reconstruct_base(df_base, tiles_base, tile_w_overall, tile_h_overall, (base_W, base_H), (ox_all, oy_all))
    other_overall = reconstruct_other(df_other, tiles_other, (base_W, base_H), tile_w_overall, tile_h_overall, (ox_all, oy_all))
    overlay_overall = alpha_overlay(base_overall, other_overall)
    save_triplet(out_root / "overall", base_overall, other_overall, overlay_overall)
    print(f"[OVERALL] canvas={base_W}x{base_H}  tiles={tile_w_overall}x{tile_h_overall}  origin=({ox_all},{oy_all}) via {reason_all}")

    # ---------- PER-SOURCE ----------
    sources = sorted(set(df_base["source"].unique()).union(set(df_other["source"].unique())))
    for src in sources:
        b = df_base[df_base["source"] == src]
        o = df_other[df_other["source"] == src]
        if b.empty and o.empty:
            continue

        # Per-source canvas from SOURCES_DIR
        lookup = lookup_canvas_for_source(src, src_index)
        if lookup is not None:
            src_W, src_H, src_path = lookup
        else:
            rows_s, cols_s = (_rows_cols_extent(b) if not b.empty else _rows_cols_extent(df_base))
            # tile size temporarily (we'll refine below)
            src_W, src_H = cols_s * tile_w_fallback, rows_s * tile_h_fallback
            src_path = None

        # Per-source tile size (best effort)
        try:
            tile_w_s, tile_h_s = _infer_tile_size(b, tiles_base, b["index"].unique())
        except Exception:
            tile_w_s, tile_h_s = tile_w_fallback, tile_h_fallback

        # Per-source origin using all hints
        ox, oy, reason = _fuse_origin_from_all_hints(b, o, tile_w_s, tile_h_s)

        base_img   = reconstruct_base(b, tiles_base, tile_w_s, tile_h_s, (src_W, src_H), (ox, oy))
        other_img  = reconstruct_other(o, tiles_other, (src_W, src_H), tile_w_s, tile_h_s, (ox, oy))
        overlay_img = alpha_overlay(base_img, other_img)

        # Save all images directly into reconstructed/
        base_path = out_root / f"reconstruct_base_{_safe_dirname(src)}.png"
        other_path = out_root / f"reconstruct_other_{_safe_dirname(src)}.png"
        overlay_path = out_root / f"reconstruct_overlay_{_safe_dirname(src)}.png"
        
        base_img.save(base_path, dpi=SAVE_DPI)
        other_img.save(other_path, dpi=SAVE_DPI)
        overlay_img.save(overlay_path, dpi=SAVE_DPI)
        
        msg = f"[SRC] {src}: canvas={src_W}x{src_H} tiles={tile_w_s}x{tile_h_s} origin=({ox},{oy}) via {reason}"
        if src_path:
            msg += f"  file={src_path.name}"
        print(msg)


    print(f"[DONE] Saved to {out_root.resolve()}")

if __name__ == "__main__":
    main()
