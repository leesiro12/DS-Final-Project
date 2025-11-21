# -*- coding: utf-8 -*-
"""
make_zoomable_grid_viewer_spyder_dropdown_axes_follow.py

Spyder-friendly: run directly. Generates an OpenSeadragon HTML viewer with:
- Insets (margins) so visuals aren’t glued to the window edge.
- Minor+major grid lines for visibility.
- Dropdown to switch between images in the same folder + local file loader.
- **Axes (labels + tick marks) that FOLLOW the viewport** (pinned to top/left screen edges).

Writes `grid_viewer.html` beside your selected image and opens it.
"""

import webbrowser
from pathlib import Path
from textwrap import dedent

# Optional file picker if IMAGE_PATH is blank
try:
    import tkinter as _tk
    from tkinter import filedialog as _fd
    _HAS_TK = True
except Exception:
    _HAS_TK = False

# PARAMS 
IMAGE_PATH    = "../Validation/tloz7_2_png_overlay_walkable.png"         # e.g., r"C:\...\tloz1_1.png"; leave "" to choose via dialog
TILE_W        = 16         # tile width  (px, image space)
TILE_H        = 16         # tile height (px, image space)
OFFSET_X      = 0          # grid origin offset X (px)
OFFSET_Y      = 0          # grid origin offset Y (px)
INDEX_START   = 0          # 0 for zero-based labels, 1 for one-based
LABEL_EVERY   = 4          # label every Nth row/col
GRID_EVERY    = 1          # draw minor lines every Nth tile
MAJOR_EVERY   = 4          # draw a thicker/brighter major line every N tiles
VIEWER_MARGIN = 16         # px margin around the viewer (pads the page)
AUTO_OPEN     = True       # open the HTML in your default browser

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Zoomable Grid Viewer</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root {{ --viewer-margin: {viewer_margin}px; }}
  html, body {{ height: 100%; margin: 0; background:#0f0f12; color:#e8e8ef; }}
  body {{
    padding: var(--viewer-margin);
    box-sizing: border-box;
    font: 12px system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  }}
  .toolbar {{ display:flex; gap:8px; align-items:center; margin-bottom:10px; flex-wrap:wrap; }}
  .toolbar select, .toolbar input[type="file"] {{
    background:#1d1f24; color:#e8e8ef; border:1px solid #2f3240; border-radius:6px; padding:6px 8px;
  }}
  .toolbar label {{ display:flex; align-items:center; gap:6px; }}
  .viewer-wrap {{
    position: relative; width: 100%; height: calc(100% - 56px);
    background:#15171c; border:1px solid #2b2f3a; border-radius:12px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.35), inset 0 0 0 1px rgba(255,255,255,0.03);
    overflow: hidden;
  }}
  #viewer {{ width:100%; height:100%; position:relative; }}
  #gridCanvas {{ position:absolute; left:0; top:0; pointer-events:none; }}
  .hud {{ position:absolute; left:12px; top:12px; z-index:10; background:rgba(0,0,0,.45); color:#fff;
         padding:6px 10px; border-radius:8px; }}
</style>
</head>
<body>
  <div class="toolbar">
    <label>Image:
      <select id="imageSelect"></select>
    </label>
    <input id="fileInput" type="file" accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff" />
    <label><input id="toggleGrid" type="checkbox" checked> Grid</label>
    <label><input id="toggleLabels" type="checkbox" checked> Axes</label>
  </div>

  <div class="viewer-wrap">
    <div id="viewer"></div>
    <canvas id="gridCanvas"></canvas>
    <div class="hud">Zoom & pan — grid and axes follow</div>
  </div>

<script src="https://cdn.jsdelivr.net/npm/openseadragon@4.1/build/openseadragon/openseadragon.min.js"></script>
<script>
/* ================== CONFIG (auto-filled) ================== */
const INITIAL_IMG   = {initial_img_json};
const IMAGE_LIST    = {image_list_json};
const TILE_W        = {tile_w};
const TILE_H        = {tile_h};
const OFFSET_X      = {offset_x};
const OFFSET_Y      = {offset_y};
const INDEX_START   = {index_start};
const LABEL_EVERY   = {label_every};
const GRID_EVERY    = {grid_every};
const MAJOR_EVERY   = {major_every};

const GRID_MINOR    = "rgba(255,255,255,0.30)";
const GRID_MAJOR    = "rgba(255,255,255,0.78)";
const LABEL_RGBA    = "rgba(255,255,255,0.98)";
const LABEL_STROKE  = "rgba(0,0,0,1)";
const AXIS_BAND     = "rgba(20,22,27,0.55)";  // subtle dark band behind axis labels
const AXIS_TICK     = "rgba(255,255,255,0.9)";
const DOT_RADIUS    = 0;  // QC centers per tile (0 = off)
/* ========================================================== */

const viewer = OpenSeadragon({{
  id: "viewer",
  prefixUrl: "https://cdn.jsdelivr.net/npm/openseadragon@4.1/build/openseadragon/images/",
  showNavigator: true,
  maxZoomPixelRatio: 2.5,
  minZoomLevel: 0,
  visibilityRatio: 1.0,
  tileSources: {{ type: "image", url: INITIAL_IMG }}
}});

const gridCanvas = document.getElementById("gridCanvas");
const selectEl   = document.getElementById("imageSelect");
const fileInput  = document.getElementById("fileInput");
const toggleGrid   = document.getElementById("toggleGrid");
const toggleLabels = document.getElementById("toggleLabels");

function populateDropdown(list, initial) {{
  selectEl.innerHTML = "";
  list.forEach(name => {{
    const opt = document.createElement("option");
    opt.value = name; opt.textContent = name;
    if (name === initial) opt.selected = true;
    selectEl.appendChild(opt);
  }});
}}
populateDropdown(IMAGE_LIST, INITIAL_IMG);

selectEl.addEventListener("change", () => {{
  viewer.open({{ type:"image", url: selectEl.value }});
}});

fileInput.addEventListener("change", (ev) => {{
  const f = ev.target.files && ev.target.files[0];
  if (!f) return;
  const url = URL.createObjectURL(f);
  const label = f.name + " (local)";
  let opt = Array.from(selectEl.options).find(o => o.value === url);
  if (!opt) {{
    opt = document.createElement("option");
    opt.value = url; opt.textContent = label;
    selectEl.appendChild(opt);
  }}
  selectEl.value = url;
  viewer.open({{ type:"image", url }});
}});

function setCanvasSizeToViewer() {{
  const dpr = window.devicePixelRatio || 1;
  const rect = viewer.container.getBoundingClientRect();
  gridCanvas.style.width  = rect.width + "px";
  gridCanvas.style.height = rect.height + "px";
  gridCanvas.width  = Math.round(rect.width  * dpr);
  gridCanvas.height = Math.round(rect.height * dpr);
  return {{ dpr, rect }};
}}

function drawGrid() {{
  const world = viewer.world;
  if (!world || world.getItemCount() === 0) return;
  const item = world.getItemAt(0);

  const {{ dpr, rect }} = setCanvasSizeToViewer();
  const ctx = gridCanvas.getContext("2d");
  ctx.setTransform(1,0,0,1,0,0);
  ctx.clearRect(0,0,gridCanvas.width, gridCanvas.height);

  // Viewport bounds in image pixels
  const bounds = viewer.viewport.getBounds(true);
  const tl  = item.viewportToImageCoordinates(bounds.getTopLeft());
  const br  = item.viewportToImageCoordinates(bounds.getBottomRight());
  const minX = Math.max(0, Math.min(tl.x, br.x));
  const maxX = Math.max(tl.x, br.x);
  const minY = Math.max(0, Math.min(tl.y, br.y));
  const maxY = Math.max(tl.y, br.y);

  // Grid range in col/row indices
  const startCol = Math.floor((minX - OFFSET_X) / TILE_W);
  const endCol   = Math.ceil((maxX - OFFSET_X) / TILE_W);
  const startRow = Math.floor((minY - OFFSET_Y) / TILE_H);
  const endRow   = Math.ceil((maxY - OFFSET_Y) / TILE_H);

  // Image(px) -> Screen(px) (HiDPI aware)
  const imgToScreen = (ix, iy) => {{
    const vp = item.imageToViewportCoordinates(ix, iy);
    const px = viewer.viewport.pixelFromPoint(vp, true);
    const k = (window.devicePixelRatio || 1);
    return {{ x: px.x * k, y: px.y * k }};
  }};

  // ========== GRID (minor + major) ==========
  if (toggleGrid.checked) {{
    // Minor
    ctx.lineWidth = Math.max(1, Math.floor(dpr));
    ctx.strokeStyle = GRID_MINOR;
    ctx.beginPath();
    for (let c = startCol; c <= endCol; c++) {{
      if ((c % MAJOR_EVERY) === 0) continue;
      if ((c - startCol) % GRID_EVERY !== 0) continue;
      const xImg = OFFSET_X + c * TILE_W;
      const p1 = imgToScreen(xImg, minY);
      const p2 = imgToScreen(xImg, maxY);
      crispLine(ctx, p1.x, p1.y, p2.x, p2.y);
    }}
    for (let r = startRow; r <= endRow; r++) {{
      if ((r % MAJOR_EVERY) === 0) continue;
      if ((r - startRow) % GRID_EVERY !== 0) continue;
      const yImg = OFFSET_Y + r * TILE_H;
      const p1 = imgToScreen(minX, yImg);
      const p2 = imgToScreen(maxX, yImg);
      crispLine(ctx, p1.x, p1.y, p2.x, p2.y);
    }}
    ctx.stroke();

    // Major
    ctx.lineWidth = Math.max(2, Math.ceil(1.5 * dpr));
    ctx.strokeStyle = GRID_MAJOR;
    ctx.beginPath();
    for (let c = startCol; c <= endCol; c++) {{
      if ((c % MAJOR_EVERY) !== 0) continue;
      const xImg = OFFSET_X + c * TILE_W;
      const p1 = imgToScreen(xImg, minY);
      const p2 = imgToScreen(xImg, maxY);
      crispLine(ctx, p1.x, p1.y, p2.x, p2.y);
    }}
    for (let r = startRow; r <= endRow; r++) {{
      if ((r % MAJOR_EVERY) !== 0) continue;
      const yImg = OFFSET_Y + r * TILE_H;
      const p1 = imgToScreen(minX, yImg);
      const p2 = imgToScreen(maxX, yImg);
      crispLine(ctx, p1.x, p1.y, p2.x, p2.y);
    }}
    ctx.stroke();
  }}

  // ========== AXES THAT FOLLOW VIEWPORT ==========
  if (toggleLabels.checked) {{
    // Draw translucent bands along top & left edges so labels pop
    const bandH = Math.max(18 * dpr, 22); // px
    const bandW = Math.max(28 * dpr, 32); // px
    ctx.fillStyle = AXIS_BAND;
    ctx.fillRect(0, 0, gridCanvas.width, bandH);   // top band
    ctx.fillRect(0, 0, bandW, gridCanvas.height);  // left band

    ctx.strokeStyle = AXIS_TICK;

    // Top axis: ticks & labels for columns intersecting the viewport
    const topY = bandH - 1; // where tick bottoms hit
    const tickLenTop = Math.max(6 * dpr, 6);
    ctx.lineWidth = Math.max(1, Math.floor(dpr));
    for (let c = startCol; c <= endCol; c++) {{
      const xImg = OFFSET_X + c * TILE_W;
      const sp   = imgToScreen(xImg, minY);
      // tick mark
      crispLine(ctx, sp.x, 0, sp.x, tickLenTop);
    }}
    ctx.stroke();

    // Left axis: ticks & labels for rows intersecting the viewport
    const leftX = bandW - 1;
    const tickLenLeft = Math.max(6 * dpr, 6);
    ctx.beginPath();
    for (let r = startRow; r <= endRow; r++) {{
      const yImg = OFFSET_Y + r * TILE_H;
      const sp   = imgToScreen(minX, yImg);
      crispLine(ctx, 0, sp.y, tickLenLeft, sp.y);
    }}
    ctx.stroke();

    // Labels
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    ctx.lineWidth = 2 * dpr;
    const fontPx = Math.max(11, 12 * dpr);
    ctx.font = `${{fontPx}}px system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif`;

    // Column labels along top band (every LABEL_EVERY-th)
    for (let c = startCol; c <= endCol; c++) {{
      if ((c - INDEX_START) % {label_every} !== 0) continue;
      const centerImgX = OFFSET_X + c * TILE_W + TILE_W/2;
      const sp = imgToScreen(centerImgX, minY);
      drawLabel(ctx, (c + INDEX_START).toString(), sp.x, Math.max(2, 2*dpr));
    }}

    // Row labels along left band (every LABEL_EVERY-th)
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let r = startRow; r <= endRow; r++) {{
      if ((r - INDEX_START) % {label_every} !== 0) continue;
      const centerImgY = OFFSET_Y + r * TILE_H + TILE_H/2;
      const sp = imgToScreen(minX, centerImgY);
      drawLabel(ctx, (r + INDEX_START).toString(), Math.max(2, (bandW - 4) ), sp.y);
    }}
  }}

  // Optional QC dots at tile centers
  if (DOT_RADIUS > 0 && toggleGrid.checked) {{
    ctx.fillStyle = "rgba(255,255,0,0.95)";
    for (let c = startCol; c <= endCol; c++) {{
      for (let r = startRow; r <= endRow; r++) {{
        const cxImg = OFFSET_X + c * TILE_W + TILE_W/2;
        const cyImg = OFFSET_Y + r * TILE_H + TILE_H/2;
        const p = imgToScreen(cxImg, cyImg);
        ctx.beginPath();
        ctx.arc(p.x, p.y, DOT_RADIUS * dpr, 0, Math.PI*2);
        ctx.fill();
      }}
    }}
  }}
}}

function crispLine(ctx, x1, y1, x2, y2) {{
  const half = 0.5 * (ctx.lineWidth % 2);
  ctx.beginPath();
  ctx.moveTo(Math.round(x1)+half, Math.round(y1)+half);
  ctx.lineTo(Math.round(x2)+half, Math.round(y2)+half);
  ctx.stroke();
}}

function drawLabel(ctx, text, x, y) {{
  ctx.strokeStyle = "{label_stroke}";
  ctx.fillStyle = "{label_rgba}";
  ctx.strokeText(text, x, y);
  ctx.fillText(text, x, y);
}}

viewer.addHandler("open", drawGrid);
viewer.addHandler("animation", drawGrid);
viewer.addHandler("animation-finish", drawGrid);
viewer.addHandler("update-viewport", drawGrid);
viewer.addHandler("resize", drawGrid);
window.addEventListener("resize", drawGrid);
</script>
</body>
</html>
"""

def _pick_image_dialog() -> Path:
    if not _HAS_TK:
        raise RuntimeError("tkinter not available; set IMAGE_PATH explicitly.")
    root = _tk.Tk()
    root.withdraw()
    fname = _fd.askopenfilename(
        title="Select base image",
        filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All files", "*.*")]
    )
    root.update()
    root.destroy()
    if not fname:
        raise SystemExit("No image selected.")
    return Path(fname)

def _list_images_in_dir(directory: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [p.name for p in sorted(directory.iterdir()) if p.suffix.lower() in exts]

def generate_html(image_path: Path) -> Path:
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    out_dir = image_path.parent
    out_html = out_dir / "grid_viewer.html"

    image_names = _list_images_in_dir(out_dir)
    if image_path.name not in image_names:
        image_names.insert(0, image_path.name)

    html = _HTML_TEMPLATE.format(
        initial_img_json = '"' + image_path.name.replace('\\', '\\\\').replace('"', '\\"') + '"',
        image_list_json  = '[' + ','.join('"' + n.replace('\\','\\\\').replace('"','\\"') + '"' for n in image_names) + ']',
        tile_w=TILE_W, tile_h=TILE_H,
        offset_x=OFFSET_X, offset_y=OFFSET_Y,
        index_start=INDEX_START, label_every=LABEL_EVERY,
        grid_every=GRID_EVERY, major_every=MAJOR_EVERY,
        viewer_margin=int(VIEWER_MARGIN),
        label_rgba="rgba(255,255,255,0.98)",
        label_stroke="rgba(0,0,0,1)",
    )

    out_html.write_text(dedent(html), encoding="utf-8")
    print(f"[OK] Wrote HTML → {out_html}")
    if AUTO_OPEN:
        try:
            webbrowser.open(out_html.as_uri())
        except Exception as e:
            print(f"[WARN] Could not open in browser automatically: {e}")
    return out_html

# MAIN
if __name__ == "__main__" or True:
    try:
        img = IMAGE_PATH.strip()
        if not img:
            img = _pick_image_dialog().as_posix()
        generate_html(img)
    except SystemExit as se:
        print(se)
    except Exception as ex:
        print(f"[ERROR] {ex}")
