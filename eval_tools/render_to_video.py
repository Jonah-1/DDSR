#!/usr/bin/env python3
"""
Convert eval_rendering.py output (image sequences) to videos and composite grids.

Expected input structure:
  <input-dir>/
    sample_0000/
      gt_views/
        280x518_cam_0_pred.png  280x518_cam_0_gt.png
      left_0.5m/  left_1.0m/  right_0.5m/  right_1.0m/
        280x518_cam_0.png
    sample_0006/ ...

Usage:
  # List discovered views and cameras
  python eval_tools/render_to_video.py --input-dir .../scene_004 --list

  # Convert all view types to per-cam videos
  python eval_tools/render_to_video.py \\
    --input-dir work_dirs/inference_results/scene_004 \\
    --output-dir work_dirs/videos/scene_004 --fps 10

  # Convert + 2D composite with text labels (rows separated by ';', cols by ',')
  python eval_tools/render_to_video.py \\
    --input-dir work_dirs/inference_results/scene_004 \\
    --output-dir work_dirs/videos/scene_004 --fps 10 --cams 0 \\
    --layout "left_1.0m,gt_views_gt,right_1.0m;left_2.0m,gt_views_pred,right_2.0m" \\
    --labels "左移1m,原图,右移1m;左移2m,渲染图,右移2m"

  # Compose only (individual videos already exist)
  python eval_tools/render_to_video.py \\
    --input-dir work_dirs/inference_results/scene_004 \\
    --output-dir work_dirs/videos/scene_004 --fps 10 --cams 0 \\
    --layout "left_1.0m,gt_views_gt,right_1.0m;left_2.0m,gt_views_pred,right_2.0m" \\
    --labels "左移1m,原图,右移1m;左移2m,渲染图,右移2m" \\
    --composite-only

  # 3-cam composite: rows = view types, cols = cameras
  python eval_tools/render_to_video.py \\
    --input-dir work_dirs/inference_results/scene_004 \\
    --output-dir work_dirs/videos/scene_004 --fps 10 \\
    --cams 0,1,2 \\
    --layout "gt_views_gt;gt_views_pred" \\
    --labels "GT;渲染" \\
    --cam-per-col
"""

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_view_cams(root_dir: Path) -> dict:
    """Scan sample_* dirs → {view_key: [cam_ids]}. gt_views → _pred/_gt."""
    view_cams: dict = defaultdict(set)
    cam_re = re.compile(r"_cam_(\d+)")
    for sample_dir in sorted(root_dir.glob("sample_*")):
        for view_dir in sorted(sample_dir.iterdir()):
            if not view_dir.is_dir():
                continue
            vname = view_dir.name
            for img in view_dir.glob("*.png"):
                m = cam_re.search(img.stem)
                if not m:
                    continue
                cam_id = int(m.group(1))
                if vname == "gt_views":
                    if img.stem.endswith("_pred"):
                        view_cams["gt_views_pred"].add(cam_id)
                    elif img.stem.endswith("_gt"):
                        view_cams["gt_views_gt"].add(cam_id)
                else:
                    view_cams[vname].add(cam_id)
    return {k: sorted(v) for k, v in sorted(view_cams.items())}


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------

def collect_images(root_dir: Path, view_key: str, cam_id: int) -> list:
    cam_tag = f"_cam_{cam_id}"
    if view_key.startswith("gt_views_"):
        view_dir_name = "gt_views"
        suffix = "_" + view_key.split("_")[-1]
    else:
        view_dir_name = view_key
        suffix = None
    items = []
    for sample_dir in sorted(root_dir.glob("sample_*")):
        m = re.match(r"sample_(\d+)$", sample_dir.name)
        if not m:
            continue
        view_path = sample_dir / view_dir_name
        if not view_path.exists():
            continue
        for img in sorted(view_path.glob("*.png")):
            if cam_tag not in img.stem:
                continue
            if suffix and not img.stem.endswith(suffix):
                continue
            items.append((int(m.group(1)), img))
            break
    return [p for _, p in sorted(items)]


# ---------------------------------------------------------------------------
# Single-video creation
# ---------------------------------------------------------------------------

def make_video(images: list, out: Path, fps: int) -> bool:
    if not images:
        print(f"  [skip] no images -> {out.name}")
        return False
    list_file = out.with_suffix(".txt")
    with open(list_file, "w") as f:
        for p in images:
            f.write(f"file '{Path(p).resolve()}'\n")
            f.write(f"duration {1.0 / fps:.6f}\n")
    try:
        import cv2
        img = cv2.imread(str(images[0]))
        if img is None:
            raise RuntimeError(f"cannot read {images[0]}")
        h, w = img.shape[:2]
        w, h = (w // 2) * 2, (h // 2) * 2
    except Exception as e:
        print(f"  [error] {e}")
        list_file.unlink(missing_ok=True)
        return False
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-vf", f"scale={w}:{h}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-pix_fmt", "yuv420p", str(out),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(f"  [ffmpeg error] {r.stderr[-300:]}")
            return False
        print(f"  -> {out.name}  ({len(images)} frames @ {fps}fps)")
        return True
    except Exception as e:
        print(f"  [error] {e}")
        return False
    finally:
        list_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Composite grid
# ---------------------------------------------------------------------------

def get_video_size(path: Path):
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            w, h = r.stdout.strip().split(",")
            return int(w), int(h)
    except Exception:
        pass
    return None, None


def _find_cjk_font() -> str:
    """Return path to a CJK-capable font, or empty string to use ffmpeg default."""
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return p
    return ""


def compose_grid(video_map: dict, grid: list, labels: list,
                 default_cam: int, out: Path, fps: int) -> bool:
    """
    Build composite video.

    grid:        list[list[tuple[str, int|None]]]  - (view_key, cam_or_None) per cell
    labels:      list[list[str]]  - text label per cell (same shape), may be None
    default_cam: camera used when a cell has no explicit @cam annotation
    """
    n_rows = len(grid)
    n_cols = max(len(row) for row in grid)

    # Collect input video paths and validate
    cell_videos = {}   # (r, c) -> Path
    for r, row in enumerate(grid):
        for c, (view_key, cell_cam) in enumerate(row):
            effective_cam = cell_cam if cell_cam is not None else default_cam
            key = (view_key, effective_cam)
            if key not in video_map:
                print(f"  [error] missing video: {view_key} cam{effective_cam}")
                return False
            cell_videos[(r, c)] = video_map[key]

    # Get cell dimensions from first video
    w, h = get_video_size(next(iter(cell_videos.values())))
    if w is None:
        print("  [error] cannot read video dimensions")
        return False
    w, h = (w // 2) * 2, (h // 2) * 2

    TEXT_H = 40          # pixels for label bar
    FONT_SIZE = 22
    font_file = _find_cjk_font()
    font_arg = f":fontfile='{font_file}'" if font_file else ""

    print(f"  Grid {n_rows}x{n_cols}  cell {w}x{h}  text_bar {TEXT_H}px")
    if font_file:
        print(f"  Font: {font_file}")
    else:
        print("  Font: ffmpeg default (CJK may not render; install noto-cjk or wqy-microhei)")

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y"]
    input_idx = {}   # (r, c) -> ffmpeg input index
    for (r, c), path in sorted(cell_videos.items()):
        input_idx[(r, c)] = len(input_idx)
        cmd.extend(["-i", str(path)])

    filters = []

    # Per-cell: scale → pad (add label bar) → drawtext
    for (r, c), i in input_idx.items():
        cell_h_total = h + TEXT_H
        label = (labels[r][c] if labels and r < len(labels) and c < len(labels[r])
                 else "")

        filters.append(f"[{i}:v]scale={w}:{h}[s{i}]")
        filters.append(
            f"[s{i}]pad={w}:{cell_h_total}:0:0:color=black[p{i}]"
        )
        if label:
            # Escape special chars for ffmpeg drawtext
            safe_label = (label
                          .replace("\\", "\\\\")
                          .replace("'", "\\'")
                          .replace(":", "\\:"))
            text_y = h + (TEXT_H - FONT_SIZE) // 2
            filters.append(
                f"[p{i}]drawtext=text='{safe_label}'"
                f"{font_arg}"
                f":fontsize={FONT_SIZE}:fontcolor=white"
                f":x=(w-text_w)/2:y={text_y}[cell{i}]"
            )
        else:
            filters.append(f"[p{i}]copy[cell{i}]")

    # hstack each row
    row_labels = []
    for r in range(n_rows):
        cols_in_row = [c for c in range(n_cols) if (r, c) in input_idx]
        parts = "".join(f"[cell{input_idx[(r, c)]}]" for c in cols_in_row)
        label = f"r{r}"
        row_labels.append(label)
        if len(cols_in_row) == 1:
            filters.append(f"{parts}copy[{label}]")
        else:
            filters.append(f"{parts}hstack=inputs={len(cols_in_row)}[{label}]")

    # vstack all rows
    parts = "".join(f"[{l}]" for l in row_labels)
    if len(row_labels) == 1:
        filters.append(f"{parts}copy[out]")
    else:
        filters.append(f"{parts}vstack=inputs={n_rows}[out]")

    cmd.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[out]",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-pix_fmt", "yuv420p", str(out),
    ])

    print("  Running ffmpeg...")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            print(f"  [ffmpeg error] {r.stderr[-500:]}")
            return False
        print(f"  -> {out.name}")
        return True
    except Exception as e:
        print(f"  [error] {e}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_grid_arg(s: str) -> list:
    """
    Parse layout string into rows of (view_key, cam_or_None) tuples.
    'a,b@1;c@2,d' → [[('a',None),('b',1)],[('c',2),('d',None)]]
    """
    result = []
    for row in s.split(";"):
        row_cells = []
        for cell in row.split(","):
            cell = cell.strip()
            if "@" in cell:
                view, cam = cell.rsplit("@", 1)
                row_cells.append((view.strip(), int(cam)))
            else:
                row_cells.append((cell, None))
        result.append(row_cells)
    return result


def parse_labels_arg(s: str) -> list:
    """'a,b;c,d' → [['a','b'],['c','d']]"""
    return [[cell.strip() for cell in row.split(",")] for row in s.split(";")]


def main():
    parser = argparse.ArgumentParser(
        description="Convert eval_rendering.py output to videos and composite grids"
    )
    parser.add_argument("--input-dir", required=True,
                        help="Scene directory containing sample_* subdirectories")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <input-dir>/videos)")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--views", type=str, default=None,
                        help="Comma-separated view keys to convert (default: all)")
    parser.add_argument("--cams", type=str, default=None,
                        help="Comma-separated cam IDs for composite (default: all)")
    parser.add_argument(
        "--layout", type=str, default=None,
        help="2D grid of view keys. Rows separated by ';', cols by ','. "
             "e.g. 'left_1.0m,gt_views_gt,right_1.0m;left_2.0m,gt_views_pred,right_2.0m'"
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help="Text labels matching --layout shape. "
             "e.g. '左移1m,原图,右移1m;左移2m,渲染图,右移2m'"
    )
    parser.add_argument("--composite-only", action="store_true",
                        help="Skip individual video creation; load existing .mp4 files")
    parser.add_argument("--cam-per-col", action="store_true",
                        help="Multi-cam composite: --layout rows = view types, columns = cameras "
                             "from --cams. Produces one composite_allcams video instead of one "
                             "per camera. --labels rows are auto-expanded to match cam count.")
    parser.add_argument("--list", action="store_true",
                        help="Print discovered views/cameras and exit")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "videos"

    if not input_dir.exists():
        print(f"[error] not found: {input_dir}")
        sys.exit(1)

    # --- Discover ---
    print(f"\nScanning {input_dir} ...")
    view_cams = discover_view_cams(input_dir)
    if not view_cams:
        print("[error] no images found under sample_* directories")
        sys.exit(1)

    print("Discovered:")
    for v, cams in view_cams.items():
        print(f"  {v:30s}  cams: {cams}")

    if args.list:
        return

    if args.views:
        wanted = {v.strip() for v in args.views.split(",")}
        view_cams = {k: v for k, v in view_cams.items() if k in wanted}

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: individual videos ---
    video_map: dict = {}   # {(view_key, cam_id): Path}

    if not args.composite_only:
        print(f"\n{'='*60}")
        print("Stage 1  individual videos")
        print(f"{'='*60}")
        # Apply --cams filter to stage 1 as well
        filter_cams = {int(c.strip()) for c in args.cams.split(",")} if args.cams else None
        # When --cam-per-col is active, auto-restrict views to those in --layout
        if args.cam_per_col and args.layout and not args.views:
            layout_views = {v.strip() for v in args.layout.split(";")}
            view_cams = {k: v for k, v in view_cams.items() if k in layout_views}
        ok = total = 0
        for view_key, cam_ids in view_cams.items():
            for cam_id in cam_ids:
                if filter_cams and cam_id not in filter_cams:
                    continue
                total += 1
                imgs = collect_images(input_dir, view_key, cam_id)
                out = output_dir / f"{view_key}_cam{cam_id}_{args.fps}fps.mp4"
                if make_video(imgs, out, args.fps):
                    video_map[(view_key, cam_id)] = out
                    ok += 1
        print(f"\nStage 1: {ok}/{total} videos  ->  {output_dir}")
    else:
        for vf in sorted(output_dir.glob("*.mp4")):
            m = re.match(r"(.+)_cam(\d+)_\d+fps\.mp4$", vf.name)
            if m:
                video_map[(m.group(1), int(m.group(2)))] = vf
        print(f"\nLoaded {len(video_map)} existing videos from {output_dir}")

    # --- Stage 2: composite ---
    if not args.layout:
        print("\nTip: use --layout to create a composite video.")
        print('  e.g. --layout "left_1.0m,gt_views_gt,right_1.0m;left_2.0m,gt_views_pred,right_2.0m"')
        print('       --labels "左移1m,原图,右移1m;左移2m,渲染图,右移2m"')
        print('  3-cam: --layout "gt_views_gt;gt_views_pred" --cams 0,1,2 --cam-per-col')
        return

    print(f"\n{'='*60}")
    print("Stage 2  composite grid")
    print(f"{'='*60}")

    if args.cam_per_col:
        # Multi-cam composite: rows = view types, cols = cameras
        if not args.cams:
            print("[error] --cam-per-col requires --cams")
            return
        cam_ids_list = [int(c.strip()) for c in args.cams.split(",")]
        row_views = [r.strip() for r in args.layout.split(";")]

        # Build @cam-annotated layout
        layout_rows = [",".join(f"{v}@{c}" for c in cam_ids_list) for v in row_views]
        grid = parse_grid_arg(";".join(layout_rows))

        # Expand labels: if a row has 1 label, repeat for each cam
        if args.labels:
            expanded = []
            for lr in args.labels.split(";"):
                parts = [l.strip() for l in lr.split(",")]
                if len(parts) == 1:
                    parts = parts * len(cam_ids_list)
                expanded.append(",".join(parts))
            labels = parse_labels_arg(";".join(expanded))
        else:
            cam_labels = ",".join(f"cam{c}" for c in cam_ids_list)
            labels = parse_labels_arg(";".join([cam_labels] * len(row_views)))

        print(f"\n  --- all cams: {cam_ids_list} ---")
        composite_out = output_dir / f"composite_allcams_{args.fps}fps.mp4"
        if compose_grid(video_map, grid, labels, cam_ids_list[0], composite_out, args.fps):
            print(f"  Saved: {composite_out}")
        else:
            print(f"  [error] composite failed")
    else:
        grid = parse_grid_arg(args.layout)
        labels = parse_labels_arg(args.labels) if args.labels else None

        # Determine which cameras to composite
        if args.cams:
            cam_ids = [int(c.strip()) for c in args.cams.split(",")]
        else:
            all_views_in_grid = {v for row in grid for v, _ in row}
            cam_sets = [set(view_cams.get(v, [])) for v in all_views_in_grid]
            cam_ids = sorted(set.intersection(*cam_sets)) if cam_sets else []
            if not cam_ids:
                print(f"[error] no common cameras for layout views")
                return

        for cam_id in cam_ids:
            print(f"\n  --- cam {cam_id} ---")
            composite_out = output_dir / f"composite_cam{cam_id}_{args.fps}fps.mp4"
            if compose_grid(video_map, grid, labels, cam_id, composite_out, args.fps):
                print(f"  Saved: {composite_out}")
            else:
                print(f"  [error] composite failed for cam {cam_id}")


if __name__ == "__main__":
    main()
