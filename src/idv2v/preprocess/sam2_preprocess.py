"""
SAM 2.1 automatic foreground segmentation (drop-in replacement for sam3.py).

No text prompt — SAM 2.1 AutomaticMaskGenerator detects all objects per frame.
Post-selection: keep masks whose area is >= --min-area-ratio of frame, and union
them. Result format is 100% compatible with sam3.py so orig_pixel.py works as-is.

Writes <output_dir>/frame_{k}/:
  sam3Mask_id_all.png                union mask of selected objects (0/255)
  sam3Mask_id_{i}.png                per-object mask (0/255)
  cropped_bbox_x.._y.._w.._h.._m{i}_input.png    RGB crop
  cropped_bbox_x.._y.._w.._h.._m{i}_sam3mask.png mask crop

Also writes <output_dir>/*__sam3_overlay.mp4 (debug overlay video).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from .secret_panda import secret_panda


# ---------------------------------------------------------------------------
# SAM 2.1 helpers (delayed import — pip install sam2 first)
# ---------------------------------------------------------------------------
def build_sam2_automatic(predictor_path: str, config_path: str, device="cuda"):
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2 = build_sam2(config_path, predictor_path)
    gen = SAM2AutomaticMaskGenerator(sam2)
    return gen


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_video_frames(video_path: str, max_frames: Optional[int] = None) -> List[Image.Image]:
    """Load video as list of PIL RGB frames via imageio/ffmpeg."""
    import imageio.v3 as iio

    frames = iio.imread(video_path, plugin="pyav")  # (T, H, W, C) uint8
    frames = [Image.fromarray(frames[i], mode="RGB") for i in range(frames.shape[0])]
    if max_frames is not None:
        frames = frames[:max_frames]
    return frames


def _filter_by_area(masks_info: List[Dict[str, Any]], H: int, W: int,
                    min_area_ratio: float = 0.005, max_area_ratio: float = 0.95,
                    top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    """Keep masks whose area is within [min, max] ratio, optionally capped to top_n by area."""
    frame_area = H * W
    kept = []
    for m in masks_info:
        area = float(m.get("area", m["segmentation"].sum()))
        ratio = area / frame_area
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue
        kept.append(m)
    if top_n is not None and len(kept) > top_n:
        kept.sort(key=lambda d: float(d.get("area", 0)), reverse=True)
        kept = kept[:top_n]
    return kept


# ---------------------------------------------------------------------------
# Save per-frame folders (EXACT same format as sam3.py)
# ---------------------------------------------------------------------------
def save_sam2_per_frame_folders(
    video_frames: List[Image.Image],
    per_frame_instances: Dict[int, List[Dict[str, Any]]],
    output_dir: str,
    mask_bin_threshold: float = 0.5,
    close_kernel: int = 10,
    bridge_distance: int = 15,
    joint_mask_post_proc: bool = True,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    H = video_frames[0].height
    W = video_frames[0].width

    for frame_idx in tqdm(range(len(video_frames)), desc="Saving per-frame results"):
        frame_dir = output_dir / f"frame_{frame_idx}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        instances = per_frame_instances.get(frame_idx, [])
        if len(instances) == 0:
            Image.fromarray(np.zeros((H, W), dtype=np.uint8), mode="L").save(
                frame_dir / "sam3Mask_id_all.png")
            continue

        # Sort by area
        instances_sorted = sorted(instances, key=lambda d: float(d.get("area", 0)), reverse=True)

        union_mask = np.zeros((H, W), dtype=bool)
        frame_np = np.asarray(video_frames[frame_idx].convert("RGB"))

        for id_index, inst in enumerate(instances_sorted):
            m = np.asarray(inst["segmentation"], dtype=bool)
            if m.shape != (H, W):
                continue

            m = secret_panda(
                m.astype(np.uint8),
                fill_holes_first=True,
                close_kernel=close_kernel,
                bridge_distance=bridge_distance,
            ).astype(bool)

            union_mask |= m

            Image.fromarray((m.astype(np.uint8) * 255), mode="L").save(
                frame_dir / f"sam3Mask_id_{id_index}.png")

            ys, xs = np.where(m)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            w, h = x2 - x1, y2 - y1

            Image.fromarray(frame_np[y1:y2, x1:x2, :]).save(
                frame_dir / f"cropped_bbox_x{x1}_y{y1}_w{w}_h{h}_m{id_index}_input.png")
            Image.fromarray((m[y1:y2, x1:x2].astype(np.uint8) * 255), mode="L").save(
                frame_dir / f"cropped_bbox_x{x1}_y{y1}_w{w}_h{h}_m{id_index}_sam3mask.png")

        if joint_mask_post_proc:
            union_mask = secret_panda(
                union_mask.astype(np.uint8),
                fill_holes_first=True,
                close_kernel=close_kernel,
                bridge_distance=bridge_distance,
            ).astype(bool)

        Image.fromarray((union_mask.astype(np.uint8) * 255), mode="L").save(
            frame_dir / "sam3Mask_id_all.png")


def write_overlay_video(video_frames: List[Image.Image],
                        mask_dir: Path,
                        out_path: str,
                        alpha: float = 0.4) -> None:
    """Write frame_N.png overlaid with its sam3Mask_id_all.png as red tint."""
    import imageio.v3 as iio

    H, W = video_frames[0].height, video_frames[0].width
    writer = iio.imopen(out_path, "w", plugin="pyav")
    for i, frame in enumerate(video_frames):
        frame_dir = mask_dir / f"frame_{i}"
        mask_path = frame_dir / "sam3Mask_id_all.png"
        if mask_path.exists():
            mask = np.asarray(Image.open(mask_path).convert("L")).astype(np.float32) / 255.0
        else:
            mask = np.zeros((H, W), dtype=np.float32)
        frame_np = np.asarray(frame).astype(np.float32)
        overlay = frame_np * (1 - alpha * mask[..., None]) + np.array([255, 0, 0], dtype=np.float32) * (alpha * mask[..., None])
        writer.write_frame(overlay.astype(np.uint8))
    writer.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SAM 2.1 automatic segmentation (SAM3 replacement)")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path",
                        default="checkpoints/VACE-Annotators/sam2/sam2.1_hq_hiera_large.pt")
    parser.add_argument("--config_path",
                        default="checkpoints/VACE-Annotators/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_area_ratio", type=float, default=0.005,
                        help="Discard masks smaller than this fraction of frame (default 0.5%)")
    parser.add_argument("--max_area_ratio", type=float, default=0.95,
                        help="Discard masks larger than this (default 95%)")
    parser.add_argument("--top_n", type=int, default=5,
                        help="Keep at most N largest surviving masks per frame")
    parser.add_argument("--joint_mask_post_proc", action="store_true", default=True)
    parser.add_argument("--close_kernel", type=int, default=10)
    parser.add_argument("--bridge_distance", type=int, default=15)
    args = parser.parse_args()

    frames = load_video_frames(args.video_path)
    H, W = frames[0].height, frames[0].width
    print(f"[sam2_preprocess] Loaded {len(frames)} frames, {W}x{H}")

    gen = build_sam2_automatic(args.model_path, args.config_path, device=args.device)
    print(f"[sam2_preprocess] SAM 2.1 loaded from {args.model_path}")

    per_frame_instances: Dict[int, List[Dict[str, Any]]] = {}
    for i, frame in enumerate(tqdm(frames, desc="SAM2 auto-segmentation")):
        result = gen.generate(np.asarray(frame))
        # automatic generator returns list of dicts with keys: area, segmentation (bool HxW), bbox, ...
        filtered = _filter_by_area(result, H, W,
                                   min_area_ratio=args.min_area_ratio,
                                   max_area_ratio=args.max_area_ratio,
                                   top_n=args.top_n)
        # Add deterministic "id" (index in sorted-area order)
        filtered.sort(key=lambda d: float(d.get("area", 0)), reverse=True)
        for idx, d in enumerate(filtered):
            d["id"] = idx
        per_frame_instances[i] = filtered

    save_sam2_per_frame_folders(
        video_frames=frames,
        per_frame_instances=per_frame_instances,
        output_dir=args.output_dir,
        close_kernel=args.close_kernel,
        bridge_distance=args.bridge_distance,
        joint_mask_post_proc=args.joint_mask_post_proc,
    )

    # Overlay debug video
    out_stem = Path(args.video_path).stem
    overlay_path = str(Path(args.output_dir) / f"{out_stem}__sam3_overlay.mp4")
    write_overlay_video(frames, Path(args.output_dir), overlay_path)
    print(f"[sam2_preprocess] Done. Masks in {args.output_dir}")
    print(f"  Overlay debug video: {overlay_path}")


if __name__ == "__main__":
    main()
