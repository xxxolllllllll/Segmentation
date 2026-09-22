# -*- coding: utf-8 -*-
"""Offline teacher soft-label generation for Phase-3 soft-label distillation.

For each unlabeled image, slides the *same* windows Stage-C uses (top-left crop,
``window=imgsz``, ``stride=window_stride``), runs one or more frozen Stage-B
teachers, and stores the crack logit map (fp16) per window.

Storage note: at full 1024x1024 fp16 a single-channel crack logit is ~2 MiB per
window; with ~275k windows that is ~550 GB. Use ``--save-res`` to downsample or
``--only-positive`` to filter if space is limited.

Teacher/leakage: pass the *fold's own* Stage-B teacher (trained without that
fold's test images) to avoid leakage. Multiple ``--teacher-ckpts`` are ensembled.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_ROLE = Path(__file__).resolve().parents[1]
if str(_ROLE) not in sys.path:
    sys.path.insert(0, str(_ROLE))

from checkpoint_io import torch_load_compat  # noqa: E402
from models.dino_stage_b_unet import DINOv3StageBUNet  # noqa: E402

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def axis_window_starts(dim: int, patch: int, stride: int) -> list[int]:
    if patch <= 0 or stride <= 0:
        raise ValueError("patch and stride must be positive")
    if dim <= patch:
        return [0]
    starts = list(range(0, dim - patch + 1, stride))
    last = dim - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def sliding_window_toplefts(height: int, width: int, patch: int, stride: int) -> list[tuple[int, int]]:
    ys = axis_window_starts(height, patch, stride)
    xs = axis_window_starts(width, patch, stride)
    return [(y, x) for y in ys for x in xs]


def crop_top_left_rgb(img: np.ndarray, y0: int, x0: int, ph: int, pw: int) -> np.ndarray:
    """Top-left crop with zero padding at the right/bottom edge (matches Stage C)."""
    out = np.zeros((ph, pw, img.shape[2]), dtype=img.dtype)
    h = min(ph, img.shape[0] - y0)
    w = min(pw, img.shape[1] - x0)
    out[:h, :w] = img[y0 : y0 + h, x0 : x0 + w]
    return out


def window_to_tensor(window_rgb: np.ndarray, imgsz: int) -> torch.Tensor:
    img = Image.fromarray(window_rgb)
    if img.size != (imgsz, imgsz):
        img = img.resize((imgsz, imgsz), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(x, (2, 0, 1)))


def discover_images(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() in IMAGE_SUFFIXES:
            out.append(root)
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                out.append(p)
    return out


def load_teacher(ckpt: Path, weights: str, device: torch.device, no_pretrained: bool) -> DINOv3StageBUNet:
    cfg_path = ckpt.parent / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    teacher = DINOv3StageBUNet(
        weights_dir=weights or cfg.get("teacher_weights", ""),
        pretrained=not no_pretrained,
        device=device,
        num_classes=int(cfg.get("num_classes", 2)),
        bottleneck_dim=int(cfg.get("adapter_bottleneck", 64)),
        adapter_dropout=float(cfg.get("adapter_dropout", 0.1)),
    ).to(device)
    state = torch_load_compat(ckpt, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def rel_key(image_path: Path, roots: list[Path]) -> str:
    for root in roots:
        try:
            return str(image_path.relative_to(root))
        except ValueError:
            continue
    return image_path.name


def npy_for(out_dir: Path, key: str, y0: int, x0: int) -> Path:
    stem = key.replace("\\", "/").rsplit(".", 1)[0]
    return out_dir / "windows" / f"{stem}_y{y0}_x{x0}.npy"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate offline Stage-B soft labels (crack logits)")
    p.add_argument("--teacher-ckpts", type=Path, nargs="+", required=True, help="Stage-B best.pt (one per ensemble member)")
    p.add_argument("--teacher-weights", type=str, default="", help="Local HF DINOv3 dir (else read from ckpt config.json)")
    p.add_argument("--teacher-no-pretrained", action="store_true")
    p.add_argument("--image-roots", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--window-stride", type=int, default=800)
    p.add_argument("--save-res", type=int, default=0, help="Downsample stored logit to this side (0 = full imgsz)")
    p.add_argument("--save-logits", choices=("crack", "both"), default="crack", help="crack = single channel fp16; both = 2 channels")
    p.add_argument("--ensemble", choices=("logit_mean", "prob_mean"), default="logit_mean")
    p.add_argument("--batch-size", type=int, default=4, help="Windows per teacher forward")
    p.add_argument("--only-positive", action="store_true", help="Store only windows whose teacher prob exceeds --positive-thresh")
    p.add_argument("--positive-thresh", type=float, default=0.3)
    p.add_argument("--max-images", type=int, default=0, help="Debug: limit number of images")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    roots = [r.expanduser().resolve() for r in args.image_roots]
    out_dir = args.output_dir.expanduser().resolve()
    (out_dir / "windows").mkdir(parents=True, exist_ok=True)

    images = discover_images(roots)
    if args.max_images > 0:
        images = images[: args.max_images]
    if not images:
        raise FileNotFoundError(f"No images under: {roots}")

    teachers = [
        load_teacher(c.expanduser().resolve(), args.teacher_weights.strip(), device, args.teacher_no_pretrained)
        for c in args.teacher_ckpts
    ]
    print(f"[softlabel] {len(images)} images, {len(teachers)} teacher(s), imgsz={args.imgsz}, stride={args.window_stride}", flush=True)

    manifest_path = out_dir / "manifest.jsonl"
    n_windows = 0
    n_stored = 0
    with manifest_path.open("a", encoding="utf-8") as mf:
        for ii, image_path in enumerate(images, start=1):
            try:
                with Image.open(image_path) as im:
                    image_rgb = np.array(im.convert("RGB"), dtype=np.uint8)
            except Exception:
                continue
            h, w = image_rgb.shape[:2]
            windows = sliding_window_toplefts(h, w, args.imgsz, args.window_stride)
            key = rel_key(image_path, roots)

            pending: list[tuple[int, int, torch.Tensor]] = []
            for y0, x0 in windows:
                n_windows += 1
                npy_path = npy_for(out_dir, key, y0, x0)
                if npy_path.is_file() and not args.overwrite:
                    continue
                pending.append((y0, x0, window_to_tensor(crop_top_left_rgb(image_rgb, y0, x0, args.imgsz, args.imgsz), args.imgsz)))

            for start in range(0, len(pending), args.batch_size):
                chunk = pending[start : start + args.batch_size]
                batch = torch.stack([c[2] for c in chunk], dim=0).to(device)
                logits = [t(batch).float() for t in teachers]
                if args.ensemble == "prob_mean":
                    probs = [torch.softmax(lg, dim=1) for lg in logits]
                    ensemble = torch.stack(probs, 0).mean(0)
                    ensemble = torch.log(ensemble.clamp_min(1e-6))
                else:
                    ensemble = torch.stack(logits, 0).mean(0)  # [B, C, H, W]

                for (y0, x0, _), lg in zip(chunk, ensemble):
                    if args.save_logits == "crack":
                        arr = lg[1:2]
                    else:
                        arr = lg
                    if args.only_positive and args.save_logits == "crack":
                        if float(torch.sigmoid(lg[1]).max()) < args.positive_thresh:
                            continue
                    if args.save_res > 0 and args.save_res != arr.shape[-1]:
                        arr = torch.nn.functional.interpolate(
                            arr.unsqueeze(0), size=(args.save_res, args.save_res), mode="bilinear", align_corners=False
                        ).squeeze(0)
                    npy_path = npy_for(out_dir, key, y0, x0)
                    npy_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(npy_path, arr.half().cpu().numpy().astype(np.float16))
                    mf.write(json.dumps({"rel": key, "y": y0, "x": x0, "npy": str(npy_path.relative_to(out_dir))}) + "\n")
                    n_stored += 1

            if args.log_every > 0 and ii % args.log_every == 0:
                mf.flush()
                print(f"[softlabel] {ii}/{len(images)} images, stored {n_stored}/{n_windows} windows", flush=True)

    meta = {
        "image_roots": [str(r) for r in roots],
        "imgsz": args.imgsz,
        "window_stride": args.window_stride,
        "save_res": args.save_res,
        "save_logits": args.save_logits,
        "teacher_ckpts": [str(c) for c in args.teacher_ckpts],
        "ensemble": args.ensemble,
        "n_images": len(images),
        "n_windows_seen": n_windows,
        "n_windows_stored": n_stored,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[softlabel] done: stored {n_stored}/{n_windows} windows -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
