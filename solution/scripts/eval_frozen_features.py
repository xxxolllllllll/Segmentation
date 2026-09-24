# -*- coding: utf-8 -*-
"""L2 evaluation: frozen DINOv3 features + a linear 1x1 segmentation head.

Native-resolution sliding windows (window = --img-size, stride = --window-stride),
matching the Stage-B/C data pipeline -- the full image is NOT downscaled.

For each window the backbone hidden state at --feature-layer is taken (adapters
applied if --stage-a-ckpt is given), cached to disk, and a single 1x1 conv head
is trained on the fold's train windows. Evaluation stitches per-window logits
into a full-image prediction and reports crack IoU (within the valid region).

Run three times (shallow/mid/deep layer) with/without --stage-a-ckpt to compare
raw vs adapted frozen features.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from checkpoint_io import torch_load_compat  # noqa: E402
from folds import discover_samples, fold_train_val_test, load_folds  # noqa: E402
from models.dino_stage_a import BottleneckResidualAdapter  # noqa: E402
from models.teacher_vit import _load_vit_backbone  # noqa: E402
from scripts.labelme_crack_copy_paste import parse_csv_set, rasterize_masks, read_labelme  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IGNORE_INDEX = 255


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Native-window frozen-feature linear segmentation probe (L2)")
    p.add_argument("--teacher-weights", type=str, required=True)
    p.add_argument("--stage-a-ckpt", type=Path, default=None, help="Optional Stage-A ckpt (adapters). Omit for raw.")
    p.add_argument("--adapter-state-key", type=str, default="teacher_adapters_ema", choices=("teacher_adapters_ema", "student_adapters"))
    p.add_argument("--adapter-bottleneck", type=int, default=64)
    p.add_argument("--adapter-dropout", type=float, default=0.1)
    p.add_argument("--labelme-dir", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, default=None)
    p.add_argument("--fold-split", type=Path, default=None)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--feature-layer", type=int, default=11, help="Single hidden-state index for this run")
    p.add_argument("--img-size", type=int, default=1024, help="Window side (= backbone input)")
    p.add_argument("--window-stride", type=int, default=800, help="Sliding stride (pipeline value)")
    p.add_argument("--cache-dir", type=Path, default=None, help="Feature cache dir (default: <output-dir>/cache)")
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--max-train-images", type=int, default=0)
    p.add_argument("--max-test-images", type=int, default=0)
    p.add_argument("--windows-per-epoch", type=int, default=0, help="Cap train windows per epoch (0 = all)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--crack-labels", type=str, default="crack")
    p.add_argument("--ignore-labels", type=str, default="ignore")
    p.add_argument("--component-labels", type=str, default="component,wood")
    p.add_argument("--output-dir", type=Path, default=Path("runs/analysis/frozen_probe"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def axis_window_starts(dim: int, patch: int, stride: int) -> list[int]:
    if dim <= patch:
        return [0]
    starts = list(range(0, dim - patch + 1, stride))
    if starts[-1] != dim - patch:
        starts.append(dim - patch)
    return starts


def sliding_window_toplefts(h: int, w: int, patch: int, stride: int) -> list[tuple[int, int]]:
    return [(y, x) for y in axis_window_starts(h, patch, stride) for x in axis_window_starts(w, patch, stride)]


def crop_top_left_rgb(img: np.ndarray, y0: int, x0: int, ph: int, pw: int) -> np.ndarray:
    out = np.zeros((ph, pw, img.shape[2]), dtype=img.dtype)
    h = min(ph, img.shape[0] - y0)
    w = min(pw, img.shape[1] - x0)
    out[:h, :w] = img[y0 : y0 + h, x0 : x0 + w]
    return out


def build_adapters(ckpt_path: Path, state_key: str, bottleneck: int, dropout: float) -> nn.ModuleDict:
    ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
    indices = tuple(int(i) for i in ckpt.get("adapter_indices", (3, 4, 7, 8, 11, 12)))
    hidden = int(ckpt.get("hidden_size", 768))
    sd = ckpt.get(state_key)
    if sd is None:
        raise KeyError(f"checkpoint missing {state_key}")
    mods = nn.ModuleDict({str(i): BottleneckResidualAdapter(hidden, bottleneck, dropout) for i in indices})
    mods.load_state_dict(sd, strict=True)
    mods.eval()
    for p in mods.parameters():
        p.requires_grad = False
    return mods


@torch.no_grad()
def extract_window_feature(
    backbone: nn.Module,
    adapters: nn.ModuleDict | None,
    window_rgb: np.ndarray,
    layer: int,
    img_size: int,
    patch_size: int,
    num_register: int,
    device: torch.device,
) -> torch.Tensor:
    x = np.asarray(Image.fromarray(window_rgb), dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0).to(device)
    out = backbone(pixel_values=t, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[layer]
    if adapters is not None and str(layer) in adapters:
        h = adapters[str(layer)](h.float())
    gh = gw = img_size // patch_size
    n_patch = gh * gw
    n_skip = 1 + num_register
    seq = h.shape[1]
    if seq >= n_skip + n_patch:
        patches = h[:, n_skip : n_skip + n_patch, :]
    elif seq == n_patch:
        patches = h
    else:
        patches = h[:, -n_patch:, :]
    return patches.reshape(gh, gw, -1).permute(2, 0, 1).contiguous().float().cpu().half()


def cache_path(cache_dir: Path, stem: str, layer: int) -> Path:
    return cache_dir / f"L{layer}" / f"{stem}.pt"


def build_image_cache(
    backbone, adapters, sample, layer, args, patch_size, num_register, device,
    crack_labels, ignore_labels, component_labels,
) -> dict:
    ann = Path(sample["json"])
    img_path = Path(sample["image"])
    data = read_labelme(ann)
    with Image.open(img_path) as im:
        img = im.convert("RGB")
        w, h = img.size
        image_rgb = np.array(img, dtype=np.uint8)

    crack, ignore, component, _ = rasterize_masks(
        data, (w, h), crack_labels, ignore_labels, component_labels, treat_empty_component_as_full_image=False
    )
    ignore_eff = ignore | (~component)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[crack & ~ignore_eff] = 1
    mask[ignore_eff] = IGNORE_INDEX
    valid = component & (~ignore_eff)

    gh = gw = args.img_size // patch_size
    coords, crops, feats, masks_down = [], [], [], []
    for y0, x0 in sliding_window_toplefts(h, w, args.img_size, args.window_stride):
        ch = min(args.img_size, h - y0)
        cw = min(args.img_size, w - x0)
        win_rgb = crop_top_left_rgb(image_rgb, y0, x0, args.img_size, args.img_size)
        win_mask = np.full((args.img_size, args.img_size), IGNORE_INDEX, dtype=np.uint8)
        win_mask[:ch, :cw] = mask[y0 : y0 + ch, x0 : x0 + cw]
        feats.append(extract_window_feature(backbone, adapters, win_rgb, layer, args.img_size, patch_size, num_register, device))
        masks_down.append(
            torch.from_numpy(np.asarray(Image.fromarray(win_mask).resize((gw, gh), Image.NEAREST)).astype(np.int64))
        )
        coords.append((y0, x0))
        crops.append((ch, cw))

    return {
        "stem": sample["stem"],
        "feats": torch.stack(feats, 0),          # [W, C, gh, gw] fp16
        "mask_down": torch.stack(masks_down, 0),  # [W, gh, gw] int64
        "coords": torch.tensor(coords, dtype=torch.int32),
        "crops": torch.tensor(crops, dtype=torch.int32),
        "gt_full": torch.from_numpy((mask == 1).astype(np.uint8)),
        "ignore_full": torch.from_numpy((mask == IGNORE_INDEX) | (~valid)),
    }


def ensure_cache(backbone, adapters, samples, layer, args, patch_size, num_register, device, crack_labels, ignore_labels, component_labels) -> None:
    total = len(samples)
    for i, s in enumerate(samples, start=1):
        cp = cache_path(args.cache_dir, s["stem"], layer)
        if cp.is_file() and not args.overwrite_cache:
            continue
        cp.parent.mkdir(parents=True, exist_ok=True)
        d = build_image_cache(backbone, adapters, s, layer, args, patch_size, num_register, device, crack_labels, ignore_labels, component_labels)
        torch.save(d, cp)
        if i % 20 == 0 or i == total:
            print(f"[frozen-probe] cached {i}/{total} ({'adapted' if adapters is not None else 'raw'})", flush=True)


def crack_iou(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray) -> float:
    valid = ~ignore
    p = (pred == 1) & valid
    g = (gt == 1) & valid
    union = int((p | g).sum())
    return int((p & g).sum()) / union if union > 0 else 0.0


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    images_dir = (args.images_dir or args.labelme_dir).expanduser().resolve()
    labelme_dir = args.labelme_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir = (args.cache_dir or (out_dir / "cache")).expanduser().resolve()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.fold_split is not None:
        split = load_folds(args.fold_split.expanduser().resolve())
        train_s, _val, test_s = fold_train_val_test(split, args.fold, args.val_ratio, args.seed)
    else:
        samples = discover_samples(labelme_dir, images_dir)
        rng = random.Random(args.seed)
        rng.shuffle(samples)
        n_val = max(1, int(len(samples) * 0.2))
        test_s, train_s = samples[:n_val], samples[n_val:]
    if args.max_train_images > 0:
        train_s = train_s[: args.max_train_images]
    if args.max_test_images > 0:
        test_s = test_s[: args.max_test_images]

    backbone = _load_vit_backbone(args.teacher_weights, pretrained=True, device=device).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    patch_size = int(getattr(backbone.config, "patch_size", 16))
    num_register = int(getattr(backbone.config, "num_register_tokens", 0))

    adapters = None
    if args.stage_a_ckpt is not None:
        adapters = build_adapters(
            args.stage_a_ckpt.expanduser().resolve(), args.adapter_state_key, args.adapter_bottleneck, args.adapter_dropout
        ).to(device)

    crack_labels = parse_csv_set(args.crack_labels)
    ignore_labels = parse_csv_set(args.ignore_labels)
    component_labels = parse_csv_set(args.component_labels)

    print(f"[frozen-probe] mode={'adapted' if adapters is not None else 'raw'} layer={args.feature_layer} "
          f"window={args.img_size} stride={args.window_stride} train={len(train_s)} test={len(test_s)}", flush=True)
    ensure_cache(backbone, adapters, train_s, args.feature_layer, args, patch_size, num_register, device, crack_labels, ignore_labels, component_labels)
    ensure_cache(backbone, adapters, test_s, args.feature_layer, args, patch_size, num_register, device, crack_labels, ignore_labels, component_labels)

    # infer channel count from first cached sample
    probe0 = torch.load(cache_path(args.cache_dir, train_s[0]["stem"], args.feature_layer), map_location="cpu", weights_only=False)
    in_ch = probe0["feats"].shape[1]
    del probe0

    head = nn.Conv2d(in_ch, 2, kernel_size=1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def seg_loss(logits, target, eps=1e-6):
        valid = target != IGNORE_INDEX
        if int(valid.sum()) == 0:
            return logits.sum() * 0.0
        ce = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX, reduction="none")
        ce = (ce * valid).sum() / valid.sum()
        prob = torch.softmax(logits.float(), dim=1)
        p = prob[:, 1][valid]
        t = (target[valid] == 1).float()
        dice = 1.0 - (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
        return ce + dice

    for epoch in range(1, args.epochs + 1):
        head.train()
        order = list(range(len(train_s)))
        random.shuffle(order)
        seen, tot, nsteps = 0, 0.0, 0
        for j in order:
            d = torch.load(cache_path(args.cache_dir, train_s[j]["stem"], args.feature_layer), map_location="cpu", weights_only=False)
            feats, masks = d["feats"], d["mask_down"]
            n = feats.shape[0]
            perm = torch.randperm(n)
            for i in range(0, n, args.batch_size):
                sel = perm[i : i + args.batch_size]
                x = feats[sel].float().to(device)
                y = masks[sel].long().to(device)
                if int((y != IGNORE_INDEX).sum()) == 0:
                    continue
                opt.zero_grad(set_to_none=True)
                logits = head(x)
                loss = seg_loss(logits, y)
                loss.backward()
                opt.step()
                tot += float(loss.detach())
                nsteps += 1
                seen += len(sel)
            if args.windows_per_epoch > 0 and seen >= args.windows_per_epoch:
                break
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[frozen-probe] epoch {epoch}/{args.epochs} windows={seen} loss={tot/max(1,nsteps):.4f}", flush=True)

    head.eval()
    ious = []
    with torch.no_grad():
        for j, s in enumerate(test_s):
            d = torch.load(cache_path(args.cache_dir, s["stem"], args.feature_layer), map_location="cpu", weights_only=False)
            feats, coords, crops = d["feats"], d["coords"], d["crops"]
            gt, ign = d["gt_full"].numpy(), d["ignore_full"].numpy()
            H, W = gt.shape
            prob_sum = np.zeros((2, H, W), dtype=np.float32)
            cnt = np.zeros((H, W), dtype=np.float32)
            for k in range(feats.shape[0]):
                y0, x0 = int(coords[k, 0]), int(coords[k, 1])
                ch, cw = int(crops[k, 0]), int(crops[k, 1])
                logits = head(feats[k].float().unsqueeze(0).to(device))
                up = F.interpolate(logits, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False)
                prob = torch.softmax(up, dim=1)[0, :, :ch, :cw].cpu().numpy()
                prob_sum[:, y0 : y0 + ch, x0 : x0 + cw] += prob
                cnt[y0 : y0 + ch, x0 : x0 + cw] += 1.0
            pred = np.argmax(prob_sum / np.maximum(cnt[None], 1.0), axis=0).astype(np.uint8)
            ious.append(crack_iou(pred, gt, ign))
    iou_mean = float(np.mean(ious)) if ious else 0.0

    result = {
        "config": vars(args),
        "mode": "adapted" if adapters is not None else "raw",
        "feature_layer": args.feature_layer,
        "in_channels": in_ch,
        "num_train_images": len(train_s),
        "num_test_images": len(test_s),
        "test_iou_mean": iou_mean,
        "test_iou_per_image": ious,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[frozen-probe] DONE mode={'adapted' if adapters is not None else 'raw'} L{args.feature_layer} test_iou={iou_mean:.4f} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
