#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate crack segmentation on a LabelMe test set.

Valid region V = inside component AND not ignore (same as Stage-C training).
Metrics on V for crack class: IoU, F1, Precision, Recall.

Supports:
  - Stage-B teacher (DINOv3StageBUNet checkpoint)
  - Stage-C semantic students (YoloUNetSemanticStudent checkpoints)

Example (repo root):

  python solution/scripts/eval_labelme_crack_test.py ^
    --labelme-dir tmp/labelme_datasets/test ^
    --output-dir runs/eval_manual_test ^
    --stage-b-ckpt solution/runs/stage_b_teacher_crack_cp/best.pt ^
    --teacher-weights solution/weights/dinov3-vitb16-pretrain-lvd1689m ^
    --student-ckpt S0=solution/runs/stage_c_semantic_yolo_unet_attn/S0/best.pt ^
    --student-ckpt S4=solution/runs/stage_c_semantic_yolo_unet_attn/S4/best.pt ^
    --student-ckpt S5=solution/runs/stage_c_semantic_yolo_unet_attn/S5/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from checkpoint_io import torch_load_compat
from models.dino_stage_b_unet import DINOv3StageBUNet
from models.yolo_unet_semseg import YoloUNetSemanticStudent
from scripts.labelme_crack_copy_paste import parse_csv_set, rasterize_masks, read_labelme
from train_seg_stage_c_mixed import (  # noqa: E402
    effective_ignore,
    load_labelme_image,
    resolve_labelme_image,
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CRACK_CLASS = 1


@dataclass
class CrackMetrics:
    tp: int
    fp: int
    fn: int
    valid_pixels: int

    @property
    def iou(self) -> float:
        return float(self.tp) / float(self.tp + self.fp + self.fn + 1e-6)

    @property
    def precision(self) -> float:
        return float(self.tp) / float(self.tp + self.fp + 1e-6)

    @property
    def recall(self) -> float:
        return float(self.tp) / float(self.tp + self.fn + 1e-6)

    @property
    def f1(self) -> float:
        return 2.0 * self.precision * self.recall / (self.precision + self.recall + 1e-6)

    def as_dict(self) -> dict[str, float]:
        return {
            "iou": self.iou,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "tp": float(self.tp),
            "fp": float(self.fp),
            "fn": float(self.fn),
            "valid_pixels": float(self.valid_pixels),
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate crack seg models on LabelMe test set.")
    p.add_argument("--labelme-dir", type=Path, default=Path("tmp/labelme_datasets/test"))
    p.add_argument("--images-dir", type=Path, default=None, help="Default: same as --labelme-dir")
    p.add_argument("--output-dir", type=Path, default=Path("runs/eval_labelme_test"))
    p.add_argument("--stage-b-ckpt", type=Path, default=None)
    p.add_argument("--teacher-weights", type=str, default="solution/weights/dinov3-vitb16-pretrain-lvd1689m")
    p.add_argument(
        "--student-ckpt",
        action="append",
        default=[],
        help="NAME=PATH to student best.pt; repeatable (e.g. S0=.../S0/best.pt)",
    )
    p.add_argument("--crack-labels", type=str, default="crack")
    p.add_argument("--ignore-labels", type=str, default="ignore")
    p.add_argument("--component-labels", type=str, default="component,wood")
    p.add_argument("--imgsz", type=int, default=0, help="Window size; 0 uses checkpoint/config default 1024")
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-pred-masks", action="store_true")
    return p.parse_args()


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_name_path(text: str) -> tuple[str, Path]:
    if "=" not in text:
        path = Path(text)
        return path.parent.name, path
    name, raw = text.split("=", 1)
    return name.strip(), Path(raw.strip())


def parse_decoder_channels(text: str) -> list[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) != 4:
        raise ValueError(f"decoder_channels expects 4 ints, got {text!r}")
    return vals


def sliding_starts(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def crop_bottom_right_pad(image: np.ndarray, x: int, y: int, size: int) -> tuple[np.ndarray, int, int]:
    h, w = image.shape[:2]
    crop_w = min(size, w - x)
    crop_h = min(size, h - y)
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    patch[:crop_h, :crop_w] = image[y : y + crop_h, x : x + crop_w]
    return patch, crop_w, crop_h


def crop_center_pad(image: np.ndarray, x: int, y: int, size: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = image.shape[:2]
    x2 = min(x + size, w)
    y2 = min(y + size, h)
    crop = image[y:y2, x:x2]
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - crop.shape[0]) // 2
    left = (size - crop.shape[1]) // 2
    patch[top : top + crop.shape[0], left : left + crop.shape[1]] = crop
    return patch, (top, left, crop.shape[0], crop.shape[1])


def image_to_tensor_imagenet(image_np: np.ndarray) -> torch.Tensor:
    x = image_np.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(np.transpose(x, (2, 0, 1)))


def build_gt_masks(
    data: dict[str, Any],
    size: tuple[int, int],
    *,
    crack_labels: set[str],
    ignore_labels: set[str],
    component_labels: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    w, h = size
    crack, ignore, component, _other = rasterize_masks(
        data,
        (w, h),
        crack_labels,
        ignore_labels,
        component_labels,
        treat_empty_component_as_full_image=False,
    )
    ignore_eff = effective_ignore(ignore, component)
    gt = np.zeros((h, w), dtype=np.uint8)
    gt[crack] = CRACK_CLASS
    gt[ignore_eff] = 255
    valid = component & (~ignore_eff)
    return gt, valid


def compute_crack_metrics(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> CrackMetrics:
    if not bool(valid.any()):
        return CrackMetrics(0, 0, 0, 0)
    gt_c = (gt == CRACK_CLASS) & valid
    pred_c = (pred == CRACK_CLASS) & valid
    tp = int(np.logical_and(gt_c, pred_c).sum())
    fp = int(np.logical_and(~gt_c, pred_c).sum())
    fn = int(np.logical_and(gt_c, ~pred_c).sum())
    return CrackMetrics(tp=tp, fp=fp, fn=fn, valid_pixels=int(valid.sum()))


def aggregate_micro(metrics_list: list[CrackMetrics]) -> CrackMetrics:
    return CrackMetrics(
        tp=sum(m.tp for m in metrics_list),
        fp=sum(m.fp for m in metrics_list),
        fn=sum(m.fn for m in metrics_list),
        valid_pixels=sum(m.valid_pixels for m in metrics_list),
    )


def aggregate_macro(metrics_list: list[CrackMetrics]) -> dict[str, float]:
    if not metrics_list:
        return {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    vals = [m.as_dict() for m in metrics_list if m.valid_pixels > 0]
    if not vals:
        return {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    return {
        "iou": float(np.mean([v["iou"] for v in vals])),
        "f1": float(np.mean([v["f1"] for v in vals])),
        "precision": float(np.mean([v["precision"] for v in vals])),
        "recall": float(np.mean([v["recall"] for v in vals])),
    }


def load_stage_b(ckpt_path: Path, teacher_weights: str, device: torch.device) -> tuple[DINOv3StageBUNet, int]:
    cfg_path = ckpt_path.parent / "config.json"
    cfg = read_json(cfg_path) if cfg_path.is_file() else {}
    tw = teacher_weights.strip() or str(cfg.get("teacher_weights", ""))
    if not tw:
        raise ValueError("Set --teacher-weights or ensure Stage-B config.json has teacher_weights")
    model = DINOv3StageBUNet(
        weights_dir=tw,
        pretrained=True,
        device=device,
        num_classes=int(cfg.get("num_classes", 2)),
        bottleneck_dim=int(cfg.get("adapter_bottleneck", 64)),
        adapter_dropout=float(cfg.get("adapter_dropout", 0.1)),
    ).to(device)
    ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    imgsz = int(cfg.get("imgsz", 1024))
    return model, imgsz


def load_student(ckpt_path: Path, device: torch.device) -> tuple[YoloUNetSemanticStudent, int, int]:
    ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args: dict[str, Any] = ckpt.get("args") or {}
    num_classes = int(ckpt_args.get("num_classes", 2))
    imgsz = int(ckpt_args.get("imgsz", 1024))
    student_weights = str(ckpt_args.get("student_weights", "solution/yolo11m-seg.pt"))
    decoder_channels = parse_decoder_channels(str(ckpt_args.get("decoder_channels", "256,192,128,64")))
    student = YoloUNetSemanticStudent(
        student_weights,
        num_classes=num_classes,
        device=device,
        decoder_channels=decoder_channels,
    ).to(device)
    student.load_state_dict(ckpt["student"], strict=True)
    student.eval()
    return student, imgsz, num_classes


@torch.no_grad()
def predict_teacher(
    model: DINOv3StageBUNet,
    image_rgb: np.ndarray,
    *,
    imgsz: int,
    stride: int,
    device: torch.device,
) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    num_classes = model.num_classes
    acc = np.zeros((num_classes, h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    use_amp = device.type == "cuda"
    for y in sliding_starts(h, imgsz, stride):
        for x in sliding_starts(w, imgsz, stride):
            patch, (top, left, crop_h, crop_w) = crop_center_pad(image_rgb, x, y, imgsz)
            t = image_to_tensor_imagenet(patch).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(t)
            logits_np = logits[0, :, top : top + crop_h, left : left + crop_w].float().cpu().numpy()
            acc[:, y : y + crop_h, x : x + crop_w] += logits_np
            cnt[y : y + crop_h, x : x + crop_w] += 1.0
    acc /= np.maximum(cnt[None, :, :], 1e-6)
    pred = np.argmax(acc, axis=0).astype(np.uint8)
    if num_classes > 2:
        pred = (pred == CRACK_CLASS).astype(np.uint8)
    return pred


@torch.no_grad()
def predict_student(
    model: YoloUNetSemanticStudent,
    image_rgb: np.ndarray,
    *,
    imgsz: int,
    stride: int,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    prob_sum = np.zeros((num_classes, h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    for y in sliding_starts(h, imgsz, stride):
        for x in sliding_starts(w, imgsz, stride):
            patch, crop_w, crop_h = crop_bottom_right_pad(image_rgb, x, y, imgsz)
            arr = patch.astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
            logits, _ = model(tensor)
            prob = F.softmax(logits.float(), dim=1)[0, :, :crop_h, :crop_w].cpu().numpy()
            prob_sum[:, y : y + crop_h, x : x + crop_w] += prob
            count[y : y + crop_h, x : x + crop_w] += 1.0
    prob_sum /= np.maximum(count[None, :, :], 1.0)
    return np.argmax(prob_sum, axis=0).astype(np.uint8)


def list_test_samples(labelme_dir: Path) -> list[Path]:
    return sorted(labelme_dir.glob("*.json"))


def evaluate_model(
    model_name: str,
    model_kind: Literal["teacher", "student"],
    model_obj: Any,
    *,
    imgsz: int,
    stride: int,
    num_classes: int,
    device: torch.device,
    samples: list[dict[str, Any]],
    output_dir: Optional[Path],
    save_pred_masks: bool,
) -> dict[str, Any]:
    per_image: list[dict[str, Any]] = []
    metric_objs: list[CrackMetrics] = []

    pred_dir = None
    if save_pred_masks and output_dir is not None:
        pred_dir = output_dir / "pred_masks" / model_name
        pred_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        image_rgb = sample["image_rgb"]
        gt = sample["gt"]
        valid = sample["valid"]
        stem = sample["stem"]

        if model_kind == "teacher":
            pred = predict_teacher(model_obj, image_rgb, imgsz=imgsz, stride=stride, device=device)
        else:
            pred = predict_student(
                model_obj,
                image_rgb,
                imgsz=imgsz,
                stride=stride,
                num_classes=num_classes,
                device=device,
            )

        m = compute_crack_metrics(gt, pred, valid)
        metric_objs.append(m)
        per_image.append({"stem": stem, **m.as_dict()})

        if pred_dir is not None:
            Image.fromarray(pred, mode="L").save(pred_dir / f"{stem}_pred.png")

    micro = aggregate_micro(metric_objs)
    macro = aggregate_macro(metric_objs)
    return {
        "model": model_name,
        "kind": model_kind,
        "num_images": len(samples),
        "micro": micro.as_dict(),
        "macro_mean": macro,
        "per_image": per_image,
    }


def main() -> int:
    args = parse_args()
    labelme_dir = args.labelme_dir.expanduser().resolve()
    images_dir = (args.images_dir or labelme_dir).expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    crack_labels = parse_csv_set(args.crack_labels)
    ignore_labels = parse_csv_set(args.ignore_labels)
    component_labels = parse_csv_set(args.component_labels)

    ann_paths = list_test_samples(labelme_dir)
    if not ann_paths:
        raise RuntimeError(f"No LabelMe JSON under {labelme_dir}")

    samples: list[dict[str, Any]] = []
    for ann_path in ann_paths:
        data = read_labelme(ann_path)
        img_path = resolve_labelme_image(images_dir, ann_path, data)
        if img_path is None:
            print(f"[warn] skip {ann_path.name}: image not found", flush=True)
            continue
        image = load_labelme_image(data, img_path)
        w, h = image.size
        gt, valid = build_gt_masks(
            data,
            (w, h),
            crack_labels=crack_labels,
            ignore_labels=ignore_labels,
            component_labels=component_labels,
        )
        if not bool(valid.any()):
            print(f"[warn] skip {ann_path.name}: empty valid region (no component?)", flush=True)
            continue
        samples.append(
            {
                "stem": ann_path.stem,
                "ann_path": str(ann_path),
                "image_path": str(img_path),
                "image_rgb": np.asarray(image, dtype=np.uint8),
                "gt": gt,
                "valid": valid,
            }
        )

    if not samples:
        raise RuntimeError("No evaluable test samples with component + image.")

    student_ckpts: list[tuple[str, Path]] = []
    for item in args.student_ckpt:
        student_ckpts.append(parse_name_path(item))

    if not student_ckpts and not args.stage_b_ckpt:
        student_ckpts = [
            ("S0", Path("solution/runs/stage_c_semantic_yolo_unet_attn/S0/best.pt")),
            ("S4", Path("solution/runs/stage_c_semantic_yolo_unet_attn/S4/best.pt")),
            ("S5", Path("solution/runs/stage_c_semantic_yolo_unet_attn/S5/best.pt")),
        ]
        args.stage_b_ckpt = Path("solution/runs/stage_b_teacher_crack_cp/best.pt")

    stride = int(args.stride)
    results: list[dict[str, Any]] = []

    if args.stage_b_ckpt is not None:
        ckpt = args.stage_b_ckpt.expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Stage-B checkpoint not found: {ckpt}")
        teacher, teacher_imgsz = load_stage_b(ckpt, args.teacher_weights, device)
        imgsz = int(args.imgsz or teacher_imgsz)
        print(f"[eval] Stage-B teacher imgsz={imgsz} stride={stride} n={len(samples)}", flush=True)
        results.append(
            evaluate_model(
                "stage_b",
                "teacher",
                teacher,
                imgsz=imgsz,
                stride=stride,
                num_classes=teacher.num_classes,
                device=device,
                samples=samples,
                output_dir=out_dir,
                save_pred_masks=args.save_pred_masks,
            )
        )

    for name, ckpt_path in student_ckpts:
        ckpt = ckpt_path.expanduser().resolve()
        if not ckpt.is_file():
            print(f"[warn] skip {name}: checkpoint not found {ckpt}", flush=True)
            continue
        student, student_imgsz, num_classes = load_student(ckpt, device)
        imgsz = int(args.imgsz or student_imgsz)
        print(f"[eval] {name} imgsz={imgsz} stride={stride} n={len(samples)}", flush=True)
        results.append(
            evaluate_model(
                name,
                "student",
                student,
                imgsz=imgsz,
                stride=stride,
                num_classes=num_classes,
                device=device,
                samples=samples,
                output_dir=out_dir,
                save_pred_masks=args.save_pred_masks,
            )
        )

    if not results:
        raise RuntimeError("No models evaluated. Check checkpoint paths.")

    summary = {
        "labelme_dir": str(labelme_dir),
        "images_dir": str(images_dir),
        "num_samples": len(samples),
        "stride": stride,
        "valid_region": "component AND NOT ignore",
        "metrics": ["iou", "f1", "precision", "recall"],
        "models": results,
    }
    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out_dir / "metrics_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "agg", "iou", "f1", "precision", "recall", "tp", "fp", "fn", "valid_pixels"])
        for r in results:
            for agg_name, block in ("micro", r["micro"]), ("macro_mean", r["macro_mean"]):
                writer.writerow(
                    [
                        r["model"],
                        agg_name,
                        f"{block['iou']:.6f}",
                        f"{block['f1']:.6f}",
                        f"{block['precision']:.6f}",
                        f"{block['recall']:.6f}",
                        block.get("tp", ""),
                        block.get("fp", ""),
                        block.get("fn", ""),
                        block.get("valid_pixels", ""),
                    ]
                )

    print("\n=== Test set crack metrics (valid = component \\ ignore) ===", flush=True)
    print(f"{'model':<10} {'agg':<10} {'IoU':>8} {'F1':>8} {'Prec':>8} {'Recall':>8}", flush=True)
    for r in results:
        micro = r["micro"]
        macro = r["macro_mean"]
        print(
            f"{r['model']:<10} {'micro':<10} {micro['iou']:8.4f} {micro['f1']:8.4f} "
            f"{micro['precision']:8.4f} {micro['recall']:8.4f}",
            flush=True,
        )
        print(
            f"{r['model']:<10} {'macro':<10} {macro['iou']:8.4f} {macro['f1']:8.4f} "
            f"{macro['precision']:8.4f} {macro['recall']:8.4f}",
            flush=True,
        )

    print(f"\n[done] metrics -> {json_path}", flush=True)
    print(f"[done] csv     -> {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
