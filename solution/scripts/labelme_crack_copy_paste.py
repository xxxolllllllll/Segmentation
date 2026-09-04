#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Same-image crack Copy-Paste augmentation for one LabelMe annotation.

Input:
  - one LabelMe JSON
  - its corresponding image

Output:
  - augmented image
  - augmented LabelMe JSON, preserving original shapes and appending pasted
    cracks as new "crack" polygon shapes.

The implementation follows docs/copy&paste:
  - crack instances are extracted from the current image's annotated crack mask
  - paste positions are constrained by component/wood, ignore and existing labels
  - brightness / texture similarity are checked before pasting
  - alpha blending uses a dilated and feathered crack mask
  - brightness matching adjusts the Lab L channel of the source patch
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def parse_csv_set(text: str) -> set[str]:
    return {x.strip().lower() for x in text.split(",") if x.strip()}


def read_labelme(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="utf-8-sig"))


def shape_points(shape: dict[str, Any]) -> list[tuple[float, float]]:
    points = shape.get("points", [])
    if not isinstance(points, list):
        return []
    out: list[tuple[float, float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            out.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    return out


def draw_labelme_shape(draw: ImageDraw.ImageDraw, shape: dict[str, Any], fill: int) -> None:
    points = shape_points(shape)
    if len(points) < 2:
        return
    shape_type = str(shape.get("shape_type", "polygon")).lower()
    if shape_type == "rectangle" and len(points) >= 2:
        draw.rectangle([points[0], points[1]], fill=fill)
    elif shape_type == "circle" and len(points) >= 2:
        (x1, y1), (x2, y2) = points[0], points[1]
        radius = math.hypot(x2 - x1, y2 - y1)
        draw.ellipse([x1 - radius, y1 - radius, x1 + radius, y1 + radius], fill=fill)
    elif shape_type in {"line", "linestrip"} and len(points) >= 2:
        draw.line(points, fill=fill, width=3)
    elif len(points) >= 3:
        draw.polygon(points, fill=fill)


def rasterize_masks(
    data: dict[str, Any],
    size: tuple[int, int],
    crack_labels: set[str],
    ignore_labels: set[str],
    component_labels: set[str],
    *,
    treat_empty_component_as_full_image: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return crack, ignore, component and non-crack-label masks as bool arrays."""
    width, height = size
    crack = Image.new("L", (width, height), 0)
    ignore = Image.new("L", (width, height), 0)
    component = Image.new("L", (width, height), 0)
    other_label = Image.new("L", (width, height), 0)
    draws = {
        "crack": ImageDraw.Draw(crack),
        "ignore": ImageDraw.Draw(ignore),
        "component": ImageDraw.Draw(component),
        "other": ImageDraw.Draw(other_label),
    }

    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        shapes = []

    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label", "")).strip().lower()
        if not label:
            continue
        if label in crack_labels:
            draw_labelme_shape(draws["crack"], shape, 1)
        elif label in ignore_labels:
            draw_labelme_shape(draws["ignore"], shape, 1)
        elif label in component_labels:
            draw_labelme_shape(draws["component"], shape, 1)
        else:
            # Existing defect/knot/decay/etc. are kept as blockers for paste positions.
            draw_labelme_shape(draws["other"], shape, 1)

    component_np = np.asarray(component, dtype=bool)
    if not component_np.any() and treat_empty_component_as_full_image:
        component_np[:] = True
    return (
        np.asarray(crack, dtype=bool),
        np.asarray(ignore, dtype=bool),
        component_np,
        np.asarray(other_label, dtype=bool),
    )


def expanded_bbox_from_mask(mask: np.ndarray, padding: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return 0, 0, 0, 0
    height, width = mask.shape
    x1 = max(int(xs.min()) - padding, 0)
    y1 = max(int(ys.min()) - padding, 0)
    x2 = min(int(xs.max()) + 1 + padding, width)
    y2 = min(int(ys.max()) + 1 + padding, height)
    return x1, y1, x2, y2


def extract_instances(
    image: np.ndarray,
    crack_mask: np.ndarray,
    min_area: int,
    padding: int,
) -> list[dict[str, Any]]:
    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        crack_mask.astype(np.uint8), connectivity=8
    )
    instances: list[dict[str, Any]] = []
    for idx in range(1, n_labels):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == idx
        x1, y1, x2, y2 = expanded_bbox_from_mask(component, padding)
        if x2 <= x1 or y2 <= y1:
            continue
        inst_mask = component[y1:y2, x1:x2].astype(np.uint8)
        instances.append(
            {
                "patch": image[y1:y2, x1:x2].copy(),
                "mask": inst_mask,
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                "area": area,
            }
        )
    return instances


def transform_instance(
    patch: np.ndarray,
    mask: np.ndarray,
    rng: random.Random,
    max_rotate_deg: float,
    scale_min: float,
    scale_max: float,
    padding: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_rotate_deg <= 0 and abs(scale_min - 1.0) < 1e-6 and abs(scale_max - 1.0) < 1e-6:
        return patch, mask

    height, width = mask.shape
    angle = rng.uniform(-max_rotate_deg, max_rotate_deg)
    scale = rng.uniform(scale_min, scale_max)
    center = (width / 2.0, height / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle, scale)
    warped_patch = cv2.warpAffine(
        patch,
        mat,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    warped_mask = cv2.warpAffine(
        mask.astype(np.uint8),
        mat,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if not (warped_mask > 0).any():
        return patch, mask
    x1, y1, x2, y2 = expanded_bbox_from_mask(warped_mask > 0, padding)
    return warped_patch[y1:y2, x1:x2].copy(), (warped_mask[y1:y2, x1:x2] > 0).astype(np.uint8)


def context_mask_from_crack(mask: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return (dilated > 0) & (mask == 0)


def estimate_texture_angle(gray: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) < 20:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gxv = gx[mask]
    gyv = gy[mask]
    if gxv.size < 20:
        return 0.0
    jxx = float(np.mean(gxv * gxv))
    jyy = float(np.mean(gyv * gyv))
    jxy = float(np.mean(gxv * gyv))
    return float(np.rad2deg(0.5 * np.arctan2(2.0 * jxy, jxx - jyy + 1e-8)))


def angle_diff(a: float, b: float) -> float:
    diff = abs(a - b) % 180.0
    return min(diff, 180.0 - diff)


def candidate_score(
    src_patch: np.ndarray,
    src_mask: np.ndarray,
    dst_patch: np.ndarray,
    brightness_mean_threshold: float,
    brightness_std_threshold: float,
    texture_angle_threshold: float,
) -> float | None:
    context = context_mask_from_crack(src_mask)
    if int(context.sum()) < 20:
        return 0.0

    src_gray = cv2.cvtColor(src_patch, cv2.COLOR_RGB2GRAY)
    dst_gray = cv2.cvtColor(dst_patch, cv2.COLOR_RGB2GRAY)
    mean_diff = float(abs(src_gray[context].mean() - dst_gray[context].mean()))
    std_diff = float(abs(src_gray[context].std() - dst_gray[context].std()))
    texture_diff = angle_diff(
        estimate_texture_angle(src_gray, context),
        estimate_texture_angle(dst_gray, context),
    )
    if mean_diff > brightness_mean_threshold:
        return None
    if std_diff > brightness_std_threshold:
        return None
    if texture_diff > texture_angle_threshold:
        return None
    return mean_diff + 0.5 * std_diff + 2.0 * texture_diff


def find_paste_location(
    image: np.ndarray,
    src_patch: np.ndarray,
    src_mask: np.ndarray,
    source_center: tuple[int, int],
    crack_mask: np.ndarray,
    ignore_mask: np.ndarray,
    component_mask: np.ndarray,
    other_label_mask: np.ndarray,
    rng: random.Random,
    max_tries: int,
    search_radius: int,
    inside_component_threshold: float,
    max_crack_overlap: float,
    max_other_overlap: float,
    brightness_mean_threshold: float,
    brightness_std_threshold: float,
    texture_angle_threshold: float,
) -> tuple[int, int] | None:
    height, width = image.shape[:2]
    ph, pw = src_mask.shape
    paste_pixels = src_mask > 0
    if ph <= 0 or pw <= 0 or ph > height or pw > width or not paste_pixels.any():
        return None

    best: tuple[float, int, int] | None = None
    cx0, cy0 = source_center
    for _ in range(max_tries):
        cx = cx0 + rng.randint(-search_radius, search_radius)
        cy = cy0 + rng.randint(-search_radius, search_radius)
        x = int(np.clip(cx - pw // 2, 0, width - pw))
        y = int(np.clip(cy - ph // 2, 0, height - ph))

        dst_component = component_mask[y : y + ph, x : x + pw]
        if float(dst_component[paste_pixels].mean()) < inside_component_threshold:
            continue

        dst_ignore = ignore_mask[y : y + ph, x : x + pw]
        if bool(dst_ignore[paste_pixels].any()):
            continue

        dst_crack = crack_mask[y : y + ph, x : x + pw]
        if float(dst_crack[paste_pixels].mean()) > max_crack_overlap:
            continue

        dst_other = other_label_mask[y : y + ph, x : x + pw]
        if float(dst_other[paste_pixels].mean()) > max_other_overlap:
            continue

        dst_patch = image[y : y + ph, x : x + pw]
        score = candidate_score(
            src_patch,
            src_mask,
            dst_patch,
            brightness_mean_threshold,
            brightness_std_threshold,
            texture_angle_threshold,
        )
        if score is None:
            continue
        if best is None or score < best[0]:
            best = (score, x, y)
    if best is None:
        return None
    return best[1], best[2]


def match_brightness_lab(src_patch: np.ndarray, dst_patch: np.ndarray, src_mask: np.ndarray) -> np.ndarray:
    context = context_mask_from_crack(src_mask)
    if int(context.sum()) < 20:
        return src_patch
    src_lab = cv2.cvtColor(src_patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    dst_lab = cv2.cvtColor(dst_patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    delta = float(dst_lab[:, :, 0][context].mean() - src_lab[:, :, 0][context].mean())
    src_lab[:, :, 0] = np.clip(src_lab[:, :, 0] + delta, 0, 255)
    return cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


def alpha_blend_paste(
    image: np.ndarray,
    src_patch: np.ndarray,
    src_mask: np.ndarray,
    x: int,
    y: int,
    alpha_dilate: int,
    alpha_blur: int,
) -> None:
    ph, pw = src_mask.shape
    dst_patch = image[y : y + ph, x : x + pw].copy()
    adjusted = match_brightness_lab(src_patch, dst_patch, src_mask)

    kernel = np.ones((alpha_dilate, alpha_dilate), np.uint8)
    dilated = cv2.dilate(src_mask.astype(np.uint8), kernel, iterations=1)
    blur = alpha_blur if alpha_blur % 2 == 1 else alpha_blur + 1
    alpha = cv2.GaussianBlur(dilated.astype(np.float32), (blur, blur), 0)
    alpha = alpha / max(float(alpha.max()), 1e-6)
    alpha = alpha[:, :, None]

    blended = adjusted.astype(np.float32) * alpha + dst_patch.astype(np.float32) * (1.0 - alpha)
    image[y : y + ph, x : x + pw] = np.clip(blended, 0, 255).astype(np.uint8)


def pasted_mask_to_shapes(
    src_mask: np.ndarray,
    x: int,
    y: int,
    label: str,
    min_contour_area: float,
    approx_epsilon: float,
) -> list[dict[str, Any]]:
    contours, _hierarchy = cv2.findContours(src_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes: list[dict[str, Any]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue
        approx = cv2.approxPolyDP(contour, epsilon=approx_epsilon, closed=True)
        points = approx.reshape(-1, 2)
        if len(points) < 3:
            continue
        shifted = [[float(px + x), float(py + y)] for px, py in points]
        shapes.append(
            {
                "label": label,
                "points": shifted,
                "group_id": None,
                "description": "copy_paste",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }
        )
    return shapes


def apply_copy_paste(args: argparse.Namespace) -> dict[str, Any]:
    labelme_path = args.labelme_json.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    output_image = args.output_image.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    data = read_labelme(labelme_path)
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    height, width = image.shape[:2]

    crack_labels = parse_csv_set(args.crack_labels)
    ignore_labels = parse_csv_set(args.ignore_labels)
    component_labels = parse_csv_set(args.component_labels)
    crack_mask, ignore_mask, component_mask, other_label_mask = rasterize_masks(
        data,
        (width, height),
        crack_labels,
        ignore_labels,
        component_labels,
    )
    instances = extract_instances(image, crack_mask, args.min_crack_area, args.bbox_padding)
    if not instances:
        raise RuntimeError(f"No crack instances found in {labelme_path}")

    rng = random.Random(args.seed)
    out_image_np = image.copy()
    out_crack_mask = crack_mask.copy()
    out_shapes: list[dict[str, Any]] = []
    attempts = max(args.num_pastes, 1) * args.instance_attempt_multiplier

    for _ in range(attempts):
        if len(out_shapes) >= args.num_pastes:
            break
        instance = instances[rng.randrange(len(instances))]
        src_patch, src_mask = transform_instance(
            instance["patch"],
            instance["mask"],
            rng,
            args.max_rotate_deg,
            args.scale_min,
            args.scale_max,
            args.bbox_padding,
        )
        if int(src_mask.sum()) < args.min_crack_area:
            continue
        location = find_paste_location(
            out_image_np,
            src_patch,
            src_mask,
            instance["center"],
            out_crack_mask,
            ignore_mask,
            component_mask,
            other_label_mask,
            rng,
            args.max_tries_per_instance,
            args.search_radius,
            args.inside_component_threshold,
            args.max_crack_overlap,
            args.max_other_overlap,
            args.brightness_mean_threshold,
            args.brightness_std_threshold,
            args.texture_angle_threshold,
        )
        if location is None:
            continue
        x, y = location
        alpha_blend_paste(out_image_np, src_patch, src_mask, x, y, args.alpha_dilate, args.alpha_blur)
        ph, pw = src_mask.shape
        out_crack_mask[y : y + ph, x : x + pw] |= src_mask.astype(bool)
        out_shapes.extend(
            pasted_mask_to_shapes(
                src_mask,
                x,
                y,
                args.output_crack_label,
                args.min_contour_area,
                args.approx_epsilon,
            )
        )

    out_data = copy.deepcopy(data)
    out_data["imagePath"] = output_image.name
    out_data["imageData"] = None
    out_data["imageHeight"] = int(height)
    out_data["imageWidth"] = int(width)
    shapes = out_data.get("shapes")
    if not isinstance(shapes, list):
        shapes = []
    out_data["shapes"] = shapes + out_shapes

    Image.fromarray(out_image_np).save(output_image)
    output_json.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "input_json": str(labelme_path),
        "input_image": str(image_path),
        "output_json": str(output_json),
        "output_image": str(output_image),
        "num_source_instances": len(instances),
        "num_pasted_shapes": len(out_shapes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply same-image crack Copy-Paste to one LabelMe sample.")
    parser.add_argument("--labelme-json", type=Path, required=True, help="Input LabelMe JSON")
    parser.add_argument("--image", type=Path, required=True, help="Input image corresponding to --labelme-json")
    parser.add_argument("--output-image", type=Path, required=True, help="Augmented output image path")
    parser.add_argument("--output-json", type=Path, required=True, help="Augmented output LabelMe JSON path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-pastes", type=int, default=3)
    parser.add_argument("--instance-attempt-multiplier", type=int, default=4)
    parser.add_argument("--min-crack-area", type=int, default=30)
    parser.add_argument("--bbox-padding", type=int, default=16)
    parser.add_argument("--search-radius", type=int, default=1200)
    parser.add_argument("--max-tries-per-instance", type=int, default=150)
    parser.add_argument("--inside-component-threshold", type=float, default=0.98)
    parser.add_argument("--max-crack-overlap", type=float, default=0.05)
    parser.add_argument("--max-other-overlap", type=float, default=0.02)
    parser.add_argument("--brightness-mean-threshold", type=float, default=25.0)
    parser.add_argument("--brightness-std-threshold", type=float, default=20.0)
    parser.add_argument("--texture-angle-threshold", type=float, default=30.0)
    parser.add_argument("--max-rotate-deg", type=float, default=10.0)
    parser.add_argument("--scale-min", type=float, default=0.9)
    parser.add_argument("--scale-max", type=float, default=1.1)
    parser.add_argument("--alpha-dilate", type=int, default=3)
    parser.add_argument("--alpha-blur", type=int, default=7)
    parser.add_argument("--min-contour-area", type=float, default=2.0)
    parser.add_argument("--approx-epsilon", type=float, default=1.5)
    parser.add_argument("--crack-labels", type=str, default="crack")
    parser.add_argument("--ignore-labels", type=str, default="ignore")
    parser.add_argument("--component-labels", type=str, default="component,wood")
    parser.add_argument("--output-crack-label", type=str, default="crack")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = apply_copy_paste(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
