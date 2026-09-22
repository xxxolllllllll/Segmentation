#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-A adapter patch-drift diagnostics (read-only analysis + a small probe).

Hypothesis: Stage-A is a CLS-level DINO self-distillation, but the adapters
modify every token. So patch-token features may drift / be perturbed without any
dense objective constraining them, which could weaken dense (segmentation)
distillation.

This script quantifies and visualises:
  1. adapter perturbation on CLS / register / patch tokens per layer;
  2. per-patch raw-vs-adapted cosine similarity + perturbation heatmaps;
  3. PCA-RGB feature maps (raw vs adapted);
  4. crack-vs-background separability (Fisher ratio) per layer/source;
  5. linear CKA (raw vs adapted) per layer;
  6. a small linear/MLP probe trained on raw vs adapted patch features.
Both EMA-teacher and student adapters are analysed.

Example:
  python solution/scripts/analyze_stage_a_features.py \
    --teacher-weights weights/dinov3-vitb16-pretrain-lvd1689m \
    --stage-a-ckpt runs/stage_a/stage_a_last.pt \
    --labelme-dir data/labelme/all --images-dir data/labelme/all \
    --output-dir runs/analysis/stage_a_features \
    --num-images 50 --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from checkpoint_io import torch_load_compat  # noqa: E402
from folds import discover_samples, resolve_labelme_image  # noqa: E402
from models.dino_stage_a import BottleneckResidualAdapter  # noqa: E402
from models.teacher_vit import _load_vit_backbone  # noqa: E402
from scripts.labelme_crack_copy_paste import parse_csv_set, rasterize_masks, read_labelme  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EPS = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-A adapter patch-drift diagnostics")
    p.add_argument("--teacher-weights", type=str, required=True, help="Local HF DINOv3 dir (with config.json)")
    p.add_argument("--stage-a-ckpt", type=Path, required=True, help="Stage-A checkpoint (stage_a_last.pt)")
    p.add_argument("--labelme-dir", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, default=None, help="Default: same as --labelme-dir")
    p.add_argument("--output-dir", type=Path, default=Path("runs/analysis/stage_a_features"))
    p.add_argument("--num-images", type=int, default=50, help="Random images to analyse")
    p.add_argument("--img-size", type=int, default=512, help="Input size fed to the backbone (divisible by 16)")
    p.add_argument("--layers", type=str, default="3,4,7,8,11,12")
    p.add_argument("--max-patches-per-image", type=int, default=400, help="Subsample patches per image for probe/separability")
    p.add_argument("--fg-frac-thresh", type=float, default=0.0, help="Patch is crack if crack-area fraction > this (0=any crack pixel, foreground priority)")
    p.add_argument("--min-valid-frac", type=float, default=0.5, help="Keep patch only if its valid-area fraction >= this")
    p.add_argument("--probe-train-images", type=int, default=40, help="Images used to train the probe (rest are val)")
    p.add_argument("--probe-epochs", type=int, default=300)
    p.add_argument("--probe-hidden", type=int, default=256)
    p.add_argument("--num-viz-samples", type=int, default=6)
    p.add_argument("--viz-layers", type=str, default="3,4,7,8,11,12", help="Layers to visualise per sample")
    p.add_argument("--crack-labels", type=str, default="crack")
    p.add_argument("--ignore-labels", type=str, default="ignore")
    p.add_argument("--component-labels", type=str, default="component,wood")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_adapters(ckpt: dict, state_key: str, bottleneck: int, dropout: float) -> nn.ModuleDict:
    indices = tuple(int(i) for i in ckpt.get("adapter_indices", (3, 4, 7, 8, 11, 12)))
    hidden = int(ckpt.get("hidden_size", 768))
    sd = ckpt.get(state_key)
    if sd is None:
        raise KeyError(f"checkpoint missing {state_key}")
    mods = nn.ModuleDict({str(i): BottleneckResidualAdapter(hidden, bottleneck, dropout) for i in indices})
    mods.load_state_dict(sd, strict=True)
    return mods


def preprocess(sample: dict, size: int, images_dir: Path, crack_labels, ignore_labels, component_labels):
    ann_path = Path(sample["json"])
    data = read_labelme(ann_path)
    img_path = resolve_labelme_image(images_dir, ann_path, data)
    img = Image.open(img_path).convert("RGB") if img_path is not None else None
    if img is None:
        raise FileNotFoundError(f"no image for {ann_path}")
    w, h = img.size
    crack, ignore, component, _other = rasterize_masks(
        data, (w, h), crack_labels, ignore_labels, component_labels, treat_empty_component_as_full_image=False
    )
    ignore_eff = ignore | (~component)
    valid = component & (~ignore_eff)

    img_s = img.resize((size, size), Image.BILINEAR)
    # Area-average (BOX) + >0 threshold keeps thin cracks (foreground priority).
    crack_s = np.asarray(Image.fromarray(crack.astype(np.uint8) * 255).resize((size, size), Image.BOX)) > 0
    valid_s = np.asarray(Image.fromarray(valid.astype(np.uint8) * 255).resize((size, size), Image.BOX)) > 0

    x = np.asarray(img_s, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0)
    return x, crack_s, valid_s


def token_parts(tokens: torch.Tensor, n_register: int, g: int):
    """Return (cls[B,C], reg[B,R,C], patch_map[B,C,g,g])."""
    b, seq, c = tokens.shape
    cls = tokens[:, 0, :]
    reg = tokens[:, 1 : 1 + n_register, :] if n_register > 0 else tokens[:, :0, :]
    if seq >= 1 + n_register + g * g:
        patch = tokens[:, 1 + n_register : 1 + n_register + g * g, :]
    elif seq == g * g:
        patch = tokens
    else:
        patch = tokens[:, -g * g :, :]
    patch_map = patch.reshape(b, g, g, c).permute(0, 3, 1, 2).contiguous()
    return cls, reg, patch, patch_map


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    xty = torch.linalg.norm(x.t() @ y) ** 2
    xtx = torch.linalg.norm(x.t() @ x)
    yty = torch.linalg.norm(y.t() @ y)
    return float(xty / (xtx * yty + EPS))


def fisher_ratio(x: torch.Tensor, y: torch.Tensor) -> float:
    m1 = x[y].mean(0)
    m0 = x[~y].mean(0)
    v1 = x[y].var(0).mean()
    v0 = x[~y].var(0).mean()
    return float(((m1 - m0) ** 2).sum() / (v0 + v1 + EPS))


class Probe(nn.Module):
    def __init__(self, c: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(c, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_probe(xtr, ytr, xva, yva, hidden: int, epochs: int, device: torch.device) -> dict:
    mu = xtr.mean(0)
    sd = xtr.std(0) + 1e-5
    xtr = (xtr - mu) / sd
    xva = (xva - mu) / sd
    probe = Probe(xtr.shape[1], hidden).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = float((ytr == 1).sum())
    neg = float((ytr == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    xtr = xtr.to(device)
    ytr = ytr.to(device)
    n = xtr.shape[0]
    bs = min(4096, n)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad(set_to_none=True)
            loss = lossf(probe(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
    with torch.no_grad():
        p = probe(xva.to(device)).cpu()
    scores = p.numpy()
    yv = yva.numpy()
    thr = 0.0
    pred = (scores > thr).astype(np.float32)
    tp = float(((pred == 1) & (yv == 1)).sum())
    fp = float(((pred == 1) & (yv == 0)).sum())
    fn = float(((pred == 0) & (yv == 1)).sum())
    iou = tp / (tp + fp + fn + EPS)
    f1 = (2 * tp) / (2 * tp + fp + fn + EPS)
    acc = float((pred == yv).mean())

    # threshold-free average precision + best-threshold IoU
    order = np.argsort(-scores)
    ys = yv[order]
    tps = np.cumsum(ys)
    fps = np.cumsum(1 - ys)
    prec = tps / np.maximum(tps + fps, 1e-9)
    rec = tps / max(float(yv.sum()), 1e-9)
    ap = 0.0
    prev_r = 0.0
    for pi, ri in zip(prec, rec):
        ap += (ri - prev_r) * pi
        prev_r = ri
    best_iou = 0.0
    for t in np.unique(scores):
        pr = scores >= t
        tp2 = float((pr & (yv == 1)).sum())
        fp2 = float((pr & (yv == 0)).sum())
        fn2 = float((~pr & (yv == 1)).sum())
        best_iou = max(best_iou, tp2 / (tp2 + fp2 + fn2 + EPS))
    return {
        "probe_iou@0": iou,
        "probe_f1@0": f1,
        "probe_acc": acc,
        "probe_ap": float(ap),
        "probe_best_iou": float(best_iou),
        "n_train": int(n),
        "n_val": int(xva.shape[0]),
        "pos_rate": float(yv.mean()),
    }


def heat_rgb(values: np.ndarray) -> np.ndarray:
    v = values.astype(np.float32)
    v = (v - v.min()) / (v.max() - v.min() + EPS)
    return (v * 255).astype(np.uint8)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    if args.img_size % 16 != 0:
        raise ValueError("--img-size must be divisible by 16")
    device = torch.device(args.device)
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    viz_layers = [int(x.strip()) for x in args.viz_layers.split(",") if x.strip()]
    images_dir = (args.images_dir or args.labelme_dir).expanduser().resolve()

    # data
    samples = discover_samples(args.labelme_dir.expanduser().resolve(), images_dir)
    if len(samples) < args.num_images:
        raise RuntimeError(f"only {len(samples)} samples < --num-images {args.num_images}")
    rng = random.Random(args.seed)
    chosen = rng.sample(samples, args.num_images)
    crack_labels = parse_csv_set(args.crack_labels)
    ignore_labels = parse_csv_set(args.ignore_labels)
    component_labels = parse_csv_set(args.component_labels)

    # backbone + adapters
    ckpt = torch_load_compat(args.stage_a_ckpt.expanduser().resolve(), map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args") or {}
    bottleneck = int(ckpt_args.get("adapter_bottleneck", 64))
    dropout = float(ckpt_args.get("adapter_dropout", 0.1))
    adapters = {
        "ema": build_adapters(ckpt, "teacher_adapters_ema", bottleneck, dropout).to(device).eval(),
        "student": build_adapters(ckpt, "student_adapters", bottleneck, dropout).to(device).eval(),
    }
    backbone = _load_vit_backbone(args.teacher_weights, pretrained=True, device=device).to(device).eval()
    n_register = int(getattr(backbone.config, "num_register_tokens", 0))
    patch_size = int(getattr(backbone.config, "patch_size", 16))
    g = args.img_size // patch_size

    sources = ["raw", "ema", "student"]
    # accumulators
    delta_rel: dict[str, dict[int, dict[str, list[float]]]] = {
        s: {l: {"cls": [], "reg": [], "patch": []} for l in layers} for s in sources
    }
    cos_stats: dict[str, dict[int, dict[str, list[float]]]] = {
        s: {l: {"cls": [], "reg": [], "patch": []} for l in layers} for s in sources
    }
    cka_vals: dict[str, dict[int, list[float]]] = {s: {l: [] for l in layers} for s in ("ema", "student")}
    fisher_vals: dict[str, dict[int, list[float]]] = {s: {l: [] for l in layers} for s in sources}
    # probe features per (layer, source)
    feats: dict[str, dict[int, list[torch.Tensor]]] = {s: {l: [] for l in layers} for s in sources}
    labels: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    image_index: dict[str, int] = {}
    pos_ratios: list[float] = []

    for ii, sample in enumerate(chosen):
        image_index[sample["stem"]] = ii
        x, crack_s, valid_s = preprocess(sample, args.img_size, images_dir, crack_labels, ignore_labels, component_labels)
        x = x.to(device)
        with torch.no_grad():
            out = backbone(pixel_values=x, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states
            # patch labels at feature grid resolution: area fractions, foreground priority
            ps = args.img_size // g
            crack_t = torch.from_numpy(crack_s)
            valid_t = torch.from_numpy(valid_s)
            valid_f = valid_t.reshape(g, ps, g, ps).float().mean((1, 3)).reshape(-1)
            crack_f = (crack_t & valid_t).reshape(g, ps, g, ps).float().mean((1, 3)).reshape(-1)
            keep = valid_f >= args.min_valid_frac
            lab = torch.full((g * g,), -1, dtype=torch.long)
            lab[keep & (crack_f > args.fg_frac_thresh)] = 1
            lab[keep & (crack_f <= args.fg_frac_thresh)] = 0
            sel = torch.nonzero(lab >= 0).squeeze(1)
            if sel.numel() > args.max_patches_per_image:
                sel = sel[torch.randperm(sel.numel())[: args.max_patches_per_image]]
            sel = sel.cpu()
            lab_sel = lab[sel]
            if sel.numel() > 0:
                pos_ratios.append(float((lab_sel == 1).float().mean()))

            viz_rows: list[dict] = []
            for l in layers:
                if l >= len(hs):
                    raise RuntimeError(f"hidden_states has {len(hs)} entries, missing {l}")
                raw_tok = hs[l][0].float()  # adapters are fp32; cast tokens accordingly
                _, _, _, pmap_r = token_parts(raw_tok.unsqueeze(0), n_register, g)
                adapted_pmaps: dict[str, torch.Tensor] = {}
                for src in sources:
                    if src == "raw":
                        adapted_tok = raw_tok
                    else:
                        adapted_tok = adapters[src][str(l)](raw_tok.unsqueeze(0))[0]
                    _, _, _, pmap_a = token_parts(adapted_tok.unsqueeze(0), n_register, g)
                    if src in ("ema", "student"):
                        adapted_pmaps[src] = pmap_a
                    d = (adapted_tok - raw_tok)
                    rel = (d.norm(dim=-1) / (raw_tok.norm(dim=-1) + EPS))
                    cs = F.cosine_similarity(raw_tok, adapted_tok, dim=-1)
                    cls_rr, reg_rr, patch_rr = token_parts((rel.unsqueeze(0).unsqueeze(-1)), n_register, g)[:3]
                    cls_c, reg_c, patch_c = token_parts((cs.unsqueeze(0).unsqueeze(-1)), n_register, g)[:3]
                    delta_rel[src][l]["cls"].append(float(cls_rr.mean()))
                    delta_rel[src][l]["reg"].append(float(reg_rr.mean()) if reg_rr.numel() else 0.0)
                    delta_rel[src][l]["patch"].append(float(patch_rr.mean()))
                    cos_stats[src][l]["cls"].append(float(cls_c.mean()))
                    cos_stats[src][l]["reg"].append(float(reg_c.mean()) if reg_c.numel() else 1.0)
                    cos_stats[src][l]["patch"].append(float(patch_c.mean()))

                    pflat = pmap_r.reshape(-1, pmap_r.shape[1])
                    pflat_a = pmap_a.reshape(-1, pmap_a.shape[1])
                    if src in ("ema", "student"):
                        cka_vals[src][l].append(linear_cka(pflat, pflat_a))
                    # separability + probe features on selected patches
                    if sel.numel() > 4:
                        yy = lab_sel == 1
                        if int(yy.sum()) >= 2 and int((~yy).sum()) >= 2:
                            fisher_vals[src][l].append(fisher_ratio(pflat_a[sel], yy))
                        feats[src][l].append(pflat_a[sel].cpu())
                labels[l].append(lab_sel)

                if ii < args.num_viz_samples and l in viz_layers and adapted_pmaps:
                    viz_rows.append(
                        {"layer": l, "raw": pmap_r, "ema": adapted_pmaps.get("ema"), "student": adapted_pmaps.get("student")}
                    )

            if viz_rows:
                _save_sample_viz(out_dir / "samples", sample["stem"], x[0].cpu(), crack_s, valid_s, viz_rows, g)

    # ---- aggregate + save ----
    summary: dict = {"config": vars(args), "layers": layers, "img_size": args.img_size, "grid": g}
    summary["pos_ratio"] = float(np.mean(pos_ratios)) if pos_ratios else None
    summary["delta_rel"] = {s: {l: {k: float(np.mean(v)) for k, v in delta_rel[s][l].items()} for l in layers} for s in sources}
    summary["cos"] = {s: {l: {k: float(np.mean(v)) for k, v in cos_stats[s][l].items()} for l in layers} for s in sources}
    summary["cka"] = {s: {l: float(np.mean(v)) if v else None for l, v in cka_vals[s].items()} for s in ("ema", "student")}
    summary["fisher"] = {s: {l: float(np.mean(v)) if v else None for l, v in fisher_vals[s].items()} for s in sources}

    # probe per layer/source
    summary["probe"] = {s: {} for s in sources}
    xva = None
    for s in sources:
        for l in layers:
            if not feats[s][l]:
                continue
            X = torch.cat(feats[s][l], 0)
            Y = torch.cat(labels[l], 0).float()
            img_ids = []
            cur = 0
            for idx, chunk in enumerate(feats[s][l]):
                img_ids.extend([idx] * chunk.shape[0])
            img_ids = torch.tensor(img_ids)
            tr_mask = img_ids < args.probe_train_images
            va_mask = ~tr_mask
            if int(tr_mask.sum()) < 10 or int(va_mask.sum()) < 10:
                continue
            if Y[tr_mask].min() == Y[tr_mask].max() or Y[va_mask].min() == Y[va_mask].max():
                continue
            res = train_probe(X[tr_mask], Y[tr_mask], X[va_mask], Y[va_mask], args.probe_hidden, args.probe_epochs, device)
            summary["probe"][s][str(l)] = res

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _save_plots(out_dir, sources, layers, delta_rel, cos_stats, cka_vals, fisher_vals, summary)
    print(f"[done] analysis -> {out_dir}")
    print(json.dumps({k: summary[k] for k in ("delta_rel", "cos", "cka", "fisher")}, ensure_ascii=False, indent=2)[:2000])
    return 0


def _pca_fit(feats: torch.Tensor, n: int = 3):
    mean = feats.mean(0, keepdim=True)
    _, _, v = torch.linalg.svd(feats - mean, full_matrices=False)
    return mean, v[:n]


def _pca_project(feats: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor, g: int) -> np.ndarray:
    p = (feats - mean) @ basis.t()
    p = p - p.min(0, keepdim=True)[0]
    p = p / (p.max(0, keepdim=True)[0] + EPS)
    return (p.reshape(g, g, 3).cpu().numpy() * 255).astype(np.uint8)


def _save_sample_viz(out_dir: Path, stem: str, x01, crack_s, valid_s, rows: list[dict], g: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = (x01.permute(1, 2, 0).cpu().numpy() * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)).clip(0, 1)
    img_u8 = (img * 255).astype(np.uint8)
    gt = np.zeros((*crack_s.shape, 3), np.uint8)
    gt[crack_s] = [255, 0, 0]
    gt[valid_s & ~crack_s] = [0, 255, 0]

    col_titles = [
        "image", "gt(crack=red)", "PCA raw", "PCA ema", "PCA student",
        "cos raw-ema", "cos raw-student", "|Δ|/|x| ema", "PCA Δema",
    ]
    n = len(rows)
    fig, axes = plt.subplots(n, len(col_titles), figsize=(2.6 * len(col_titles), 2.6 * n), squeeze=False)
    for r, d in enumerate(rows):
        raw = d["raw"].reshape(-1, d["raw"].shape[1])
        ema = d["ema"].reshape(-1, d["ema"].shape[1])
        stu = d["student"].reshape(-1, d["student"].shape[1])
        mean, basis = _pca_fit(raw, 3)
        pca_raw = _pca_project(raw, mean, basis, g)
        pca_ema = _pca_project(ema, mean, basis, g)
        pca_stu = _pca_project(stu, mean, basis, g)
        cos_ema = F.cosine_similarity(raw, ema, dim=-1).reshape(g, g).cpu().numpy()
        cos_stu = F.cosine_similarity(raw, stu, dim=-1).reshape(g, g).cpu().numpy()
        dvec = ema - raw
        dmag = (dvec.norm(dim=-1) / (raw.norm(dim=-1) + EPS)).reshape(g, g).cpu().numpy()
        dm, db = _pca_fit(dvec, 3)
        pca_delta = _pca_project(dvec, dm, db, g)
        panels = [
            img_u8, gt, pca_raw, pca_ema, pca_stu,
            np.stack([heat_rgb(cos_ema)] * 3, -1), np.stack([heat_rgb(cos_stu)] * 3, -1),
            np.stack([heat_rgb(dmag)] * 3, -1), pca_delta,
        ]
        for c, arr in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(arr)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=8)
        axes[r][0].set_ylabel(f"L{d['layer']}", fontsize=10)
    fig.suptitle(stem[:60], fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem[:48]}_analysis.png", dpi=85)
    plt.close(fig)


def _save_plots(out_dir, sources, layers, delta_rel, cos_stats, cka_vals, fisher_vals, summary) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for src in sources:
        xs = [str(l) for l in layers]
        axes[0].plot(xs, [summary["delta_rel"][src][l]["patch"] for l in layers], marker="o", label=src)
        axes[0].plot(xs, [summary["delta_rel"][src][l]["cls"] for l in layers], marker="x", ls="--", label=f"{src}-cls")
        axes[1].plot(xs, [summary["cos"][src][l]["patch"] for l in layers], marker="o", label=src)
        axes[2].plot(xs, [summary["fisher"][src][l] if summary["fisher"][src][l] is not None else np.nan for l in layers], marker="o", label=src)
    axes[0].set_title("adapter relative perturbation (patch solid / cls dashed)")
    axes[1].set_title("cosine(raw, adapted) patch")
    axes[2].set_title("Fisher ratio (crack vs bg)")
    for ax in axes:
        ax.set_xlabel("layer")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "metrics.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, src in zip(axes, ("ema", "student")):
        vals = [summary["cka"][src][l] for l in layers]
        ax.bar([str(l) for l in layers], [v if v is not None else 0 for v in vals])
        ax.set_title(f"linear CKA raw vs {src}")
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "cka.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
