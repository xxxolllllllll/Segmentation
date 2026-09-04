from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from checkpoint_io import torch_load_compat
from .dino_stage_a import BottleneckResidualAdapter, default_teacher_weights_dir
from .teacher_vit import _load_vit_backbone


class PairAdaptiveFuse(nn.Module):
    """Adaptive weighted fusion for a pair of same-shape feature maps."""

    def __init__(self, channels: int):
        super().__init__()
        self.head = nn.Conv2d(2 * channels, 2, kernel_size=1)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        z = torch.cat([a, b], dim=1)
        w = torch.softmax(self.head(z), dim=1)
        return w[:, 0:1] * a + w[:, 1:2] * b


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DINOv3StageBUNet(nn.Module):
    """
    Stage-B teacher fine-tuning model:
    frozen DINOv3 backbone + trainable adapters + U-Net-like decoder.
    """

    _IDX_LOW = (3, 4)
    _IDX_MID = (7, 8)
    _IDX_DEEP = (11, 12)

    def __init__(
        self,
        *,
        weights_dir: str | Path | None = None,
        pretrained: bool = True,
        device: torch.device | None = None,
        num_classes: int = 4,
        bottleneck_dim: int = 64,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()
        root = Path(weights_dir) if weights_dir is not None else default_teacher_weights_dir()
        root = root.resolve()
        cfg = root / "config.json"
        if not root.is_dir() or not cfg.is_file():
            raise FileNotFoundError(
                f"Invalid DINOv3 weights dir or missing config.json: {root}\n"
                "Please download local HF snapshot first."
            )

        self.weights_dir = str(root)
        self.backbone = _load_vit_backbone(self.weights_dir, pretrained=pretrained, device=device)
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.patch_size = int(getattr(self.backbone.config, "patch_size", 16))
        self.num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
        self.num_classes = int(num_classes)

        # Freeze backbone as required by stage-B.
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        adapter_idx = sorted(set(self._IDX_LOW + self._IDX_MID + self._IDX_DEEP))
        self.adapters = nn.ModuleDict(
            {str(i): BottleneckResidualAdapter(self.hidden_size, bottleneck_dim, adapter_dropout) for i in adapter_idx}
        )

        self.fuse_low = PairAdaptiveFuse(self.hidden_size)
        self.fuse_mid = PairAdaptiveFuse(self.hidden_size)
        self.fuse_deep = PairAdaptiveFuse(self.hidden_size)

        # Bridge layers.
        self.bridge_low = nn.Sequential(nn.Conv2d(self.hidden_size, 128, 1), nn.GELU())
        self.bridge_mid = nn.Sequential(nn.Conv2d(self.hidden_size, 192, 1), nn.GELU())
        self.bridge_deep = nn.Sequential(nn.Conv2d(self.hidden_size, 256, 1), nn.GELU())

        # U-Net-like decoder.
        self.dec8 = ConvBlock(256 + 192, 192)   # H/8
        self.dec4 = ConvBlock(192 + 128, 128)   # H/4
        self.dec2 = ConvBlock(128, 64)          # H/2
        self.dec1 = ConvBlock(64, 64)           # H
        self.head = nn.Conv2d(64, self.num_classes, kernel_size=1)

    def _tokens_to_map(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        b, seq, c = tokens.shape
        gh, gw = h // self.patch_size, w // self.patch_size
        n_patch = gh * gw
        n_skip = 1 + self.num_register_tokens
        if seq >= n_skip + n_patch:
            patch_tokens = tokens[:, n_skip : n_skip + n_patch, :]
        elif seq == n_patch:
            patch_tokens = tokens
        elif seq > n_patch:
            patch_tokens = tokens[:, -n_patch:, :]
        else:
            raise RuntimeError(f"Unexpected token shape: seq={seq}, needed patches={n_patch}")
        return patch_tokens.reshape(b, gh, gw, c).permute(0, 3, 1, 2).contiguous()

    def _adapt_map(self, hs: Sequence[torch.Tensor], idx: int, h: int, w: int) -> torch.Tensor:
        x = hs[idx]
        x = self.adapters[str(idx)](x)
        return self._tokens_to_map(x, h=h, w=w)

    @torch.no_grad()
    def extract_adapted_feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Return Stage-B adapted DINO feature maps [F3,F4,F7,F8,F11,F12].

        This is used by Stage-C semantic distillation. It keeps the Stage-B
        teacher frozen, but includes the trained adapters so the student mimics
        the domain-adapted representation rather than the raw foundation model.
        """
        b, _, h, w = x.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"Input H/W must be divisible by patch_size={self.patch_size}, got {(h, w)}")
        self.eval()
        out = self.backbone(pixel_values=x, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states
        if hs is None:
            raise RuntimeError("Backbone did not return hidden_states")
        indices = [*self._IDX_LOW, *self._IDX_MID, *self._IDX_DEEP]
        return [self._adapt_map(hs, idx, h=h, w=w) for idx in indices]

    def load_stage_a_adapters(self, ckpt_path: str | Path) -> None:
        ckpt_path = Path(ckpt_path)
        ckpt = torch_load_compat(ckpt_path, map_location="cpu", weights_only=True)
        sd = ckpt.get("teacher_adapters_ema", None) or ckpt.get("student_adapters", None)
        if sd is None:
            raise KeyError("stage-A checkpoint missing adapters state dict")
        missing, unexpected = self.adapters.load_state_dict(sd, strict=False)
        if missing:
            print(f"[stageB] adapter missing keys: {missing}")
        if unexpected:
            print(f"[stageB] adapter unexpected keys: {unexpected}")

    def trainable_parameters(self):
        for p in self.adapters.parameters():
            yield p
        for m in (
            self.fuse_low,
            self.fuse_mid,
            self.fuse_deep,
            self.bridge_low,
            self.bridge_mid,
            self.bridge_deep,
            self.dec8,
            self.dec4,
            self.dec2,
            self.dec1,
            self.head,
        ):
            for p in m.parameters():
                yield p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"Input H/W must be divisible by patch_size={self.patch_size}, got {(h, w)}")

        with torch.no_grad():
            out = self.backbone(pixel_values=x, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states
        if hs is None:
            raise RuntimeError("Backbone did not return hidden_states")

        low3 = self._adapt_map(hs, self._IDX_LOW[0], h, w)
        low4 = self._adapt_map(hs, self._IDX_LOW[1], h, w)
        mid7 = self._adapt_map(hs, self._IDX_MID[0], h, w)
        mid8 = self._adapt_map(hs, self._IDX_MID[1], h, w)
        dep11 = self._adapt_map(hs, self._IDX_DEEP[0], h, w)
        dep12 = self._adapt_map(hs, self._IDX_DEEP[1], h, w)

        low = self.fuse_low(low3, low4)
        mid = self.fuse_mid(mid7, mid8)
        deep = self.fuse_deep(dep11, dep12)

        low = self.bridge_low(low)    # H/16
        mid = self.bridge_mid(mid)    # H/16
        deep = self.bridge_deep(deep) # H/16

        x8 = F.interpolate(deep, scale_factor=2.0, mode="bilinear", align_corners=False)
        s8 = F.interpolate(mid, scale_factor=2.0, mode="bilinear", align_corners=False)
        x8 = self.dec8(torch.cat([x8, s8], dim=1))

        x4 = F.interpolate(x8, scale_factor=2.0, mode="bilinear", align_corners=False)
        s4 = F.interpolate(low, scale_factor=4.0, mode="bilinear", align_corners=False)
        x4 = self.dec4(torch.cat([x4, s4], dim=1))

        x2 = F.interpolate(x4, scale_factor=2.0, mode="bilinear", align_corners=False)
        x2 = self.dec2(x2)
        x1 = F.interpolate(x2, scale_factor=2.0, mode="bilinear", align_corners=False)
        x1 = self.dec1(x1)
        logits = self.head(x1)
        return logits

