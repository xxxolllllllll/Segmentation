from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn as nn

try:
    # Newer teacher_vit.py exports this helper.
    from .teacher_vit import _load_vit_backbone as _load_vit_backbone_from_teacher
except Exception:
    _load_vit_backbone_from_teacher = None

try:
    from .teacher_vit import default_teacher_weights_dir as _default_teacher_weights_dir_from_teacher
except Exception:
    _default_teacher_weights_dir_from_teacher = None


def default_teacher_weights_dir() -> Path:
    if _default_teacher_weights_dir_from_teacher is not None:
        return _default_teacher_weights_dir_from_teacher()
    # Fallback for older repositories: derive path locally.
    return Path(__file__).resolve().parent.parent / "weights" / "dinov3-vitb16-pretrain-lvd1689m"


def _teacher_load_dtype(device: torch.device | None) -> torch.dtype | None:
    if device is None or device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _load_vit_backbone(model_id: str, *, pretrained: bool, device: torch.device | None) -> nn.Module:
    """
    Compatibility loader:
    - uses teacher_vit private helper when available
    - falls back to local transformers loading when helper is absent
    """
    if _load_vit_backbone_from_teacher is not None:
        return _load_vit_backbone_from_teacher(model_id, pretrained=pretrained, device=device)

    from transformers import AutoConfig, AutoModel

    if not pretrained:
        config = AutoConfig.from_pretrained(model_id, local_files_only=True)
        return AutoModel.from_config(config)

    kwargs: dict = {"local_files_only": True}
    torch_dtype = _teacher_load_dtype(device)
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    if device is not None and device.type == "cuda":
        try:
            return AutoModel.from_pretrained(model_id, device_map={"": str(device)}, **kwargs)
        except Exception:
            pass
    return AutoModel.from_pretrained(model_id, **kwargs)


class BottleneckResidualAdapter(nn.Module):
    """LN -> down -> GELU -> drop -> up + residual."""

    def __init__(self, hidden_size: int, bottleneck_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError(f"bottleneck_dim must be > 0, got {bottleneck_dim}")
        self.norm = nn.LayerNorm(hidden_size)
        self.down = nn.Linear(hidden_size, bottleneck_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Linear(bottleneck_dim, hidden_size)

        # Zero-init up projection to keep initial behavior close to identity.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.down(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.up(y)
        return x + y


class ProjectionHead(nn.Module):
    """Simple 2-layer MLP head for self-distillation logits."""

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 1024, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DINOv3StageAModel(nn.Module):
    """
    Frozen DINOv3 backbone + adapters on selected blocks + projection head.

    This model is designed for Stage-A self-supervised continued pretraining.
    """

    def __init__(
        self,
        *,
        weights_dir: str | Path | None = None,
        pretrained: bool = True,
        device: torch.device | None = None,
        adapter_indices: Sequence[int] = (3, 4, 7, 8, 11, 12),
        bottleneck_dim: int = 64,
        adapter_dropout: float = 0.1,
        proj_hidden_dim: int = 2048,
        proj_out_dim: int = 1024,
        proj_dropout: float = 0.0,
        freeze_backbone: bool = True,
    ) -> None:
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
        self.adapter_indices = tuple(int(i) for i in adapter_indices)
        if len(self.adapter_indices) == 0:
            raise ValueError("adapter_indices cannot be empty")

        self.backbone = _load_vit_backbone(self.weights_dir, pretrained=pretrained, device=device)
        self.hidden_size = int(self.backbone.config.hidden_size)

        self.adapters = nn.ModuleDict(
            {str(i): BottleneckResidualAdapter(self.hidden_size, bottleneck_dim, adapter_dropout) for i in self.adapter_indices}
        )
        self.projector = ProjectionHead(
            in_dim=self.hidden_size,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_out_dim,
            dropout=proj_dropout,
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def backbone_hidden_states(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Run frozen backbone and return hidden states tuple.
        """
        with torch.no_grad():
            outputs = self.backbone(pixel_values=x, output_hidden_states=True, return_dict=True)
        hs = outputs.hidden_states
        if hs is None:
            raise RuntimeError("Backbone did not return hidden_states.")
        return hs

    def adapt_hidden_states(self, hidden_states: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        adapted: list[torch.Tensor] = []
        for idx in self.adapter_indices:
            if idx >= len(hidden_states):
                raise RuntimeError(f"hidden_states length={len(hidden_states)} missing index={idx}")
            x = hidden_states[idx]
            x = self.adapters[str(idx)](x)
            adapted.append(x)
        return adapted

    @staticmethod
    def pool_cls(adapted_hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(adapted_hidden_states) == 0:
            raise ValueError("adapted_hidden_states cannot be empty")
        cls_tokens = [h[:, 0, :] for h in adapted_hidden_states]
        return torch.stack(cls_tokens, dim=0).mean(dim=0)

    def project_from_hidden_states(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        adapted = self.adapt_hidden_states(hidden_states)
        pooled = self.pool_cls(adapted)
        return self.projector(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = self.backbone_hidden_states(x)
        return self.project_from_hidden_states(hs)

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        return self.adapters.parameters()

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        for p in self.adapters.parameters():
            yield p
        for p in self.projector.parameters():
            yield p

