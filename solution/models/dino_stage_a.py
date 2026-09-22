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


class DenseHead(nn.Module):
    """3-layer MLP head used for the iBOT / dense patch self-distillation loss."""

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
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
        ibot_hidden_dim: int = 2048,
        ibot_out_dim: int = 2048,
        ibot_dropout: float = 0.0,
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
        self.patch_size = int(getattr(self.backbone.config, "patch_size", 16))
        self.num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))

        self.adapters = nn.ModuleDict(
            {str(i): BottleneckResidualAdapter(self.hidden_size, bottleneck_dim, adapter_dropout) for i in self.adapter_indices}
        )
        self.projector = ProjectionHead(
            in_dim=self.hidden_size,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_out_dim,
            dropout=proj_dropout,
        )
        # Dense (iBOT) heads: one per scale group P3/P4/P5 = consecutive adapter pairs.
        self.layer_groups = [pos // 2 for pos in range(len(self.adapter_indices))]
        self.num_ibot_groups = max(self.layer_groups) + 1 if self.layer_groups else 0
        self.ibot_heads = nn.ModuleDict(
            {
                str(g): DenseHead(
                    in_dim=self.hidden_size,
                    hidden_dim=ibot_hidden_dim,
                    out_dim=ibot_out_dim,
                    dropout=ibot_dropout,
                )
                for g in range(self.num_ibot_groups)
            }
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def backbone_hidden_states(
        self, x: torch.Tensor, bool_masked_pos: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, ...]:
        """
        Run frozen backbone and return hidden states tuple.

        ``bool_masked_pos`` ([B, num_patches], True = masked) is forwarded to the
        DINOv3 embeddings so masked patches are replaced by the (pretrained) mask token.
        """
        with torch.no_grad():
            outputs = self.backbone(
                pixel_values=x,
                output_hidden_states=True,
                return_dict=True,
                bool_masked_pos=bool_masked_pos,
            )
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

    def _tokens_to_map(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """[B, seq, C] -> patch tokens -> [B, C, H, W] (H=H_in/patch, W=W_in/patch)."""
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

    @torch.no_grad()
    def extract_adapted_feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return Stage-A adapted DINO feature maps for the adapter indices.

        Used as the S2/S2_attn distillation target. Applies the (frozen) adapters
        to the backbone hidden states and reshapes the patch tokens to a grid.
        """
        b, _, h, w = x.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"Input H/W must be divisible by patch_size={self.patch_size}, got {(h, w)}")
        hs = self.backbone_hidden_states(x)
        maps = []
        for idx in self.adapter_indices:
            if idx >= len(hs):
                raise RuntimeError(f"hidden_states length={len(hs)} missing index={idx}")
            maps.append(self._tokens_to_map(self.adapters[str(idx)](hs[idx]), h=h, w=w))
        return maps

    def project_from_hidden_states(self, hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
        adapted = self.adapt_hidden_states(hidden_states)
        pooled = self.pool_cls(adapted)
        return self.projector(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = self.backbone_hidden_states(x)
        return self.project_from_hidden_states(hs)

    def tokens_to_patch_seq(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """[B, seq, C] -> patch tokens [B, num_patches, C] (drops CLS + register tokens)."""
        b, seq, c = tokens.shape
        gh, gw = h // self.patch_size, w // self.patch_size
        n_patch = gh * gw
        n_skip = 1 + self.num_register_tokens
        if seq >= n_skip + n_patch:
            return tokens[:, n_skip : n_skip + n_patch, :]
        if seq == n_patch:
            return tokens
        if seq > n_patch:
            return tokens[:, -n_patch:, :]
        raise RuntimeError(f"Unexpected token shape: seq={seq}, needed patches={n_patch}")

    def forward_dense(
        self,
        x: torch.Tensor,
        bool_masked_pos: torch.Tensor | None = None,
        return_patch: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
        """Student forward for dense + CLS self-distillation.

        Returns ``(cls_logits, patch_logits_per_adapter_layer)`` where each patch
        logit tensor is ``[B, num_patches, ibot_out_dim]``. ``bool_masked_pos``
        ([B, num_patches]) replaces masked patches with the backbone mask token.
        """
        b, _, h, w = x.shape
        hs = self.backbone_hidden_states(x, bool_masked_pos=bool_masked_pos)
        adapted = self.adapt_hidden_states(hs)
        cls_logits = self.projector(self.pool_cls(adapted))
        if not return_patch:
            return cls_logits, None

        patch_logits = [
            self.ibot_heads[str(self.layer_groups[i])](self.tokens_to_patch_seq(tokens, h, w))
            for i, tokens in enumerate(adapted)
        ]
        return cls_logits, patch_logits

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        return self.adapters.parameters()

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        for p in self.adapters.parameters():
            yield p
        for p in self.projector.parameters():
            yield p
        for p in self.ibot_heads.parameters():
            yield p

