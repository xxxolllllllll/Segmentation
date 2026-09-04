"""
================================================================================
【教师网络】冻结 ViT，输出 6 个尺度的「空间特征图」供蒸馏
================================================================================

要抽哪几层：与实验方案一致，取第 3/4/7/8/11/12 个 Transformer block 的输出，
再 reshape 成 [B, C, H_grid, W_grid]，与学生 P3/P4/P5 对齐用。

实现：**仅**通过 Hugging Face ``transformers`` 的 ``from_pretrained`` 从**本地目录**
加载（目录内需含 ``config.json`` 与权重）。默认目录为::

    solution/weights/dinov3-vitb16-pretrain-lvd1689m/

若该目录缺失，请先在本机执行 ``python scripts/download_dinov3_teacher.py``（需 Hub 登录）。

DINOv3 带 register token，``HFViTTeacher`` 会跳过 CLS+register，只取 patch 序列再铺成网格。
================================================================================
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn


def _teacher_load_dtype(device: Optional[torch.device]) -> Optional[torch.dtype]:
    """CUDA 上用半精度加载，权重体积约减半、拷贝更快；CPU 上保持 float32。"""
    if device is None or device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _load_vit_backbone(
    model_id: str,
    *,
    pretrained: bool,
    device: Optional[torch.device],
) -> nn.Module:
    from transformers import AutoConfig, AutoModel

    if not pretrained:
        config = AutoConfig.from_pretrained(model_id, local_files_only=True)
        return AutoModel.from_config(config)

    torch_dtype = _teacher_load_dtype(device)
    kwargs: dict = {"local_files_only": True}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    # 直接映射到 GPU 可避免「先占满 CPU 内存再整块 .to(cuda)」，通常明显快于先 CPU 加载
    if device is not None and device.type == "cuda":
        try:
            return AutoModel.from_pretrained(
                model_id,
                device_map={"": str(device)},
                **kwargs,
            )
        except Exception:
            pass
    return AutoModel.from_pretrained(model_id, **kwargs)

# 与 scripts/download_dinov3_teacher.py 中的子目录名一致
_DEFAULT_TEACHER_SUBDIR = "dinov3-vitb16-pretrain-lvd1689m"


def default_teacher_weights_dir() -> Path:
    """``solution/weights/dinov3-vitb16-pretrain-lvd1689m`` 的绝对路径。"""
    return Path(__file__).resolve().parent.parent / "weights" / _DEFAULT_TEACHER_SUBDIR


class HFViTTeacher(nn.Module):
    """
    Hugging Face ``DINOv3ViTModel`` 等 ViT 教师（**仅从本地快照加载**）。

    ``hidden_states[k]``：第 0 项为 embedding 后，第 ``k`` 项为第 ``k`` 个 encoder 层输出。
    取索引 ``[3,4,7,8,11,12]`` 与实验方案中「第 3/4/7/8/11/12 个 block 输出」语义对齐。

    ``model_id`` 应为含 ``config.json`` 的目录路径（一般为 ``default_teacher_weights_dir()``）。
    """

    _HIDDEN_STATE_INDICES = (3, 4, 7, 8, 11, 12)

    def __init__(
        self,
        model_id: str,
        img_size: int = 1024,
        pretrained: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        try:
            import transformers  # noqa: F401
        except ImportError as e:
            raise ImportError("使用教师 ViT 时需安装: pip install transformers") from e

        self.model_id = model_id
        self.img_size = img_size

        self.backbone = _load_vit_backbone(model_id, pretrained=pretrained, device=device)

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.eval()

        cfg = self.backbone.config
        self.patch_size = int(getattr(cfg, "patch_size", 16))
        self.num_register_tokens = int(getattr(cfg, "num_register_tokens", 0))
        self.embed_dim = int(cfg.hidden_size)

        assert img_size % self.patch_size == 0, (
            f"img_size {img_size} 应能被 patch_size {self.patch_size} 整除（DINOv3 ViT-B/16 为 16）"
        )
        self.grid_size = img_size // self.patch_size
        self._n_patch = self.grid_size * self.grid_size

        n_layers = int(getattr(cfg, "num_hidden_layers", 12))
        max_idx = max(self._HIDDEN_STATE_INDICES)
        if max_idx > n_layers:
            raise ValueError(
                f"模型仅 {n_layers} 层，无法取 hidden_states[{max_idx}]；请换更大模型或改 _HIDDEN_STATE_INDICES"
            )

    def _tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        """[B, seq, C] -> 仅 patch 部分 -> [B, C, H, W]。"""
        B, seq, C = x.shape
        n_skip = 1 + self.num_register_tokens
        # DINOv3: [CLS] + register_tokens + patches（patches 在序列尾部连续）
        if seq >= n_skip + self._n_patch:
            patch_tokens = x[:, n_skip : n_skip + self._n_patch, :]
        elif seq == self._n_patch:
            patch_tokens = x
        elif seq > self._n_patch:
            patch_tokens = x[:, -self._n_patch :, :]
        else:
            raise RuntimeError(
                f"序列长度 {seq} 与期望 patch 数 {self._n_patch} 不匹配（img_size={self.img_size}, patch={self.patch_size}）"
            )
        g = self.grid_size
        out = patch_tokens.reshape(B, g, g, C).permute(0, 3, 1, 2).contiguous()
        return out

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [N, 3, H, W]，H=W=img_size，已按 ImageNet 归一化（与检测蒸馏里对教师支路的预处理一致）

        Returns:
            长度为 6 的 list，每项 [N, embed_dim, grid, grid]
        """
        self.eval()
        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = outputs.hidden_states
        if hs is None:
            raise RuntimeError("模型未返回 hidden_states，请确认实现为 Transformer ViT 且支持 output_hidden_states")

        feats: List[torch.Tensor] = []
        for idx in self._HIDDEN_STATE_INDICES:
            if idx >= len(hs):
                raise RuntimeError(f"hidden_states 长度为 {len(hs)}，缺少索引 {idx}")
            feats.append(self._tokens_to_map(hs[idx]))
        return feats


def build_teacher(
    img_size: int,
    pretrained: bool = True,
    *,
    weights_dir: Optional[str | Path] = None,
    device: Optional[torch.device] = None,
) -> HFViTTeacher:
    """
    构建教师网络：仅从本地 HF 格式目录 ``from_pretrained``。

    Args:
        img_size: 教师输入边长（须能被 patch_size 整除）。
        pretrained: True 加载权重；False 仅按 config 随机初始化结构（调试用）。
        weights_dir: 含 ``config.json`` 的目录；``None`` 时使用 ``default_teacher_weights_dir()``。
        device: 训练用设备。若为 CUDA，加载时优先使用 ``device_map`` 直拷到 GPU，并用 bf16/fp16
            减小读盘与搬运量（需 ``accelerate`` 时 ``device_map`` 才可用；否则自动退化为 CPU 半精度加载再 ``.to(device)``）。

    Raises:
        FileNotFoundError: 目录不存在或缺少 ``config.json``。
    """
    root = Path(weights_dir) if weights_dir is not None else default_teacher_weights_dir()
    root = root.resolve()
    cfg = root / "config.json"
    if not root.is_dir() or not cfg.is_file():
        raise FileNotFoundError(
            f"教师权重目录无效或缺少 config.json: {root}\n"
            f"请在 solution 目录执行: python scripts/download_dinov3_teacher.py"
        )
    return HFViTTeacher(model_id=str(root), img_size=img_size, pretrained=pretrained, device=device)
