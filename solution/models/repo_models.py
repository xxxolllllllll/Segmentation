"""Adapter to reuse the SMG-Net repo's single-file segmentation models as
Stage-C students.

Each repo model is a self-contained ``nn.Module`` (or factory) that takes a
3-channel image and returns a 1-channel crack map (raw logits or sigmoid).
``RepoStudent`` wraps it to the Stage-C interface: ``forward(x) -> (logits[2],
feats)`` where ``logits`` is 2-class (background/crack).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_MODELS = Path("/home/zgt/Segmentation/SMG-Net/HuiZhenxing/SMG-Net/models")

# key -> (file, class/factory, args, kwargs)
SPECS: dict[str, tuple[str, str, tuple, dict]] = {
    "attunet": ("Attention_UNet.py", "AttU_Net", (), {"in_channel": 3, "num_classes": 1}),
    "cmunet": ("CMUNet.py", "CMUNet", (), {"img_ch": 3, "output_ch": 1}),
    "swinunet_repo": ("SwingUNet.py", "SwingUNet", (), {"img_size": 1024, "in_ch": 3, "num_classes": 1}),
    "maresunet": ("MAResUNet.py", "MAResUNet", (), {"num_channels": 3, "num_classes": 1}),
    "macunet": ("MACUNet.py", "MACUNet", (), {"band_num": 3, "class_num": 1}),
    "abcnet": ("ABCNet.py", "ABCNet", (), {"band": 3, "n_classes": 1}),
    "segnet_repo": ("SegNet.py", "SegNet", (1,), {}),
    "pspnet_repo": ("PSPNet.py", "PSPNet", (1,), {}),
    "swinsys": ("SegFormer.py", "SwinSys", (), {"img_size": 1024, "num_classes": 1}),
}


def _load_module(path: Path, tag: str):
    name = f"smgnet_repo_{tag}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


class RepoStudent(nn.Module):
    def __init__(self, key: str, num_classes: int = 2):
        super().__init__()
        if key not in SPECS:
            raise KeyError(f"unknown repo student {key!r}")
        fname, cls, args, kwargs = SPECS[key]
        mod = _load_module(REPO_MODELS / fname, key)
        obj = getattr(mod, cls)
        self.net = obj(*args, **kwargs) if callable(obj) else obj
        self.neck_channels = (1, 1, 1)  # unused (no distillation in baselines)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        z = out
        if z.shape[1] == 1:
            if float(z.min()) >= -1e-6 and float(z.max()) <= 1.0 + 1e-6:
                p = z.clamp(1e-6, 1.0 - 1e-6)
                z = torch.log(p / (1.0 - p))
            logits = torch.cat([-z, z], dim=1)
        else:
            logits = z
        return logits, ()


class MACUNetKDStudent(nn.Module):
    """MACUNet exposed for feature distillation (plan A).

    Distillation taps (stride-aligned with the teacher's FPN p3/p4/p5):
      f0 = conv14  @ H/8  (c=128)  <-> p3 (H/8)
      f1 = conv5   @ H/16 (c=256)  <-> p4 (H/16)
      f2 = maxpool2(conv5) @ H/32 (c=256)  <-> p5 (H/32)
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        mod = _load_module(REPO_MODELS / "MACUNet.py", "macunet_kd")
        self.net = mod.MACUNet(band_num=3, class_num=1)
        self._f0: torch.Tensor | None = None
        self._f1: torch.Tensor | None = None
        self.net.conv14.register_forward_hook(self._hook0)
        self.net.conv5.register_forward_hook(self._hook1)
        self.neck_channels = (128, 256, 256)

    def _hook0(self, _m, _i, out):
        self._f0 = out

    def _hook1(self, _m, _i, out):
        self._f1 = out

    def forward(self, x: torch.Tensor):
        self._f0 = None
        self._f1 = None
        out = self.net(x)
        z = out
        if float(z.min()) >= -1e-6 and float(z.max()) <= 1.0 + 1e-6:
            p = z.clamp(1e-6, 1.0 - 1e-6)
            z = torch.log(p / (1.0 - p))
        logits = torch.cat([-z, z], dim=1)
        f0, f1 = self._f0, self._f1
        if f0 is None or f1 is None:
            raise RuntimeError("MACUNet feature hooks did not fire (check module names conv14/conv5)")
        f2 = F.max_pool2d(f1, kernel_size=2, stride=2)
        return logits, (f0, f1, f2)
