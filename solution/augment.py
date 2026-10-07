"""Shared lightweight geometric augmentation for the K-fold few-shot pipeline.

Per the refactored protocol, training samples receive a simple geometric
transform (90-degree-multiple rotation + horizontal flip) with probability
``prob`` each epoch. The same helper is used by Stage B (teacher) and Stage C
(student) so both see identical augmentation.
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image


def _rot90(arr: np.ndarray, k: int) -> np.ndarray:
    if k == 0:
        return arr
    return np.rot90(arr, k).copy()


def _hflip(arr: np.ndarray) -> np.ndarray:
    return arr[:, ::-1].copy()


def geometric_augment(
    img: np.ndarray,
    masks: list[np.ndarray],
    *,
    prob: float = 0.5,
    rng: random.Random | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Apply rotation+flip consistently to an image and its label masks.

    Args:
        img: HxWx3 uint8 image.
        masks: list of HxW arrays (dtypes may differ, e.g. uint8 label + bool).
        prob: probability of applying the geometric transform.
        rng: optional ``random.Random`` instance; uses module ``random`` when None.

    Returns:
        ``(img, masks)`` transformed with the same sampled rotation/flip.
    """
    rng = rng if rng is not None else random
    if rng.random() >= prob:
        return img, masks
    k = rng.randint(0, 3)
    hflip = rng.random() < 0.5

    img = _rot90(img, k)
    if hflip:
        img = _hflip(img)

    out: list[np.ndarray] = []
    for m in masks:
        m = _rot90(m, k)
        if hflip:
            m = _hflip(m)
        out.append(m)
    return img, out


def geometric_augment_pil(
    img: Image.Image,
    mask: Image.Image,
    *,
    prob: float = 0.5,
    rng: random.Random | None = None,
) -> tuple[Image.Image, Image.Image]:
    """PIL-wrapper for a single image+mask pair (used by Stage B transforms)."""
    img_out, (mask_out,) = geometric_augment(
        np.asarray(img), [np.asarray(mask)], prob=prob, rng=rng
    )
    return Image.fromarray(img_out), Image.fromarray(mask_out)


def photometric_augment(
    img: np.ndarray,
    *,
    rng: random.Random,
    strength: float = 0.2,
    blur_prob: float = 0.3,
    noise_std: float = 6.0,
) -> np.ndarray:
    """Random photometric augmentation for an HxWx3 uint8 image (colour only).

    brightness / contrast / gamma are sampled in ``1 ± strength``; optional
    Gaussian blur and additive Gaussian noise add capture-condition robustness.
    The label mask is left unchanged.
    """
    from PIL import ImageFilter

    x = img.astype(np.float32)
    x = x * (1.0 + rng.uniform(-strength, strength))                 # brightness
    mean = float(x.mean())
    x = (x - mean) * (1.0 + rng.uniform(-strength, strength)) + mean  # contrast
    gamma = max(0.2, 1.0 + rng.uniform(-strength, strength))          # gamma
    x = np.power(np.clip(x, 0.0, 255.0) / 255.0, gamma) * 255.0
    if blur_prob > 0.0 and rng.random() < blur_prob:
        radius = rng.uniform(0.4, 1.5)
        x = np.asarray(
            Image.fromarray(np.clip(x, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius)),
            dtype=np.float32,
        )
    if noise_std > 0.0:
        nprng = np.random.default_rng(rng.randrange(2 ** 32))
        x = x + nprng.normal(0.0, noise_std, size=x.shape)
    return np.clip(x, 0.0, 255.0).astype(np.uint8)
