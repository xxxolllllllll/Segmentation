"""
================================================================================
【蒸馏积木】教师 ViT 与学生 CNN/YOLO 之间的对齐与损失（可复用模块）
================================================================================

包含什么：
  StudentChannelAlign — 把学生三层特征通道数统一投到「教师通道数」（如 768）；
  AdaptiveTeacherFusion — 把教师相邻两层的特征图融合成一路，再与学生同尺度比；
  SingleLayerTeacherTargets — 消融用：每层只取单层教师特征（F3/F7/F11），不做相邻层融合；
  FeatureDistillLoss — 对多层特征做 SmoothL1（加权求和 = L_feat）；
  NWDLoss — 归一化 Wasserstein 距离，用作检测框回归的替代损失；
  total_loss — 把分类/定位/蒸馏/DFL 等按系数相加（通用模板）。

依赖：仅 PyTorch。
================================================================================
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentChannelAlign(nn.Module):
    """
    学生网络多尺度特征通道对齐模块。

    将 YOLO 颈部输出的 P3、P4、P5 的通道数线性投影到与教师（如 DINOv3 ViT-B）
    patch 嵌入维度一致（默认 768），使异构网络间可在同一通道语义空间计算蒸馏损失。

    默认通道须与所用检测模型 neck 一致（如 yolo11m 常见 P3/P4/P5=256/512/1024；小模型可能是 192/384/512）。
    """

    def __init__(self, in_channels: Sequence[int] = (256, 512, 512), out_channels: int = 768):
        """
        参数:
            in_channels: 三元组，依次为 P3、P4、P5 的输入通道数。
            out_channels: 投影后的通道数，需与教师特征通道数 C_t 一致（如 768）。
        """
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels])

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对三层学生特征分别做 1×1 卷积。

        参数:
            p3: [N, C3, H3, W3]，通常对应相对高分辨率浅层特征。
            p4: [N, C4, H4, W4]。
            p5: [N, C5, H5, W5]，通常对应低分辨率深层语义。

        返回:
            三个张量，通道均为 out_channels，空间尺寸与输入一致，
            分别记为 P3'、P4'、P5'，用于与教师融合特征计算 L_feat。
        """
        return self.proj[0](p3), self.proj[1](p4), self.proj[2](p5)


class AdaptiveTeacherFusion(nn.Module):
    """
    教师相邻编码层特征的自适应加权融合模块。

    对每组相邻教师特征 F_s、F_d（如第 3/4、7/8、11/12 层）：
    1. 双线性插值到与学生对应层（P3/P4/P5）相同的空间尺寸；
    2. 在通道维拼接后，经 1×1 卷积得到 2 通道 logits；
    3. 在通道维做 softmax 得到空间位置相关的权重 w1、w2；
    4. F_target = w1 * F_s + w2 * F_d。

    三组融合分别使用独立的 weight head，避免不同尺度间参数纠缠。
    """

    def __init__(self, channels: int = 768):
        """
        参数:
            channels: 教师单路特征的通道数（ViT-B 常见为 768）。
                      拼接后输入卷积的通道数为 2 * channels。
        """
        super().__init__()
        self.weight_heads = nn.ModuleList(
            [
                nn.Conv2d(2 * channels, 2, kernel_size=1),  # P3 尺度（浅层）
                nn.Conv2d(2 * channels, 2, kernel_size=1),  # P4 尺度（中层）
                nn.Conv2d(2 * channels, 2, kernel_size=1),  # P5 尺度（深层）
            ]
        )

    @staticmethod
    def _to_nchw(feat: torch.Tensor) -> torch.Tensor:
        """
        将特征张量规范为 NCHW 布局，供 interpolate 与卷积使用。

        支持:
            - 已是 [N, C, H, W] 则原样返回；
            - 若为 [N, H, W, C]（通道在最后且 C 小于空间维度的启发式），则 permute 为 NCHW。

        参数:
            feat: 四维张量。

        返回:
            contiguous 的 [N, C, H, W]。

        异常:
            ValueError: 非四维输入。
        """
        if feat.dim() != 4:
            raise ValueError(f"Expected 4D feature, got shape={tuple(feat.shape)}")
        n, a, b, c = feat.shape
        # 仅当明显是 NHWC 时才 permute：空间维相等且与通道维不同（如 [N,64,64,768]）。
        # 禁止用「dim1 > dim[-1]」判断：ViT 教师是标准 NCHW [N,768,H,W]，768>H 会被误判成 NHWC，
        # permute 后通道错位，融合头会看到 128 维而非 2*768=1536。
        if a == b and c != a:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def _fuse_one(
        self,
        fs: torch.Tensor,
        fd: torch.Tensor,
        target_hw: Tuple[int, int],
        head: nn.Conv2d,
    ) -> torch.Tensor:
        """
        对一对相邻教师特征完成「空间对齐 + 可学习加权融合」。

        参数:
            fs: 教师较浅层特征（如第 3 层），可为 NCHW 或 NHWC。
            fd: 教师较深层特征（如第 4 层），布局与 fs 一致。
            target_hw: (H, W)，与学生对应层特征图空间尺寸一致，如 P3 对应 (128,128)。
            head: 该尺度专用的 Conv2d(2C→2)，输出两路 logits 经 softmax 得 w1、w2。

        返回:
            [N, C, H, W] 的融合教师目标特征，用于与 student_aligned 计算 SmoothL1。
        """
        fs = self._to_nchw(fs)
        fd = self._to_nchw(fd)
        fs = F.interpolate(fs, size=target_hw, mode="bilinear", align_corners=False)
        fd = F.interpolate(fd, size=target_hw, mode="bilinear", align_corners=False)

        concat = torch.cat([fs, fd], dim=1)
        logits = head(concat)  # [N, 2, H, W]
        weight = torch.softmax(logits, dim=1)
        w1 = weight[:, 0:1]
        w2 = weight[:, 1:2]
        return w1 * fs + w2 * fd

    def forward(
        self,
        teacher_feats: Sequence[torch.Tensor],
        target_sizes: Sequence[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对六张教师特征做三组「相邻层融合」，输出三个尺度的教师目标特征。

        参数:
            teacher_feats: 长度为 6 的序列，顺序必须为
                [F3, F4, F7, F8, F11, F12]（与专利中编码块索引一致）。
            target_sizes: 长度为 3，依次为 P3、P4、P5 的空间尺寸 (H, W)，
                例如输入 1024×1024 时常见为 (128,128)、(64,64)、(32,32)。
                需与 forward 学生特征的实际 H、W 一致，否则蒸馏空间不对齐。

        返回:
            (T1, T2, T3): 分别与 P3'、P4'、P5' 同空间尺寸、同通道数 C 的三张融合特征。

        异常:
            ValueError: teacher_feats 长度不为 6 或 target_sizes 长度不为 3。
        """
        if len(teacher_feats) != 6:
            raise ValueError("teacher_feats must contain 6 feature maps: [F3,F4,F7,F8,F11,F12].")
        if len(target_sizes) != 3:
            raise ValueError("target_sizes must contain 3 tuples for P3/P4/P5.")

        t1 = self._fuse_one(teacher_feats[0], teacher_feats[1], target_sizes[0], self.weight_heads[0])
        t2 = self._fuse_one(teacher_feats[2], teacher_feats[3], target_sizes[1], self.weight_heads[1])
        t3 = self._fuse_one(teacher_feats[4], teacher_feats[5], target_sizes[2], self.weight_heads[2])
        return t1, t2, t3


class SingleLayerTeacherTargets(nn.Module):
    """
    消融 **A4**：不使用相邻层自适应融合。

    对三组尺度分别仅使用教师 **单层** patch 特征图（与实验方案一致：F3→P3、F7→P4、F11→P5），
    双线性插值到与学生对应层相同空间尺寸后直接作为蒸馏目标。**无可学习参数**。
    """

    def forward(
        self,
        teacher_feats: Sequence[torch.Tensor],
        target_sizes: Sequence[Tuple[int, int]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(teacher_feats) != 6:
            raise ValueError("teacher_feats must be length 6: [F3,F4,F7,F8,F11,F12].")
        if len(target_sizes) != 3:
            raise ValueError("target_sizes must be length 3 for P3/P4/P5.")
        outs = []
        for feat_idx, hw in zip((0, 2, 4), target_sizes):
            f = AdaptiveTeacherFusion._to_nchw(teacher_feats[feat_idx])
            outs.append(F.interpolate(f, size=hw, mode="bilinear", align_corners=False))
        return outs[0], outs[1], outs[2]


class FeatureDistillLoss(nn.Module):
    """
    多尺度特征蒸馏损失。

    L_feat = Σ_l λ_l · SmoothL1(student_aligned_l, teacher_target_l)

    默认 λ = (0.5, 0.3, 0.2) 对应浅层 P3 权重更大，与专利中强调局部细节迁移一致。
    """

    def __init__(self, lambdas: Sequence[float] = (0.5, 0.3, 0.2)):
        """
        参数:
            lambdas: 三个非负标量，依次对应 P3、P4、P5 三层蒸馏项的权重。
        """
        super().__init__()
        if len(lambdas) != 3:
            raise ValueError("lambdas must be length 3.")
        self.register_buffer("lambdas", torch.tensor(lambdas, dtype=torch.float32))
        self.criterion = nn.SmoothL1Loss(reduction="mean")

    def forward(self, student_aligned: Sequence[torch.Tensor], teacher_targets: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        计算加权后的总特征蒸馏损失标量。

        参数:
            student_aligned: 长度为 3，依次为 P3'、P4'、P5'（已与教师同通道、同空间尺寸）。
            teacher_targets: 长度为 3，依次为 AdaptiveTeacherFusion 输出的 T1、T2、T3。

        返回:
            标量张量 L_feat。

        异常:
            ValueError: 任一侧序列长度不为 3。
        """
        if len(student_aligned) != 3 or len(teacher_targets) != 3:
            raise ValueError("Need 3 feature levels for student and teacher.")
        device, dtype = student_aligned[0].device, student_aligned[0].dtype
        total = torch.zeros((), device=device, dtype=dtype)
        for i in range(3):
            w = self.lambdas[i]
            if float(w) <= 0.0:
                continue
            total = total + w * self.criterion(student_aligned[i], teacher_targets[i])
        return total


class NWDLoss(nn.Module):
    """
    归一化 Wasserstein 距离（NWD）定位损失。

    将预测框与真值框建模为轴对齐二维高斯（对角协方差），在仅使用中心 (cx,cy) 与宽高 (w,h) 时，
    使用专利中的 W_2^2 闭式近似，再定义:
        NWD = exp(-sqrt(W_2^2) / C)
        L_loc = 1 - NWD

    相比 IoU 类损失，在极小目标或弱重叠时仍可提供梯度，有利于古建细小缺陷检测训练。

    注意: 输入框坐标需与训练设定一致（通常为归一化到 [0,1] 的相对坐标）。
    """

    def __init__(self, c: float = 0.7, eps: float = 1e-6):
        """
        参数:
            c: 归一化尺度常数，与数据集/框尺度相关，需调参。
            eps: 防止 sqrt(0) 的数值稳定项。
        """
        super().__init__()
        self.c = c
        self.eps = eps

    def forward(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        计算一批框对的 NWD 定位损失。

        参数:
            pred_boxes: [..., 4]，每行为 (cx, cy, w, h)。
            gt_boxes: 与 pred_boxes 相同形状广播规则下的真值框，格式相同。
            mask: 可选，与 loss 逐元素相乘后做归一化求和；常用于只统计正样本 anchor。
                  形状需可与 loss 广播（例如 [N] 或 [N,1]）。

        返回:
            标量：若 mask 非空则为 sum(loss*mask)/(sum(mask)+eps)，否则为 loss.mean()。
            ------------------
            mask啥意思？？？？？？？？？？？？？？？？？？？
            ------------------
        """
        px, py, pw, ph = pred_boxes.unbind(-1)
        gx, gy, gw, gh = gt_boxes.unbind(-1)

        sigma_aw = pw / 2.0
        sigma_ah = ph / 2.0
        sigma_bw = gw / 2.0
        sigma_bh = gh / 2.0

        w2 = (px - gx) ** 2 + (py - gy) ** 2 + (sigma_aw - sigma_bw) ** 2 + (sigma_ah - sigma_bh) ** 2
        nwd = torch.exp(-torch.sqrt(w2 + self.eps) / self.c)
        loss = 1.0 - nwd

        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + self.eps)
        return loss.mean()


def total_loss(
    l_cls: torch.Tensor,
    l_loc: torch.Tensor,
    l_feat: torch.Tensor,
    l_dfl: torch.Tensor,
    weights: Sequence[float] = (1.0, 1.0, 0.5, 1.0),
) -> torch.Tensor:
    """
    将分类、定位、特征蒸馏、DFL 四项损失按权重线性组合为总损失。

    对应专利形式:
        L_total = λ1·L_cls + λ2·L_loc + λ3·L_feat + λ4·L_DFL

    参数:
        l_cls: 分类损失标量或已规约张量。
        l_loc: 定位损失（如 NWDLoss 输出）。
        l_feat: 特征蒸馏损失（FeatureDistillLoss 输出）。
        l_dfl: 分布焦点损失等 YOLO 头中的 DFL 项。
        weights: (w1, w2, w3, w4)，默认与实验方案中 λ 示例一致。

    返回:
        标量总损失，用于 backward。
    """
    w1, w2, w3, w4 = weights
    return w1 * l_cls + w2 * l_loc + w3 * l_feat + w4 * l_dfl
