# 简历项目：古建筑木构件裂缝细粒度分割（跨架构知识蒸馏）

> 数字均来自本项目**统一评测协议**（整图裁剪拼接 + 在验证集调阈值、在测试集报告，K=5 折、fold-0 hold-out）。

---

## 一句话概述

面向**古建筑木构件裂缝**的细粒度语义分割：以冻结的 **DINOv3 基础模型**为教师，设计**跨架构知识蒸馏**训练轻量学生，并建立统一评测；在少标注条件下逼近教师性能、并超越公开方法 **SMG-Net**。

---

## 详细版（研究 / 算法岗，6 条）

**古建筑木构件裂缝细粒度分割（跨架构知识蒸馏）** — 算法 / 研究工程师（独立完成）

- 面向裂缝“细长、方向多变、弱纹理、标注稀缺”的难点，构建 **K=5 折统一评测协议**，覆盖 **700+ 标注图**与 **1 万+ 无标注图**，保证教师 / 学生 / 基线同源划分、可比可复现。
- 设计**三阶段跨架构蒸馏框架（DINOv3 → 轻量学生）**：
  - **Stage A**：在无标注图上以 **iBOT 式自监督**适配冻结 ViT 基础模型（多尺度 adapter + 多尺度 dense head，**8 卡 DDP** 训练）；
  - **Stage B**：将其改造为 **FPN 密集教师**；
  - **Stage C**：以**多尺度特征蒸馏（Smooth-L1 + 注意力）+ logit KD + 离线软标签 KD** 训练轻量学生。
- 搭建端到端训练工程：滑窗训练 / 推理（1024 / stride 800）、copy-paste 裂缝合成、类权重 / 正样本采样 / 尺度增强、软标签离线缓存、AMP、断点续训与早停、k 折编排与并行调度。
- 建立**统一校准评估**（裁剪拼接整图 + 验证集调阈值 → 测试集报告 + 完整 P-R 曲线分析），纠正了“固定 0.5 阈值低估召回”的评测偏差，使教师 / 学生 / 第三方方法可在同一协议下对比。
- **结果**：教师 DINOv3+FPN 在 hold-out 折达 **0.72 IoU**；轻量学生从零训练达 **0.60 IoU（约 22M 参数）**，**优于复现的公开方法 SMG-Net（0.50 IoU，0.36M）**；并通过加权 CE / 注意力蒸馏系统调控 P-R 折衷，将**召回从 0.64 提升到 0.80**。
- 复现并对标 **PLOS One 2025 的 SMG-Net**（含论文级 640×480 裁剪协议与像素级指标），验证其在该任务整图协议下并不占优，为方法选型提供依据。

---

## 精简版（综合 / 工程岗，4 条）

- 面向古建筑木构件裂缝分割，独立搭建 **K=5 折少样本评测 + 滑窗训练 / 推理**全流程（700+ 标注、1 万+ 无标注）。
- 实现 **DINOv3 基础模型 → 轻量学生的跨架构知识蒸馏**（自监督适配 + FPN 教师 + 特征 / logit / 软标签多目标 KD），支持 **8 卡 DDP、AMP、断点续训**。
- 建立**统一校准评估与 P-R 曲线分析**，使不同架构、不同方法可公平对比。
- 轻量学生达 **0.60 IoU**，超过公开方法 **SMG-Net（0.50）**，逼近教师 **0.72**；系统优化召回（**0.64 → 0.80**）。

---

## English version（optional）

**Fine-grained crack segmentation for ancient timber structures (cross-architecture knowledge distillation)** — Research / ML Engineer (sole author)

- Built a K=5-fold few-shot benchmark (700+ labeled / 10k+ unlabeled images) for thin, multi-orientation, low-contrast cracks.
- Designed a 3-stage distillation pipeline: iBOT-style self-supervised adaptation of a frozen DINOv3 ViT (multi-scale adapters / dense heads, 8-GPU DDP), an FPN dense teacher, and a lightweight student trained with multi-scale feature + logit + offline soft-label KD.
- Engineered the full stack: sliding-window train / infer (1024 / stride 800), copy-paste synthesis, class weighting, positive-window sampling, scale augmentation, AMP, checkpointing / early stopping, k-fold orchestration.
- Built a unified calibrated evaluation (crop-stitch whole-image, val-tuned threshold, P-R curve analysis) and reproduced the PLOS One 2025 SMG-Net baseline.
- Teacher 0.72 IoU; lightweight student 0.60 IoU (~22M) beating SMG-Net 0.50; recall improved 0.64 → 0.80 via class-weighted CE / attention distillation.

---

## 使用建议

- **研究岗**：突出“跨架构蒸馏框架 + 统一评测 + 对标公开方法”；可加“论文撰写中 / 投稿 IEEE JCH”。
- **工程岗**：突出“8 卡 DDP、断点续训、k 折编排、统一评估工程化”。
- **诚实边界**：SMG-Net 的 0.81 是其**自有数据 / 裁剪口径**下的数字，**不要**写成自己的结果；本人数字为统一整图协议下的 0.60 / 0.72 / 0.50。
- 可量化补充：数据规模、参数量、训练卡数、KD 目标数、评测图数（144 张测试图）。
- 如有开源 / 论文链接，一并写上更加分。
