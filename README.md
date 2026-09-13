# 古建筑木构件裂缝分割 · 跨架构视觉大模型知识蒸馏（5 折交叉验证版）

面向少样本古建筑木构件裂缝分割的跨架构视觉大模型知识蒸馏方法复现实验包。

## 目录结构

```text
paper_repro/
  data/
    stage_a_external/        # Stage A 自监督无标签数据（Zenodo 外部池，非 labelme 标注数据）
    labelme/
      all/                   # 全部标注样本（json+image），5 折划分的数据源
  weights/
    dinov3-vitb16-pretrain-lvd1689m/
    yolo11m-seg.pt
  runs/
    kfold/                   # K 折实验产物
      folds.json             # 5 折划分清单（唯一事实来源）
      fold0/…fold4/          # 每折的 stage_b 教师 + 各学生实验 + eval
      reports/               # 均值±标准差汇总
    stage_a/                 # 全局 Stage A 自监督训练产物
  solution/                  # 训练/评估脚本
  scripts/                   # shell 包装与 K 折编排器
```

## 实验矩阵（每折 K=5）

| ID | 教师 | 蒸馏特征来源 | 蒸馏损失 |
|----|------|--------------|----------|
| S0 | 无 | — | — |
| S1 | 裸 DINOv3（无 A/B） | h3‖h4, h7‖h8, h11‖h12 拼接 | feat |
| S1_attn | 裸 DINOv3 | 同 S1 | feat + attn |
| S2 | Stage A（仅 adapter，无 fuse/bridge） | adapter 输出拼接 | feat |
| S2_attn | Stage A | 同 S2 | feat + attn |
| S3 | Stage A + B（Stage B 冻结 adapter） | bridge_low/mid/deep | feat |
| S3_attn | Stage A + B | bridge_low/mid/deep | feat + attn |

- **S4（copy-paste）已删除**，copy-paste 增益实验暂不开展。
- 蒸馏对齐：`S1/S2` 将相邻教师层拼接后与 P3/P4/P5 对齐（通道 2×768=1536）；`S3` 将
  `bridge_low(128)/mid(192)/deep(256)` 映射到 P3/P4/P5。
- 教师特征来源（`dino_stage_a.py::extract_adapted_feature_maps`、
  `dino_stage_b_unet.py::extract_bridge_feature_maps`）。

### 架构对比基线（单一变量：只换模型）

与 S0（YOLO11-U-Net）对比的基线，**除学生架构外全部与 S0 相同**（同 CE+Dice 损失、AdamW 超参、
100 epoch + val-IoU 早停、同 K-fold、同数据/增广/窗口/步长），且**全部 from scratch 初始化**：

| ID | 学生架构 | 说明 |
|----|----------|------|
| S0 | `yolo_unet` | YOLO11m backbone/neck + U-Net 解码头（预训练 `yolo11m-seg.pt`） |
| YOLOSeg | `yolo_seg` | 原生 YOLO11m-seg（backbone/neck/Segment head），从 `yolo11m-seg.yaml` 随机初始化，用 mask-prototype 聚合出语义裂缝 logits |
| UNet | `unet` | 标准 U-Net（base=64），随机初始化 |
| DeepLab | `deeplab` | torchvision DeepLabV3-ResNet50（`weights=None`），随机初始化 |

- 由 `solution/models/student_factory.py::build_student` 统一构建；训练脚本用 `--student-arch` 选择，
  评估脚本按 checkpoint 里的 `student_arch` 自动分发。
- 通过 `EXPERIMENT_FILTER=YOLOSeg,UNet,DeepLab`（或与 S0 一起）运行，报告自动累加。
- **DeepLab 需要 batch ≥ 2**（ASPP 全局池化含 BatchNorm；`KFOLD_BATCH_SIZE` 默认 2，满足）。
- 依赖：`torchvision`（DeepLab）与 `ultralytics`（YOLOSeg/Unet），均已随环境安装。

## K 折协议

- 在 `data/labelme/all` 上做 **K=5** 确定性划分（seed 42），清单缓存于 `runs/kfold/folds.json`，
  教师训练、学生蒸馏、评估三者共享同一份划分。
- 对每一折 `f`（0..4）：测试集 = 折 f；训练集 = 其余 4 折；再从训练集内划出 **~10%** 作为
  早停验证集。
- 教师-学生对应关系：`S3/S3_attn` 使用「同折的 Stage B 教师」蒸馏学生（教师与学生使用相同的
  4 折训练）；`S1/S1_attn/S2/S2_attn` 教师与折无关（裸 DINOv3 / 全局 Stage A）。
- Stage A 为全局一次性自监督训练，数据来自外部无标签池 `data/stage_a_external/`。

## 训练协议（统一）

- 最大轮次 `--epochs 100`，**早停基于验证集裂缝 IoU**（patience 10），best.pt 按 val IoU 选取。
- 训练数据每轮以 0.5 概率施加简单几何变换（90° 整数倍旋转 + 水平翻转），Stage B/C 共用
  `solution/augment.py`。
- Stage B 教师训练 **冻结 adapter**（仅训练 decoder / PairAdaptiveFuse / bridge），保留
  Stage A 学到的领域特征。

## 安装（WSL）

```bash
bash scripts/setup_wsl.sh
```

## 运行 5 折实验

```bash
bash scripts/run_kfold.sh
```

编排器 `scripts/run_kfold_experiments.py` 依次完成：建折 → Stage A（全局，按需）→ 每折 Stage B（按需）→
每折所选学生实验 → 每折评估 → 汇总 `mean ± std` 报告（`runs/kfold/reports/`）。
各步骤幂等：已存在的 checkpoint / metrics 会被复用。

### 选择性运行 `EXPERIMENT_FILTER`

用 `EXPERIMENT_FILTER`（逗号分隔的实验 ID）只跑部分实验；教师阶段会自动按需执行：

- `S0`：无教师 → **跳过 Stage A 与 Stage B**。
- `S1` / `S1_attn`：裸 DINOv3 教师 → 跳过 Stage A/B。
- `S2` / `S2_attn`：Stage A 教师 → 只跑 Stage A。
- `S3` / `S3_attn`：Stage A+B 教师 → 跑 Stage A 与每折 Stage B。

只跑 S0（服务器示例，数据在外部卷）：

```bash
cd <PROJECT_ROOT>
DATA_ROOT=/mnt/volume3/home/zgt/data \
EXPERIMENT_FILTER=S0 \
YOLO_WEIGHTS=weights/yolo11m-seg.pt \
NUM_WORKERS=4 PREFETCH_FACTOR=2 \
bash scripts/run_kfold.sh
```

- `LABELME_ALL_DIR` 由 `DATA_ROOT` 推导为 `$DATA_ROOT/labelme/all`。
- 评估会在**现有 checkpoint 集合扩大时自动重跑**并累加进报告，因此可以分批运行（先 S0，后
  `EXPERIMENT_FILTER=S0,S1,S1_attn,S2,S2_attn,S3,S3_attn` 补齐），不会重复训练已完成的模型。

常用环境变量：

```bash
DATA_ROOT=/mnt/volume3/home/zgt/data \
RUNS_ROOT=$PWD/runs \
KFOLD_K=5 KFOLD_SEED=42 KFOLD_MAX_EPOCHS=100 KFOLD_EARLY_STOP_PATIENCE=10 \
KFOLD_VAL_RATIO=0.10 KFOLD_VAL_STRIDE=512 KFOLD_BATCH_SIZE=2 \
NUM_WORKERS=4 PREFETCH_FACTOR=2 \
EXPERIMENT_FILTER=S0 \
bash scripts/run_kfold.sh
```

## 单折/单模型调试

直接调用训练脚本（以折 0、S3 为例）：

```bash
python solution/train_seg_stage_c_mixed.py \
  --curated-labelme-dir data/labelme/all --images-dir data/labelme/all \
  --student-weights weights/yolo11m-seg.pt \
  --teacher-mode stage_b --teacher-weights weights/dinov3-vitb16-pretrain-lvd1689m \
  --teacher-stage-b-ckpt runs/kfold/fold0/stage_b/best.pt \
  --output-dir runs/kfold/fold0/S3 \
  --fold-split runs/kfold/folds.json --fold 0 \
  --epochs 100 --early-stop-patience 10 --val-ratio 0.10 \
  --lambda-feat-curated 0.5 --lambda-attn-curated 0.0
```

Stage B 教师（冻结 adapter）：

```bash
python solution/train_teacher_stage_b.py \
  --labelme-dir data/labelme/all --images-dir data/labelme/all \
  --teacher-weights weights/dinov3-vitb16-pretrain-lvd1689m \
  --stage-a-ckpt runs/stage_a/stage_a_last.pt \
  --output-dir runs/kfold/fold0/stage_b \
  --fold-split runs/kfold/folds.json --fold 0 \
  --labelme-sliding-window --epochs 100 --early-stop-patience 10
```

## 评估

```bash
python solution/scripts/eval_labelme_crack_test.py \
  --labelme-dir data/labelme/all \
  --fold-split runs/kfold/folds.json --fold 0 \
  --stage-b-ckpt runs/kfold/fold0/stage_b/best.pt \
  --teacher-weights weights/dinov3-vitb16-pretrain-lvd1689m \
  --student-ckpt S3=runs/kfold/fold0/S3/best.pt
```

## 数据假设

LabelMe 标注语义：

- `crack`：前景裂缝像素。
- `component` / `wood`：有效构件区域。
- `ignore`：损失中排除的不可靠区域。
- 其余（`defect`/`knot`/`decay` 等）作为 copy-paste 的遮挡区域（当前 copy-paste 未启用）。
- `ncp`：该样本禁用 copy-paste（保留为未来选项）。

有效区域 `V = component ∧ ¬ignore`，早停与最终评估均在此区域上计算裂缝 IoU。
