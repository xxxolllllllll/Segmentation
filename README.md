## Clone Repo & Get Assests 

git clone https://github.com/xxxolllllllll/Segmentation.git  
  
cd Segmentation  

bash scripts/download_assets.sh  


## Linux/WSL Repro Package

This directory is a cleaned, Linux/WSL-oriented experiment package for reproducing the small-paper training and evaluation chain. It keeps only the Stage A / Stage B / Stage C / evaluation code needed for the paper and standardizes paths around `data/`, `weights/`, and `runs/`.

## Standard Layout

Use the following layout under `paper_repro/`:

```text
paper_repro/
  data/
    stage_a/                  # raw images or LabelMe JSON roots for Stage A
    labelme/
      curated/                # optional alias; in your current setup this equals trainval
      trainval/               # LabelMe train+val JSONs and their images
      test/                   # LabelMe test JSONs and their images
  weights/
    dinov3-vitb16-pretrain-lvd1689m/
    yolo11m-seg.pt
  runs/
  solution/
  scripts/
```

The shell wrappers default to this layout, but you can override paths with environment variables such as `DATA_ROOT`, `WEIGHTS_ROOT`, `RUNS_ROOT`, `LABELME_CURATED_DIR`, `LABELME_TRAINVAL_DIR`, and `LABELME_TEST_DIR`. If `LABELME_CURATED_DIR` is not set, it falls back to `LABELME_TRAINVAL_DIR`.

## Paper Experiment IDs

The wrappers in this folder use the paper's final numbering:

- `S0`: student-only baseline, no teacher distillation.
- `S1`: raw `DINOv3` teacher feature distillation.
- `S2`: Stage A + Stage B adapted teacher feature distillation.
- `S3`: `S2` plus crack-attention distillation.
- `S4`: `S3` plus student-side Copy-Paste augmentation.

Legacy mapping in the original project:

- paper `S2` corresponds to the old code/output naming `S3`.
- paper `S3` corresponds to the old code/output naming `S4`.
- paper `S4` corresponds to the old code/output naming `S5`.
- the old "stage-B-only teacher" ablation is kept as an optional `generic` Stage B run, but it is not part of the final paper numbering.

## Install on WSL

From `paper_repro/`, run:

```bash
bash scripts/setup_wsl.sh
```

The setup script installs `torch`/`torchvision` with CUDA wheels first, then installs the remaining Python dependencies from `requirements-linux.txt`.

## Quick Throughput Benchmark

To verify that WSL gives better Stage C throughput than Windows:

```bash
MAX_STEPS=200 NUM_WORKERS=4 bash scripts/benchmark_stage_c.sh
```

This benchmark uses the same Stage C configuration as the adapted-teacher feature-distillation setting, so it is a good proxy for the paper's main training bottleneck.

## Training Entry Points

```bash
bash scripts/run_stage_a.sh
bash scripts/run_stage_b.sh full
bash scripts/run_stage_c_suite.sh S0
bash scripts/run_stage_c_suite.sh S1
bash scripts/run_stage_c_suite.sh S2
bash scripts/run_stage_c_suite.sh S3
bash scripts/run_stage_c_suite.sh S4
bash scripts/run_stagec_repeat_stability.sh
```

Optional Stage B ablation without Stage A initialization:

```bash
bash scripts/run_stage_b.sh generic
```

The Stage B / Stage C wrappers forward any extra CLI arguments to the underlying Python script, so short smoke runs such as `bash scripts/run_stage_b.sh generic --max-steps 2 --max-val-batches 1` are supported.

The repeat-stability runner reuses the paper's existing seed-42 checkpoints, resumes any copied `last.pt` runs for seeds `52/62`, evaluates each seed on the test set, and writes `mean +- std` reports under `runs/repeat_stagec_stability/reports/`.

## Evaluation

To evaluate available Stage C checkpoints on the independent test set:

```bash
bash scripts/eval_students.sh
```

## Data Assumptions

The LabelMe-based Stage B / Stage C pipeline assumes the following label semantics:

- `crack`: foreground crack pixels.
- `component` or `wood`: valid component region.
- `ignore`: unreliable region excluded from loss.
- `ncp`: disables Copy-Paste for that image but is not a prediction class.

If your Stage A inputs live outside `data/stage_a/`, set `STAGE_A_ROOT` before running `scripts/run_stage_a.sh`.
