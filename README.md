# Multi-Objective NAS for Vision Transformers on Small Datasets

**Master's Thesis** — Mats Adel, HAW Hamburg

> Evolutionary multi-objective Neural Architecture Search (NAS) for Vision Transformers on the Food11 dataset, benchmarked against CNN baselines and accelerated with zero-cost proxy filtering.

---

## Overview

Fine-tuning a large pretrained Vision Transformer on a small domain-specific dataset is non-trivial: the optimal architecture, fine-tuning strategy, and data augmentation depend heavily on dataset size and class structure. This thesis builds an evolutionary NAS framework that searches for Pareto-optimal ViT configurations with respect to **validation accuracy** and **parameter count**, and compares the found architectures against standard CNN baselines (ResNet-18, EfficientNet-B0, MobileNetV3-Large).

The framework extends a prior proof-of-concept (Hauptprojekt) by replacing a scalar fitness function with true multi-objective Pareto optimization (NSGA-II), adding zero-cost proxy pre-filtering to reduce search cost, and fixing broken weight inheritance. All experiments run on the [Food-11](https://www.epfl.ch/labs/mmspg/downloads/food-recognition-benchmark/) dataset.

---

## Research Questions

| | Question |
|---|---|
| **F1** | At what dataset size does a NAS-optimized Vision Transformer lose its accuracy advantage over CNN baselines at comparable parameter count? |
| **F2** | Under what conditions does NAS yield a measurable accuracy gain over CNN fine-tuning? |
| **F3** | Can zero-cost proxies as a pre-filter reduce NAS search cost without degrading Pareto-front quality? |

---

## Key Results

| Model | Params | Val-Acc |
|---|---|---|
| NAS top-1 (proxy-filtered) | 21.7 M | 96.7 % |
| NAS top-1 (unfiltered) | 21.7 M | 95.7 % |
| Smallest NAS-ViT beating all CNNs | 14.6 M | 92.9 % |
| EfficientNet-B0 (best CNN baseline) | 4.0 M | 92.2 % |
| ResNet-18 | 11.2 M | 90.2 % |
| MobileNetV3-Large | 4.2 M | 92.2 % |

**Proxy filtering** reduced training cost by ~47 % (120 → 56 evaluated architectures, 7.7 h → 4.0 h on H100) while improving Pareto-front quality and eliminating SPT artifacts from the front.

---

## NAS Framework

### Search Space

The NAS explores 9 hyperparameters simultaneously:

| Parameter | Options |
|---|---|
| `depth` | 4, 6, 8, 10, 12 |
| `num_heads` | 4, 6, 8 |
| `use_spt` | 0, 1 (Shifted Patch Tokenization) |
| `use_lsa` | 0, 1 (Local Self-Attention) |
| `finetune_mode` | head, last2, last4, full |
| `mixup_alpha` | 0.0, 0.2, 0.4 |
| `cutmix_prob` | 0.0, 0.5, 1.0 |
| `lr` | 1e-4, 3e-4, 1e-3 |
| `weight_decay` | 1e-4, 5e-4 |

Total search space: **8,640 possible configurations**. Backbone: `vit_small_patch16_224` pretrained on ImageNet-21k/1k via timm.

### Algorithm (NSGA-II)

1. **Initialization** — random population of 15 architectures
2. **Zero-cost proxy filter** *(optional)* — SynFlow + NASWOT + GradNorm rank ensemble, top 50 % advance
3. **Proxy training** — 12 epochs per candidate (vs. 30 for full training → ~60 % faster evaluation)
4. **NSGA-II selection** — non-dominated sorting + crowding distance, survivor pool of size 15
5. **Crossover** — uniform crossover across 9 parameters
6. **Mutation** — per-parameter probability 0.3, expected ~2.7 changes per child
7. **Weight inheritance** — child inherits parent weights when architectures are structurally compatible
8. Repeat for **8 generations**

### Zero-Cost Proxies

Three complementary proxies are combined into a rank ensemble (no training required):

| Proxy | Measures |
|---|---|
| **SynFlow** | Gradient flow through the network at initialization |
| **NASWOT** | Diversity of activation patterns (kernel matrix rank) |
| **GradNorm** | Gradient magnitude at initialization |

Proxy scores are converted to ranks per proxy, summed, and used to filter the bottom 50 % before any training occurs.

---

## Project Structure

```
src/
  models/
    vit.py              # ViT-Small with optional SPT & LSA, build_vit_from_cfg
    cnn_baseline.py     # ResNet-18, EfficientNet-B0, MobileNetV3-Large via timm
  nas/
    search_space.py     # Search space definition, random config sampling
    evolution.py        # Main NSGA-II loop
    pareto.py           # Dominance check, non-dominated sort, crowding distance
    crossover.py        # Uniform crossover operator
    mutation.py         # Per-parameter mutation
    evaluate.py         # Proxy-training evaluation of a single config
  proxies/
    synflow.py          # SynFlow implementation
    naswot.py           # NASWOT implementation
    gradnorm.py         # GradNorm implementation
    evaluate_proxy.py   # Rank ensemble of all three proxies
  utils/
    training.py         # Training loop with AMP, mixup, cutmix, early stopping
    data.py             # Food11 data loaders with subsample_frac support
    metrics.py          # Parameter count, accuracy
    seed.py             # Reproducibility, device selection

scripts/
  run_nas.py              # NAS search — with or without proxy filter
  run_f1_experiment.py    # Dataset-scaling: ViT-NAS + CNNs across data fractions
  run_proxy_validation.py # Spearman-ρ validation of proxy quality
  retrain_analysis.py     # Full retrain of best NAS architecture

notebooks/
  auswertung_ergebnisse.ipynb   # Full analysis: Pareto fronts, CNN comparison, plots

results/
  nas_baseline.csv      # 120 NAS architectures (unfiltered)
  nas_proxies.csv       # 56 NAS architectures (proxy-filtered)
  f1_scaling.csv        # ViT-Baseline + CNNs across 6 data fractions
  f1_scaling_nas.csv    # NAS top-1 across 6 data fractions
  proxy_validation.csv  # Proxy scores + val-acc for 30 random architectures
  plots/                # All figures used in the thesis

configs/                # Search space and experiment configurations
```

---

## Setup

### Local

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Docker (recommended for GPU cluster)

```bash
docker build -t nas-vit .

docker run --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  --shm-size=8g \
  -v /path/to/food11:/data/food11 \
  -v /path/to/results:/workspace/results \
  -e PYTHONUNBUFFERED=1 \
  nas-vit python scripts/run_nas.py \
    --data-root /data/food11 \
    --results-dir /workspace/results \
    --use-proxies
```

---

## Running Experiments

### NAS Search

```bash
# With proxy filter (recommended)
python scripts/run_nas.py --data-root data/food11 --results-dir results --use-proxies

# Without proxy filter (baseline)
python scripts/run_nas.py --data-root data/food11 --results-dir results
```

### Dataset-Scaling Experiment

```bash
python scripts/run_f1_experiment.py \
  --data-root data/food11 \
  --results-dir results \
  --epochs 30
```

### Proxy Validation

```bash
python scripts/run_proxy_validation.py \
  --data-root data/food11 \
  --results-dir results \
  --n-configs 30
```

---

## Dataset

**Food-11** — 16,643 images across 11 food categories, fixed train/val/test splits.

| Split | Images |
|---|---|
| Training | 9,866 |
| Validation | 3,430 |
| Evaluation | 3,347 |

Download from the [EPFL MMSPG page](https://www.epfl.ch/labs/mmspg/downloads/food-recognition-benchmark/) and place under `data/food11/` with subdirectories `training/`, `validation/`, `evaluation/`.

---

## Hardware

All experiments were run on the HAW Hamburg GPU cluster:

- NVIDIA H100 NVL (80 GB)
- NVIDIA A100-SXM4-40 GB (×4)
- PyTorch 2.5.1 · CUDA 12.4 · cuDNN 9
