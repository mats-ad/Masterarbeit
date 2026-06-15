# Multi-Objective NAS for Vision Transformers on Small Datasets

Master's Thesis — Mats Adel

## Research Questions

- **F1** — Can a NAS-optimized ViT outperform CNN baselines (MobileNet, EfficientNet, ResNet) on small domain-specific datasets at comparable parameter counts?
- **F2** — How close can NAS get to SOTA with fewer parameters?
- **F3** — When does NAS justify its search cost over simple fine-tuning?

## Project Structure

```
src/
  models/     # ViT building blocks: SPT, LSA, build_vit_from_cfg
  nas/        # Evolutionary NAS: search space, mutation, crossover, Pareto selection
  proxies/    # Zero-cost proxies: SynFlow, NASWOT, GradNorm
  utils/      # Training loops, data loaders, metrics, evaluation framework

notebooks/
  baseline_hauptprojekt.ipynb   # Original Hauptprojekt (reference)

results/
  hauptprojekt/   # NAS results from Hauptprojekt (CIFAR-100, EuroSAT)

configs/          # Search space and experiment configurations

data/             # Datasets (gitignored, download separately)
```

## Baseline (Hauptprojekt)

Built in Semester 3 as a proof of concept. Key findings:
- Scalar fitness function systematically penalized SPT via time penalty
- Weight inheritance was designed but not correctly wired
- No comparison against CNN baselines

This thesis fixes all of the above and adds Pareto optimization + zero-cost proxies.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
