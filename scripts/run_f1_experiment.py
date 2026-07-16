"""
F1-Experiment: Wie wirkt sich die Datenmenge auf ViT vs. CNN aus?
Trainiert ViT-Small und CNN-Baselines auf verschiedenen subsample_frac Werten.
"""
import argparse
import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seed import set_seed, get_device
from src.utils.data import make_food11_loaders
from src.utils.training import run_training
from src.utils.metrics import count_params
from src.models.vit import build_vit_from_cfg
from src.models.cnn_baseline import build_cnn_baseline


FRACS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

VIT_CFG = {
    "depth": 12, "num_heads": 6, "use_spt": 0, "use_lsa": 0,
    "finetune_mode": "last2", "lr": 3e-4, "weight_decay": 1e-4,
    "mixup_alpha": 0.0, "cutmix_prob": 0.0,
}

CNN_MODELS = [
    ("resnet18",         "resnet18.a1_in1k"),
    ("efficientnet_b0",  "efficientnet_b0.ra_in1k"),
    ("mobilenetv3_large","mobilenetv3_large_100.ra_in1k"),
]


def parse_args():
    p = argparse.ArgumentParser(description="F1 Dataset-Scaling Experiment")
    p.add_argument("--data-root",   default="/mnt/data/mats_adel/food11")
    p.add_argument("--results-dir", default="/mnt/data/mats_adel/results")
    p.add_argument("--epochs",      type=int, default=30)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--models",      nargs="+",
                   default=["vit"] + [name for name, _ in CNN_MODELS],
                   help="Welche Modelle trainieren (vit, resnet18, ...)")
    return p.parse_args()


def train_model(model, train_loader, val_loader, device, lr, epochs):
    best_val, history = run_training(
        model, train_loader, val_loader, device,
        epochs=epochs, lr=lr, weight_decay=1e-4, patience=10,
    )
    return best_val


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)
    log_path = os.path.join(args.results_dir, "f1_scaling.csv")

    print(f"Device: {device}")
    print(f"Fractions: {FRACS} | Epochs: {args.epochs}\n")

    fieldnames = ["model", "subsample_frac", "train_size", "params_M",
                  "best_val_acc", "epochs"]

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for frac in FRACS:
            train_loader, val_loader, _, num_classes = make_food11_loaders(
                root=args.data_root,
                batch_size=args.batch_size,
                num_workers=4,
                seed=args.seed,
                subsample_frac=frac,
            )
            train_size = len(train_loader.dataset)
            print(f"\n=== subsample_frac={frac} | {train_size} Trainingsbilder ===")

            # ViT
            if "vit" in args.models:
                print(f"  Training ViT...")
                model = build_vit_from_cfg(num_classes, VIT_CFG)
                best_val = train_model(model, train_loader, val_loader, device,
                                       lr=VIT_CFG["lr"], epochs=args.epochs)
                row = {"model": "vit_small", "subsample_frac": frac,
                       "train_size": train_size, "params_M": count_params(model)/1e6,
                       "best_val_acc": best_val, "epochs": args.epochs}
                writer.writerow(row)
                f.flush()
                print(f"  ViT → val_acc={best_val:.4f}")

            # CNNs
            for name, timm_name in CNN_MODELS:
                if name not in args.models:
                    continue
                print(f"  Training {name}...")
                model = build_cnn_baseline(timm_name, num_classes)
                best_val = train_model(model, train_loader, val_loader, device,
                                       lr=1e-3, epochs=args.epochs)
                row = {"model": name, "subsample_frac": frac,
                       "train_size": train_size, "params_M": count_params(model)/1e6,
                       "best_val_acc": best_val, "epochs": args.epochs}
                writer.writerow(row)
                f.flush()
                print(f"  {name} → val_acc={best_val:.4f}")

    print(f"\nFertig. Ergebnisse: {log_path}")


if __name__ == "__main__":
    main()
