"""
Proxy-Validation Experiment (F2):
Für N zufällige Architekturen: Proxy-Scores messen + trainieren.
Korrelation zwischen Proxy-Score und Val-Accuracy zeigt ob Proxies sinnvoll sind.
"""
import argparse
import csv
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seed import set_seed, get_device
from src.utils.data import make_food11_loaders
from src.nas.search_space import sample_config
from src.nas.evaluate import evaluate_cfg
from src.proxies.evaluate_proxy import evaluate_proxies


def spearman_r(x, y):
    """Spearman Rang-Korrelation ohne scipy."""
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return 1 - 6 * np.sum((rx - ry) ** 2) / (n * (n ** 2 - 1))


def parse_args():
    p = argparse.ArgumentParser(description="Proxy-Validation Experiment")
    p.add_argument("--data-root",    default="/mnt/data/mats_adel/food11")
    p.add_argument("--results-dir",  default="/mnt/data/mats_adel/results")
    p.add_argument("--n-configs",    type=int, default=30)
    p.add_argument("--proxy-epochs", type=int, default=12)
    p.add_argument("--batch-size",   type=int, default=32)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)
    log_path = os.path.join(args.results_dir, "proxy_validation.csv")

    print(f"Device: {device}")
    print(f"Configs: {args.n_configs} | Epochs: {args.proxy_epochs}\n")

    train_loader, val_loader, _, num_classes = make_food11_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=4,
        seed=args.seed,
    )
    proxy_batch, proxy_labels = next(iter(train_loader))

    fieldnames = [
        "config_id", "depth", "num_heads", "use_spt", "use_lsa",
        "finetune_mode", "lr", "weight_decay", "mixup_alpha", "cutmix_prob",
        "synflow", "naswot", "gradnorm", "params", "val_acc", "time_sec",
    ]

    configs = [sample_config() for _ in range(args.n_configs)]

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, cfg in enumerate(configs):
            print(f"[{i+1:02d}/{args.n_configs}] Proxy-Scores messen...")
            proxy_scores = evaluate_proxies(cfg, num_classes, proxy_batch, proxy_labels, device)

            print(f"  synflow={proxy_scores['synflow']:.2f} | "
                  f"naswot={proxy_scores['naswot']:.2f} | "
                  f"gradnorm={proxy_scores['gradnorm']:.2f}")
            print(f"  Training ({args.proxy_epochs} Epochen)...")

            result = evaluate_cfg(
                cfg, num_classes, train_loader, val_loader, device,
                proxy_epochs=args.proxy_epochs,
            )

            print(f"  val_acc={result['val_acc']:.4f} | params={result['params']/1e6:.1f}M\n")

            row = {
                "config_id":    i,
                "synflow":      proxy_scores["synflow"],
                "naswot":       proxy_scores["naswot"],
                "gradnorm":     proxy_scores["gradnorm"],
                "params":       result["params"],
                "val_acc":      result["val_acc"],
                "time_sec":     result["time_sec"],
                **{k: cfg[k] for k in ["depth", "num_heads", "use_spt", "use_lsa",
                                        "finetune_mode", "lr", "weight_decay",
                                        "mixup_alpha", "cutmix_prob"]},
            }
            writer.writerow(row)
            f.flush()

    # Korrelation ausgeben
    import numpy as np
    data = [row for row in csv.DictReader(open(log_path))]
    val_accs  = np.array([float(r["val_acc"])  for r in data])
    synflows  = np.array([float(r["synflow"])  for r in data])
    naswots   = np.array([float(r["naswot"])   for r in data])
    gradnorms = np.array([float(r["gradnorm"]) for r in data])

    print("=== Spearman-Korrelation mit val_acc ===")
    print(f"  SynFlow:  ρ = {spearman_r(synflows,  val_accs):.3f}")
    print(f"  NASWOT:   ρ = {spearman_r(naswots,   val_accs):.3f}")
    print(f"  GradNorm: ρ = {spearman_r(gradnorms, val_accs):.3f}")
    print(f"\nErgebnisse: {log_path}")


if __name__ == "__main__":
    main()
