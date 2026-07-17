"""
Trainiert die Top-N und Bottom-N Architekturen aus einem NAS-Lauf vollständig.
Zeigt ob die Proxy-Epoch-Rangliste nach vollem Training stabil bleibt.
"""
import argparse
import csv
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.seed import set_seed, get_device
from src.utils.data import make_food11_loaders
from src.utils.training import run_training, evaluate
from src.utils.metrics import count_params
from src.models.vit import build_vit_from_cfg


CONFIG_KEYS = ["depth", "num_heads", "use_spt", "use_lsa",
               "finetune_mode", "lr", "weight_decay", "mixup_alpha", "cutmix_prob"]


def parse_args():
    p = argparse.ArgumentParser(description="Top/Bottom Architekturen vollständig trainieren")
    p.add_argument("--nas-csv",      required=True, help="z.B. results/nas_baseline.csv")
    p.add_argument("--data-root",    default="/mnt/data/mats_adel/food11")
    p.add_argument("--results-dir",  default="/mnt/data/mats_adel/results")
    p.add_argument("--top-n",        type=int, default=3)
    p.add_argument("--bottom-n",     type=int, default=3)
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--batch-size",   type=int, default=32)
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def load_candidates(csv_path, top_n, bottom_n):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        row["val_acc"] = float(row["val_acc"])
        for k in ["depth", "num_heads", "use_spt", "use_lsa"]:
            row[k] = int(row[k])
        for k in ["lr", "weight_decay", "mixup_alpha", "cutmix_prob"]:
            row[k] = float(row[k])

    sorted_rows = sorted(rows, key=lambda r: r["val_acc"], reverse=True)
    top    = sorted_rows[:top_n]
    bottom = sorted_rows[-bottom_n:]
    return top, bottom


def train_candidate(cfg, label, num_classes, train_loader, val_loader,
                    test_loader, device, epochs, results_dir):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  depth={cfg['depth']} heads={cfg['num_heads']} "
          f"spt={cfg['use_spt']} lsa={cfg['use_lsa']} ft={cfg['finetune_mode']}")
    print(f"  Proxy val_acc={cfg['val_acc']:.4f}")
    print(f"{'='*50}")

    model = build_vit_from_cfg(num_classes, cfg)
    print(f"  Parameter: {count_params(model)/1e6:.2f}M")

    best_val, history = run_training(
        model, train_loader, val_loader, device,
        epochs=epochs,
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        mixup_alpha=float(cfg["mixup_alpha"]),
        cutmix_prob=float(cfg["cutmix_prob"]),
        patience=15,
    )

    _, test_acc = evaluate(model, test_loader, device)

    print(f"\n  Proxy val_acc:    {cfg['val_acc']:.4f}")
    print(f"  Full val_acc:     {best_val:.4f}")
    print(f"  Test acc:         {test_acc:.4f}")

    result = {
        "label":          label,
        "proxy_val_acc":  cfg["val_acc"],
        "full_val_acc":   best_val,
        "test_acc":       test_acc,
        "params_M":       count_params(model) / 1e6,
        "cfg":            {k: cfg[k] for k in CONFIG_KEYS},
        "history":        history,
    }

    safe_label = label.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(results_dir, f"retrain_{safe_label}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Lade Kandidaten aus: {args.nas_csv}")

    top, bottom = load_candidates(args.nas_csv, args.top_n, args.bottom_n)

    print(f"\nTop {args.top_n}:")
    for r in top:
        print(f"  val_acc={r['val_acc']:.4f} | depth={r['depth']} heads={r['num_heads']}")
    print(f"\nBottom {args.bottom_n}:")
    for r in bottom:
        print(f"  val_acc={r['val_acc']:.4f} | depth={r['depth']} heads={r['num_heads']}")

    train_loader, val_loader, test_loader, num_classes = make_food11_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=4,
        seed=args.seed,
    )

    all_results = []

    for i, cfg in enumerate(top):
        result = train_candidate(cfg, f"top_{i+1}", num_classes,
                                 train_loader, val_loader, test_loader,
                                 device, args.epochs, args.results_dir)
        all_results.append(result)

    for i, cfg in enumerate(bottom):
        result = train_candidate(cfg, f"bottom_{i+1}", num_classes,
                                 train_loader, val_loader, test_loader,
                                 device, args.epochs, args.results_dir)
        all_results.append(result)

    print(f"\n{'='*50}")
    print("ZUSAMMENFASSUNG")
    print(f"{'='*50}")
    print(f"{'Label':<12} {'Proxy':>8} {'Full Val':>10} {'Test':>8}")
    print("-" * 42)
    for r in all_results:
        print(f"{r['label']:<12} {r['proxy_val_acc']:>8.4f} "
              f"{r['full_val_acc']:>10.4f} {r['test_acc']:>8.4f}")

    summary_path = os.path.join(args.results_dir, "retrain_analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nGespeichert: {summary_path}")


if __name__ == "__main__":
    main()
