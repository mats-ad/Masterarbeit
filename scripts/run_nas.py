import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seed import set_seed, get_device
from src.utils.data import make_food11_loaders
from src.nas.evolution import run_evolution


def parse_args():
    p = argparse.ArgumentParser(description="NAS Run für ViT auf Food11")
    p.add_argument("--data-root",      default="/mnt/data/mats_adel/food11")
    p.add_argument("--results-dir",    default="/mnt/data/mats_adel/results")
    p.add_argument("--run-name",       default="nas_run")
    p.add_argument("--pop-size",       type=int,   default=15)
    p.add_argument("--generations",    type=int,   default=8)
    p.add_argument("--proxy-epochs",   type=int,   default=12)
    p.add_argument("--use-proxies",    action="store_true")
    p.add_argument("--proxy-keep-frac",type=float, default=0.5)
    p.add_argument("--batch-size",     type=int,   default=32)
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    os.makedirs(args.results_dir, exist_ok=True)
    log_path = os.path.join(args.results_dir, f"{args.run_name}.csv")

    print(f"Device: {device}")
    print(f"Config: pop={args.pop_size} | gen={args.generations} | "
          f"proxy_epochs={args.proxy_epochs} | use_proxies={args.use_proxies}")
    print(f"Log: {log_path}\n")

    train_loader, val_loader, _, num_classes = make_food11_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=4,
        seed=args.seed,
    )

    results = run_evolution(
        num_classes=num_classes,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        pop_size=args.pop_size,
        generations=args.generations,
        proxy_epochs=args.proxy_epochs,
        seed=args.seed,
        log_path=log_path,
        use_proxies=args.use_proxies,
        proxy_keep_frac=args.proxy_keep_frac,
    )

    print(f"\nFertig — {len(results)} Architekturen evaluiert.")
    print(f"Ergebnisse gespeichert: {log_path}")


if __name__ == "__main__":
    main()
