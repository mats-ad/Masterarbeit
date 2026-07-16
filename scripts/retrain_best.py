"""
Beste NAS-Architektur vollständig trainieren (50 Epochen).
Config manuell aus nas_run.csv übernehmen.
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seed import set_seed, get_device
from src.utils.data import make_food11_loaders
from src.utils.training import run_training
from src.utils.metrics import count_params, accuracy_top1
from src.models.vit import build_vit_from_cfg
import torch
import json


def parse_args():
    p = argparse.ArgumentParser(description="Beste Architektur vollständig trainieren")
    p.add_argument("--data-root",    default="/mnt/data/mats_adel/food11")
    p.add_argument("--results-dir",  default="/mnt/data/mats_adel/results")
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--seed",         type=int,   default=42)
    # Architektur-Parameter (aus nas_run.csv übernehmen)
    p.add_argument("--depth",        type=int,   default=12)
    p.add_argument("--num-heads",    type=int,   default=6)
    p.add_argument("--use-spt",      type=int,   default=0)
    p.add_argument("--use-lsa",      type=int,   default=0)
    p.add_argument("--finetune-mode",default="last2")
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mixup-alpha",  type=float, default=0.0)
    p.add_argument("--cutmix-prob",  type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)

    cfg = {
        "depth":        args.depth,
        "num_heads":    args.num_heads,
        "use_spt":      args.use_spt,
        "use_lsa":      args.use_lsa,
        "finetune_mode":args.finetune_mode,
        "lr":           args.lr,
        "weight_decay": args.weight_decay,
        "mixup_alpha":  args.mixup_alpha,
        "cutmix_prob":  args.cutmix_prob,
    }

    print(f"Device: {device}")
    print(f"Config: {cfg}\n")

    train_loader, val_loader, test_loader, num_classes = make_food11_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=4,
        seed=args.seed,
    )

    model = build_vit_from_cfg(num_classes, cfg)
    print(f"Parameter: {count_params(model)/1e6:.2f}M\n")

    best_val, history = run_training(
        model, train_loader, val_loader, device,
        epochs=args.epochs,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        mixup_alpha=cfg["mixup_alpha"],
        cutmix_prob=cfg["cutmix_prob"],
        patience=15,
    )

    # Test-Set Evaluation
    from src.utils.training import evaluate
    test_loss, test_acc = evaluate(model, test_loader, device)

    print(f"\nBest Val Acc:  {best_val:.4f}")
    print(f"Test Acc:      {test_acc:.4f}")

    # Modell + History speichern
    run_name = f"best_arch_d{cfg['depth']}_h{cfg['num_heads']}_spt{cfg['use_spt']}"
    torch.save(model.state_dict(),
               os.path.join(args.results_dir, f"{run_name}.pth"))
    with open(os.path.join(args.results_dir, f"{run_name}_history.json"), "w") as f:
        json.dump({"cfg": cfg, "history": history,
                   "best_val_acc": best_val, "test_acc": test_acc}, f, indent=2)

    print(f"\nGespeichert: {args.results_dir}/{run_name}.*")


if __name__ == "__main__":
    main()
