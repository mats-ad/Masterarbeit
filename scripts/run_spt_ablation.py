"""
SPT-Ablation: Vergleich von SPT=0, SPT=1 (random init), SPT=1 (warm-start init)
für eine fixe Architektur (typischerweise die beste aus dem NAS-Lauf).
Testet ob SPT im Fine-Tuning-Setting mit pretrained Gewichten funktioniert.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.seed import set_seed, get_device
from src.utils.data import make_food11_loaders
from src.utils.training import run_training, evaluate
from src.utils.metrics import count_params
from src.models.vit import build_vit_from_cfg

CONDITIONS = [
    {"label": "spt_none",   "use_spt": 0, "warm_start": False},
    {"label": "spt_random", "use_spt": 1, "warm_start": False},
    {"label": "spt_warm",   "use_spt": 1, "warm_start": True},
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-json", help="Pfad zu retrain_top_1.json (liest cfg automatisch)")
    # Fallback: manuelle Angabe
    p.add_argument("--depth",         type=int,   default=12)
    p.add_argument("--num-heads",     type=int,   default=6)
    p.add_argument("--finetune-mode", default="last2")
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--weight-decay",  type=float, default=1e-4)
    p.add_argument("--mixup-alpha",   type=float, default=0.0)
    p.add_argument("--cutmix-prob",   type=float, default=0.0)
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--data-root",     default="/mnt/data/mats_adel/food11")
    p.add_argument("--results-dir",   default="/mnt/data/mats_adel/results")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--batch-size",    type=int,   default=32)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)

    # Config aus JSON laden oder aus CLI-Args bauen
    if args.config_json:
        with open(args.config_json) as f:
            base_cfg = json.load(f)["cfg"]
    else:
        base_cfg = {
            "depth": args.depth, "num_heads": args.num_heads,
            "finetune_mode": args.finetune_mode, "lr": args.lr,
            "weight_decay": args.weight_decay, "mixup_alpha": args.mixup_alpha,
            "cutmix_prob": args.cutmix_prob,
        }

    train_loader, val_loader, test_loader, num_classes = make_food11_loaders(
        root=args.data_root, batch_size=args.batch_size,
        num_workers=4, seed=args.seed,
    )

    all_results = []
    for cond in CONDITIONS:
        cfg = {**base_cfg, "use_spt": cond["use_spt"], "use_lsa": 0}
        print(f"\n{'='*50}\n  {cond['label']}  (warm_start={cond['warm_start']})\n{'='*50}")

        model = build_vit_from_cfg(num_classes, cfg,
                                   spt_warm_start=cond["warm_start"])
        best_val, history = run_training(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
            mixup_alpha=cfg["mixup_alpha"], cutmix_prob=cfg["cutmix_prob"],
            patience=15,
        )
        _, test_acc = evaluate(model, test_loader, device)
        print(f"  val_acc={best_val:.4f}  test_acc={test_acc:.4f}")

        result = {"label": cond["label"], "val_acc": best_val,
                  "test_acc": test_acc, "history": history}
        all_results.append(result)

        out = os.path.join(args.results_dir, f"spt_ablation_{cond['label']}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)

    print(f"\n{'Label':<14} {'Val':>8} {'Test':>8}")
    print("-" * 32)
    for r in all_results:
        print(f"{r['label']:<14} {r['val_acc']:>8.4f} {r['test_acc']:>8.4f}")

    with open(os.path.join(args.results_dir, "spt_ablation_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()