import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.metrics import accuracy_top1
import numpy as np

def _rand_bbox(size, lam):
    """Zufällige Bounding Box für CutMix."""
    W, H = size[2], size[3]
    cut_ratio = (1.0 - lam) ** 0.5
    cut_w, cut_h = int(W * cut_ratio), int(H * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2

def train_one_epoch(model, loader, optimizer, scaler, device,
                    mixup_alpha=0.0, cutmix_prob=0.0):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        use_cutmix = cutmix_prob > 0.0 and np.random.random() < cutmix_prob
        use_mixup  = mixup_alpha > 0.0 and not use_cutmix
        y_b, lam   = None, 1.0

        if use_cutmix:
            lam = np.random.beta(1.0, 1.0)
            idx = torch.randperm(x.size(0), device=device)
            x1, y1, x2, y2 = _rand_bbox(x.size(), lam)
            x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
            lam = 1 - (y2 - y1) * (x2 - x1) / (x.size(2) * x.size(3))
            y_b = y[idx]

        elif use_mixup:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(x.size(0), device=device)
            x   = lam * x + (1 - lam) * x[idx]
            y_b = y[idx]

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(x)
            if y_b is not None:
                loss = lam * loss_fn(logits, y) + (1 - lam) * loss_fn(logits, y_b)
            else:
                loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy_top1(logits, y) * bs
        n += bs

    return total_loss / n, total_acc / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type,enabled=(device.type == "cuda")):
            logits = model(x)
            loss = loss_fn(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_top1(logits, y) * bs
        n += bs

    return total_loss / n, total_acc / n

def run_training(model, train_loader, val_loader, device, epochs=30, lr=3e-4, weight_decay=1e-4, patience=None, use_scheduler=True, mixup_alpha=0.0, cutmix_prob=0.0):
    params = [p for p in model.parameters() if p.requires_grad] # Nur die Parameter werden ausgewählt die trainiert werden sollen
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Mixed Precision für schnellere Trainingszeit und weniger Speicher verbrauch
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    history = []
    
    # Lernrate startet hoch und fällt über die Epochen ab, wodurch das Modell am Anfang große Schritte mach und am Ende nur noch verfeinert
    # Standard für ViTs ist Cosine Annealing, die LR fällt wie eine halbe Kosinuskurve
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) if use_scheduler else None
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, mixup_alpha=mixup_alpha, cutmix_prob=cutmix_prob)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        if scheduler:
            scheduler.step()
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        
        print(f"Epoch {epoch+1:03d}/{epochs} | "
              f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")
        
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if patience is not None and no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(best_state)
    return best_val_acc, history