import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.metrics import accuracy_top1

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_acc, n = 0.0, 0.0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(x)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_top1(logits, y) * bs
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

def run_training(model, train_loader, val_loader, device, epochs=30, lr=3e-4, weight_decay=1e-4, patience=None):
    params = [p for p in model.parameters() if p.requires_grad] # Nur die Parameter werden ausgewählt die trainiert werden sollen
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Mixed Precision für schnellere Trainingszeit und weniger Speicher verbrauch
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    history = []
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
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