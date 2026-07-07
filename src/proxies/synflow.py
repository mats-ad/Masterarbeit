import torch

def compute_synflow(model, input_size=(1, 3, 224, 224)):
    device = next(model.parameters()).device
    model.eval()
    
    # Schritt 1: θ → |θ| (verhindert Layer Collapse)
    signs = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            signs[name] = p.data.sign().clone()
            p.data.abs_()
    
    # All-ones Input, keine echten Daten hierfür nötig
    x = torch.ones(input_size, device=device)
    
    # Forward + Backward
    model.zero_grad()
    model(x).sum().backward()
    
    # Score = Σ (p · ∂L/∂p)
    score = sum(
        (p * p.grad).sum().item()
        for p in model.parameters()
        if p.grad is not None
    )
    
    # Ursprüngliche vorzeichen wieder herstellen
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.data.mul_(signs[name])
    model.zero_grad()
    
    return score