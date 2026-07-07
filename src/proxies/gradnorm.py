import torch
import torch.nn as nn

def compute_gradnorm(model, data_batch, labels, device):
    model.eval()
    model.zero_grad()
    
    x = data_batch.to(device)
    y = labels.to(device)
    
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    
    loss.backward()
    
    score = sum(
        p.grad.norm().item() ** 2
        for p in model.parameters()
        if p.grad is not None
    ) ** 0.5
    
    model.zero_grad()
    return score