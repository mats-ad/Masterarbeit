import torch


def accuracy_top1(logits, targets):
    """Fraction of correct predictions. logits: (B, C), targets: (B,)"""
    return (logits.argmax(dim=1) == targets).float().mean().item()

def count_params(model):
    """Total number of parameters (including frozen)"""
    return sum(p.numel() for p in model.parameters())

def count_trainable_params(model):
    """Number of parameters with requires_grad=True"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)