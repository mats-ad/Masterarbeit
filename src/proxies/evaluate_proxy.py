import torch
from src.models.vit import build_vit_from_cfg
from src.utils.metrics import count_params
from src.proxies.synflow import compute_synflow
from src.proxies.naswot import compute_naswot
from src.proxies.gradnorm import compute_gradnorm


def evaluate_proxies(cfg, num_classes, data_batch, labels, device, input_size=(1, 3, 224, 224)):
    model = build_vit_from_cfg(num_classes, cfg).to(device)
    model.eval()
    
    results = {"params": count_params(model)}
    results["synflow"] = compute_synflow(model, input_size=input_size)
    results["naswot"] = compute_naswot(model, data_batch, device)
    results["gradnorm"] = compute_gradnorm(model, data_batch, labels, device)
    
    return results
