import time
import torch
from src.models.vit import build_vit_from_cfg
from src.utils.training import run_training
from src.utils.metrics import count_params

'''
SPT ändert patch_embed von 3 auf 15 Input-Channels —
die Gewichte sind nicht kompatibel. Gleiches gilt für 
use_lsa (verändert die Attention-Struktur), depth (andere Anzahl Blöcke)
und num_heads (andere Q/K/V-Dimensionen).
'''

COMPATIBLE_KEYS = {"depth", "num_heads", "use_spt", "use_lsa"}

def is_compatible(cfg_a, cfg_b):
    """True wenn beide Configs in den architektur-definierenden Keys übereinstimmen."""
    return all(cfg_a[k] == cfg_b[k] for k in COMPATIBLE_KEYS)

def evaluate_cfg(cfg, num_classes, train_loader, val_loader, device, proxy_epochs, parent_cfg=None, parent_state_dict=None):
    model = build_vit_from_cfg(num_classes, cfg)
    
    # Weight Inheritance nur wenn Architektur kompatibel
    if parent_state_dict is not None and parent_cfg is not None:
        if is_compatible(cfg, parent_cfg):
            model.load_state_dict(parent_state_dict, strict=False)
        
    params = count_params(model)
    t0 = time.time()
    
    best_val, _ = run_training(
        model, train_loader, val_loader, device, 
        epochs=proxy_epochs,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    
    return {
        **cfg,
        "val_acc":     best_val,
        "params":      params,
        "time_sec":    time.time() - t0,
        "_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
    }