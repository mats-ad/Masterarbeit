import random

SEARCH_SPACE = {
    "depth": [4, 6, 8, 10, 12],
    "num_heads": [4, 6, 8],
    "use_spt": [0, 1],
    "use_lsa": [0, 1],
    "finetune_mode": ["head", "full", "last2", "last4"],
    "mixup_alpha": [0.0, 0.2, 0.4], # mischt zwei Bilder und ihre Labels linear neues_bild  = α · bild_1 + (1-α) · bild_2, neues_label = α · label_1 + (1-α) · label_2, je höher desto stärker
    "cutmix_prob": [0.0, 0.5, 1.0], # schneidet rechteckigen Bereich aus Bild 1 aus und fügt ihn in Bild 2 ein [  Bild 2  | Bild 1 ]   → label = 0.6 · label_2 + 0.4 · label_1 [  Bild 2  | Bild 1 ]
    "lr": [1e-4, 3e-4, 1e-3],
    "weight_decay": [1e-4, 5e-4],
}

EMBED_DIM = 384

def validate_config(cfg):
    """
        Gibt ein korrigiertes Dict zurück, was später für 
        mutation und crossover benötigt wird, da Mutation und 
        Crossover versehentlich ungültige num_heads erzeugen können
    """
    cfg = dict(cfg)
    if EMBED_DIM % cfg["num_heads"] != 0:
        valid_heads = [h for h in SEARCH_SPACE["num_heads"] if EMBED_DIM % h == 0]
        cfg["num_heads"] = random.choice(valid_heads)
    return cfg

def sample_config():
    cfg = {k: random.choice(v) for k, v in SEARCH_SPACE.items()}
    return validate_config(cfg)