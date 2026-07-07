import random
from src.nas.search_space import SEARCH_SPACE, validate_config

def crossover_configs(cfg_a, cfg_b) -> dict:
    """Uniform crossover: jeder Parameter wird zufällig von parent A oder parent B geerbt"""
    child = {
        key: random.choice([cfg_a[key], cfg_b[key]])
        for key in SEARCH_SPACE
    }
    return validate_config(child)