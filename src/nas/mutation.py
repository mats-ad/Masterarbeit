import random
from src.nas.search_space import SEARCH_SPACE, validate_config

def mutate_config(cfg, p_mut=0.3):
    '''
        Mutation nimmt bestehende Konfiguration und verändert jeden Parameter mit 
        Wahrscheinlichkeit p_mut. Es muss mindestens ein Parameter geändert werden,
        sonst ist das Kind identisch mit Eltern
    '''
    child = dict(cfg)
    
    for key, choices in SEARCH_SPACE.items():
        if random.random() < p_mut:
            # nur andere Werte als den aktuellen wählen
            options = [c for c in choices if c != child[key]]
            if options:
                child[key] = random.choice(options)
    
    return validate_config(child)