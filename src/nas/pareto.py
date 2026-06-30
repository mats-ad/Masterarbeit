def dominates(a, b):
    """True wenn a Pareto-dominant über B ist
        (höhere Accuracy, weniger Parameter)"""
    better_in_one = (a["val_acc"] > b["val_acc"]) or (a["params"] < b["params"])
    not_worse     = (a["val_acc"] >= b["val_acc"]) and (a["params"] <= b["params"])
    return not_worse and better_in_one

def get_pareto_front(population):
    """Gibt alle nicht dominierten Architekturen zurück"""
    front = []
    
    for candidate in population:
        dominated = any(dominates(other, candidate) for other in population)
        if not dominated:
            front.append(candidate)
    return front

def non_dominated_sort(population):
    """Teilt Population in Pareto Ränge auf. Gibt Liste von Fronten zurück"""
    remaining = list(population)
    fronts = []
    
    while remaining:
        front = get_pareto_front(remaining)
        fronts.append(front)
        front_set = {id(x) for x in front}
        remaining = [x for x in remaining if id(x) not in front_set]
    
    return fronts

def crowding_distance(front):
    """Berechnet Crowding Distance für jede Architektur in einer Front.
    Gibt die Front zurück mit zusätzlichem '_cd' Key"""
    
    n = len(front)
    if n <= 2:
        for ind in front:
            ind["_cd"] = float("inf")
        return front
    
    for ind in front:
        ind["_cd"] = 0.0
        
    for objective in ("val_acc", "params"):
        sorted_front = sorted(front, key=lambda x: x[objective])
        obj_min = sorted_front[0][objective]
        obj_max = sorted_front[-1][objective]
        obj_range = obj_max - obj_min or 1e-9
        
        sorted_front[0]["_cd"]  = float("inf")
        sorted_front[-1]["_cd"] = float("inf")

        for i in range(1, n - 1):
            sorted_front[i]["_cd"] += (
                sorted_front[i + 1][objective] - sorted_front[i - 1][objective]
            ) / obj_range

    return front
        