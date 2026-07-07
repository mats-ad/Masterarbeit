import random
import csv
import torch

from src.nas.search_space import sample_config
from src.nas.mutation import mutate_config
from src.nas.crossover import crossover_configs
from src.nas.pareto import non_dominated_sort, crowding_distance
from src.nas.evaluate import evaluate_cfg
from src.proxies.evaluate_proxy import evaluate_proxies

def _rank(scores, descending=True):
    """Gibt Rang-Liste zurück: 0 = bester Kandidat"""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=descending)
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks

def run_evolution(
    num_classes, 
    train_loader, 
    val_loader, 
    device, 
    pop_size=10, 
    generations=10, 
    proxy_epochs=10, 
    p_mut=0.3, 
    seed=42, 
    log_path="results/nas_run.csv",
    use_proxies=False, proxy_keep_frac=0.5):
    random.seed(seed)
    if use_proxies:
        proxy_batch, proxy_labels = next(iter(train_loader))
    population = [sample_config() for _ in range(pop_size)]
    all_results = []
    
    fieldnames = list(population[0].keys()) + ["val_acc", "params", "time_sec", "generation"]
    
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for gen in range(generations):
            print(f"\n=== Generation {gen} | {len(population)} Architekturen ===")
            if use_proxies:
                proxy_scores = [
                    evaluate_proxies(cfg, num_classes, proxy_batch, proxy_labels, device)
                    for cfg in population
                ]
                syn_ranks  = _rank([s["synflow"]  for s in proxy_scores])
                nas_ranks  = _rank([s["naswot"]   for s in proxy_scores])
                grad_ranks = _rank([s["gradnorm"] for s in proxy_scores])
                combined   = [syn_ranks[i] + nas_ranks[i] + grad_ranks[i]
                              for i in range(len(population))]
                keep_n     = max(1, int(len(population) * proxy_keep_frac))
                top_idx    = sorted(range(len(population)), key=lambda i: combined[i])[:keep_n]
                population_to_eval = [population[i] for i in top_idx]
                print(f"  Proxies: {len(population_to_eval)}/{len(population)} Kandidaten behalten")
            else:
                population_to_eval = population
                
            gen_results = []
            
            for cfg in population_to_eval:
                result = evaluate_cfg(
                    cfg, num_classes, train_loader, val_loader, device,
                    proxy_epochs=proxy_epochs,
                    parent_cfg=cfg.get("_parent_cfg"),
                    parent_state_dict=cfg.get("_parent_state"),
                )
                
                result["generation"] = gen
                gen_results.append(result)
                
                row = {k: result[k] for k in fieldnames}
                writer.writerow(row)
                print(f" val_acc={result['val_acc']:.4f} | params={result['params']/1e6:.1f}M")
                
            all_results.extend(gen_results)
            
            # Pareto-Front als Eltern
            fronts = non_dominated_sort(gen_results)
            pareto_pool = []
            for front in fronts:
                crowding_distance(front)
                if len(pareto_pool) + len(front) <= pop_size:
                    pareto_pool.extend(front)
                else:
                    remaining = pop_size - len(pareto_pool)
                    front_sorted = sorted(front, key=lambda x: x["_cd"], reverse=True)
                    pareto_pool.extend(front_sorted[:remaining])
                    break
            
            # Neue population
            new_population = []
            while len(new_population) < pop_size:
                parents = random.sample(pareto_pool, min(2, len(pareto_pool)))
                p1 = parents[0]
                p2 = parents[-1]
                child_cfg = crossover_configs(p1, p2)
                child_cfg = mutate_config(child_cfg, p_mut=p_mut)
                
                # Weight Inheritance vorbereiten
                child_cfg["_parent_cfg"] = p1
                child_cfg["_parent_state"] = p1.get("_state_dict")
                new_population.append(child_cfg)
            
            population = new_population
                
    return all_results    