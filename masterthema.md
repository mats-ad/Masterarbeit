# Master Thesis: Konzept & Plan

---

## Titel

**"Multi-Objective Neural Architecture Search for Vision Transformers on Small Datasets:
Efficiency, Accuracy, and Search Cost Analysis"**

### Alternativer Titel (deutsch)

**"Multi-objectives NAS für Vision Transformer auf kleinen Datensätzen:
Ein Framework zur systematischen Bewertung von Effizienz, Genauigkeit und Suchaufwand"**

---

## Kurzbeschreibung

Die vorliegende Arbeit untersucht, unter welchen Datenbedingungen Neural Architecture Search (NAS) für Vision Transformers gegenüber handdesignten CNN-Baselines vorteilhaft ist. Dazu wird ein evolutionäres Multi-Objective NAS-Framework entwickelt, das Pareto-optimale ViT-Konfigurationen hinsichtlich Genauigkeit und Parametereffizienz identifiziert. Im Mittelpunkt steht nicht die Architektursuche selbst, sondern ein systematisches Evaluation-Framework, das Suchaufwand, Genauigkeitsgewinn und Generalisierbarkeit der gefundenen Architekturen messbar macht.

Der zentrale wissenschaftliche Beitrag ist nicht nur "NAS findet bessere Architekturen",
sondern die systematische Antwort auf die Frage: **Wann und warum lohnt sich der Einsatz
von NAS gegenüber einfachem Fine-Tuning von CNN-Baselines?**

---

## Ausgangslage: Das Hauptprojekt als Baseline

Im Hauptprojekt wurde ein funktionierendes evolutionäres NAS gebaut mit:

- Pretrained ViT-Small (timm, ImageNet-Gewichte)
- Shifted Patch Tokenization (SPT) und Local Self-Attention (LSA)
- Evolutionärer Suche: Population → Elite Selection → Mutation → Generationen
- Soft-Constraint Fitness: `val_acc - λ·params - μ·time`
- Proxy-Training (5 Epochen) zur schnellen Architekturbewertung

### Identifizierte Probleme im Hauptprojekt

| Problem                           | Auswirkung                                                                                                            |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Skalare Fitness-Funktion          | SPT systematisch benachteiligt durch Zeitstrafe; NAS konvergiert zu einer einzigen Lösung statt Tradeoffs zu erkunden |
| Weight Inheritance defekt         | `parent_state` wurde extrahiert aber nie übergeben — jedes Modell von Grund auf neu initialisiert                     |
| Nur 5 Proxy-Epochen auf CIFAR-100 | Zu rauschige Fitness-Schätzungen; falsche Architekturen wurden bevorzugt                                              |
| Kein systematischer Vergleich     | Kein Vergleich mit CNN-Baselines; keine Antwort auf "lohnt sich NAS?"                                                 |
| Nur Mutation, kein Crossover      | Eingeschränkte Exploration des Suchraums                                                                              |
| Kein Zero-Cost Proxy              | Jede Architektur muss trainiert werden — hoher Rechenaufwand                                                          |

Das Hauptprojekt war ein **Proof of Concept** — die Pipeline funktioniert, aber die
Ergebnisse sind durch die oben genannten Probleme verzerrt. Die Masterarbeit behebt
diese Probleme systematisch und liefert damit erstmals valide, vergleichbare Ergebnisse.

---

## Die 3 zentralen Forschungsfragen

### F1 — Ab welcher Datensatzgröße verliert ein NAS-optimierter Vision Transformer seinen Genauigkeitsvorteil gegenüber CNN-Baselines?

> Erreicht ein NAS-optimierter Vision Transformer auf kleinen domänenspezifischen Datensätzen
> eine höhere Accuracy als handdesignte CNN-Baselines (MobileNet, EfficientNet, ResNet)
> bei vergleichbarer Parameteranzahl?

MobileNet, EfficientNet und ResNet sind die Standard-Wahl für kleine Datensätze.
Diese Frage beantwortet ob der Einsatz eines ViT mit NAS überhaupt sinnvoll ist.

### F2 — Wie nah kommt NAS an den SOTA, mit weniger Parametern?

> Kann das NAS-Framework eine ViT-Konfiguration finden, die bei weniger Parametern als
> der bisherige State of the Art eine wettbewerbsfähige oder bessere Accuracy erreicht?

Es existiert ein SOTA-Modell auf dem gewählten Datensatz (Referenz: Doktoranden-Paper).
Ziel ist nicht diesen zu übertreffen, sondern einen Punkt auf der Pareto-Front zu finden
der effizienter ist — also weniger Parameter bei ähnlicher Genauigkeit.

### F3 — Wann lohnt sich NAS, und wann nicht?

> Wie groß ist der Accuracy-Gewinn durch NAS im Vergleich zum einfachen Fine-Tuning,
> und wie viel Rechenaufwand (GPU-Stunden) ist dafür notwendig?

Das ist die "Mehrwert vs. Aufwand"-Frage des Professors. NAS kostet Rechenzeit.
Diese Arbeit quantifiziert erstmals systematisch ob und wann dieser Aufwand gerechtfertigt ist.

---

## Wissenschaftliche Beiträge (Contributions)

### Beitrag 1: Multi-Objective Pareto-NAS für ViTs

Ersatz der skalaren Fitness-Funktion durch echte Pareto-Optimierung.
Statt einer einzigen "besten" Architektur liefert das Framework eine **Pareto-Front**:
eine Kurve von Architekturen, die jeweils optimal sind für einen bestimmten Accuracy-Effizienz-Tradeoff.

Der Nutzer/Anwendungsfall bestimmt welchen Punkt auf der Kurve er wählt —
je nach verfügbarer Hardware oder Genauigkeitsanforderung.

### Beitrag 2: Zero-Cost Proxy Integration

Architekturen werden zuerst **ohne Training** gerankt durch Gradient-basierte Proxies
(SynFlow, NASWOT, GradNorm). Nur die Top-K Architekturen werden wirklich trainiert.

Dadurch wird die Suche 10-50x schneller — und dieser Speedup ist direkt messbar
und Teil der Antwort auf F3 (Aufwand).

### Beitrag 3: Korrektes Weight Inheritance mit Kompatibilitätsprüfung

Korrekte Implementierung der Weight Inheritance — Kindarchitekturen erben Gewichte
vom Elternmodell wenn die Struktur kompatibel ist. Mit empirischer Ablation:
bringt Weight Inheritance tatsächlich etwas oder nicht?

### Beitrag 4: Systematisches NAS-Evaluation-Framework

Das ist der Hauptbeitrag der Thesis. Ein Framework das beantwortet:

- Wie vergleicht man NAS-Ergebnisse fair mit CNN-Baselines?
- Wie bewertet man ob der Suchaufwand gerechtfertigt war?
- Wie stellt man sicher dass die gefundenen Architekturen generalisieren?

---

## Methodik im Detail

### Suchraum

```python
SEARCH_SPACE = {
    # Architektur
    "depth":          [4, 6, 8, 10, 12],     # Anzahl Transformer-Blöcke
    "num_heads":      [4, 6, 8],              # Attention Heads (muss embed_dim teilen)
    "use_spt":        [0, 1],                 # Shifted Patch Tokenization
    "use_lsa":        [0, 1],                 # Local Self-Attention

    # Fine-Tuning Strategie
    "finetune_mode":  ["head", "last2", "last4", "full"],

    # Augmentierung (NEU gegenüber Hauptprojekt)
    "mixup_alpha":    [0.0, 0.2, 0.4],
    "cutmix_prob":    [0.0, 0.5, 1.0],

    # Optimizer
    "lr":             [1e-4, 3e-4, 1e-3],
    "weight_decay":   [1e-4, 5e-4],
}
```

### Evolutionärer Algorithmus (verbessert)

```
1. Initialisierung
   └── Zufällige Population aus SEARCH_SPACE

2. Zero-Cost Proxy Ranking (NEU)
   └── Alle Architekturen werden ohne Training gerankt
   └── Nur Top-K (z.B. 50%) gehen weiter

3. Proxy-Training (verbessert)
   └── 10-15 Epochen statt 5
   └── Mit korrektem Weight Inheritance

4. Pareto-Selektion (NEU, ersetzt Elite-Selektion)
   └── Pareto-Dominanz: Architektur A dominiert B wenn
       A besser in Accuracy UND besser in Params
   └── Nicht-dominierte Architekturen bilden die Pareto-Front

5. Variation (verbessert)
   └── Mutation (wie Hauptprojekt, p_mut angepasst)
   └── Crossover (NEU): Parameter von zwei Eltern kombinieren

6. Iteration über Generationen
```

### Pareto-Dominanz Definition

Architektur A dominiert Architektur B wenn gilt:

- `accuracy(A) >= accuracy(B)` UND `params(A) <= params(B)`
- mindestens eine Ungleichung ist strikt

Nicht-dominierte Architekturen bilden die Pareto-Front und werden als Eltern für die nächste Generation gewählt.

### Zero-Cost Proxies (NEU)

| Proxy        | Idee                                          | Aufwand  |
| ------------ | --------------------------------------------- | -------- |
| **SynFlow**  | Summiert Gradienten ohne Datenpunkte          | Sekunden |
| **NASWOT**   | Misst Kernel-Matrix-Rang bei zufälligem Batch | Sekunden |
| **GradNorm** | Norm der Gradienten nach einem Forward-Pass   | Sekunden |

Evaluierung: Korrelation der Proxy-Rankings mit echter Validation Accuracy
(Spearman-Korrelation) → zeigt welcher Proxy am zuverlässigsten ist.

### Korrektes Weight Inheritance

```python
# Fix: parent_state_dict korrekt weitergeben
res = evaluate_cfg(cfg, parent_state_dict=parent_state)

# Kompatibilitätsbedingung bleibt:
# Nur erben wenn depth, num_heads, use_spt, use_lsa identisch
```

---

## Das Evaluation-Framework (Hauptbeitrag)

### Schritt 1: Baseline-Messungen

Alle Baselines auf demselben Datensatz, identische Trainings-Pipeline:

| Modell                | Typ | Params   | Trainingsart           |
| --------------------- | --- | -------- | ---------------------- |
| MobileNetV3-Large     | CNN | 5.4M     | Fine-Tuning pretrained |
| EfficientNet-B1       | CNN | 7.8M     | Fine-Tuning pretrained |
| ResNet-18             | CNN | 11.7M    | Fine-Tuning pretrained |
| ViT-Small vanilla     | ViT | 22M      | Fine-Tuning pretrained |
| **NAS-ViT (Beitrag)** | ViT | variabel | NAS-gefunden           |

### Schritt 2: Pareto-Front Vergleich

```
Frage F1: Liegt die NAS-Pareto-Front oberhalb der CNN-Baselines?
→ Beantwortet: "Schlägt NAS-ViT MobileNet/EfficientNet/ResNet?"

Frage F2: Wo liegt die Pareto-Front relativ zum SOTA?
→ Beantwortet: "Weniger Parameter bei ähnlicher Accuracy als SOTA?"
```

### Schritt 3: Suchkosten-Analyse (F3 — Mehrwert vs. Aufwand)

| Methode                          | GPU-Stunden | Beste Accuracy | Accuracy-Gewinn vs. EfficientNet |
| -------------------------------- | ----------- | -------------- | -------------------------------- |
| EfficientNet-B0 Fine-Tuning      | ~0.5h       | X%             | Baseline                         |
| ViT-Small Fine-Tuning (kein NAS) | ~1h         | Y%             | +Δ1%                             |
| NAS ohne Zero-Cost Proxy         | ~10h        | Z%             | +Δ2%                             |
| NAS mit Zero-Cost Proxy          | ~2h         | Z%             | +Δ2%                             |

→ Zeigt: Zero-Cost Proxies reduzieren Suchkosten um 80% bei gleichem Ergebnis.
→ Zeigt: NAS lohnt sich wen n Accuracy-Gewinn > Schwellwert für den Anwendungsfall.

### Schritt 4: Ablation Studies

Jede Komponente wird einzeln getestet:

| Ablation                           | Frage                          |
| ---------------------------------- | ------------------------------ |
| NAS ohne Pareto (skalare Fitness)  | Wie viel bringt Pareto?        |
| NAS ohne Zero-Cost Proxy           | Wie viel schneller ist ZCP?    |
| NAS ohne Weight Inheritance        | Bringt Weight Inheritance was? |
| NAS ohne Crossover                 | Bringt Crossover was?          |
| NAS ohne SPT/LSA                   | Bringen diese Komponenten was? |
| NAS ohne Augmentierung im Suchraum | Wie viel bringt Augmentierung? |

---

## Anwendungsfall & Datensatz

### Primärer Datensatz: Food11

- ~16.600 Bilder, 11 Lebensmittelklassen
- Bread, Dairy, Dessert, Egg, Fried Food, Meat, Noodles, Rice, Seafood, Soup, Vegetable/Fruit
- Realistisches Klassifikationsproblem mit hoher Intra-Klassen-Variation

**Konkreter Anwendungsfall:**
Automatisches Dietary Assessment — eine Ernährungs-App fotografiert das Essen
und klassifiziert es automatisch für Kalorientracking und Ernährungsmonitoring.
Ziel: möglichst genaues Modell mit möglichst wenig Parametern (schnelle Inferenz).

**Warum dieser Anwendungsfall die Forschungsfragen motiviert:**

- F1: Brauche ich einen ViT oder reicht EfficientNet?
- F2: Wie nah komme ich an den SOTA bei gegebenen Ressourcen?
- F3: Lohnt es sich NAS zu laufen oder nehme ich einfach EfficientNet-B0?

### Sekundärer Datensatz (Generalisierung)

Ein zweiter Datensatz zeigt dass das Framework nicht auf Food11 beschränkt ist.
Kandidaten: HAM10000 (Dermatologie), MSTAR (SAR), RESISC45 (Remote Sensing).

---
