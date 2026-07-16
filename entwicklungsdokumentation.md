# Entwicklungsdokumentation

## Zweck dieser Datei

Diese Datei ist das Rohmaterial für Kapitel 4 (Methodik) und 5 (Evaluation) der
Masterarbeit (siehe Gliederung in [masterthema.md](masterthema.md)). Jede Komponente
wird dokumentiert nach dem Schema **Was / Warum / Wie implementiert**, damit beim
Schreiben nicht rekonstruiert werden muss, welche Entscheidung aus welchem Grund
getroffen wurde.

Zwei Teile:

- **Teil A — Komponenten-Dokumentation**: Stand der Implementierung, wird aktualisiert
  wenn sich Code/Design ändert. Referenz-Charakter, kein Datum pro Eintrag nötig.
- **Teil B — Experiment-Log**: chronologische Einträge, ein Eintrag pro Lauf/Session.
  Wird nach jedem Experiment ergänzt (ab 2026-07-16, siehe Platzhalter am Ende).

Stand: 2026-07-15. Bisher wurden nur **Smoke Tests** gefahren (Pipeline läuft
End-to-End ohne Absturz), keine echten Experimente — diese beginnen morgen
(2026-07-16).

---

## Teil A — Komponenten-Dokumentation

### Übersichtstabelle: Bezug zur Gliederung

| Komponente                          | Datei(en)                                                          | Gliederungspunkt          | Status |
| ------------------------------------ | ------------------------------------------------------------------ | -------------------------- | ------ |
| Suchraum                             | `src/nas/search_space.py`                                          | 4.3                         | fertig |
| ViT-Baumodell + SPT + LSA            | `src/models/vit.py`                                                 | 2.1, 4.2                    | fertig |
| CNN-Baselines                        | `src/models/cnn_baseline.py`                                        | 2.5                         | fertig |
| Fitnessfunktion / Pareto-Dominanz    | `src/nas/pareto.py`                                                 | 4.4                         | fertig |
| Zero-Cost Proxies                    | `src/proxies/synflow.py`, `naswot.py`, `gradnorm.py`, `evaluate_proxy.py` | 2.3, 4.5              | fertig |
| Evolutionärer Algorithmus            | `src/nas/evolution.py`, `mutation.py`, `crossover.py`               | 4.6, 4.6.1                  | fertig |
| Weight Inheritance                   | `src/nas/evaluate.py`                                               | 4.7                         | fertig |
| Trainings-Pipeline (Mixup/CutMix/AMP)| `src/utils/training.py`                                             | 4.8                         | fertig |
| Daten-Pipeline (Food11, Subsampling) | `src/utils/data.py`                                                 | 5.1                         | fertig |
| Experiment-Skripte                   | `scripts/*.py`                                                      | 5.2                         | fertig |
| Infrastruktur (Docker, Remote-GPU)   | `Dockerfile`, `.dockerignore`                                       | 5.2                         | fertig |
| NAS-Runs / F1-Experiment / Ablationen | —                                                                   | 5.3–5.7                     | **offen — ab 2026-07-16** |

---

### 1. Suchraum (`src/nas/search_space.py`) — Gliederung 4.3

**Was:** Definiert den Hyperparameter-Raum, aus dem Architekturen gezogen werden:
`depth` (4–12 Blöcke), `num_heads` (4/6/8), `use_spt`, `use_lsa` (je 0/1),
`finetune_mode` (head/last2/last4/full), `mixup_alpha`, `cutmix_prob`, `lr`,
`weight_decay`. `EMBED_DIM = 384` ist fix (ViT-Small).

**Warum:** Der Suchraum kombiniert Architektur-Entscheidungen (depth, heads, SPT,
LSA) mit Trainings-/Regularisierungs-Entscheidungen (finetune_mode, mixup, cutmix,
lr, weight_decay) in einem gemeinsamen Genom. Das ist eine bewusste Erweiterung
gegenüber dem Hauptprojekt: dort wurde nur die Architektur durchsucht, hier auch
die Trainingsstrategie — weil auf einem kleinen Datensatz (Food11, ~16.6k Bilder)
die Regularisierung (Mixup/CutMix) genauso entscheidend sein kann wie die Architektur
selbst.

**Wie implementiert:** `sample_config()` zieht für jeden Key einen zufälligen Wert.
`validate_config()` repariert danach ungültige Kombinationen — konkret:
`num_heads` muss `EMBED_DIM` (384) teilen, sonst wird ein zufälliger gültiger Wert
aus den Kandidaten nachgezogen. Diese Validierung läuft nach *jeder* Mutation und
jedem Crossover, weil beide Operatoren sonst ungültige `num_heads`-Werte erzeugen
könnten (z. B. durch Crossover zweier an sich gültiger Eltern-Configs mit
unterschiedlichen `num_heads`-Werten, die dann Kompatibilitätsprobleme mit anderen
Genen erzeugen — hier konkret weil ein gemischter Head-Wert nicht mehr zwingend
384 teilt, falls künftig weitere `embed_dim`-Werte hinzukämen; aktuell ist embed_dim
konstant, die Prüfung ist aber die Absicherung für den Fall inkompatibler Head-Zahlen
nach Crossover).

---

### 2. ViT-Modell mit SPT und LSA (`src/models/vit.py`) — Gliederung 2.1, 4.2

**Was:** `build_vit_from_cfg()` baut ein `timm`-ViT-Small (`vit_small_patch16_224`,
ImageNet-pretrained) und wendet optional zwei architektonische Varianten an:

- **SPT (Shifted Patch Tokenization)**: ersetzt `patch_embed` durch eine Version,
  die das Eingabebild diagonal in 4 Richtungen verschiebt, mit dem Original
  konkateniert (3 → 15 Kanäle) und dann patcht.
- **LSA (Local Self-Attention)**: ersetzt die Attention in jedem Block durch eine
  Variante mit lernbarer Temperatur pro Head und einer Diagonalmaske (kein
  Self-Token-Attention).

Zusätzlich implementiert `finetune_mode` vier Freeze-Strategien: `head` (nur
Klassifikationskopf trainierbar), `last2`, `last4`, `full`.

**Warum:** SPT und LSA sind Techniken aus der Literatur (Lee et al., "Vision
Transformer for Small-Size Datasets"), die ViTs einen stärkeren induktiven Bias
für lokale Strukturen geben — relevant für die Forschungsfrage, ob ViTs auf
kleinen/mittleren Datensätzen mit CNNs mithalten können (F1). Die vier
Finetune-Modi sind Teil des Suchraums, weil bei einem kleinen, pretrained Modell
die Frage "wie viel darf sich verändern" mindestens so wichtig ist wie die
Architektur selbst — Overfitting-Risiko bei `full` vs. Underfitting-Risiko bei
`head`.

**Wie implementiert:** `enable_spt()` tauscht `model.patch_embed` aus (Conv2d
nimmt jetzt 15 statt 3 Eingabekanäle, deshalb inkompatibel mit vortrainierten
Gewichten dieses Layers — siehe Weight-Inheritance-Kompatibilitätsprüfung unten).
`enable_lsa()` ersetzt `blk.attn` für jeden Transformer-Block durch
`LSA_Attention`, die die relevanten Teile (qkv, proj, num_heads, head_dim) aus dem
Original-Attention-Modul übernimmt und nur `scale` und das Masking-Verhalten
ändert. Freeze-Logik: zunächst werden alle Parameter eingefroren
(`requires_grad = False`), dann werden abhängig vom Modus gezielt Blöcke wieder
aufgetaut.

---

### 3. CNN-Baselines (`src/models/cnn_baseline.py`) — Gliederung 2.5

**Was:** Dünner Wrapper um `timm.create_model()` für drei pretrained CNNs:
`resnet18.a1_in1k`, `efficientnet_b0.ra_in1k`, `mobilenetv3_large_100.ra_in1k`.

**Warum:** Diese drei Modelle sind die Vergleichsklasse für F1/F2 — sie decken
unterschiedliche Effizienzklassen ab (ResNet18 als klassische Baseline,
EfficientNet-B0 als parametereffizientes Modell, MobileNetV3 als
mobile-optimiertes Modell) und werden identisch zum ViT (gleiche Trainings-Pipeline,
gleicher Datensatz) fine-getuned, damit der Vergleich fair ist.

**Wie implementiert:** Keine Sonderbehandlung nötig — alle drei Modelle werden
komplett fine-getuned (kein Freezing wie beim ViT), da sie deutlich kleiner sind
und günstiger komplett zu trainieren.

---

### 4. Fitnessfunktion / Pareto-Dominanz (`src/nas/pareto.py`) — Gliederung 4.4

**Was:** Ersetzt die skalare Fitness aus dem Hauptprojekt
(`val_acc - λ·params - μ·time`) durch echte Multi-Objective-Optimierung über zwei
Ziele: `val_acc` (maximieren) und `params` (minimieren). Implementiert:

- `dominates(a, b)`: A dominiert B, wenn A in mindestens einem Ziel besser und in
  keinem schlechter ist.
- `non_dominated_sort()`: NSGA-II-artige Einteilung der Population in Ränge/Fronten
  (Front 0 = nicht dominiert, Front 1 = nur von Front 0 dominiert, usw.).
- `crowding_distance()`: misst, wie "isoliert" ein Individuum innerhalb seiner
  Front bezüglich der Zielwerte liegt (Randpunkte bekommen `inf`), um bei
  gleichem Rang Diversität statt Häufung zu bevorzugen.

**Warum:** Das zentrale identifizierte Problem im Hauptprojekt (siehe
[masterthema.md](masterthema.md), Zeile "Skalare Fitness-Funktion") war, dass eine
gewichtete Summe SPT systematisch benachteiligt hat, weil SPT langsamer trainiert
und die Zeitstrafe das dominierte. Eine Pareto-Front macht diese Gewichtung
überflüssig: es gibt keine einzige "beste" Architektur mehr, sondern eine Menge
von Kompromissen, aus der der Anwendungsfall (Beitrag 1 in masterthema.md)
denjenigen Punkt wählt, der zu seinen Ressourcen passt.

**Wie implementiert:** Klassischer NSGA-II-Ansatz, aber ohne Referenzpunkt-Logik
(nicht nötig bei 2 Zielen). `run_evolution()` nutzt die Pareto-Front (mit
Crowding-Distance als Tie-Breaker, wenn eine Front nicht vollständig in die neue
Elternpopulation passt) als Eltern-Pool für die nächste Generation.

---

### 5. Zero-Cost Proxies (`src/proxies/*.py`) — Gliederung 2.3, 4.5

**Was:** Drei gradientenbasierte Metriken, die die Qualität einer Architektur
**ohne Training** schätzen, auf Basis eines einzigen Batches:

- **SynFlow** (`synflow.py`): Setzt alle Gewichte auf ihren Betrag (`|θ|`, verhindert
  "Layer Collapse" durch Vorzeichen-Auslöschung), macht einen Forward+Backward-Pass
  mit einem All-Ones-Input (keine echten Daten nötig) und summiert `Σ(p · ∂L/∂p)`
  über alle Parameter. Danach werden die ursprünglichen Vorzeichen wiederhergestellt.
- **NASWOT** (`naswot.py`): Hookt sich in alle ReLU/GELU-Aktivierungen, binarisiert
  die Outputs (>0 oder nicht) für einen Batch, bildet daraus eine
  Kernel-Matrix (`K = binary @ binary.T`) und nimmt deren Log-Determinante als
  Score. Idee: je unterschiedlicher die Aktivierungsmuster verschiedener Inputs,
  desto ausdrucksstärker das Netzwerk.
- **GradNorm** (`gradnorm.py`): Ein normaler Forward+Backward-Pass mit echten
  Labels, Score = L2-Norm aller Parametergradienten.

`evaluate_proxy.py` (`evaluate_proxies()`) baut das Modell aus der Config und
berechnet alle drei Scores für einen Kandidaten.

**Warum:** Jede Architektur zu trainieren (auch nur mit Proxy-Epochen) ist teuer.
Zero-Cost Proxies sollen einen großen Teil eindeutig schlechter Kandidaten schon
vor dem Training aussortieren (Beitrag 2 in masterthema.md). Drei verschiedene
Proxies statt einem, weil aus der Literatur bekannt ist, dass kein einzelner
Proxy robust über alle Architektur-Familien hinweg korreliert — die Korrelation
mit echter Val-Accuracy soll empirisch gemessen werden (Spearman, siehe
`run_proxy_validation.py`), bevor final entschieden wird, welcher(n) Proxy(s) im
NAS-Lauf verwendet werden.

**Wie implementiert:** In `run_evolution()` (siehe Punkt 6) werden bei
`use_proxies=True` alle drei Proxy-Scores pro Kandidat berechnet, pro Proxy in
einen Rang umgewandelt (`_rank()`, 0 = bester Kandidat) und die drei Ränge
aufsummiert (einfaches Rangsummen-Ensemble statt gewichteter Kombination — bewusst
simpel gehalten, um keine weitere Hyperparameter-Entscheidung in den Suchprozess
einzuführen). Nur die `proxy_keep_frac` (Standard 50 %) besten Kandidaten nach
Rangsumme werden tatsächlich trainiert.

---

### 6. Evolutionärer Algorithmus (`src/nas/evolution.py`, `mutation.py`, `crossover.py`) — Gliederung 4.6, 4.6.1

**Was:** `run_evolution()` orchestriert den vollständigen Zyklus:

1. Zufällige Startpopulation (`sample_config()` × `pop_size`)
2. Optional: Zero-Cost-Proxy-Filterung (siehe Punkt 5)
3. Proxy-Training jeder verbleibenden Config (`evaluate_cfg`, siehe Punkt 7)
4. Pareto-Sortierung der Ergebnisse (siehe Punkt 4) → Eltern-Pool für nächste Generation
5. Neue Population durch Crossover + Mutation aus dem Eltern-Pool
6. Wiederholen über `generations` Generationen, alles wird nach jeder Config in eine CSV geschrieben (Zeile für Zeile, damit bei Abbruch nichts verloren geht)

**Mutation** (`mutation.py`): Für jeden Parameter unabhängig mit Wahrscheinlichkeit
`p_mut` ein anderer Wert als der aktuelle aus dem Suchraum gezogen.

**Crossover** (`crossover.py`): Uniform Crossover — jeder Parameter wird
unabhängig zufällig von Elternteil A oder B übernommen.

**Warum:** Uniform Crossover statt z. B. 1-Punkt-Crossover, weil die Parameter im
Genom keine natürliche Reihenfolge/Nachbarschaftsstruktur haben (anders als z. B.
bei einer Bit-String-Kodierung eines Layer-Stacks) — jeder Parameter ist
unabhängig sinnvoll kombinierbar. Crossover war im Hauptprojekt komplett nicht
vorhanden (nur Mutation) und ist eine der in masterthema.md explizit benannten
Lücken ("Nur Mutation, kein Crossover → Eingeschränkte Exploration des
Suchraums").

**Wie implementiert:** Nach Crossover *und* nach Mutation läuft immer
`validate_config()`, um durch die Rekombination erzeugte ungültige
`num_heads`-Werte zu reparieren. Für Weight Inheritance trägt jedes Kind seine
Eltern-Referenz (`_parent_cfg`, `_parent_state`) mit sich, bevor es evaluiert wird
(siehe Punkt 7).

---

### 7. Weight Inheritance (`src/nas/evaluate.py`) — Gliederung 4.7

**Was:** `evaluate_cfg()` baut das Modell, lädt — falls möglich — die Gewichte des
Elternmodells (`parent_state_dict`) via `load_state_dict(..., strict=False)`,
trainiert dann `proxy_epochs` Epochen und gibt u. a. `val_acc`, `params`,
`time_sec` und den eigenen `_state_dict` (für die nächste Generation) zurück.

**Warum:** Im Hauptprojekt wurde `parent_state` zwar extrahiert, aber nie an
`evaluate_cfg` übergeben — jedes Kind startete faktisch bei Null statt von den
Eltern-Gewichten zu profitieren (siehe masterthema.md, "Weight Inheritance
defekt"). Das macht Proxy-Training-Ergebnisse für Kinder tendenziell schlechter
als sie bei korrekter Vererbung wären, und verzerrt damit die Selektion. Dieser
Bug wird hier behoben; zusätzlich ist eine echte Ablation geplant (Beitrag 3 in
masterthema.md): NAS-Lauf mit vs. ohne Weight Inheritance, um den tatsächlichen
Effekt zu quantifizieren statt ihn nur anzunehmen.

**Wie implementiert:** `is_compatible()` prüft, ob Eltern- und Kind-Config in den
vier architektur-*definierenden* Keys übereinstimmen: `depth`, `num_heads`,
`use_spt`, `use_lsa`. Nur wenn diese vier identisch sind, werden die Gewichte
geladen — andernfalls hätten z. B. SPT (ändert `patch_embed` von 3 auf 15
Eingabekanäle) oder eine andere `depth`/`num_heads` eine andere Tensor-Shape und
`load_state_dict` würde entweder crashen oder (mit `strict=False`) stillschweigend
falsche/keine Gewichte übernehmen. `strict=False` wird trotzdem verwendet, weil
selbst bei kompatibler Architektur der Klassifikationskopf (`head`) bei
unterschiedlicher `num_classes` oder freien/eingefrorenen Layern abweichen könnte.

---

### 8. Trainings-Pipeline (`src/utils/training.py`) — Gliederung 4.8

**Was:** `run_training()` trainiert ein Modell mit AdamW, Cosine-Annealing-LR-Schedule,
Mixed-Precision (AMP, nur auf CUDA aktiv), optionalem Early Stopping (`patience`)
und Mixup/CutMix-Augmentierung. Gibt die beste Val-Accuracy und die volle
Trainings-History zurück; lädt am Ende automatisch die Gewichte der besten Epoche
(nicht die der letzten).

**Warum:** Cosine Annealing ist Standard für ViT-Fine-Tuning (großer Schritt am
Anfang, feine Anpassung am Ende). Early Stopping ist nötig, weil im Proxy-Training
(nur 10–15 Epochen) und im NAS-Loop viele Modelle trainiert werden — Zeit sparen,
wenn ein Kandidat offensichtlich nicht mehr besser wird. Mixup/CutMix sind Teil
des Suchraums (nicht fix), weil auf einem kleinen Datensatz (Food11) die richtige
Stärke der Augmentierung stark vom Modell/Finetune-Modus abhängen kann — das ist
eine der in masterthema.md explizit benannten Suchraum-Erweiterungen gegenüber dem
Hauptprojekt.

**Wie implementiert:** Pro Batch wird zunächst entschieden, ob CutMix (mit
Wahrscheinlichkeit `cutmix_prob`) oder — falls nicht — Mixup (falls
`mixup_alpha > 0`) angewendet wird; beide sind nie gleichzeitig aktiv. CutMix
schneidet eine zufällige Bounding Box aus einem anderen Bild im Batch aus und
gewichtet den Loss anteilig zur überschriebenen Fläche; Mixup mischt zwei Bilder
linear und gewichtet den Loss mit demselben `lam`.

---

### 9. Daten-Pipeline (`src/utils/data.py`) — Gliederung 5.1

**Was:** `make_food11_loaders()` lädt Food11 über `torchvision.datasets.ImageFolder`
aus den drei vorgegebenen Split-Ordnern (`training`, `validation`, `evaluation`),
mit Standard-ImageNet-Normalisierung, Resize auf 224×224, und für den Trainingssplit
zusätzlich Random-Horizontal-Flip + Random-Resized-Crop. Unterstützt `subsample_frac`,
um nur einen zufälligen Anteil der Trainingsdaten zu verwenden.

**Warum:** `subsample_frac` ist direkt die technische Grundlage für F1
("ab welcher Datensatzgröße verliert ein NAS-ViT seinen Vorteil"): das
F1-Experiment (`scripts/run_f1_experiment.py`) trainiert dieselben Modelle über
eine Reihe von `subsample_frac`-Werten (0.1 bis 1.0) und vergleicht die
Accuracy-Kurven.

**Wie implementiert:** Subsampling zieht mit `numpy`s `default_rng(seed)` eine
zufällige, seed-fixierte Teilmenge der Trainingsindizes (ohne Zurücklegen) und
wrapped das Original-Dataset in ein `torch.utils.data.Subset` — Validation- und
Test-Split bleiben dabei immer vollständig, damit die Vergleichbarkeit der
Metriken über verschiedene `subsample_frac`-Werte hinweg erhalten bleibt.

---

### 10. Experiment-Skripte (`scripts/*.py`) — Gliederung 5.2

| Skript                        | Zweck                                                                                     |
| ------------------------------ | ------------------------------------------------------------------------------------------ |
| `run_nas.py`                   | Startet einen vollständigen evolutionären NAS-Lauf (CLI-Wrapper um `run_evolution()`), mit/ohne Zero-Cost-Proxy-Filter |
| `run_proxy_validation.py`      | F2-Vorstudie: zieht `n_configs` zufällige Architekturen, misst alle drei Proxy-Scores *und* trainiert sie tatsächlich, berechnet danach die Spearman-Korrelation zwischen jedem Proxy und der echten Val-Accuracy |
| `run_f1_experiment.py`         | F1-Hauptexperiment: trainiert ViT (feste, sinnvolle Config) und die drei CNN-Baselines über eine Reihe von `subsample_frac`-Werten, um die Accuracy-Kreuzungspunkte zu finden |
| `retrain_best.py`              | Nimmt eine per CLI übergebene (aus einer NAS-Run-CSV manuell ausgewählte) Config und trainiert sie vollständig (50 Epochen statt Proxy-Epochen) inkl. finaler Test-Set-Auswertung; speichert Gewichte + History als JSON |

**Warum eigene Skripte statt Notebooks für die eigentlichen Experimente:** Die
Notebooks (`notebooks/experimente_*.ipynb`) dienen dem interaktiven Testen der
Pipeline-Bausteine; die eigentlichen, länger laufenden Experimente (Stunden bis
Tage auf Remote-GPU) laufen als Python-Skripte, damit sie unabhängig von einer
Notebook-Session im Hintergrund/über SSH laufen können und bei Absturz nicht die
gesamte bisherige Ausgabe verloren geht (CSV wird zeilenweise geschrieben und
geflusht).

---

### 11. Infrastruktur (`Dockerfile`, `.dockerignore`)

**Was:** Docker-Image auf Basis von `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`,
installiert `requirements.txt`, kopiert das Repo, setzt `PYTHONPATH=/workspace`.

**Warum:** Die Experimente laufen auf einer Remote-GPU (Pfade in den Skripten wie
`/mnt/data/mats_adel/food11` deuten auf eine gemountete Remote-Umgebung hin, nicht
auf den lokalen Mac). Ein Docker-Image stellt sicher, dass CUDA-/PyTorch-Version
zwischen lokaler Entwicklung und Remote-Ausführung nicht auseinanderlaufen — relevant
für Reproduzierbarkeit (Kapitel 4.8/5.2).

**Wie implementiert:** Minimalistisches Image ohne Multi-Stage-Build o. Ä., da kein
Deployment-Ziel, sondern reine Experiment-Ausführung.

---

## Teil B — Experiment-Log

> Format pro Eintrag: Datum, Was wurde gemacht, Warum (welche Frage/welches Risiko
> adressiert dieser Lauf), Wie (Parameter/Kommando), Ergebnis, Interpretation.

### 2026-07-07 — Smoke Tests (Pipeline-Validierung, keine echten Ergebnisse)

**Was:** Zwei kurze NAS-Läufe zur Validierung, dass die komplette
Evolution-Pipeline (Sampling → Proxy-Filter → Training → Pareto-Selektion →
Crossover/Mutation → nächste Generation) fehlerfrei durchläuft:
`results/nas_smoke_test.csv` (ohne Zero-Cost-Proxies) und
`results/nas_smoke_proxies.csv` (mit Zero-Cost-Proxy-Filterung).

**Warum:** Vor den eigentlichen, rechenintensiven Experimenten auf der Remote-GPU
musste sichergestellt werden, dass keine Laufzeitfehler auftreten (inkompatible
Shapes bei Weight Inheritance, Crashes bei bestimmten Suchraum-Kombinationen wie
`use_spt=1` + bestimmten `num_heads`, CSV-Schreibfehler, etc.) — Smoke Tests laufen
mit sehr wenigen Epochen/Generationen und dienen nicht der Ergebnisinterpretation.

**Wie:** Kleine Population, wenige Generationen (aus den CSV-Zeilen ablesbar: 2
Generationen, Populationsgröße im niedrigen zweistelligen Bereich), lokal auf dem
Mac gefahren (kein `--data-root`-Remote-Pfad in den Ergebnissen ersichtlich).

**Ergebnis:** Beide Läufe liefen ohne Absturz durch; Val-Accuracies streuen stark
(0.30–0.91), was für Smoke Tests erwartet und irrelevant ist (zu wenige
Proxy-Epochen für aussagekräftige Werte).

**Interpretation:** Pipeline ist lauffähig. Keine inhaltliche Aussage über
Architektur-Qualität ableitbar — das ist erst Aufgabe der echten Läufe ab
2026-07-16.

---

### 2026-07-16 — [Platzhalter: erster echter Experiment-Tag]

<!--
Beim Ausfüllen: Was wurde gestartet (welches Skript, welche Parameter),
warum genau dieser Lauf zuerst (z. B. Proxy-Validierung vor NAS-Runs, weil
F3 von der Proxy-Wahl abhängt), erwartete Laufzeit, tatsächliches Ergebnis,
erste Interpretation, was das für die nächsten Schritte laut Zeitplan
(Woche 11: "NAS Runs mit Pareto Front") bedeutet.
-->

