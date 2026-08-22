# Entwicklungsdokumentation

## Zweck dieser Datei

Diese Datei ist das Rohmaterial für Kapitel 4 (Methodik) und Kapitel 5 (Evaluation)
der Masterarbeit (siehe Gliederung in [masterthema.md](masterthema.md)). Jede
Komponente wird dokumentiert nach dem Schema **Was / Warum / Wie implementiert /
Designentscheidungen und verworfene Alternativen**, damit beim Schreiben der Arbeit
nicht rekonstruiert werden muss, welche Entscheidung aus welchem Grund getroffen
wurde und welche Alternativen in Betracht gezogen wurden.

Zwei Teile:

- **Teil A — Komponenten-Dokumentation**: Beschreibt den Stand der Implementierung
  für jede Systemkomponente. Wird aktualisiert, wenn sich Code oder Design ändern.
  Hat Referenz-Charakter; kein Datum pro Eintrag nötig.
- **Teil B — Experiment-Log**: Chronologische Einträge, ein Eintrag pro
  Experiment-Session. Enthält Parameter, Laufzeiten, Ergebnisse und Interpretation.

Stand: 2026-07-17.

---

## Teil A — Komponenten-Dokumentation

### Übersichtstabelle: Bezug zur Gliederung

| Komponente                            | Datei(en)                                                                 | Gliederungspunkt | Status        |
| ------------------------------------- | ------------------------------------------------------------------------- | ---------------- | ------------- |
| Suchraum                              | `src/nas/search_space.py`                                                 | 4.3              | fertig        |
| ViT-Baumodell + SPT + LSA             | `src/models/vit.py`                                                       | 2.1, 4.2         | fertig        |
| CNN-Baselines                         | `src/models/cnn_baseline.py`                                              | 2.5              | fertig        |
| Fitnessfunktion / Pareto-Dominanz     | `src/nas/pareto.py`                                                       | 4.4              | fertig        |
| Zero-Cost Proxies                     | `src/proxies/synflow.py`, `naswot.py`, `gradnorm.py`, `evaluate_proxy.py` | 2.3, 4.5         | fertig        |
| Evolutionärer Algorithmus             | `src/nas/evolution.py`, `mutation.py`, `crossover.py`                     | 4.6, 4.6.1       | fertig        |
| Weight Inheritance                    | `src/nas/evaluate.py`                                                     | 4.7              | fertig        |
| Trainings-Pipeline (Mixup/CutMix/AMP) | `src/utils/training.py`                                                   | 4.8              | fertig        |
| Daten-Pipeline (Food11, Subsampling)  | `src/utils/data.py`                                                       | 5.1              | fertig        |
| Experiment-Skripte                    | `scripts/*.py`                                                            | 5.2              | fertig        |
| Infrastruktur (Docker, Remote-GPU)    | `Dockerfile`, `.dockerignore`                                             | 5.2              | fertig        |
| NAS-Runs, F1-Experiment, Ablationen   | —                                                                         | 5.3–5.7          | ab 2026-07-16 |

---

### 0. Evaluations-Framework — Gliederung 4.1, 4.1.1, 4.1.2, 4.1.3, 4.1.4

#### 4.1 Evaluations-Framework: Überblick und Motivation

Das zentrale methodische Problem dieser Arbeit ist nicht die Architektursuche selbst,
sondern die Frage: **Wann und warum lohnt sich NAS gegenüber einfachem Fine-Tuning
einer CNN-Baseline?** Diese Frage ist nur dann wissenschaftlich beantwortbar, wenn
der Vergleich zwischen NAS-gefundenen Architekturen und CNN-Baselines nach einheitlichen,
nachvollziehbaren Kriterien durchgeführt wird.

Ein naiver Vergleich ("NAS-ViT erreicht X%, CNN erreicht Y%") ist aus mehreren Gründen
problematisch:

1. **Unterschiedliche Parameteranzahl:** Ein größeres Modell hat typischerweise höhere
   Accuracy, unabhängig von der Suchstrategie. Ein fairer Vergleich muss die
   Modellgröße als Dimension explizit einbeziehen.
2. **Unterschiedlicher Trainingsaufwand:** NAS evaluiert viele Architekturen, eine
   CNN-Baseline wird einmal trainiert. Der Mehraufwand muss dem Mehrwert gegenübergestellt
   werden.
3. **Mögliches Overfitting auf den Validierungssplit:** Wenn der NAS-Algorithmus
   val_acc als Fitness-Signal nutzt, besteht das Risiko, dass selektierte Architekturen
   spezifisch für diesen Split optimiert sind, nicht generell besser.

Das Evaluations-Framework adressiert diese Probleme durch vier Komponenten:
Anforderungsanalyse (4.1.1), Baseline-Vergleichsmethodik (4.1.2),
Suchkostenanalyse (4.1.3) und Generalisierbarkeit (4.1.4).

---

#### 4.1.1 Anforderungsanalyse

Für einen validen Vergleich zwischen NAS-gefundenen Architekturen und CNN-Baselines
müssen folgende Anforderungen erfüllt sein:

**Anforderung 1 — Einheitliche Trainingspipeline:**
Alle Modelle werden mit identischer Trainings-Pipeline trainiert: gleicher Optimizer
(AdamW), gleicher LR-Scheduler (CosineAnnealingLR), gleiche Standard-Augmentierungen
(RandomHorizontalFlip, RandomResizedCrop), gleiche Batch-Größe (32) und gleiche
Evaluation-Metrik (Top-1-Accuracy). Unterschiede in den Ergebnissen sind damit
ausschließlich auf Architektur und Konfiguration zurückzuführen, nicht auf
unterschiedliche Trainingsregimes.

**Anforderung 2 — Vorgegebene und unveränderliche Datensatz-Splits:**
Food11 stellt drei feste Splits bereit (training / validation / evaluation). Diese
werden nicht modifiziert. Training erfolgt ausschließlich auf dem training-Split,
Modellselektion anhand des validation-Splits. Das evaluation-Split fungiert als
echtes Test-Set und wird erst nach Abschluss der Suche und Modellselektion einmalig
genutzt.

**Anforderung 3 — Parameter-bewusster Vergleich:**
Accuracy-Vergleiche zwischen Modellen unterschiedlicher Größe sind nur in Verbindung
mit der Parameteranzahl aussagekräftig. Die Pareto-Front (Accuracy vs. Parameter)
dient als Hauptvergleichsinstrument: Sie bildet den vollständigen Tradeoff-Raum ab,
statt eine einzelne "beste" Architektur herauszustellen. CNN-Baselines werden als
Referenzpunkte in denselben Accuracy-Parameter-Raum eingetragen.

**Anforderung 4 — Trennung von Architektur- und Trainingsqualität:**
Der Suchraum enthält sowohl Architekturparameter (depth, num_heads, use_spt, use_lsa)
als auch Trainingsparameter (finetune_mode, lr, weight_decay, mixup_alpha, cutmix_prob).
Die Analyse unterscheidet daher zwischen Architektur-Effizienz (Accuracy pro Parameter)
und Training-Effizienz (Accuracy pro Trainingsaufwand), um zu verstehen welche
Kategorie den größten Einfluss auf das Ergebnis hat.

**Anforderung 5 — Reproduzierbarkeit:**
Alle Experimente verwenden einen fixierten Seed (42). Ergebnisse werden nach jeder
Architektur-Evaluierung zeilenweise in CSV-Dateien geschrieben und geflusht, um
Datenverlust bei Serverabsturz zu verhindern. Hyperparameter, Docker-Befehle und
Laufzeiten sind vollständig dokumentiert.

---

#### 4.1.2 Vergleich: NAS vs. CNN-Baselines

Der Vergleich zwischen NAS-gefundenen Architekturen und CNN-Baselines erfolgt auf
drei strukturellen Ebenen:

**Ebene 1 — Absoluter Accuracy-Vergleich beim vollen Datensatz:**
Die beste durch NAS gefundene Architektur wird nach vollständigem Training (50 Epochen)
und finaler Test-Set-Evaluation direkt den drei CNN-Baselines gegenübergestellt.
Diese Ebene beantwortet: Welches Modell ist bei maximalem Dateneinsatz am genauesten?

**Ebene 2 — Pareto-Front-Vergleich:**
Die vollständige NAS-Pareto-Front wird gegen die CNN-Baselines als Referenzpunkte
verglichen. Dabei wird untersucht ob und in welchem Parameterbereich NAS-Architekturen
die CNNs dominieren (d. h. höhere Accuracy bei gleicher oder geringerer Parameterzahl
erreichen). Dieser Vergleich beantwortet: Ist NAS im Accuracy-Effizienz-Tradeoff
besser als handdesignte CNNs?

**Ebene 3 — Datensatz-Skalierungsvergleich:**
Alle Modelle werden auf mehreren Datensatzgrößen (10 % bis 100 % des Trainingssets)
evaluiert. Dieser Vergleich beantwortet F1: Gibt es eine Datenmenge unterhalb derer
CNN-Baselines dem ViT überlegen sind, und wo liegt dieser Schwellenwert?

---

#### 4.1.3 Bewertung der Suchkostenrechtfertigung

Die Suchkostenrechtfertigung stellt den Mehraufwand von NAS dem erzielbaren
Genauigkeitsgewinn gegenüber. Die Analyse erfolgt in drei Schritten:

**Schritt 1 — Messung der absoluten Suchkosten:**
Für jeden Ansatz werden gemessen: Anzahl der trainierten Architekturen, Gesamtlaufzeit
auf der genutzten Hardware (GPU-Stunden) und die beste gefundene Val-Accuracy.
Verglichen werden: einfaches CNN-Fine-Tuning, einfaches ViT-Fine-Tuning ohne NAS,
NAS ohne Zero-Cost-Proxy-Filter und NAS mit Zero-Cost-Proxy-Filter.

**Schritt 2 — Berechnung der Proxy-Effizienz:**
Die Effizienz des Proxy-Filters wird als Verhältnis von eingesparter Rechenzeit zu
erreichter Accuracy-Qualität quantifiziert. Konkret: Wie viel Prozent der
Evaluierungen werden eingespart, und wie verändert sich die beste gefundene
Architektur-Qualität?

**Schritt 3 — Schwellenwertbetrachtung:**
Der Accuracy-Gewinn von NAS gegenüber einfachem Fine-Tuning wird dem Zeitaufwand
gegenübergestellt. Die Frage lautet: Ab welchem Accuracy-Gewinn amortisiert sich
der NAS-Mehraufwand für einen typischen Anwendungsfall? Diese Betrachtung ist
qualitativ und anwendungsfallabhängig — sie liefert keine universelle Zahl, sondern
eine begründete Einordnung.

---

#### 4.1.4 Retrain-Analyse

Der NAS-Lauf bewertet Architekturen nach 12 Proxy-Epochen. Damit stellt sich die
Frage, ob die so ermittelte Rangliste unter vollständigem Training stabil bleibt —
oder ob kurzes Proxy-Training bestimmte Architektur-Typen systematisch über- oder
unterschätzt.

Um das zu überprüfen, wurden die drei besten und drei schlechtesten Architekturen
aus dem NAS-Lauf (nach Proxy-Val-Accuracy) für 50 Epochen vollständig trainiert
und anschließend auf dem Food11 evaluation-Split bewertet. Die Auswahl dieser sechs
Architekturen als Stichprobe deckt beide Extreme der Qualitätsverteilung ab. Bleibt
die Rangliste nach 50 Epochen konsistent mit der nach 12 Epochen, ist die
Proxy-Selektion valide. Kehrt sie sich um oder ändern sich die Abstände erheblich,
deutet das auf eine systematische Verzerrung durch das kurze Proxy-Training hin.

Die finale Bewertung erfolgt ausschließlich auf dem Food11 evaluation-Split, der
während der gesamten Suche und Modellselektion nicht verwendet wurde.

---

### 0b. Systemübersicht der NAS-Pipeline — Gliederung 4.2

#### 4.2 Systemübersicht der NAS-Pipeline

Die NAS-Pipeline realisiert eine evolutionäre Suche über einen diskret-gemischten
Hyperparameter-Suchraum. Sie besteht aus sieben Phasen, die in jeder Generation
sequenziell durchlaufen werden, mit einer abschließenden Retrain-Phase nach der
letzten Generation. Abbildung 4.1 zeigt den Gesamtablauf; die folgenden Abschnitte
beschreiben jede Phase im Detail.

---

#### 4.2.1 Überblick über die sieben Phasen

**Phase 1 — Suchraum-Definition (→ Abschnitt 4.3):**
Der Suchraum `SEARCH_SPACE` definiert neun diskrete Parameter in vier Kategorien
(Architektur, Fine-Tuning-Strategie, Augmentierung, Optimierung). Jede gültige
Kombination dieser Parameter ist ein sogenanntes _Genom_ — ein Dictionaries mit
neun Schlüssel-Wert-Paaren. `validate_config()` prüft harte Nebenbedingungen
(z. B. `num_heads` muss `embed_dim=384` teilen).

**Phase 2 — Population-Initialisierung:**
Zu Beginn des ersten Durchlaufs werden `pop_size=15` Genome zufällig aus dem
Suchraum gesampelt. Jedes Genom ist gleichwahrscheinlich — es gibt keine
Initialisierungsbiases, die bestimmte Regionen des Suchraums bevorzugen.
Ab Generation 2 besteht die initiale Population aus den Überlebenden der
Pareto-Selektion plus neuen Offspring-Kandidaten aus Crossover und Mutation.

**Phase 3 — Zero-Cost Proxy Filter (→ Abschnitt 4.5):**
Vor dem Training wird jede Architektur mit drei trainingsfreien Proxies bewertet:
SynFlow (Gradientfluss-Analyse), NASWOT (Kernel-Matrix-Rang bei Forward-Pass) und
GradNorm (L2-Norm der Gradienten nach einem Single-Batch). Aus den drei Rang-Scores
wird ein gewichteter Ensemble-Score berechnet. Die `top_k=8` Architekturen mit den
höchsten Ensemble-Scores werden in Phase 4 trainiert; die restlichen `15 - 8 = 7`
werden verworfen. Dieser Schritt reduziert den Trainingsaufwand um ~47 % pro
Generation.

**Phase 4 — Proxy-Training und Fitness-Evaluation (→ Abschnitte 4.8, 4.4):**
Die verbleibenden 8 Architekturen werden je 12 Epochen auf dem Food11 training-Split
trainiert. Die Fitness wird durch zwei Metriken gemessen: `val_acc` (Accuracy auf
dem validation-Split nach Epoche 12) und `params` (Anzahl trainierbarer Parameter).
Diese zwei Metriken definieren den zweidimensionalen Objektraum, in dem die
Pareto-Dominanz berechnet wird. Training erfolgt mit AMP (Automatic Mixed Precision)
und CosineAnnealingLR; Augmentierungsparameter (`mixup_alpha`, `cutmix_prob`) sind
Teil des Genoms und werden per-Architektur gesetzt.

**Phase 5 — Multi-Objective Pareto-Selektion (→ Abschnitt 4.4):**
Alle `pop_size` Individuen (Überlebende vorheriger Generationen plus neu evaluierte
Offspring) werden gemeinsam nach Pareto-Dominanz sortiert. Ein Individuum dominiert
ein anderes, wenn es in mindestens einem Ziel besser ist und in keinem schlechter:

```
a dominiert b  ⟺  val_acc(a) ≥ val_acc(b)  ∧  params(a) ≤ params(b)
                   ∧  (val_acc(a) > val_acc(b)  ∨  params(a) < params(b))
```

Die nicht-dominierte Front (Rank 1) bildet die aktuelle Pareto-Front. Bei
Platzmangel innerhalb eines Ranks wird über Crowding Distance entschieden, um
die Diversität der Population zu erhalten. Es werden die besten `pop_size`
Individuen in die nächste Generation übernommen.

**Phase 6 — Crossover und Mutation (→ Abschnitt 4.6):**
Aus den Überlebenden werden neue Offspring-Kandidaten erzeugt. Crossover kombiniert
zwei Eltern durch zufälligen Parameter-Austausch (Uniform Crossover): jedes
Gen des Kindes stammt mit Wahrscheinlichkeit 0.5 vom ersten, sonst vom zweiten
Elternteil. Mutation ersetzt jeden Parameter eines Individuums unabhängig mit
Wahrscheinlichkeit `mutation_rate=0.2` durch einen zufällig anderen Wert aus dem
Suchraum. Duplicates (bereits evaluierte Genome) werden erkannt und verworfen.

**Phase 7 — Weight Inheritance (→ Abschnitt 4.7):**
Vor dem Training eines Offspring wird geprüft, ob ein kompatibles Elternmodell
existiert. Kompatibel heißt: gleiche `depth`, gleiche `num_heads`, gleiche
`use_spt`, gleiche `use_lsa`. Ist ein kompatibles Elternmodell vorhanden, werden
dessen gespeicherte Gewichte als Startpunkt verwendet — nur die geänderten Trainings-
parameter (lr, weight_decay, finetune_mode) wirken sich auf das Training aus, die
Modellarchitektur ist bereits teilweise optimiert. Ist kein kompatibles Elternteil
vorhanden, startet das Modell von den ImageNet-Gewichten.

---

#### 4.2.2 Ablauf über Generationen

Die Phasen 2–7 werden für `num_generations=8` Generationen wiederholt. Nach
jeder Generation wird der aktuelle Zustand — Pareto-Front, beste Val-Accuracy,
durchschnittliche Params der Pareto-Front — in eine CSV-Datei geflushd. Nach
der letzten Generation liegt eine finale Pareto-Front mit typischerweise 3–6
nicht-dominierten Architekturen vor.

Die besten Architekturen dieser Front (Top-1, Top-2, Top-3 nach `val_acc` sowie
Bottom-1, Bottom-2, Bottom-3 als Negativkontrolle) werden in einer separaten
Retrain-Phase 50 Epochen vollständig trainiert. Die finale Bewertung erfolgt
auf dem Food11 evaluation-Split (Test-Set), der bis zu diesem Punkt nicht
verwendet wurde.

---

#### 4.2.3 Design-Entscheidungen auf Pipeline-Ebene

**Evolutionäre Suche statt Bayesian Optimization oder Differenzierbarer NAS:**
Der Suchraum ist diskret und gemischt (Integer + Kategorisch). Bayesian Optimization
mit kategorischen Variablen skaliert schlecht bei mehr als ~5 Parametern. DARTS
(differenzierbarer NAS) setzt Differenzierbarkeit über den Suchraum voraus und
ist auf kontinuierliche Architektur-Relaxierungen beschränkt. Evolutionäre
Algorithmen (insbesondere NSGA-II) sind für diskrete Multi-Objective-Probleme
gut etabliert und erfordern keine speziellen Annahmen über die Suchlandschaft.

**Zwei-Phasen-Evaluierung (Proxy + Training) statt einmaliger Evaluierung:**
Jede Architektur einmalig für die volle Anzahl Epochen zu trainieren wäre bei
`pop_size × num_generations = 120` Evaluierungen nicht praktikabel. Zwei Phasen
ermöglichen eine grobe Vorselektion durch trainingsfreie Proxies (< 1 Sekunde
pro Architektur) und eine präzisere Bewertung durch 12-Epochen-Training nur
für die vielversprechendsten Kandidaten.

**Multi-Objective statt skalarer Fitness:**
Eine skalare Fitness-Funktion `val_acc - λ·params` erfordert die manuelle Wahl
von λ, die inhärent willkürlich ist und eine implizite Gewichtung von Accuracy
gegenüber Modellgröße einbettet. Die Pareto-Formulierung macht diese Gewichtung
explizit und verschiebt sie aus dem Algorithmus in die Modellselektion: Der
Nutzer (oder die Anwendung) wählt den gewünschten Punkt auf der Pareto-Front.
Dies ist der wesentliche methodische Unterschied zum Hauptprojekt.

**Seed-Fixierung und stateless Reproduzierbarkeit:**
Alle stochastischen Operationen (Population-Initialisierung, Crossover, Mutation,
Training) verwenden einen fixierten Seed (42). Damit sind alle Experimente
vollständig reproduzierbar — bei identischem Seed, identischen Daten und
identischer Hardware wird exakt dieselbe Sequenz von Populationen erzeugt.

---

### 1. Suchraum (`src/nas/search_space.py`) — Gliederung 4.3

#### Was

Der Suchraum definiert den diskreten Hyperparameter-Raum, aus dem der evolutionäre
Algorithmus Architekturen sampelt. Er umfasst neun Parameter in vier Kategorien:

**Architektur-Parameter:**

- `depth` ∈ {4, 6, 8, 10, 12}: Anzahl der Transformer-Blöcke.
- `num_heads` ∈ {4, 6, 8}: Anzahl der Attention-Heads pro Block.
- `use_spt` ∈ {0, 1}: Ob Shifted Patch Tokenization verwendet wird.
- `use_lsa` ∈ {0, 1}: Ob Local Self-Attention verwendet wird.

**Trainings-Strategie:**

- `finetune_mode` ∈ {head, last2, last4, full}: Welche Teile des Modells trainiert werden.

**Regularisierung / Augmentierung:**

- `mixup_alpha` ∈ {0.0, 0.2, 0.4}: Stärke der Mixup-Augmentierung (0 = aus).
- `cutmix_prob` ∈ {0.0, 0.5, 1.0}: Wahrscheinlichkeit, CutMix pro Batch anzuwenden.

**Optimierung:**

- `lr` ∈ {1e-4, 3e-4, 1e-3}: Initiale Lernrate für AdamW.
- `weight_decay` ∈ {1e-4, 5e-4}: L2-Regularisierung.

`EMBED_DIM = 384` ist nicht Teil des Suchraums, sondern fest auf den ViT-Small-Standard
gesetzt.

Die Gesamtgröße des Suchraums beträgt:
5 × 3 × 2 × 2 × 4 × 3 × 3 × 3 × 2 = **12.960 mögliche Konfigurationen**.

#### Warum diese Parameterwahl

**`depth` 4–12:** ViT-Varianten aus der Literatur verwenden typischerweise 6 (ViT-Tiny),
12 (ViT-Small) oder 24 (ViT-Large) Blöcke. Der Bereich 4–12 deckt die kleinen bis
mittleren Varianten ab, die für einen kleinen Datensatz wie Food11 relevant sind.
Tiefere Modelle (>12 Blöcke) würden bei ~16k Trainingsdaten stark overfitting, flachere
(<4 Blöcke) verlieren den Vorteil der tiefen Repräsentation gegenüber CNNs.

**`EMBED_DIM = 384` (fix):** Die Embedding-Dimension ist der stärkste Einflussfaktor
auf Modellgröße und Speicherbedarf. Eine Variable `embed_dim` würde den Suchraum
exponentiell vergrößern und die Vergleichbarkeit zwischen Architekturen erschweren, da
`num_heads` immer `embed_dim` teilen muss. Die fixe Wahl auf 384 (ViT-Small) stellt
sicher, dass alle NAS-Kandidaten zur gleichen Gewichtsklasse gehören und Pareto-Vergleiche
über `params` sinnvoll sind.

**`num_heads` ∈ {4, 6, 8}:** Alle drei Werte teilen EMBED_DIM=384 ganzzahlig
(384/4=96, 384/6=64, 384/8=48 Dimensionen pro Head). Dies ist eine harte Nebenbedingung:
ein Head-Wert der 384 nicht teilt, würde zu einem `ValueError` in der Attention-Berechnung
führen. `validate_config()` sichert dies ab.

**`finetune_mode`:** Fine-Tuning-Strategie ist auf kleinen Datensätzen mindestens so
entscheidend wie die Architektur selbst. `full` (alle Parameter trainierbar) maximiert
die Anpassungsfähigkeit, erhöht aber das Overfitting-Risiko erheblich. `head` (nur der
Klassifikationskopf) ist maximale Regularisierung, kann aber wichtige Features für
das spezifische Domänenproblem nicht anpassen. `last2` und `last4` sind pragmatische
Kompromisse, die in der Transfer-Learning-Literatur gut etabliert sind. Dass dieser
Parameter Teil des Suchraums ist (und nicht fest vorgegeben), ist eine bewusste
Erweiterung gegenüber dem Hauptprojekt.

**`mixup_alpha` und `cutmix_prob`:** Beide Augmentierungsverfahren können auf kleinen
Datensätzen stark helfen, zu hohe Werte führen aber zu zu aggressiver Interpolation und
schlechter Konvergenz. {0.0, 0.2, 0.4} für alpha bzw. {0.0, 0.5, 1.0} für cutmix_prob
deckt einen sinnvollen Bereich von "aus" bis "moderat" ab.

**`lr` ∈ {1e-4, 3e-4, 1e-3}:** Für ViT-Fine-Tuning sind sehr niedrige Lernraten
(<<1e-4) zu konservativ und konvergieren nicht in 12 Proxy-Epochen, zu hohe Raten
(>1e-3) destabilisieren das Training. 3e-4 ist de facto Standard für AdamW auf
Transformer-Modellen (Devlin et al., BERT; Dosovitskiy et al., ViT).

#### Designentscheidung: Gemeinsamer Suchraum für Architektur und Training

Eine wichtige Entscheidung ist, Architektur-Parameter und Trainings-Hyperparameter
in einem einzigen "Genom" zu kodieren und gemeinsam zu suchen. Die Alternative wäre,
zunächst nur die Architektur zu suchen und danach separat Hyperparameter zu optimieren
(z. B. mit Random Search oder Bayesian Optimization). Der gemeinsame Ansatz hat den
Vorteil, Wechselwirkungen zwischen Architektur und Training zu erfassen — z. B. kann
eine flache Architektur (`depth=4`) von aggressiverer Augmentierung profitieren, während
eine tiefe Architektur (`depth=12`, `finetune_mode=full`) mehr Regularisierung braucht.
Der Nachteil ist ein größerer Suchraum, der bei gegebener Evaluierungskapazität
schlechter abgedeckt wird. Mit pop=15 und gen=8 werden 120 der 12.960 Konfigurationen
evaluiert (~0.9 %), was für einen evolutionären Algorithmus aber ausreichend ist.

---

### 2. ViT-Modell mit SPT und LSA (`src/models/vit.py`) — Gliederung 2.1, 4.2

#### Was

`build_vit_from_cfg(num_classes, cfg)` baut ein `timm`-ViT-Small
(`vit_small_patch16_224`, pretrained auf ImageNet-21k, dann fine-getuned auf
ImageNet-1k) und wendet optional zwei architektonische Erweiterungen an:

**Shifted Patch Tokenization (SPT):**
Das originale ViT teilt das Eingabebild in nicht-überlappende 16×16-Patches und
bettet diese linear ein. SPT erweitert dies: Das Eingabebild wird um jeweils 8 Pixel
in vier Diagonalrichtungen (↖, ↗, ↙, ↘) verschoben. Die fünf Versionen (Original +
4 Shifts) werden entlang der Kanal-Dimension konkateniert:

    x_spt = Concat[x, shift(x,↖), shift(x,↗), shift(x,↙), shift(x,↘)]

Statt 3 Eingabekanälen hat `patch_embed` jetzt 15 Kanäle. Die nachfolgende Patch-
Einbettung (Conv2d mit Stride 16, Kernel 16) arbeitet weiterhin auf derselben räumlichen
Auflösung und erzeugt einen Patch-Token-Embedding-Vektor der Dimension 384.

**Local Self-Attention (LSA):**
Standard Multi-Head Self-Attention (MHSA) berechnet für jedes Token die Attention zu
allen anderen Token inklusive sich selbst:

    Attention(Q, K, V) = softmax(QK^T / √d_k) · V

LSA modifiziert dies in zwei Punkten:

1. **Diagonalmaske:** Das Self-Token (Attention eines Tokens zu sich selbst) wird
   maskiert (Maske = −∞ vor dem Softmax), um den Gradienten zu zwingen, echte
   Kontextinformationen zu lernen statt die Identität zu kopieren.
2. **Lernbare Temperatur:** Der Skalierungsfaktor `1/√d_k` wird durch einen
   lernbaren, kopfspezifischen Parameter τ_h ersetzt:

   Attention_LSA(Q, K, V) = softmax((QK^T · exp(τ_h)) ⊙ M) · V

   wobei M die Diagonalmaske ist (M_ii = −∞, sonst 0) und τ_h für jeden der
   `num_heads` Heads separat gelernt wird. Dies erlaubt dem Modell, die Schärfe
   der Attention-Verteilung pro Head adaptiv zu steuern.

**Finetune-Modi:** Zunächst werden alle Parameter eingefroren. Dann werden
je nach Modus gezielt Parameter wieder aktiviert:

- `head`: Nur `model.head` (Linear-Layer, ~384×11 Parameter) trainierbar.
- `last2`: `model.blocks[-2:]` + `model.norm` trainierbar.
- `last4`: `model.blocks[-4:]` + `model.norm` trainierbar.
- `full`: Alle Parameter trainierbar.

#### Warum SPT und LSA

Das zentrale Problem von Standard-ViTs auf kleinen Datensätzen ist, dass der
Self-Attention-Mechanismus einen sehr schwachen räumlichen induktiven Bias hat.
CNNs lernen durch ihre lokale Faltungsstruktur automatisch, dass benachbarte Pixel
zusammengehören (Translation Equivarianz). ViTs müssen diese Struktur aus Daten
lernen, was mit ~10k Trainingsbildern oft nicht gelingt.

SPT stärkt den lokalen Bias, indem es die Patch-Tokens mit Kontextinformation aus
benachbarten Patches anreichert. Statt eines 16×16-Patches nur aus dem originalen
Bildfenster enthält jeder Token implizit Informationen über angrenzende Bereiche
durch die überlagerten Shifts.

LSA verhindert, dass die Attention degeneriert (alle Gewichte auf dem Self-Token
oder uniform verteilt). Die lernbare Temperatur erlaubt dem Modell, Head-spezifisch
entweder "scharfe" (lokale) oder "diffuse" (globale) Attention zu entwickeln.

Beide Verfahren stammen aus Lee et al. (2021), "Vision Transformer for Small-Size
Datasets", und wurden speziell für den hier vorliegenden Fall (kleiner Datensatz +
ViT) entwickelt. Sie sind als optionale Parameter im Suchraum implementiert, weil
nicht a priori klar ist, ob sie für Food11 tatsächlich helfen — das wird durch den
NAS-Lauf empirisch ermittelt (eine Pareto-Front die bevorzugt SPT/LSA-Architekturen
enthält, wäre ein Befund für Kapitel 5).

#### Warum pretrained auf ImageNet statt random initialization

Fine-Tuning eines ImageNet-vortrainierten Modells ist für kleine Datensätze klar
überlegen gegenüber Training from scratch. Das vortrainierte Modell kennt allgemeine
visuelle Konzepte (Kanten, Texturen, Formen, semantische Konzepte), die für die
Lebensmittel-Klassifikation in Food11 nützlich sind. Bei ~10k Trainingsbildern würde
ein ViT-Small from scratch massiv overfitting zeigen und nicht die Accuracy eines
feingefeinten vortrainierten Modells erreichen.

#### Designentscheidung: `enable_spt()` als Post-hoc-Modifikation

SPT wird nicht durch eine andere `timm`-Architektur realisiert, sondern durch
direktes Austauschen des `model.patch_embed` Attributs nach dem Laden des
vortrainierten Modells. Dies hat einen wichtigen Nebeneffekt: Die pretrained Gewichte
von `patch_embed` (der Original-Conv2d mit in_channels=3) sind für SPT nicht mehr
kompatibel (in_channels=15). Das Modell startet also für den SPT-Teil bei
zufälligen Gewichten, während alle anderen Teile (Transformer-Blöcke, Klassifikationskopf)
von den ImageNet-Gewichten profitieren. Das ist eine pragmatische, in der Literatur
übliche Kompromisslösung.

---

### 3. CNN-Baselines (`src/models/cnn_baseline.py`) — Gliederung 2.5

#### Was

`build_cnn_baseline(model_name, num_classes)` ist ein dünner Wrapper um
`timm.create_model(model_name, pretrained=True, num_classes=num_classes)`. Drei Modelle:

| Modell-Name       | timm-ID                         | Parameter | ImageNet Top-1 |
| ----------------- | ------------------------------- | --------- | -------------- |
| ResNet-18         | `resnet18.a1_in1k`              | 11.7 M    | ~71.5 %        |
| EfficientNet-B0   | `efficientnet_b0.ra_in1k`       | 5.3 M     | ~77.7 %        |
| MobileNetV3-Large | `mobilenetv3_large_100.ra_in1k` | 5.5 M     | ~75.8 %        |

Alle drei werden vollständig fine-getuned (kein Freezing), mit identischer
Trainings-Pipeline wie der ViT (gleicher AdamW-Optimierer, gleiche LR, gleicher
Scheduler, gleiche Augmentierung). Lernrate: 1e-3 (höher als beim ViT, da CNNs
auf kleinen Datensätzen robuster gegen hohe LRs sind und schneller konvergieren).

#### Warum genau diese drei Modelle

Die drei Modelle decken unterschiedliche Klassen moderner CNN-Architekturen ab:

- **ResNet-18:** Klassische Skip-Connection-Architektur, einfach und gut verstanden.
  Dient als "minimale Baseline" ohne architektonische Tricks.
- **EfficientNet-B0:** Compound Scaling (gleichzeitige Skalierung von Breite, Tiefe
  und Auflösung), entwickelt mit Neural Architecture Search. Besonders parametereffizient.
- **MobileNetV3-Large:** Depthwise Separable Convolutions + SE-Module, optimiert für
  mobile Inferenz. Ähnliche Parameterzahl wie EfficientNet-B0, aber anderes
  Architektur-Paradigma.

Diese drei Modelle erlauben in der Evaluation drei unterschiedliche Aussagen:

- ViT vs. klassischer CNN (ResNet)
- ViT vs. NAS-optimierter CNN (EfficientNet)
- ViT vs. Effizienz-optimierter CNN (MobileNet)

Alle drei wurden ebenfalls auf ImageNet vortrainiert, was den Vergleich fair macht.

---

### 4. Fitnessfunktion / Pareto-Dominanz (`src/nas/pareto.py`) — Gliederung 4.4

#### Was

Das Modul implementiert Multi-Objective-Optimierung mit zwei Zielen:

- **Ziel 1:** `val_acc` maximieren (Genauigkeit auf dem Validierungsset)
- **Ziel 2:** `params` minimieren (Parameteranzahl als Proxy für Modellgröße/Inferenzkosten)

Drei Funktionen:

**`dominates(a, b)`:** Architektur `a` dominiert `b` genau dann, wenn:

    a dominiert b ⟺ (val_acc(a) ≥ val_acc(b)) ∧ (params(a) ≤ params(b))
                    ∧ (val_acc(a) > val_acc(b) ∨ params(a) < params(b))

d. h. `a` ist in mindestens einem Ziel strikt besser und in keinem Ziel schlechter.

**`get_pareto_front(population)`:** Gibt alle nicht-dominierten Architekturen zurück
(Pareto-Rang 0): die Menge aller Architekturen, für die kein anderes Mitglied der
Population existiert, das sie dominiert.

**`non_dominated_sort(population)`:** Implementiert die NSGA-II Non-Dominated-Sortierung:
Die Population wird iterativ in Fronten eingeteilt. Front 0 sind alle nicht-dominierten
Architekturen. Aus der Restpopulation wird erneut die Pareto-Front gebildet (= Front 1),
usw. Das Ergebnis ist eine geordnete Liste von Fronten, wobei Front 0 die qualitativ
besten Architekturen enthält.

**`crowding_distance(front)`:** Für jede Architektur in einer Front wird die
Crowding Distance berechnet — ein Maß dafür, wie "isoliert" sie bezüglich der
Zielwerte ist. Für jedes Ziel m wird sortiert und die Distanz berechnet als:

    CD(i, m) = (f_m(i+1) - f_m(i-1)) / (f_m_max - f_m_min)

Die Gesamt-Crowding-Distance ist die Summe über alle Ziele. Randpunkte (kleinste
und größte Werte in einem Ziel) erhalten CD = ∞.

#### Warum Multi-Objective statt skalarer Fitness

Im Hauptprojekt wurde eine gewichtete Summe als Fitness verwendet:

    fitness = val_acc - λ·params - μ·time

Das Problem: Die Gewichte λ und μ bestimmen maßgeblich, welche Architekturen
bevorzugt werden. Ein zu großes λ bestraft SPT-Architekturen (mehr Parameter durch
15-Kanal-Patch-Embed) systematisch, unabhängig von ihrer tatsächlichen Accuracy.
Ein zu kleines λ ignoriert den Compute-Aufwand vollständig. Jede Wahl ist arbiträr
und schwer zu begründen.

Die Pareto-Front macht diese Gewichtsentscheidung überflüssig: Alle
Kompromiss-Architekturen zwischen "maximaler Accuracy" und "minimalen Parametern"
sind gleichwertig optimal und verbleiben in der Selektionsmenge. Die Entscheidung,
welche Architektur final eingesetzt wird, hängt vom konkreten Anwendungsfall ab (z. B.
Deployment auf einem ressourcenbeschränkten Gerät → kleinste Architektur auf der
Pareto-Front; Deployment auf einem Server → Architektur mit maximaler Accuracy).

#### Warum Crowding Distance

Ohne Crowding Distance würde NSGA-II bei gleicher Front-Zugehörigkeit zufällig
selektieren, was zu Häufungen in "dichten" Regionen der Pareto-Front führt. Die
Crowding Distance bevorzugt Architekturen in wenig besetzten Regionen — das hält
die Population divers und verbessert die Abdeckung der gesamten Pareto-Front statt
nur einzelner Punkte. Konkret wird in `run_evolution()` die Crowding Distance als
Tie-Breaker verwendet: wenn eine Front nicht vollständig in den Eltern-Pool passt,
werden bevorzugt Architekturen mit hoher Crowding Distance (= isolierte Punkte)
übernommen.

---

### 5. Zero-Cost Proxies (`src/proxies/`) — Gliederung 2.3, 4.5

#### Was und mathematische Grundlage

Zero-Cost Proxies schätzen die Qualität einer Architektur **ohne Training** —
typischerweise mit einem einzigen Forward- oder Backward-Pass.

**SynFlow (`synflow.py`):**
SynFlow (Tanaka et al., 2020) misst den "synaptischen Fluss" durch das Netzwerk.
Die Idee: Gute Architekturen nutzen alle ihre Parameter effektiv; in schlechten
Architekturen "kollabieren" bestimmte Layer (ihre Gradienten werden Null, sogenannter
Layer Collapse).

Um Layer Collapse zu vermeiden, werden zunächst alle Parameter durch ihren Betrag
ersetzt: θ̃ = |θ|. Dann wird ein Forward-Pass mit einem All-Ones-Input durchgeführt
(keine echten Daten nötig) und der Loss als Summe aller Outputs definiert:

    R(θ̃) = 1^T · f(θ̃, 1)

Der SynFlow-Score ist:

    SynFlow(θ) = Σ_l Σ_i |θ_l^i · ∂R/∂θ_l^i|

Mit |θ| im Forward-Pass sind alle Intermediate-Aktivierungen positiv, wodurch kein
Gradient durch Vorzeichenwechsel annulliert werden kann. Nach der Score-Berechnung
werden die ursprünglichen Vorzeichen wiederhergestellt.

**NASWOT (`naswot.py`):**
NASWOT (Mellor et al., 2021) misst, wie gut eine Architektur verschiedene Inputs
voneinander unterscheiden kann — ebenfalls ohne Training.

Für einen Batch von B Bildern werden Forward-Hooks auf alle ReLU/GELU-Aktivierungen
registriert. Für jede Aktivierungsschicht wird eine binäre Aktivierungsmaske erzeugt:

    K_binary(x_i) = 1[σ(x_i) > 0]  (elementweise)

Die binären Aktivierungsmuster aller B Samples werden zu einem Matrix-Vektor
der Länge |F| (Gesamtzahl der Aktivierungen) konkateniert. Daraus wird die
Kernel-Matrix K ∈ ℝ^{B×B} gebildet:

    K_{ij} = <K_binary(x_i), K_binary(x_j)>  (Dot-Produkt)

Der NASWOT-Score ist:

    NASWOT = log|det(K + ε·I)|  (ε = 1e-6 für numerische Stabilität)

Interpretation: Ein hoher Score bedeutet, dass die Aktivierungsmuster der B Samples
möglichst verschieden sind — das Netzwerk nutzt seinen Repräsentationsraum gut aus
und kann gut diskriminieren, bevor es je trainiert wurde.

**GradNorm (`gradnorm.py`):**
GradNorm ist das einfachste der drei Proxies: Ein normaler Forward- und Backward-Pass
mit echten Daten und echten Labels, gefolgt von der L2-Norm aller Parametergradienten:

    GradNorm(θ) = ||∇_θ L(θ)||_2 = √(Σ_l Σ_i (∂L/∂θ_l^i)²)

Die Intuition: Architekturen mit großen Gradienten bei Initialisierung haben ein
stärkeres Lernsignal und trainieren in der Regel schneller und besser als Architekturen
mit sehr kleinen Gradienten (Vanishing Gradient Problem).

#### Warum genau diese drei Proxies

Die drei Proxies wurden nach zwei Kriterien ausgewählt:

1. **Diversität der Messkategorie:** SynFlow misst globalen Informationsfluss durch
   Parametergewichte (kein echter Input nötig). NASWOT misst Aktivierungsdiversität
   über einen Batch. GradNorm misst die Gradient-Signalstärke. Die drei messen
   komplementäre Eigenschaften — ein Ensemble verspricht robustere Rankings als ein
   einzelner Proxy.

2. **Etablierung in der Literatur:** Alle drei erscheinen prominent im
   Zero-Cost-NAS-Benchmark (Kolmogorov et al., 2021), der auf mehreren
   Architektur-Suchräumen zeigt, dass kein einzelner Proxy durchgängig besser als
   andere ist — Ensembles sind konsistent stabiler.

Die Schwäche aller drei Proxies ist ihre schwache Korrelation auf kleinen Datensätzen
(Food11 mit ~10k Bildern) im Vergleich zu ImageNet-Skala. Das ist bekannt und
motiviert gerade den empirischen Vergleich (`run_proxy_validation.py`).

#### Wie die Proxies kombiniert werden (Rangsummen-Ensemble)

`evaluate_proxies()` gibt einen Dict mit allen drei Scores zurück. In `run_evolution()`
werden die Scores nicht direkt addiert (unterschiedliche Skalen: SynFlow ~60,
NASWOT ~200–400, GradNorm ~10–30), sondern über eine Rang-Transformation normiert:

    _rank(scores): sortiere Architekturen nach Score, weise Rang 0 (= bester) bis
    n-1 (= schlechtester) zu

Die Rangsumme für jede Architektur i ist:

    combined_rank(i) = rank_SynFlow(i) + rank_NASWOT(i) + rank_GradNorm(i)

Architekturen mit kleiner Rangsumme werden für das vollständige Proxy-Training
ausgewählt (Top `proxy_keep_frac`, Standard 50 %). Dieses Verfahren ist
skaleninvariant und benötigt keine Gewichtsentscheidung zwischen den Proxies.

**Alternative: gewichtetes Ensemble.** Es wäre möglich, die Proxies mit Gewichten
entsprechend ihrer gemessenen Korrelation (ρ_NASWOT=0.444 > ρ_SynFlow=0.192 >
ρ_GradNorm≈0) zu kombinieren. Das wurde für eine zukünftige Ablation vorgesehen,
aber nicht als Standard implementiert, um den Suchprozess nicht von
Proxy-Validierungsergebnissen abhängig zu machen (Zirkularität).

---

### 6. Evolutionärer Algorithmus (`src/nas/evolution.py`, `mutation.py`, `crossover.py`) — Gliederung 4.6

#### 4.6 Evolutionärer Suchalgorithmus

Der Suchalgorithmus realisiert eine Generation-basierte Evolutionsstrategie nach
dem NSGA-II-Prinzip (Non-dominated Sorting Genetic Algorithm II, Deb et al. 2002).
Er operiert auf einer Population von Konfigurationen — im Folgenden _Individuen_
genannt — und verbessert die Population iterativ durch Selektion, Crossover und
Mutation.

---

#### 4.6.1 Populationsinitialisierung und Generationsschleife

Zu Beginn wird eine Population der Größe `P = 15` durch gleichverteiltes Zufalls-
sampling aus dem Suchraum erzeugt. Jedes Individuum ist eine vollständige
Konfiguration mit neun Parametern. Die Population wird für `G = 8` Generationen
weiterentwickelt.

In jeder Generation durchläuft die Population vier Schritte: Proxy-Filterung,
Proxy-Training und Fitness-Messung, Pareto-Selektion und Offspring-Erzeugung.
Alle Schritte sind in Abschnitt 4.2.1 als Phasen 3–6 überblicksartig beschrieben;
dieser Abschnitt beschreibt die algorithmischen Details.

---

#### 4.6.2 Selektion: NSGA-II mit Crowding Distance

Nach der Fitness-Evaluierung werden alle bisher bewerteten Individuen gemeinsam
nach Pareto-Dominanz sortiert. Zwei Ziele werden minimiert bzw. maximiert:
Modellgröße (Parameteranzahl, minimieren) und Validierungsgenauigkeit (maximieren).
Individuum _a_ dominiert Individuum _b_, wenn gilt:

```
val_acc(a) ≥ val_acc(b)  und  params(a) ≤ params(b)
mit mindestens einer strikten Ungleichung
```

Alle nicht-dominierten Individuen bilden Rang-1 (die aktuelle Pareto-Front). Aus
der verbleibenden Menge wird erneut die nicht-dominierte Front bestimmt (Rang 2),
und so weiter. Die Population der nächsten Generation wird front-weise aufgefüllt:
zuerst Rang 1, dann Rang 2 usw. Wenn ein Rang die Population überfüllen würde,
wird innerhalb dieses Rangs nach _Crowding Distance_ sortiert: einem Maß für den
mittleren Abstand eines Individuums zu seinen Nachbarn im Zielfunktionsraum.
Individuen mit hoher Crowding Distance werden bevorzugt, da sie seltener besetzte
Regionen der Pareto-Front repräsentieren und die Populationsdiversität sichern.

---

#### 4.6.3 Crossover-Operator

Für jedes neue Individuum werden zwei Eltern zufällig aus dem Pareto-Pool gezogen.
Das Kind-Individuum entsteht durch _Uniform Crossover_: Jeder der neun Parameter
wird unabhängig voneinander mit gleicher Wahrscheinlichkeit vom ersten oder zweiten
Elternteil übernommen. Im Erwartungswert trägt jedes Elternteil 4,5 Parameter bei.

Uniform Crossover wurde gegenüber Ein-Punkt- oder Zwei-Punkt-Crossover gewählt,
weil die neun Parameter keine inhärente Reihenfolge haben — es gibt keine biologische
Analogie zu "benachbarten" Genen. Uniform Crossover behandelt alle Parameter
gleichwertig und erlaubt beliebige Kombinationen beider Elternteile.

Das Hauptprojekt verwendete kein Crossover — neue Kandidaten entstanden ausschließlich
durch Mutation. Crossover ermöglicht es, Merkmale aus zwei unabhängig gut performenden
Regionen des Suchraums zu kombinieren, was reine Mutation (lokale Nachbarschaftssuche)
nicht leisten kann.

---

#### 4.6.4 Mutations-Operator

Nach dem Crossover wird jeder Parameter des Kindes unabhängig mit
Mutations-Wahrscheinlichkeit `p_mut = 0.3` mutiert. Bei einer Mutation wird
ein anderer Wert als der aktuelle aus den Optionen des jeweiligen Parameters
gleichverteilt gezogen — der aktuelle Wert ist dabei explizit ausgeschlossen,
sodass jede Mutation eine echte Veränderung bewirkt. Bei neun Parametern und
`p_mut = 0.3` werden im Erwartungswert 2,7 Parameter pro Individuum verändert.

Nach Crossover und Mutation wird die Gültigkeit des Genoms geprüft (z. B.
`num_heads` muss `embed_dim=384` ganzzahlig teilen). Ungültige Konfigurationen
werden verworfen und neu erzeugt. Bereits evaluierte Konfigurationen (Duplikate)
werden ebenfalls verworfen.

---

#### 4.6.5 Begründung der Hyperparameter

**Populationsgröße `P = 15`:** Zu kleine Populationen (< 10) haben zu wenig
Diversität und konvergieren frühzeitig zu lokalen Optima. Zu große Populationen
(> 30) sind bei 12 Proxy-Epochen pro Kandidat und beschränktem Compute-Budget
nicht praktikabel. `P = 15` ist ein in der evolutionären NAS-Literatur etablierter
Startwert für begrenzte Budgets (Liu et al., 2018; Real et al., 2019).

**Generationsanzahl `G = 8`:** Mit `P = 15` und `G = 8` werden maximal 120
Architekturen evaluiert (ohne Proxy-Filter; mit Filter ~60). Studien zu evolutionärem
NAS mit vergleichbarem Budget berichten typischerweise Konvergenz nach 5–10
Generationen. `G = 8` liegt in diesem Bereich und hält den Gesamtaufwand unter
~12 GPU-Stunden.

**Mutations-Wahrscheinlichkeit `p_mut = 0.3`:** Eine zu niedrige Rate (< 0.1)
führt zu früher Konvergenz, weil Offspring-Individuen zu ähnlich zu den Eltern
sind. Eine zu hohe Rate (> 0.5) negiert den Selektionsdruck, da zu viele Eltern-
Gene verloren gehen. `p_mut = 0.3` ergibt bei neun Parametern im Erwartungswert
2,7 Änderungen pro Mutation — aggressiv genug für echte Exploration, konservativ
genug um nützliche Elternmerkmale zu erhalten.

**Proxy-Filterrate `proxy_keep_frac = 0.5`:** 50 % der Kandidaten werden durch
Proxy-Scoring ohne Training aussortiert. Der empirisch gemessene Spearman-Korrelations-
koeffizient der Proxies (ρ ≈ 0.44, vgl. Abschnitt 4.5) rechtfertigt eine moderate
Filterung. Eine aggressivere Rate (z. B. 25 % behalten) würde voraussetzen, dass
die Proxies sehr zuverlässig die schlechtesten Architekturen identifizieren — was
bei ρ ≈ 0.44 nicht gegeben ist.

---

### 7. Weight Inheritance (`src/nas/evaluate.py`) — Gliederung 4.7

#### Was

`evaluate_cfg()` prüft vor dem Training, ob ein Elternmodell kompatibel ist, und
lädt in diesem Fall dessen Gewichte:

    is_compatible(cfg_a, cfg_b) ⟺ ∀ key ∈ {depth, num_heads, use_spt, use_lsa}:
        cfg_a[key] == cfg_b[key]

Falls kompatibel: `model.load_state_dict(parent_state_dict, strict=False)`

Das Kind-Modell startet das Proxy-Training also nicht bei den ImageNet-Gewichten,
sondern bei den Gewichten des Elternmodells nach dessen Proxy-Training.

Der eigene `_state_dict` wird nach dem Training gespeichert und in den Rückgabe-Dict
aufgenommen, damit Kindmodelle der nächsten Generation davon profitieren können.

#### Die vier Kompatibilitätsbedingungen und ihre Begründung

**`depth`:** Unterschiedliche Depth = unterschiedliche Anzahl Transformer-Blöcke =
unterschiedliche State-Dict-Struktur (`blocks.0`, ..., `blocks.depth-1`). Ein
Elternmodell mit depth=12 hat Tensoren `blocks.10.*`, `blocks.11.*`, die in einem
Kind mit depth=8 schlicht nicht existieren. `strict=False` würde diese Keys ignorieren
(kein Crash), aber die vorhandenen Blöcke 0–7 würden korrekt übernommen. Dies ist
ein sinnvoller Kompromiss: Der gemeinsame Kern wird vererbt.

**`num_heads`:** `num_heads` bestimmt die Dimension des QKV-Projektions-Tensors:
`W_qkv ∈ ℝ^{embed_dim × 3·embed_dim}`, wobei die Gewichte intern in `num_heads`
Blöcke aufgeteilt werden. Unterschiedliche `num_heads` bedeuten unterschiedliche
interne Tensor-Shapes im State Dict → Shape-Mismatch → `load_state_dict` mit
`strict=False` würde diese Layer vollständig ignorieren, was Weight Inheritance
ineffektiv macht.

**`use_spt`:** SPT ändert `patch_embed` von Conv2d(in_channels=3) zu
Conv2d(in_channels=15). Die Gewicht-Tensor-Shape ist grundlegend verschieden
(Faktor 5 in der Eingangsdimension). Inkompatibel.

**`use_lsa`:** LSA ersetzt den Standard-Attention-Block durch `LSA_Attention`
mit zusätzlichem `temperature`-Parameter (pro Head ein Skalar). Ein Standard-Eltern
hat kein `temperature`-Tensor im State Dict; ein LSA-Kind würde dieses zufällig
initialisiert lassen. Der umgekehrte Fall (LSA-Eltern, Standard-Kind) würde unbekannte
Keys erzeugen. Beide Fälle sind technisch handhabbar mit `strict=False`, aber konzeptuell
problematisch — deshalb Inkompatibilitäts-Deklaration.

#### Warum Weight Inheritance wichtig ist

Proxy-Training mit 12 Epochen ist für viele Architekturen zu kurz für vollständige
Konvergenz. Ein Kind, das von den Eltern-Gewichten startet, hat nach 12 Epochen
eine realistischere Val-Accuracy als ein Kind, das bei ImageNet-Gewichten startet
und erst den Transfer-Learning-Prozess beginnt. Ohne Weight Inheritance unterschätzt
das Proxy-Training systematisch Architekturen, die in ihrer Initialphase langsam
lernen (z. B. tiefere Netze oder `finetune_mode=full`).

Der Bug im Hauptprojekt (Gewichte wurden extrahiert aber nie übergeben) ist deshalb
nicht trivial: Er führt dazu, dass spät-lernende Architekturen aus der Pareto-Front
herausfallen, obwohl sie bei vollem Training gut wären.

---

### 8. Trainings-Pipeline (`src/utils/training.py`) — Gliederung 4.8

#### Was

`run_training()` trainiert ein gegebenes Modell mit folgenden Komponenten:

**Optimizer:** AdamW mit den konfigurierten `lr` und `weight_decay`-Werten aus der
Config. Nur Parameter mit `requires_grad=True` werden übergeben (respektiert
`finetune_mode`).

**Mixed Precision (AMP):** `torch.amp.autocast` (nur auf CUDA aktiv, MPS/CPU
laufen in fp32) kombiniert mit `torch.amp.GradScaler`. AMP berechnet Forward-Pass
und Loss in fp16, hält aber kritische Operationen (Batch-Norm, Loss-Reduktion) in
fp32 und skaliert den Loss, um Underflow bei fp16-Gradienten zu verhindern. Auf
A100 mit fp16-Tensor-Cores ist die Beschleunigung signifikant (~1.5–2×).

**LR-Schedule:** CosineAnnealingLR mit T_max=epochs und η_min=1e-6:

    η_t = η_min + 0.5 · (η_max - η_min) · (1 + cos(π · t / T_max))

Die Lernrate beginnt bei `lr`, fällt kosinusförmig auf 1e-6 ab. Für Fine-Tuning
von ViTs ist Cosine Annealing Standard, da es am Ende des Trainings sehr feine
Anpassungen ermöglicht ohne dass die Lernrate plötzlich auf Null springt.

**Early Stopping:** Bei gesetztem `patience`-Parameter wird das Training abgebrochen,
wenn sich die Val-Accuracy über `patience` aufeinanderfolgende Epochen nicht um
mehr als 1e-4 verbessert. Das beste Modell (höchste Val-Accuracy) wird gespeichert
und nach dem Training wiederhergestellt. Im Proxy-Training (12 Epochen) ist Early
Stopping optional, bei vollem Training (50 Epochen) mit patience=15 aktiv.

**Mixup:** Per Batch wird mit Wahrscheinlichkeit 1 (wenn alpha > 0) eine zufällige
Mischrate λ ~ Beta(alpha, alpha) gezogen. Zwei Bilder im Batch werden linear
interpoliert:

    x̃ = λ · x_i + (1-λ) · x_j
    L = λ · CE(logits, y_i) + (1-λ) · CE(logits, y_j)

**CutMix:** Mit Wahrscheinlichkeit `cutmix_prob` wird CutMix statt Mixup angewendet.
Aus λ ~ Beta(1,1) (=Uniform[0,1]) wird eine zufällige Bounding Box berechnet:

    cut_ratio = √(1 - λ)
    Box = [cx±cut_w/2, cy±cut_h/2]  (geclippt auf Bildgrenzen)

Ein rechteckiger Bereich des ersten Bildes wird durch den entsprechenden Bereich
des zweiten ersetzt. Das effektive λ wird aus der tatsächlichen überschriebenen
Fläche neu berechnet:

    λ_eff = 1 - (Box_Fläche / Bild_Fläche)

**Best-Model-Tracking:** Das Modell der besten Val-Accuracy wird als CPU-Kopie
gespeichert (`{k: v.cpu().clone() for k, v in model.state_dict().items()}`), um
GPU-Speicher zu sparen. Nach dem Training wird dieses State-Dict geladen.

#### Warum AdamW statt Adam oder SGD

**Adam vs. AdamW:** Standard Adam koppelt Weight Decay (L2-Regularisierung) mit
dem adaptiven Lernraten-Mechanismus, was bei Momentum-Schätzungen zu inkonsistenter
Regularisierung führt. AdamW entkoppelt Weight Decay vom Gradient-Update (Loshchilov
& Hutter, 2019) und ist deshalb für Transformer-Modelle Standard.

**AdamW vs. SGD:** SGD mit Momentum konvergiert bei Transformern deutlich langsamer
als AdamW und ist empfindlicher gegenüber der Lernraten-Wahl. Für ein NAS-Experiment
mit beschränkten Proxy-Epochen ist AdamW die zuverlässigere Wahl.

---

### 9. Daten-Pipeline (`src/utils/data.py`) — Gliederung 5.1

#### Was

**Food11-Datensatz:** 16.643 Bilder in 11 Lebensmittelkategorien (Bread, Dairy
Product, Dessert, Egg, Fried Food, Meat, Noodles/Pasta, Rice, Seafood, Soup,
Vegetable/Fruit), aufgeteilt in drei vorgegebene Splits:

| Split      | Bilder | Verwendung                  |
| ---------- | ------ | --------------------------- |
| training   | 9.866  | Training                    |
| validation | 3.430  | Modellselektion (Proxy Val) |
| evaluation | 3.347  | Finale Test-Evaluation      |

Alle Bilder werden auf 224×224 Pixel resized (bicubic, von `transforms.Resize`).
Normalisierung mit ImageNet-Statistiken (μ=(0.485, 0.456, 0.406),
σ=(0.229, 0.224, 0.225)), da das Modell auf ImageNet vortrainiert ist.

**Trainings-Augmentierungen:** RandomHorizontalFlip (p=0.5) +
RandomResizedCrop(224, scale=(0.8, 1.0)). Keine starken Augmentierungen wie
ColorJitter oder RandAugment — die Stärke der Augmentierung (Mixup/CutMix)
ist Teil des Suchraums.

**Subsampling:** `subsample_frac ∈ (0, 1]` zieht eine reproduzierbare
Teilmenge der Trainingsindizes (seed-fixiert via `numpy.random.default_rng(seed)`).
Validation- und Evaluation-Split bleiben vollständig, damit die Metric-Vergleichbarkeit
über alle Datensatzgrößen gewährleistet ist.

**DataLoader:** `num_workers=4`, `persistent_workers=True` (wenn workers > 0),
`pin_memory=True` (wenn CUDA verfügbar). Auf dem Remote-Server wurde die
Trainings-Data lokal auf `/tmp/food11` kopiert (NFS war ~18× langsamer als
lokaler Speicher für sequentielles Image-Lesen mit 4 parallelen Worker-Prozessen
× 4 Container).

#### Warum Food11 als primärer Datensatz

Food11 wurde als primärer Datensatz gewählt, weil:

1. **Vorgegebene Splits:** Kein eigenes Train/Val/Test-Splitting nötig, reproduzierbare
   Vergleiche ohne zufällige Datenteilung.
2. **Mittlere Größe:** ~10k Trainingsbilder sind groß genug für Fine-Tuning aber
   klein genug, dass die Skalierungsfrage (F1) interessant ist. Bei Datensätzen
   mit >100k Bildern würde der ViT klar dominieren, die Forschungsfrage wäre trivial.
3. **Domäne:** Lebensmittelklassifikation ist eine reale Anwendungsdomäne, für die
   pretrained ImageNet-Features eingeschränkt nützlich sind (ImageNet enthält zwar
   Lebensmittel, aber keine gezielte Annotation). Das macht den Fine-Tuning-Vergleich
   zwischen ViT und CNN interessant.

---

### 10. Experiment-Skripte (`scripts/*.py`) — Gliederung 5.2

| Skript                    | Zweck                                                                                                                                                                       |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_nas.py`              | CLI-Wrapper für `run_evolution()`. Konfigurierbar: `pop-size`, `generations`, `proxy-epochs`, `use-proxies`, `proxy-keep-frac`, `data-root`, `results-dir`.                 |
| `run_proxy_validation.py` | Sampelt `n_configs` zufällige Architekturen, misst alle drei Proxy-Scores, trainiert jeden für `proxy_epochs` Epochen, berechnet Spearman-ρ.                                |
| `run_f1_experiment.py`    | Trainiert ViT und drei CNNs auf sechs `subsample_frac`-Werten (0.1–1.0), 30 Epochen. Output: `f1_scaling.csv`.                                                              |
| `retrain_best.py`         | Vollständiges Training einer per CLI spezifizierten Config (50 Epochen), inkl. Test-Set-Evaluation. Speichert Modellgewichte + History.                                     |
| `retrain_analysis.py`     | Liest eine NAS-CSV, selektiert Top-N und Bottom-N nach Proxy-Val-Accuracy, trainiert alle vollständig (50 Epochen). Ausgabe: Vergleichstabelle Proxy-Rang vs. finaler Rang. |

**Warum eigene Skripte statt Notebooks für lange Trainings:** Die Notebooks
(`notebooks/experimente_*.ipynb`) dienen dem interaktiven Testen. Für Experimente
mit Laufzeiten von 4–8 Stunden auf Remote-GPUs sind Skripte überlegen:
(a) kein Absturz der Notebook-Kernel-Session trennt den laufenden Prozess,
(b) `argparse` macht Parameter reproduzierbar dokumentiert,
(c) CSV-Schreibung nach jeder Architektur verhindert Datenverlust bei Abbruch.

---

### 11. Infrastruktur (`Dockerfile`, `.dockerignore`, Remote-Server) — Gliederung 5.2

#### Docker-Image

Base-Image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`. Diese Version wurde
gewählt, weil:

- **CUDA 12.4** ist kompatibel mit dem NVIDIA-Treiber 580 auf dem Server
  (CUDA 13.0 driver, rückwärtskompatibel mit CUDA 12.x Containern).
- **Runtime-Image** (nicht `devel`): Enthält CUDA Runtime und cuDNN, aber keine
  Compiler/Header — kleiner, ausreichend für PyTorch-Inferenz und Training.
- PyTorch 2.5.1 unterstützt `torch.amp.autocast(device_type=...)` mit der
  generischen Schnittstelle (statt des alten `torch.cuda.amp`).

Umgebungsvariablen im Container:

- `PYTHONPATH=/workspace`: Stellt sicher, dass `from src.models.vit import ...`
  ohne relative Pfad-Probleme funktioniert.
- `PYTHONUNBUFFERED=1`: Verhindert Python-Output-Buffering in Docker-Detached-Mode,
  damit `docker logs` Epoch-Output sofort anzeigt (nicht erst wenn der 8kB-Buffer voll ist).
- `HF_HOME=/mnt/data/mats_adel/hf_cache` (per `-e` bei `docker run`): Shared
  Cache für HuggingFace/timm-Pretrained-Weights über alle Container hinweg.

`.dockerignore` schließt aus: `data/` (Food11, ~2 GB), `masterenv/` (virtuelle
Python-Umgebung, nicht nötig im Container), `results/` (werden per Volume gemountet),
`.git/`, `__pycache__/`, `.ipynb_checkpoints/`, `*.pdf`.

#### Remote-GPU-Cluster (HAW Hamburg, ML AG)

| Host            | GPU(s)                   | VRAM   | CPU        | RAM     |
| --------------- | ------------------------ | ------ | ---------- | ------- |
| gpu-a100        | 4× NVIDIA A100-SXM4-40GB | 4×40GB | 2×32 Kerne | 1024 GB |
| gpu-h100-01     | 1× NVIDIA H100           | 80GB   | 2×24 Kerne | 512 GB  |
| gpu-v100s-01/03 | 1× NVIDIA V100s          | 32GB   | 16 Kerne   | 128 GB  |

Zugang: FortiClient VPN (SSL-VPN, Gateway `connect.informatik.haw-hamburg.de:6443`,
Profil `/ml`), danach SSH mit infw-Kennung. Reservierung über MRBS
(Meeting-Room-Booking-System, https://mrbs-mlag.informatik.haw-hamburg.de).

Netzwerkspeicher: `/mnt/data/` (NFS, 60 TB), persistente Speicherung unter
`/mnt/data/mats_adel/`. Lokaler Speicher (`/tmp/`) für temporäre Daten (z. B.
Food11 für Training), da NFS-I/O bei parallelem Lesen aus mehreren Containern
einen starken Flaschenhals darstellt.

**Ressourcen-Zuteilung für die Experimente:**
Auf dem A100-Server wurden alle vier Experimente gleichzeitig gestartet, je auf
einer eigenen GPU (`--gpus '"device=X"'`). Dadurch laufen vier unabhängige
Python-Prozesse parallel, ohne sich GPU-Rechenzeit zu teilen.

---

## Teil B — Experiment-Log

> Format pro Eintrag: Datum, Was wurde gemacht, Warum (welche Frage/welches Risiko
> adressiert dieser Lauf), Wie (Parameter/Kommando), Ergebnis, Interpretation,
> nächste Schritte.

---

### 2026-07-07 — Smoke Tests (Pipeline-Validierung, keine echten Ergebnisse)

**Was:** Zwei kurze NAS-Läufe zur End-to-End-Validierung der Pipeline:

- `results/nas_smoke_test.csv`: NAS ohne Proxy-Filter (pop=3, gen=2, proxy_epochs=2)
- `results/nas_smoke_proxies.csv`: NAS mit Proxy-Filter (use_proxies=True,
  proxy_keep_frac=0.5, sonst gleich)

**Warum:** Vor den langen, ressourcenintensiven Experimenten auf dem Remote-GPU-Cluster
musste sichergestellt werden, dass keine Laufzeitfehler auftreten. Spezifisch getestet:

- Korrekte Ausführung von Weight Inheritance (Gewichte werden tatsächlich übertragen)
- Keine Shape-Fehler bei `load_state_dict(strict=False)` für inkompatible Architekturen
- Korrekte Proxy-Filterung (7/15 oder ähnlich Architekturen behalten)
- CSV wird korrekt geschrieben (alle Spalten vorhanden, keine fehlenden Werte)
- Pareto-Selektion funktioniert auch bei Population aus nur 2 Generationen

**Ergebnis:** Beide Läufe liefen ohne Absturz durch. Val-Accuracies stark streuend
(0.30 bis 0.91), was bei 2 Epochen und zufälligen Architekturen erwartet wird.
Auffällig: Ein Sprung von Gen 0 (max. 0.38) zu Gen 1 (0.91) in `nas_smoke_proxies.csv`
— dieser Sprung ist auf Weight Inheritance zurückzuführen: das Kind-Modell in Gen 1
startete von den Eltern-Gewichten und erreichte daher in 2 Epochen eine deutlich
höhere Accuracy als Gen-0-Modelle.

**Interpretation:** Pipeline ist funktionsfähig. Keine inhaltliche Aussage
über Architektur-Qualität aus diesen Läufen ableitbar.

---

### 2026-07-16 — Erster echter Experiment-Tag auf dem HAW-GPU-Cluster

#### Infrastruktur-Setup und aufgetretene Probleme

**Hardware:** NVIDIA A100-SXM4-40GB (4×), NVIDIA-Treiber 580.126.16, CUDA 13.0.

Das Setup verlief mit mehreren Problemen, die dokumentiert werden, weil sie
reproduzierbare Lösungen erfordern und für zukünftige Remote-GPU-Experimente
(z. B. spätere NAS-Ablationen) direkt verwendbar sind:

**Problem 3 — Docker Shared Memory (shm):**
DataLoader mit `num_workers=4` crashte im Container mit:
`ERROR: Unexpected bus error — insufficient shared memory (shm)`.
Ursache: Docker begrenzt Shared Memory standardmäßig auf 64 MB. PyTorch DataLoader
mit mehreren Worker-Prozessen nutzt `/dev/shm` für die Inter-Process-Kommunikation
beim Weitergeben von Batches an den Haupt-Prozess. Lösung: `--shm-size=8g` in allen
`docker run`-Aufrufen.

**Problem 4 — NFS als Flaschenhals:**
Nach dem Start der vier Container war die GPU-Auslastung nach 45 Minuten noch immer
nahe 0 % und kein Ergebnis erschien. Diagnose: Alle Container lasen Food11
gleichzeitig vom NFS-Mount (`/mnt/data/mats_adel/food11`), was bei 4 Containern
× 4 DataLoader-Workers = 16 parallele NFS-Reader-Prozesse zu extremem I/O-Bottleneck
führte. Messung:

- 50 Batches aus NFS: >50 Sekunden
- 50 Batches aus lokalem Speicher (`/tmp/food11`): **2.73 Sekunden** (~18× schneller)

Lösung: `cp -r /mnt/data/mats_adel/food11 /tmp/food11` vor dem Container-Start,
alle Container mit `-v /tmp/food11:/tmp/food11` und `--data-root /tmp/food11`.

**Problem 5 — HuggingFace/timm Rate-Limiting:**
Ohne gemeinsamen Pretrained-Weights-Cache versuchten alle Container simultan die
ViT-Small-Gewichte von HuggingFace herunterzuladen (~90 MB pro Container).
HuggingFace drosselt unauthentifizierte Anfragen von derselben IP-Adresse.
Lösung: Einmaliger Pre-Download in einen NFS-Pfad:
`docker run --rm -e HF_HOME=/mnt/data/mats_adel/hf_cache nas-vit python -c
"import timm; timm.create_model('vit_small_patch16_224', pretrained=True)"`
Alle Container verwenden denselben Cache via `-e HF_HOME=/mnt/data/mats_adel/hf_cache`.

**Problem 6 — Python Output Buffering in Docker:**
Auch nach dem NFS-Fix erschienen keine Epoch-Logs in `docker logs` trotz aktiver
Trainings-GPU-Auslastung. Ursache: Im Docker-Detached-Mode puffert Python `stdout`
vollständig (kein Line-Buffering), da kein interaktives Terminal vorhanden ist.
Die `print()`-Ausgaben werden erst ausgegeben wenn der 8 kB-Buffer voll ist oder
das Programm beendet wird. Nach 17 Minuten Training waren ~0.5 kB Ausgabe vorhanden
— deutlich unter dem Buffer-Limit. Lösung: `-e PYTHONUNBUFFERED=1` in allen
`docker run`-Aufrufen. Dauerhaft im Dockerfile aufgenommen.

**Problem 7 — timm Unexpected-Keys-Warnings:**
Für jede Architektur mit `depth < 12` gibt `timm.create_model(pretrained=True)` eine
mehrzeilige Warning aus (`Unexpected keys: blocks.4.*, blocks.5.*, ...`), da die
pretrained Gewichte für depth=12 geladen werden, aber nur weniger Blöcke im Modell
vorhanden sind. Diese Warnings sind korrekt (überzählige Keys werden von `timm`
ignoriert) und störten ausschließlich die Log-Lesbarkeit.
Lösung: `logging.getLogger("timm").setLevel(logging.ERROR)` in `src/models/vit.py`.

#### Gestartete Experimente

Alle vier Experimente wurden gleichzeitig gestartet, je auf einer eigenen A100-GPU:

| Container      | Skript                    | GPU | Parameter                                                    |
| -------------- | ------------------------- | --- | ------------------------------------------------------------ |
| `proxy_val`    | `run_proxy_validation.py` | 0   | `n_configs=100, proxy_epochs=12`                             |
| `nas_baseline` | `run_nas.py`              | 1   | `pop=15, gen=8, proxy_epochs=12, use_proxies=False`          |
| `nas_proxies`  | `run_nas.py`              | 2   | `pop=15, gen=8, proxy_epochs=12, use_proxies=True, keep=0.5` |
| `f1_exp`       | `run_f1_experiment.py`    | 3   | `epochs=30, fracs=[0.1, 0.2, 0.3, 0.5, 0.75, 1.0]`           |

#### Ergebnisse und Laufzeiten

**`f1_exp` — F1-Datensatz-Skalierungs-Experiment:**

Laufzeit: 07:11 – 09:38 UTC = **2 Stunden 27 Minuten**.
Ergebnisse in `results/f1_scaling.csv`. Trainiert ViT-Small und drei CNN-Baselines
auf sechs Datensatzgrößen (10 % – 100 % von Food11 Trainingsset).

Dieses Experiment liefert die primären Daten für F1: Ab welcher Datensatzgröße
beginnt der ViT (mit Fine-Tuning) gegenüber CNN-Baselines zu dominieren?

**`nas_proxies` — NAS mit Zero-Cost-Proxy-Vorfilterung:**

Laufzeit: 08:40 – 12:42 UTC = **4 Stunden 2 Minuten**.
56 Architekturen evaluiert (von maximal 15 × 8 × 0.5 = 60 möglich; leichte
Abweichung durch variable Pareto-Pool-Größe in einzelnen Generationen).
Ergebnisse in `results/nas_proxies.csv`.

**`proxy_val` — Proxy-Validation-Experiment:**

Laufzeit: 08:40 – 15:11 UTC = **6 Stunden 31 Minuten**.
100 Architekturen evaluiert (Proxy-Score + 12 Epochen Training).
Ergebnisse in `results/proxy_validation.csv`.

Gemessene Spearman-Rang-Korrelation zwischen Proxy-Score und echter Val-Accuracy
(12 Epochen) über 100 Architekturen:

| Proxy    | Spearman-ρ | p-Wert (geschätzt) | Interpretation                                 |
| -------- | ---------- | ------------------ | ---------------------------------------------- |
| NASWOT   | **0.444**  | < 0.001            | Moderate, signifikante Korrelation             |
| SynFlow  | 0.192      | ~0.06              | Schwache, grenzwertig signifikante Korrelation |
| GradNorm | −0.037     | ~0.72              | Keine Korrelation                              |

**Interpretation Proxy-Ergebnisse:**

NASWOT hat die stärkste Korrelation mit echter Val-Accuracy auf Food11. Dieser Wert
(ρ=0.444) ist niedrig verglichen mit Proxy-Korrelationen auf ImageNet-Skala
(typisch ρ=0.6–0.8 in der Literatur), aber konsistent mit dem bekannten Befund,
dass Zero-Cost Proxies auf kleinen Datensätzen weniger zuverlässig sind.

SynFlow (ρ=0.192) zeigt ein schwaches Signal, das bei 100 Architekturen statistisch
grenzwertig ist. GradNorm (ρ=−0.037) zeigt keinerlei Korrelation — bei ViT-Fine-Tuning
auf einem kleinen Datensatz scheint die Gradientennorm bei Initialisierung kein
valider Indikator für Trainierbarkeit zu sein, möglicherweise weil die Norm stark
von `finetune_mode` abhängt (bei `head` sind fast keine Gradienten vorhanden, bei
`full` viele) und nicht nur von der Architektur.

Diese Ergebnisse haben direkte Konsequenzen für das Rangsummen-Ensemble: GradNorm
schadet potenziell der Rankingqualität (zufälliges Rauschen im Ensemble). Eine
Ablation mit nur NASWOT als Proxy wäre für zukünftige Arbeiten sinnvoll.

**`nas_baseline` — NAS ohne Proxy-Vorfilterung:**

Laufzeit: 08:40 – 16:26 UTC = **7 Stunden 46 Minuten**.
120 Architekturen evaluiert (15 × 8 Generationen, alle Kandidaten trainiert).
Ergebnisse in `results/nas_baseline.csv`.

**Compute-Vergleich (direkte Evidenz für F2):**

| NAS-Variante      | Evaluierungen | Laufzeit | Compute-Einsparung |
| ----------------- | ------------- | -------- | ------------------ |
| Ohne Proxy-Filter | 120           | 7h 46min | —                  |
| Mit Proxy-Filter  | 56            | 4h 02min | **−48 %**          |

Die Proxy-Vorfilterung halbiert die Laufzeit. Ob die Qualität der gefundenen
Pareto-Front dadurch leidet, wird durch den Vergleich von `nas_baseline.csv` und
`nas_proxies.csv` quantifiziert (Kapitel 5): Dominieren die Architekturen aus
`nas_baseline` die aus `nas_proxies`, oder finden beide ähnlich gute Fronten?

---

### 2026-07-17 — Vollständiges Training: Top-3 und Bottom-3 Architekturen aus NAS

**Was:** `retrain_analysis.py` wird auf dem Server gestartet, um die drei besten
und drei schlechtesten Architekturen aus `nas_baseline.csv` (nach 12-Epoch-Val-Accuracy)
für jeweils 50 Epochen vollständig zu trainieren.

**Warum:** Proxy-Training mit 12 Epochen ist eine Näherung — es beantwortet die Frage,
welche Architekturen nach kurzer Trainingszeit gut aussehen. Aber entspricht das der
finalen Qualität nach vollständigem Training?

Diese Analyse beantwortet drei mögliche Szenarien:

1. **Stabile Rangliste:** Die Top-3 bleiben Top-3 nach 50 Epochen. → Proxy-Epochen
   sind ein zuverlässiger Indikator. Die NAS-Ergebnisse aus `nas_baseline.csv` sind
   direkt interpretierbar.

2. **Bottom-3 schließen auf:** Architekturen, die nach 12 Epochen schlecht aussehen,
   profitieren stark von längeren Training. → Proxy-Epochen unterschätzen systematisch
   bestimmte Architektur-Typen (z. B. tiefere Netze, `finetune_mode=full`), die
   mehr Epochen zum Einlaufen brauchen. Dies würde bedeuten, dass die NAS-Ergebnisse
   die tatsächlich besten Architekturen möglicherweise nicht finden.

3. **Top-3 fallen zurück:** Architekturen, die kurz gut aussehen, zeigen bei vollem
   Training Overfitting oder Konvergenz-Probleme. → Proxy-Epochen begünstigen
   bestimmte Architektur-Typen systematisch zu Unrecht.

Alle drei Szenarien sind für F2 relevante Befunde.

**Startbefehl:**

```bash
docker run -d --name retrain_analysis \
  --runtime=nvidia --gpus '"device=0"' \
  --shm-size=8g \
  -e HF_HOME=/mnt/data/mats_adel/hf_cache \
  -e PYTHONUNBUFFERED=1 \
  -v /mnt/data/mats_adel:/mnt/data/mats_adel \
  -v /tmp/food11:/tmp/food11 \
  nas-vit python scripts/retrain_analysis.py \
    --nas-csv /mnt/data/mats_adel/results/nas_baseline.csv \
    --data-root /tmp/food11 \
    --top-n 3 --bottom-n 3 --epochs 50
```

Tatsächliche Laufzeit: 07:32 – 09:02 UTC = **1 Stunde 30 Minuten** (6 Architekturen × 50 Epochen auf A100).

#### Ergebnisse (`retrain_analysis_summary.json`)

| Label    | Proxy Val-Acc | Full Val-Acc |  Test-Acc  | Δ (Proxy→Full Val) |
| -------- | :-----------: | :----------: | :--------: | :----------------: |
| top_1    |    0.9566     |    0.9554    | **0.9692** |      −0.0012       |
| top_2    |    0.9531     |    0.9542    |   0.9624   |      +0.0011       |
| top_3    |    0.9464     |    0.9466    |   0.9591   |      +0.0002       |
| bottom_1 |    0.2650     |    0.3117    |   0.3325   |      +0.0467       |
| bottom_2 |    0.2644     |    0.3111    |   0.3269   |      +0.0467       |
| bottom_3 |    0.2633     |    0.2904    |   0.3131   |      +0.0271       |

#### Interpretation

**Befund: Szenario 1 — Stabile Rangliste.** Die drei eingangs formulierten Szenarien
waren (1) stabile Rangliste, (2) Bottom-3 schließen auf, (3) Top-3 fallen zurück.
Die Daten belegen klar Szenario 1:

**Stabile Rangliste für Top-Architekturen:** Die Reihenfolge top_1 > top_2 > top_3
ist nach 50 vollständigen Epochen identisch zur Reihenfolge nach 12 Proxy-Epochen.
Die absoluten Val-Accuracy-Änderungen liegen unter 0.2 % (innerhalb des Messrauschens).
Dies bedeutet, dass 12 Proxy-Epochen **ausreichend** sind, um gute Architekturen
zuverlässig zu identifizieren — eine direkte Rechtfertigung des NAS-Ablaufs.

**Permanente Trennung von Top und Bottom:** Die Bottom-3 Architekturen verbessern sich
zwar beim Übergang von 12 auf 50 Epochen um ~3–5 Prozentpunkte, bleiben aber mit
Test-Accuracies von 31–33 % weit von den Top-Architekturen (96–97 %) entfernt.
Der Abstand von ~63 Prozentpunkten zeigt, dass es sich um grundlegend ungeeignete
Architekturen handelt, nicht um Architekturen, die einfach mehr Zeit zum Einlaufen
brauchen.

**Test-Accuracy übertrifft Val-Accuracy:** Alle drei Top-Architekturen erzielen
auf dem Test-Set (3.347 Bilder, nie während NAS oder Training gesehen) höhere
Accuracies als auf dem Validierungsset. Dies ist ein positives Zeichen — kein
Overfitting auf den Validierungssplit durch die Architektur-Selektion. Die
Differenz (~1.3 pp bei top_1) liegt im erwartbaren Bereich natürlicher
Set-zu-Set-Varianz bei ~3.000 Bildern.

**Bestes Ergebnis: top_1 mit 96.92 % Test-Accuracy** auf Food11 mit ~9.900
Trainingsbildern — ein starkes Resultat für einen ViT auf einem kleinen Datensatz.
Die genaue Konfiguration dieser Architektur ist aus `nas_baseline.csv` (höchste
val_acc nach 12 Epochen) ablesbar und wird als beste Architektur für die finale
Evaluation herangezogen.

---

### 2026-07-17 — SPT-Ablation-Study: Kompatibilität von SPT mit Fine-Tuning

**Was:** Kontrolliertes Experiment zur Frage: Funktioniert Shifted Patch Tokenization
(SPT) auch im Fine-Tuning-Setting mit einem auf ImageNet vortrainierten ViT, oder
ist SPT auf Training from Scratch angewiesen?

**Motivation:** Im NAS-Lauf (`nas_baseline.csv`) zeigten Architekturen mit `use_spt=1`
systematisch schlechtere Val-Accuracies. Die eigene Beobachtung und anschließende
Hypothese: Dies liegt nicht an SPT als Architektur-Konzept, sondern an einem
Implementierungs-Konflikt mit dem Pretrained-Weight-Transfer. In unserem Setup wird
`patch_embed` (Conv2d, in_channels=3, pretrained auf ImageNet) nach dem Gewichtsladen
durch ein neues SPT-Patch-Embed (Conv2d, in_channels=15) ersetzt — mit **zufälliger
Initialisierung** für die neuen 15-Kanal-Gewichte.

Das originale SPT-Paper (Lee et al., 2021) trainiert ausschließlich from scratch auf
CIFAR-10/100 und Tiny ImageNet und adressiert den Fine-Tuning-Fall gar nicht.

**Drei Bedingungen:** Identische Architektur (top_1-Config aus `nas_baseline.csv`,
`use_lsa=0`, 50 Epochen, Patience 15) mit drei SPT-Varianten:

| Bedingung    | `use_spt` | Patch-Embed-Init                                                     |
| ------------ | :-------: | -------------------------------------------------------------------- |
| `spt_none`   |     0     | Originales pretrained 3-Kanal-Conv (unverändert)                     |
| `spt_random` |     1     | Zufällig initialisiertes 15-Kanal-Conv (bisheriger NAS-Standard)     |
| `spt_warm`   |     1     | 15-Kanal-Conv initialisiert durch: `W_orig.repeat(1, 5, 1, 1) / 5.0` |

**Warm-Start-Idee:** Die originalen 3-Kanal-Gewichte werden fünfmal entlang der
Eingangskanal-Dimension wiederholt und durch 5 dividiert. Wenn alle fünf
SPT-Eingaben identisch wären (kein Shift), wäre der Output identisch zum Original:
`5 × (W/5 · x) = W · x`. Dies gibt dem Patch-Embed einen Startpunkt, der mit den
pretrained Transformer-Blöcken kompatibel ist.

**Laufzeit:** ~30 Minuten auf NVIDIA H100 NVL (80GB).

**Ergebnisse (`results/spt_ablation_summary.json`):**

| Bedingung    | Val-Acc (Ep. 0) | Best Val-Acc |  Test-Acc  | Δ vs. spt_none |
| ------------ | :-------------: | :----------: | :--------: | :------------: |
| `spt_none`   |     0.8813      |  **0.9595**  | **0.9701** |       —        |
| `spt_warm`   |     0.6437      |    0.9149    |   0.9274   |    −4.5 pp     |
| `spt_random` |     0.3402      |    0.7341    |   0.7451   |  **−22.5 pp**  |

**Interpretation:**

**Befund 1 — Hypothese bestätigt:** `spt_random` bricht mit −22.5 pp drastisch ein.
Der Epoch-0-Val-Wert von 34.0 % (nahe am Zufallslevel für 11 Klassen = 9.1 %) zeigt,
dass die zufällig initialisierte 15-Kanal-Faltung die pretrained Repräsentationen
der Transformer-Blöcke vollständig unbrauchbar macht. Die Blöcke erwarten eine
Einbettungsverteilung aus einem auf ImageNet trainierten 3-Kanal-Embed — was sie
stattdessen bekommen, ist strukturiertes Rauschen. Das Modell muss das gesamte
Verständnis neu aufbauen, was in 50 Epochen nur auf ~73 % führt.

**Befund 2 — Warm-Start hilft, reicht aber nicht:** `spt_warm` startet deutlich
besser (Epoch 0: 64.4 % vs. 34.0 %) und erreicht 91.5 %, bleibt aber 4.5 pp
hinter `spt_none`. Auch nach 50 Epochen zeigt `spt_warm` noch keine Sättigung —
die Val-Accuracy steigt bis zur letzten Epoche monoton an. Mit mehr Epochen würde
die Lücke möglicherweise kleiner, nicht aber verschwinden: SPT verändert die
Statistik der Patch-Token-Aktivierungen auch bei optimaler Initialisierung, was
einen Mehraufwand im Training bedeutet.

**Befund 3 — Konsequenz für den NAS-Lauf:** Im `nas_baseline`-Lauf wurden
SPT-Architekturen mit 12 Proxy-Epochen evaluiert. Da `spt_random` auch nach 50
Epochen auf ~73 % bleibt, hat der NAS-Algorithmus SPT-Architekturen korrekt als
schlechter eingestuft — aber nicht weil SPT als Konzept schlecht ist, sondern weil
Post-hoc-SPT-Integration fundamental mit dem Pretrained-Weight-Transfer konfligiert.

**Designimplikation / Limitation:** SPT ist für Training from Scratch konzipiert.
Ein valider Vergleich von "ViT mit SPT vs. ohne SPT" würde beide Varianten from
scratch trainieren — auf Food11 mit ~10k Bildern wäre das aber keine sinnvolle
Baseline, da ViTs ohne große Datenmengen nicht konvergieren. Die Entscheidung, SPT
als optionalen NAS-Parameter beizubehalten, ist rückblickend eine Limitation: Der
NAS-Algorithmus kann SPT nicht fair evaluieren. Dies wird in der Arbeit als
Limitation diskutiert.

---

### 2026-07-29 — Explorative Datenanalyse (`notebooks/auswertung_ergebnisse.ipynb`)

**Was:** Systematische Auswertung aller vier Experiment-Ergebnisse im Notebook.
Erzeugte Plots unter `results/plots/`.

#### F1: ViT vs. CNN — Datensatz-Skalierung

**Vollständige Ergebnistabelle** (`results/f1_scaling.csv`, 30 Epochen, frac=1.0 = ~9.866 Bilder):

| Datensatzanteil | Trainingsbilder | ViT-Small  | ResNet-18 | EfficientNet-B0 | MobileNetV3 |
| :-------------: | :-------------: | :--------: | :-------: | :-------------: | :---------: |
|      10 %       |      ~987       | **0.9207** |  0.7913   |     0.7895      |   0.7953    |
|      20 %       |     ~1.973      | **0.9283** |  0.8297   |     0.8385      |   0.8402    |
|      30 %       |     ~2.960      | **0.9344** |  0.8557   |     0.8752      |   0.8653    |
|      50 %       |     ~4.933      | **0.9364** |  0.8819   |     0.8980      |   0.8933    |
|      75 %       |     ~7.400      | **0.9569** |  0.8968   |     0.9131      |   0.9026    |
|      100 %      |     ~9.866      | **0.9536** |  0.9020   |     0.9224      |   0.9216    |

**Befund F1 — ViT dominiert bei allen Datensatzgrößen:**
Es gibt keinen Crossover-Punkt. Der pretrained ViT-Small übertrifft alle drei CNN-Baselines
bereits bei 10 % der Trainingsdaten (~987 Bilder) mit einem Vorsprung von ~12 pp gegenüber
ResNet-18. Dies widerspricht der verbreiteten Annahme, dass ViTs viele Daten benötigen —
im Fine-Tuning-Setting sind die ImageNet-pretrained Features so stark, dass auch wenige
Ziel-Domänen-Bilder für eine gute Anpassung ausreichen.

**Zusätzlicher Befund — Lücke schließt sich:** Mit mehr Daten holen die CNNs anteilig
stärker auf (ResNet-18: +11 pp von 10 % auf 100 %, ViT: +3 pp). Der ViT-Vorteil schrumpft
von ~12 pp (bei 10 %) auf ~5 pp (bei 100 %), bleibt aber klar positiv. CNNs profitieren
also stärker von zusätzlichen Daten — was bedeutet: Bei deutlich größeren Datensätzen
(>>10k Bilder) könnte sich das Verhältnis verändern.

---

#### NAS Baseline — Beste Architektur (top_1)

| Parameter         | Wert            |
| ----------------- | --------------- |
| depth             | 12              |
| num_heads         | 8               |
| use_spt           | 0 (kein SPT)    |
| use_lsa           | 0 (kein LSA)    |
| finetune_mode     | **full**        |
| lr                | 1e-4            |
| weight_decay      | 5e-4            |
| mixup_alpha       | 0.0             |
| cutmix_prob       | 0.5             |
| Proxy val_acc     | 0.9566 (12 Ep.) |
| Full val_acc      | 0.9554 (50 Ep.) |
| **Test-Accuracy** | **0.9692**      |
| Parameter         | 21.67M          |

Die beste Architektur nutzt maximale Tiefe (depth=12), vollständiges Fine-Tuning aller
Parameter (`full`) und CutMix-Augmentierung (cutmix_prob=0.5), aber weder SPT noch LSA.
Das `full`-Fine-Tuning erklärt die hohe Parameteranzahl (21.67M trainierbar), erfordert
aber auch die niedrige Lernrate (1e-4) um Catastrophic Forgetting zu vermeiden.

---

#### Pareto-Front Vergleich: Baseline vs. Proxy-gefilterter NAS

| Metrik                         | NAS Baseline    | NAS mit Proxies |
| ------------------------------ | --------------- | --------------- |
| Evaluierte Architekturen       | 120             | 56 (−53 %)      |
| Architekturen auf Pareto-Front | 6               | 8               |
| Beste Val-Acc (Pareto)         | 0.9566          | **0.9668**      |
| Val-Acc-Bereich (Pareto)       | 0.6554 – 0.9566 | 0.8362 – 0.9668 |

**Befund F2 — Proxy-NAS übertrifft Baseline trotz halbiertem Aufwand:**
Der proxy-gefilterte NAS-Lauf findet mit nur 56 Evaluierungen (statt 120) eine beste
Architektur mit 0.9668 Val-Acc — 1.0 pp besser als das Baseline-Best von 0.9566.
Zudem ist die Pareto-Front der Proxy-Variante kompakter (val_acc ≥ 0.836 für alle
8 Punkte) als die der Baseline (schlechtester Pareto-Punkt: 0.655). Der Proxy-Filter
verhindert, dass der evolutionäre Algorithmus Rechenzeit auf qualitativ schwache
Architekturen verschwendet, und fokussiert die Population stärker auf vielversprechende
Regionen des Suchraums.

---

#### EDA: Parameter-Einflussranking

Gemessen als Δ Mean Val-Accuracy (max − min der Gruppen-Mittelwerte, NAS Baseline n=120):

| Rang | Parameter     | Δ Mean Val-Acc | Interpretation                                  |
| :--: | ------------- | :------------: | ----------------------------------------------- |
|  1   | use_spt       |   **0.340**    | Stärkster Faktor — SPT random init inkompatibel |
|  2   | finetune_mode |     0.237      | head vs. full macht 24 pp Unterschied           |
|  3   | depth         |     0.191      | Tiefere Netze klar besser auf Food11            |
|  4   | lr            |     0.126      | Lernrate wichtiger als Architektur-Details      |
|  5   | num_heads     |     0.065      | Moderater Einfluss                              |
|  6   | mixup_alpha   |     0.049      | Schwacher Einfluss                              |
|  7   | cutmix_prob   |     0.035      | Schwacher Einfluss                              |
|  8   | weight_decay  |     0.017      | Vernachlässigbar                                |
|  9   | use_lsa       |     0.017      | Kein messbarer Einfluss                         |

`use_spt` ist der bei weitem stärkste Einflussfaktor — konsistent mit der SPT-Ablation.
`finetune_mode` auf Rang 2 zeigt, dass die Wahl des Fine-Tuning-Modus mindestens so
wichtig ist wie die Architekturtiefe. `use_lsa` am Ende bestätigt, dass LSA im
Fine-Tuning-Setting keinen Vorteil bringt.

---

#### NAS vs. CNN bei vergleichbarer Modellgröße (≤ 12M Parameter)

Die CNN-Baselines haben 5.3–11.7M Parameter. Unter den 120 NAS-Architekturen haben
23 eine Parameterzahl ≤ 12M (durch `depth ≤ 6` oder `finetune_mode` mit weniger
trainierbaren Parametern). Beste dieser Architekturen:

| Architektur                      | Val-Acc | Params |
| -------------------------------- | :-----: | :----: |
| NAS beste ≤12M (depth=6, full)   | 0.8869  | 11.02M |
| ResNet-18 (CNN Baseline)         | 0.9020  | 11.7M  |
| EfficientNet-B0 (CNN Baseline)   | 0.9224  |  5.3M  |
| MobileNetV3-Large (CNN Baseline) | 0.9216  |  5.5M  |

**Befund:** Kleinere NAS-ViT-Architekturen (≤12M) sind bei 12 Proxy-Epochen schlechter
als CNN-Baselines in der gleichen Parameterklasse. Erst die voll trainierten großen
Architekturen (21.67M, depth=12, full) erreichen mit 96.92 % Test-Acc eine klar
überlegene Leistung. Der ViT-Vorteil ist also vor allem ein **Transfer-Learning-Vorteil**
großer pretrained Modelle, nicht ein Effizienzvorsprung in der Parameteranzahl.

---

#### F3: NAS-ViT vs. CNN — Größenvergleich und Klassifikationsgenauigkeit

F3 fragt: _Inwiefern gelingt es einem NAS-optimierten ViT, bei reduzierter Parameteranzahl
eine mit CNN-Baselines vergleichbare Klassifikationsgenauigkeit zu erzielen?_

**Vollständige Gegenüberstellung (Food11, vollständiges Trainingsset):**

| Modell                     | Typ     | Parameter | Val-Acc |  Test-Acc  |
| -------------------------- | ------- | :-------: | :-----: | :--------: |
| NAS top_1 (depth=12, full) | ViT-NAS |  21.67M   | 0.9536  | **0.9692** |
| NAS beste ≤12M (depth=6)   | ViT-NAS |  11.02M   | 0.8869  |     —      |
| EfficientNet-B0            | CNN     |   5.3M    | 0.9224  |     —      |
| MobileNetV3-Large          | CNN     |   5.5M    | 0.9216  |     —      |
| ResNet-18                  | CNN     |   11.7M   | 0.9020  |     —      |

**Befund 1 — Kein Effizienzgewinn durch NAS in der Parameterklasse:**
Die beste NAS-gefundene ViT-Architektur mit ≤12M Parametern (11.02M, val_acc=0.8869)
erreicht nicht die Genauigkeit der CNN-Baselines in vergleichbarer Parameterklasse
(EfficientNet-B0: 5.3M, val_acc=0.9224). F3 wird damit **negativ beantwortet**: Ein
NAS-optimierter ViT erzielt bei reduzierter Parameteranzahl keine mit CNNs vergleichbare
Klassifikationsgenauigkeit auf Food11.

**Befund 2 — Accuracy-Gewinn nur bei größerem Modell:**
Der klare Gewinn des NAS-ViT (96.92 % Test-Acc) gegenüber den CNN-Baselines
(≤92.2 % Val-Acc) kommt mit fast 4× mehr Parametern (21.67M vs. 5.3M). Der
Mehrwert von NAS liegt hier nicht in der Parametereffizienz, sondern darin,
die optimale Trainings-Konfiguration für das große pretrained Modell zu finden
(depth=12, finetune_mode=full, cutmix_prob=0.5, lr=1e-4).

**Befund 3 — Pareto-Perspektive:**
Die Pareto-Front des NAS-Baseline-Laufs (6 Architekturen) zeigt, dass es keine
ViT-Architektur gibt, die gleichzeitig CNN-Parametereffizienz und CNN-Genauigkeit
übertrifft. Alle Pareto-optimalen Punkte mit val_acc > 0.90 haben Parameteranzahlen

> 15M. Dies ist eine direkte Konsequenz der ViT-Architektur: Die minimale Tiefe
> (depth=4–6) für halbwegs gute Ergebnisse erzeugt bereits ~11M Parameter durch den
> großen MLP-Block (embed_dim=384, mlp_ratio=4 → 384×4×384×2 pro Block).

**Einordnung:** Die negative Antwort auf F3 ist selbst ein wissenschaftlicher Befund.
Sie zeigt, dass NAS auf kleinen Datensätzen mit pretrained ViTs nicht die Effizienz-
Lücke zu CNNs schließt, sondern stattdessen die Konfiguration für maximale Accuracy
optimiert — ein anderes, aber ebenfalls wertvolles Ergebnis.
