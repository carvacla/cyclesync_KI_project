# AI Applications Project Documentation Template

Use this template to document your project concisely and completely.
Fill in all required fields. Keep answers short and precise.

## Documentation Hint

Important:
When possible, reference the corresponding code location directly in your description.

### Example: Reference to a notebook section
Reference to the header `## Data Preprocessing` in the notebook `analysis.ipynb`:

> See *Data Preprocessing* in
> [`analysis.ipynb`](analysis.ipynb#data-preprocessing)

### Example: Reference to Python code

Reference to a single line in `model.py`, line 42:
> [`model.py`, line 42](model.py#L42)

Reference to multiple lines in `train.py`, lines 15-38:
> [`train.py`, lines 15-38](train.py#L15-L38)

## Project Metadata

- Project title: CycleSync — Zyklusbasierter Trainings- & Recovery-Coach
- Student: Claudia Carvalho Paula (Wirtschaftsinformatik, Klasse TZaBIS)
- GitHub repository URL: https://github.com/carvacla/cyclesync_KI_project
- Deployment URL: https://huggingface.co/spaces/CarvalhoClaudia/cyclesync
- Submission date: 07.06.2026

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [ ] Computer Vision

Primary blocks used for core solution (choose 2):
- Primary block 1: ML Numeric Data
- Primary block 2: NLP (Retrieval-Augmented Generation)

If a third block is selected, it is documented and graded separately as extra work.

Guidance hint: Keep the project idea short and consistent. Focus most details on the selected blocks.
Evidence hint: Show where each selected block contributes to the final system.

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition
- Problem statement: Sportwissenschaftliche Forschung und Mainstream-Trainings-Apps berücksichtigen die zyklischen physiologischen Veränderungen weiblicher Athletinnen bislang kaum. Empfehlungen basieren überwiegend auf männlichen Studienpopulationen, was für weibliche Nutzerinnen zu suboptimalen Trainings- und Recovery-Empfehlungen führen kann.
- Goal: Eine Web-App, die aus Zyklus- und Wellness-Daten eine begründete Trainingsempfehlung erzeugt und diese mit aktueller wissenschaftlicher Literatur (PubMed) erklärt.
- Success criteria:
  1. ML-Klassifikator erreicht F1 (macro) ≥ 0.75 auf dem Hold-out-Set
  2. RAG-Pipeline liefert pro Empfehlung ≥ 2 thematisch passende PubMed-Quellen
  3. End-to-End-Antwort < 10s in der deployten App
  4. Öffentlich erreichbares Deployment auf HuggingFace Spaces

### 1.2 Integration Logic
- How the selected blocks interact: Der ML-Block klassifiziert die Trainingsempfehlung (low/moderate/high) aus strukturierten User-Daten (Zyklustag, Symptome, Schlafqualität, etc.). Die strukturierte Vorhersage (Phase, Intensität, Recovery-Stunden, Risiko, Konfidenz) wird als Kontext an die RAG-Pipeline weitergegeben. Diese retrieved passende PubMed-Abstracts und ein LLM generiert eine personalisierte, deutschsprachige Erklärung mit Studien-Zitaten.
- Data and output flow between blocks: User-Input → ML-Pipeline ([`src/ml_model.py`](../src/ml_model.py)) → strukturierte Vorhersage → RAG ([`src/rag_pipeline.py`](../src/rag_pipeline.py)) → kombinierte Antwort mit Empfehlung, Erklärung und Quellen. Orchestriert durch [`src/recommender.py`](../src/recommender.py), präsentiert via [`app.py`](../app.py).

Guidance hint: This section should be short. The detailed work belongs in block sections.
Evidence hint: Include one clear pipeline overview.

---

## 2. Block Documentation

Complete only selected blocks. Mark non-selected block sections as N/A.

### 2A. ML Numeric Data (If selected)

#### 2A.1 Data Source(s)
List every usage of a data source as a separate entry. If the same source is used twice for different roles, add it twice.

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | Synthetisches Cycle-Tracking-Dataset (generiert in [`notebooks/01_data_acquisition.ipynb`](../notebooks/01_data_acquisition.ipynb), basierend auf publizierten physiologischen Mustern: Schmalenberger et al. 2021, Carmichael et al. 2021) | Numeric/categorical (Zyklustag, Phase, BBT, Schlaf, Symptome, Ruhepuls, Alter, Fitness-Level) | 500 Userinnen × 3 Zyklen × 28 Tage = 42'000 Tagesrecords | Trainings- und Evaluations-Set für den Empfehlungs-Klassifikator |
| 2 | Synthetisches Workout-Dataset (generiert in Notebook 01, basierend auf McNulty et al. 2020 Meta-Analyse) | Numeric/categorical (Sportart, Dauer, HF, RPE, Recovery, Wetter) | ~10'000 Workouts | Validierung der Phasen-Performance-Annahmen, ergänzende Datenbasis für Realismus |
| 3 | — | — | — | — |

#### 2A.2 Preprocessing and Features
- Cleaning steps: Symptome werden aus ';' getrennten Strings in 6 binäre Features expandiert (`sym_cramps`, `sym_fatigue`, etc.). Keine fehlenden Werte (synthetische Daten). Siehe [`notebooks/02_eda.ipynb`](../notebooks/02_eda.ipynb#4-processed-data-speichern).
- Preprocessing steps: `ColumnTransformer` mit `StandardScaler` für 13 numerische Features und `OneHotEncoder` für 2 kategorische Features (Phase, Fitness-Level). Stratifizierter 70/15/15 Split (Train/Val/Test). Siehe [`notebooks/03_ml_modeling.ipynb`](../notebooks/03_ml_modeling.ipynb#1-daten-laden-features-definieren).
- Feature engineering and selection: Abgeleitete Features `symptom_count` (Summe der Symptome) und 6 binäre Symptom-Indikatoren `sym_<symptom>`. Feature-Importance via Random Forest analysiert. Die wichtigsten Features sind: `sleep_quality` (~0.24), `day_in_cycle` (~0.19), `phase_follicular` (~0.11), `symptom_count` (~0.09), `phase_luteal` (~0.08).

#### 2A.3 Model Selection
- Models tested: Logistic Regression (Baseline), Random Forest, XGBoost.
- Why these models were chosen: Logistic Regression als interpretierbare lineare Baseline; Random Forest als robustes Ensemble-Modell mit guter Performance bei strukturierten Daten und gemischten Feature-Typen; XGBoost als Gradient-Boosting-Vergleich (State-of-the-Art für Tabellendaten).

#### 2A.4 Model Comparison and Iterations
| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline | Standardparameter, multinomial | Logistic Regression | Val F1 (macro): 0.881, Val Accuracy: 0.898 | — |
| 2 | Ensemble | n_estimators=200, max_depth=15 | Random Forest | Val F1 (macro): 1.000, Val Accuracy: 1.000 | +0.119 F1 |
| 3 | Gradient Boosting | n_estimators=300, max_depth=6, lr=0.1 | XGBoost | Val F1 (macro): 1.000, Val Accuracy: 1.000 | ±0.000 vs Iter 2 |

#### 2A.5 Evaluation and Error Analysis
- Metrics used: Accuracy, F1 (macro), Confusion Matrix, Feature Importance.
- Final results: Random Forest auf Test-Set: **F1 (macro) = 1.000**, **Accuracy = 1.000**. Confusion Matrix zeigt perfekte Klassifikation: low (3031), moderate (1059), high (2210), keine Fehlklassifikationen.
- Error patterns and likely causes: Die perfekten Scores von Random Forest und XGBoost sind methodisch erklärbar und reflexionswürdig: Das Target `recommended_intensity` wird in Notebook 01 durch eine **deterministische Regelfunktion** (`recommend_intensity()`) aus Phase, Schlafqualität und Symptomen generiert. Tree-basierte Modelle wie Random Forest und XGBoost können diese Regel praktisch perfekt rekonstruieren, weil ihre Splits genau die Schwellwerte (sleep_quality ≤ 4, symptom_count ≥ 2 etc.) abbilden können. Logistic Regression als lineares Modell erreicht "nur" 88% F1, weil sie diese diskreten Schwellen schlechter modellieren kann. Für ein produktives System mit realen, verrauschten Labels wäre dieses Verhalten nicht zu erwarten — die deterministische Label-Funktion ist eine Limitierung der synthetischen Datengrundlage.

#### 2A.6 Integration with Other Block(s)
- Inputs received from other block(s): Keine direkten Inputs aus dem NLP-Block in der aktuellen Version.
- Outputs provided to other block(s): Strukturierte Vorhersage `{phase, intensity, recovery_hours, risk, confidence}` wird via [`CycleSyncRecommender`](../src/recommender.py#L8-L19) an die RAG-Pipeline übergeben. Die Phase und Intensität steuern dort die Retrieval-Query, und alle Werte fliessen als Kontext in den Prompt ein.

Guidance hint: Keep entries practical and evidence-based.
Evidence hint: Add values, not only claims.

### 2B. NLP (If selected)

#### 2B.1 Data Source(s)
List every usage of a data source as a separate entry. If the same source is used twice for different roles, add it twice.

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | PubMed Abstracts via NCBI E-utilities (8 Queries: menstrual cycle exercise, luteal phase training, etc.) | Text (wissenschaftliche Abstracts inkl. Titel, PMID, Jahr, Autoren) | ~300 Abstracts | Wissensbasis für RAG-Retrieval |
| 2 | Strukturierte ML-Vorhersage aus Block 2A | JSON-strukturiert | pro Request | Kontext im Prompt + steuert Retrieval-Query |
| 3 | User-Profil aus Streamlit-UI ([`app.py`](../app.py)) | strukturiert (Zyklustag, Symptome, Schlaf, etc.) | pro Request | Personalisierung der Erklärung |

#### 2B.2 Preprocessing and Prompt Design
- Text preprocessing: Filterung von Abstracts mit < 100 Zeichen, Deduplication per PMID, Metadata-Anreicherung (Jahr, Autoren). Kein Chunking nötig, da Abstracts inhärent kurz sind. Siehe [`notebooks/04_rag_pipeline.ipynb`](../notebooks/04_rag_pipeline.ipynb#1-dokumente-vorbereiten).
- Prompt design or retrieval setup: Lokales Embedding-Modell `sentence-transformers/all-MiniLM-L6-v2` (kostenlos, kein API-Aufruf), Chroma als persistente Vektor-DB unter [`models/chroma_db/`](../models/chroma_db/), Top-4 Retrieval. Prompt enthält strukturierten User- und ML-Kontext sowie retrieved Abstracts; verlangt deutsche Antwort in 4 strukturierten Teilen: konkrete Empfehlung, physiologische Begründung, mindestens 2 Studien-Zitate per `[PMID:xxxx]`, Disclaimer. Siehe [`src/rag_pipeline.py`, lines 8-31](../src/rag_pipeline.py#L8-L31).

#### 2B.3 Approach Selection
- Approach used (classical NLP, transformer, RAG, prompt engineering): Retrieval-Augmented Generation (RAG) mit GPT-4o-mini als Generator und einem lokalen Sentence-Transformer als Encoder.
- Alternatives considered: (1) Reines Prompt-Engineering ohne Retrieval — verworfen wegen Halluzinationsrisiko bei medizinischen Aussagen und fehlender Quellenbelege. (2) Klassisches Information-Retrieval mit TF-IDF — verworfen wegen schlechterer semantischer Trefferqualität bei sportwissenschaftlichen Fachbegriffen.

#### 2B.4 Comparison and Iterations
| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline-RAG | MiniLM-Embeddings, Top-4 Retrieval, Zero-Shot Prompt (PROMPT_A) | gpt-4o-mini, temp=0.3 | Qualitative Bewertung 5 Testszenarien: Quellen relevant, aber Zitate inkonsistent | — |
| 2 | Strukturierter Prompt | Explizite 4-Punkt-Antwortstruktur (Empfehlung, Begründung, Zitate, Disclaimer) | gpt-4o-mini, temp=0.3, PROMPT_B | Konsistentere Zitierungen, klarere Antworten, Disclaimer immer enthalten | Deutliche Verbesserung |
| 3 | — | — | — | — | — |

#### 2B.5 Evaluation and Error Analysis
- Evaluation strategy: Qualitative Bewertung von 3 Test-Szenarien (frühe Follikelphase, späte Lutealphase mit Symptomen, Menstruationstag mit Krämpfen) in [`notebooks/05_integration_test.ipynb`](../notebooks/05_integration_test.ipynb). Bewertungs-Kriterien: Relevanz der Retrieval-Treffer, Faktentreue der Erklärung, Tonfall, Disclaimer-Konformität.
- Results: PROMPT_B (strukturiert) liefert konsistenter Zitate und enthält den Disclaimer zuverlässig. Beispiel-Output für Tag 7 (Follikelphase), Lauftraining: empfiehlt hohe Intensität mit Bezug auf Östrogen-Effekt auf Performance, zitiert PMID:36129579 (iron homeostasis) und PMID:39189220 (pelvic floor muscles).
- Error patterns and likely causes: (1) Bei sehr generischen User-Inputs (keine Symptome, normaler Schlaf) sind retrieved Abstracts manchmal nur lose mit der konkreten Phase verbunden. (2) Das Modell tendiert dazu, denselben Abstract mehrfach zu zitieren wenn er thematisch dominant ist. Ursache: Top-4 Retrieval gibt keine Diversity-Garantie; MMR (Maximum Marginal Relevance) wäre eine Erweiterung.

#### 2B.6 Integration with Other Block(s)
- Inputs received from other block(s): Strukturierte ML-Vorhersage `{phase, intensity, recovery_hours, risk, confidence}` aus Block 2A — wird Teil des Prompt-Kontexts und steuert die Retrieval-Query (siehe [`src/rag_pipeline.py`, lines 53-58](../src/rag_pipeline.py#L53-L58)).
- Outputs provided to other block(s): Natürlichsprachliche Erklärung + Liste der zitierten Studien (PMID, Titel, Jahr) zurück an die App-Schicht ([`app.py`](../app.py)) zur Anzeige.

Guidance hint: Show concrete prompt or retrieval decisions.
Evidence hint: Include representative outputs or failure cases.

### 2C. Computer Vision (If selected)

N/A — Computer Vision ist in diesem Projekt nicht ausgewählt.

#### 2C.1 Data Source(s)
N/A

#### 2C.2 Preprocessing and Augmentation
N/A

#### 2C.3 Model Selection
N/A

#### 2C.4 Model Comparison and Iterations
N/A

#### 2C.5 Evaluation and Error Analysis
N/A

#### 2C.6 Integration with Other Block(s)
N/A

Guidance hint: Use concise examples from real predictions.
Evidence hint: Include sample outputs and observed failure cases.

---

## 3. Deployment

- Deployment URL: https://huggingface.co/spaces/CarvalhoClaudia/cyclesync
- Main user flow:
  1. Nutzerin trägt in der Sidebar Zyklustag, Symptome, Schlafdaten, Ruhepuls und das geplante Workout ein.
  2. App ruft `CycleSyncRecommender.recommend()` auf ([`src/recommender.py`](../src/recommender.py)).
  3. ML-Vorhersage erscheint in 4 Metric-Kacheln (Zyklusphase, Intensität, Recovery, Risiko).
  4. LLM-Erklärung erscheint darunter, mit aufklappbarer Quellenliste (PubMed-Links) und Modell-Konfidenz.
- Screenshot or short demo: Siehe [`docs/screenshots/`](screenshots/) für UI-Screenshots, Modell-Vergleich, Confusion Matrix und Feature Importance.

Guidance hint: Deployment must be usable.
Evidence hint: Add screenshots or short demo references.

---

## 4. Execution Instructions

- Environment setup:
```bash
  git clone https://github.com/carvacla/cyclesync_KI_project.git
  cd cyclesync_KI_project
  python -m venv .venv
  source .venv/bin/activate    # Mac/Linux
  pip install -r requirements.txt
  cp .env.example .env         # OPENAI_API_KEY eintragen
```
- Data setup: Notebook [`notebooks/01_data_acquisition.ipynb`](../notebooks/01_data_acquisition.ipynb) ausführen — generiert synthetische Cycle- und Workout-Daten (deterministisch via `RANDOM_STATE=42`) und lädt ~300 PubMed-Abstracts via Biopython (NCBI E-utilities API).
- Training command(s): Notebooks 02 → 03 → 04 in Reihenfolge ausführen. Ergebnisse: `models/best_classifier.joblib`, `models/feature_meta.json`, `models/chroma_db/`.
- Inference/run command(s):
```bash
  streamlit run app.py
```
- Reproducibility notes: Alle Notebooks nutzen `RANDOM_STATE=42`. Versionen sind in [`requirements.txt`](../requirements.txt) gepinnt (insbesondere `scikit-learn==1.6.1`, weil das Modell mit dieser Version trainiert wurde). Python 3.11+ empfohlen. Für HF Spaces Deployment: `OPENAI_API_KEY` als Secret unter Space Settings → Variables and secrets.

Guidance hint: Another person should be able to run your project from this section.
Evidence hint: Include exact commands and versions.

---

## 5. Optional Bonus Evidence

Use this section for exceptional work beyond the core requirements.

- [ ] Third selected block implemented with strong quality
- [x] More than two data sources used with clear added value
- [ ] A core section is done exceptionally well
- [x] Extended evaluation
- [x] Ethics, bias, or fairness analysis
- [x] Creative or exceptional use case

Evidence for selected bonus items:

**More than two data sources** — Drei verschiedene Datenquellen mit klarem Mehrwert: (1) synthetisches Cycle-Tracking-Dataset für Phase- und Symptom-Modellierung, (2) synthetisches Workout-Dataset für Trainings-Performance-Annahmen, (3) echte PubMed-Abstracts als wissenschaftliche Wissensbasis. Die Trennung erlaubt es, ML-Vorhersage und natürlichsprachliche Begründung unabhängig zu validieren.

**Extended evaluation** — Über die Pflicht-Metriken hinaus wurde Feature-Importance-Analyse durchgeführt (zeigt, dass `sleep_quality` und `day_in_cycle` die wichtigsten Prädiktoren sind). Die methodische Reflexion zu den perfekten Tree-Modell-Scores (siehe 2A.5) ist ein Beispiel für kritische Selbstbewertung statt unkritischer Metrik-Maximierung.

**Ethics, bias, or fairness analysis** — Sportwissenschaftliche Studien sind historisch von männlichen Probanden dominiert (McNulty et al. 2020 berichten, dass < 39% der sportwissenschaftlichen Studien weibliche Teilnehmerinnen einschliessen). Daraus abgeleitete Empfehlungen können für weibliche Athletinnen systematisch suboptimal sein. CycleSync adressiert diese Lücke explizit. Limitationen werden in der App durch einen sichtbaren Disclaimer markiert (keine medizinische Beratung). Datenschutz: keine Persistenz der Nutzereingaben, alle Berechnungen pro Request. Limitationen der synthetischen Daten werden in der Doku transparent gemacht.

**Creative or exceptional use case** — Die Kombination eines Zyklusphasen-abhängigen Trainings-Klassifikators mit einer RAG-Pipeline über aktuelle PubMed-Literatur in einem deutschsprachigen Streamlit-Frontend mit medizinischer Disclaimern und PubMed-Quellenangaben adressiert einen real existierenden Bedarf in einer unterforschten Domäne.
