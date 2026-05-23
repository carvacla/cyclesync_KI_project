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
- Student: *<Vor- und Nachname eintragen>*
- GitHub repository URL: *<wird ergänzt>*
- Deployment URL: *<wird ergänzt nach HF Spaces Deployment>*
- Submission date: 07.06.2026

### Mandatory Setup Checks

- [ ] At least 2 blocks selected
- [ ] Multiple and different data sources used
- [ ] Deployment URL provided
- [ ] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

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
- Problem statement: Sportwissenschaftliche Forschung und gängige Trainings-Apps
  ignorieren weitgehend, dass sich die physiologischen Voraussetzungen für Training
  und Recovery über den Menstruationszyklus hinweg verändern. Frauen erhalten daher
  oft Empfehlungen, die auf Daten überwiegend männlicher Probanden basieren.
- Goal: Eine Web-App bereitstellen, die aus Zyklus- und Wellness-Daten eine
  begründete Trainingsempfehlung erzeugt und diese mit aktueller wissenschaftlicher
  Literatur erklärt.
- Success criteria:
  1. ML-Modell klassifiziert die Zyklusphase mit F1 (macro) ≥ 0.70 auf dem Hold-out-Set.
  2. RAG-Pipeline liefert pro Empfehlung mindestens 2 thematisch passende PubMed-Quellen.
  3. End-to-End-Latenz < 10s in der deployten App.
  4. Deployment auf HuggingFace Spaces öffentlich erreichbar.

### 1.2 Integration Logic
- How the selected blocks interact: Der ML-Block klassifiziert die Zyklusphase und
  leitet daraus eine strukturierte Trainingsempfehlung ab (Intensität, Recovery-Zeit,
  Belastungsrisiko). Diese Vorhersage wird zusammen mit dem User-Profil als
  Kontext an die NLP/RAG-Pipeline weitergegeben. Das RAG-System retrieved passende
  PubMed-Abstracts und das LLM generiert eine personalisierte Erklärung mit
  Quellenangaben.
- Data and output flow between blocks: User-Input → ML-Pipeline (Phase + Empfehlung)
  → Output als strukturierter Kontext → RAG-Retrieval auf PubMed-Vektor-DB →
  LLM-Generation → kombinierte Ausgabe an User. Siehe Pipeline-Übersicht in
  [`README.md`](../README.md) und End-to-End-Test in
  [`notebooks/05_integration_test.ipynb`](../notebooks/05_integration_test.ipynb).

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
| 1 | FitRec Endomondo Dataset (UCSD, Ni et al.) — https://cseweb.ucsd.edu/~jmcauley/datasets/fitrec.html | Numeric/structured (workout logs, HR, sport, weather) | ~250k workouts | Trainings- und Herzfrequenz-Profile zur Anreicherung der Empfehlungs-Features |
| 2 | Menstrual Cycle / Fertility Awareness Dataset (Kaggle) | Numeric/categorical (Zyklustag, Symptome, BBT, Schlaf) | *<Zeilenzahl ergänzen nach Download>* | Training des Phase-Klassifikators |
| 3 |  |  |  |  |

#### 2A.2 Preprocessing and Features
- Cleaning steps: *<nach EDA ergänzen — Beispiele: Entfernen unplausibler Zyklustage (>40), Imputation fehlender Schlafwerte, Filterung von FitRec auf weibliche User und Ausdauersportarten>*
- Preprocessing steps: *<Beispiele: One-Hot-Encoding der Sportart und Symptome, StandardScaler für numerische Features, Train/Val/Test-Split 70/15/15 stratifiziert nach Phase>*
- Feature engineering and selection: *<Beispiele: Phase-Indikator aus Zyklustag, gleitender 7-Tage-Schlafdurchschnitt, Symptom-Count, Phase × Symptom-Interaktion>*

Siehe [`notebooks/02_eda.ipynb`](../notebooks/02_eda.ipynb) für EDA und Findings
sowie [`notebooks/03_ml_modeling.ipynb`](../notebooks/03_ml_modeling.ipynb#feature-engineering)
für Feature Engineering.

#### 2A.3 Model Selection
- Models tested: Logistic Regression (Baseline), Random Forest, XGBoost.
- Why these models were chosen: Logistic Regression als interpretierbare Baseline,
  Random Forest und XGBoost wegen Robustheit bei gemischten Features und
  Standardwahl für strukturierte Tabellendaten.

#### 2A.4 Model Comparison and Iterations
| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline | Rohfeatures, keine Skalierung | Logistic Regression | F1 (macro) | — |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |

#### 2A.5 Evaluation and Error Analysis
- Metrics used: Accuracy, F1 (macro), Confusion Matrix; für Recovery-Regression MAE/R².
- Final results: *<einfügen>*
- Error patterns and likely causes: *<einfügen, z. B. Verwechslung zwischen später Follikel- und früher Lutealphase wegen ähnlicher Symptomprofile>*

#### 2A.6 Integration with Other Block(s)
- Inputs received from other block(s): Keine direkten Inputs aus dem NLP-Block in
  Iteration 1. (Optional in einer Erweiterung: Symptom-Stichworte aus Freitext-Input
  per LLM strukturiert extrahieren und als zusätzliche Features einspeisen.)
- Outputs provided to other block(s): Strukturierte Vorhersage (Phase, Intensität,
  Recovery-Stunden, Risiko, Konfidenz) wird als Kontext an die RAG-Pipeline
  übergeben. Siehe [`src/recommender.py`](../src/recommender.py).

Guidance hint: Keep entries practical and evidence-based.
Evidence hint: Add values, not only claims.

### 2B. NLP (If selected)

#### 2B.1 Data Source(s)
List every usage of a data source as a separate entry. If the same source is used twice for different roles, add it twice.

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | PubMed Abstracts via NCBI E-utilities | Text (wissenschaftliche Abstracts) | ~300 Abstracts | Wissensbasis für RAG-Retrieval |
| 2 | User-Eingabe (Symptome, Sportart) aus Streamlit-UI | Text/strukturiert | pro Request | Eingangskontext für Prompt-Konstruktion |
| 3 | ML-Output (Phase, Empfehlung) aus 2A | Strukturiert (JSON) | pro Request | Strukturierter Kontext im Prompt |

#### 2B.2 Preprocessing and Prompt Design
- Text preprocessing: *<einfügen — Beispiele: Filterung leerer Abstracts, Deduplication nach PMID, Metadata-Anreicherung (Jahr, Autoren), Chunking nicht nötig, da Abstracts ohnehin kurz sind>*
- Prompt design or retrieval setup: Lokale Embeddings mit Sentence-Transformers
  (`all-MiniLM-L6-v2`), Chroma als persistente Vektor-DB, Top-k Retrieval (k=4).
  Prompt enthält strukturierten User-Kontext, ML-Vorhersage, retrieved Abstracts
  und verlangt eine knappe Erklärung mit `[PMID:xxxx]`-Zitaten plus Disclaimer.
  Siehe [`src/rag_pipeline.py`](../src/rag_pipeline.py) und
  [`notebooks/04_rag_pipeline.ipynb`](../notebooks/04_rag_pipeline.ipynb).

#### 2B.3 Approach Selection
- Approach used (classical NLP, transformer, RAG, prompt engineering): Retrieval-Augmented
  Generation mit GPT-4o-mini als Generator und einem lokalen Sentence-Transformer
  als Encoder.
- Alternatives considered: Reines Prompt-Engineering ohne Retrieval (verworfen wegen
  fehlender Quellenbelege und Halluzinationsrisiko bei medizinischen Aussagen);
  klassisches Information-Retrieval mit TF-IDF (verworfen wegen schlechterer
  semantischer Trefferqualität).

#### 2B.4 Comparison and Iterations
| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline-RAG | MiniLM-Embeddings, Top-4 Retrieval, Zero-Shot-Prompt | gpt-4o-mini | Qualitative Bewertung 5 Testfälle | — |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |

#### 2B.5 Evaluation and Error Analysis
- Evaluation strategy: Qualitative Bewertung von 10 Test-Szenarien (Relevanz der
  Retrieval-Treffer, Faktentreue der Erklärung, Tonfall, Disclaimer-Konformität);
  quantitativ optional Retrieval-Hit-Rate gegen manuell annotierte Gold-Quellen.
- Results: *<einfügen>*
- Error patterns and likely causes: *<einfügen — typische Risiken: zu generische
  Erklärungen, Zitate, die im Prompt enthalten waren, aber nicht im retrieved Doc;
  Themendrift bei sehr kurzen User-Inputs>*

#### 2B.6 Integration with Other Block(s)
- Inputs received from other block(s): Strukturierte ML-Vorhersage (Phase,
  Intensität, Recovery, Risiko, Konfidenz) — wird Teil des Prompt-Kontexts und
  steuert die Retrieval-Query.
- Outputs provided to other block(s): Natürlichsprachliche Erklärung + Liste der
  zitierten Studien (PMID, Titel, Jahr) zurück an die App-Schicht zur Anzeige.

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

- Deployment URL: *<HuggingFace Spaces URL nach Deployment einfügen>*
- Main user flow:
  1. Nutzerin trägt in der Sidebar Zyklustag, Symptome, Schlafdaten und geplantes
     Workout ein.
  2. App ruft `CycleSyncRecommender.recommend()` auf
     (siehe [`src/recommender.py`](../src/recommender.py)).
  3. ML-Vorhersage wird in drei Metric-Kacheln angezeigt (Intensität, Recovery, Risiko).
  4. LLM-Erklärung erscheint darunter, mit aufklappbarer Quellenliste (PubMed-Links).
- Screenshot or short demo: Siehe [`docs/screenshots/`](screenshots/).

Guidance hint: Deployment must be usable.
Evidence hint: Add screenshots or short demo references.

---

## 4. Execution Instructions

- Environment setup:
  ```bash
  git clone <repo-url>
  cd cyclesync
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  cp .env.example .env   # OPENAI_API_KEY eintragen
  ```
- Data setup: Notebook [`notebooks/01_data_acquisition.ipynb`](../notebooks/01_data_acquisition.ipynb)
  ausführen — lädt FitRec, Kaggle-Zyklusdaten und PubMed-Abstracts nach `data/`.
- Training command(s): Notebooks 02 → 03 → 04 in Reihenfolge ausführen.
  Modell wird unter `models/best_classifier.joblib` und Vektor-DB unter
  `models/chroma_db/` persistiert.
- Inference/run command(s):
  ```bash
  streamlit run app/streamlit_app.py
  ```
- Reproducibility notes: Alle Notebooks verwenden `RANDOM_STATE=42`. Versionen
  in [`requirements.txt`](../requirements.txt) sind gepinnt. Python 3.11 empfohlen.

Guidance hint: Another person should be able to run your project from this section.
Evidence hint: Include exact commands and versions.

---

## 5. Optional Bonus Evidence

Use this section for exceptional work beyond the core requirements.

- [ ] Third selected block implemented with strong quality
- [ ] More than two data sources used with clear added value
- [ ] A core section is done exceptionally well
- [ ] Extended evaluation
- [x] Ethics, bias, or fairness analysis
- [x] Creative or exceptional use case

Evidence for selected bonus items:

**Ethics, bias, or fairness analysis** — Sportwissenschaftliche Studien sind
historisch von männlichen Probanden dominiert; daraus abgeleitete Trainings-
empfehlungen können für weibliche Athletinnen systematisch suboptimal sein.
CycleSync adressiert diese Lücke explizit. Limitationen werden in der App durch
einen sichtbaren Disclaimer markiert (keine medizinische Beratung). Datenschutz:
keine Persistenz der Nutzerdaten, alle Berechnungen pro Request.

**Creative or exceptional use case** — Die Kombination eines Zyklusphasen-
Klassifikators mit RAG über aktuelle PubMed-Literatur in einem zugänglichen
Web-Frontend ist nach unserer Recherche in der Form weder als kommerzielles
Produkt noch in studentischen Projekten verbreitet.
