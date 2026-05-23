# CycleSync — Zyklusbasierter Trainings- & Recovery-Coach

> Semesterprojekt im Modul **KI-Anwendungen** (ZHAW, FS 2026)
> Kombination der Blöcke **ML Numeric Data** + **NLP (RAG)**

## Projektidee

CycleSync ist eine Web-App, die Trainingsempfehlungen basierend auf der aktuellen
Zyklusphase einer Nutzerin generiert und diese mit wissenschaftlicher Literatur erklärt.

- **ML-Komponente**: Sagt aus Zyklusphase, Symptomen, Schlafdaten und Workout-Historie
  eine empfohlene Trainingsintensität, Recovery-Zeit und ein Belastungsrisiko voraus.
- **NLP-Komponente (RAG)**: Erklärt die Empfehlung mit Bezug auf aktuelle
  sportwissenschaftliche Studien aus PubMed.

**Social Impact**: Sportwissenschaft ist historisch männerlastig — Tools wie dieses
adressieren eine reale Lücke in personalisierter Trainingsberatung.

## Architektur

```
User Input (strukturiert)
    ↓
ML-Pipeline  →  Phase-Klassifikation + Trainings-Empfehlung
    ↓
ML-Output als Kontext für NLP
    ↓
RAG (PubMed-Vektor-DB + LLM)  →  Personalisierte Erklärung mit Quellen
    ↓
User Output (Empfehlung + Begründung + Studien-Links)
```

## Repository-Struktur

```
cyclesync/
├── notebooks/           # Datenexploration, ML-Training, RAG-Aufbau
├── src/                 # Wiederverwendbare Python-Module
├── app/                 # Streamlit-Frontend (Deployment-Einstiegspunkt)
├── data/                # Daten (raw + processed, via .gitignore ausgeschlossen)
├── models/              # Trainierte Modelle (joblib)
├── docs/                # Dokumentation + Screenshots
└── tests/               # Optionale Unit-Tests
```

## Reproduktion der Pipeline

Die Notebooks sind nummeriert und müssen in dieser Reihenfolge ausgeführt werden:

1. `01_data_acquisition.ipynb` — Lädt FitRec, Zyklus-Daten und PubMed-Abstracts
2. `02_eda.ipynb` — Exploratory Data Analysis
3. `03_ml_modeling.ipynb` — Trainiert und vergleicht ML-Modelle, speichert das beste
4. `04_rag_pipeline.ipynb` — Baut die Vektor-DB und testet das RAG-System
5. `05_integration_test.ipynb` — End-to-End-Test der Pipeline

## Deployment

Live-App: `https://huggingface.co/spaces/carvacla/cyclesync`'

## Dokumentation

Die vollständige Projekt-Dokumentation nach Q&A-Template findest du in
[`docs/documentation.md`](docs/documentation.md).

## Disclaimer

CycleSync ist ein studentisches Forschungsprojekt und **kein medizinisches Gerät**.
Die Empfehlungen ersetzen keine professionelle medizinische oder sportwissenschaftliche
Beratung. Bei gesundheitlichen Beschwerden bitte einen Facharzt konsultieren.

