"""Lädt die persistierten Datensätze."""

import json
from pathlib import Path
import pandas as pd

DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
ABSTRACTS_PATH = Path(__file__).resolve().parents[1] / "data" / "pubmed_abstracts"


def load_cycle_data(processed: bool = True) -> pd.DataFrame:
    path = (DATA_PROCESSED / "cycle_processed.csv") if processed else (DATA_RAW / "cycle_tracking.csv")
    return pd.read_csv(path)


def load_workouts() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "workouts.csv")


def load_pubmed_abstracts() -> list[dict]:
    with open(ABSTRACTS_PATH / "pubmed_abstracts.jsonl") as f:
        return [json.loads(line) for line in f]
