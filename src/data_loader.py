from pathlib import Path
import pandas as pd

DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_fitrec(filtered: bool = True) -> pd.DataFrame:
    """Lädt den FitRec-Datensatz.

    Args:
        filtered: Wenn True, wird auf weibliche User gefiltert.

    Returns:
        DataFrame mit Workout-Daten.
    """
    raise NotImplementedError("In Notebook 01/02 implementieren und hierher migrieren.")


def load_cycle_data() -> pd.DataFrame:
    """Lädt den Menstruationszyklus-Datensatz."""
    raise NotImplementedError("In Notebook 01/02 implementieren und hierher migrieren.")


def load_pubmed_abstracts() -> list[dict]:
    """Lädt PubMed-Abstracts als Liste von Dicts."""
    raise NotImplementedError("In Notebook 01 implementieren und hierher migrieren.")
