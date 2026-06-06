"""ML-Inferenz: Trainiertes Modell laden und Empfehlung liefern.

Training erfolgt in notebooks/03_ml_modeling.ipynb. Hier wird das gespeicherte
Modell nur GELADEN (Trennung Training/Inferenz).
"""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

MODELS_PATH = Path(__file__).resolve().parents[1] / "models"


def get_phase(day_in_cycle: int) -> str:
    if day_in_cycle <= 5:
        return "menstruation"
    elif day_in_cycle <= 13:
        return "follicular"
    elif day_in_cycle <= 16:
        return "ovulation"
    else:
        return "luteal"


class CyclePredictor:
    SYMPTOM_LIST = [
        "cramps", "fatigue", "mood_low", "headache", "bloating", "tender_breasts",
    ]

    def __init__(self, model_filename: str = "best_classifier.joblib"):
        self.model_path = MODELS_PATH / model_filename
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modell nicht gefunden: {self.model_path}\n"
                "Bitte zuerst notebooks/03_ml_modeling.ipynb ausführen."
            )
        self.model = joblib.load(self.model_path)
        with open(MODELS_PATH / "feature_meta.json") as f:
            self.meta = json.load(f)

    def _build_features(self, user_input: dict) -> pd.DataFrame:
        """Konvertiert User-Input in das vom Modell erwartete Format."""
        symptoms = user_input.get("symptoms", ["none"])
        if not symptoms:
            symptoms = ["none"]

        # Standard-Werte für nicht angegebene Felder
        day_in_cycle = int(user_input["day_in_cycle"])
        phase = get_phase(day_in_cycle)

        # BBT schätzen, falls nicht gegeben
        bbt_default = {"menstruation": 36.4, "follicular": 36.4,
                       "ovulation": 36.55, "luteal": 36.75}
        bbt = user_input.get("bbt_celsius", bbt_default[phase])

        row = {
            "day_in_cycle": day_in_cycle,
            "bbt_celsius": float(bbt),
            "sleep_hours": float(user_input.get("sleep_hours", 7.5)),
            "sleep_quality": int(user_input.get("sleep_quality", 7)),
            "resting_hr": int(user_input.get("resting_hr", 65)),
            "age": int(user_input.get("age", 30)),
            "symptom_count": len([s for s in symptoms if s != "none"]),
        }
        for s in self.SYMPTOM_LIST:
            row[f"sym_{s}"] = int(s in symptoms)

        row["phase"] = phase
        row["fitness_level"] = user_input.get("fitness_level", "intermediate")
        return pd.DataFrame([row])

    def predict(self, user_input: dict) -> dict[str, Any]:
        X = self._build_features(user_input)
        intensity = str(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0]
        confidence = float(max(proba))

        # Recovery-Stunden heuristisch aus Intensität ableiten
        recovery_map = {"low": 12, "moderate": 24, "high": 36}
        recovery_hours = recovery_map.get(intensity, 24)

        # Risiko aus Schlaf + Symptomen
        symptom_count = int(X["symptom_count"].iloc[0])
        sleep_quality = int(X["sleep_quality"].iloc[0])
        if sleep_quality <= 4 or symptom_count >= 3:
            risk = "high"
        elif sleep_quality <= 6 or symptom_count >= 2:
            risk = "moderate"
        else:
            risk = "low"

        return {
            "phase": str(X["phase"].iloc[0]),
            "intensity": intensity,
            "recovery_hours": int(recovery_hours),
            "risk": risk,
            "confidence": round(confidence, 3),
        }
