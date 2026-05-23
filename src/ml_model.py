from pathlib import Path
from typing import Any
import joblib

MODELS_PATH = Path(__file__).resolve().parents[1] / "models"


class CyclePredictor:
    """Lädt das trainierte ML-Modell und liefert Empfehlungen."""

    def __init__(self, model_filename: str = "best_classifier.joblib"):
        self.model_path = MODELS_PATH / model_filename
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modell nicht gefunden: {self.model_path}\n"
                "Bitte zuerst notebook 03_ml_modeling.ipynb ausführen."
            )
        self.model = joblib.load(self.model_path)

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Liefert Empfehlung als Dict.

        Returns:
            {
                'phase': 'follicular' | 'ovulation' | 'luteal' | 'menstruation',
                'intensity': 'low' | 'moderate' | 'high',
                'recovery_hours': int,
                'risk': 'low' | 'moderate' | 'high',
                'confidence': float,
            }
        """
        # TODO: Feature-Vektor bauen, model.predict() + predict_proba()
        raise NotImplementedError
