from typing import Any

from .ml_model import CyclePredictor
from .rag_pipeline import RAGPipeline


class CycleSyncRecommender:
    """Vereint ML-Vorhersage und RAG-Erklärung."""

    def __init__(self):
        self.predictor = CyclePredictor()
        self.rag = RAGPipeline()

    def recommend(self, user_input: dict[str, Any]) -> dict[str, Any]:
        """Hauptmethode für die App.

        Args:
            user_input: Dict mit User-Eingaben aus dem Frontend.

        Returns:
            Dict mit:
            - 'prediction': Ergebnis vom ML-Modell
            - 'explanation': Erklärung vom LLM
            - 'sources': Liste der zitierten Studien
        """
        # 1) ML-Vorhersage
        prediction = self.predictor.predict(user_input)

        # 2) RAG-Erklärung mit ML-Output als Kontext
        rag_output = self.rag.explain(user_profile=user_input, ml_prediction=prediction)

        return {
            "prediction": prediction,
            "explanation": rag_output["explanation"],
            "sources": rag_output["sources"],
        }
