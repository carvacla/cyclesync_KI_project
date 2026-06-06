"""Orchestriert ML-Vorhersage und RAG-Erklärung zu einer Gesamtempfehlung."""

from typing import Any

from .ml_model import CyclePredictor
from .rag_pipeline import RAGPipeline


class CycleSyncRecommender:
    def __init__(self):
        self.predictor = CyclePredictor()
        self.rag = RAGPipeline()

    def recommend(self, user_input: dict[str, Any]) -> dict[str, Any]:
        # 1) ML-Vorhersage
        prediction = self.predictor.predict(user_input)
        # 2) RAG-Erklärung
        rag_output = self.rag.explain(user_profile=user_input, ml_prediction=prediction)
        return {
            "prediction": prediction,
            "explanation": rag_output["explanation"],
            "sources": rag_output["sources"],
        }
