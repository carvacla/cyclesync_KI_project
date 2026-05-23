import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = Path(__file__).resolve().parents[1] / "models" / "chroma_db"


class RAGPipeline:
    """RAG-Pipeline für sportwissenschaftliche Erklärungen."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o-mini",
        top_k: int = 4,
    ):
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.top_k = top_k

        # TODO: Embeddings, Vector Store und LLM hier initialisieren
        # self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        # self.vectorstore = Chroma(persist_directory=str(CHROMA_PATH), ...)
        # self.llm = ChatOpenAI(model=llm_model, temperature=0.3)

    def retrieve(self, query: str) -> list[dict]:
        """Liefert die top_k relevantesten PubMed-Abstracts."""
        raise NotImplementedError

    def explain(self, user_profile: dict, ml_prediction: dict) -> dict[str, Any]:
        """Generiert eine personalisierte Erklärung mit Quellenangabe.

        Returns:
            {
                'explanation': str,
                'sources': list[{'pmid': str, 'title': str, 'year': int}],
            }
        """
        raise NotImplementedError
