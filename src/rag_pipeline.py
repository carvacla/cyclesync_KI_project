"""RAG-Pipeline für CycleSync."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = Path(__file__).resolve().parents[1] / "models" / "chroma_db"

PROMPT_TEMPLATE = """\
Rolle: Du bist sportwissenschaftliche Beraterin für zyklusbasiertes Training.

Aufgabe: Erkläre der Nutzerin die folgende Trainingsempfehlung.

Vorgabe an die Antwort:
1. Beginne mit der konkreten Empfehlung in einem Satz.
2. Erkläre die physiologische Begründung in 1-2 Sätzen.
3. Verweise auf mindestens 2 der bereitgestellten Studien per [PMID:xxxx].
4. Schließe mit einem kurzen Disclaimer.

Nutzerin-Daten:
- Zyklusphase: {phase} (Tag {day_in_cycle})
- Geplante Sportart: {sport}
- Schlafqualität: {sleep_quality}/10
- Symptome: {symptoms}
- Vorhergesagte Intensität: {intensity}

Verfügbare Studien:
{retrieved_docs}

Antwort auf Deutsch:
"""


class RAGPipeline:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o-mini",
        top_k: int = 4,
    ):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        if not CHROMA_PATH.exists():
            raise FileNotFoundError(
                f"Vektor-DB nicht gefunden unter {CHROMA_PATH}. "
                "Bitte zuerst notebooks/04_rag_pipeline.ipynb ausführen."
            )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
        )
        self.vectorstore = Chroma(
            persist_directory=str(CHROMA_PATH),
            embedding_function=self.embeddings,
            collection_name="pubmed_cyclesync",
        )
        self.llm = ChatOpenAI(model=llm_model, temperature=0.3)
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.top_k = top_k

    @staticmethod
    def _format_docs(docs) -> str:
        parts = []
        for d in docs:
            pmid = d.metadata.get("pmid", "?")
            title = d.metadata.get("title", "")
            snippet = d.page_content[:400]
            parts.append(f"[PMID:{pmid}] {title}\n{snippet}\n")
        return "\n---\n".join(parts)

    def explain(self, user_profile: dict, ml_prediction: dict) -> dict[str, Any]:
        symptoms = user_profile.get("symptoms", ["none"])
        query = (
            f"{ml_prediction['phase']} phase {user_profile['planned_sport']} "
            f"intensity {ml_prediction['intensity']} {' '.join(symptoms)}"
        )
        retrieved = self.vectorstore.similarity_search(query, k=self.top_k)

        chain = self.prompt | self.llm
        response = chain.invoke({
            "phase": ml_prediction["phase"],
            "day_in_cycle": user_profile.get("day_in_cycle", "?"),
            "sport": user_profile["planned_sport"],
            "sleep_quality": user_profile.get("sleep_quality", "?"),
            "symptoms": ", ".join(symptoms),
            "intensity": ml_prediction["intensity"],
            "retrieved_docs": self._format_docs(retrieved),
        })

        return {
            "explanation": response.content,
            "sources": [
                {
                    "pmid": d.metadata.get("pmid", "?"),
                    "title": d.metadata.get("title", ""),
                    "year": d.metadata.get("year", 0),
                }
                for d in retrieved
            ],
        }
