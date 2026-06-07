"""RAG-Pipeline für CycleSync."""

import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHROMA_PATH = ROOT / "models" / "chroma_db"
ABSTRACTS_PATH = ROOT / "data" / "pubmed_abstracts" / "pubmed_abstracts.jsonl"

PROMPT_TEMPLATE = """Rolle: Du bist sportwissenschaftliche Beraterin fuer zyklusbasiertes Training.

Aufgabe: Erklaere der Nutzerin die folgende Trainingsempfehlung.

Vorgabe an die Antwort:
1. Beginne mit der konkreten Empfehlung in einem Satz.
2. Erklaere die physiologische Begruendung in 1-2 Saetzen.
3. Verweise auf mindestens 2 der bereitgestellten Studien per [PMID:xxxx].
4. Schliesse mit einem kurzen Disclaimer.

Nutzerin-Daten:
- Zyklusphase: {phase} (Tag {day_in_cycle})
- Geplante Sportart: {sport}
- Schlafqualitaet: {sleep_quality}/10
- Symptome: {symptoms}
- Vorhergesagte Intensitaet: {intensity}

Verfuegbare Studien:
{retrieved_docs}

Antwort auf Deutsch:
"""


def _build_vectorstore_from_jsonl(embeddings):
    """Baut die Vektor-DB aus dem JSONL-File neu auf (in-memory, kein Persist)."""
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document

    docs = []
    with open(ABSTRACTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            a = json.loads(line)
            content = f"Title: {a['title']}\n\nAbstract: {a['abstract']}"
            metadata = {
                "pmid": str(a.get("pmid", "")),
                "title": str(a.get("title", ""))[:200],
                "year": int(a.get("year") or 0),
            }
            docs.append(Document(page_content=content, metadata=metadata))

    # In-memory Chroma (kein persist_directory)
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="pubmed_cyclesync",
    )


class RAGPipeline:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 llm_model="gpt-4o-mini", top_k=4):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model, model_kwargs={"device": "cpu"})

        # Vektor-DB beim Start aus JSONL bauen (vermeidet Versions-Inkompatibilitaet)
        print("Baue Vektor-DB aus PubMed-Abstracts...")
        self.vectorstore = _build_vectorstore_from_jsonl(self.embeddings)
        print("Vektor-DB bereit.")

        self.llm = ChatOpenAI(model=llm_model, temperature=0.3)
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        self.top_k = top_k

    @staticmethod
    def _format_docs(docs):
        parts = []
        for d in docs:
            pmid = d.metadata.get("pmid", "?")
            title = d.metadata.get("title", "")
            snippet = d.page_content[:400]
            parts.append("[PMID:" + str(pmid) + "] " + str(title) + "\n" + snippet + "\n")
        return "\n---\n".join(parts)

    def explain(self, user_profile, ml_prediction):
        symptoms = user_profile.get("symptoms", ["none"])
        phase = ml_prediction["phase"]
        sport = user_profile["planned_sport"]
        intensity = ml_prediction["intensity"]
        symptoms_str = " ".join(symptoms)
        query = phase + " phase " + sport + " intensity " + intensity + " " + symptoms_str

        retrieved = self.vectorstore.similarity_search(query, k=self.top_k)
        chain = self.prompt | self.llm
        response = chain.invoke({
            "phase": phase,
            "day_in_cycle": user_profile.get("day_in_cycle", "?"),
            "sport": sport,
            "sleep_quality": user_profile.get("sleep_quality", "?"),
            "symptoms": ", ".join(symptoms),
            "intensity": intensity,
            "retrieved_docs": self._format_docs(retrieved),
        })
        return {
            "explanation": response.content,
            "sources": [{
                "pmid": d.metadata.get("pmid", "?"),
                "title": d.metadata.get("title", ""),
                "year": d.metadata.get("year", 0),
            } for d in retrieved],
        }
