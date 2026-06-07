"""CycleSync — Streamlit Frontend für HuggingFace Spaces."""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(
    page_title="CycleSync",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_recommender():
    from src.recommender import CycleSyncRecommender
    return CycleSyncRecommender()


st.title("🌸 CycleSync")
st.markdown("**Zyklusbasierte Trainings- & Recovery-Empfehlungen, erklärt mit Wissenschaft.**")
st.caption("Forschungsprojekt der ZHAW — keine medizinische Beratung.")

with st.sidebar:
    st.header("Deine Daten")
    day_in_cycle = st.number_input("Tag im Zyklus (1 = erster Tag der Periode)", 1, 40, 14)
    symptoms = st.multiselect(
        "Symptome heute",
        options=["cramps", "fatigue", "mood_low", "headache", "bloating", "tender_breasts"],
        default=[],
    )
    if not symptoms:
        symptoms = ["none"]
    sleep_hours = st.slider("Schlafstunden letzte Nacht", 3.0, 12.0, 7.5, 0.5)
    sleep_quality = st.slider("Schlafqualität (1=schlecht, 10=top)", 1, 10, 7)
    resting_hr = st.number_input("Ruhepuls (bpm)", 40, 100, 65)
    planned_sport = st.selectbox(
        "Geplante Sportart",
        ["running", "cycling", "strength", "yoga", "hiit", "walking"],
    )
    planned_duration = st.number_input("Geplante Dauer (Min)", 10, 180, 45)
    age = st.number_input("Alter", 14, 70, 28)
    fitness_level = st.selectbox("Fitness-Level", ["beginner", "intermediate", "advanced"], index=1)
    submit = st.button("Empfehlung generieren", type="primary", use_container_width=True)

if submit:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY fehlt. In HuggingFace Spaces unter Settings → Secrets eintragen.")
        st.stop()

    user_input = {
        "day_in_cycle": day_in_cycle,
        "symptoms": symptoms,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "resting_hr": resting_hr,
        "planned_sport": planned_sport,
        "planned_duration_min": planned_duration,
        "age": age,
        "fitness_level": fitness_level,
    }

    with st.spinner("Analysiere Daten und suche relevante Studien…"):
        try:
            recommender = load_recommender()
            result = recommender.recommend(user_input)
        except FileNotFoundError as e:
            st.error(f"Setup unvollständig: {e}")
            st.stop()

    pred = result["prediction"]
    intensity_emoji = {"low": "🟢", "moderate": "🟡", "high": "🔴"}.get(pred["intensity"], "")
    risk_emoji = {"low": "✅", "moderate": "⚠️", "high": "🚨"}.get(pred["risk"], "")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Zyklusphase", pred["phase"])
    col2.metric("Empfohlene Intensität", f"{intensity_emoji} {pred['intensity']}")
    col3.metric("Recovery-Zeit", f"{pred['recovery_hours']} h")
    col4.metric("Belastungsrisiko", f"{risk_emoji} {pred['risk']}")

    st.subheader("Warum diese Empfehlung?")
    st.write(result["explanation"])

    with st.expander("📚 Zitierte Studien"):
        for src in result["sources"]:
            st.markdown(
                f"- **{src['title']}** ({src['year']}) — "
                f"[PubMed:{src['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{src['pmid']}/)"
            )

    with st.expander("🔬 Modell-Konfidenz"):
        st.write(f"Konfidenz der Vorhersage: **{pred['confidence']:.0%}**")
else:
    st.info("👈 Trage links deine Daten ein und klicke auf **Empfehlung generieren**.")

st.markdown("---")
st.caption("Datenquellen: Synthetische Daten + PubMed via NCBI E-utilities.")
