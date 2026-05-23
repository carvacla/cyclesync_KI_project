import sys
from pathlib import Path

# src/-Module verfügbar machen
sys.path.append(str(Path(__file__).resolve().parents[1]))

import streamlit as st

# Optional importieren, weil zum Testen der UI vielleicht noch nicht alle
# Modelle existieren
try:
    from src.recommender import CycleSyncRecommender
    RECOMMENDER_AVAILABLE = True
except Exception as e:
    RECOMMENDER_AVAILABLE = False
    INIT_ERROR = e

# -------- Page Config --------
st.set_page_config(
    page_title="CycleSync — Zyklusbasiertes Training",
    page_icon="🌸",
    layout="wide",
)

# -------- Header --------
st.title("🌸 CycleSync")
st.markdown(
    "**Zyklusbasierte Trainings- & Recovery-Empfehlungen, erklärt mit Wissenschaft.**"
)
st.caption(
    "Forschungsprojekt der ZHAW — keine medizinische Beratung. "
    "Empfehlungen ersetzen keinen Arztbesuch."
)

# -------- Sidebar: Input --------
with st.sidebar:
    st.header("Deine Daten")

    day_in_cycle = st.number_input(
        "Tag im Zyklus (1 = erster Tag der Periode)",
        min_value=1, max_value=40, value=14,
    )

    symptoms = st.multiselect(
        "Symptome heute",
        options=[
            "none", "fatigue", "bloating", "cramps",
            "mood_low", "headache", "tender_breasts",
        ],
        default=["none"],
    )

    sleep_hours = st.slider("Schlafstunden letzte Nacht", 3.0, 12.0, 7.5, 0.5)
    sleep_quality = st.slider("Schlafqualität (1=schlecht, 10=top)", 1, 10, 7)

    planned_sport = st.selectbox(
        "Geplante Sportart",
        ["running", "cycling", "strength", "yoga", "hiit", "walking"],
    )
    planned_duration = st.number_input(
        "Geplante Dauer (Min)", min_value=10, max_value=180, value=45,
    )

    age = st.number_input("Alter", min_value=14, max_value=70, value=28)
    fitness_level = st.selectbox(
        "Fitness-Level", ["beginner", "intermediate", "advanced"], index=1,
    )

    submit = st.button("Empfehlung generieren", type="primary", use_container_width=True)

# -------- Main Output --------
if submit:
    if not RECOMMENDER_AVAILABLE:
        st.error(
            f"Recommender konnte nicht geladen werden: {INIT_ERROR}\n\n"
            "Bitte zuerst die Notebooks ausführen, damit Modelle und Vektor-DB existieren."
        )
    else:
        user_input = {
            "day_in_cycle": day_in_cycle,
            "symptoms": symptoms,
            "sleep_hours": sleep_hours,
            "sleep_quality": sleep_quality,
            "planned_sport": planned_sport,
            "planned_duration_min": planned_duration,
            "age": age,
            "fitness_level": fitness_level,
        }

        with st.spinner("Analysiere Daten und suche relevante Studien…"):
            try:
                recommender = CycleSyncRecommender()
                result = recommender.recommend(user_input)
            except NotImplementedError:
                st.warning(
                    "Pipeline noch nicht vollständig implementiert. "
                    "Bitte zuerst Notebooks 01–04 ausführen."
                )
                st.stop()

        # Vorhersage anzeigen
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Empfohlene Intensität", result["prediction"]["intensity"])
        with col2:
            st.metric("Recovery-Zeit", f"{result['prediction']['recovery_hours']} h")
        with col3:
            st.metric("Belastungsrisiko", result["prediction"]["risk"])

        st.subheader("Warum diese Empfehlung?")
        st.write(result["explanation"])

        with st.expander("📚 Zitierte Studien"):
            for src in result["sources"]:
                st.markdown(
                    f"- **{src['title']}** ({src['year']}) — "
                    f"[PubMed:{src['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{src['pmid']}/)"
                )
else:
    st.info("👈 Trage links deine Daten ein und klicke auf **Empfehlung generieren**.")

# -------- Footer --------
st.markdown("---")
st.caption(
    "Datenquellen: FitRec (UCSD), Kaggle Menstrual Cycle Dataset, "
    "PubMed via NCBI E-utilities. Code: github.com/<dein-user>/cyclesync"
)
