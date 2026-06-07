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


SYMPTOM_LABELS = {
    "cramps": "Krämpfe",
    "fatigue": "Müdigkeit / Erschöpfung",
    "mood_low": "Niedergeschlagene Stimmung",
    "headache": "Kopfschmerzen",
    "bloating": "Blähbauch / Wassereinlagerungen",
    "tender_breasts": "Brustspannen",
}

SPORT_LABELS = {
    "running": "Laufen / Joggen",
    "cycling": "Radfahren",
    "strength": "Krafttraining",
    "yoga": "Yoga / Stretching",
    "hiit": "HIIT (Hochintensives Intervalltraining)",
    "walking": "Spazieren / Wandern",
}

FITNESS_LABELS = {
    "beginner": "Einsteigerin",
    "intermediate": "Fortgeschritten",
    "advanced": "Sehr sportlich",
}

PHASE_LABELS = {
    "menstruation": "Menstruation",
    "follicular": "Follikelphase",
    "ovulation": "Eisprung",
    "luteal": "Lutealphase",
}

INTENSITY_LABELS = {
    "low": "niedrig",
    "moderate": "moderat",
    "high": "hoch",
}

RISK_LABELS = {
    "low": "niedrig",
    "moderate": "moderat",
    "high": "hoch",
}


# --- Header ---
st.title("🌸 CycleSync")
st.markdown(
    "**Zyklusbasierte Trainings- und Recovery-Empfehlungen, "
    "wissenschaftlich begründet.**"
)
st.caption(
    "📚 Studentisches Forschungsprojekt der ZHAW (Modul KI-Anwendungen, FS 2026). "
    "Diese App ist **keine medizinische Beratung** und ersetzt keinen Arztbesuch."
)

# --- Sidebar ---
with st.sidebar:
    st.header("📝 Deine Daten")
    st.caption("Trage die Daten so genau wie möglich ein. Je präziser deine Angaben, "
               "desto passender wird die Empfehlung.")

    st.subheader("🩸 Zyklus")
    day_in_cycle = st.number_input(
        "Tag im Zyklus",
        min_value=1, max_value=40, value=14,
        help="Tag 1 ist der erste Tag deiner Periode. "
             "Zähle ab dort durch (üblicherweise 28-Tage-Zyklus)."
    )

    symptoms_de = st.multiselect(
        "Wie fühlst du dich heute?",
        options=list(SYMPTOM_LABELS.values()),
        default=[],
        help="Wähle alle Symptome aus, die du heute hast."
    )
    # Deutsche Labels zurück auf englische Keys mappen für das ML-Modell
    reverse_symptom = {v: k for k, v in SYMPTOM_LABELS.items()}
    symptoms = [reverse_symptom[s] for s in symptoms_de] if symptoms_de else ["none"]

    st.subheader("😴 Schlaf")
    sleep_hours = st.slider(
        "Schlafstunden letzte Nacht",
        3.0, 12.0, 7.5, 0.5,
        help="Wie viele Stunden hast du tatsächlich geschlafen?"
    )
    sleep_quality = st.slider(
        "Schlafqualität",
        1, 10, 7,
        help="1 = sehr schlecht (ständig wach geworden), "
             "10 = perfekt erholt aufgewacht."
    )

    st.subheader("❤️ Körper")
    resting_hr = st.number_input(
        "Ruhepuls (Schläge pro Minute)",
        40, 100, 65,
        help="Idealerweise morgens nach dem Aufwachen messen. "
             "Üblicher Bereich: 60–80 bpm."
    )

    st.subheader("🏃‍♀️ Geplantes Training")
    sport_de = st.selectbox(
        "Welche Sportart hast du geplant?",
        options=list(SPORT_LABELS.values()),
        index=0,
    )
    reverse_sport = {v: k for k, v in SPORT_LABELS.items()}
    planned_sport = reverse_sport[sport_de]

    planned_duration = st.number_input(
        "Geplante Dauer (Minuten)",
        10, 180, 45,
        help="Wie lange möchtest du heute trainieren?"
    )

    st.subheader("👤 Über dich")
    age = st.number_input("Alter", 14, 70, 28)
    fitness_de = st.selectbox(
        "Wie sportlich bist du?",
        options=list(FITNESS_LABELS.values()),
        index=1,
        help="Einsteigerin: 0-2× pro Woche aktiv. "
             "Fortgeschritten: 3-4× pro Woche. "
             "Sehr sportlich: ≥5× pro Woche oder Wettkampfsport."
    )
    reverse_fitness = {v: k for k, v in FITNESS_LABELS.items()}
    fitness_level = reverse_fitness[fitness_de]

    submit = st.button(
        "🔮 Empfehlung generieren",
        type="primary",
        use_container_width=True
    )


# --- Main Output ---
if submit:
    if not os.getenv("OPENAI_API_KEY"):
        st.error(
            "⚠️ OPENAI_API_KEY fehlt. In HuggingFace Spaces "
            "unter Settings → Secrets eintragen."
        )
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

    with st.spinner("🔬 Analysiere deine Daten und suche relevante Studien…"):
        try:
            recommender = load_recommender()
            result = recommender.recommend(user_input)
        except FileNotFoundError as e:
            st.error(f"Setup unvollständig: {e}")
            st.stop()

    pred = result["prediction"]
    intensity_emoji = {"low": "🟢", "moderate": "🟡", "high": "🔴"}.get(pred["intensity"], "")
    risk_emoji = {"low": "✅", "moderate": "⚠️", "high": "🚨"}.get(pred["risk"], "")

    st.success("✨ Deine personalisierte Empfehlung")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Zyklusphase",
        PHASE_LABELS.get(pred["phase"], pred["phase"])
    )
    col2.metric(
        "Empfohlene Intensität",
        f"{intensity_emoji} {INTENSITY_LABELS.get(pred['intensity'], pred['intensity'])}"
    )
    col3.metric(
        "Recovery-Zeit",
        f"{pred['recovery_hours']} h"
    )
    col4.metric(
        "Belastungsrisiko",
        f"{risk_emoji} {RISK_LABELS.get(pred['risk'], pred['risk'])}"
    )

    st.subheader("💡 Begründung")
    st.write(result["explanation"])

    with st.expander("📚 Wissenschaftliche Quellen"):
        st.caption("Diese Studien aus der PubMed-Datenbank wurden für deine Empfehlung herangezogen:")
        for src in result["sources"]:
            st.markdown(
                f"- **{src['title']}** ({src['year']}) — "
                f"[PubMed:{src['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{src['pmid']}/)"
            )

    with st.expander("🔬 Technische Details"):
        st.write(f"**Modell-Konfidenz**: {pred['confidence']:.0%}")
        st.write(f"**Erkannte Zyklusphase**: {PHASE_LABELS.get(pred['phase'], pred['phase'])}")
        st.caption(
            "Die Konfidenz gibt an, wie sicher das Machine-Learning-Modell "
            "bei der Empfehlung ist. Werte über 80% deuten auf eine klare "
            "Datenlage hin."
        )
else:
    st.info(
        "👈 **So funktioniert's:** Trage links in der Seitenleiste deine Daten ein "
        "und klicke auf **🔮 Empfehlung generieren**."
    )

    with st.expander("ℹ️ Wie funktioniert CycleSync?"):
        st.markdown("""
        **CycleSync kombiniert zwei KI-Ansätze:**

        1. **Machine Learning** – Ein trainiertes Modell analysiert deine Eingaben
           (Zyklustag, Symptome, Schlaf, Ruhepuls) und sagt voraus, wie intensiv du
           heute trainieren solltest, wie lange du danach brauchst zur Erholung
           und welches Belastungsrisiko besteht.

        2. **Retrieval-Augmented Generation (RAG)** – Ein KI-Sprachmodell sucht
           passende wissenschaftliche Studien aus der medizinischen Datenbank
           PubMed und formuliert daraus eine verständliche Begründung für die
           Empfehlung.

        **Warum das wichtig ist:**
        Sportwissenschaftliche Studien basieren historisch überwiegend auf
        männlichen Probanden. Empfehlungen aus Mainstream-Trainings-Apps
        berücksichtigen die zyklischen Veränderungen weiblicher Physiologie
        meist nicht. CycleSync möchte diese Lücke schliessen.

        **Wichtig:**
        CycleSync ist ein studentisches Forschungsprojekt und **kein
        medizinisches Produkt**. Die Empfehlungen ersetzen keine ärztliche
        oder sportwissenschaftliche Beratung.
        """)

st.markdown("---")
st.caption(
    "🌱 Daten: Synthetische Zyklus- und Workout-Daten (basierend auf Schmalenberger et al. 2021, "
    "McNulty et al. 2020), wissenschaftliche Abstracts von PubMed via NCBI E-utilities. "
    "💻 Code: [github.com/carvacla/cyclesync_KI_project](https://github.com/carvacla/cyclesync_KI_project)"
)
