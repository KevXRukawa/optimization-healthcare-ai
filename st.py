import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Smart Triage",
    page_icon="🚑",
    layout="centered"
)

MODEL_PATH = "./finished_triage_model"

# Maximum sequence length supported by the underlying BERT model.
# Inputs longer than this must be truncated or inference will crash.
MAX_TOKENS = 512

# Predictions below this confidence are held for human review
# before dispatch is allowed.
CONFIDENCE_THRESHOLD = 0.70

DISCLAIMER = (
    "This tool is a clinical decision-support aid only. It does not replace "
    "assessment by qualified medical personnel. Always confirm the risk level "
    "with a trained clinician before acting on any recommendation."
)


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )


try:
    classifier = load_model()
except Exception as error:
    st.error("❌ Failed to load the triage model.")
    st.caption(
        f"Check that '{MODEL_PATH}' exists and contains all model files. "
        f"Details: {error}"
    )
    st.stop()

HIGH = "HIGH_RISK"
MEDIUM = "MEDIUM_RISK"
LOW = "LOW_RISK"

RISK_LEVELS = [HIGH, MEDIUM, LOW]


def classify_symptoms(text):
    """Classify symptom text into a risk level.

    Returns a (risk, confidence, was_truncated) tuple. Text longer than
    MAX_TOKENS is truncated so that inference cannot crash on long input.
    """
    token_count = len(classifier.tokenizer(text)["input_ids"])
    was_truncated = token_count > MAX_TOKENS

    result = classifier(
        text,
        truncation=True,
        max_length=MAX_TOKENS
    )[0]

    return result["label"], result["score"], was_truncated


class Hospital:
    def __init__(self, name, hospital_class, capacity, ugd_capacity=0, location=(0, 0)):
        self.name = name
        self.hospital_class = hospital_class
        self.capacity = capacity
        self.ugd_capacity = ugd_capacity
        self.current = 0
        self.ugd_current = 0
        self.location = location

    def can_handle(self, risk):
        if self.hospital_class == 1 and risk == HIGH:
            return (
                self.ugd_current < self.ugd_capacity
                or self.current < self.capacity
            )
        return self.current < self.capacity

    def admit(self, risk):
        if (
            self.hospital_class == 1
            and risk == HIGH
            and self.ugd_current < self.ugd_capacity
        ):
            self.ugd_current += 1
        else:
            self.current += 1


if "hospitals" not in st.session_state:
    st.session_state.hospitals = [
        Hospital(
            "RSUP Cipto Mangunkusumo",
            1,
            2,
            ugd_capacity=1,
            location=(-6.197, 106.846)
        ),
        Hospital(
            "RS Fatmawati",
            2,
            2,
            location=(-6.293, 106.797)
        ),
        Hospital(
            "RSUD Pasar Minggu",
            3,
            3,
            location=(-6.284, 106.842)
        ),
    ]

if "patient" not in st.session_state:
    st.session_state.patient = None

if "recommendation" not in st.session_state:
    st.session_state.recommendation = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "needs_review" not in st.session_state:
    st.session_state.needs_review = False

if "was_truncated" not in st.session_state:
    st.session_state.was_truncated = False

if "dispatched" not in st.session_state:
    st.session_state.dispatched = False


def reset_patient():
    st.session_state.patient = None
    st.session_state.recommendation = None
    st.session_state.confidence = None
    st.session_state.needs_review = False
    st.session_state.was_truncated = False
    st.session_state.dispatched = False


def hospital_order(risk, hospitals):
    if risk == HIGH:
        return sorted(hospitals, key=lambda h: h.hospital_class)

    elif risk == MEDIUM:
        return sorted(
            hospitals,
            key=lambda h: abs(h.hospital_class - 2)
        )

    else:
        return sorted(
            hospitals,
            key=lambda h: -h.hospital_class
        )


def find_hospital(risk):
    """Return the first hospital able to admit a patient at this risk level."""
    for hospital in hospital_order(risk, st.session_state.hospitals):
        if hospital.can_handle(risk):
            return hospital
    return None


st.markdown(
    """
    <h1 style='text-align: center;'>🚑 Smart Triage & Hospital Dispatch</h1>
    <p style='text-align: center; color: gray;'>
        AI-assisted emergency risk assessment and hospital allocation
    </p>
    """,
    unsafe_allow_html=True
)

st.info(f"⚕️ {DISCLAIMER}", icon="⚕️")

st.divider()

# INPUT PASIENT

if st.session_state.patient is None:

    st.subheader("🧑‍⚕️ Patient Information")

    with st.container(border=True):

        name = st.text_input("Patient Name")

        symptoms = st.text_area(
            "Describe Symptoms",
            placeholder=(
                "e.g. chest pain, shortness of breath, unconscious...\n"
                "Bahasa Indonesia juga didukung, mis. nyeri dada, sesak napas"
            )
        )

        st.markdown("")

        if st.button("🔍 Assess Patient", use_container_width=True):

            if name.strip() and symptoms.strip():

                try:
                    with st.spinner("Analysing symptoms..."):
                        risk, confidence, was_truncated = classify_symptoms(
                            symptoms.strip()
                        )

                except Exception as error:
                    st.error(
                        "❌ Assessment failed. The patient was not recorded. "
                        "Please try again or escalate manually."
                    )
                    st.caption(f"Details: {error}")

                else:
                    st.session_state.patient = {
                        "name": name.strip(),
                        "risk": risk
                    }

                    st.session_state.confidence = confidence
                    st.session_state.was_truncated = was_truncated
                    st.session_state.needs_review = (
                        confidence < CONFIDENCE_THRESHOLD
                    )
                    st.session_state.recommendation = find_hospital(risk)

                    st.rerun()

            else:
                st.warning("Please complete all fields")

# TRIAGE RESULT

elif not st.session_state.dispatched:

    p = st.session_state.patient
    h = st.session_state.recommendation

    st.subheader("📋 Triage Result")

    with st.container(border=True):

        st.caption(f"Patient: **{p['name']}**")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Risk Level",
                value=p["risk"]
            )

        with col2:
            st.metric(
                label="Confidence",
                value=f"{st.session_state.confidence:.2%}"
            )

        st.divider()

        if st.session_state.was_truncated:
            st.warning(
                "✂️ The symptom description exceeded the model limit of "
                f"{MAX_TOKENS} tokens and was truncated. Only the earlier "
                "part of the text was assessed. Review the full description "
                "manually before dispatch."
            )

        if p["risk"] == HIGH:
            st.error("⚠️ Critical condition detected")

        elif p["risk"] == MEDIUM:
            st.warning("⚠️ Moderate risk detected")

        else:
            st.success("✅ Low risk condition detected")

        if h is None:
            st.error(
                "❌ No available hospital can currently handle this patient."
            )
        else:
            st.success(
                f"🏥 Recommended Hospital: "
                f"{h.name} (Class {h.hospital_class})"
            )

    # HUMAN REVIEW GATE FOR LOW-CONFIDENCE PREDICTIONS

    if st.session_state.needs_review:

        st.error(
            "🧑‍⚕️ **Human review required.** Model confidence is below "
            f"{CONFIDENCE_THRESHOLD:.0%}, so this prediction is not reliable "
            "enough to act on unchecked. Confirm or correct the risk level "
            "before dispatch."
        )

        with st.container(border=True):

            reviewed_risk = st.selectbox(
                "Confirmed risk level (clinician decision)",
                RISK_LEVELS,
                index=RISK_LEVELS.index(p["risk"])
            )

            if st.button(
                "✅ Confirm Risk Level",
                use_container_width=True
            ):
                st.session_state.patient["risk"] = reviewed_risk
                st.session_state.recommendation = find_hospital(reviewed_risk)
                st.session_state.needs_review = False
                st.rerun()

    st.markdown("### Confirm Dispatch")

    col1, col2 = st.columns(2)

    with col1:

        if h is not None:

            if st.button(
                "🚑 Dispatch Patient",
                use_container_width=True,
                disabled=st.session_state.needs_review,
                help=(
                    "Confirm the risk level first"
                    if st.session_state.needs_review
                    else None
                )
            ):
                # Re-check capacity in case it changed since assessment
                if not h.can_handle(p["risk"]):
                    st.error(
                        "❌ That hospital is no longer available. "
                        "Please reassess the patient."
                    )
                else:
                    h.admit(p["risk"])

                    st.session_state.dispatched = True
                    st.rerun()

    with col2:

        if st.button(
            "↩️ Reassess",
            use_container_width=True
        ):
            reset_patient()
            st.rerun()

# DISPATCH SUCCESS

else:

    h = st.session_state.recommendation

    st.success("🚑 Patient Successfully Dispatched")

    st.markdown(
        f"**🏥 Hospital:** {h.name}"
    )

    with st.container(border=True):

        m = folium.Map(
            location=h.location,
            zoom_start=13
        )

        folium.Marker(
            h.location,
            popup=h.name
        ).add_to(m)

        st_folium(
            m,
            height=300
        )

    if st.button(
        "➕ New Patient",
        use_container_width=True
    ):
        reset_patient()
        st.rerun()


# HOSPITAL STATUS

with st.expander("🏨 Hospital Capacity Status"):

    for h in st.session_state.hospitals:

        status = (
            f"**{h.name}** (Class {h.hospital_class})\n"
            f"- Normal: {h.current}/{h.capacity}"
        )

        if h.ugd_capacity > 0:
            status += f"\n- UGD: {h.ugd_current}/{h.ugd_capacity}"

        st.write(status)

    st.caption(
        "⚠️ Capacity is tracked per browser session for simulation purposes "
        "and resets when the page reloads. It is not shared between users."
    )
