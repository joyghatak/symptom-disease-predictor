import streamlit as st
import time
from typing import Any, Dict, Optional, Tuple

from src.predictor import predict_disease, get_feature_columns
from src.agent import ai_agent_response

# ====================== HARDCODED API ======================
HARDCODED_GEMINI_API_KEY = "AIzaSyApeJTaKk58bT3vvq149S2H85KiWEB9tT0"

# ====================== FALLBACK SYMPTOMS ======================
FALLBACK_SYMPTOMS = [
    "fever", "cough", "headache", "nausea", "vomiting",
    "fatigue", "chills", "shortness of breath",
    "chest pain", "dizziness", "abdominal pain",
    "diarrhea", "skin rash", "back pain",
    "joint pain", "muscle pain"
]

# ====================== SESSION ======================
def init_session_state():
    defaults = {
        "last_result": None,
        "last_symptoms": [],
        "last_confidence": 0.0,
        "last_layer": "",
        "chat_history": [],
        "agent_enabled": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ====================== LOAD SYMPTOMS ======================
@st.cache_data
def load_symptoms():
    try:
        data = list(get_feature_columns())
        if not data:
            raise ValueError
        return data
    except:
        st.warning("⚠️ Using fallback symptoms (dataset not loaded)")
        return FALLBACK_SYMPTOMS

ALL_SYMPTOMS = load_symptoms()

# ====================== UI ======================
st.set_page_config(page_title="Disease Prediction", layout="wide")

st.title("🏥 Disease Prediction System")
st.caption("AI + ML Based Medical Assistant")

st.warning("⚠️ Educational purpose only. Not real medical advice.")

# ====================== INPUT ======================
selected = st.multiselect(
    "Select Symptoms",
    ALL_SYMPTOMS,
    placeholder="Type symptoms..."
)

if st.button("🚀 Predict"):
    if not selected:
        st.warning("Select at least one symptom")
        st.stop()

    with st.spinner("Analyzing..."):
        result, conf, layer = predict_disease(selected)

    st.session_state.last_result = result
    st.session_state.last_symptoms = selected
    st.session_state.last_confidence = conf
    st.session_state.last_layer = layer

# ====================== RESULT ======================
if st.session_state.last_result:
    st.success(f"Prediction: {st.session_state.last_result}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{st.session_state.last_confidence}%")
    c2.metric("Layer", st.session_state.last_layer)
    c3.metric("Symptoms", len(st.session_state.last_symptoms))

    st.info("💬 Ask questions like: Is it serious? Home remedy?")

# ====================== CHAT ======================
def call_ai(user_msg):
    try:
        text, _ = ai_agent_response(
            st.session_state.last_symptoms,
            st.session_state.last_result,
            user_message=user_msg,
            api_key=HARDCODED_GEMINI_API_KEY
        )
        return text
    except:
        return "⚠️ AI unavailable. Please try later."

# Display chat
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["text"])

user_input = st.chat_input("Ask about your condition...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "text": user_input})

    if not st.session_state.last_result:
        reply = "Please predict disease first."
    else:
        with st.spinner("Thinking..."):
            reply = call_ai(user_input)

    st.session_state.chat_history.append({"role": "assistant", "text": reply})
    st.rerun()

st.caption("© Joy Ghatak • 2026")
