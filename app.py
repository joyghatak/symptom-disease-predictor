import streamlit as st
import time
from typing import Any, Dict, Optional, Tuple

from src.predictor import predict_disease, get_feature_columns
from src.agent import ai_agent_response, get_key_usage, MAX_REQUESTS_PER_KEY

# Direct in-code key fallback (requested).
HARDCODED_GEMINI_API_KEY = "AIzaSyApeJTaKk58bT3vvq149S2H85KiWEB9tT0"

try:
    dotenv_module = __import__("dotenv")
    getattr(dotenv_module, "load_dotenv", lambda: None)()
except Exception:
    pass


def init_session_state() -> None:
    """Initialize all session state keys at startup in one place."""
    defaults: Dict[str, Any] = {
        "last_result": None,
        "last_symptoms": [],
        "last_confidence": 0.0,
        "last_layer": "",
        "chat_history": [],
        "agent_enabled": True,
        "explanation": None,
        "settings_gemini_key": "",
        "selected_symptoms_input": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_settings_key() -> None:
    """Clear only the settings key so environment key can take precedence."""
    st.session_state["settings_gemini_key"] = ""


def reset_chat_and_prediction() -> None:
    """Clear prediction outputs, selected symptoms, explanation, and chat history."""
    st.session_state["last_result"] = None
    st.session_state["last_symptoms"] = []
    st.session_state["last_confidence"] = 0.0
    st.session_state["last_layer"] = ""
    st.session_state["explanation"] = None
    st.session_state["chat_history"] = []
    st.session_state["selected_symptoms_input"] = []


def get_api_key_with_source() -> Tuple[str, str]:
    """Resolve API key in order: session state -> hardcoded -> environment -> secrets."""
    session_key = (st.session_state.get("settings_gemini_key") or "").strip()
    if session_key:
        return session_key, "settings"

    hardcoded_key = (HARDCODED_GEMINI_API_KEY or "").strip()
    if hardcoded_key:
        return hardcoded_key, "hardcoded"

    import os

    env_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if env_key:
        return env_key, "environment"

    try:
        secrets_key = (st.secrets.get("GEMINI_API_KEY") or "").strip()
        if secrets_key:
            return secrets_key, "st.secrets"
    except Exception:
        pass

    return "", "none"


def get_api_key() -> str:
    key, _ = get_api_key_with_source()
    return key


def call_agent_with_retry(
    symptoms,
    prediction,
    user_message: Optional[str],
    api_key: str,
    max_attempts: int = 2,
) -> Tuple[str, bool]:
    """Call AI agent with up to max_attempts and graceful fallback handling."""
    last_text = "AI response unavailable."
    last_success = False

    for attempt in range(max_attempts):
        try:
            text, success = ai_agent_response(
                symptoms,
                prediction,
                user_message=user_message,
                api_key=api_key,
            )
        except Exception as e:
            text, success = (f"⚠️ AI call failed unexpectedly: {str(e)[:160]}", False)

        text = (text or "").strip()
        if text:
            last_text = text
        last_success = bool(success)

        if last_success:
            return last_text, True

        if attempt < max_attempts - 1:
            time.sleep(0.6)

    return last_text, last_success

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================== SESSION STATE ======================
init_session_state()

# ====================== SYMPTOM LIST (dynamic) ======================
# Derived from the dataset columns — stays in sync automatically
@st.cache_data
def load_symptom_list():
    try:
        return list(get_feature_columns())
    except Exception:
        return []   # graceful fallback if dataset isn't present yet

ALL_SYMPTOMS = load_symptom_list()

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.big-title {
    font-size: 50px;
    font-weight: 700;
    letter-spacing: -1.8px;
    background: linear-gradient(135deg, #4ade80, #67e8f9, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.disclaimer-box {
    background: #7f1d1d;
    border: 2px solid #ef4444;
    border-radius: 14px;
    padding: 18px 24px;
    margin: 20px 0 30px 0;
    color: #fee2e2;
}

.agent-box {
    background: linear-gradient(135deg, rgba(103,232,249,0.15), rgba(74,222,128,0.15));
    border: 2px solid #67e8f9;
    border-radius: 16px;
    padding: 24px;
    margin: 20px 0;
}

.about-chat-link {
    display: block;
    text-align: center;
    text-decoration: none;
    font-weight: 700;
    border-radius: 12px;
    padding: 10px 12px;
    margin-top: 10px;
    color: #082f49 !important;
    background: linear-gradient(135deg, #bae6fd, #a7f3d0);
    border: 2px solid #38bdf8;
}

.floating-chat-fab {
    position: fixed;
    right: 28px;
    bottom: 24px;
    z-index: 999;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    text-decoration: none;
    font-size: 34px;
    font-weight: 700;
    width: 1.92cm;
    height: 1.92cm;
    color: #f8fafc !important;
    background: #0f172a;
    border: 2px solid #67e8f9;
    border-radius: 50%;
    box-shadow: 0 10px 24px rgba(2, 6, 23, 0.35);
}

.floating-chat-fab:hover {
    border-color: #4ade80;
    transform: translateY(-1px);
}

.floating-chat-toggle-input {
    display: none;
}

.floating-chat-panel {
    position: fixed;
    right: 28px;
    bottom: calc(24px + 2.1cm + 12px);
    z-index: 998;
    width: min(300px, calc(100vw - 32px));
    background: #0f172a;
    border: 2px solid #67e8f9;
    border-radius: 14px;
    box-shadow: 0 14px 30px rgba(2, 6, 23, 0.45);
    padding: 12px;
    color: #e2e8f0;
    display: none;
}

.floating-chat-panel-title {
    margin: 0 0 8px 0;
    font-size: 14px;
    font-weight: 700;
    color: #f8fafc;
}

.floating-chat-panel-text {
    margin: 0 0 10px 0;
    font-size: 12px;
    color: #cbd5e1;
}

.floating-chat-panel-link {
    display: inline-block;
    text-decoration: none;
    font-size: 12px;
    font-weight: 700;
    color: #082f49 !important;
    background: linear-gradient(135deg, #a7f3d0, #67e8f9);
    border-radius: 10px;
    border: 1px solid #38bdf8;
    padding: 8px 10px;
}

.floating-chat-toggle-input:checked ~ .floating-chat-panel {
    display: block;
}

@media (max-width: 768px) {
    .floating-chat-fab {
        right: 16px;
        bottom: 16px;
    }

    .floating-chat-panel {
        right: 16px;
        bottom: calc(16px + 2.1cm + 10px);
    }
}
</style>
""", unsafe_allow_html=True)

# ====================== DISCLAIMER ======================
st.markdown("""
<div class="disclaimer-box">
    <strong>⚠️ MEDICAL DISCLAIMER</strong><br>
    This is an educational demonstration only. Not a real medical tool.
    Always consult a qualified doctor.
</div>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    tab_about, tab_settings = st.tabs(["📌 About", "⚙️ Settings"])

    with tab_about:
        st.markdown("### 🏥 Disease Prediction System")
        st.write("""
**This app predicts possible health conditions from symptoms.**

**3-Layer System:**
- 🔴 Red Layer → Emergency detection
- 🟢 Green Layer → Common known patterns
- 🤖 ML Layer → BernoulliNB Machine Learning

**AI Medical Agent** explains the result in simple language.
        """)
        st.button(
            "Clear chat + reset prediction",
            use_container_width=True,
            on_click=reset_chat_and_prediction,
        )
        st.markdown(
            '<a class="about-chat-link" href="#ai-medical-chat">🩺 AI Medical Agent Chat</a>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption("👨‍💻 Joy Ghatak • Educational Demo 2026")

    with tab_settings:
        st.markdown("### Agent Settings")
        enabled = st.toggle("Enable AI Medical Agent", value=st.session_state.agent_enabled)
        st.session_state.agent_enabled = enabled

        active_key, key_source = get_api_key_with_source()
        if active_key:
            used = get_key_usage(active_key)
            remaining = max(0, MAX_REQUESTS_PER_KEY - used)
            st.caption(f"AI key usage: {used}/{MAX_REQUESTS_PER_KEY} • Remaining: {remaining}")
            st.caption(f"Active key source: {key_source}")
        else:
            st.caption("No Gemini API key configured. AI responses will use local fallback mode.")

# ====================== HEADER ======================
st.markdown('<div class="big-title">🏥 Disease Prediction System</div>', unsafe_allow_html=True)
st.caption("3-Layer Intelligence + AI Medical Agent")
st.divider()
st.markdown(
    """
    <div class="floating-chat-toggle">
        <input class="floating-chat-toggle-input" type="checkbox" id="floating-chat-toggle" />
        <label class="floating-chat-fab" for="floating-chat-toggle">🤖</label>
        <div class="floating-chat-panel">
            <p class="floating-chat-panel-title">AI Medical Agent Chat</p>
            <p class="floating-chat-panel-text">Tap below to open the full chat section.</p>
            <a class="floating-chat-panel-link" href="#ai-medical-chat">Open Chat</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ====================== SYMPTOM INPUT ======================
if not ALL_SYMPTOMS:
    st.warning("⚠️ Symptom list could not be loaded. Make sure your dataset CSV is in the `data/` folder.")

selected_symptoms = st.multiselect(
    "🧾 Select your symptoms",
    options=ALL_SYMPTOMS,
    placeholder="Start typing symptoms...",
    key="selected_symptoms_input",
)

if st.button("🚀 Predict Disease", type="primary", use_container_width=True):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
        st.stop()

    try:
        with st.spinner("Analyzing with 3-Layer System..."):
            progress = st.progress(0)
            time.sleep(0.3)
            progress.progress(50)

            # predict_disease always returns (result, confidence, layer)
            result, confidence, layer = predict_disease(selected_symptoms)

            progress.progress(100)
    except Exception as e:
        st.error(
            "Prediction failed. Ensure dataset and model files are available. "
            "If missing, run src/data_loader.py and src/model.py first."
        )
        st.caption(f"Technical detail: {str(e)[:220]}")
        st.stop()

    # Store in session state
    st.session_state.last_result     = result
    st.session_state.last_symptoms   = selected_symptoms
    st.session_state.last_confidence = confidence
    st.session_state.last_layer      = layer
    st.session_state.explanation     = None   # reset cached explanation

# ====================== PREDICTION RESULT ======================
if st.session_state.last_result:
    result     = st.session_state.last_result
    confidence = st.session_state.last_confidence
    layer      = st.session_state.last_layer

    st.success(f"**Predicted Condition:** {result}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{confidence:.1f}%")
    c2.metric("Layer",      layer)              # shows the actual layer that fired
    c3.metric("Symptoms",   len(st.session_state.last_symptoms))

    if st.session_state.agent_enabled:
        if st.button("🤖 Get AI Medical Explanation", type="secondary", use_container_width=True):
            # Only call the API if we don't already have the explanation cached
            if st.session_state.explanation is None:
                with st.spinner("AI Agent is thinking..."):
                    response_text, _ = call_agent_with_retry(
                        st.session_state.last_symptoms,
                        result,
                        user_message=None,
                        api_key=get_api_key(),
                    )
                    st.session_state.explanation = response_text or "Offline response unavailable."

        if st.session_state.explanation is not None:
            # Use plain markdown — no unsafe_allow_html to avoid XSS from API output
            with st.container():
                st.markdown("---")
                st.markdown(st.session_state.explanation or "No explanation was generated.")

        st.info("💬 You can now ask questions like:\n- Is this serious?\n- What should I do?\n- Any home remedies?")

    st.divider()

# ====================== AI CHAT SECTION ======================
if st.session_state.agent_enabled:
    st.markdown('<div id="ai-medical-chat"></div>', unsafe_allow_html=True)
    st.subheader("💬 AI Medical Agent Chat")

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.info(f"**You:** {msg['text']}")
        else:
            st.markdown(msg["text"])

    user_input = st.chat_input(
        "Ask anything about your symptoms...",
        disabled=not bool(st.session_state.last_result),
    )
    if user_input:
        st.session_state.chat_history.append({"role": "user", "text": user_input})

        if not st.session_state.last_result or not st.session_state.last_symptoms:
            st.session_state.chat_history.append(
                {
                    "role": "bot",
                    "text": (
                        "Please select symptoms and click **Predict Disease** first, "
                        "then ask follow-up questions in chat."
                    ),
                }
            )
            st.rerun()

        with st.spinner("Agent is replying..."):
            reply, success = call_agent_with_retry(
                symptoms=st.session_state.last_symptoms,
                prediction=st.session_state.last_result,
                user_message=user_input,
                api_key=get_api_key(),
            )

        if not (reply or "").strip():
            reply = "Offline mode: no AI response was generated."

        if not success:
            reply = "⚠️ AI is in fallback mode right now.\n\n" + reply

        st.session_state.chat_history.append({"role": "bot", "text": reply})
        st.rerun()

st.caption("Disease Prediction System • Educational Demo • Joy Ghatak • March 2026")