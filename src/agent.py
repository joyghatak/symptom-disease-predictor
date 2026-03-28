import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Iterable, Optional, Tuple

try:
    from google import genai
except Exception:
    genai = None


MAX_REQUESTS_PER_KEY = 20
DEFAULT_MODEL = "gemini-2.5-flash"
VALID_MODELS = {"gemini-2.0-flash", "gemini-2.5-flash"}
REQUEST_TIMEOUT_SECONDS = 12
_KEY_USAGE = {}


def get_api_key() -> str:
    """Resolve Gemini API key from environment only.

    Streamlit-specific lookups are intentionally handled in app.py to avoid
    circular imports in src modules.
    """
    return os.getenv("GEMINI_API_KEY", "").strip()


def _normalize_key(api_key: Optional[str]) -> str:
    return (api_key or "").strip()


def get_key_usage(api_key: Optional[str]) -> int:
    """Return usage count for the provided API key in current process."""
    key = _normalize_key(api_key)
    if not key:
        return 0
    return int(_KEY_USAGE.get(key, 0))


def _remaining_requests(api_key: Optional[str]) -> int:
    return max(0, MAX_REQUESTS_PER_KEY - get_key_usage(api_key))


def _register_request(api_key: Optional[str]) -> bool:
    key = _normalize_key(api_key)
    if not key:
        return False
    _KEY_USAGE[key] = _KEY_USAGE.get(key, 0) + 1
    return True


def _is_emergency_context(symptoms: Iterable[str], prediction: Optional[str]) -> bool:
    prediction_text = (prediction or "").lower()
    symptom_texts = [str(s).lower().strip() for s in (symptoms or [])]

    emergency_keywords = [
        "emergency",
        "chest pain",
        "shortness of breath",
        "difficulty breathing",
        "severe pain",
        "sharp chest pain",
        "fainting",
        "unconscious",
    ]

    critical_prediction_keywords = [
        "hemorrhage",
        "intracranial",
        "stroke",
        "myocardial infarction",
        "heart attack",
        "sepsis",
        "pulmonary embol",
        "aneurysm",
        "meningitis",
        "respiratory failure",
    ]

    if any(keyword in prediction_text for keyword in emergency_keywords):
        return True

    if any(keyword in prediction_text for keyword in critical_prediction_keywords):
        return True

    return any(keyword in symptom for symptom in symptom_texts for keyword in emergency_keywords)


def _fallback_guidance(symptoms: Iterable[str], prediction: Optional[str]) -> str:
    if _is_emergency_context(symptoms, prediction):
        return (
            "Urgent guidance: potential emergency indicators detected. "
            "Call local emergency services now or go to the nearest emergency department immediately. "
            "Do not rely on this app for urgent decisions."
        )

    return (
        "General advice: rest, hydrate, and monitor symptoms. "
        "If symptoms worsen or feel severe, consult a qualified doctor."
    )


def _offline_fallback(symptoms: Iterable[str], prediction: Optional[str], user_message: Optional[str] = None) -> str:
    symptom_str = ", ".join(symptoms) if symptoms else "none selected"
    prediction_str = prediction or "not available"

    if user_message:
        return (
            "🤖 **Medical Assistant (Offline Fallback)**\n\n"
            "I could not reach the AI service right now, but based on your context:\n"
            f"- Predicted condition: **{prediction_str}**\n"
            f"- Symptoms: **{symptom_str}**\n\n"
            f"You asked: **{user_message}**\n\n"
            + _fallback_guidance(symptoms, prediction)
            + "\n\n"
            "**DISCLAIMER**: Educational support only, not medical advice."
        )

    return (
        "🤖 **Medical Assistant (Offline Fallback)**\n\n"
        "I could not reach the AI service right now.\n"
        f"- Predicted condition: **{prediction_str}**\n"
        f"- Symptoms: **{symptom_str}**\n\n"
        + _fallback_guidance(symptoms, prediction)
        + "\n\n"
        "**DISCLAIMER**: Educational support only, not medical advice."
    )


def _resolve_model_name(requested_model: Optional[str]) -> Tuple[str, Optional[str]]:
    model = (requested_model or DEFAULT_MODEL).strip()
    if model in VALID_MODELS:
        return model, None
    return DEFAULT_MODEL, (
        f"Model '{model}' is not supported in this app; auto-switched to '{DEFAULT_MODEL}'."
    )


def get_gemini_client(api_key: Optional[str] = None):
    """Get Gemini client using the provided key or environment fallback."""
    key = _normalize_key(api_key) or get_api_key()
    if not key:
        return None
    if genai is None:
        return None
    try:
        return genai.Client(api_key=key)
    except Exception:
        return None


def _generate_with_timeout(client, model: str, prompt: str, timeout_seconds: int):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(client.models.generate_content, model=model, contents=prompt)
        return future.result(timeout=timeout_seconds)


def ai_agent_response(
    symptoms: Iterable[str],
    prediction: Optional[str],
    user_message: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Tuple[str, bool]:
    """Return AI response text and success status.

    Returns:
        tuple[str, bool]: (response_text_or_error, success)
    """
    resolved_key = _normalize_key(api_key) or get_api_key()

    if not resolved_key:
        return (
            "⚠️ No Gemini API key found. Add it in Streamlit settings, st.secrets, or environment variable GEMINI_API_KEY.\n\n"
            + _offline_fallback(symptoms, prediction, user_message),
            False,
        )

    if _remaining_requests(resolved_key) <= 0:
        return (
            "⚠️ **API key request limit reached**\n\n"
            f"This Gemini API key already used **{MAX_REQUESTS_PER_KEY}/{MAX_REQUESTS_PER_KEY}** requests in this app session. "
            "Please add a different key or restart the app process.\n\n"
            + _offline_fallback(symptoms, prediction, user_message),
            False,
        )

    client = get_gemini_client(api_key=resolved_key)
    if not client:
        return (
            "⚠️ Could not create Gemini client. The key may be invalid or the SDK is unavailable.\n\n"
            + _offline_fallback(symptoms, prediction, user_message),
            False,
        )

    resolved_model, model_warning = _resolve_model_name(model_name)
    symptom_str = ", ".join(symptoms) if symptoms else "None"

    if user_message:
        prompt = f"""
You are a helpful educational medical assistant.

Context:
- Reported symptoms: {symptom_str}
- Preliminary prediction: {prediction}

The user asked: {user_message}

Answer clearly and professionally.
Keep it concise. End with a brief disclaimer that this is not real medical advice.
"""
    else:
        prompt = f"""
You are a helpful educational medical assistant.

Symptoms: {symptom_str}
Preliminary prediction: {prediction}

Give a short, clear response including:
- Simple explanation
- Home precautions
- When to see a doctor

End with a strong disclaimer that this is not real medical advice.
"""

    try:
        _register_request(resolved_key)
        response = _generate_with_timeout(
            client=client,
            model=resolved_model,
            prompt=prompt,
            timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        )
        text = (getattr(response, "text", "") or "").strip()

        if not text:
            fallback = _offline_fallback(symptoms, prediction, user_message)
            warning = f"⚠️ Empty response from model '{resolved_model}'."
            if model_warning:
                warning = model_warning + "\n" + warning
            return (warning + "\n\n" + fallback, False)

        if model_warning:
            return (f"⚠️ {model_warning}\n\n{text}", True)

        return text, True

    except FutureTimeoutError:
        prefix = f"⚠️ {model_warning}\n\n" if model_warning else ""
        return (
            prefix
            + f"⚠️ AI request timed out after {REQUEST_TIMEOUT_SECONDS}s using model '{resolved_model}'.\n\n"
            + _offline_fallback(symptoms, prediction, user_message),
            False,
        )
    except Exception as e:
        error_text = str(e)
        normalized = error_text.lower()

        if (
            "resource_exhausted" in normalized
            or "quota" in normalized
            or "429" in normalized
            or "rate limit" in normalized
        ):
            prefix = f"⚠️ {model_warning}\n\n" if model_warning else ""
            return (
                prefix
                + "⚠️ **AI quota reached**\n\n"
                "The Gemini API limit is temporarily exhausted for this key. "
                "Showing local fallback guidance instead.\n\n"
                + _offline_fallback(symptoms, prediction, user_message),
                False,
            )

        prefix = f"⚠️ {model_warning}\n\n" if model_warning else ""
        return (
            prefix
            + f"⚠️ AI request failed using model '{resolved_model}'.\n\n"
            + _offline_fallback(symptoms, prediction, user_message),
            False,
        )