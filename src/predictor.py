from functools import lru_cache
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
from joblib import load
from pathlib import Path

# ====================== CONFIG ======================
MODEL_PATH = "model.joblib"

# Google Drive CSV
FILE_ID = "1-vtq4LIJI2JdKM6dWFhdUM9ub8Zcll10"
DATA_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"


# ====================== HELPERS ======================
def _clamp_confidence(value: float) -> float:
    return round(max(0.0, min(100.0, float(value))), 1)


# ====================== DATA LOADING ======================
@lru_cache(maxsize=1)
def load_dataset():
    """Load dataset from Google Drive (cached for performance)"""
    try:
        df = pd.read_csv(DATA_URL)
        return df
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load dataset from Google Drive: {e}")


@lru_cache(maxsize=1)
def get_feature_columns() -> Tuple[str, ...]:
    """Extract feature columns dynamically from dataset"""
    df = load_dataset()

    if "diseases" not in df.columns:
        raise ValueError("Dataset missing 'diseases' column.")

    features = tuple(col for col in df.columns if col != "diseases")

    if not features:
        raise ValueError("No feature columns found.")

    return features


@lru_cache(maxsize=1)
def get_feature_index() -> Dict[str, int]:
    columns = get_feature_columns()
    return {col: idx for idx, col in enumerate(columns)}


@lru_cache(maxsize=1)
def get_model():
    model_file = Path(MODEL_PATH)

    if not model_file.exists():
        raise FileNotFoundError(
            f"❌ model.joblib not found. Run model.py first."
        )

    try:
        return load(model_file)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


# ====================== PREDICTION LAYERS ======================

def red_layer(symptoms: Iterable[str]) -> Optional[Tuple[str, float, str]]:
    """Emergency detection layer"""
    critical = [
        "chest pain", "shortness of breath",
        "difficulty breathing", "severe pain",
        "sharp chest pain",
    ]

    lower = [s.lower() for s in symptoms]

    for symptom in lower:
        for keyword in critical:
            if keyword in symptom:
                return (
                    "⚠️ Emergency: Seek immediate medical attention!",
                    100.0,
                    "🔴 Emergency Layer"
                )
    return None


def green_layer(symptoms: Iterable[str]) -> Optional[Tuple[str, float, str]]:
    """Rule-based fast detection"""
    if "fever" in symptoms and "cough" in symptoms:
        return "Common Flu", 95.0, "🟢 Rule Layer"
    return None


def ml_layer(symptoms: Iterable[str]) -> Tuple[str, float, str]:
    """ML prediction"""
    model = get_model()
    columns = list(get_feature_columns())
    col_index = get_feature_index()

    input_data = [0] * len(columns)
    matched = 0

    for symptom in symptoms:
        if symptom in col_index:
            input_data[col_index[symptom]] = 1
            matched += 1

    if matched == 0:
        return (
            "No matching known symptoms found.",
            0.0,
            "🟡 Validation Layer",
        )

    input_df = pd.DataFrame([input_data], columns=columns)

    prediction = model.predict(input_df)[0]

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        confidence = _clamp_confidence(max(probs) * 100)

    return prediction, confidence, "🤖 ML Model"


# ====================== MAIN FUNCTION ======================
def predict_disease(symptoms: Iterable[str]) -> Tuple[str, float, str]:

    if not symptoms:
        return "Please select at least one symptom.", 0.0, "🟡 Validation Layer"

    normalized = [str(s).lower().strip() for s in symptoms if str(s).strip()]

    if not normalized:
        return "Please select valid symptoms.", 0.0, "🟡 Validation Layer"

    # 🔴 Emergency Layer
    red = red_layer(normalized)
    if red:
        return red[0], _clamp_confidence(red[1]), red[2]

    # 🟢 Rule Layer
    green = green_layer(normalized)
    if green:
        return green[0], _clamp_confidence(green[1]), green[2]

    # 🤖 ML Layer
    prediction, confidence, layer = ml_layer(normalized)
    return prediction, _clamp_confidence(confidence), layer


# ====================== TEST ======================
if __name__ == "__main__":
    test_cases = [
        ["fever", "cough"],
        ["chest pain", "shortness of breath"],
        ["headache", "nausea", "dizziness"],
    ]

    for symptoms in test_cases:
        result, confidence, layer = predict_disease(symptoms)
        print(f"Symptoms : {symptoms}")
        print(f"Result   : {result} ({confidence}%) via {layer}\n")
