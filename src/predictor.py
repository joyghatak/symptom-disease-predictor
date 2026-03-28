from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
from joblib import load

# Paths
DATA_PATH = "data/Final_Augmented_dataset_Diseases_and_Symptoms.csv"
MODEL_PATH = "model.joblib"


def _clamp_confidence(value: float) -> float:
    return round(max(0.0, min(100.0, float(value))), 1)


@lru_cache(maxsize=1)
def get_feature_columns() -> Tuple[str, ...]:
    """Load feature columns from dataset with existence checks."""
    data_file = Path(DATA_PATH)
    if not data_file.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'. Run data_loader.py first and ensure the CSV exists."
        )

    try:
        df = pd.read_csv(data_file, nrows=1)
    except Exception as e:
        raise RuntimeError(f"Failed to read dataset header: {e}") from e

    if "diseases" not in df.columns:
        raise ValueError(
            "Dataset is missing required 'diseases' column. "
            "Please verify data/Final_Augmented_dataset_Diseases_and_Symptoms.csv format."
        )

    features = tuple(col for col in df.columns if col != "diseases")
    if not features:
        raise ValueError("No feature columns found in dataset.")
    return features


@lru_cache(maxsize=1)
def get_feature_index() -> Dict[str, int]:
    """Create a symptom-to-column index mapping."""
    columns = get_feature_columns()
    return {col: idx for idx, col in enumerate(columns)}


@lru_cache(maxsize=1)
def get_model():
    """Load the trained model with validation and clear errors."""
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        raise FileNotFoundError(
            f"❌ model.joblib not found at '{MODEL_PATH}'. "
            "Run model.py to train and save the model first."
        )
    try:
        return load(model_file)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{MODEL_PATH}': {e}") from e


# ---------------------------------------------------------------------------
# Prediction layers
# ---------------------------------------------------------------------------

def red_layer(symptoms: Iterable[str]) -> Optional[Tuple[str, float, str]]:
    """Emergency detection layer returning prediction tuple when matched."""
    critical = [
        "chest pain", "shortness of breath",
        "difficulty breathing", "severe pain",
        "sharp chest pain",
    ]
    lower_symptoms = [s.lower() for s in symptoms]
    for symptom in lower_symptoms:
        for keyword in critical:
            if keyword in symptom:
                return "⚠️ Emergency: Seek immediate medical attention!", 100.0, "🔴 Emergency Layer"
    return None


def green_layer(symptoms: Iterable[str]) -> Optional[Tuple[str, float, str]]:
    """Fast-path rule layer returning prediction tuple when matched."""
    if "fever" in symptoms and "cough" in symptoms:
        return "Common Flu", 95.0, "🟢 Rule Layer"
    return None


def ml_layer(symptoms: Iterable[str]) -> Tuple[str, float, str]:
    """Machine learning prediction layer with robust confidence handling."""
    model = get_model()
    columns = list(get_feature_columns())
    col_index = get_feature_index()

    if not columns:
        raise ValueError("No feature columns available for prediction.")

    input_data = [0] * len(columns)
    matched = 0
    for symptom in symptoms:
        if symptom in col_index:
            input_data[col_index[symptom]] = 1
            matched += 1

    if matched == 0:
        return (
            "No matching known symptoms found in dataset columns. Please choose listed symptoms.",
            0.0,
            "🟡 Validation Layer",
        )

    input_frame = pd.DataFrame([input_data], columns=columns)

    prediction = model.predict(input_frame)[0]

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_frame)[0]
        confidence = _clamp_confidence(max(probabilities) * 100)

    return prediction, confidence, "🤖 ML Model"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def predict_disease(symptoms: Iterable[str]) -> Tuple[str, float, str]:
    """Run the 3-layer prediction pipeline.

    Returns:
        tuple: (prediction: str, confidence: float, layer: str)
               Always returns all three values regardless of which layer fires.
    """
    if not symptoms:
        return "Please select at least one symptom.", 0.0, "🟡 Validation Layer"

    # Normalize once here so all layers see the same format.
    normalized = [str(s).lower().strip() for s in symptoms if str(s).strip()]
    if not normalized:
        return "Please select at least one valid symptom.", 0.0, "🟡 Validation Layer"

    # 🔴 Red layer — emergency
    red = red_layer(normalized)
    if red:
        return red[0], _clamp_confidence(red[1]), red[2]

    # 🟢 Green layer — common patterns
    green = green_layer(normalized)
    if green:
        return green[0], _clamp_confidence(green[1]), green[2]

    # 🤖 ML layer
    prediction, confidence, layer = ml_layer(normalized)
    return prediction, _clamp_confidence(confidence), layer


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
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