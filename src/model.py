import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from joblib import dump

# ---------------------------------------------------------------------------
# Paths — relative so this works locally, in Docker, and on Streamlit Cloud
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "Final_Augmented_dataset_Diseases_and_Symptoms.csv")
MODEL_PATH = "model.joblib"


def train_and_save_model(nrows=50000):
    """Train a BernoulliNB model and save it as model.joblib."""

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'.\n"
            "Make sure the CSV is in the data/ folder."
        )

    print(f"📂 Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, nrows=nrows)

    X = df.drop("diseases", axis=1)
    y = df["diseases"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   Training rows : {len(X_train):,}")
    print(f"   Test rows     : {len(X_test):,}")
    print(f"   Classes       : {y.nunique()}")
    print("\nTraining model... ⏳")

    model = BernoulliNB(alpha=0.1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("✅ Model trained successfully")
    print(f"🎯 Accuracy: {acc * 100:.2f}%")

    dump(model, MODEL_PATH)
    print(f"💾 Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()