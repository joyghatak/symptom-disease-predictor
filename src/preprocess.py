import pandas as pd
import pickle

# Load dataset (for column reference)
df = pd.read_csv("data/Final_Augmented_dataset_Diseases_and_Symptoms.csv")
X = df.drop("diseases", axis=1)

# Load saved model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# 🟢 GREEN LAYER
def green_layer(symptoms):
    if "fever" in symptoms and "cough" in symptoms:
        return "Common Flu"
    return None


# 🔴 RED LAYER
def red_layer(symptoms):
    if "chest pain" in symptoms or "shortness of breath" in symptoms:
        return "⚠️ Emergency: Seek immediate medical attention!"
    return None


# 🤖 ML LAYER
def ml_layer(symptoms):
    # Create input dataframe (fix warning + professional)
    input_data = pd.DataFrame([0]*len(X.columns)).T
    input_data.columns = X.columns

    for symptom in symptoms:
        if symptom in input_data.columns:
            input_data[symptom] = 1

    prediction = model.predict(input_data)
    return prediction[0]


# 🎯 MAIN FUNCTION
def predict_disease(symptoms):
    red = red_layer(symptoms)
    if red:
        return red

    green = green_layer(symptoms)
    if green:
        return green

    return ml_layer(symptoms)


# 🔍 TEST
if __name__ == "__main__":
    test = ["fever", "cough"]
    print("Prediction:", predict_disease(test))