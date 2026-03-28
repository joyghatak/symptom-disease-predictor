import pandas as pd

def load_data():
    path = "data/Final_Augmented_dataset_Diseases_and_Symptoms.csv"
    df = pd.read_csv(path)

    print("✅ Dataset Loaded Successfully")
    print("Shape:", df.shape)
    print("\nColumns:\n", df.columns)
    print("\nFirst 5 rows:\n", df.head())

    return df


if __name__ == "__main__":
    load_data()