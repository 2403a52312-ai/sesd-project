from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "fake_job_postings.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "fake_job_model.joblib"

TEXT_COLUMNS = [
    "title", "company_profile", "description", "requirements",
    "benefits", "employment_type", "required_experience",
    "required_education", "industry", "function", "location"
]

def combine_text(row):
    parts = []
    for col in TEXT_COLUMNS:
        if col in row.index and pd.notna(row[col]):
            value = str(row[col]).strip()
            if value and value.lower() != "nan":
                parts.append(f"{col}: {value}")
    return " ".join(parts)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download the Kaggle CSV and save it as data/fake_job_postings.csv"
        )

    df = pd.read_csv(DATA_PATH)
    if "fraudulent" not in df.columns:
        raise ValueError("Dataset must contain a 'fraudulent' target column.")

    df = df.dropna(subset=["fraudulent"]).copy()
    df["fraudulent"] = pd.to_numeric(df["fraudulent"], errors="coerce")
    df = df.dropna(subset=["fraudulent"])
    df["fraudulent"] = df["fraudulent"].astype(int)

    if not set(df["fraudulent"].unique()).issubset({0, 1}):
        raise ValueError("The 'fraudulent' column must contain 0 and 1 labels.")

    X = df.apply(combine_text, axis=1)
    y = df["fraudulent"]

    if y.nunique() != 2:
        raise ValueError("The dataset must contain both genuine (0) and fake (1) postings.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
