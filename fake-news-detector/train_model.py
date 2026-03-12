import json
import os
import pickle
import re
from pathlib import Path

import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


DATA_PATH = Path("dataset/news.csv")
MODEL_PATH = Path("model.pkl")
VECTORIZER_PATH = Path("vectorizer.pkl")

# Endee configuration (can be changed with environment variables)
ENDEE_ENABLE = os.getenv("ENDEE_ENABLE", "1") == "1"
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")
ENDEE_TOKEN = os.getenv("ENDEE_TOKEN", "")
ENDEE_INDEX = os.getenv("ENDEE_INDEX", "fake-news")
ENDEE_MAX_INDEX_ROWS = int(os.getenv("ENDEE_MAX_INDEX_ROWS", "2000"))
ENDEE_BATCH_SIZE = int(os.getenv("ENDEE_BATCH_SIZE", "128"))


def clean_text(text: str) -> str:
    """Basic text cleaning for beginner-friendly NLP.

    Steps:
    1. Lowercase
    2. Remove URLs
    3. Remove non-letter characters
    4. Collapse extra spaces
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def index_vectors_in_endee(df: pd.DataFrame, vectorizer: TfidfVectorizer) -> None:
    """Send TF-IDF vectors to Endee for semantic search.

    This is optional and will be skipped if the Endee server is not running.
    """
    if not ENDEE_ENABLE:
        print("Endee indexing disabled (ENDEE_ENABLE=0).")
        return

    headers = {"Content-Type": "application/json"}
    if ENDEE_TOKEN:
        headers["Authorization"] = ENDEE_TOKEN

    # Determine vector dimension
    try:
        dim = len(vectorizer.get_feature_names_out())
    except AttributeError:
        dim = len(vectorizer.vocabulary_)

    # Create index (safe to call even if it already exists)
    create_payload = {
        "index_name": ENDEE_INDEX,
        "dim": dim,
        "space_type": "cosine",
    }

    try:
        resp = requests.post(
            f"{ENDEE_URL}/api/v1/index/create",
            json=create_payload,
            headers=headers,
            timeout=10,
        )
    except requests.exceptions.RequestException as exc:
        print(f"Skipping Endee indexing (server not reachable): {exc}")
        return

    if resp.status_code not in (200, 409):
        print(f"Endee index create failed ({resp.status_code}): {resp.text}")
        return

    # Limit how many rows we index to keep the demo lightweight
    if ENDEE_MAX_INDEX_ROWS > 0:
        df_index = df.head(ENDEE_MAX_INDEX_ROWS).copy()
    else:
        df_index = df.copy()

    # Vectorize content and send in batches
    all_vectors = vectorizer.transform(df_index["content"])
    total = all_vectors.shape[0]

    for start in range(0, total, ENDEE_BATCH_SIZE):
        end = min(start + ENDEE_BATCH_SIZE, total)
        batch_vectors = all_vectors[start:end].toarray()

        payload = []
        for i, vec in enumerate(batch_vectors):
            row = df_index.iloc[start + i]
            meta = {
                "title": str(row.get("title", "")),
                "label": str(row.get("label", "")),
            }

            payload.append(
                {
                    "id": str(start + i),
                    "vector": vec.tolist(),
                    "meta": json.dumps(meta),
                }
            )

        try:
            resp = requests.post(
                f"{ENDEE_URL}/api/v1/index/{ENDEE_INDEX}/vector/insert",
                json=payload,
                headers=headers,
                timeout=30,
            )
        except requests.exceptions.RequestException as exc:
            print(f"Endee insert failed: {exc}")
            return

        if resp.status_code != 200:
            print(f"Endee insert failed ({resp.status_code}): {resp.text}")
            return

    print(f"Indexed {total} vectors into Endee index '{ENDEE_INDEX}'.")


def main() -> None:
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Ensure expected columns exist
    required_cols = {"title", "text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns: {sorted(required_cols)}. "
            f"Found: {list(df.columns)}"
        )

    # Combine title and text for richer context
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).astype(str)

    # Clean text
    df["content"] = df["content"].apply(clean_text)

    # Features and labels
    X = df["content"]
    y = df["label"].astype(str)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert text to numerical features
    # max_features keeps vectors small enough for a beginner-friendly demo
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train classifier
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")

    # Save model and vectorizer
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")

    # Optional: index vectors into Endee for semantic search
    index_vectors_in_endee(df, vectorizer)


if __name__ == "__main__":
    main()
