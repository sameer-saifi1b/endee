import json
import os
import pickle
from pathlib import Path

import msgpack
import requests


MODEL_PATH = Path("model.pkl")
VECTORIZER_PATH = Path("vectorizer.pkl")

# Endee configuration (can be changed with environment variables)
ENDEE_ENABLE = os.getenv("ENDEE_ENABLE", "1") == "1"
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")
ENDEE_TOKEN = os.getenv("ENDEE_TOKEN", "")
ENDEE_INDEX = os.getenv("ENDEE_INDEX", "fake-news")
ENDEE_TOP_K = int(os.getenv("ENDEE_TOP_K", "3"))


def load_artifacts():
    """Load the trained model and vectorizer from disk."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            "Model or vectorizer not found. Run train_model.py first."
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def _normalize_result(item):
    """Handle both list-based and dict-based msgpack formats."""
    if isinstance(item, dict):
        return {
            "similarity": item.get("similarity"),
            "id": item.get("id"),
            "meta": item.get("meta"),
            "filter": item.get("filter"),
            "norm": item.get("norm"),
            "vector": item.get("vector"),
        }

    if isinstance(item, (list, tuple)) and len(item) >= 6:
        return {
            "similarity": item[0],
            "id": item[1],
            "meta": item[2],
            "filter": item[3],
            "norm": item[4],
            "vector": item[5],
        }

    return {}


def search_similar_in_endee(text: str, vectorizer) -> None:
    """Search Endee for similar articles using the same TF-IDF vector."""
    if not ENDEE_ENABLE:
        return

    headers = {"Content-Type": "application/json"}
    if ENDEE_TOKEN:
        headers["Authorization"] = ENDEE_TOKEN

    vector = vectorizer.transform([text]).toarray()[0].tolist()
    payload = {
        "k": ENDEE_TOP_K,
        "vector": vector,
        "include_vectors": False,
    }

    try:
        resp = requests.post(
            f"{ENDEE_URL}/api/v1/index/{ENDEE_INDEX}/search",
            json=payload,
            headers=headers,
            timeout=10,
        )
    except requests.exceptions.RequestException:
        return

    if resp.status_code != 200:
        return

    try:
        data = msgpack.unpackb(resp.content, raw=False)
    except Exception:
        return

    results = data.get("results", []) if isinstance(data, dict) else []
    if not results:
        return

    print("\nSimilar articles from Endee:")
    for item in results[:ENDEE_TOP_K]:
        result = _normalize_result(item)
        meta_bytes = result.get("meta")
        title = ""
        label = ""

        if isinstance(meta_bytes, (bytes, bytearray)):
            try:
                meta_obj = json.loads(meta_bytes.decode("utf-8"))
                title = str(meta_obj.get("title", ""))
                label = str(meta_obj.get("label", ""))
            except Exception:
                pass

        similarity = result.get("similarity")
        if similarity is not None:
            print(f"- score: {similarity:.4f} | label: {label} | title: {title}")


def main() -> None:
    model, vectorizer = load_artifacts()

    print("Enter a news title (optional):")
    title = input("> ").strip()

    print("Enter the news text (required):")
    text = input("> ").strip()

    if not text:
        print("No text provided. Please try again with news content.")
        return

    combined = (title + " " + text).strip()
    features = vectorizer.transform([combined])
    prediction = model.predict(features)[0]

    print(f"Prediction: {prediction}")

    # Optional: show similar articles using Endee
    search_similar_in_endee(combined, vectorizer)


if __name__ == "__main__":
    main()
