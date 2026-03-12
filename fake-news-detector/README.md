# Fake News Detector (Beginner Friendly + Endee Vector Search)

This project is a simple, beginner-friendly machine learning pipeline that classifies news
articles as **Fake** or **Real** using **TF-IDF** and **Logistic Regression**. It also
stores TF-IDF vectors in the **Endee** vector database so you can run similarity search
against your news corpus.

## What This Project Does
- Cleans text and converts it into TF-IDF vectors
- Trains a Logistic Regression classifier
- Saves the trained model and vectorizer
- Indexes vectors into Endee for semantic search
- Provides a CLI for predictions

## Project Structure
```
fake-news-detector/
├ dataset/
│  └ news.csv
├ train_model.py
├ predict.py
├ requirements.txt
├ README.md
```

## Dataset Format (Required)
Your dataset must be a CSV at `dataset/news.csv` with **exactly** these columns:
- `title`
- `text`
- `label` (values like `Fake` or `Real`)

### If You Have Fake.csv and True.csv (Kaggle)
Use this to combine them into `dataset/news.csv`:
```
python - <<'PY'
import pandas as pd

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = "Fake"
true["label"] = "Real"

fake = fake[["title", "text", "label"]]
true = true[["title", "text", "label"]]

news = pd.concat([fake, true], ignore_index=True)
news.to_csv("dataset/news.csv", index=False)
print("Saved dataset/news.csv with", len(news), "rows")
PY
```

## Installation
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start (TL;DR)
```
cd /Users/sameersaifi/endee/fake-news-detector
source .venv/bin/activate
python train_model.py
python predict.py
```

## Start the Endee Vector Database (Optional but Recommended)
This project uses Endee for vector search. Start Endee with Docker:
```
docker run \
  --ulimit nofile=100000:100000 \
  -p 8080:8080 \
  -v ./endee-data:/data \
  --name endee-server \
  --restart unless-stopped \
  endeeio/endee-server:latest
```

Health check:
```
curl http://localhost:8080/api/v1/health
```

Stop Endee:
```
docker stop endee-server
```

## Train the Model
```
python train_model.py
```
This will:
- Train the model
- Print accuracy
- Save `model.pkl` and `vectorizer.pkl`
- Index vectors into Endee (if the server is running)

## Run Prediction
```
python predict.py
```
Then enter a title (optional) and the news text when prompted.
If Endee is running, it will also show similar articles.

### Sample Input
Title:
```
Scientists confirm new water discovery on Mars
```
Text:
```
NASA scientists announced evidence of ancient water flow on Mars after analyzing satellite imagery and rover data.
```

## Endee Configuration (Optional)
You can control Endee behavior with environment variables:
- `ENDEE_ENABLE` (default `1`) enable or disable Endee indexing/search
- `ENDEE_URL` (default `http://localhost:8080`)
- `ENDEE_TOKEN` (optional auth token)
- `ENDEE_INDEX` (default `fake-news`)
- `ENDEE_MAX_INDEX_ROWS` (default `2000`) limit how many rows to index
- `ENDEE_TOP_K` (default `3`) number of similar results in `predict.py`

Index all rows:
```
export ENDEE_MAX_INDEX_ROWS=0
python train_model.py
```

## Troubleshooting
- **Docker not found**: install Docker Desktop and run `docker --version`
- **Endee not reachable**: start the server and retry `python train_model.py`
- **Only one class error**: your dataset has only `Real` or only `Fake`. Add both classes.
- **Accuracy = 1.00**: this is normal with tiny or unbalanced data. Use a larger dataset.

## Notes
- The included sample dataset is tiny and only for quick testing.
- For real performance, use a full Kaggle dataset.
