import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUT_PATH = BASE_DIR / "visualization_data.json"
TEMPLATE_PATH = BASE_DIR / "model_visual_dashboard.html"
READY_HTML_PATH = BASE_DIR / "model_visual_dashboard_ready.html"


EMBEDDING_RESULTS = [
    {
        "model": "Logistic Regression (Baseline)",
        "accuracy": 0.9262,
        "f1": 0.9606,
        "precision": 0.9437,
        "recall": 0.9781,
        "auc": None,
        "source": "data_preprocess.py / local rerun",
    },
    {
        "model": "Word2Vec",
        "accuracy": 0.9195,
        "f1": 0.9039,
        "precision": None,
        "recall": None,
        "auc": 0.7322,
        "source": "word_embedding.ipynb output",
    },
    {
        "model": "GloVe",
        "accuracy": 0.9060,
        "f1": 0.8914,
        "precision": None,
        "recall": None,
        "auc": 0.7927,
        "source": "word_embedding.ipynb output",
    },
    {
        "model": "BERT",
        "accuracy": 0.8926,
        "f1": 0.8881,
        "precision": None,
        "recall": None,
        "auc": 0.7310,
        "source": "word_embedding.ipynb output",
    },
]


def find_split_file(name: str) -> Path:
    candidates = [
        PROJECT_ROOT / name,
        PROJECT_ROOT.parent / name,
        PROJECT_ROOT.parent / "527_project" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to locate {name}. Checked: {candidates}")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = find_split_file("train_fixed_split.csv")
    test_path = find_split_file("test_fixed_split.csv")
    train_df = pd.read_csv(train_path).dropna(subset=["text"]).copy()
    test_df = pd.read_csv(test_path).dropna(subset=["text"]).copy()
    return train_df, test_df


def build_models() -> dict:
    return {
        "Logistic Regression (Baseline)": LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced",
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            random_state=42,
        ),
    }


def collect_keywords(texts: pd.Series, extra_stopwords: set[str]) -> list[dict]:
    base_stopwords = set(ENGLISH_STOP_WORDS) | {
        "album",
        "albums",
        "dark",
        "dont",
        "floyd",
        "good",
        "great",
        "just",
        "like",
        "moon",
        "movie",
        "movies",
        "music",
        "netflix",
        "one",
        "pink",
        "que",
        "really",
        "series",
        "show",
        "shows",
        "song",
        "songs",
        "story",
        "time",
        "watch",
        "watched",
        "watching",
    }
    stopwords = base_stopwords | extra_stopwords
    counter: Counter[str] = Counter()
    for text in texts.astype(str):
        tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
        counter.update(token for token in tokens if token not in stopwords)
    return [{"word": word, "value": value} for word, value in counter.most_common(24)]


def evaluate_classic_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    vectorizer = TfidfVectorizer(max_features=10000, min_df=2, ngram_range=(1, 2))
    x_train = vectorizer.fit_transform(train_df["text"])
    x_test = vectorizer.transform(test_df["text"])
    y_train = train_df["label"]
    y_test = test_df["label"]

    metrics = []
    confusion = {}

    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics.append(
            {
                "model": model_name,
                "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
                "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
                "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            }
        )
        confusion[model_name] = confusion_matrix(y_test, predictions).tolist()

    return {
        "metrics": metrics,
        "confusionMatrices": confusion,
    }


def build_dataset_stats(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df["word_count"] = full_df["text"].astype(str).str.split().str.len()
    label_counts = full_df["label"].value_counts().sort_index()
    return {
        "trainSamples": int(len(train_df)),
        "testSamples": int(len(test_df)),
        "negativeCount": int(label_counts.get(0, 0)),
        "positiveCount": int(label_counts.get(1, 0)),
        "meanWordCount": round(float(full_df["word_count"].mean()), 2),
        "varianceWordCount": round(float(full_df["word_count"].var()), 2),
        "stdWordCount": round(float(full_df["word_count"].std()), 2),
        "medianWordCount": round(float(full_df["word_count"].median()), 2),
    }


def build_payload() -> dict:
    train_df, test_df = load_data()
    classic = evaluate_classic_models(train_df, test_df)
    stats = build_dataset_stats(train_df, test_df)
    positive_keywords = collect_keywords(
        train_df.loc[train_df["label"] == 1, "text"],
        extra_stopwords={"positive"},
    )
    negative_keywords = collect_keywords(
        train_df.loc[train_df["label"] == 0, "text"],
        extra_stopwords={"negative"},
    )

    insights = [
        "Bar charts are split into two groups: LR + machine learning models, and LR + embedding models.",
        "The classic-model chart now includes Accuracy, Precision, Recall and F1 with a true 0-1 scale.",
        "The class distribution is shown with a pie chart, matching the requested visual style for presentation.",
        "Embedding Precision, Recall and confusion matrices are not recoverable from the current local files and should be requested from the teammate who ran those notebooks.",
    ]

    return {
        "meta": {
            "title": "Visualization + Data Statistics",
            "subtitle": "Revised model comparison dashboard based on feedback in Visual fix(2).docx",
            "classLabels": ["Negative", "Positive"],
        },
        "datasetStats": stats,
        "classDistribution": [
            {"label": "Negative", "count": stats["negativeCount"]},
            {"label": "Positive", "count": stats["positiveCount"]},
        ],
        "classicModels": classic["metrics"],
        "classicConfusionMatrices": classic["confusionMatrices"],
        "embeddingModels": EMBEDDING_RESULTS,
        "insights": insights,
        "positiveKeywords": positive_keywords,
        "negativeKeywords": negative_keywords,
        "notes": {
            "embeddingLimitation": "Word2Vec / GloVe / BERT Precision, Recall and confusion matrices are not present in the saved local outputs. Ask the teammate who ran word_embedding.ipynb to export the full classification reports if you need those figures on the final slide."
        },
    }


def main() -> None:
    payload = build_payload()
    payload_json = json.dumps(payload, indent=2)
    OUTPUT_PATH.write_text(payload_json, encoding="utf-8")

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    ready_html = template.replace("__VISUALIZATION_DATA__", payload_json)
    READY_HTML_PATH.write_text(ready_html, encoding="utf-8")

    print(f"Wrote visualization data to {OUTPUT_PATH}")
    print(f"Wrote ready-to-open dashboard to {READY_HTML_PATH}")


if __name__ == "__main__":
    main()
