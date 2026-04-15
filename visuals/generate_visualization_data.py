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


def evaluate_models(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
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


def build_payload() -> dict:
    train_df, test_df = load_data()
    model_data = evaluate_models(train_df, test_df)
    positive_keywords = collect_keywords(
        train_df.loc[train_df["label"] == 1, "text"],
        extra_stopwords={"positive"},
    )
    negative_keywords = collect_keywords(
        train_df.loc[train_df["label"] == 0, "text"],
        extra_stopwords={"negative"},
    )

    insights = [
        "Baseline logistic regression reaches the best overall accuracy and the best balance across both classes.",
        "Random Forest and Gradient Boosting recall every positive review, but both miss almost all negatives.",
        "Decision Tree improves minority detection slightly, yet still struggles with the negative class.",
        "Keyword differences can support the presentation narrative alongside model-level metrics.",
    ]

    return {
        "meta": {
            "title": "Model Visualization Dashboard",
            "subtitle": "Sentiment Classification Comparison for Presentation Use",
            "classLabels": ["Negative", "Positive"],
        },
        "datasetSummary": {
            "trainSamples": int(len(train_df)),
            "testSamples": int(len(test_df)),
            "negativeTrain": int((train_df["label"] == 0).sum()),
            "positiveTrain": int((train_df["label"] == 1).sum()),
        },
        "insights": insights,
        **model_data,
        "positiveKeywords": positive_keywords,
        "negativeKeywords": negative_keywords,
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
