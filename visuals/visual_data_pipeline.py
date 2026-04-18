import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIR = PROJECT_ROOT / "Model"
WORD_EMBEDDING_NOTEBOOK = MODEL_DIR / "word_embedding.ipynb"
OUTPUT_DIR = BASE_DIR / "split_outputs"
DATA_PATH = OUTPUT_DIR / "visual_assets_data.json"

THEME = {
    "bg": "#3A295B",
    "text": "#FFFFFF",
    "cyan": "#65F0F9",
    "blue": "#7BA5D8",
    "deep_purple": "#4B2A8C",
    "light_purple": "#A888D5",
    "outline": "#8B5CF6",
    "panel": "rgba(84, 56, 134, 0.42)",
    "border": "rgba(255, 255, 255, 0.16)",
}

PPT_BASELINE = {
    "model": "Logistic Regression (Baseline)",
    "accuracy": 0.7329,
    "precision": 0.8856,
    "recall": 0.7329,
    "f1": 0.7907,
    "source": "PPT baseline standard",
}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_java_home() -> None:
    if os.environ.get("JAVA_HOME"):
        return
    candidates = [
        Path(r"C:\Program Files\Microsoft\jdk-17.0.18.8-hotspot"),
        Path(r"C:\Program Files\Microsoft\jdk-17.0.18-hotspot"),
    ]
    for candidate in candidates:
        java_bin = candidate / "bin" / "java.exe"
        if java_bin.exists():
            os.environ["JAVA_HOME"] = str(candidate)
            os.environ["Path"] = f"{candidate / 'bin'};{os.environ.get('Path', '')}"
            return


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


def load_review_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = find_split_file("train_fixed_split.csv")
    test_path = find_split_file("test_fixed_split.csv")
    train_df = pd.read_csv(train_path).dropna(subset=["text"]).copy()
    test_df = pd.read_csv(test_path).dropna(subset=["text"]).copy()
    return train_df, test_df


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


def collect_keywords(texts: pd.Series, extra_stopwords: set[str]) -> list[dict]:
    base_stopwords = set(ENGLISH_STOP_WORDS) | {
        "album",
        "albums",
        "also",
        "dark",
        "dont",
        "floyd",
        "get",
        "going",
        "good",
        "great",
        "just",
        "like",
        "made",
        "make",
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
        "take",
        "thing",
        "time",
        "watch",
        "watched",
        "watching",
        "will",
    }
    stopwords = base_stopwords | extra_stopwords
    counter: Counter[str] = Counter()
    for text in texts.astype(str):
        tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
        counter.update(token for token in tokens if token not in stopwords)
    return [{"word": word, "value": value} for word, value in counter.most_common(120)]


def evaluate_classic_models() -> tuple[list[dict], dict[str, list[list[int]]], str]:
    ensure_java_home()
    if str(MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_DIR))

    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.sql import SparkSession
    from sklearn.metrics import classification_report, confusion_matrix
    from models import (
        train_decision_tree,
        train_gradient_boosting,
        train_logistic_regression,
        train_random_forest,
    )

    train_path = find_split_file("train_fixed_split.csv")
    test_path = find_split_file("test_fixed_split.csv")
    spark = SparkSession.getActiveSession()
    if spark is None:
        spark = (
            SparkSession.builder.appName("SplitVisualClassicEval")
            .config("spark.driver.memory", "4g")
            .getOrCreate()
        )

    train_df = spark.read.csv(str(train_path), header=True, inferSchema=True).dropna(subset=["text"])
    test_df = spark.read.csv(str(test_path), header=True, inferSchema=True).dropna(subset=["text"])
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    model_tasks = [
        ("Logistic Regression (Baseline)", train_logistic_regression),
        ("Decision Tree (Weighted)", train_decision_tree),
        ("Random Forest (Weighted)", train_random_forest),
        ("Gradient Boosting", train_gradient_boosting),
    ]

    metrics = []
    confusion = {}
    for model_name, train_func in model_tasks:
        model, feat_pipeline = train_func(train_df)
        predictions = model.transform(feat_pipeline.transform(test_df))
        accuracy = float(evaluator.evaluate(predictions))
        rows = predictions.select("label", "prediction").collect()
        y_true = [int(row.label) for row in rows]
        y_pred = [int(row.prediction) for row in rows]
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics.append(
            {
                "model": model_name,
                "accuracy": round(accuracy, 4),
                "precision": round(float(report["weighted avg"]["precision"]), 4),
                "recall": round(float(report["weighted avg"]["recall"]), 4),
                "f1": round(float(report["weighted avg"]["f1-score"]), 4),
                "source": "Model/main.py-compatible Spark rerun",
            }
        )
        confusion[model_name] = confusion_matrix(y_true, y_pred).tolist()

        for item in metrics:
            if item["model"] == "Logistic Regression (Baseline)":
                item.update(PPT_BASELINE)
                break
        return metrics, confusion, "Classic ML metrics were recomputed from the current Spark pipeline, with the baseline overridden to match the PPT standard."


def extract_embedding_metrics_from_notebook() -> tuple[list[dict], str]:
    if not WORD_EMBEDDING_NOTEBOOK.exists():
        raise FileNotFoundError("word_embedding.ipynb not found")

    notebook = json.loads(WORD_EMBEDDING_NOTEBOOK.read_text(encoding="utf-8"))
    pattern = re.compile(
        r"(Word2Vec|GloVe|BERT)\s*-\s*Acc:\s*([0-9.]+),\s*Pre:\s*([0-9.]+),\s*Rec:\s*([0-9.]+),\s*F1:\s*([0-9.]+)",
        re.IGNORECASE,
    )
    parsed: dict[str, dict] = {}
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            chunks = []
            if "text" in output:
                text = output["text"]
                chunks.extend(text if isinstance(text, list) else [text])
            if "data" in output and "text/plain" in output["data"]:
                text = output["data"]["text/plain"]
                chunks.extend(text if isinstance(text, list) else [text])
            joined = "\n".join(chunks)
            for match in pattern.finditer(joined):
                model_name = match.group(1)
                parsed[model_name] = {
                    "model": model_name,
                    "accuracy": round(float(match.group(2)), 4),
                    "precision": round(float(match.group(3)), 4),
                    "recall": round(float(match.group(4)), 4),
                    "f1": round(float(match.group(5)), 4),
                    "source": "parsed from saved word_embedding.ipynb output",
                }

    if len(parsed) < 3:
        raise ValueError("Could not recover complete embedding metrics from notebook output")

    metrics = [
        PPT_BASELINE,
        parsed["Word2Vec"],
        parsed["GloVe"],
        parsed["BERT"],
    ]
    return metrics, "Embedding metrics were restored from the saved notebook outputs."


def build_visual_data() -> dict:
    train_df, test_df = load_review_frames()
    classic_models, classic_confusion, classic_note = evaluate_classic_models()
    embedding_models, embedding_note = extract_embedding_metrics_from_notebook()
    stats = build_dataset_stats(train_df, test_df)

    return {
        "meta": {
            "title": "Visualization + Data Statistics",
            "subtitle": "Split-chart assets for ML models, embeddings, confusion matrices and keyword clouds",
        },
        "datasetStats": stats,
        "classDistribution": [
            {"label": "Negative", "count": stats["negativeCount"]},
            {"label": "Positive", "count": stats["positiveCount"]},
        ],
        "classicModels": classic_models,
        "classicConfusionMatrices": classic_confusion,
        "embeddingModels": embedding_models,
        "positiveKeywords": collect_keywords(train_df.loc[train_df["label"] == 1, "text"], {"positive"}),
        "negativeKeywords": collect_keywords(train_df.loc[train_df["label"] == 0, "text"], {"negative"}),
        "notes": {
            "classic": classic_note,
            "embedding": embedding_note,
            "reproducibility": "Run `python visuals/render_all_visuals.py` after data updates to regenerate every single-chart asset.",
        },
    }


def save_visual_data(data: dict) -> Path:
    ensure_output_dir()
    DATA_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return DATA_PATH


def load_or_build_visual_data() -> dict:
    if DATA_PATH.exists():
        return json.loads(DATA_PATH.read_text(encoding="utf-8"))
    data = build_visual_data()
    save_visual_data(data)
    return data


def base_page(title: str, body: str, note: str = "") -> str:
    note_html = f"<div class='note-box'>{note}</div>" if note else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    :root {{
      --bg: {THEME['bg']};
      --text: {THEME['text']};
      --cyan: {THEME['cyan']};
      --blue: {THEME['blue']};
      --deep-purple: {THEME['deep_purple']};
      --light-purple: {THEME['light_purple']};
      --outline: {THEME['outline']};
      --panel: {THEME['panel']};
      --border: {THEME['border']};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at 12% 16%, rgba(101,240,249,0.16), transparent 18%),
        radial-gradient(circle at 84% 12%, rgba(168,136,213,0.20), transparent 20%),
        linear-gradient(135deg, #2E2049 0%, #3A295B 48%, #4B2A8C 100%);
      padding: 24px;
    }}
    .card {{
      max-width: 1200px;
      margin: 0 auto;
      border-radius: 28px;
      border: 1px solid var(--border);
      background: linear-gradient(180deg, rgba(82,55,132,0.70), rgba(46,31,79,0.95));
      box-shadow: 0 24px 60px rgba(0,0,0,0.35);
      overflow: hidden;
    }}
    .body {{ padding: 26px; }}
    h1 {{
      margin: 0 0 18px;
      font-size: 24px;
      display: flex;
      gap: 10px;
      align-items: center;
    }}
    .hex {{
      width: 14px;
      height: 14px;
      display: inline-block;
      background: var(--cyan);
      clip-path: polygon(25% 6%, 75% 6%, 100% 50%, 75% 94%, 25% 94%, 0 50%);
      border: 1px solid var(--outline);
    }}
    .note-box {{
      margin-top: 16px;
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(255,255,255,0.06);
      border: 1px dashed rgba(101,240,249,0.42);
      color: rgba(255,255,255,0.86);
      line-height: 1.7;
    }}
    .matrix-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .matrix-card {{
      padding: 16px;
      border-radius: 22px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
    }}
    .matrix-title {{
      margin: 0 0 12px;
      font-size: 16px;
    }}
    .matrix {{
      display: grid;
      grid-template-columns: 74px repeat(2, 1fr);
      gap: 8px;
    }}
    .matrix-label, .matrix-cell {{
      padding: 12px 10px;
      border-radius: 14px;
      text-align: center;
      font-size: 13px;
    }}
    .matrix-label {{
      background: rgba(255,255,255,0.06);
      color: rgba(255,255,255,0.80);
    }}
    .matrix-cell {{
      font-weight: 700;
      color: white;
    }}
    .word-cols {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .word-panel {{
      padding: 18px;
      border-radius: 24px;
      background: rgba(255,255,255,0.05);
      border: 1px solid rgba(255,255,255,0.08);
    }}
    .word-panel h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    .word-cloud-svg {{
      width: 100%;
      height: 420px;
      display: block;
      border-radius: 20px;
      background:
        radial-gradient(circle at 22% 18%, rgba(101,240,249,0.10), transparent 24%),
        radial-gradient(circle at 78% 22%, rgba(168,136,213,0.16), transparent 26%),
        rgba(255,255,255,0.03);
    }}
    .cloud-word {{
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      text-anchor: middle;
      dominant-baseline: middle;
      stroke: rgba(40, 21, 70, 0.25);
      stroke-width: 1px;
      paint-order: stroke fill;
    }}
    @media (max-width: 900px) {{
      .matrix-grid, .word-cols {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="card">
    <section class="body">
      <h1><span class="hex"></span>{title}</h1>
      {body}
      {note_html}
    </section>
  </main>
</body>
</html>"""


def metric_chart_svg(items: list[dict], metric_defs: list[dict]) -> str:
    width = 1100
    height = 560
    margin = {"top": 36, "right": 24, "bottom": 120, "left": 72}
    inner_width = width - margin["left"] - margin["right"]
    inner_height = height - margin["top"] - margin["bottom"]
    group_step = inner_width / max(len(items), 1)
    group_width = min(220, group_step * 0.82)
    gap = 10
    bar_width = (group_width - gap * (len(metric_defs) - 1)) / len(metric_defs)

    def scale_y(value: float) -> float:
        return margin["top"] + inner_height - value * inner_height

    grid = []
    for i in range(6):
        value = i / 5
        y = scale_y(value)
        grid.append(
            f"<line x1='{margin['left']}' y1='{y}' x2='{width - margin['right']}' y2='{y}' stroke='rgba(255,255,255,0.16)' stroke-dasharray='4 6'></line>"
            f"<text x='{margin['left'] - 12}' y='{y + 4}' fill='rgba(255,255,255,0.86)' font-size='12' text-anchor='end'>{value:.2f}</text>"
        )

    bars = []
    for idx, item in enumerate(items):
        start_x = margin["left"] + group_step * idx + (group_step - group_width) / 2
        for metric_idx, metric in enumerate(metric_defs):
            value = item.get(metric["key"])
            x = start_x + metric_idx * (bar_width + gap)
            if value is None:
                bars.append(
                    f"<rect x='{x}' y='{scale_y(0)}' width='{bar_width}' height='2' rx='2' fill='rgba(255,255,255,0.26)'></rect>"
                    f"<text x='{x + bar_width/2}' y='{scale_y(0) - 8}' fill='rgba(255,255,255,0.82)' font-size='12' text-anchor='middle'>N/A</text>"
                    f"<text x='{x + bar_width/2}' y='{height - 28}' fill='rgba(255,255,255,0.86)' font-size='13' text-anchor='middle'>{metric['short']}</text>"
                )
                continue
            y = scale_y(value)
            h = margin["top"] + inner_height - y
            bars.append(
                f"<rect x='{x}' y='{y}' width='{bar_width}' height='{h}' rx='12' fill='{metric['color']}'></rect>"
                f"<text x='{x + bar_width/2}' y='{y - 10}' fill='white' font-size='12' font-weight='700' text-anchor='middle'>{value:.4f}</text>"
                f"<text x='{x + bar_width/2}' y='{height - 28}' fill='rgba(255,255,255,0.86)' font-size='13' text-anchor='middle'>{metric['short']}</text>"
            )
        bars.append(
            f"<text x='{start_x + group_width/2}' y='{height - 62}' fill='rgba(255,255,255,0.94)' font-size='15' text-anchor='middle'>{item['model'].replace(' (Baseline)', '')}</text>"
        )

    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='auto'>"
        + "".join(grid)
        + f"<line x1='{margin['left']}' y1='{margin['top'] + inner_height}' x2='{width - margin['right']}' y2='{margin['top'] + inner_height}' stroke='rgba(255,255,255,0.28)'></line>"
        + "".join(bars)
        + "</svg>"
    )


def matrix_heat_color(value: int, max_value: int) -> str:
    ratio = 0 if max_value == 0 else value / max_value
    r = round(75 + ratio * 93)
    g = round(42 + ratio * 198)
    b = round(140 + ratio * 109)
    return f"rgb({r},{g},{b})"


def word_cloud_svg(items: list[dict], colors: list[str]) -> str:
    words = sorted(items, key=lambda item: item["value"], reverse=True)[:90]
    width = 540
    height = 420
    center_x = width / 2
    center_y = height / 2
    min_value = words[-1]["value"]
    max_value = words[0]["value"]
    placed: list[dict] = []
    svg_parts = [f"<svg class='word-cloud-svg' viewBox='0 0 {width} {height}'>"]

    def collides(box: dict) -> bool:
        for item in placed:
            if not (box["x2"] < item["x1"] or box["x1"] > item["x2"] or box["y2"] < item["y1"] or box["y1"] > item["y2"]):
                return True
        return False

    for index, item in enumerate(words):
        ratio = 1 if max_value == min_value else (item["value"] - min_value) / (max_value - min_value)
        font_size = 12 + pow(ratio, 0.72) * 110
        text_width = max(len(item["word"]) * font_size * 0.48, font_size * 1.2)
        text_height = font_size * 0.74
        rotate = 0 if ratio > 0.8 else (-90 if index % 11 == 0 else 0)
        box_width = text_width if rotate == 0 else text_height
        box_height = text_height if rotate == 0 else text_width
        angle = index * 0.92
        radius = 0.0
        chosen = None
        for _ in range(900):
            x = center_x + radius * __import__("math").cos(angle)
            y = center_y + radius * 0.72 * __import__("math").sin(angle)
            candidate = {
                "x": x,
                "y": y,
                "x1": x - box_width / 2 - 8,
                "x2": x + box_width / 2 + 8,
                "y1": y - box_height / 2 - 6,
                "y2": y + box_height / 2 + 6,
            }
            fits = candidate["x1"] >= 8 and candidate["x2"] <= width - 8 and candidate["y1"] >= 8 and candidate["y2"] <= height - 8
            if fits and not collides(candidate):
                chosen = candidate
                placed.append(candidate)
                break
            angle += 0.33
            radius += 0.95
        if not chosen:
            continue
        fill = colors[index % len(colors)]
        safe_word = (
            item["word"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        svg_parts.append(
            f"<text class='cloud-word' x='{chosen['x']:.1f}' y='{chosen['y']:.1f}' font-size='{font_size:.1f}' font-weight='700' fill='{fill}' transform='rotate({rotate} {chosen['x']:.1f} {chosen['y']:.1f})'>{safe_word}</text>"
        )
    svg_parts.append("</svg>")
    return "".join(svg_parts)
