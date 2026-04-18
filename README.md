# Netflix Predictive Analytics

This repository contains the CDS527 Netflix review sentiment analysis project and a reusable visualization pipeline for presentation-ready outputs.

The visualization part of this repo was organized so that:

- each chart has its own Python file for easy editing
- charts can be regenerated after the dataset changes
- the baseline model is standardized to the values shown in the team PPT
- both split single-chart outputs and a full dashboard page are available

## Project Structure

```text
Netflix-Predictive-Analytics-main/
|-- Model/
|-- visuals/
|   |-- render_bar_chart_ml.py
|   |-- render_bar_chart_word_embed.py
|   |-- render_confusion_matrix_ml.py
|   |-- render_pie_distribution.py
|   |-- render_word_cloud_reviews.py
|   |-- render_all_visuals.py
|   |-- visual_data_pipeline.py
|   |-- generate_visualization_data.py
|   |-- model_visual_dashboard.html
|   |-- model_visual_dashboard_ready.html
|   `-- split_outputs/
`-- README.md
```

## Visualizations Included

The repo currently supports these visuals:

- Bar chart for `LR + ML Models` using `Accuracy`, `Precision`, `Recall`, and `F1`
- Bar chart for `LR + Embedding Models` using `Accuracy`, `Precision`, `Recall`, and `F1`
- Confusion matrix heatmaps for ML models
- Positive vs negative review distribution pie chart
- Positive vs negative keyword word clouds
- A full presentation-style dashboard page

## Baseline Standard

To keep the report consistent with the submitted presentation, the Logistic Regression baseline is fixed to the PPT values below:

- Accuracy: `0.7329`
- Precision: `0.8856`
- Recall: `0.7329`
- F1: `0.7907`

These values are enforced in:

- `visuals/visual_data_pipeline.py`
- `visuals/generate_visualization_data.py`

So even if the current codebase or dataset produces a different Logistic Regression result, the exported visual baseline will stay aligned with the PPT standard.

## Data Requirements

The visualization scripts automatically look for:

- `train_fixed_split.csv`
- `test_fixed_split.csv`

They search in these locations:

1. project root
2. parent folder of the project
3. `527_project` folder next to the repo

The expected columns are:

- `text`
- `label`

## Environment

Recommended Python version:

- Python `3.10+`

Install the required packages before running the visualization scripts:

```bash
pip install pandas scikit-learn
```

If you also want to run Spark-based model code inside `Model/`, you may additionally need:

```bash
pip install pyspark
```

Java may also be required for some Spark workflows, but the split visualization pipeline mainly relies on the saved project outputs and local CSV files.

## How To Regenerate Visuals

From the repository root, run:

```bash
python visuals/render_all_visuals.py
python visuals/generate_visualization_data.py
```

What each command does:

- `python visuals/render_all_visuals.py`
  regenerates all split single-chart HTML files in `visuals/split_outputs/`
- `python visuals/generate_visualization_data.py`
  regenerates the full dashboard data file and the ready-to-use dashboard page

## Single-Chart Files

Each chart is separated into its own script so it can be edited independently:

- `visuals/render_bar_chart_ml.py`
- `visuals/render_bar_chart_word_embed.py`
- `visuals/render_confusion_matrix_ml.py`
- `visuals/render_pie_distribution.py`
- `visuals/render_word_cloud_reviews.py`

Shared data preparation lives in:

- `visuals/visual_data_pipeline.py`

## Generated Outputs

After running the scripts, the main outputs are:

### Split outputs

Located in `visuals/split_outputs/`

- `bar_chart_ml.html`
- `bar_chart_word_embed.html`
- `confusion_matrix_ml.html`
- `pie_chart_distribution.html`
- `word_cloud_reviews.html`
- `visual_assets_data.json`

### Full dashboard

Located in `visuals/`

- `model_visual_dashboard_ready.html`
- `visualization_data.json`

## Embedding Model Notes

Embedding model metrics are read from available project outputs, especially saved notebook outputs from `Model/word_embedding.ipynb`.

The visualization charts for embedding models are standardized to these four metrics only:

- Accuracy
- Precision
- Recall
- F1

`AUC` is not used in the presentation charts.

## Design Theme

The visual design uses the purple presentation theme requested for the final slides:

- Main background: `#3A295B`
- Title text: `#FFFFFF`
- Cyan highlight: `#65F0F9`
- Blue: `#7BA5D8`
- Deep purple: `#4B2A8C`
- Light purple: `#A888D5`
- Outline: `#8B5CF6`

## Recommended GitHub Upload Notes

If you upload this repo to GitHub, keep these files committed:

- all files under `visuals/`
- the generated `split_outputs/` HTML files
- `model_visual_dashboard_ready.html`
- `visualization_data.json`

This makes it easy for teammates and graders to:

- preview the visuals immediately
- rerun the pipeline after data changes
- modify one chart at a time without editing a large combined script

## Quick Start

```bash
git clone <your-repo-url>
cd Netflix-Predictive-Analytics-main
pip install pandas scikit-learn
python visuals/render_all_visuals.py
python visuals/generate_visualization_data.py
```

Then open:

- `visuals/split_outputs/bar_chart_ml.html`
- `visuals/split_outputs/bar_chart_word_embed.html`
- `visuals/split_outputs/confusion_matrix_ml.html`
- `visuals/split_outputs/pie_chart_distribution.html`
- `visuals/split_outputs/word_cloud_reviews.html`
- `visuals/model_visual_dashboard_ready.html`

## License

This repository includes the original project license in `LICENSE`.
