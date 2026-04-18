from visual_data_pipeline import OUTPUT_DIR, THEME, base_page, load_or_build_visual_data, metric_chart_svg


def main() -> None:
    data = load_or_build_visual_data()
    metric_defs = [
        {"key": "accuracy", "short": "Acc", "color": THEME["cyan"]},
        {"key": "precision", "short": "Pre", "color": THEME["blue"]},
        {"key": "recall", "short": "Rec", "color": THEME["deep_purple"]},
        {"key": "f1", "short": "F1", "color": THEME["light_purple"]},
    ]
    html = base_page(
        "Bar Chart ML",
        metric_chart_svg(data["classicModels"], metric_defs),
        data["notes"]["classic"],
    )
    (OUTPUT_DIR / "bar_chart_ml.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
