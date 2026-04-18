from visual_data_pipeline import OUTPUT_DIR, base_page, load_or_build_visual_data, matrix_heat_color


def main() -> None:
    data = load_or_build_visual_data()
    parts = ["<div class='matrix-grid'>"]
    for model, matrix in data["classicConfusionMatrices"].items():
        max_value = max(value for row in matrix for value in row)
        tn, fp = matrix[0]
        fn, tp = matrix[1]
        parts.append(
            f"""
            <article class="matrix-card">
              <h2 class="matrix-title">{model}</h2>
              <div class="matrix">
                <div class="matrix-label"></div>
                <div class="matrix-label">Pred Neg</div>
                <div class="matrix-label">Pred Pos</div>
                <div class="matrix-label">True Neg</div>
                <div class="matrix-cell" style="background:{matrix_heat_color(tn, max_value)}">{tn}</div>
                <div class="matrix-cell" style="background:{matrix_heat_color(fp, max_value)}">{fp}</div>
                <div class="matrix-label">True Pos</div>
                <div class="matrix-cell" style="background:{matrix_heat_color(fn, max_value)}">{fn}</div>
                <div class="matrix-cell" style="background:{matrix_heat_color(tp, max_value)}">{tp}</div>
              </div>
            </article>
            """
        )
    parts.append("</div>")
    html = base_page("Confusion Matrix ML", "".join(parts), data["notes"]["classic"])
    (OUTPUT_DIR / "confusion_matrix_ml.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
