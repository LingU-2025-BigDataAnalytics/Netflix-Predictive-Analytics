from visual_data_pipeline import build_visual_data, save_visual_data

import render_bar_chart_ml
import render_bar_chart_word_embed
import render_confusion_matrix_ml
import render_pie_distribution
import render_word_cloud_reviews


def main() -> None:
    data = build_visual_data()
    save_visual_data(data)
    render_bar_chart_ml.main()
    render_bar_chart_word_embed.main()
    render_confusion_matrix_ml.main()
    render_pie_distribution.main()
    render_word_cloud_reviews.main()
    print("Generated split visual assets in visuals/split_outputs")


if __name__ == "__main__":
    main()
