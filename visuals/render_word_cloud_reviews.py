from visual_data_pipeline import OUTPUT_DIR, base_page, load_or_build_visual_data, word_cloud_svg


def main() -> None:
    data = load_or_build_visual_data()
    body = f"""
    <div class="word-cols">
      <section class="word-panel">
        <h2 style="color:#65F0F9">Positive Reviews</h2>
        {word_cloud_svg(data["positiveKeywords"], ["#65F0F9", "#FFFFFF", "#7BA5D8", "#D8F9FF", "#A888D5"])}
      </section>
      <section class="word-panel">
        <h2 style="color:#A888D5">Negative Reviews</h2>
        {word_cloud_svg(data["negativeKeywords"], ["#A888D5", "#FFFFFF", "#65F0F9", "#7BA5D8", "#D9C6FF"])}
      </section>
    </div>
    """
    html = base_page("Word Cloud Reviews", body)
    (OUTPUT_DIR / "word_cloud_reviews.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
