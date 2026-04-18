from visual_data_pipeline import OUTPUT_DIR, THEME, base_page, load_or_build_visual_data


def main() -> None:
    data = load_or_build_visual_data()
    items = data["classDistribution"]
    total = sum(item["count"] for item in items)
    positive = next(item["count"] for item in items if item["label"] == "Positive")
    positive_ratio = positive / total if total else 0
    negative = total - positive
    body = f"""
    <div style="display:grid;grid-template-columns:300px 1fr;gap:24px;align-items:center;">
      <div style="width:260px;height:260px;border-radius:50%;margin:0 auto;position:relative;background:conic-gradient({THEME['cyan']} 0 {(positive_ratio * 360):.1f}deg, {THEME['light_purple']} {(positive_ratio * 360):.1f}deg 360deg);box-shadow:inset 0 0 0 1px rgba(255,255,255,0.08);">
        <div style="position:absolute;inset:40px;border-radius:50%;background:rgba(44,28,74,0.96);border:1px solid rgba(255,255,255,0.1);display:flex;align-items:center;justify-content:center;flex-direction:column;">
          <div style="font-size:36px;font-weight:800;">{positive_ratio * 100:.1f}%</div>
          <div style="font-size:13px;letter-spacing:0.16em;text-transform:uppercase;color:rgba(255,255,255,0.75);">Positive share</div>
        </div>
      </div>
      <div>
        <div style="display:grid;gap:14px;">
          <div style="display:flex;justify-content:space-between;padding:16px 18px;border-radius:18px;background:rgba(255,255,255,0.06);">
            <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{THEME['light_purple']};margin-right:10px;"></span>Negative</div>
            <strong>{negative}</strong>
          </div>
          <div style="display:flex;justify-content:space-between;padding:16px 18px;border-radius:18px;background:rgba(255,255,255,0.06);">
            <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{THEME['cyan']};margin-right:10px;"></span>Positive</div>
            <strong>{positive}</strong>
          </div>
          <div style="padding:16px 18px;border-radius:18px;background:rgba(255,255,255,0.06);line-height:1.7;color:rgba(255,255,255,0.84);">
            Train / Test: <strong>{data['datasetStats']['trainSamples']} / {data['datasetStats']['testSamples']}</strong><br>
            Mean word count: <strong>{data['datasetStats']['meanWordCount']:.2f}</strong><br>
            Median word count: <strong>{data['datasetStats']['medianWordCount']:.2f}</strong>
          </div>
        </div>
      </div>
    </div>
    """
    html = base_page("Pie Chart Distribution", body, data["notes"]["reproducibility"])
    (OUTPUT_DIR / "pie_chart_distribution.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
