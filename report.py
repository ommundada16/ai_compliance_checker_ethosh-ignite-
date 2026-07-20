def build_html(report) -> str:
    SEV_COLOR = {
        "Critical": "#ef4444",
        "High": "#f97316",
        "Medium": "#eab308",
        "Low": "#22c55e"
    }

    # Count statistics
    total_findings = len(report.findings)
    avg_confidence = int(sum(f.confidence for f in report.findings) / total_findings * 100) if total_findings else 0
    sev_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    category_counts = {}
    for f in report.findings:
        sev_counts[f.severity] = sev_counts.get(f.severity, 0) + 1
        category_counts[f.category] = category_counts.get(f.category, 0) + 1

    crit_high_count = sev_counts["Critical"] + sev_counts["High"]

    # Generate category list HTML
    category_html = ""
    if category_counts:
        max_count = max(category_counts.values())
        sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        for cat, count in sorted_cats:
            pct = (count / max_count) * 100
            category_html += f"""
            <div style="margin-bottom: 12px;">
              <div style="display: flex; justify-content: space-between; font-size: 11px; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 4px; letter-spacing: 0.5px;">
                <span>{cat}</span>
                <span>{count}</span>
              </div>
              <div style="background-color: #f1f5f9; height: 6px; border-radius: 3px; overflow: hidden;">
                <div style="background-color: #0b57d0; height: 100%; width: {pct}%; border-radius: 3px;"></div>
              </div>
            </div>"""

    # Generate findings cards HTML
    cards_html = ""
    severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    sorted_findings = sorted(report.findings, key=lambda x: severity_order.get(x.severity, 4))
    
    for f in sorted_findings:
        sev_lower = f.severity.lower()
        conf_pct = int(f.confidence * 100)
        
        conf_class_color = "#22c55e" # High
        if conf_pct < 70:
            conf_class_color = "#cbd5e1" # Low
        elif conf_pct < 85:
            conf_class_color = "#0b57d0" # Medium

        suggested_fix_html = ""
        if f.suggested_correction:
            text = f.suggested_correction
            if not text.lower().startswith("suggested fix"):
                text = f"Suggested fix: {text}"
            suggested_fix_html = f"""
            <div class="suggested-fix-box">
              <div class="suggested-fix-icon">💡</div>
              <div class="suggested-fix-text">{text}</div>
            </div>"""

        badge_bg = "#fee2e2"
        badge_fg = "#ef4444"
        if f.severity == "High":
            badge_bg = "#ffedd5"
            badge_fg = "#f97316"
        elif f.severity == "Medium":
            badge_bg = "#fef9c3"
            badge_fg = "#ca8a04"
        elif f.severity == "Low":
            badge_bg = "#dcfce7"
            badge_fg = "#22c55e"

        cards_html += f"""
        <div class="finding-card" style="border-left: 4px solid {SEV_COLOR[f.severity]};">
          <div class="finding-card-header">
            <div class="finding-card-badges">
              <span class="badge-severity" style="background-color: {badge_bg}; color: {badge_fg};">{f.severity}</span>
              <span class="badge-clause">{f.guideline_clause}</span>
            </div>
            <div class="finding-card-confidence">
              <span>CONFIDENCE</span>
              <div class="confidence-bar-bg">
                <div class="confidence-bar-fill" style="width: {conf_pct}%; background-color: {conf_class_color};"></div>
              </div>
              <span style="color: #0f172a; font-weight: 700;">{conf_pct}%</span>
            </div>
          </div>
          <div class="finding-card-title">{f.violating_statement}</div>
          <div class="finding-card-desc">{f.explanation}</div>
          {suggested_fix_html}
        </div>"""

    # Generate Severity Distribution segment percentages
    if total_findings > 0:
        crit_pct = (sev_counts["Critical"] / total_findings) * 100
        high_pct = (sev_counts["High"] / total_findings) * 100
        med_pct = (sev_counts["Medium"] / total_findings) * 100
        low_pct = (sev_counts["Low"] / total_findings) * 100
    else:
        crit_pct = high_pct = med_pct = low_pct = 0

    severity_bar_html = f"""
    <div style="display:flex; height:8px; border-radius:4px; overflow:hidden; background:#e2e8f0; margin-bottom:16px;">
      <div style="width: {crit_pct}%; background: #ef4444;"></div>
      <div style="width: {high_pct}%; background: #f97316;"></div>
      <div style="width: {med_pct}%; background: #eab308;"></div>
      <div style="width: {low_pct}%; background: #22c55e;"></div>
    </div>
    <div style="display:flex; justify-content:space-between; font-size:11px; font-weight:600; color:#64748b;">
      <span><span style="color:#ef4444; margin-right:4px;">●</span>Critical ({sev_counts['Critical']})</span>
      <span><span style="color:#f97316; margin-right:4px;">●</span>High ({sev_counts['High']})</span>
      <span><span style="color:#eab308; margin-right:4px;">●</span>Medium ({sev_counts['Medium']})</span>
      <span><span style="color:#22c55e; margin-right:4px;">●</span>Low ({sev_counts['Low']})</span>
    </div>"""

    score = int(report.readiness_score)
    crit_high_pill = '<span class="metric-card-pill">Requires Attention</span>' if crit_high_count > 0 else ''

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Regulatory Compliance Audit Report</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    body {{
      font-family: 'Outfit', Arial, sans-serif;
      margin: 0;
      padding: 40px;
      background-color: #f8fafc;
      color: #0f172a;
    }}
    .report-container {{
      max-width: 1000px;
      margin: 0 auto;
    }}
    .header {{
      background-color: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 16px;
      padding: 32px;
      margin-bottom: 24px;
    }}
    .title-area {{
      margin-bottom: 24px;
    }}
    .title {{
      font-size: 28px;
      font-weight: 700;
      color: #0f172a;
      margin: 0 0 6px 0;
    }}
    .subtitle {{
      font-size: 15px;
      color: #64748b;
      margin: 0;
    }}
    
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 16px;
      margin-bottom: 24px;
    }}
    .metric-card {{
      background-color: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      min-height: 100px;
    }}
    .metric-card-label {{
      font-size: 10px;
      font-weight: 700;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 6px;
    }}
    .metric-card-value {{
      font-size: 24px;
      font-weight: 700;
      color: #0f172a;
    }}
    .metric-card-subvalue {{
      font-size: 12px;
      color: #64748b;
      margin-top: 2px;
    }}
    .metric-card-pill {{
      background-color: #ffedd5;
      color: #ea580c;
      font-size: 11px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 9999px;
      margin-left: 6px;
    }}
    
    .analytics-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-bottom: 32px;
    }}
    .analytics-card {{
      background-color: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 24px;
    }}
    .analytics-card-title {{
      font-size: 15px;
      font-weight: 700;
      color: #0f172a;
      margin-bottom: 16px;
    }}

    .finding-card {{
      background-color: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 16px;
      position: relative;
    }}
    .finding-card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }}
    .finding-card-badges {{
      display: flex;
      gap: 8px;
      align-items: center;
    }}
    .badge-severity {{
      font-size: 10px;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 5px;
      text-transform: uppercase;
    }}
    .badge-clause {{
      background-color: #f1f5f9;
      color: #475569;
      font-size: 10px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 5px;
    }}
    .finding-card-confidence {{
      font-size: 10px;
      font-weight: 700;
      color: #64748b;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .confidence-bar-bg {{
      background-color: #e2e8f0;
      width: 50px;
      height: 4px;
      border-radius: 2px;
      overflow: hidden;
    }}
    .confidence-bar-fill {{
      height: 100%;
      border-radius: 2px;
    }}
    
    .finding-card-title {{
      font-size: 16px;
      font-weight: 700;
      color: #0f172a;
      margin-bottom: 6px;
    }}
    .finding-card-desc {{
      font-size: 13.5px;
      color: #475569;
      line-height: 1.45;
      margin-bottom: 14px;
    }}
    .suggested-fix-box {{
      background-color: #f0fdf4;
      border: 1px solid #bbf7d0;
      border-radius: 8px;
      padding: 12px 14px;
      display: flex;
      gap: 10px;
      align-items: flex-start;
    }}
    .suggested-fix-icon {{
      font-size: 15px;
      color: #22c55e;
    }}
    .suggested-fix-text {{
      font-size: 13px;
      color: #166534;
      line-height: 1.45;
    }}
  </style>
</head>
<body>
  <div class="report-container">
    <div class="header">
      <div class="title-area">
        <h1 class="title">Regulatory Compliance Audit Report</h1>
        <p class="subtitle">Document: <b>{report.document_name}</b></p>
      </div>
      
      <div class="metrics-grid">
        <div class="metric-card">
          <div>
            <div class="metric-card-label">Readiness Score</div>
            <div class="metric-card-value" style="color: #0b57d0;">{score}/100</div>
          </div>
        </div>
        <div class="metric-card">
          <div>
            <div class="metric-card-label">Total Findings</div>
            <div class="metric-card-value">{total_findings}</div>
            <div class="metric-card-subvalue">Across {len(category_counts)} sections</div>
          </div>
        </div>
        <div class="metric-card">
          <div>
            <div class="metric-card-label">Critical + High</div>
            <div style="display: flex; align-items: center;">
              <span class="metric-card-value">{crit_high_count}</span>
              {crit_high_pill}
            </div>
          </div>
        </div>
        <div class="metric-card">
          <div style="width: 100%;">
            <div class="metric-card-label">Avg Confidence</div>
            <div class="metric-card-value">{avg_confidence}%</div>
          </div>
        </div>
      </div>
      
      <div class="analytics-grid">
        <div class="analytics-card">
          <div class="analytics-card-title">Severity Distribution</div>
          {severity_bar_html}
        </div>
        <div class="analytics-card">
          <div class="analytics-card-title">Issues by Category</div>
          {category_html}
        </div>
      </div>
    </div>

    <h2 style="font-size: 18px; font-weight: 700; color: #475569; margin: 32px 0 16px 0; letter-spacing: 0.5px;">AUDIT FINDINGS</h2>
    {cards_html}
  </div>
</body>
</html>"""