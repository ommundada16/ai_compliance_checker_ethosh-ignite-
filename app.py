import streamlit as st
import tempfile
import json
import os
import textwrap
from schema import AuditReport
from report import build_html

# Anchor data paths to this file's own directory, not the process's current
# working directory (which depends on where `streamlit run` was launched from).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Import heavy ML dependencies (sentence-transformers/torch) once at module
# load time instead of on the first "Run Compliance Audit" click -- these
# imports alone can take 10-20s the first time, which otherwise shows up as a
# silent freeze right after clicking the button.
from ingest import extract_text, extract_pages, chunk_text, quote_locatable
from retriever import Retriever
from auditor import audit_section, readiness_score, verify_correction


@st.cache_resource(show_spinner=False)
def get_guideline_retriever(guideline_path: str, mtime: float):
    """Build (and cache) the guideline retriever once per server process.
    Recomputing embeddings for the guideline PDF on every single audit was
    pure wasted time since the guideline never changes between runs.
    """
    guideline_text = extract_text(guideline_path)
    guideline_chunks = chunk_text(guideline_text)
    return Retriever(guideline_chunks)

# Check query params for home navigation
if st.query_params.get("screen") == "home":
    st.session_state.current_screen = "home"
    st.session_state.report = None
    st.query_params.clear()
    st.rerun()

# Set page config
st.set_page_config(page_title="The Auditor", layout="wide", initial_sidebar_state="expanded")

# Initialize session state
if "current_screen" not in st.session_state:
    st.session_state.current_screen = "home"
if "report" not in st.session_state:
    st.session_state.report = None

# Helper to get severity color matching Stitch AI design
def get_severity_color(severity):
    colors = {
        "Critical": "#ef4444", # Red
        "High": "#f97316",     # Orange
        "Medium": "#eab308",   # Yellow/Gold
        "Low": "#22c55e"       # Green
    }
    return colors.get(severity, "#cbd5e1")

# Helper to render clean HTML without markdown paragraph wrapper bugs
def render_html(html_str):
    clean_html = textwrap.dedent(html_str)
    # Remove empty lines and leading/trailing whitespace per line to avoid markdown code-block triggers
    clean_html = "\n".join([line.strip() for line in clean_html.split("\n") if line.strip()])
    st.markdown(clean_html, unsafe_allow_html=True)

import html as html_module

def _build_highlighted_page_html(page_text: str, page_findings_idx: list, resolved: dict) -> str:
    """Build page HTML with finding highlights injected as <mark> tags.

    page_findings_idx: list of (idx, Finding) tuples, where idx is the finding's
    stable position in report.findings (used as the fix-application id).
    resolved: dict of {idx: {"verified": bool, "explanation": str}} for findings
    that have already been applied to this page's text.
    """
    escaped = html_module.escape(page_text)

    # --- Unresolved findings: highlight the original flagged text, clickable ---
    active = [(idx, f) for idx, f in page_findings_idx if idx not in resolved and f.source_quote and f.source_quote.strip()]
    active.sort(key=lambda t: len(t[1].source_quote), reverse=True)

    for idx, f in active:
        quote = f.source_quote.strip()
        escaped_quote = html_module.escape(quote)
        sev_lower = f.severity.lower()
        mark_open = f'<mark class="issue-highlight severity-{sev_lower}" data-fid="{idx}" onclick="openAnnotation({idx})">'
        if escaped_quote in escaped:
            escaped = escaped.replace(escaped_quote, f'{mark_open}{escaped_quote}</mark>', 1)
        else:
            # Fuzzy fallback: match on the first 40 characters
            prefix = html_module.escape(quote[:40])
            start = escaped.find(prefix)
            if start != -1:
                end = min(start + len(escaped_quote), len(escaped))
                span = escaped[start:end]
                escaped = escaped[:start] + mark_open + span + "</mark>" + escaped[end:]

    # --- Resolved findings: highlight the now-applied correction text in green ---
    fixed = [(idx, f) for idx, f in page_findings_idx if idx in resolved and f.suggested_correction and f.suggested_correction.strip()]
    fixed.sort(key=lambda t: len(t[1].suggested_correction), reverse=True)

    for idx, f in fixed:
        corr = f.suggested_correction.strip()
        escaped_corr = html_module.escape(corr)
        if escaped_corr in escaped:
            verified = resolved[idx].get("verified", True)
            title = "Fix applied and verified compliant" if verified else "Fix applied (verification was inconclusive)"
            mark = f'<mark class="issue-highlight resolved" title="{title}">'
            escaped = escaped.replace(escaped_corr, f'{mark}{escaped_corr}</mark>', 1)

    return escaped


def _apply_finding_fix(report, resolved, idx, force=False):
    """Verify (unless forced) and persist a finding's suggested correction into
    the actual stored page text. Mutates `resolved` and `report.pages` in place."""
    finding = report.findings[idx]
    result = {"compliant": True, "explanation": ""}
    if not force:
        with st.spinner("Verifying correction against the guideline clause..."):
            result = verify_correction(
                source_quote=finding.source_quote or finding.violating_statement,
                suggested_correction=finding.suggested_correction,
                guideline_clause=finding.guideline_clause,
                category=finding.category,
            )

    if force or result.get("compliant", True):
        quote = (finding.source_quote or "").strip()
        target_page = finding.page_number or 1
        for pg in (report.pages or []):
            if pg["page_num"] == target_page and quote and quote in pg["text"]:
                pg["text"] = pg["text"].replace(quote, finding.suggested_correction, 1)
                break
        resolved[idx] = {
            "verified": result.get("compliant", True),
            "explanation": result.get("explanation", ""),
        }
        st.session_state.pop("pending_review", None)
    else:
        st.session_state["pending_review"] = {
            "fidx": idx,
            "explanation": result.get("explanation", "Verification did not confirm this correction resolves the issue."),
        }


def _render_document_view(report):
    """Render the interactive document view: issues highlighted inline in the
    real page text (click a highlight for a quick preview), plus a real,
    per-issue "Apply Fix" control that verifies the correction against the
    guideline clause and persists it into the actual stored document text.
    """
    resolved = st.session_state.setdefault("resolved_findings", {})

    pages = report.pages or []
    all_findings = report.findings

    if not pages:
        st.markdown("""
        <div class="no-issues-placeholder">
          <div class="icon">📄</div>
          <div>Document pages not available.</div>
          <div style="font-size:13px; margin-top:8px; color:#cbd5e1;">
            Run a new audit to enable the Document View.
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Group (idx, finding) tuples by page
    findings_by_page = {}
    for idx, f in enumerate(all_findings):
        pg = f.page_number or 1
        findings_by_page.setdefault(pg, []).append((idx, f))

    legend_html = """
    <div style="display:flex; gap:16px; align-items:center; margin-bottom:16px; flex-wrap:wrap;">
      <span style="font-size:12px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.5px;">Issue Severity:</span>
      <span style="background:rgba(239,68,68,0.18); color:#ef4444; font-size:12px; font-weight:600; padding:3px 10px; border-radius:4px; border-bottom:2px solid #ef4444;">● Critical</span>
      <span style="background:rgba(249,115,22,0.18); color:#f97316; font-size:12px; font-weight:600; padding:3px 10px; border-radius:4px; border-bottom:2px solid #f97316;">● High</span>
      <span style="background:rgba(234,179,8,0.18); color:#ca8a04; font-size:12px; font-weight:600; padding:3px 10px; border-radius:4px; border-bottom:2px solid #eab308;">● Medium</span>
      <span style="background:rgba(34,197,94,0.18); color:#22c55e; font-size:12px; font-weight:600; padding:3px 10px; border-radius:4px; border-bottom:2px solid #22c55e;">● Low</span>
      <span style="font-size:12px; color:#94a3b8;">— Highlights show where issues are. Expand an issue below a page to preview and apply its fix.</span>
    </div>
    """
    render_html(legend_html)

    severity_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    severity_icon = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}

    for page_data in pages:
        pg_num = page_data["page_num"]
        pg_text = page_data["text"]
        pg_findings = findings_by_page.get(pg_num, [])
        active = sorted(
            [(idx, f) for idx, f in pg_findings if idx not in resolved],
            key=lambda t: severity_order.get(t[1].severity, 4),
        )
        n_resolved = len([1 for idx, _ in pg_findings if idx in resolved])

        if active:
            issues_badge = f'<span class="doc-issues-count has-issues">{len(active)} issue{"s" if len(active) != 1 else ""}</span>'
        elif n_resolved > 0:
            issues_badge = f'<span class="doc-issues-count">✓ All fixed ({n_resolved})</span>'
        else:
            issues_badge = '<span class="doc-issues-count">✓ No issues</span>'

        page_body_html = _build_highlighted_page_html(pg_text, pg_findings, resolved)
        # Rough height estimate so the iframe doesn't clip or over-scroll
        est_height = 140 + 22 * (len(pg_text) // 80 + pg_text.count("\n") + 1)
        est_height = max(180, min(est_height, 900))

        page_html = f"""
<!DOCTYPE html><html><head><meta charset="utf-8"><style>
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 4px; background: transparent; font-family: -apple-system, 'Segoe UI', 'Outfit', sans-serif; }}
.doc-page-card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 28px 32px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); font-size: 14.5px; line-height: 1.7; color: #1e293b; white-space: pre-wrap; word-break: break-word; }}
.doc-page-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 18px; padding-bottom: 14px; border-bottom: 1px solid #f1f5f9; }}
.doc-page-number {{ font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }}
.doc-issues-count {{ font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 20px; background: #f1f5f9; color: #475569; }}
.doc-issues-count.has-issues {{ background: #fef3c7; color: #92400e; }}
mark.issue-highlight {{ border-radius: 3px; padding: 1px 2px; text-decoration-line: underline; text-decoration-style: wavy; text-underline-offset: 3px; font-weight: inherit; display: inline; }}
mark.issue-highlight.severity-critical {{ background-color: rgba(239, 68, 68, 0.18); text-decoration-color: #ef4444; color: inherit; }}
mark.issue-highlight.severity-high {{ background-color: rgba(249, 115, 22, 0.18); text-decoration-color: #f97316; color: inherit; }}
mark.issue-highlight.severity-medium {{ background-color: rgba(234, 179, 8, 0.18); text-decoration-color: #eab308; color: inherit; }}
mark.issue-highlight.severity-low {{ background-color: rgba(34, 197, 94, 0.18); text-decoration-color: #22c55e; color: inherit; }}
mark.issue-highlight.resolved {{ background-color: rgba(34, 197, 94, 0.14); text-decoration: none; }}
</style></head><body>
<div class="doc-page-card">
  <div class="doc-page-header">
    <span class="doc-page-number">Page {pg_num}</span>
    {issues_badge}
  </div>
  <div>{page_body_html}</div>
</div>
</body></html>
"""
        st.iframe(page_html, height=est_height)

        # Native, guaranteed-working preview + apply control for each active issue
        for idx, f in active:
            icon = severity_icon.get(f.severity, "⚪")
            with st.expander(f"{icon} {f.severity} — {f.violating_statement}", expanded=False):
                st.markdown(
                    f"<span class='sev-pill sev-pill-{f.severity.lower()}'>{f.severity.upper()}</span> "
                    f"&nbsp; **{f.guideline_clause}** &nbsp; · &nbsp; Confidence: {int(f.confidence*100)}%",
                    unsafe_allow_html=True,
                )
                if f.source_quote:
                    st.markdown(f"> {f.source_quote}")
                st.markdown(f"**Explanation:** {f.explanation}")
                st.success(f"**Suggested fix:** {f.suggested_correction}")

                pending = st.session_state.get("pending_review")
                if pending and pending["fidx"] == idx:
                    st.warning(
                        f"Verification flagged this correction as not clearly compliant: {pending['explanation']}"
                    )
                    if st.button("Apply Anyway", key=f"force_apply_{idx}"):
                        _apply_finding_fix(report, resolved, idx, force=True)
                        st.rerun()
                else:
                    if st.button("✓ Apply Fix to Document", key=f"apply_{idx}", type="primary"):
                        _apply_finding_fix(report, resolved, idx, force=False)
                        st.rerun()

        if n_resolved:
            fixed_labels = []
            for idx, f in pg_findings:
                if idx in resolved:
                    verified = resolved[idx].get("verified", True)
                    fixed_labels.append(f"{'✓' if verified else '⚠'} {f.violating_statement}")
            st.caption("Fixed on this page: " + " · ".join(fixed_labels))



# Custom CSS matching Stitch AI's premium, state-of-the-art design
css_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Global overrides */
.stApp {
    background-color: #f8fafc;
    font-family: 'Outfit', sans-serif !important;
}

/* Hide default streamlit header/decorations */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
div[data-testid="stDecoration"] {display: none;}

/* Adjust block container padding for full width navbar and alignment */
[data-testid="stAppViewBlockContainer"], .main .block-container {
    padding-top: 100px !important;
    padding-left: 48px !important;
    padding-right: 48px !important;
    max-width: 100% !important;
}

/* Custom top navbar styling */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 14px 40px;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 70px;
    z-index: 1000;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
}
.navbar-left {
    display: flex;
    align-items: center;
    gap: 12px;
}
.navbar-logo-link {
    text-decoration: none !important;
    display: flex;
    align-items: center;
    gap: 12px;
    cursor: pointer;
}
.navbar-logo-icon {
    background-color: #0b57d0;
    color: white;
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 18px;
    font-family: 'Outfit', sans-serif;
}
.navbar-logo-text {
    font-size: 20px;
    font-weight: 700;
    color: #0f172a;
    transition: color 0.2s ease;
}
.navbar-logo-link:hover .navbar-logo-text {
    color: #0b57d0;
}
.navbar-right {
    display: flex;
    align-items: center;
    gap: 20px;
}
.navbar-doc-title {
    font-size: 14px;
    color: #475569;
    font-weight: 500;
    max-width: 300px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin-right: 180px; /* Leave space for absolute-positioned streamlit download button */
}
.navbar-pill {
    background-color: #f1f5f9;
    color: #475569;
    font-size: 13px;
    font-weight: 600;
    padding: 6px 14px;
    border-radius: 9999px;
}
.navbar-link {
    font-size: 14px;
    color: #64748b;
    text-decoration: none;
    font-weight: 500;
}

/* Home Screen Layout */
.home-container {
    max-width: 960px;
    margin: 40px auto 20px auto;
    text-align: center;
}
.home-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background-color: #e0f2fe;
    color: #0369a1;
    font-size: 13px;
    font-weight: 600;
    padding: 6px 16px;
    border-radius: 9999px;
    margin-bottom: 28px;
}
.home-title {
    font-size: 44px;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.15;
    margin-bottom: 20px;
    letter-spacing: -0.8px;
}
.home-subtitle {
    font-size: 18px;
    color: #475569;
    max-width: 720px;
    margin: 0 auto 48px auto;
    line-height: 1.6;
}

/* Home Features Grid Layout */
.features-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 24px;
    margin: 48px auto 0 auto;
    max-width: 900px;
}
.feature-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    text-align: left;
    box-shadow: 0 1px 3px rgba(0,0,0,0.01);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
}
.feature-icon {
    font-size: 28px;
    background-color: #f0f4f9;
    width: 48px;
    height: 48px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 18px;
}
.feature-title {
    font-size: 16px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 8px;
}
.feature-desc {
    font-size: 13.5px;
    color: #475569;
    line-height: 1.5;
}

.home-footer-features {
    display: flex;
    justify-content: center;
    gap: 40px;
    margin-top: 64px;
    border-top: 1px solid #e2e8f0;
    padding-top: 32px;
}
.home-footer-feature {
    font-size: 13px;
    font-weight: 700;
    color: #64748b;
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: 0.5px;
}

/* Uploader components */
.uploader-label {
    font-size: 16px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 12px;
    margin-left: 4px;
}

/* Native Sidebar Styling integration */
[data-testid="stSidebar"] {
    background-color: #f0f4f9 !important;
    border-right: 1px solid #e2e8f0 !important;
    top: 70px !important;
    height: calc(100vh - 70px) !important;
    width: 280px !important;
}
[data-testid="stSidebarContent"] {
    background-color: #f0f4f9 !important;
}
[data-testid="stSidebarUserContent"] {
    padding-top: 24px !important;
    padding-left: 20px !important;
    padding-right: 20px !important;
}
[data-testid="stSidebarCollapseButton"] {
    display: none !important;
}

/* Sidebar inner components */
.sidebar-profile {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 32px;
}
.sidebar-avatar {
    background-color: #0b57d0;
    color: white;
    font-weight: 700;
    font-size: 16px;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
}
.sidebar-profile-info {
    display: flex;
    flex-direction: column;
}
.sidebar-profile-name {
    font-size: 15px;
    font-weight: 600;
    color: #0f172a;
}
.sidebar-profile-title {
    font-size: 11px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 2px;
}
.sidebar-menu {
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.sidebar-item {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14.5px;
    font-weight: 500;
    color: #475569;
    padding: 10px 16px;
    border-radius: 8px;
    cursor: default;
    transition: all 0.2s ease;
}
.sidebar-item:hover {
    background-color: rgba(203, 213, 225, 0.4);
    color: #0f172a;
}
.sidebar-item.active {
    background-color: #cbd5e1;
    color: #0f172a;
    font-weight: 600;
}
.sidebar-footer-item {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14.5px;
    font-weight: 500;
    color: #475569;
    cursor: default;
    padding: 8px 12px;
    border-radius: 6px;
}

/* Dashboard Header */
.dashboard-header {
    margin-bottom: 24px;
}
.dashboard-title {
    font-size: 32px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 4px;
}
.dashboard-subtitle {
    font-size: 16px;
    color: #64748b;
}

/* Executive Summary styling */
.summary-card {
    background: linear-gradient(135deg, #0f2b6b 0%, #0b57d0 100%);
    border-radius: 16px;
    padding: 24px;
    color: #ffffff;
    margin-bottom: 32px;
    box-shadow: 0 4px 15px rgba(11, 87, 208, 0.15);
}
.summary-title {
    font-size: 15px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
    opacity: 0.9;
}
.summary-body {
    font-size: 15.5px;
    line-height: 1.55;
    font-weight: 400;
}

/* Metric Cards Layout */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 24px;
    margin-bottom: 32px;
}
.metric-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    min-height: 120px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.01);
    transition: transform 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
}
.metric-card-left {
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-card-label {
    font-size: 11px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}
.metric-card-value {
    font-size: 28px;
    font-weight: 700;
    color: #0f172a;
}
.metric-card-subvalue {
    font-size: 13px;
    color: #64748b;
    margin-top: 4px;
}
.metric-card-pill {
    background-color: #ffedd5;
    color: #ea580c;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 9999px;
    margin-left: 10px;
    display: inline-block;
}

/* Analytics Row Layout */
.analytics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-bottom: 32px;
}
.analytics-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.01);
}
.analytics-card-title {
    font-size: 16px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 24px;
}

/* Issues by Category items */
.category-item {
    margin-bottom: 16px;
}
.category-header {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    margin-bottom: 6px;
    letter-spacing: 0.5px;
}
.category-bar-bg {
    background-color: #f1f5f9;
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
}
.category-bar-fill {
    background-color: #0b57d0;
    height: 100%;
    border-radius: 4px;
}

/* Filter Card Wrap */
.filter-wrapper {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 16px 24px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.01);
}

/* Findings Card Layout */
.finding-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    position: relative;
    box-shadow: 0 1px 3px rgba(0,0,0,0.01);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.finding-card:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
}
.finding-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
}
.finding-card-badges {
    display: flex;
    gap: 8px;
    align-items: center;
}
.badge-severity {
    font-size: 11px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 6px;
    text-transform: uppercase;
    font-family: 'Outfit', sans-serif;
}
.badge-severity.critical { background-color: #fee2e2; color: #ef4444; }
.badge-severity.high { background-color: #ffedd5; color: #f97316; }
.badge-severity.medium { background-color: #fef9c3; color: #ca8a04; }
.badge-severity.low { background-color: #dcfce7; color: #22c55e; }

.badge-clause {
    background-color: #f1f5f9;
    color: #475569;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 6px;
}
.finding-card-confidence {
    font-size: 11px;
    font-weight: 700;
    color: #64748b;
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: 0.5px;
}
.confidence-bar-bg {
    background-color: #e2e8f0;
    width: 60px;
    height: 6px;
    border-radius: 3px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 3px;
}
.confidence-bar-fill.high { background-color: #22c55e; }
.confidence-bar-fill.medium { background-color: #0b57d0; }
.confidence-bar-fill.low { background-color: #cbd5e1; }

.finding-card-title {
    font-size: 18px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 8px;
    line-height: 1.3;
}
.finding-card-desc {
    font-size: 14.5px;
    color: #475569;
    line-height: 1.5;
    margin-bottom: 16px;
}
.suggested-fix-box {
    background-color: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 14px 18px;
    display: flex;
    gap: 12px;
    align-items: flex-start;
}
.suggested-fix-icon {
    font-size: 18px;
    color: #22c55e;
    margin-top: 1px;
}
.suggested-fix-text {
    font-size: 14px;
    color: #166534;
    line-height: 1.5;
}

/* =================== DOCUMENT VIEW STYLES =================== */
.doc-view-container {
    display: flex;
    gap: 24px;
    position: relative;
    align-items: flex-start;
}
.doc-pages-panel {
    flex: 1 1 65%;
    min-width: 0;
}
.doc-annotation-panel {
    flex: 0 0 340px;
    position: sticky;
    top: 90px;
    max-height: calc(100vh - 110px);
    overflow-y: auto;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 0;
    box-shadow: 0 4px 24px rgba(11,87,208,0.08);
    display: none;
    flex-direction: column;
}
.doc-annotation-panel.open {
    display: flex;
}
.doc-annotation-header {
    padding: 20px 20px 16px 20px;
    border-bottom: 1px solid #e2e8f0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.doc-annotation-title {
    font-size: 15px;
    font-weight: 700;
    color: #0f172a;
}
.doc-annotation-close {
    cursor: pointer;
    color: #94a3b8;
    font-size: 20px;
    line-height: 1;
    transition: color 0.2s;
    background: none;
    border: none;
    padding: 0 4px;
}
.doc-annotation-close:hover {
    color: #0f172a;
}
.doc-annotation-body {
    padding: 20px;
    flex: 1;
    overflow-y: auto;
}
.doc-annotation-severity {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 14px;
}
.doc-annotation-clause {
    font-size: 12px;
    font-weight: 600;
    color: #475569;
    background: #f1f5f9;
    padding: 4px 10px;
    border-radius: 6px;
}
.doc-annotation-section {
    margin-bottom: 16px;
}
.doc-annotation-section-label {
    font-size: 11px;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.doc-annotation-section-text {
    font-size: 14px;
    color: #334155;
    line-height: 1.55;
}
.doc-annotation-quote {
    background: #f8fafc;
    border-left: 3px solid #cbd5e1;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 13px;
    color: #475569;
    font-style: italic;
    line-height: 1.5;
    margin-bottom: 16px;
}
.doc-correction-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 16px;
}
.doc-correction-label {
    font-size: 11px;
    font-weight: 700;
    color: #166534;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.doc-correction-text {
    font-size: 13.5px;
    color: #166534;
    line-height: 1.5;
    font-family: 'Outfit', sans-serif;
}
.doc-apply-btn {
    width: 100%;
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-size: 15px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: 'Outfit', sans-serif;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin-bottom: 8px;
}
.doc-apply-btn:hover {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
    box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3);
    transform: translateY(-1px);
}
.doc-apply-btn:active {
    transform: translateY(0);
}
.doc-dismiss-btn {
    width: 100%;
    background: transparent;
    color: #64748b;
    border: 1.5px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    font-family: 'Outfit', sans-serif;
}
.doc-dismiss-btn:hover {
    background: #f8fafc;
    border-color: #cbd5e1;
    color: #334155;
}
/* Document page cards */
.doc-page-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    font-size: 14.5px;
    line-height: 1.7;
    color: #1e293b;
    font-family: 'Outfit', sans-serif;
    white-space: pre-wrap;
    word-break: break-word;
}
.doc-page-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    padding-bottom: 14px;
    border-bottom: 1px solid #f1f5f9;
}
.doc-page-number {
    font-size: 11px;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.doc-issues-count {
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    background: #f1f5f9;
    color: #475569;
}
.doc-issues-count.has-issues {
    background: #fef3c7;
    color: #92400e;
}
/* Issue highlight marks */
mark.issue-highlight {
    border-radius: 3px;
    padding: 1px 2px;
    cursor: pointer;
    transition: filter 0.15s ease, box-shadow 0.15s ease;
    text-decoration-line: underline;
    text-decoration-style: wavy;
    text-underline-offset: 3px;
    font-weight: inherit;
    display: inline;
}
mark.issue-highlight:hover {
    filter: brightness(0.92);
    box-shadow: 0 2px 8px rgba(0,0,0,0.12);
}
mark.issue-highlight.severity-critical {
    background-color: rgba(239, 68, 68, 0.18);
    text-decoration-color: #ef4444;
    color: inherit;
}
mark.issue-highlight.severity-high {
    background-color: rgba(249, 115, 22, 0.18);
    text-decoration-color: #f97316;
    color: inherit;
}
mark.issue-highlight.severity-medium {
    background-color: rgba(234, 179, 8, 0.18);
    text-decoration-color: #eab308;
    color: inherit;
}
mark.issue-highlight.severity-low {
    background-color: rgba(34, 197, 94, 0.18);
    text-decoration-color: #22c55e;
    color: inherit;
}
mark.issue-highlight.resolved {
    background-color: rgba(34, 197, 94, 0.10);
    text-decoration: none;
    cursor: default;
}
/* Tab bar for dashboard */
.tab-bar {
    display: flex;
    gap: 4px;
    background: #f1f5f9;
    border-radius: 12px;
    padding: 4px;
    margin-bottom: 28px;
    width: fit-content;
}
.tab-btn {
    padding: 8px 20px;
    border-radius: 9px;
    font-size: 14px;
    font-weight: 600;
    border: none;
    cursor: pointer;
    transition: all 0.2s ease;
    background: transparent;
    color: #64748b;
    font-family: 'Outfit', sans-serif;
}
.tab-btn.active {
    background: #ffffff;
    color: #0f172a;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.tab-btn:hover:not(.active) {
    color: #334155;
    background: rgba(255,255,255,0.6);
}
/* Severity pill in annotation panel */
.sev-pill-critical { background: #fee2e2; color: #ef4444; }
.sev-pill-high { background: #ffedd5; color: #f97316; }
.sev-pill-medium { background: #fef9c3; color: #ca8a04; }
.sev-pill-low { background: #dcfce7; color: #22c55e; }
.sev-pill {
    font-size: 11px;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 6px;
    text-transform: uppercase;
    font-family: 'Outfit', sans-serif;
}
/* No issues placeholder */
.no-issues-placeholder {
    text-align: center;
    padding: 60px 20px;
    color: #94a3b8;
    font-size: 15px;
}
.no-issues-placeholder .icon { font-size: 48px; margin-bottom: 16px; }
/* Custom styling for Streamlit widgets */
div.stDownloadButton > button {
    background-color: #ffffff !important;
    color: #0b57d0 !important;
    border: 1.5px solid #0b57d0 !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    transition: all 0.2s ease !important;
    font-size: 14px !important;
    box-shadow: none !important;
}
div.stDownloadButton > button:hover {
    background-color: #0b57d0 !important;
    color: #ffffff !important;
}
div.stButton > button {
    background-color: #0b57d0 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    font-size: 16px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
}
div.stButton > button:hover {
    background-color: #0045b3 !important;
    box-shadow: 0 4px 12px rgba(11, 87, 208, 0.15) !important;
}
div[data-testid="stFileUploader"] {
    background-color: #ffffff !important;
    border: 2px dashed #cbd5e1 !important;
    border-radius: 16px !important;
    padding: 24px !important;
    min-height: 204px !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
}
div[data-testid="stFileUploader"] section {
    background-color: transparent !important;
    padding: 0 !important;
    border: none !important;
    width: 100% !important;
}
div[data-testid="stFileUploaderDropzone"] {
    border: none !important;
    background-color: transparent !important;
    padding: 0 !important;
}
div[data-testid="stSidebar"] button {
    background-color: transparent !important;
    color: #475569 !important;
    border: none !important;
    text-align: left !important;
    justify-content: flex-start !important;
    font-size: 14.5px !important;
    font-weight: 500 !important;
    padding: 10px 16px !important;
    box-shadow: none !important;
    width: 100% !important;
    border-radius: 8px !important;
    margin-bottom: 6px !important;
}
div[data-testid="stSidebar"] button:hover {
    background-color: rgba(203, 213, 225, 0.4) !important;
    color: #0f172a !important;
}
</style>
"""
st.markdown(css_style, unsafe_allow_html=True)

# ----------------- SCREEN 1: HOME SCREEN -----------------
if st.session_state.current_screen == "home":
    # Ensure sidebar is collapsed on Home screen
    st.markdown("""
    <style>
    div[data-testid="stSidebar"] {display: none;}
    [data-testid="stAppViewBlockContainer"], .main .block-container {
        padding-left: 48px !important;
        padding-right: 48px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Home navbar with clickable logo that links to /?screen=home
    render_html("""
    <div class="navbar">
      <div class="navbar-left">
        <a href="/?screen=home" target="_self" class="navbar-logo-link">
          <div class="navbar-logo-icon">🛡️</div>
          <div class="navbar-logo-text">The Auditor</div>
        </a>
      </div>
      <div class="navbar-right">
        <span class="navbar-pill">EU MDR 2017/745</span>
      </div>
    </div>
    """)

    # Home Content
    render_html("""
    <div class="home-container">
      <div class="home-badge">
        <span>✓</span> Regulatory AI Engine v4.2
      </div>
      <h1 class="home-title">AI-powered medical device<br>compliance audit</h1>
      <p class="home-subtitle">Upload your Clinical Evaluation Report and we check it against EU MDR regulations in seconds. Reduce audit preparation time by up to 80%.</p>
    </div>
    """)

    # Centered File Uploader
    st.markdown('<div class="uploader-label" style="text-align: center; font-size: 18px; margin-bottom: 16px;">Clinical Evaluation Report (PDF)</div>', unsafe_allow_html=True)
    c_upload = st.columns([1, 2, 1])
    with c_upload[1]:
        cer_file = st.file_uploader("Clinical Evaluation Report (PDF)", type="pdf", label_visibility="collapsed", key="cer_file")

    # Options and Run Audit
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    c_pages = st.columns([1, 2, 1])
    with c_pages[1]:
        max_pages_selected = st.slider(
            "Pages to audit", min_value=5, max_value=10, value=8, step=1,
            help="Runs locally via Ollama, so this is limited to keep audit time reasonable while we validate the flow.",
        )
    c_check = st.columns([1, 2, 1])
    with c_check[1]:
        run_real = st.button("Run Compliance Audit", use_container_width=True)

    # Platform Capabilities Grid under uploader to fill space nicely
    render_html("""
    <div class="features-grid">
      <div class="feature-card">
        <div class="feature-icon">🤖</div>
        <div class="feature-title">LLM Regulatory Agent</div>
        <div class="feature-desc">Powered by a local Ollama model to verify protocol wording and identify hidden regulatory gaps &mdash; no API rate limits.</div>
      </div>
      <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <div class="feature-title">Semantic Retrieval</div>
        <div class="feature-desc">Uses semantic search to cross-reference report contents with the exact matching EU MDR guidelines.</div>
      </div>
      <div class="feature-card">
        <div class="feature-icon">📊</div>
        <div class="feature-title">Readiness Scoring</div>
        <div class="feature-desc">Provides a quantified readiness rating based on the severity and category density of identified violations.</div>
      </div>
    </div>
    """)

    if run_real:
        if cer_file:
            def save(f):
                t = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                t.write(f.read())
                t.close()
                return t.name
            
            # Placeholders for dynamic progress feedback
            status_placeholder = st.empty()
            progress_bar = st.progress(0.0)
            
            def update_status(title, estimate, steps_list, progress_val):
                progress_bar.progress(progress_val)
                steps_html = ""
                for step, state in steps_list:
                    if state == "active":
                        steps_html += f"<li style='color:#0b57d0; font-weight:bold; margin-bottom:8px;'>⏳ {step}</li>"
                    elif state == "done":
                        steps_html += f"<li style='color:#22c55e; margin-bottom:8px;'>✓ {step}</li>"
                    else:
                        steps_html += f"<li style='color:#94a3b8; margin-bottom:8px;'>○ {step}</li>"
                
                status_box = f"""
                <div style="background-color:#ffffff; border:1px solid #e2e8f0; border-radius:12px; padding:24px; margin-bottom:24px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05);">
                  <h3 style="margin-top:0; color:#0f172a; font-family:'Outfit', sans-serif;">{title}</h3>
                  <p style="color:#64748b; font-size:14px; margin-bottom:16px;">Estimated total time: <b>{estimate}</b></p>
                  <ul style="padding-left:0; list-style-type:none; font-family:'Outfit', sans-serif; font-size:15px; margin:0;">
                    {steps_html}
                  </ul>
                </div>
                """
                status_placeholder.markdown(textwrap.dedent(status_box), unsafe_allow_html=True)

            audit_steps = [
                ("Extracting and chunking Clinical Evaluation Report", "active"),
                ("Searching MDR regulatory guidelines", "pending"),
                ("Running compliance audit with AI models", "pending")
            ]
            time_estimate = f"~{max_pages_selected * 15}-{max_pages_selected * 30} seconds (local model)"
            update_status("Starting Compliance Audit...", time_estimate, audit_steps, 0.1)

            # Native rotating spinner shown the instant the button is clicked,
            # for feedback while the (now-cached, module-level-imported) audit
            # pipeline warms up and runs.
            spinner_cm = st.spinner("Running compliance audit — this can take a few minutes on a local model...")
            spinner_cm.__enter__()

            try:
                # Step 1: Ingest
                cer_path = save(cer_file)
                cer_text = extract_text(cer_path)
                cer_chunks = chunk_text(cer_text)
                
                if len(cer_chunks) == 0:
                    status_placeholder.empty()
                    progress_bar.empty()
                    spinner_cm.__exit__(None, None, None)
                    st.error("Error: No text could be extracted from the uploaded PDF. Please make sure the PDF contains selectable text (not scanned images without OCR).")
                    st.stop()
                
                audit_steps[0] = ("Extracting and chunking Clinical Evaluation Report", "done")
                audit_steps[1] = ("Searching MDR regulatory guidelines", "active")
                update_status("Analyzing guidelines...", time_estimate, audit_steps, 0.3)
                
                # Step 2: Retrieve from system default guideline
                guideline_path = os.path.join(BASE_DIR, "data", "guideline.pdf")
                if not os.path.exists(guideline_path):
                    status_placeholder.empty()
                    progress_bar.empty()
                    spinner_cm.__exit__(None, None, None)
                    st.error("Error: Default guideline.pdf not found in data/ directory.")
                    st.stop()
                    
                retriever = get_guideline_retriever(guideline_path, os.path.getmtime(guideline_path))

                audit_steps[1] = ("Searching MDR regulatory guidelines", "done")
                audit_steps[2] = ("Running compliance audit with AI models", "active")
                update_status("Executing compliance checks...", time_estimate, audit_steps, 0.5)

                # Extract page-level text for Document View: this scans past any
                # cover/TOC/abbreviations pages and returns max_pages_selected
                # pages of real content, starting wherever it actually begins.
                doc_pages, skipped_pages = extract_pages(cer_path, max_pages=max_pages_selected)

                audit_pages = doc_pages
                if not audit_pages:
                    # Fallback: use existing chunks mapped to page 1
                    audit_pages = [{'page_num': i+1, 'text': c} for i, c in enumerate(cer_chunks[:5])]

                findings = []
                for idx, page_data in enumerate(audit_pages):
                    progress_val = 0.5 + 0.5 * ((idx + 1) / len(audit_pages))
                    sec = page_data['text']
                    pg = page_data['page_num']

                    audit_steps[2] = (f"Running compliance audit with AI models (page {idx+1}/{len(audit_pages)})...", "active")
                    update_status("Executing compliance checks...", time_estimate, audit_steps, progress_val)

                    clauses = "\n---\n".join(retriever.search(sec, k=3))
                    findings.extend(audit_section(sec, clauses, page_number=pg))

                # Drop "phantom" findings whose quote can't actually be located in that
                # page's text -- these can't be highlighted and just erode trust.
                page_text_by_num = {p["page_num"]: p["text"] for p in doc_pages}
                findings = [
                    f for f in findings
                    if quote_locatable(page_text_by_num.get(f.page_number or 1, ""), f.source_quote or "")
                ]

                audit_steps[2] = ("Running compliance audit with AI models", "done")
                update_status("Audit complete! Redirecting...", "0 seconds", audit_steps, 1.0)

                # Dynamic category counts for summary
                from collections import Counter
                category_counts = Counter()
                for f in findings:
                    category_counts[f.category] += 1

                st.session_state.report = AuditReport(
                    document_name=cer_file.name,
                    findings=findings,
                    readiness_score=readiness_score(findings),
                    summary={
                        "total_findings": len(findings),
                        "categories": dict(category_counts),
                        "skipped_pages": skipped_pages,
                    },
                    pages=doc_pages
                )
                # Clear any resolved/pending state from a previous document
                st.session_state.resolved_findings = {}
                st.session_state.pop("pending_review", None)
                st.session_state.current_screen = "dashboard"
                spinner_cm.__exit__(None, None, None)
                st.rerun()

            except Exception as e:
                spinner_cm.__exit__(None, None, None)
                status_placeholder.empty()
                progress_bar.empty()
                err_text = str(e).lower()
                model_name = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
                if "not found" in err_text and "model" in err_text:
                    hint = f"The model `{model_name}` hasn't been pulled yet. Run `ollama pull {model_name}` in a terminal, then try again."
                elif "connection" in err_text or "connect" in err_text:
                    hint = "Can't reach Ollama. Make sure the Ollama app is running (it should be listed in your system tray/menu bar), or run `ollama serve` in a terminal."
                else:
                    hint = f"Make sure the Ollama app is running and that `ollama pull {model_name}` has been run."
                st.error(f"Error during compliance audit: {e}\n\n{hint}")
        else:
            st.warning("Please upload a Clinical Evaluation Report (PDF) to run the audit.")

    # Sample data handler removed

    # Home Footer features
    render_html("""
    <div class="home-footer-features">
      <div class="home-footer-feature"><span>🔗</span> CLAUSE-GROUNDED</div>
      <div class="home-footer-feature"><span>📊</span> CONFIDENCE-SCORED</div>
      <div class="home-footer-feature"><span>📄</span> EXPORTABLE REPORT</div>
    </div>
    """)


# ----------------- SCREEN 2: DASHBOARD SCREEN -----------------
elif st.session_state.current_screen == "dashboard" and st.session_state.report:
    report = st.session_state.report

    # Dashboard Navbar with clickable logo that links to /?screen=home
    render_html(f"""
    <div class="navbar">
      <div class="navbar-left">
        <a href="/?screen=home" target="_self" class="navbar-logo-link">
          <div class="navbar-logo-icon">🛡️</div>
          <div class="navbar-logo-text">The Auditor</div>
        </a>
      </div>
      <div class="navbar-right">
        <span class="navbar-doc-title">{report.document_name}</span>
        <div id="dl-btn-container"></div>
      </div>
    </div>
    """)
    
    # We overlay the streamlit download button over the top-right navbar placeholder using CSS
    st.markdown("""
    <style>
    /* Position the streamlit download button absolute over the navbar right side */
    .stDownloadButton {
        position: fixed;
        top: 14px;
        right: 40px;
        z-index: 1001;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Actually render the download button (Streamlit handles PDF export with build_html)
    st.download_button("⬇ Download report", build_html(report), file_name="audit_report.html", mime="text/html")

    # Left Sidebar content
    with st.sidebar:
        sidebar_html = """
        <div class="sidebar-profile">
          <div class="sidebar-avatar">MA</div>
          <div class="sidebar-profile-info">
            <span class="sidebar-profile-name">Medical Auditor</span>
            <span class="sidebar-profile-title">Level 4 Certification</span>
          </div>
        </div>
        <div class="sidebar-menu">
          <div class="sidebar-item active"><span>📄</span> Document View</div>
        </div>
        """
        render_html(sidebar_html)
        
        # Clickable native button for Back to Home
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        if st.button("🏠 Back to Home", key="btn_sidebar_home", use_container_width=True):
            st.session_state.current_screen = "home"
            st.session_state.report = None
            st.rerun()

    # Dashboard Header + compact summary strip
    total_findings = len(report.findings)
    sev_counts = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
    for f in report.findings:
        sev_counts[f.severity] = sev_counts.get(f.severity, 0) + 1
    crit_high_count = sev_counts["Critical"] + sev_counts["High"]
    score = int(report.readiness_score)
    score_color = "#ef4444" if score < 50 else ("#f97316" if score < 75 else "#22c55e")

    skipped_pages = []
    if isinstance(report.summary, dict):
        skipped_pages = report.summary.get("skipped_pages", []) or []
    reviewed_page_nums = [p["page_num"] for p in (report.pages or [])]
    skipped_note = ""
    if skipped_pages:
        skipped_note = (
            f" &nbsp;·&nbsp; skipped PDF page{'s' if len(skipped_pages) != 1 else ''} "
            f"{', '.join(str(p) for p in skipped_pages)} (front matter)"
        )
    pages_note = ""
    if reviewed_page_nums:
        pages_note = f"PDF page{'s' if len(reviewed_page_nums) != 1 else ''} {min(reviewed_page_nums)}-{max(reviewed_page_nums)} reviewed"

    render_html(f"""
    <div class="dashboard-header">
      <h1 class="dashboard-title">Audit Results</h1>
      <p class="dashboard-subtitle">{report.document_name} &nbsp;·&nbsp; {pages_note}{skipped_note}</p>
    </div>
    <div style="display:flex; gap:20px; align-items:center; margin-bottom:24px; flex-wrap:wrap;">
      <div style="background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:14px 20px; display:flex; align-items:center; gap:12px;">
        <span style="font-size:22px; font-weight:800; color:{score_color};">{score}/100</span>
        <span style="font-size:12px; color:#64748b; font-weight:600;">READINESS<br>SCORE</span>
      </div>
      <div style="background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:14px 20px;">
        <span style="font-size:22px; font-weight:800; color:#0f172a;">{total_findings}</span>
        <span style="font-size:12px; color:#64748b; font-weight:600;"> total findings</span>
      </div>
      <div style="background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:14px 20px;">
        <span style="font-size:22px; font-weight:800; color:#ef4444;">{crit_high_count}</span>
        <span style="font-size:12px; color:#64748b; font-weight:600;"> critical + high</span>
      </div>
    </div>
    """)

    _render_document_view(report)
