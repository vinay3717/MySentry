import datetime
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.detection import PoisoningDetector
from data.schema_mapper import (
    suggest_schema,
    binarize_label,
    reshape_to_contract,
    validate_dataset_mappable,
)

# ── Singleton detector ────────────────────────────────────────────────────────
_detector_instance: Optional[PoisoningDetector] = None

def get_detector() -> PoisoningDetector:
    """Singleton getter for the backend PoisoningDetector engine."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PoisoningDetector()
    return _detector_instance


def generate_report(name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """Runs the real SENTRY PoisoningDetector on the dataset and returns the audit report."""
    detector = get_detector()
    return detector.generate_report(name, X, y)


# ── Theme CSS ─────────────────────────────────────────────────────────────────
_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">

<style>
/* ── Global typography ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: #e2e2eb !important;
}

/* ── App background ── */
.stApp {
    background-color: #111319 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #191b22 !important;
    border-right: 1px solid #2a2d36 !important;
}
[data-testid="stSidebar"] * { color: #bcc9cd !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e2e2eb !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #1e1f26 !important;
    border: 1px solid #2a2d36 !important;
    border-radius: 6px !important;
    padding: 0.9rem 1.1rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #bcc9cd !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: #4cd7f6 !important;
    letter-spacing: -0.03em !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: #0c0e14 !important;
    border: 1px solid #2a2d36 !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── Info / alert banners ── */
[data-testid="stAlert"] {
    background: #1e1f26 !important;
    border-left: 3px solid #4cd7f6 !important;
    border-radius: 4px !important;
    font-size: 0.875rem !important;
    color: #e2e2eb !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: #191b22 !important;
    border-bottom: 1px solid #2a2d36 !important;
    border-radius: 4px 4px 0 0 !important;
    gap: 0 !important;
}
[data-baseweb="tab"] {
    font-size: 0.75rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #869397 !important;
    padding: 0.5rem 1rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #4cd7f6 !important;
    border-bottom: 2px solid #4cd7f6 !important;
    background: #1e1f26 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #1e1f26 !important;
    border: 1px solid #2a2d36 !important;
    border-radius: 4px !important;
    color: #4cd7f6 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    font-family: 'Inter', sans-serif !important;
    transition: background 0.15s, border-color 0.15s;
}
.stButton > button:hover {
    background: #282a30 !important;
    border-color: #4cd7f6 !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: #0c0e14 !important;
    border: 1px solid #4cd7f6 !important;
    color: #4cd7f6 !important;
    border-radius: 4px !important;
    font-size: 0.78rem !important;
}

/* ── Page title ── */
h1 {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: #e2e2eb !important;
}

/* ── Section headings ── */
h2, h3 {
    font-family: 'Inter', sans-serif !important;
    color: #e2e2eb !important;
}

/* ── Verdict badge helper classes (used in custom HTML) ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-clean    { background: rgba(79,219,200,0.15); color: #4fdbc8; border: 1px solid #4fdbc8; }
.badge-minor    { background: rgba(76,215,246,0.12); color: #4cd7f6; border: 1px solid #4cd7f6; }
.badge-sig      { background: rgba(255,180,171,0.12); color: #ffb4ab; border: 1px solid #ffb4ab; }
.badge-critical { background: rgba(147,0,10,0.25);   color: #ffb4ab; border: 1px solid #93000a; }

/* ── Recommendation banner ── */
.rec-banner {
    background: #1e1f26;
    border-left: 3px solid #4cd7f6;
    border-radius: 0 4px 4px 0;
    padding: 0.75rem 1rem;
    margin: 0.75rem 0 1rem 0;
    font-size: 0.875rem;
    color: #e2e2eb;
}
.rec-banner .rec-label {
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4cd7f6;
    margin-bottom: 0.25rem;
}
.rec-banner.rec-danger {
    border-left-color: #ffb4ab;
    background: rgba(147,0,10,0.12);
}
.rec-banner.rec-danger .rec-label { color: #ffb4ab; }
</style>
"""

def _inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _score_band(score: float) -> str:
    """Returns clean/minor/sig/critical band for a given poisoning score."""
    if score < 5:   return "clean"
    if score < 20:  return "minor"
    if score < 50:  return "sig"
    return "critical"

def get_verdict_badge(score: float) -> str:
    """Plain-text verdict label used in st.metric and report text."""
    band = _score_band(score)
    return {"clean": "✅ CLEAN", "minor": "⚠️ MINOR ANOMALIES",
            "sig": "🚨 SIGNIFICANT", "critical": "🔴 CRITICAL"}[band]

def _verdict_html(score: float) -> str:
    band = _score_band(score)
    labels = {"clean": "✅ CLEAN", "minor": "⚠️ MINOR ANOMALIES",
               "sig": "🚨 SIGNIFICANT", "critical": "🔴 CRITICAL"}
    return f'<span class="badge badge-{band}">{labels[band]}</span>'

def _recommendation_banner(report: Dict[str, Any]) -> None:
    score = report.get('poisoning_score', 0.0)
    danger = score >= 20
    cls = "rec-banner rec-danger" if danger else "rec-banner"
    label = "⚠ THREAT DETECTED — RECOMMENDATION" if danger else "✓ RECOMMENDATION"
    text = report.get('recommendation', 'N/A')
    st.markdown(
        f'<div class="{cls}"><div class="rec-label">{label}</div>{text}</div>',
        unsafe_allow_html=True
    )


# ── Charts ────────────────────────────────────────────────────────────────────
def create_comparison_chart(report_clean: Dict[str, Any], report_poison: Dict[str, Any]) -> go.Figure:
    """Grouped bar chart comparing clean vs poisoned datasets across all three scores."""
    categories = ['Poisoning Score', 'Anomaly Score', 'Label Flip Score']
    clean_values  = [report_clean.get('poisoning_score', 0.0),
                     report_clean.get('anomaly_score', 0.0),
                     report_clean.get('label_flip_score', 0.0)]
    poison_values = [report_poison.get('poisoning_score', 0.0),
                     report_poison.get('anomaly_score', 0.0),
                     report_poison.get('label_flip_score', 0.0)]

    fig = go.Figure(data=[
        go.Bar(name='Clean Dataset',    x=categories, y=clean_values,
               marker_color='rgba(79,219,200,0.80)',
               text=[f"{v:.1f}%" for v in clean_values], textposition='auto',
               textfont=dict(family='JetBrains Mono', size=12, color='#0c0e14')),
        go.Bar(name='Poisoned Dataset', x=categories, y=poison_values,
               marker_color='rgba(255,180,171,0.80)',
               text=[f"{v:.1f}%" for v in poison_values], textposition='auto',
               textfont=dict(family='JetBrains Mono', size=12, color='#0c0e14')),
    ])
    fig.update_layout(
        barmode='group',
        yaxis_title="Detection Score (%)",
        yaxis=dict(range=[0, 100], gridcolor='#1e1f26', zerolinecolor='#2a2d36',
                   tickfont=dict(family='JetBrains Mono', size=11, color='#bcc9cd')),
        xaxis=dict(tickfont=dict(family='Inter', size=12, color='#bcc9cd')),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(family='Inter', size=12, color='#e2e2eb'),
                    bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='#1e1f26',
        plot_bgcolor='#1e1f26',
        margin=dict(l=40, r=40, t=50, b=40),
        height=380,
        font=dict(family='Inter', color='#e2e2eb'),
    )
    return fig

def create_single_dataset_chart(report: Dict[str, Any]) -> go.Figure:
    """Bar chart showing three detection scores for a single uploaded dataset."""
    categories = ['Poisoning Score', 'Anomaly Score', 'Label Flip Score']
    values = [
        report.get('poisoning_score', 0.0),
        report.get('anomaly_score', 0.0),
        report.get('label_flip_score', 0.0),
    ]
    # Color each bar individually: green if < 20, red if >= 20
    bar_colors = [
        'rgba(79,219,200,0.80)' if v < 20 else 'rgba(255,180,171,0.80)'
        for v in values
    ]
    dataset_name = report.get('dataset', 'Dataset')
    fig = go.Figure(data=[
        go.Bar(
            name=dataset_name,
            x=categories,
            y=values,
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in values],
            textposition='auto',
            textfont=dict(family='JetBrains Mono', size=12, color='#0c0e14'),
        )
    ])
    fig.update_layout(
        yaxis_title="Detection Score (%)",
        yaxis=dict(range=[0, 100], gridcolor='#1e1f26', zerolinecolor='#2a2d36',
                   tickfont=dict(family='JetBrains Mono', size=11, color='#bcc9cd')),
        xaxis=dict(tickfont=dict(family='Inter', size=12, color='#bcc9cd')),
        showlegend=False,
        paper_bgcolor='#1e1f26',
        plot_bgcolor='#1e1f26',
        margin=dict(l=40, r=40, t=50, b=40),
        height=340,
        font=dict(family='Inter', color='#e2e2eb'),
    )
    return fig

def generate_text_report(report: Dict[str, Any]) -> str:
    verdict = get_verdict_badge(report.get('poisoning_score', 0.0))
    indices_str = ", ".join(map(str, report.get('suspicious_indices', []))) or "None"
    return f"""======================================================================
SENTRY: SYNTHETIC DATA POISONING DETECTION AUDIT REPORT
======================================================================
Generated at: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Dataset Name: {report.get('dataset', 'Unknown')}
Total Samples Analyzed: {report.get('total_samples', 0)}

----------------------------------------------------------------------
DETECTION METRICS SUMMARY:
----------------------------------------------------------------------
- Overall Poisoning Score:          {report.get('poisoning_score', 0.0):.1f}%
- Feature Anomaly / Outlier Score:  {report.get('anomaly_score', 0.0):.1f}%
- Label-Flipping Mismatch Score:    {report.get('label_flip_score', 0.0):.1f}%
- Suspicious Samples Flagged:       {report.get('suspicious_samples', 0)}
- Dataset Security Verdict:         {verdict}

----------------------------------------------------------------------
OFFICIAL VERDICT & RECOMMENDATION:
----------------------------------------------------------------------
{report.get('recommendation', 'N/A')}

----------------------------------------------------------------------
FLAGGED SUSPICIOUS ROW INDICES:
----------------------------------------------------------------------
{indices_str}

======================================================================
End of SENTRY Audit Report — Sentry Poisoning Detection Engine
======================================================================
"""


# ── Dashboard renderer ────────────────────────────────────────────────────────
def render_dataset_dashboard(dataset_name: str, df: pd.DataFrame,
                              report: Dict[str, Any], show_benchmark: bool = True):
    score = report.get('poisoning_score', 0.0)

    # Header row: dataset name + verdict badge
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.5rem;">'
        f'<span style="font-size:1rem;font-weight:600;color:#e2e2eb;font-family:Inter,sans-serif;">'
        f'📊 {dataset_name}</span>'
        f'{_verdict_html(score)}</div>',
        unsafe_allow_html=True
    )

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Poisoning Score",  f"{report.get('poisoning_score', 0.0):.1f}%")
    col2.metric("Anomaly Score",    f"{report.get('anomaly_score', 0.0):.1f}%")
    col3.metric("Label-Flip Score", f"{report.get('label_flip_score', 0.0):.1f}%")
    col4.metric("Suspicious / Total",
                f"{report.get('suspicious_samples', 0)} / {report.get('total_samples', 0)}")

    # Recommendation banner
    _recommendation_banner(report)

    # Download
    report_txt = generate_text_report(report)
    clean_filename = f"sentry_report_{os.path.splitext(dataset_name)[0]}.txt"
    st.download_button(
        label="📥 Download Audit Report (.txt)",
        data=report_txt, file_name=clean_filename, mime="text/plain",
        key=f"dl_{dataset_name}"
    )

    # Tabs
    tab_overview, tab_label_flip, tab_anomaly, tab_inspector = st.tabs([
        "📊 Overview & Comparison",
        "🏷️ Label-Flip Analysis",
        "🌲 Anomaly / Outlier Analysis",
        "🔍 Suspicious Samples Inspector"
    ])

    clean_csv_path   = os.path.join("data", "clean_dataset.csv")
    poisoned_csv_path = os.path.join("data", "poisoned_dataset.csv")
    demo_ready = os.path.exists(clean_csv_path) and os.path.exists(poisoned_csv_path)

    with tab_overview:
        if show_benchmark:
            chart_label = "Detection Benchmark — Clean vs. Poisoned"
        else:
            chart_label = f"Detection Scores — {dataset_name}"
        st.markdown(
            f'<p style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;'
            f'color:#bcc9cd;margin-bottom:0.5rem;">{chart_label}</p>',
            unsafe_allow_html=True
        )
        if show_benchmark and demo_ready:
            # Demo mode: compare synthetic clean vs poisoned datasets
            df_clean  = pd.read_csv(clean_csv_path)
            df_poison = pd.read_csv(poisoned_csv_path)
            report_clean  = generate_report("clean_dataset.csv",
                                            df_clean.drop('Label', axis=1).values,
                                            df_clean['Label'].values)
            report_poison = generate_report("poisoned_dataset.csv",
                                            df_poison.drop('Label', axis=1).values,
                                            df_poison['Label'].values)
            fig = create_comparison_chart(report_clean, report_poison)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Uploaded file: show this dataset's own detection scores
            fig = create_single_dataset_chart(report)
            st.plotly_chart(fig, use_container_width=True)


        st.markdown(
            '<p style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;'
            'color:#bcc9cd;margin:1rem 0 0.5rem;">Dataset Preview (first 10 rows)</p>',
            unsafe_allow_html=True
        )
        st.dataframe(df.head(10), use_container_width=True)

    with tab_label_flip:
        st.markdown(
            '<p style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;'
            'color:#bcc9cd;">Label-Flip Attack Detection</p>',
            unsafe_allow_html=True
        )
        st.write("Detects inverted or mislabeled ground truth targets by measuring decision boundary "
                 "confidence mismatch against random forest estimators.")
        col_lf1, col_lf2 = st.columns(2)
        col_lf1.metric("Label-Flip Score", f"{report.get('label_flip_score', 0.0):.1f}%")
        col_lf2.metric("Flagged / Total",
                       f"{report.get('suspicious_samples', 0)} / {report.get('total_samples', 0)}")

    with tab_anomaly:
        st.markdown(
            '<p style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;'
            'color:#bcc9cd;">Feature Outlier & Distribution Anomalies</p>',
            unsafe_allow_html=True
        )
        st.write("Evaluates multidimensional feature deviations using an ensemble of Isolation Forest "
                 "and Local Outlier Factor (LOF) estimators to identify corrupted or out-of-distribution values.")
        col_an1, col_an2 = st.columns(2)
        col_an1.metric("Anomaly Score",         f"{report.get('anomaly_score', 0.0):.1f}%")
        col_an2.metric("Contamination Estimate", f"{(report.get('anomaly_score', 0.0) / 100):.2f}")

    with tab_inspector:
        st.markdown(
            '<p style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;'
            'color:#bcc9cd;">Suspicious Samples Inspector</p>',
            unsafe_allow_html=True
        )
        flagged_idx = report.get('suspicious_indices', [])
        st.write(f"Total flagged rows: **{report.get('suspicious_samples', 0)}**")
        if flagged_idx:
            valid_idx = [int(i) for i in flagged_idx if 0 <= int(i) < len(df)]
            st.dataframe(df.iloc[valid_idx].copy(), use_container_width=True)
        else:
            st.success("No suspicious samples detected in this dataset.")


# ── Side-by-side demo renderer ────────────────────────────────────────────────
def render_side_by_side_demo(clean_path: str, poison_path: str):
    df_clean  = pd.read_csv(clean_path)
    df_poison = pd.read_csv(poison_path)

    report_clean  = generate_report("clean_dataset.csv",
                                    df_clean.drop('Label', axis=1).values,
                                    df_clean['Label'].values)
    report_poison = generate_report("poisoned_dataset.csv",
                                    df_poison.drop('Label', axis=1).values,
                                    df_poison['Label'].values)

    st.markdown(
        '<p style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;'
        'color:#4cd7f6;margin-bottom:0.25rem;">Side-by-Side Demo</p>'
        '<h2 style="margin-top:0;">⚡ Clean vs. Poisoned Analysis</h2>',
        unsafe_allow_html=True
    )

    col_c, col_p = st.columns(2)
    with col_c:
        st.markdown(
            f'<div style="font-size:0.75rem;font-weight:600;color:#4fdbc8;'
            f'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.5rem;">'
            f'📗 Clean Dataset</div>', unsafe_allow_html=True
        )
        st.dataframe(df_clean.head(6), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Poisoning Score", f"{report_clean.get('poisoning_score', 0.0):.1f}%")
        c2.metric("Status", get_verdict_badge(report_clean.get('poisoning_score', 0.0)))
        _recommendation_banner(report_clean)
        txt_clean = generate_text_report(report_clean)
        st.download_button("📥 Download Clean Report (.txt)", data=txt_clean,
                           file_name="sentry_report_clean_dataset.txt",
                           mime="text/plain", key="dl_demo_clean")

    with col_p:
        st.markdown(
            f'<div style="font-size:0.75rem;font-weight:600;color:#ffb4ab;'
            f'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.5rem;">'
            f'📕 Poisoned Dataset</div>', unsafe_allow_html=True
        )
        st.dataframe(df_poison.head(6), use_container_width=True)
        p1, p2 = st.columns(2)
        p1.metric("Poisoning Score", f"{report_poison.get('poisoning_score', 0.0):.1f}%")
        p2.metric("Status", get_verdict_badge(report_poison.get('poisoning_score', 0.0)))
        _recommendation_banner(report_poison)
        txt_poison = generate_text_report(report_poison)
        st.download_button("📥 Download Poisoned Report (.txt)", data=txt_poison,
                           file_name="sentry_report_poisoned_dataset.txt",
                           mime="text/plain", key="dl_demo_poison")

    st.markdown('<hr style="border-color:#2a2d36;margin:1.5rem 0;">', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;'
        'color:#bcc9cd;">Detection Metric Comparison</p>', unsafe_allow_html=True
    )
    fig = create_comparison_chart(report_clean, report_poison)
    st.plotly_chart(fig, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="SENTRY — Data Poisoning Detector", layout="wide")
    _inject_css()

    # Header
    st.markdown(
        '<p style="font-size:0.6rem;letter-spacing:0.2em;text-transform:uppercase;'
        'color:#4cd7f6;margin-bottom:0.15rem;">Synthetic Data Security Audit</p>',
        unsafe_allow_html=True
    )
    st.title("🔍 SENTRY: Data Poisoning Detector")

    clean_csv_path    = os.path.join("data", "clean_dataset.csv")
    poisoned_csv_path = os.path.join("data", "poisoned_dataset.csv")
    demo_ready = os.path.exists(clean_csv_path) and os.path.exists(poisoned_csv_path)

    # ── Sidebar ──
    st.sidebar.markdown(
        '<p style="font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;'
        'color:#4cd7f6;margin:0.5rem 0 0.25rem;">Data Ingestion</p>',
        unsafe_allow_html=True
    )
    uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type="csv")

    st.sidebar.markdown('<hr style="border-color:#2a2d36;">', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<p style="font-size:0.6rem;letter-spacing:0.15em;text-transform:uppercase;'
        'color:#4cd7f6;margin:0.5rem 0 0.25rem;">Demo Mode</p>',
        unsafe_allow_html=True
    )

    if "demo_view" not in st.session_state:
        st.session_state.demo_view = None

    if demo_ready:
        if st.sidebar.button("⚡ Side-by-Side Demo", use_container_width=True):
            st.session_state.demo_view = "side_by_side"

        col_b1, col_b2 = st.sidebar.columns(2)
        if col_b1.button("📗 Clean",   use_container_width=True):
            st.session_state.demo_view = "clean"
        if col_b2.button("📕 Poisoned", use_container_width=True):
            st.session_state.demo_view = "poisoned"

        if st.session_state.demo_view is not None:
            if st.sidebar.button("🔄 Reset", use_container_width=True):
                st.session_state.demo_view = None
                st.rerun()

    # ── Routing ──
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)

            # Verify label column exists
            if 'Label' not in raw_df.columns:
                st.error("⚠️ Dataset is missing required 'Label' column. Please check schema.")
                return

            # Detect if dataset follows the standard synthetic schema
            has_standard_schema = (
                all(col.startswith('Feature_') for col in raw_df.columns if col != 'Label')
                and raw_df['Label'].dropna().nunique() <= 2
            )

            if has_standard_schema:
                # Standard processing path
                feature_cols = [c for c in raw_df.columns if c != 'Label']
                X = raw_df[feature_cols].values
                y = raw_df['Label'].values
                report = generate_report(uploaded_file.name, X, y)
                render_dataset_dashboard(uploaded_file.name, raw_df, report, show_benchmark=True)
            else:
                # Flexible schema handling (partner's logic)
                is_valid, validation_msg = validate_dataset_mappable(raw_df)
                if not is_valid:
                    st.error(f"⚠️ Dataset cannot be processed: {validation_msg}")
                    return

                schema_guess = suggest_schema(raw_df)

                st.sidebar.markdown("---")
                st.sidebar.subheader("⚙️ Flexible Schema Mapping")
                st.sidebar.caption("Map real‑world tabular data into SENTRY's detection contract.")

                col_options = list(raw_df.columns)
                default_label_idx = (
                    col_options.index(schema_guess['label_guess'])
                    if schema_guess['label_guess'] in col_options
                    else len(col_options) - 1
                )

                label_col = st.sidebar.selectbox(
                    "Which column is the Label / Target?",
                    col_options,
                    index=default_label_idx,
                    help="Ground‑truth classification target to inspect for label flipping."
                )

                available_exclude_cols = [c for c in col_options if c != label_col]
                default_exclude = [c for c in schema_guess['id_cols'] if c in available_exclude_cols]

                exclude_cols = st.sidebar.multiselect(
                    "Columns to exclude (IDs, indices, timestamps):",
                    available_exclude_cols,
                    default=default_exclude,
                    help="Non‑feature metadata that should not be fed to the ML detector."
                )

                # Label binarization if needed
                binarize_threshold = None
                label_series = raw_df[label_col].dropna()
                unique_label_count = label_series.nunique()

                if unique_label_count > 2:
                    st.sidebar.markdown("##### 🔀 Label Binarization")
                    st.sidebar.info(
                        f"Target `{label_col}` has **{unique_label_count}** unique values. Detector requires binary 0/1."
                    )
                    if pd.api.types.is_numeric_dtype(label_series):
                        min_v = float(label_series.min())
                        max_v = float(label_series.max())
                        med_v = float(label_series.median())
                        is_integer = issubclass(label_series.dtype.type, (int, np.integer))
                        step_v = 1.0 if is_integer else (max_v - min_v) / 100.0

                        binarize_threshold = st.sidebar.slider(
                            "Binarization Threshold (≥ threshold → 1, < threshold → 0):",
                            min_value=min_v,
                            max_value=max_v,
                            value=med_v,
                            step=max(step_v, 0.01)
                        )
                        c0 = (label_series < binarize_threshold).sum()
                        c1 = (label_series >= binarize_threshold).sum()
                        st.sidebar.caption(f"Preview split: Class 0 = **{c0}**, Class 1 = **{c1}**")

                # Transform to contract
                mapped_df, dropped_non_numeric = reshape_to_contract(
                    raw_df,
                    label_col=label_col,
                    exclude_cols=exclude_cols,
                    binarize_threshold=binarize_threshold
                )

                if dropped_non_numeric:
                    st.sidebar.warning(
                        f"⚠️ {len(dropped_non_numeric)} non‑numeric feature column(s) excluded: "
                        f"{', '.join(dropped_non_numeric)}"
                    )

                if len(mapped_df.columns) <= 1:
                    st.error("⚠️ No numeric feature columns remain after exclusions. Adjust the sidebar selections.")
                    return

                feature_cols = [c for c in mapped_df.columns if c != 'Label']
                X = mapped_df[feature_cols].values
                y = mapped_df['Label'].values

                st.sidebar.success(f"✅ Transformed to {len(feature_cols)} features + binary label")
                report = generate_report(uploaded_file.name, X, y)
                render_dataset_dashboard(uploaded_file.name, raw_df, report, show_benchmark=True)
        except Exception as e:
            st.error(f"Error processing dataset: {e}")

    elif st.session_state.demo_view == "side_by_side" and demo_ready:
        render_side_by_side_demo(clean_csv_path, poisoned_csv_path)

    elif st.session_state.demo_view == "clean" and demo_ready:
        df_clean = pd.read_csv(clean_csv_path)
        report_clean = generate_report("clean_dataset.csv",
                                       df_clean.drop('Label', axis=1).values,
                                       df_clean['Label'].values)
        render_dataset_dashboard("clean_dataset.csv", df_clean, report_clean, show_benchmark=True)

    elif st.session_state.demo_view == "poisoned" and demo_ready:
        df_poison = pd.read_csv(poisoned_csv_path)
        report_poison = generate_report("poisoned_dataset.csv",
                                        df_poison.drop('Label', axis=1).values,
                                        df_poison['Label'].values)
        render_dataset_dashboard("poisoned_dataset.csv", df_poison, report_poison, show_benchmark=True)

    else:
        # Landing / benchmark view
        st.markdown(
            '<p style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;'
            'color:#869397;margin-bottom:0.5rem;">Upload a CSV or use Demo Mode to begin.</p>',
            unsafe_allow_html=True
        )
        if demo_ready:
            st.markdown(
                '<p style="font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;'
                'color:#bcc9cd;margin:1.5rem 0 0.5rem;">Benchmark — Clean vs. Poisoned Detection</p>',
                unsafe_allow_html=True
            )
            df_clean  = pd.read_csv(clean_csv_path)
            df_poison = pd.read_csv(poisoned_csv_path)
            report_clean  = generate_report("clean_dataset.csv",
                                            df_clean.drop('Label', axis=1).values,
                                            df_clean['Label'].values)
            report_poison = generate_report("poisoned_dataset.csv",
                                            df_poison.drop('Label', axis=1).values,
                                            df_poison['Label'].values)
            fig = create_comparison_chart(report_clean, report_poison)
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
