import datetime
import os
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def stub_generate_report(name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Temporary stub for PoisoningDetector.generate_report.
    Returns the exact schema frozen in BUILD.md Shared Contract.
    To be swapped with backend.detection.PoisoningDetector at Task 3.
    """
    total = len(X)
    is_clean = "clean" in name.lower()

    if is_clean:
        poisoning_score = 3.5
        anomaly_score = 4.0
        label_flip_score = 3.0
        suspicious_count = min(3, total)
        recommendation = "✅ Dataset appears clean"
    else:
        poisoning_score = 54.0
        anomaly_score = 48.0
        label_flip_score = 62.0
        suspicious_count = min(40, total)
        recommendation = "🔴 CRITICAL: Dataset is heavily poisoned — reject immediately"

    suspicious_indices = list(range(suspicious_count))

    return {
        'dataset': name,
        'total_samples': total,
        'poisoning_score': poisoning_score,
        'anomaly_score': anomaly_score,
        'label_flip_score': label_flip_score,
        'suspicious_samples': suspicious_count,
        'suspicious_indices': suspicious_indices,
        'recommendation': recommendation
    }


def create_comparison_chart(report_clean: Dict[str, Any], report_poison: Dict[str, Any]) -> go.Figure:
    """
    Task 4: Grouped bar chart comparing clean vs poisoned datasets
    across all three primary scores (Poisoning, Anomaly, Label Flip).
    """
    categories = ['Poisoning Score', 'Anomaly Score', 'Label Flip Score']
    clean_values = [
        report_clean.get('poisoning_score', 0.0),
        report_clean.get('anomaly_score', 0.0),
        report_clean.get('label_flip_score', 0.0)
    ]
    poison_values = [
        report_poison.get('poisoning_score', 0.0),
        report_poison.get('anomaly_score', 0.0),
        report_poison.get('label_flip_score', 0.0)
    ]

    fig = go.Figure(data=[
        go.Bar(
            name='Clean Dataset',
            x=categories,
            y=clean_values,
            marker_color='rgba(34, 197, 94, 0.85)',
            text=[f"{v:.1f}%" for v in clean_values],
            textposition='auto'
        ),
        go.Bar(
            name='Poisoned Dataset',
            x=categories,
            y=poison_values,
            marker_color='rgba(239, 68, 68, 0.85)',
            text=[f"{v:.1f}%" for v in poison_values],
            textposition='auto'
        ),
    ])

    fig.update_layout(
        title="<b>Detection Metrics Comparison: Clean vs. Poisoned</b>",
        barmode='group',
        yaxis_title="Detection Score (%)",
        yaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40),
        height=400
    )
    return fig


def get_verdict_badge(score: float) -> str:
    """Verdict badge mapped to BUILD.md frozen scoring thresholds."""
    if score < 5:
        return "✅ CLEAN"
    elif score < 20:
        return "⚠️ MINOR ANOMALIES"
    elif score < 50:
        return "🚨 SIGNIFICANT"
    else:
        return "🔴 CRITICAL"


def generate_text_report(report: Dict[str, Any]) -> str:
    """
    Task 5: Format a plain-text audit summary report for download.
    """
    verdict = get_verdict_badge(report.get('poisoning_score', 0.0))
    indices_str = ", ".join(map(str, report.get('suspicious_indices', [])))
    if not indices_str:
        indices_str = "None"

    return f"""======================================================================
SENTRY: SYNTHETIC DATA POISONING DETECTION AUDIT REPORT
======================================================================
Generated at: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Dataset Name: {report.get('dataset', 'Unknown')}
Total Samples Analyzed: {report.get('total_samples', 0)}

----------------------------------------------------------------------
DETECTION METRICS SUMMARY:
----------------------------------------------------------------------
- Overall Poisoning Score: {report.get('poisoning_score', 0.0):.1f}%
- Feature Anomaly / Outlier Score: {report.get('anomaly_score', 0.0):.1f}%
- Label-Flipping Mismatch Score: {report.get('label_flip_score', 0.0):.1f}%
- Suspicious Samples Flagged: {report.get('suspicious_samples', 0)}
- Dataset Security Verdict: {verdict}

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


def render_dataset_dashboard(dataset_name: str, df: pd.DataFrame, report: Dict[str, Any], show_benchmark: bool = True):
    """
    Helper to render the complete analysis view for a given dataset and report.
    """
    st.subheader(f"📊 Dataset Preview — `{dataset_name}`")
    st.dataframe(df.head(10), use_container_width=True)

    # Top KPI Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Poisoning Score", f"{report['poisoning_score']:.1f}%")
    col2.metric("Anomaly Score", f"{report['anomaly_score']:.1f}%")
    col3.metric("Label-Flip Score", f"{report['label_flip_score']:.1f}%")
    col4.metric("Status", get_verdict_badge(report['poisoning_score']))

    st.info(f"**Recommendation:** {report['recommendation']}")

    # Task 5: Plain-text report download button
    report_txt = generate_text_report(report)
    clean_filename = f"sentry_report_{os.path.splitext(dataset_name)[0]}.txt"
    st.download_button(
        label="📥 Download Audit Report (.txt)",
        data=report_txt,
        file_name=clean_filename,
        mime="text/plain",
        key=f"dl_{dataset_name}"
    )

    # Attack-Type Breakdown Tabs
    tab_overview, tab_label_flip, tab_anomaly, tab_inspector = st.tabs([
        "📊 Overview & Comparison",
        "🏷️ Label-Flip Analysis",
        "🌲 Anomaly / Outlier Analysis",
        "🔍 Suspicious Samples Inspector"
    ])

    clean_csv_path = os.path.join("data", "clean_dataset.csv")
    poisoned_csv_path = os.path.join("data", "poisoned_dataset.csv")
    demo_ready = os.path.exists(clean_csv_path) and os.path.exists(poisoned_csv_path)

    with tab_overview:
        st.markdown("### 📈 Detection Benchmark Comparison")
        if show_benchmark and demo_ready:
            df_clean = pd.read_csv(clean_csv_path)
            df_poison = pd.read_csv(poisoned_csv_path)

            report_clean = stub_generate_report(
                "clean_dataset.csv",
                df_clean.drop('Label', axis=1).values,
                df_clean['Label'].values
            )
            report_poison = stub_generate_report(
                "poisoned_dataset.csv",
                df_poison.drop('Label', axis=1).values,
                df_poison['Label'].values
            )

            fig = create_comparison_chart(report_clean, report_poison)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(f"Analyzed {report['total_samples']} samples. Poisoning score is {report['poisoning_score']:.1f}%.")

    with tab_label_flip:
        st.markdown("### 🏷️ Label-Flipping Attack Detection")
        st.write(
            "Detects inverted or mislabeled ground truth targets by measuring decision boundary "
            "confidence mismatch against random forest estimators."
        )
        col_lf1, col_lf2 = st.columns(2)
        col_lf1.metric("Label-Flip Score", f"{report['label_flip_score']:.1f}%")
        col_lf2.metric("Flagged Sample Ratio", f"{report['suspicious_samples']}/{report['total_samples']}")

    with tab_anomaly:
        st.markdown("### 🌲 Feature Outlier & Distribution Anomalies")
        st.write(
            "Evaluates multidimensional feature deviations using an ensemble of Isolation Forest "
            "and Local Outlier Factor (LOF) estimators to identify corrupted or out-of-distribution values."
        )
        col_an1, col_an2 = st.columns(2)
        col_an1.metric("Anomaly Score", f"{report['anomaly_score']:.1f}%")
        col_an2.metric("Contamination Estimate", f"{(report['anomaly_score'] / 100):.2f}")

    with tab_inspector:
        st.markdown("### 🔍 Suspicious Samples Inspector")
        st.write(f"Total flagged rows: **{report['suspicious_samples']}**")
        flagged_idx = report.get('suspicious_indices', [])
        if flagged_idx:
            flagged_df = df.iloc[flagged_idx].copy()
            st.dataframe(flagged_df, use_container_width=True)
        else:
            st.success("No suspicious samples detected in this dataset.")


def render_side_by_side_demo(clean_path: str, poison_path: str):
    """
    Task 5: One-click side-by-side demo showing clean vs. poisoned datasets directly.
    """
    df_clean = pd.read_csv(clean_path)
    df_poison = pd.read_csv(poison_path)

    report_clean = stub_generate_report(
        "clean_dataset.csv",
        df_clean.drop('Label', axis=1).values,
        df_clean['Label'].values
    )
    report_poison = stub_generate_report(
        "poisoned_dataset.csv",
        df_poison.drop('Label', axis=1).values,
        df_poison['Label'].values
    )

    st.markdown("## ⚡ Side-by-Side Demo: Clean vs. Poisoned Analysis")

    col_c, col_p = st.columns(2)
    with col_c:
        st.markdown("### 📗 Clean Dataset (`clean_dataset.csv`)")
        st.dataframe(df_clean.head(6), use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Poisoning Score", f"{report_clean['poisoning_score']:.1f}%")
        c2.metric("Status", get_verdict_badge(report_clean['poisoning_score']))
        st.success(f"{report_clean['recommendation']}")

        txt_clean = generate_text_report(report_clean)
        st.download_button(
            label="📥 Download Clean Report (.txt)",
            data=txt_clean,
            file_name="sentry_report_clean_dataset.txt",
            mime="text/plain",
            key="dl_demo_clean"
        )

    with col_p:
        st.markdown("### 📕 Poisoned Dataset (`poisoned_dataset.csv`)")
        st.dataframe(df_poison.head(6), use_container_width=True)
        p1, p2 = st.columns(2)
        p1.metric("Poisoning Score", f"{report_poison['poisoning_score']:.1f}%")
        p2.metric("Status", get_verdict_badge(report_poison['poisoning_score']))
        st.error(f"{report_poison['recommendation']}")

        txt_poison = generate_text_report(report_poison)
        st.download_button(
            label="📥 Download Poisoned Report (.txt)",
            data=txt_poison,
            file_name="sentry_report_poisoned_dataset.txt",
            mime="text/plain",
            key="dl_demo_poison"
        )

    st.markdown("---")
    st.markdown("### 📊 Side-by-Side Metric Comparison")
    fig = create_comparison_chart(report_clean, report_poison)
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(page_title="SENTRY — Data Poisoning Detector", layout="wide")
    st.title("🔍 SENTRY: Synthetic Data Poisoning Detector")

    clean_csv_path = os.path.join("data", "clean_dataset.csv")
    poisoned_csv_path = os.path.join("data", "poisoned_dataset.csv")
    demo_ready = os.path.exists(clean_csv_path) and os.path.exists(poisoned_csv_path)

    # Sidebar: Data Ingestion & Demo Controls
    st.sidebar.header("📁 Data Ingestion")
    uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type="csv")

    st.sidebar.markdown("---")
    st.sidebar.header("🚀 Demo Mode (Task 5)")

    if "demo_view" not in st.session_state:
        st.session_state.demo_view = None

    if demo_ready:
        if st.sidebar.button("⚡ One-Click Side-by-Side Demo", use_container_width=True):
            st.session_state.demo_view = "side_by_side"

        col_b1, col_b2 = st.sidebar.columns(2)
        if col_b1.button("📗 Load Clean", use_container_width=True):
            st.session_state.demo_view = "clean"
        if col_b2.button("📕 Load Poisoned", use_container_width=True):
            st.session_state.demo_view = "poisoned"

        if st.session_state.demo_view is not None:
            if st.sidebar.button("🔄 Reset View", use_container_width=True):
                st.session_state.demo_view = None
                st.rerun()

    # Route UI rendering
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'Label' not in df.columns:
                st.error("⚠️ Dataset is missing required 'Label' column. Please check schema.")
                return

            feature_cols = [c for c in df.columns if c != 'Label']
            X = df[feature_cols].values
            y = df['Label'].values

            report = stub_generate_report(uploaded_file.name, X, y)  # swap at Task 3
            render_dataset_dashboard(uploaded_file.name, df, report, show_benchmark=True)
        except Exception as e:
            st.error(f"Error processing dataset: {e}")

    elif st.session_state.demo_view == "side_by_side" and demo_ready:
        render_side_by_side_demo(clean_csv_path, poisoned_csv_path)

    elif st.session_state.demo_view == "clean" and demo_ready:
        df_clean = pd.read_csv(clean_csv_path)
        report_clean = stub_generate_report(
            "clean_dataset.csv",
            df_clean.drop('Label', axis=1).values,
            df_clean['Label'].values
        )
        render_dataset_dashboard("clean_dataset.csv", df_clean, report_clean, show_benchmark=True)

    elif st.session_state.demo_view == "poisoned" and demo_ready:
        df_poison = pd.read_csv(poisoned_csv_path)
        report_poison = stub_generate_report(
            "poisoned_dataset.csv",
            df_poison.drop('Label', axis=1).values,
            df_poison['Label'].values
        )
        render_dataset_dashboard("poisoned_dataset.csv", df_poison, report_poison, show_benchmark=True)

    else:
        st.info("👈 Upload a CSV file or click a **Demo Mode** button in the sidebar to begin analysis.")
        if demo_ready:
            st.markdown("### 📊 Benchmark Demo: Clean vs. Poisoned Detection")
            df_clean = pd.read_csv(clean_csv_path)
            df_poison = pd.read_csv(poisoned_csv_path)

            report_clean = stub_generate_report(
                "clean_dataset.csv",
                df_clean.drop('Label', axis=1).values,
                df_clean['Label'].values
            )
            report_poison = stub_generate_report(
                "poisoned_dataset.csv",
                df_poison.drop('Label', axis=1).values,
                df_poison['Label'].values
            )

            fig = create_comparison_chart(report_clean, report_poison)
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
