import os
from typing import Any, Dict
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


def main():
    st.set_page_config(page_title="SENTRY — Data Poisoning Detector", layout="wide")
    st.title("🔍 SENTRY: Synthetic Data Poisoning Detector")

    st.sidebar.header("📁 Data Ingestion")
    uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type="csv")

    # Load demo datasets from data/ for quick comparison benchmark
    clean_csv_path = os.path.join("data", "clean_dataset.csv")
    poisoned_csv_path = os.path.join("data", "poisoned_dataset.csv")

    demo_ready = os.path.exists(clean_csv_path) and os.path.exists(poisoned_csv_path)

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader(f"📊 Dataset Preview — `{uploaded_file.name}`")
            st.dataframe(df.head(10), use_container_width=True)

            if 'Label' not in df.columns:
                st.error("⚠️ Dataset is missing required 'Label' column. Please check schema.")
                return

            feature_cols = [c for c in df.columns if c != 'Label']
            X = df[feature_cols].values
            y = df['Label'].values

            report = stub_generate_report(uploaded_file.name, X, y)  # swap at Task 3 checkpoint

            # Top KPI Summary Cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Poisoning Score", f"{report['poisoning_score']:.1f}%")
            col2.metric("Anomaly Score", f"{report['anomaly_score']:.1f}%")
            col3.metric("Label-Flip Score", f"{report['label_flip_score']:.1f}%")
            col4.metric("Status", get_verdict_badge(report['poisoning_score']))

            st.info(f"**Recommendation:** {report['recommendation']}")

            # Task 4: Attack-Type Breakdown Tabs
            tab_overview, tab_label_flip, tab_anomaly, tab_inspector = st.tabs([
                "📊 Overview & Comparison",
                "🏷️ Label-Flip Analysis",
                "🌲 Anomaly / Outlier Analysis",
                "🔍 Suspicious Samples Inspector"
            ])

            with tab_overview:
                st.markdown("### 📈 Detection Benchmark Comparison")
                if demo_ready:
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
                    st.warning("Demo datasets in `data/` not found to render side-by-side benchmark.")

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

        except Exception as e:
            st.error(f"Error processing dataset: {e}")
    else:
        st.info("👈 Upload a CSV file in the sidebar to begin analysis")

        # Fallback view: Show clean vs poisoned benchmark comparison chart directly
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
