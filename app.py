from typing import Any, Dict
import numpy as np
import pandas as pd
import streamlit as st


def stub_generate_report(name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Temporary stub for PoisoningDetector.generate_report.
    Returns the exact schema frozen in BUILD.md Shared Contract.
    To be swapped with backend.detection.PoisoningDetector at Task 3.
    """
    total = len(X)
    suspicious_count = min(20, total)
    return {
        'dataset': name,
        'total_samples': total,
        'poisoning_score': 42.0,
        'anomaly_score': 38.0,
        'label_flip_score': 46.0,
        'suspicious_samples': suspicious_count,
        'suspicious_indices': list(range(suspicious_count)),
        'recommendation': '🚨 Significant poisoning detected — do not use'
    }


def main():
    st.set_page_config(page_title="SENTRY — Data Poisoning Detector", layout="wide")
    st.title("🔍 SENTRY: Synthetic Data Poisoning Detector")

    st.sidebar.header("Data Ingestion")
    uploaded_file = st.sidebar.file_uploader("Upload Dataset (CSV)", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📊 Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

            if 'Label' not in df.columns:
                st.error("⚠️ Dataset is missing required 'Label' column. Please check schema.")
                return

            feature_cols = [c for c in df.columns if c != 'Label']
            X = df[feature_cols].values
            y = df['Label'].values

            report = stub_generate_report(uploaded_file.name, X, y)  # swap at Task 3 checkpoint

            col1, col2, col3 = st.columns(3)
            col1.metric("Poisoning Score", f"{report['poisoning_score']:.1f}%")
            col2.metric("Suspicious Samples", report['suspicious_samples'])
            col3.metric("Status", "⚠️ RISKY" if report['poisoning_score'] > 20 else "✅ CLEAN")

            st.info(f"**Recommendation:** {report['recommendation']}")
        except Exception as e:
            st.error(f"Error processing dataset: {e}")
    else:
        st.info("👈 Upload a CSV file to begin analysis")


if __name__ == "__main__":
    main()
