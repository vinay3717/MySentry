# 🔍 SENTRY: Synthetic Data Poisoning Detector

> Built for IEEE SYNAPSE 2026 — 6-Hour Offline Build Sprint (2-Person Team)

## One-Line Pitch
A tool that scans any ML training dataset and returns a poisoning score — catching label-flipping, feature-outlier, and backdoor-trigger attacks before they corrupt a model, with a breakdown of exactly which samples are suspicious and why.

## Problem Statement
ML teams routinely train on datasets they didn't fully vet — scraped data, vendor-licensed data, crowdsourced labels, public datasets. There's no accessible, easy-to-run tool that tells an engineer "is this dataset safe to train on?" Academic papers describe these attacks; almost no mainstream tool lets a practitioner check their own CSV in under a minute.

## Solution Summary
Upload a labeled CSV → the system runs three independent detectors (Isolation Forest, Local Outlier Factor, label-flip/confidence mismatch) → combines them into one 0–100% **Poisoning Score** with a per-attack-type breakdown → dashboard shows suspicious samples, a verdict, and a plain-language recommendation.

## Track
AI & Intelligent Systems (secondary relevance: Cybersecurity & Digital Trust)

## Team
- **T1 — Backend / Detection Logic Lead**
- **T2 — Frontend / Dashboard + Data Lead**

## Demo

| Clean Dataset | Poisoned Dataset |
|---|---|
| ✅ Low poisoning score, CLEAN verdict | 🚨 High poisoning score, RISKY verdict |

1. Launch the app (`streamlit run app.py`)
2. Upload `data/clean_dataset.csv` → see a low score and a clean verdict
3. Upload `data/poisoned_dataset.csv` → see the score jump and a risky verdict
4. (If enabled) View the comparison chart breaking down anomaly vs. label-flip signals

## Features
- 📤 Simple CSV upload — no setup, no account, runs fully offline
- 🧮 Ensemble poisoning score combining three independent detection signals
- 🚩 Suspicious sample flagging with row-level indices
- 📊 Clean vs. poisoned comparison visualization
- 🗣️ Plain-language recommendation, not just a raw number

## How It Works
1. **Isolation Forest + Local Outlier Factor** — two independent outlier detectors flag feature-space anomalies.
2. **Label-flip detection** — a quick classifier is trained on the uploaded data; samples where the model's prediction confidence is low are flagged as potentially mislabeled.
3. **Ensemble scoring** — the signals above are combined into a single 0–100% Poisoning Score.
4. **Verdict mapping** — the score is mapped to a plain-language recommendation using fixed thresholds (see below).

## Scoring Thresholds
| Poisoning Score | Verdict | Recommendation |
|---|---|---|
| 0–5% | Clean | ✅ Dataset appears clean |
| 5–20% | Minor anomalies | ⚠️ Minor anomalies detected — investigate further |
| 20–50% | Significant | 🚨 Significant poisoning detected — do not use |
| 50–100% | Critical | 🔴 CRITICAL: Dataset is heavily poisoned — reject immediately |

## Attack Types Detected
1. **Label flipping** — a random percentage of labels inverted (fully implemented, P0).
2. **Feature outliers** — extreme injected values unrelated to the true distribution (stretch, P1).
3. **Backdoor trigger** — a specific feature combination forced to one label (stretch, P1).

## Project Structure
```
sentry-poisoning-detector/
├── README.md                    # this file
├── requirements.txt             # shared dependency list
├── BUILD.md                     # build plan and shared contract
├── BUILD_T1_backend_detection.md
├── BUILD_T2_frontend_data.md
├── PREREQUISITES.md
├── INTEGRATION.md
├── backend/                     # detection engine (T1)
│   ├── detection.py
│   ├── scoring.py
│   └── report.py
├── data/                        # datasets + generator (T2)
│   ├── data_generator.py
│   ├── clean_dataset.csv
│   └── poisoned_dataset.csv
├── app.py                       # Streamlit dashboard (T2)
└── demo_script.md
```

## Data Format
Input CSVs must follow this schema:
- `Feature_1` ... `Feature_N` — numeric feature columns
- `Label` — binary target column (0 or 1)

The bundled synthetic datasets (`data/clean_dataset.csv`, `data/poisoned_dataset.csv`) are generated with a fixed random seed for reproducibility. The poisoned variant is created by randomly flipping 20% of labels in the clean dataset, giving a known ground truth to validate detector accuracy against.

## Getting Started

### Prerequisites
See [PREREQUISITES.md](./PREREQUISITES.md) for full setup steps.

### Installation
```bash
git clone <repo-url>
cd sentry-poisoning-detector
pip install -r requirements.txt
```

### Run the app
```bash
streamlit run app.py
```

### Regenerate demo datasets (optional)
```bash
python data/data_generator.py
```

## Scope
**In scope**
- Batch CSV upload of tabular, binary-label data
- Local, single-session, offline-first tool

**Out of scope (by design, for this 6-hour build)**
- Live/streaming detection
- Multi-class or image datasets
- Authentication / user accounts
- Cloud deployment (local run is the intended submission; Streamlit Cloud is a stretch goal only)
- Training a production-grade "victim" model

## Known Limitations
- Outlier detectors can't fully distinguish "rare but legitimate" data from injected poisoning — flagged samples should be reviewed, not auto-rejected.
- Validated against synthetic, known-ground-truth data within the project's time box; real-world datasets would need preprocessing to match the expected schema.
- Feature-outlier and backdoor-trigger detection are partially implemented stretch goals, not guaranteed to be fully scored in every build.

## Tech Stack
- **Frontend:** Streamlit, Plotly
- **Backend / ML:** scikit-learn (Isolation Forest, Local Outlier Factor, Random Forest), NumPy, pandas

## Acknowledgments
Built in a single 6-hour sprint by a 2-person team for IEEE SYNAPSE 2026.
