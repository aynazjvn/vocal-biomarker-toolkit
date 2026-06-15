# Vocal Biomarker Screening Toolkit

A research web application that screens for three neurological and respiratory conditions from a short voice recording. Built as a portfolio project by [Aynaz Javanivayeghan](https://aynazjvn.github.io).

> **Research / Educational Tool Only.** Not validated for clinical use. Do not use for diagnosis, treatment, or patient screening.

---

## Overview

This project extracts 63 acoustic biomarkers from a voice recording and runs them through three independent Random Forest classifiers — one per condition. The entire pipeline runs in real time in the browser: record, upload, get results with feature attribution.

| Condition | Dataset | LOOCV AUROC | Training samples |
|---|---|---|---|
| Parkinson's Disease | Italian Parkinson's Voice and Speech Corpus | **0.989** | 831 recordings, 61 subjects |
| Respiratory / COVID-19 | Coswara (IISc Bangalore) | **0.817** | 498 cough recordings, 498 subjects |
| Depression Affect | RAVDESS (affect proxy) | **0.737** | 864 speech clips, 24 actors |

---

## Architecture

```
Browser (MediaRecorder API)
    │
    ▼
FastAPI server  (app/server.py)
    │
    ▼
Feature Extractor  (src/features/audio_features.py)
    63 features: MFCCs (13×3 stats) · F0/YIN · Jitter · Shimmer · HNR
                 Spectral centroid · Rolloff · Flatness · ZCR · Chroma (12)
    │
    ├── Random Forest → Parkinson's score   (Italian corpus, LOOCV AUROC 0.989)
    ├── Random Forest → Respiratory score   (Coswara cough,  LOOCV AUROC 0.817)
    └── Random Forest → Depression score    (RAVDESS affect, LOOCV AUROC 0.737)
    │
    ▼
JSON report → rendered in browser with gauges + feature importance bars
```

---

## Project Structure

```
├── app/
│   ├── server.py              FastAPI backend — serves UI + /api/predict
│   └── index.html             Single-page frontend (recording, results, about)
├── src/
│   ├── data/
│   │   ├── parkinson_loader.py    Italian corpus loader
│   │   ├── respiratory_loader.py  Coswara loader (covid_status labels)
│   │   └── depression_loader.py   RAVDESS loader (affect proxy labels)
│   ├── features/
│   │   └── audio_features.py      63-feature extractor (librosa)
│   ├── inference/
│   │   └── pipeline.py            Multi-condition inference pipeline
│   ├── models/
│   │   ├── classical.py           RF + SVM builders
│   │   └── transformer_classifier.py
│   ├── training/
│   │   └── trainer.py             LOOCV training loop
│   └── evaluation/
│       └── metrics.py             AUROC, sensitivity, specificity, ECE
├── scripts/
│   ├── setup_and_train.py         One-shot setup for Parkinson's model
│   ├── train_parkinson.py
│   ├── train_respiratory.py
│   └── train_depression.py
├── results/
│   ├── parkinson/classical/       rf.pkl · svm.pkl · metrics.json
│   ├── respiratory/classical/     rf.pkl · svm.pkl · metrics.json
│   └── depression/classical/      rf.pkl · svm.pkl · metrics.json
├── data/
│   ├── coswara/                   Coswara-Data sparse clone + extracted audio
│   └── ravdess/                   RAVDESS Audio_Speech_Actors_01-24
├── notebooks/
│   └── Parkinson.ipynb            EDA and deep learning experiments
├── configs/                       Hydra configs for each condition
└── tests/                         Unit tests for features and pipeline
```

---

## Quick Start

**Prerequisites:** Python 3.12 (Anaconda recommended), `pip install -r requirements.txt`

### 1. Parkinson's model (already trained if you have the dataset)

```bash
python scripts/setup_and_train.py
```

Dataset: Italian Parkinson's Voice and Speech Corpus — place recordings in `Italian Parkinson's Voice and speech/`.

### 2. Respiratory model

```bash
# Data is already cloned and extracted in data/coswara/Extracted_data/
python scripts/train_respiratory.py --data data/coswara/Extracted_data
```

### 3. Depression model

```bash
# Data is already extracted in data/ravdess/
python scripts/train_depression.py --data data/ravdess
```

### 4. Launch the web app

```bash
python app/server.py
# Open http://127.0.0.1:8000
```

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the frontend |
| `/api/health` | GET | Model status + LOOCV metrics for all 3 conditions |
| `/api/predict` | POST | Upload audio file → JSON report with scores + top features |

### Example predict response

```json
{
  "duration_s": 7.2,
  "sample_rate": 16000,
  "conditions": {
    "parkinson":   { "score": 0.04, "label": "negative", "top_features": [...] },
    "respiratory": { "score": 0.31, "label": "negative", "top_features": [...] },
    "depression":  { "score": 0.42, "label": "negative", "top_features": [...] }
  }
}
```

---

## Datasets

| Dataset | License | Access |
|---|---|---|
| Italian Parkinson's Voice and Speech Corpus | Academic | [IEEE DataPort](https://ieee-dataport.org/open-access/italian-parkinsons-voice-and-speech) |
| Coswara | CC BY 4.0 | [GitHub](https://github.com/iiscleap/Coswara-Data) |
| RAVDESS | CC BY-NC-SA 4.0 | [Zenodo](https://zenodo.org/record/1188976) |

---

## Earlier Deep Learning Experiments

The `notebooks/Parkinson.ipynb` contains experiments fine-tuning speaker embedding models on the Italian corpus:

| Model | Test Accuracy | Notes |
|---|---|---|
| X-vector (MFCC) | 100% (original) / 90.6% (augmented) | Lightweight |
| ECAPA-TDNN (Fbank) | 100% (original) / 85.9% (augmented) | State-of-the-art speaker embedding |
| Whisper | 95.3% (original) / 70.6% (augmented) | ASR adapted for classification |

The classical ML pipeline (this repo's main contribution) achieves LOOCV AUROC 0.989, which is more reliable than held-out accuracy because it prevents same-subject data leakage across train/test.

---

## Author

**Aynaz Javanivayeghan** — MSc Computer Science, Concordia University

[Portfolio](https://aynazjvn.github.io) · [LinkedIn](https://www.linkedin.com/in/aynaz-javani) · [GitHub](https://github.com/aynazjvn)
