# Vocal Biomarker Screening Toolkit

A research web application that screens for three neurological and respiratory conditions from a short voice recording. Built as a portfolio project demonstrating end-to-end clinical speech ML — from data loading and subject-aware cross-validation to fine-tuned transformer inference with gradient-based explainability.

> **Research / Educational Tool Only.** Not validated for clinical use. Do not use for diagnosis, treatment, or patient screening.

---

## Results

| Condition | Backbone | Evaluation Protocol | AUROC | Dataset |
|---|---|---|---|---|
| Parkinson's Disease | Wav2Vec2-base fine-tuned | LOOCV — 61 subjects | **0.963** | Italian Parkinson's Voice and Speech Corpus (831 recordings) |
| Depression Affect | Wav2Vec2-base-SUPERB-ER fine-tuned | Leave-Actor-Out — 24 actors | **0.849** | RAVDESS (864 speech clips) |
| Respiratory / COVID-19 | WavLM-base fine-tuned | 10-fold subject-grouped CV | **0.733** | Coswara IISc (2,599 cough recordings) |

LOOCV (Leave-One-Subject-Out Cross-Validation) is the standard evaluation protocol for small medical audio datasets. It ensures no recording from a held-out subject ever appears in training, preventing the optimistic bias that arises from within-subject acoustic correlations.

---

## Architecture

Each condition uses an independently fine-tuned transformer encoder with a binary classification head:

```
Raw audio (16 kHz mono)
    |
    v
Specialist pretrained encoder (per condition)
    Parkinson's : facebook/wav2vec2-base              general speech
    Depression  : superb/wav2vec2-base-superb-er      emotion recognition pretrained
    Respiratory : microsoft/wavlm-base                masked denoising pretrained
    |
    v
CNN feature extractor  [frozen]
    |
    v
Transformer encoder  [bottom 8 of 12 layers frozen, top 4 fine-tuned]
    |
    v
Mean pooling over time frames
    |
    v
Dropout(0.1) -- Linear(768, 1) -- sigmoid -- P(condition)
```

**Backbone selection rationale:**
- `wav2vec2-base` was pretrained for automatic speech recognition. It provides a strong baseline for pathological speech (Parkinson's).
- `wav2vec2-base-superb-er` was further fine-tuned on the SUPERB emotion recognition benchmark. Its encoder already encodes affect-relevant representations, giving a better initialization for depression affect detection.
- `wavlm-base` uses a masked denoising pre-training objective that captures richer spectrotemporal features beyond speech content — better suited for cough audio where the signal is non-linguistic.

**Explainability:**
For transformer models, the server computes input-gradient saliency — d(sigmoid output) / d(input waveform) — and aggregates absolute gradient magnitudes into 0.5-second time blocks. The five most influential regions are returned per prediction alongside the probability score.

**Training details:**
- Optimizer: AdamW with weight decay 1e-2
- Scheduler: OneCycleLR with 10% linear warm-up
- Loss: BCEWithLogitsLoss with class-frequency pos_weight for imbalanced conditions
- All waveforms pre-loaded to RAM before LOOCV to eliminate per-fold disk I/O overhead
- Model weights cached after fold 1; subsequent folds rebuild the architecture from config without re-downloading

---

## Project Structure

```
vocal-biomarker-toolkit/
|
├── app/
│   ├── server.py                    FastAPI backend — /api/predict, /api/health
│   └── index.html                   Single-page frontend (recording, results, about)
|
├── src/
│   ├── data/
│   │   ├── base_loader.py           AudioSample dataclass and BaseLoader ABC
│   │   ├── parkinson_loader.py      Italian corpus loader
│   │   ├── respiratory_loader.py    Coswara loader (covid_status labels)
│   │   └── depression_loader.py     RAVDESS loader (affect proxy labels)
│   ├── features/
│   │   └── audio_features.py        63-feature extractor (MFCCs, jitter, shimmer, HNR, spectral)
│   ├── inference/
│   │   └── pipeline.py              Multi-condition pipeline with gradient saliency
│   ├── models/
│   │   ├── classical.py             RF and SVM builders (classical baseline)
│   │   └── wav2vec2_classifier.py   AutoModel-based binary classifier (transformer conditions)
│   ├── training/
│   │   ├── trainer.py               Classical LOOCV training loop
│   │   └── transformer_trainer.py   Transformer training — LOOCV, k-fold, RAM caching
│   └── evaluation/
│       └── metrics.py               AUROC, sensitivity, specificity, ECE
|
├── scripts/
│   ├── train_parkinson_transformer.py
│   ├── train_respiratory_transformer.py
│   └── train_depression_transformer.py
|
├── results/
│   ├── parkinson/transformer/       model.pt · config.json · metrics.json
│   ├── respiratory/transformer/     model.pt · config.json · metrics.json
│   └── depression/transformer/      model.pt · config.json · metrics.json
|
├── data/
│   ├── coswara/                     Coswara-Data git clone + Extracted_data/
│   └── ravdess/                     Audio_Speech_Actors_01-24/
|
├── notebooks/
│   └── Parkinson.ipynb              EDA and earlier deep learning experiments
|
└── tests/                           Unit tests for features and pipeline
```

---

## Setup and Training

**Prerequisites:** Python 3.10+, CUDA GPU strongly recommended.

```bash
git clone <repo-url>
cd vocal-biomarker-toolkit
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Model checkpoints are not committed (each `model.pt` is ~360 MB). Train each condition from scratch:

### Parkinson's

Requires the Italian Parkinson's Voice and Speech Corpus — place recordings under `Italian Parkinson's Voice and speech/`.

```bash
python scripts/train_parkinson_transformer.py
```

### Depression

Download RAVDESS from [Zenodo record 1188976](https://zenodo.org/record/1188976) (`Audio_Speech_Actors_01-24.zip`, 198 MB) and unzip to `data/ravdess/`.

```bash
python scripts/train_depression_transformer.py --data data/ravdess
```

### Respiratory

```bash
git clone https://github.com/iiscleap/Coswara-Data.git data/coswara
cd data/coswara && python extract_data.py && cd ../..
python scripts/train_respiratory_transformer.py --data data/coswara/Extracted_data
```

Each script runs cross-validation, trains a final model on all data, and saves the checkpoint to `results/{condition}/transformer/`.

---

## Running the Server

```bash
python app/server.py
# Open http://127.0.0.1:8000
```

On startup the server calls `VocalBiomarkerPipeline.from_auto()`, which checks for a transformer checkpoint first (`results/{condition}/transformer/model.pt`) and falls back to a classical RF model if one is absent. The `/api/health` endpoint reports which model type is loaded per condition along with its CV metrics.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Frontend |
| `/api/health` | GET | Loaded model type and AUROC per condition |
| `/api/predict` | POST (multipart) | Upload audio file, receive prediction report |

**Predict response:**

```json
{
  "audio_path": "...",
  "duration_s": 7.2,
  "sample_rate": 16000,
  "conditions": {
    "parkinson": {
      "score": 0.12,
      "label": "negative",
      "threshold": 0.5,
      "model_type": "transformer",
      "top_features": [
        { "feature": "2.0-2.5s", "importance": 0.18 },
        { "feature": "4.5-5.0s", "importance": 0.15 }
      ]
    },
    "respiratory": { "...": "..." },
    "depression":  { "...": "..." }
  },
  "disclaimer": "RESEARCH / EDUCATIONAL TOOL ONLY. ..."
}
```

For transformer models, `top_features` lists the five most salient 0.5-second time windows by gradient magnitude. For classical models they are the top-10 Random Forest feature importances by name.

---

## Datasets

| Dataset | License | Source |
|---|---|---|
| Italian Parkinson's Voice and Speech Corpus | Academic use | [IEEE DataPort](https://ieee-dataport.org/open-access/italian-parkinsons-voice-and-speech) |
| Coswara | CC BY 4.0 | [GitHub — iiscleap/Coswara-Data](https://github.com/iiscleap/Coswara-Data) |
| RAVDESS | CC BY-NC-SA 4.0 | [Zenodo 1188976](https://zenodo.org/record/1188976) |

---

## Notebooks

`notebooks/Parkinson.ipynb` documents earlier experiments fine-tuning speaker embedding models (X-Vector, ECAPA-TDNN, Whisper) on the Italian corpus. These are exploratory; the production pipeline uses the transformer training scripts described above.

---

## Author

**Aynaz Javanivayeghan** — MSc Computer Science, Concordia University

[Portfolio](https://aynazjvn.github.io) · [LinkedIn](https://www.linkedin.com/in/aynaz-javani) · [GitHub](https://github.com/aynazjvn)
