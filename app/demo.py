"""Vocal Biomarker Screening Toolkit — Voice Recording Demo.

Launch with:
    python app/demo.py

Requires the model to be trained first:
    python scripts/setup_and_train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend for Gradio

from src.inference.pipeline import VocalBiomarkerPipeline

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_DIR = Path("results/parkinson/classical")

DISCLAIMER = (
    "⚠️ **RESEARCH / EDUCATIONAL TOOL ONLY** — This system has NOT been "
    "validated for clinical use and must NOT be used to diagnose, treat, or "
    "screen patients. Results are for research and educational purposes only. "
    "Always consult a qualified healthcare professional."
)

INSTRUCTIONS = """
### How to use
1. **Record** your voice using the microphone — sustain the vowel **/aaa/** steadily for **5–10 seconds**
   *(or upload a WAV/MP3 file)*
2. Click **Analyse**
3. Review the **confidence score** and **top acoustic features**

**Tips for best results:**
- Record in a quiet room
- Hold the microphone 10–15 cm from your mouth
- Take a breath, then sustain /aaa/ as long as comfortably possible
- Avoid coughing or background noise

**What the score means:**
A score closer to 1.0 means the acoustic pattern resembles the Parkinson's group in the training data. This is NOT a diagnosis.
"""


# ── Model loading ─────────────────────────────────────────────────────────────

def load_pipeline() -> tuple[VocalBiomarkerPipeline, list[str]]:
    """Load saved RF model and feature names. Returns (pipeline, feature_names)."""
    model_path   = MODEL_DIR / "rf.pkl"
    feat_path    = MODEL_DIR / "feature_names.json"
    metrics_path = MODEL_DIR / "metrics.json"

    if not model_path.exists():
        print(f"[WARN] No model found at {model_path}. Run setup_and_train.py first.")
        return VocalBiomarkerPipeline(), []

    import pickle
    with open(model_path, "rb") as f:
        rf = pickle.load(f)

    feature_names: list[str] = []
    if feat_path.exists():
        with open(feat_path) as f:
            feature_names = json.load(f)

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        loocv = metrics.get("loocv_rf", {})
        print(
            f"[INFO] Model loaded — LOOCV AUROC={loocv.get('auroc','?'):.3f}  "
            f"Sensitivity={loocv.get('sensitivity','?'):.3f}  "
            f"Specificity={loocv.get('specificity','?'):.3f}"
        )

    pipeline = VocalBiomarkerPipeline(
        parkinson_model=rf,
        feature_names=feature_names,
    )
    return pipeline, feature_names


# ── Plots ─────────────────────────────────────────────────────────────────────

def make_waveform_plot(audio_path: str) -> plt.Figure:
    """Waveform + mel-spectrogram side by side."""
    import librosa
    import librosa.display

    y, sr = librosa.load(audio_path, sr=16_000, mono=True, duration=10.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
    fig.patch.set_facecolor("#1e1e2e")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e1e2e")
        for spine in ax.spines.values():
            spine.set_color("#555")

    # Waveform
    t = np.linspace(0, len(y) / sr, len(y))
    ax1.plot(t, y, linewidth=0.6, color="#89b4fa")
    ax1.set_xlim(0, t[-1])
    ax1.set_ylabel("Amplitude", color="#cdd6f4", fontsize=8)
    ax1.tick_params(colors="#cdd6f4", labelsize=7)
    ax1.set_title("Waveform", color="#cdd6f4", fontsize=9, pad=4)

    # Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(
        S_dB, sr=sr, hop_length=160, x_axis="time", y_axis="mel",
        ax=ax2, cmap="magma", fmax=8000
    )
    ax2.set_title("Mel Spectrogram", color="#cdd6f4", fontsize=9, pad=4)
    ax2.tick_params(colors="#cdd6f4", labelsize=7)
    ax2.set_ylabel("Frequency (Hz)", color="#cdd6f4", fontsize=8)
    ax2.set_xlabel("Time (s)", color="#cdd6f4", fontsize=8)

    plt.tight_layout(pad=1.0)
    return fig


def make_score_gauge(score: float) -> plt.Figure:
    """Semicircular gauge showing the PD confidence score."""
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.axis("off")

    # Background arc (grey track)
    theta_bg = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg), color="#313244", linewidth=18,
            solid_capstyle="round")

    # Foreground arc coloured by score
    theta_fill = np.linspace(np.pi, np.pi - score * np.pi, 200)
    colour = "#a6e3a1" if score < 0.4 else ("#f9e2af" if score < 0.65 else "#f38ba8")
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=colour, linewidth=18,
            solid_capstyle="round")

    # Score text
    ax.text(0, 0.18, f"{score:.2f}", ha="center", va="center",
            fontsize=26, fontweight="bold", color=colour)
    ax.text(0, -0.12, "Confidence Score", ha="center", va="center",
            fontsize=8, color="#cdd6f4")
    ax.text(-1.15, -0.15, "0.0", ha="center", fontsize=7, color="#6c7086")
    ax.text( 1.15, -0.15, "1.0", ha="center", fontsize=7, color="#6c7086")

    label = "LOW" if score < 0.4 else ("MODERATE" if score < 0.65 else "HIGH")
    ax.text(0, -0.28, f"SIMILARITY TO PD PATTERN: {label}", ha="center",
            fontsize=7, color=colour, fontweight="bold")

    plt.tight_layout(pad=0.2)
    return fig


def make_feature_plot(top_features: list[dict]) -> plt.Figure:
    """Horizontal bar chart of top contributing features."""
    if not top_features:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Feature importances not available",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    names = [f["feature"].replace("_", " ") for f in top_features[:10]]
    imps  = [f["importance"] for f in top_features[:10]]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#1e1e2e")
    for spine in ax.spines.values():
        spine.set_color("#555")

    colours = ["#89dceb"] * len(names)
    # Highlight clinical biomarkers
    for i, _ in enumerate(names):
        raw = top_features[i]["feature"]
        if any(k in raw for k in ("jitter", "shimmer", "hnr", "f0")):
            colours[i] = "#f38ba8"
        elif "mfcc" in raw:
            colours[i] = "#89b4fa"

    ax.barh(range(len(names))[::-1], imps, color=colours,
                   edgecolor="none", height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=7, color="#cdd6f4")
    ax.set_xlabel("Importance", color="#cdd6f4", fontsize=8)
    ax.set_title("Top Contributing Acoustic Features", color="#cdd6f4", fontsize=9, pad=6)
    ax.tick_params(axis="x", colors="#6c7086", labelsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color="#f38ba8", label="Vocal biomarker (jitter/shimmer/HNR/F0)"),
        Patch(color="#89b4fa", label="MFCC"),
        Patch(color="#89dceb", label="Spectral / chroma"),
    ]
    ax.legend(handles=legend, fontsize=6, framealpha=0.0,
              labelcolor="#cdd6f4", loc="lower right")

    plt.tight_layout(pad=0.8)
    return fig


# ── Main analysis function ────────────────────────────────────────────────────

def analyse(audio_path: str | None, pipeline: VocalBiomarkerPipeline
            ) -> tuple[plt.Figure, plt.Figure, plt.Figure, str]:
    """Process audio and return (waveform_fig, gauge_fig, features_fig, report_md)."""
    empty = plt.figure()

    if audio_path is None:
        return empty, empty, empty, "_Please record or upload an audio clip first._"

    try:
        report = pipeline.predict(audio_path)
    except Exception as exc:
        return empty, empty, empty, f"**Error:** {exc}"

    pk = report["conditions"]["parkinson"]
    waveform_fig  = make_waveform_plot(audio_path)

    if pk.get("label") == "unavailable":
        gauge_fig   = empty
        feature_fig = empty
        md = (
            "**Model not loaded.**\n\n"
            "Run `python scripts/setup_and_train.py` first, then restart the app."
        )
    else:
        score       = pk.get("score", 0.0)
        label       = pk.get("label", "?").upper()
        top_feats   = pk.get("top_features", [])
        gauge_fig   = make_score_gauge(score)
        feature_fig = make_feature_plot(top_feats)

        colour = "🟢" if score < 0.4 else ("🟡" if score < 0.65 else "🔴")
        md = (
            f"**Duration:** {report['duration_s']:.1f} s &nbsp;|&nbsp; "
            f"**Sample rate:** {report['sample_rate']} Hz\n\n"
            f"**Parkinson's Pattern Score:** {score:.3f} &nbsp; {colour} &nbsp; **{label}**\n\n"
            f"_(Score ≥ 0.5 = pattern resembles PD group in training data)_\n\n"
            "---\n" + DISCLAIMER
        )

    return waveform_fig, gauge_fig, feature_fig, md


# ── Gradio interface ──────────────────────────────────────────────────────────

def build_demo(pipeline: VocalBiomarkerPipeline) -> gr.Blocks:
    with gr.Blocks(title="Vocal Biomarker Screening Toolkit") as demo:

        gr.Markdown("# 🎙 Vocal Biomarker Screening Toolkit")
        gr.Markdown(DISCLAIMER)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(INSTRUCTIONS)
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record or upload voice sample",
                )
                analyse_btn = gr.Button("▶  Analyse", variant="primary", size="lg")
                result_md = gr.Markdown("_Record a sample and click Analyse._")

            with gr.Column(scale=1):
                gauge_plot = gr.Plot(label="Parkinson's Pattern Score")
                feature_plot = gr.Plot(label="Top Acoustic Features")

        waveform_plot = gr.Plot(label="Audio Analysis")

        analyse_btn.click(
            fn=lambda path: analyse(path, pipeline),
            inputs=[audio_in],
            outputs=[waveform_plot, gauge_plot, feature_plot, result_md],
        )

        gr.Markdown(
            "---\n*Built as a portfolio / research project by Aynaz Javani. "
            "Source: [GitHub](https://github.com) · "
            "Dataset: Italian Parkinson's Voice and Speech Corpus*"
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    args = parser.parse_args()

    pipeline, _ = load_pipeline()
    demo = build_demo(pipeline)
    theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    )
    demo.launch(server_port=args.port, share=args.share, theme=theme)
