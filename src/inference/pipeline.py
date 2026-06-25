"""Unified inference pipeline: raw audio → multi-condition JSON report.

Usage
-----
    from src.inference.pipeline import VocalBiomarkerPipeline

    # Auto-detect transformer vs classical per condition:
    pipe = VocalBiomarkerPipeline.from_auto(
        parkinson_dir="results/parkinson",
        respiratory_dir="results/respiratory",
        depression_dir="results/depression",
    )
    report = pipe.predict("path/to/audio.wav")
    print(report)           # JSON-serialisable dict

    # Legacy: load classical models only (backward compatible):
    pipe = VocalBiomarkerPipeline.from_checkpoints(
        parkinson_model_dir="results/parkinson/classical"
    )

Model priority
--------------
For each condition, ``from_auto`` loads in this order:
  1. Transformer: ``{condition_dir}/transformer/model.pt``  (Wav2Vec2 fine-tuned)
  2. Classical:   ``{condition_dir}/classical/rf.pkl``      (Random Forest fallback)

Output schema
-------------
    {
        "audio_path": "...",
        "duration_s": 4.2,
        "sample_rate": 16000,
        "conditions": {
            "parkinson": {
                "score": 0.82,          # P(positive class)
                "label": "positive",    # "positive" | "negative" | "unavailable"
                "threshold": 0.5,
                "model_type": "transformer",   # "transformer" | "classical"
                "top_features": [
                    {"feature": "0.0-0.5s", "importance": 0.31},  # time saliency
                    ...
                ]
            },
            ...
        },
        "disclaimer": "Research/educational tool only. NOT for clinical diagnosis."
    }
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from src.features.audio_features import FeatureExtractor

DISCLAIMER = (
    "RESEARCH / EDUCATIONAL TOOL ONLY. "
    "This system is NOT validated for clinical use and must NOT be used "
    "to diagnose, treat, or screen patients."
)

_TRANSFORMER_SUBDIR = "transformer"
_CLASSICAL_SUBDIR   = "classical"
_BLOCK_DURATION_S   = 0.5   # saliency block size for time-region attribution


class VocalBiomarkerPipeline:
    """Multi-condition vocal biomarker inference pipeline.

    Supports both fine-tuned Wav2Vec2 transformer models and classical
    Random Forest pipelines.  Load via :meth:`from_auto` (recommended) or
    the legacy :meth:`from_checkpoints`.

    Args:
        parkinson_model:  Loaded model for Parkinson's (RF sklearn or Wav2Vec2Classifier).
        depression_model: Loaded model for depression.
        respiratory_model: Loaded model for respiratory.
        model_types: Dict mapping condition → "transformer" | "classical".
        feature_names: Feature names for classical RF conditions.
        respiratory_feature_names: Feature names for respiratory RF.
        depression_feature_names: Feature names for depression RF.
        threshold: Default probability threshold.
        sample_rate: Audio sample rate for the feature extractor.
    """

    def __init__(
        self,
        parkinson_model=None,
        depression_model=None,
        respiratory_model=None,
        model_types: Optional[dict[str, str]] = None,
        feature_names: Optional[list[str]] = None,
        respiratory_feature_names: Optional[list[str]] = None,
        depression_feature_names: Optional[list[str]] = None,
        threshold: float = 0.5,
        sample_rate: int = 16_000,
    ) -> None:
        self.models: dict[str, object] = {
            "parkinson":   parkinson_model,
            "depression":  depression_model,
            "respiratory": respiratory_model,
        }
        self.model_types: dict[str, str] = model_types or {}
        self.feature_names = feature_names or []
        self.feature_names_by_condition: dict[str, list[str]] = {
            "parkinson":   feature_names or [],
            "respiratory": respiratory_feature_names or feature_names or [],
            "depression":  depression_feature_names or feature_names or [],
        }
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.extractor = FeatureExtractor(sr=sample_rate)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, audio_path: str | Path) -> dict:
        """Run all available condition models on *audio_path*.

        Args:
            audio_path: Path to an audio file (any format librosa supports).

        Returns:
            JSON-serialisable dict following the schema above.
        """
        import librosa

        audio_path = Path(audio_path)

        # Always extract classical features (cheap, needed for RF fallback)
        bundle = self.extractor.from_file(audio_path)
        X = bundle.vector.reshape(1, -1)

        # Load raw waveform only if any transformer model is active
        raw_waveform: Optional[np.ndarray] = None
        if any(self.model_types.get(c) == "transformer" for c in self.models):
            raw_waveform, _ = librosa.load(
                str(audio_path), sr=self.sample_rate, mono=True
            )

        report: dict = {
            "audio_path": str(audio_path),
            "duration_s": round(bundle.duration_s, 3),
            "sample_rate": bundle.sample_rate,
            "conditions": {},
            "disclaimer": DISCLAIMER,
        }

        for condition, model in self.models.items():
            if self.model_types.get(condition) == "transformer":
                report["conditions"][condition] = self._run_transformer(
                    condition, model, raw_waveform, bundle
                )
            else:
                report["conditions"][condition] = self._run_classical(
                    condition, model, X, bundle
                )

        return report

    def predict_json(self, audio_path: str | Path, indent: int = 2) -> str:
        """Same as :meth:`predict` but returns a formatted JSON string."""
        return json.dumps(self.predict(audio_path), indent=indent, default=str)

    # ------------------------------------------------------------------
    # Per-condition runners
    # ------------------------------------------------------------------

    def _run_classical(
        self,
        condition: str,
        model,
        X: np.ndarray,
        bundle,
    ) -> dict:
        if model is None:
            return {"label": "unavailable", "reason": "model not loaded"}

        try:
            prob = float(model.predict_proba(X)[0, 1])
            label = "positive" if prob >= self.threshold else "negative"

            top_features: list[dict] = []
            clf = getattr(model, "named_steps", {}).get("clf")
            if clf is not None and hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
                names = (
                    self.feature_names_by_condition.get(condition)
                    or self.feature_names
                    or [f"feature_{i}" for i in range(len(importances))]
                )
                ranked = sorted(
                    zip(names, importances.tolist()), key=lambda x: -x[1]
                )[:10]
                top_features = [{"feature": n, "importance": round(v, 4)} for n, v in ranked]

            return {
                "score": round(prob, 4),
                "label": label,
                "threshold": self.threshold,
                "model_type": "classical",
                "top_features": top_features,
            }
        except Exception as exc:
            return {"label": "error", "reason": str(exc)}

    def _run_transformer(
        self,
        condition: str,
        model,
        raw_waveform: Optional[np.ndarray],
        bundle,
    ) -> dict:
        if model is None:
            return {"label": "unavailable", "reason": "model not loaded"}
        if raw_waveform is None:
            return {"label": "error", "reason": "raw waveform unavailable"}

        try:
            import torch

            device = next(model.parameters()).device

            # Prepare input tensor: truncate to 10 s
            max_samples = int(10.0 * self.sample_rate)
            wav = raw_waveform[:max_samples].astype(np.float32)

            # Wav2Vec2 expects normalised input (zero-mean / unit-var)
            wav = (wav - wav.mean()) / (wav.std() + 1e-9)
            input_values = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(device)

            # Gradient saliency: d(sigmoid(logit)) / d(input_values)
            prob_tensor, grad = model.predict_proba_with_grad(input_values)
            prob = float(prob_tensor.item())
            label = "positive" if prob >= self.threshold else "negative"

            # Map raw-sample gradients to 0.5s time blocks
            top_features = self._time_saliency(
                grad.squeeze(0).cpu().numpy(),
                sr=self.sample_rate,
                block_s=_BLOCK_DURATION_S,
            )

            return {
                "score": round(prob, 4),
                "label": label,
                "threshold": self.threshold,
                "model_type": "transformer",
                "top_features": top_features,
            }
        except Exception as exc:
            return {"label": "error", "reason": str(exc)}

    # ------------------------------------------------------------------
    # Gradient → time-region saliency
    # ------------------------------------------------------------------

    @staticmethod
    def _time_saliency(
        grad: np.ndarray,
        sr: int,
        block_s: float = 0.5,
        n_top: int = 5,
    ) -> list[dict]:
        """Convert raw-sample gradient magnitudes to labelled time regions.

        Args:
            grad: Absolute gradient, shape ``(n_samples,)``.
            sr: Audio sample rate.
            block_s: Duration of each attribution block in seconds.
            n_top: Number of top regions to return.

        Returns:
            List of ``{"feature": "0.0-0.5s", "importance": 0.31}`` dicts,
            sorted descending by importance, normalised to sum to 1.
        """
        block_size = int(block_s * sr)
        if block_size == 0 or len(grad) == 0:
            return []

        n_blocks = max(1, len(grad) // block_size)
        blocks = np.array_split(grad[: n_blocks * block_size], n_blocks)
        scores = np.array([b.mean() for b in blocks])

        total = scores.sum() + 1e-9
        normed = scores / total

        # Build region labels and sort
        regions = [
            (f"{i * block_s:.1f}-{(i + 1) * block_s:.1f}s", float(normed[i]))
            for i in range(len(normed))
        ]
        regions.sort(key=lambda x: -x[1])

        return [
            {"feature": name, "importance": round(imp, 4)}
            for name, imp in regions[:n_top]
        ]

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_auto(
        cls,
        parkinson_dir: Optional[str | Path] = None,
        respiratory_dir: Optional[str | Path] = None,
        depression_dir: Optional[str | Path] = None,
        threshold: float = 0.5,
        sample_rate: int = 16_000,
        device: Optional[str] = None,
    ) -> "VocalBiomarkerPipeline":
        """Auto-detect and load transformer (preferred) or classical model for each condition.

        For each condition, checks ``{condition_dir}/transformer/model.pt`` first,
        then falls back to ``{condition_dir}/classical/rf.pkl``.

        Args:
            parkinson_dir: Root directory for Parkinson's models
                           (e.g. ``"results/parkinson"``).
            respiratory_dir: Root for respiratory models.
            depression_dir: Root for depression models.
            threshold: Decision threshold for all conditions.
            sample_rate: Audio sample rate.
            device: Torch device string (default: ``"cuda"`` if available else ``"cpu"``).
        """
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        dirs = {
            "parkinson":   Path(parkinson_dir)  if parkinson_dir  else None,
            "respiratory": Path(respiratory_dir) if respiratory_dir else None,
            "depression":  Path(depression_dir)  if depression_dir  else None,
        }

        models: dict[str, object] = {}
        model_types: dict[str, str] = {}
        feature_names_map: dict[str, list[str]] = {}

        for condition, root in dirs.items():
            if root is None:
                models[condition] = None
                continue

            transformer_dir = root / _TRANSFORMER_SUBDIR
            classical_dir   = root / _CLASSICAL_SUBDIR

            if (transformer_dir / "model.pt").exists():
                m = _load_transformer(transformer_dir, device=device)
                if m is not None:
                    models[condition] = m
                    model_types[condition] = "transformer"
                    feature_names_map[condition] = ["wav2vec2"]
                    print(f"[INFO] {condition}: transformer model loaded from {transformer_dir}")
                    continue

            if (classical_dir / "rf.pkl").exists():
                m = _load_rf(classical_dir)
                if m is not None:
                    models[condition] = m
                    model_types[condition] = "classical"
                    feature_names_map[condition] = _load_feature_names(classical_dir)
                    print(f"[INFO] {condition}: classical RF loaded from {classical_dir}")
                    continue

            models[condition] = None
            print(
                f"[WARN] {condition}: no model found in {root}. "
                f"Train with scripts/train_{condition}_transformer.py"
            )

        return cls(
            parkinson_model=models.get("parkinson"),
            respiratory_model=models.get("respiratory"),
            depression_model=models.get("depression"),
            model_types=model_types,
            feature_names=feature_names_map.get("parkinson", []),
            respiratory_feature_names=feature_names_map.get("respiratory", []),
            depression_feature_names=feature_names_map.get("depression", []),
            threshold=threshold,
            sample_rate=sample_rate,
        )

    @classmethod
    def from_checkpoints(
        cls,
        parkinson_model_dir: Optional[str | Path] = None,
        depression_model_dir: Optional[str | Path] = None,
        respiratory_model_dir: Optional[str | Path] = None,
        model_filename: str = "rf.pkl",
        threshold: float = 0.5,
        sample_rate: int = 16_000,
    ) -> "VocalBiomarkerPipeline":
        """Load classical RF models from checkpoint directories (legacy).

        Each directory is expected to contain ``{model_filename}`` produced by
        :class:`~src.training.trainer.ClassicalTrainer`.
        """

        def _load(directory: Optional[str | Path]) -> object:
            if directory is None:
                return None
            p = Path(directory) / model_filename
            return _load_rf(Path(directory)) if p.exists() else None

        return cls(
            parkinson_model=_load(parkinson_model_dir),
            depression_model=_load(depression_model_dir),
            respiratory_model=_load(respiratory_model_dir),
            model_types={},   # all classical
            threshold=threshold,
            sample_rate=sample_rate,
        )


# ---------------------------------------------------------------------------
# Private loaders
# ---------------------------------------------------------------------------

def _load_transformer(directory: Path, device: str = "cpu"):
    """Load a Wav2Vec2Classifier from *directory*.  Returns None on failure."""
    try:
        import torch
        from src.models.wav2vec2_classifier import Wav2Vec2Classifier
        model = Wav2Vec2Classifier.load(directory, map_location=device)
        model = model.to(device)
        model.eval()
        return model
    except Exception as exc:
        print(f"[WARN] Failed to load transformer from {directory}: {exc}")
        return None


def _load_rf(directory: Path):
    """Load an sklearn RF pipeline from *directory*.  Returns None on failure."""
    rf_path = directory / "rf.pkl"
    if not rf_path.exists():
        return None
    try:
        with open(rf_path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"[WARN] Failed to load RF from {rf_path}: {exc}")
        return None


def _load_feature_names(directory: Path) -> list[str]:
    feat_path = directory / "feature_names.json"
    if not feat_path.exists():
        return []
    try:
        with open(feat_path) as f:
            return json.load(f)
    except Exception:
        return []
