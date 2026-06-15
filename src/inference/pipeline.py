"""Unified inference pipeline: raw audio → multi-condition JSON report.

Usage
-----
    from src.inference.pipeline import VocalBiomarkerPipeline

    pipe = VocalBiomarkerPipeline.from_checkpoints(
        parkinson_model_dir="results/parkinson/classical"
    )
    report = pipe.predict("path/to/audio.wav")
    print(report)           # JSON-serialisable dict

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
                "top_features": [
                    {"feature": "jitter", "importance": 0.31},
                    ...
                ]
            },
            "depression":   {"label": "unavailable", "reason": "model not loaded"},
            "respiratory":  {"label": "unavailable", "reason": "model not loaded"}
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


class VocalBiomarkerPipeline:
    """Multi-condition vocal biomarker inference pipeline.

    Load condition models individually; unavailable conditions are reported
    as "unavailable" rather than raising an error.

    Args:
        parkinson_model: Fitted sklearn pipeline for Parkinson's (or None).
        depression_model: Fitted sklearn pipeline for depression (or None).
        respiratory_model: Fitted sklearn pipeline for respiratory (or None).
        feature_names: Feature names parallel to the model's input columns.
        threshold: Default probability threshold for all conditions.
        sample_rate: Audio sample rate for the feature extractor.
    """

    def __init__(
        self,
        parkinson_model=None,
        depression_model=None,
        respiratory_model=None,
        feature_names: Optional[list[str]] = None,
        respiratory_feature_names: Optional[list[str]] = None,
        depression_feature_names: Optional[list[str]] = None,
        threshold: float = 0.5,
        sample_rate: int = 16_000,
    ) -> None:
        self.models = {
            "parkinson":  parkinson_model,
            "depression": depression_model,
            "respiratory": respiratory_model,
        }
        self.feature_names = feature_names or []
        self.feature_names_by_condition: dict[str, list[str]] = {
            "parkinson":   feature_names or [],
            "respiratory": respiratory_feature_names or feature_names or [],
            "depression":  depression_feature_names or feature_names or [],
        }
        self.threshold = threshold
        self.extractor = FeatureExtractor(sr=sample_rate)

    # ------------------------------------------------------------------

    def predict(self, audio_path: str | Path) -> dict:
        """Run all available condition models on *audio_path*.

        Args:
            audio_path: Path to an audio file (any format supported by librosa).

        Returns:
            JSON-serialisable dict following the schema above.
        """
        audio_path = Path(audio_path)
        bundle = self.extractor.from_file(audio_path)
        X = bundle.vector.reshape(1, -1)

        report: dict = {
            "audio_path": str(audio_path),
            "duration_s": round(bundle.duration_s, 3),
            "sample_rate": bundle.sample_rate,
            "conditions": {},
            "disclaimer": DISCLAIMER,
        }

        for condition, model in self.models.items():
            report["conditions"][condition] = self._run_condition(
                condition, model, X, bundle
            )

        return report

    def predict_json(self, audio_path: str | Path, indent: int = 2) -> str:
        """Same as :meth:`predict` but returns a formatted JSON string."""
        return json.dumps(self.predict(audio_path), indent=indent, default=str)

    # ------------------------------------------------------------------

    def _run_condition(
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

            # Feature importances from RF (if available)
            top_features: list[dict] = []
            clf = getattr(model, "named_steps", {}).get("clf")
            if clf is not None and hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
                names = (self.feature_names_by_condition.get(condition)
                         or self.feature_names
                         or [f"feature_{i}" for i in range(len(importances))])
                ranked = sorted(
                    zip(names, importances.tolist()), key=lambda x: -x[1]
                )[:10]
                top_features = [{"feature": n, "importance": round(v, 4)} for n, v in ranked]

            return {
                "score": round(prob, 4),
                "label": label,
                "threshold": self.threshold,
                "top_features": top_features,
            }
        except Exception as exc:
            return {"label": "error", "reason": str(exc)}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

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
        """Load models from checkpoint directories.

        Each directory is expected to contain a ``{model_filename}`` pickle
        produced by :class:`~src.training.trainer.ClassicalTrainer`.

        Args:
            parkinson_model_dir: Directory with Parkinson's model checkpoint.
            depression_model_dir: Directory with depression model checkpoint.
            respiratory_model_dir: Directory with respiratory model checkpoint.
            model_filename: Pickle file name inside each directory.
            threshold: Decision threshold.
            sample_rate: Feature extractor sample rate.
        """

        def _load(directory: Optional[str | Path]) -> object:
            if directory is None:
                return None
            p = Path(directory) / model_filename
            if not p.exists():
                return None
            with open(p, "rb") as f:
                return pickle.load(f)

        return cls(
            parkinson_model=_load(parkinson_model_dir),
            depression_model=_load(depression_model_dir),
            respiratory_model=_load(respiratory_model_dir),
            threshold=threshold,
            sample_rate=sample_rate,
        )
