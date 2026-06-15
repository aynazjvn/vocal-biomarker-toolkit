"""Fine-tuned transformer classifier for vocal biomarker tasks.

Wraps Wav2Vec2 / HuBERT from HuggingFace for binary audio classification.
The architecture adds a single linear classification head on top of the
mean-pooled encoder output.

Status: STUB — full implementation in next iteration.
        See configs/parkinson.yaml [transformer] section for hyperparameters.

Example
-------
    from src.models.transformer_classifier import TransformerClassifier

    model = TransformerClassifier.from_pretrained(
        "facebook/wav2vec2-base", num_labels=2
    )
    logits = model(waveform_tensor)   # shape (batch, 2)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers import (
        AutoConfig,
        AutoFeatureExtractor,
        AutoModelForAudioClassification,
    )
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


class TransformerClassifier(nn.Module):
    """Thin wrapper around a HuggingFace AudioClassification model.

    Args:
        model_name: HuggingFace model ID, e.g. "facebook/wav2vec2-base".
        num_labels: Number of output classes (2 for binary screening).
        sample_rate: Expected waveform sample rate (must match model config).
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        num_labels: int = 2,
        sample_rate: int = 16_000,
    ) -> None:
        super().__init__()
        if not _HF_AVAILABLE:
            raise ImportError(
                "transformers package required: pip install transformers"
            )
        self.sample_rate = sample_rate
        self.config = AutoConfig.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.model = AutoModelForAudioClassification.from_pretrained(
            model_name, config=self.config, ignore_mismatched_sizes=True
        )
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    # ------------------------------------------------------------------

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            input_values: Raw waveform, shape (batch, time).
            attention_mask: Optional padding mask, shape (batch, time).

        Returns:
            Logits tensor, shape (batch, num_labels).
        """
        out = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        return out.logits

    def predict_proba(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper — returns softmax probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(waveform)
        return torch.softmax(logits, dim=-1)

    # ------------------------------------------------------------------

    def preprocess(self, waveforms: list, padding: bool = True) -> dict:
        """Tokenise a list of numpy waveforms for model input."""
        return self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=padding,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_labels: int = 2,
        sample_rate: int = 16_000,
    ) -> "TransformerClassifier":
        return cls(model_name=model_name, num_labels=num_labels, sample_rate=sample_rate)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.feature_extractor.save_pretrained(path)

    @classmethod
    def load(cls, path: str | Path, num_labels: int = 2) -> "TransformerClassifier":
        path = str(path)
        obj = cls.__new__(cls)
        super(TransformerClassifier, obj).__init__()
        obj.model = AutoModelForAudioClassification.from_pretrained(path)
        obj.feature_extractor = AutoFeatureExtractor.from_pretrained(path)
        obj.sample_rate = obj.feature_extractor.sampling_rate
        return obj
