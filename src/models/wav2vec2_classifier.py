"""Model-agnostic binary classifier for vocal biomarker tasks.

Architecture:
    Raw audio (16 kHz, float32)
        → AutoModel encoder  (CNN feature extractor frozen; top layers fine-tuned)
        → Mean pool over time frames
        → Dropout(0.1)
        → Linear(hidden_size, 1)  →  scalar logit  →  sigmoid  →  probability

Supported backbones (any HuggingFace encoder that accepts raw input_values):
    facebook/wav2vec2-base          — general speech
    microsoft/wavlm-base            — richer acoustic repr, better for non-speech audio
    superb/wav2vec2-base-superb-er  — emotion-pretrained, better for affect/depression

Checkpoints are saved as two files:
    model.pt     — full state dict (all parameters, including frozen backbone)
    config.json  — {model_name, freeze_feature_extractor, freeze_encoder_layers}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

try:
    from transformers import AutoModel
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


def _load_encoder(model_name: str):
    """Load the base encoder from *model_name* via AutoModel.

    Handles two cases:
    1. AutoModel returns the base encoder directly (e.g. Wav2Vec2Model, WavLMModel).
    2. AutoModel returns a task-specific model (e.g. Wav2Vec2ForSequenceClassification
       for emotion-pretrained checkpoints like superb/wav2vec2-base-superb-er).
       In that case, the inner encoder is extracted and the classification head discarded.
    """
    backbone = AutoModel.from_pretrained(model_name)
    # Strip task-specific head if present
    for attr in ("wav2vec2", "wavlm", "hubert", "data2vec_audio", "sew", "unispeech"):
        if hasattr(backbone, attr):
            candidate = getattr(backbone, attr)
            if hasattr(candidate, "config") and hasattr(candidate.config, "hidden_size"):
                return candidate
    return backbone


def _make_encoder_from_config(config):
    """Instantiate an encoder with random weights from *config* (no download)."""
    from transformers import AutoModel
    backbone = AutoModel.from_config(config)
    for attr in ("wav2vec2", "wavlm", "hubert", "data2vec_audio", "sew", "unispeech"):
        if hasattr(backbone, attr):
            candidate = getattr(backbone, attr)
            if hasattr(candidate, "config") and hasattr(candidate.config, "hidden_size"):
                return candidate
    return backbone


class Wav2Vec2Classifier(nn.Module):
    """Binary classifier on top of a frozen/partially-frozen speech encoder.

    The name ``Wav2Vec2Classifier`` is kept for checkpoint backward compatibility.
    Internally the backbone can be any AutoModel-compatible encoder.

    Args:
        model_name: HuggingFace model ID.
        freeze_feature_extractor: Freeze the CNN feature extractor.
        freeze_encoder_layers: Number of transformer encoder layers to freeze
            (from layer 0 upward).
        dropout: Dropout before the classification head.
    """

    MODEL_NAME_DEFAULT = "facebook/wav2vec2-base"

    def __init__(
        self,
        model_name: str = MODEL_NAME_DEFAULT,
        freeze_feature_extractor: bool = True,
        freeze_encoder_layers: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not _HF_AVAILABLE:
            raise ImportError("pip install transformers")

        self.model_name = model_name
        self.freeze_feature_extractor = freeze_feature_extractor
        self.freeze_encoder_layers = freeze_encoder_layers

        # Stored as self.wav2vec2 to keep state-dict keys backward-compatible
        # with existing Parkinson's checkpoint (keys: wav2vec2.*, classifier.*).
        self.wav2vec2 = _load_encoder(model_name)

        if freeze_feature_extractor and hasattr(self.wav2vec2, "feature_extractor"):
            self.wav2vec2.feature_extractor._freeze_parameters()

        if hasattr(self.wav2vec2, "encoder") and hasattr(self.wav2vec2.encoder, "layers"):
            n_layers = len(self.wav2vec2.encoder.layers)
            for i, layer in enumerate(self.wav2vec2.encoder.layers):
                if i < min(freeze_encoder_layers, n_layers):
                    for p in layer.parameters():
                        p.requires_grad_(False)

        hidden_size: int = self.wav2vec2.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    # ------------------------------------------------------------------
    # Forward

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_values: Raw waveform, shape ``(batch, time)``, float32.
            attention_mask: Optional padding mask, shape ``(batch, time)``.

        Returns:
            Scalar logits, shape ``(batch,)``.
        """
        out = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
        )
        hidden = out.last_hidden_state  # (B, T, hidden)

        if attention_mask is not None and hasattr(self.wav2vec2, "_get_feature_vector_attention_mask"):
            frame_mask = self.wav2vec2._get_feature_vector_attention_mask(
                hidden.shape[1], attention_mask
            )
            frame_mask = frame_mask.unsqueeze(-1).float()
            pooled = (hidden * frame_mask).sum(1) / frame_mask.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)

        pooled = self.dropout(pooled)
        return self.classifier(pooled).squeeze(-1)  # (B,)

    # ------------------------------------------------------------------
    # Inference helpers

    def predict_proba(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_values, attention_mask)
        return torch.sigmoid(logits)

    def predict_proba_with_grad(
        self,
        input_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (prob, |gradient|) for gradient saliency attribution."""
        self.eval()
        x = input_values.detach().requires_grad_(True)
        logit = self.forward(x)
        prob = torch.sigmoid(logit)
        prob.sum().backward()
        grad = x.grad.abs()
        return prob.detach(), grad.detach()

    # ------------------------------------------------------------------
    # Persistence

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), directory / "model.pt")
        config = {
            "model_name": self.model_name,
            "freeze_feature_extractor": self.freeze_feature_extractor,
            "freeze_encoder_layers": self.freeze_encoder_layers,
        }
        with open(directory / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(
        cls,
        directory: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "Wav2Vec2Classifier":
        directory = Path(directory)
        with open(directory / "config.json") as f:
            cfg = json.load(f)
        model = cls(
            model_name=cfg.get("model_name", cls.MODEL_NAME_DEFAULT),
            freeze_feature_extractor=cfg.get("freeze_feature_extractor", True),
            freeze_encoder_layers=cfg.get("freeze_encoder_layers", 8),
        )
        state_dict = torch.load(directory / "model.pt", map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict)
        return model

    # ------------------------------------------------------------------

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
