"""Core acoustic feature extractor for vocal biomarker analysis.

Extracts ~63 scalar features per audio clip:
  - 13 MFCCs × (mean, std, delta_mean) = 39
  - F0 mean / std / range = 3
  - Jitter, Shimmer, HNR = 3
  - Spectral centroid (mean, std), rolloff (mean), flatness (mean) = 4
  - ZCR (mean, std) = 2
  - 12 chroma means = 12
Total: 63 features

Uses librosa.yin for pitch tracking (fast, deterministic). pyin was 10–20×
slower and unnecessary for these biomarker features.

All values are float32, NaN-safe, and reproducibly ordered (sorted by name).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import librosa
except ImportError as e:  # pragma: no cover
    raise ImportError("librosa is required: pip install librosa") from e


@dataclass
class FeatureBundle:
    """Holds both a named dict (for SHAP / explainability) and a flat vector."""

    features: dict[str, float]
    vector: np.ndarray        # float32, shape (n_features,)
    feature_names: list[str]  # parallel to vector
    sample_rate: int
    duration_s: float

    def __len__(self) -> int:
        return len(self.vector)


class FeatureExtractor:
    """Extract acoustic biomarker features from an audio file or waveform.

    Parameters
    ----------
    sr : int
        Target sample rate. Audio is resampled if needed.
    n_mfcc : int
        Number of MFCC coefficients.
    hop_length : int
        Hop length in samples (default 160 = 10 ms at 16 kHz).
    frame_length : int
        Analysis frame length in samples (default 400 = 25 ms at 16 kHz).
    f0_min, f0_max : float
        F0 search range in Hz.
    max_duration_s : float or None
        Trim audio to this many seconds before feature extraction.
        None = no trimming. Recommended: 10.0 — enough to capture stable
        phonation while avoiding the slow processing of multi-minute recordings.
    """

    def __init__(
        self,
        sr: int = 16_000,
        n_mfcc: int = 13,
        hop_length: int = 160,
        frame_length: int = 400,
        f0_min: float = 80.0,   # 80 Hz covers adult speech (male ~85-180, female ~165-255)
        f0_max: float = 400.0,  # 400 Hz well above typical speech
        max_duration_s: Optional[float] = 10.0,
    ) -> None:
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.max_duration_s = max_duration_s

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def from_file(self, path: str | Path) -> FeatureBundle:
        """Load audio from *path* and extract features."""
        duration_limit = self.max_duration_s
        y, sr = librosa.load(str(path), sr=self.sr, mono=True, duration=duration_limit)
        return self._extract(y, sr)

    def from_waveform(self, y: np.ndarray, sr: Optional[int] = None) -> FeatureBundle:
        """Extract features from a pre-loaded waveform *y* (float32/64)."""
        if sr is not None and sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
        y = y.astype(np.float32)
        if self.max_duration_s is not None:
            max_samples = int(self.max_duration_s * self.sr)
            y = y[:max_samples]
        return self._extract(y, self.sr)

    @property
    def n_features(self) -> int:
        return self.n_mfcc * 3 + 3 + 3 + 4 + 2 + 12

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract(self, y: np.ndarray, sr: int) -> FeatureBundle:
        duration = len(y) / sr
        feats: dict[str, float] = {}

        # ── MFCCs ──────────────────────────────────────────────────────
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=self.n_mfcc,
            hop_length=self.hop_length, n_fft=self.frame_length,
        )
        delta_mfcc = librosa.feature.delta(mfcc)
        for i in range(self.n_mfcc):
            feats[f"mfcc_{i:02d}_mean"]       = float(np.nanmean(mfcc[i]))
            feats[f"mfcc_{i:02d}_std"]        = float(np.nanstd(mfcc[i]))
            feats[f"mfcc_{i:02d}_delta_mean"] = float(np.nanmean(delta_mfcc[i]))

        # ── Pitch (F0) via YIN — fast deterministic pitch tracker ────────
        # pyin is 10-20× slower and unnecessary for these scalar biomarkers.
        # YIN needs frame_length >= 4 * sr/fmin for reliable detection;
        # we use 2048 samples (128 ms) regardless of the MFCC frame size.
        f0 = librosa.yin(
            y, fmin=self.f0_min, fmax=self.f0_max, sr=sr,
            hop_length=self.hop_length, frame_length=2048,
        )
        # YIN returns 0.0 for unvoiced frames; filter those out
        f0_voiced = f0[f0 > self.f0_min * 0.9]

        feats["f0_mean"]  = float(np.mean(f0_voiced))  if len(f0_voiced) > 0 else 0.0
        feats["f0_std"]   = float(np.std(f0_voiced))   if len(f0_voiced) > 0 else 0.0
        feats["f0_range"] = float(np.ptp(f0_voiced))   if len(f0_voiced) > 1 else 0.0

        # ── Jitter (cycle-to-cycle F0 period variation) ─────────────────
        feats["jitter"] = self._jitter(f0_voiced)

        # ── Shimmer (amplitude variation between consecutive RMS frames) ─
        rms = librosa.feature.rms(
            y=y, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]
        feats["shimmer"] = (
            float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-9))
            if len(rms) > 1 else 0.0
        )

        # ── HNR (harmonics-to-noise ratio, autocorrelation-based) ───────
        feats["hnr"] = self._hnr(y, sr)

        # ── Spectral features ───────────────────────────────────────────
        sc = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=self.hop_length
        )[0]
        feats["spectral_centroid_mean"] = float(np.mean(sc))
        feats["spectral_centroid_std"]  = float(np.std(sc))

        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, hop_length=self.hop_length
        )[0]
        feats["spectral_rolloff_mean"] = float(np.mean(rolloff))

        flatness = librosa.feature.spectral_flatness(
            y=y, hop_length=self.hop_length
        )[0]
        feats["spectral_flatness_mean"] = float(np.mean(flatness))

        # ── Zero-crossing rate ──────────────────────────────────────────
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]
        feats["zcr_mean"] = float(np.mean(zcr))
        feats["zcr_std"]  = float(np.std(zcr))

        # ── Chroma ──────────────────────────────────────────────────────
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, hop_length=self.hop_length
        )
        for i in range(12):
            feats[f"chroma_{i:02d}_mean"] = float(np.mean(chroma[i]))

        # ── Build flat vector (sorted for reproducibility) ───────────────
        names  = sorted(feats.keys())
        vector = np.array([feats[k] for k in names], dtype=np.float32)
        np.nan_to_num(vector, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        return FeatureBundle(
            features=feats,
            vector=vector,
            feature_names=names,
            sample_rate=sr,
            duration_s=duration,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _jitter(f0_voiced: np.ndarray) -> float:
        if len(f0_voiced) < 2:
            return 0.0
        safe = f0_voiced[f0_voiced > 0]
        if len(safe) < 2:
            return 0.0
        periods = 1.0 / safe
        return float(np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-9))

    def _hnr(self, y: np.ndarray, sr: int) -> float:
        if len(y) < 2:
            return 0.0
        ac = np.correlate(y, y, mode="full")
        ac = ac[len(ac) // 2:]
        min_lag = max(1, int(sr / self.f0_max))
        max_lag = min(len(ac) - 1, int(sr / self.f0_min))
        if max_lag <= min_lag:
            return 0.0
        peak_offset = int(np.argmax(ac[min_lag: max_lag + 1]))
        peak = min_lag + peak_offset
        denom = ac[0] - ac[peak] + 1e-9
        return float(10.0 * np.log10(ac[peak] / denom + 1e-9)) if denom > 0 else 0.0
