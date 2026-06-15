"""Unit tests for the acoustic feature extractor.

These tests use synthetically generated audio (sine wave + noise) so they
run without the dataset being present and without network access.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features.audio_features import FeatureBundle, FeatureExtractor

SR = 16_000
DURATION = 2.0  # seconds


def _sine_wave(freq: float = 220.0, duration: float = DURATION, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _white_noise(duration: float = DURATION, sr: int = SR, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(sr * duration)).astype(np.float32) * 0.1


@pytest.fixture(scope="module")
def extractor() -> FeatureExtractor:
    return FeatureExtractor(sr=SR)


@pytest.fixture(scope="module")
def sine_bundle(extractor: FeatureExtractor) -> FeatureBundle:
    return extractor.from_waveform(_sine_wave())


# ------------------------------------------------------------------
# Shape and type tests
# ------------------------------------------------------------------

class TestFeatureBundleShape:
    def test_vector_is_1d_float32(self, sine_bundle: FeatureBundle):
        assert sine_bundle.vector.ndim == 1
        assert sine_bundle.vector.dtype == np.float32

    def test_feature_names_parallel_to_vector(self, sine_bundle: FeatureBundle):
        assert len(sine_bundle.feature_names) == len(sine_bundle.vector)

    def test_feature_names_sorted(self, sine_bundle: FeatureBundle):
        assert sine_bundle.feature_names == sorted(sine_bundle.feature_names)

    def test_expected_feature_count(self, sine_bundle: FeatureBundle):
        # 13 MFCCs × 3 stats + 3 F0 + 3 (jitter/shimmer/hnr) + 4 spectral + 2 ZCR + 12 chroma
        assert len(sine_bundle.vector) == 63

    def test_no_nan_or_inf(self, sine_bundle: FeatureBundle):
        assert np.all(np.isfinite(sine_bundle.vector)), (
            f"Found non-finite values: {sine_bundle.vector[~np.isfinite(sine_bundle.vector)]}"
        )

    def test_duration_approximate(self, sine_bundle: FeatureBundle):
        assert abs(sine_bundle.duration_s - DURATION) < 0.05

    def test_sample_rate_stored(self, sine_bundle: FeatureBundle):
        assert sine_bundle.sample_rate == SR


# ------------------------------------------------------------------
# Feature plausibility tests
# ------------------------------------------------------------------

class TestFeaturePlausibility:
    def test_mfcc_means_present(self, sine_bundle: FeatureBundle):
        mfcc_keys = [k for k in sine_bundle.features if "mfcc" in k and "mean" in k]
        assert len(mfcc_keys) >= 13

    def test_f0_nonzero_for_sine(self, sine_bundle: FeatureBundle):
        # A pure sine should have a clear F0
        assert sine_bundle.features["f0_mean"] > 0

    def test_zcr_low_for_sine(self, extractor: FeatureExtractor):
        # A low-frequency sine has very few zero crossings
        y = _sine_wave(freq=110.0)
        bundle = extractor.from_waveform(y)
        assert bundle.features["zcr_mean"] < 0.1

    def test_zcr_high_for_noise(self, extractor: FeatureExtractor):
        y = _white_noise()
        bundle = extractor.from_waveform(y)
        assert bundle.features["zcr_mean"] > 0.1

    def test_shimmer_positive(self, sine_bundle: FeatureBundle):
        assert sine_bundle.features["shimmer"] >= 0

    def test_jitter_positive(self, sine_bundle: FeatureBundle):
        assert sine_bundle.features["jitter"] >= 0


# ------------------------------------------------------------------
# Waveform vs file path API
# ------------------------------------------------------------------

class TestAPIConsistency:
    def test_from_waveform_matches_from_file(self, extractor: FeatureExtractor, tmp_path):
        import soundfile as sf

        y = _sine_wave()
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), y, SR, subtype="FLOAT")

        bundle_file = extractor.from_file(wav_path)
        bundle_wave = extractor.from_waveform(y)

        np.testing.assert_allclose(
            bundle_file.vector, bundle_wave.vector, rtol=1e-4, atol=1e-4,
            err_msg="from_file and from_waveform should produce identical features",
        )

    def test_different_signals_produce_different_features(self, extractor: FeatureExtractor):
        b1 = extractor.from_waveform(_sine_wave(freq=220.0))
        b2 = extractor.from_waveform(_sine_wave(freq=440.0))
        assert not np.allclose(b1.vector, b2.vector)

    def test_resampling(self, extractor: FeatureExtractor):
        """from_waveform should accept audio at a different sample rate."""
        y = _sine_wave(sr=44_100)
        bundle = extractor.from_waveform(y, sr=44_100)
        assert bundle.sample_rate == SR
        assert len(bundle.vector) == 63


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_silent_audio(self, extractor: FeatureExtractor):
        y = np.zeros(SR * 2, dtype=np.float32)
        bundle = extractor.from_waveform(y)
        assert np.all(np.isfinite(bundle.vector))

    def test_very_short_audio(self, extractor: FeatureExtractor):
        y = np.random.randn(SR // 10).astype(np.float32) * 0.1
        bundle = extractor.from_waveform(y)
        assert np.all(np.isfinite(bundle.vector))
