"""Unit tests for the unified inference pipeline.

Tests that:
  - Pipeline initialises cleanly with no models loaded
  - predict() returns a valid schema for any audio
  - Unavailable conditions report correctly
  - predict_json() returns valid JSON
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from src.inference.pipeline import DISCLAIMER, VocalBiomarkerPipeline


def _write_dummy_wav(path: Path, duration_s: float = 1.0, sr: int = 16_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * duration_s)
    data = (np.random.randn(n) * 0.1 * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    return path


@pytest.fixture(scope="module")
def audio_file(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("audio") / "test.wav"
    return _write_dummy_wav(p)


@pytest.fixture(scope="module")
def empty_pipeline() -> VocalBiomarkerPipeline:
    return VocalBiomarkerPipeline()


# ------------------------------------------------------------------

class TestPipelineSchema:
    def test_report_has_required_keys(self, empty_pipeline, audio_file):
        report = empty_pipeline.predict(audio_file)
        for key in ("audio_path", "duration_s", "sample_rate", "conditions", "disclaimer"):
            assert key in report, f"Missing key: {key}"

    def test_conditions_has_all_three(self, empty_pipeline, audio_file):
        report = empty_pipeline.predict(audio_file)
        for cond in ("parkinson", "depression", "respiratory"):
            assert cond in report["conditions"]

    def test_unavailable_conditions(self, empty_pipeline, audio_file):
        report = empty_pipeline.predict(audio_file)
        for cond, res in report["conditions"].items():
            assert res["label"] == "unavailable"
            assert "reason" in res

    def test_disclaimer_present(self, empty_pipeline, audio_file):
        report = empty_pipeline.predict(audio_file)
        assert len(report["disclaimer"]) > 10
        assert report["disclaimer"] == DISCLAIMER

    def test_duration_positive(self, empty_pipeline, audio_file):
        report = empty_pipeline.predict(audio_file)
        assert report["duration_s"] > 0

    def test_sample_rate_correct(self, empty_pipeline, audio_file):
        report = empty_pipeline.predict(audio_file)
        assert report["sample_rate"] == 16_000

    def test_predict_json_valid(self, empty_pipeline, audio_file):
        raw = empty_pipeline.predict_json(audio_file)
        parsed = json.loads(raw)       # must not raise
        assert "conditions" in parsed


class TestPipelineWithMockModel:
    def test_model_score_in_report(self, audio_file):
        from unittest.mock import MagicMock
        import numpy as np

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        mock_model.named_steps = {}  # no RF importances

        pipeline = VocalBiomarkerPipeline(parkinson_model=mock_model)
        report = pipeline.predict(audio_file)
        pk = report["conditions"]["parkinson"]

        assert pk["label"] == "positive"
        assert abs(pk["score"] - 0.7) < 0.01

    def test_low_score_is_negative(self, audio_file):
        from unittest.mock import MagicMock
        import numpy as np

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
        mock_model.named_steps = {}

        pipeline = VocalBiomarkerPipeline(parkinson_model=mock_model)
        report = pipeline.predict(audio_file)
        assert report["conditions"]["parkinson"]["label"] == "negative"


class TestPipelineFromCheckpoints:
    def test_empty_checkpoint_dir_returns_none_models(self, tmp_path):
        pipeline = VocalBiomarkerPipeline.from_checkpoints(
            parkinson_model_dir=tmp_path  # exists but no rf.pkl inside
        )
        assert pipeline.models["parkinson"] is None

    def test_nonexistent_dir_returns_none_models(self):
        pipeline = VocalBiomarkerPipeline.from_checkpoints(
            parkinson_model_dir="/nonexistent/path"
        )
        assert pipeline.models["parkinson"] is None
