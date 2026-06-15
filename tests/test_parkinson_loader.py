"""Unit tests for the Parkinson's disease data loader.

Two test layers:
  1. Synthetic fixture tests — build a tiny fake corpus directory to verify
     the loader logic without needing the real dataset.
  2. Integration smoke test — skipped unless PARKINSON_DATA_ROOT env var
     points to the actual corpus.
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

from src.data.parkinson_loader import (
    LABEL_HC,
    LABEL_PD,
    ParkinsonLoader,
    subject_split,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _write_dummy_wav(path: Path, duration_s: float = 0.5, sr: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(sr * duration_s)
    data = (np.random.randn(n_samples) * 0.1 * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


@pytest.fixture(scope="module")
def fake_corpus(tmp_path_factory) -> Path:
    """Create a minimal fake Italian Parkinson's corpus directory."""
    root = tmp_path_factory.mktemp("corpus")

    # 2 young healthy controls, 2 files each
    for subj in ["Alice", "Bob"]:
        for task in ["B1", "VA"]:
            _write_dummy_wav(root / "15 Young Healthy Control" / subj / f"{task}test.wav")

    # 2 elderly healthy controls
    for subj in ["Carol", "Dave"]:
        _write_dummy_wav(root / "22 Elderly Healthy Control" / subj / "B1test.wav")

    # 3 PD patients across two severity bins
    for subj in ["Eve", "Frank"]:
        _write_dummy_wav(
            root / "28 People with Parkinson's disease" / "1-5" / subj / "B1test.wav"
        )
    for subj in ["Grace"]:
        _write_dummy_wav(
            root / "28 People with Parkinson's disease" / "6-10" / subj / "B1test.wav"
        )

    return root


# ------------------------------------------------------------------
# Loader correctness
# ------------------------------------------------------------------

class TestParkinsonLoaderFakeCorpus:
    def test_loads_all_samples(self, fake_corpus: Path):
        loader = ParkinsonLoader()
        samples = loader.load_samples(fake_corpus)
        assert len(samples) > 0

    def test_label_distribution(self, fake_corpus: Path):
        loader = ParkinsonLoader()
        samples = loader.load_samples(fake_corpus)
        labels = [s.label for s in samples]
        assert LABEL_PD in labels
        assert LABEL_HC in labels

    def test_all_paths_exist(self, fake_corpus: Path):
        loader = ParkinsonLoader()
        for sample in loader.load_samples(fake_corpus):
            assert sample.path.exists(), f"Missing: {sample.path}"

    def test_task_filter(self, fake_corpus: Path):
        loader = ParkinsonLoader(tasks={"B1"})
        samples = loader.load_samples(fake_corpus)
        for s in samples:
            assert s.task == "B1"

    def test_task_filter_excludes_va(self, fake_corpus: Path):
        loader = ParkinsonLoader(tasks={"B1"})
        samples = loader.load_samples(fake_corpus)
        assert not any(s.task == "VA" for s in samples)

    def test_exclude_young_controls(self, fake_corpus: Path):
        loader = ParkinsonLoader(include_young_controls=False)
        samples = loader.load_samples(fake_corpus)
        assert not any(s.metadata["age_group"] == "young_healthy" for s in samples)

    def test_updrs_range_parsed(self, fake_corpus: Path):
        loader = ParkinsonLoader()
        samples = loader.load_samples(fake_corpus)
        pd_samples = [s for s in samples if s.label == LABEL_PD]
        updrs_values = {s.metadata["updrs_range"] for s in pd_samples}
        assert "1-5" in updrs_values or "6-10" in updrs_values

    def test_subject_ids_set(self, fake_corpus: Path):
        loader = ParkinsonLoader()
        samples = loader.load_samples(fake_corpus)
        for s in samples:
            assert s.subject_id, "subject_id should not be empty"

    def test_condition_field(self, fake_corpus: Path):
        loader = ParkinsonLoader()
        for s in loader.load_samples(fake_corpus):
            assert s.condition == "parkinson"

    def test_missing_root_raises(self, tmp_path: Path):
        loader = ParkinsonLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_samples(tmp_path / "nonexistent")


# ------------------------------------------------------------------
# Subject-level split
# ------------------------------------------------------------------

class TestSubjectSplit:
    def test_no_subject_leakage(self, fake_corpus: Path):
        loader = ParkinsonLoader()
        samples = loader.load_samples(fake_corpus)
        train, val, test = subject_split(samples, seed=42)

        train_ids = {s.subject_id for s in train}
        val_ids   = {s.subject_id for s in val}
        test_ids  = {s.subject_id for s in test}

        assert train_ids.isdisjoint(test_ids), "Subject leaked into train and test"
        assert train_ids.isdisjoint(val_ids),  "Subject leaked into train and val"
        assert val_ids.isdisjoint(test_ids),   "Subject leaked into val and test"

    def test_all_samples_assigned(self, fake_corpus: Path):
        loader = ParkinsonLoader()
        samples = loader.load_samples(fake_corpus)
        train, val, test = subject_split(samples, seed=42)
        assert len(train) + len(val) + len(test) == len(samples)

    def test_both_labels_in_train(self, fake_corpus: Path):
        loader = ParkinsonLoader()
        samples = loader.load_samples(fake_corpus)
        if len({s.subject_id for s in samples if s.label == LABEL_PD}) < 3:
            pytest.skip("Too few PD subjects for split with coverage guarantee")
        train, _, _ = subject_split(samples, seed=42)
        labels = {s.label for s in train}
        assert LABEL_PD in labels
        assert LABEL_HC in labels


# ------------------------------------------------------------------
# Integration test against real corpus (skipped by default)
# ------------------------------------------------------------------

REAL_CORPUS = os.environ.get("PARKINSON_DATA_ROOT")


@pytest.mark.skipif(
    not REAL_CORPUS,
    reason="Set PARKINSON_DATA_ROOT env var to run integration test",
)
class TestParkinsonLoaderRealCorpus:
    def test_loads_expected_subjects(self):
        loader = ParkinsonLoader()
        samples = loader.load_samples(REAL_CORPUS)
        subject_ids = {s.subject_id for s in samples}
        # Corpus has 65 subjects total
        assert len(subject_ids) >= 60, f"Only found {len(subject_ids)} subjects"

    def test_pd_hc_balance(self):
        loader = ParkinsonLoader()
        samples = loader.load_samples(REAL_CORPUS)
        n_pd = sum(1 for s in samples if s.label == LABEL_PD)
        n_hc = sum(1 for s in samples if s.label == LABEL_HC)
        assert n_pd > 0 and n_hc > 0
        # PD:HC ratio should be roughly 28:37 (~0.75)
        ratio = n_pd / (n_hc + 1e-9)
        assert 0.3 < ratio < 2.5, f"Unexpected PD/HC ratio: {ratio:.2f}"
