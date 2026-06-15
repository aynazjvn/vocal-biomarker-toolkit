"""Classical-model training loop for vocal biomarker classifiers.

Usage
-----
    from src.training.trainer import ClassicalTrainer

    trainer = ClassicalTrainer(cfg)
    results = trainer.run(samples, dataset_root)
    print(results.test_metrics)

The trainer handles:
  1. Feature extraction for all samples
  2. Subject-level train/val/test split (no leakage)
  3. Fitting SVM and RF pipelines
  4. LOOCV evaluation on the full dataset
  5. Saving the best model checkpoint
"""

from __future__ import annotations

import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.base_loader import AudioSample
from src.data.parkinson_loader import subject_split
from src.evaluation.metrics import BinaryMetrics, evaluate, loocv_evaluate
from src.features.audio_features import FeatureBundle, FeatureExtractor
from src.models.classical import build_rf, build_svm, feature_importances_named

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    # Feature extractor
    sample_rate: int = 16_000
    n_mfcc: int = 13

    # Classical models
    svm_C: float = 1.0
    rf_n_estimators: int = 200
    seed: int = 42

    # Splits
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Outputs
    output_dir: str = "results/parkinson/classical"
    save_model: bool = True


@dataclass
class TrainingResult:
    val_metrics_svm: Optional[BinaryMetrics] = None
    val_metrics_rf: Optional[BinaryMetrics] = None
    test_metrics_svm: Optional[BinaryMetrics] = None
    test_metrics_rf: Optional[BinaryMetrics] = None
    loocv_metrics_svm: Optional[BinaryMetrics] = None
    loocv_metrics_rf: Optional[BinaryMetrics] = None
    feature_importances_rf: dict[str, float] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


class ClassicalTrainer:
    """Train and evaluate SVM + RF on acoustic features.

    Args:
        cfg: :class:`TrainingConfig` (or an OmegaConf DictConfig with the
             same fields).
    """

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self.extractor = FeatureExtractor(sr=cfg.sample_rate, n_mfcc=cfg.n_mfcc)

    # ------------------------------------------------------------------

    def run(self, samples: list[AudioSample]) -> TrainingResult:
        """Full pipeline: extract features → split → train → evaluate.

        Args:
            samples: All AudioSample objects from the loader.

        Returns:
            :class:`TrainingResult` with all metrics.
        """
        t0 = time.time()
        result = TrainingResult()

        # ── Feature extraction ──────────────────────────────────────────
        logger.info("Extracting features for %d samples …", len(samples))
        bundles = self._extract_features(samples)

        X = np.vstack([b.vector for b in bundles])
        y = np.array([s.label for s in samples], dtype=int)
        subj_ids = [s.subject_id for s in samples]
        feature_names = bundles[0].feature_names
        result.feature_names = feature_names

        logger.info("Feature matrix: %s  classes: %s", X.shape, np.bincount(y))

        # ── Subject-level split ─────────────────────────────────────────
        train_s, val_s, test_s = subject_split(
            samples,
            val_ratio=self.cfg.val_ratio,
            test_ratio=self.cfg.test_ratio,
            seed=self.cfg.seed,
        )
        train_idx = self._index_map(samples, train_s)
        val_idx   = self._index_map(samples, val_s)
        test_idx  = self._index_map(samples, test_s)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        logger.info(
            "Split sizes — train: %d  val: %d  test: %d",
            len(train_idx), len(val_idx), len(test_idx),
        )

        # ── Build and train models ──────────────────────────────────────
        svm = build_svm(C=self.cfg.svm_C)
        rf  = build_rf(n_estimators=self.cfg.rf_n_estimators, seed=self.cfg.seed)

        logger.info("Fitting SVM …")
        svm.fit(X_train, y_train)
        logger.info("Fitting Random Forest …")
        rf.fit(X_train, y_train)

        # ── Held-out evaluation ─────────────────────────────────────────
        result.val_metrics_svm = evaluate(y_val,  svm.predict_proba(X_val)[:, 1])
        result.val_metrics_rf  = evaluate(y_val,  rf.predict_proba(X_val)[:, 1])
        result.test_metrics_svm = evaluate(y_test, svm.predict_proba(X_test)[:, 1])
        result.test_metrics_rf  = evaluate(y_test, rf.predict_proba(X_test)[:, 1])

        logger.info("Val   SVM: %s", result.val_metrics_svm)
        logger.info("Val   RF:  %s", result.val_metrics_rf)
        logger.info("Test  SVM: %s", result.test_metrics_svm)
        logger.info("Test  RF:  %s", result.test_metrics_rf)

        # ── LOOCV on the full dataset (most reliable for small N) ────────
        logger.info("Running Leave-One-Subject-Out CV …")
        svm_loocv = build_svm(C=self.cfg.svm_C)
        rf_loocv  = build_rf(n_estimators=self.cfg.rf_n_estimators, seed=self.cfg.seed)
        result.loocv_metrics_svm = loocv_evaluate(X, y, subj_ids, svm_loocv)
        result.loocv_metrics_rf  = loocv_evaluate(X, y, subj_ids, rf_loocv)
        logger.info("LOOCV SVM: %s", result.loocv_metrics_svm)
        logger.info("LOOCV RF:  %s", result.loocv_metrics_rf)

        # ── Feature importances from RF ──────────────────────────────────
        result.feature_importances_rf = feature_importances_named(rf, feature_names)

        # ── Save checkpoints ─────────────────────────────────────────────
        if self.cfg.save_model:
            out = Path(self.cfg.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "svm.pkl", "wb") as f:
                pickle.dump(svm, f)
            with open(out / "rf.pkl", "wb") as f:
                pickle.dump(rf, f)
            logger.info("Models saved to %s", out)

        result.elapsed_s = time.time() - t0
        return result

    # ------------------------------------------------------------------

    def _extract_features(self, samples: list[AudioSample]) -> list[FeatureBundle]:
        bundles = []
        for i, sample in enumerate(samples):
            try:
                b = self.extractor.from_file(sample.path)
            except Exception as exc:
                logger.warning("Skipping %s: %s", sample.path, exc)
                continue
            bundles.append(b)
            if (i + 1) % 50 == 0:
                logger.info("  … %d / %d", i + 1, len(samples))
        return bundles

    @staticmethod
    def _index_map(all_samples: list[AudioSample], subset: list[AudioSample]) -> list[int]:
        subset_paths = {s.path for s in subset}
        return [i for i, s in enumerate(all_samples) if s.path in subset_paths]
