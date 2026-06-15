"""Evaluation metrics for binary vocal biomarker classifiers.

Covers:
  - AUROC
  - Sensitivity (recall for the positive class) and specificity
  - F1, precision, accuracy
  - Expected Calibration Error (ECE)
  - Full results dict for logging

All functions accept plain numpy arrays or Python lists.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)


@dataclass
class BinaryMetrics:
    """Structured result of :func:`evaluate`."""

    auroc: float
    sensitivity: float  # = recall for the positive class
    specificity: float
    f1: float
    precision: float
    accuracy: float
    ece: float          # expected calibration error
    threshold: float    # decision threshold used for binary predictions
    n_positive: int
    n_negative: int

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"AUROC={self.auroc:.3f}  "
            f"Sens={self.sensitivity:.3f}  "
            f"Spec={self.specificity:.3f}  "
            f"F1={self.f1:.3f}  "
            f"ECE={self.ece:.3f}  "
            f"(n+={self.n_positive}, n-={self.n_negative})"
        )


def evaluate(
    y_true: np.ndarray | list,
    y_prob: np.ndarray | list,
    threshold: float = 0.5,
    n_cal_bins: int = 10,
) -> BinaryMetrics:
    """Compute a full suite of binary classification metrics.

    Args:
        y_true: Ground-truth labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        threshold: Probability cut-off for hard predictions.
        n_cal_bins: Number of bins for ECE / calibration curve.

    Returns:
        :class:`BinaryMetrics` dataclass.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn + 1e-9))
    specificity = float(tn / (tn + fp + 1e-9))

    return BinaryMetrics(
        auroc=auroc,
        sensitivity=sensitivity,
        specificity=specificity,
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        ece=expected_calibration_error(y_true, y_prob, n_bins=n_cal_bins),
        threshold=threshold,
        n_positive=int(y_true.sum()),
        n_negative=int((1 - y_true).sum()),
    )


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error (ECE).

    ECE measures the weighted average discrepancy between predicted confidence
    and actual accuracy across equal-width probability bins.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        acc  = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def loocv_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: list[str],
    pipeline,
    threshold: float = 0.5,
) -> BinaryMetrics:
    """Leave-One-Subject-Out cross-validation evaluation.

    Essential for small medical datasets (like this corpus with ~65 subjects)
    to avoid data leakage from same-subject recordings appearing in both
    train and test.

    Args:
        X: Feature matrix (n_samples × n_features).
        y: Labels (n_samples,).
        subject_ids: Subject ID for each sample — same-subject samples
                     are held out together.
        pipeline: sklearn-compatible classifier with fit / predict_proba.
        threshold: Decision threshold.

    Returns:
        :class:`BinaryMetrics` over all held-out predictions.
    """
    unique_subjects = list(dict.fromkeys(subject_ids))
    all_true, all_prob = [], []

    for held_out in unique_subjects:
        mask_test  = np.array([s == held_out for s in subject_ids])
        mask_train = ~mask_test

        if mask_test.sum() == 0 or mask_train.sum() == 0:
            continue

        pipeline.fit(X[mask_train], y[mask_train])
        prob = pipeline.predict_proba(X[mask_test])[:, 1]
        all_true.extend(y[mask_test].tolist())
        all_prob.extend(prob.tolist())

    return evaluate(np.array(all_true), np.array(all_prob), threshold=threshold)
