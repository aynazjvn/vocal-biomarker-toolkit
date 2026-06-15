"""Classical ML baselines: SVM and Random Forest pipelines.

Both pipelines accept a 2-D feature matrix X (n_samples × n_features)
and 1-D integer labels y.  They expose the standard sklearn fit / predict /
predict_proba interface, making them drop-in replaceable with each other.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_svm(
    C: float = 1.0,
    gamma: str = "scale",
) -> Pipeline:
    """RBF-SVM with z-score scaling and Platt calibration.

    Uses ``SVC(probability=True)`` which invokes libsvm's built-in Platt
    scaling — avoids the ``CalibratedClassifierCV`` inner-fold failures that
    occur with small or imbalanced datasets.

    Args:
        C: Regularisation parameter.
        gamma: RBF kernel bandwidth ("scale" = 1 / (n_features × Var(X))).

    Returns:
        sklearn Pipeline with steps [scaler, clf].
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=C, gamma=gamma, probability=True, class_weight="balanced")),
    ])


def build_rf(
    n_estimators: int = 200,
    max_depth: int | None = None,
    seed: int = 42,
) -> Pipeline:
    """Random Forest with standard scaling (for feature-importance consistency).

    RF does not strictly need scaling but we include it so feature importance
    values are comparable with the SVM pipeline.

    Args:
        n_estimators: Number of trees.
        max_depth: Maximum tree depth. ``None`` = grow until leaves are pure.
        seed: Random seed for reproducibility.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",  # handle class imbalance
        )),
    ])


def feature_importances(rf_pipeline: Pipeline) -> dict[str, float]:
    """Extract feature importances from a fitted Random Forest pipeline.

    Args:
        rf_pipeline: A fitted pipeline returned by :func:`build_rf`.

    Returns:
        Dict mapping feature index string to importance value.
    """
    rf = rf_pipeline.named_steps["clf"]
    if not hasattr(rf, "feature_importances_"):
        raise ValueError("Pipeline has not been fitted yet.")
    return {
        f"feature_{i:03d}": float(v)
        for i, v in enumerate(rf.feature_importances_)
    }


def feature_importances_named(
    rf_pipeline: Pipeline,
    feature_names: list[str],
) -> dict[str, float]:
    """Same as :func:`feature_importances` but uses the provided names.

    Args:
        rf_pipeline: Fitted RF pipeline.
        feature_names: List of feature names parallel to X columns.
    """
    rf = rf_pipeline.named_steps["clf"]
    if not hasattr(rf, "feature_importances_"):
        raise ValueError("Pipeline has not been fitted yet.")
    return dict(zip(feature_names, map(float, rf.feature_importances_)))
