"""SHAP-based feature attribution for classical vocal biomarker models.

Uses SHAP's KernelExplainer for model-agnostic explanations and
TreeExplainer for Random Forest models (faster, exact).

Usage
-----
    from src.explainability.shap_explainer import SHAPExplainer

    explainer = SHAPExplainer(rf_pipeline, X_background, feature_names)
    attribution = explainer.explain(X_single_sample)
    explainer.waterfall_plot(attribution, save_path="explanation.png")

STATUS: Core scaffold implemented; visualisation helpers in next iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False


@dataclass
class Attribution:
    """Per-prediction feature attributions."""

    feature_names: list[str]
    shap_values: np.ndarray       # shape (n_features,) for single sample
    base_value: float             # model output for an average input
    prediction_prob: float        # final model probability for the positive class
    top_k: int = 10

    def top_features(self) -> list[tuple[str, float]]:
        """Return top-k features sorted by |SHAP value| descending."""
        pairs = list(zip(self.feature_names, self.shap_values.tolist()))
        return sorted(pairs, key=lambda x: -abs(x[1]))[: self.top_k]

    def to_dict(self) -> dict:
        return {
            "base_value": float(self.base_value),
            "prediction_prob": float(self.prediction_prob),
            "top_features": [
                {"feature": name, "shap_value": float(val)}
                for name, val in self.top_features()
            ],
        }


class SHAPExplainer:
    """Wrapper around SHAP explainers for vocal biomarker models.

    For Random Forest pipelines, uses TreeExplainer (fast, exact).
    For SVM and other models, falls back to KernelExplainer (slow, model-agnostic).

    Args:
        pipeline: Fitted sklearn Pipeline.
        X_background: Background dataset for KernelExplainer (n_bg × n_features).
                      Ignored for TreeExplainer.
        feature_names: Names for each column of X.
        n_background_samples: Sub-sample background to this size for speed.
    """

    def __init__(
        self,
        pipeline,
        X_background: np.ndarray,
        feature_names: list[str],
        n_background_samples: int = 50,
    ) -> None:
        if not _SHAP_AVAILABLE:
            raise ImportError("shap package required: pip install shap")

        self.pipeline = pipeline
        self.feature_names = feature_names

        # Check if we can use the fast TreeExplainer
        from sklearn.ensemble import RandomForestClassifier
        clf = pipeline.named_steps.get("clf")
        use_tree = isinstance(clf, RandomForestClassifier)

        # Scale background through the pipeline's scaler (if any)
        scaler = pipeline.named_steps.get("scaler")
        if scaler is not None:
            X_bg_scaled = scaler.transform(X_background)
        else:
            X_bg_scaled = X_background

        if n_background_samples < len(X_bg_scaled):
            idx = np.random.default_rng(42).choice(
                len(X_bg_scaled), n_background_samples, replace=False
            )
            X_bg_scaled = X_bg_scaled[idx]

        if use_tree:
            self._explainer = shap.TreeExplainer(clf, X_bg_scaled)
            self._use_tree = True
        else:
            def _predict_proba(X_scaled):
                return clf.predict_proba(X_scaled)[:, 1]

            self._explainer = shap.KernelExplainer(_predict_proba, X_bg_scaled)
            self._use_tree = False

        self._scaler = scaler

    # ------------------------------------------------------------------

    def explain(self, X: np.ndarray, check_additivity: bool = False) -> Attribution:
        """Compute SHAP attributions for a single sample (1 × n_features).

        Args:
            X: Feature vector, shape (1, n_features) or (n_features,).
            check_additivity: Passed to TreeExplainer for sanity check.

        Returns:
            :class:`Attribution` dataclass.
        """
        X = np.atleast_2d(X)

        if self._scaler is not None:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        if self._use_tree:
            sv = self._explainer.shap_values(
                X_scaled, check_additivity=check_additivity
            )
            # TreeExplainer returns [class0, class1] — take class 1
            shap_vals = sv[1][0] if isinstance(sv, list) else sv[0]
            base = float(self._explainer.expected_value[1]
                         if isinstance(self._explainer.expected_value, (list, np.ndarray))
                         else self._explainer.expected_value)
        else:
            sv = self._explainer.shap_values(X_scaled)
            shap_vals = np.asarray(sv).flatten()
            base = float(self._explainer.expected_value)

        pred_prob = float(self.pipeline.predict_proba(X)[0, 1])

        return Attribution(
            feature_names=self.feature_names,
            shap_values=shap_vals.astype(np.float32),
            base_value=base,
            prediction_prob=pred_prob,
        )

    def summary_plot(
        self,
        X: np.ndarray,
        save_path: Optional[str | Path] = None,
        max_display: int = 20,
    ) -> None:
        """SHAP beeswarm summary plot for a batch of samples.

        Args:
            X: Feature matrix (n_samples × n_features).
            save_path: If given, save the figure to this path.
            max_display: Number of top features to display.
        """
        import matplotlib
        if save_path:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if self._scaler is not None:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        if self._use_tree:
            sv = self._explainer.shap_values(X_scaled)
            sv_pos = sv[1] if isinstance(sv, list) else sv
        else:
            sv_pos = self._explainer.shap_values(X_scaled)

        shap.summary_plot(
            sv_pos, X_scaled,
            feature_names=self.feature_names,
            max_display=max_display,
            show=save_path is None,
        )
        if save_path:
            plt.savefig(str(save_path), bbox_inches="tight", dpi=150)
            plt.close()
