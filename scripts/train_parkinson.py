#!/usr/bin/env python3
"""Train and evaluate Parkinson's disease classifiers.

Runs classical SVM + RF baselines with LOOCV evaluation.
Transformer fine-tuning is handled separately (see scripts/train_transformer.py).

Usage
-----
    # With default config:
    python scripts/train_parkinson.py

    # Override dataset root or tasks:
    python scripts/train_parkinson.py dataset.root=/path/to/corpus dataset.tasks=[B1,B2]

    # Tasks only (pass as comma-separated in shell):
    python scripts/train_parkinson.py 'dataset.tasks=[VA,VE,VI,VO,VU]'

Output
------
    results/parkinson/classical/
        svm.pkl
        rf.pkl
        metrics.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure src/ is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.parkinson_loader import ParkinsonLoader
from src.training.trainer import ClassicalTrainer, TrainingConfig


@hydra.main(config_path="../configs", config_name="parkinson", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, cfg.output.log_level, logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # ── Load dataset ───────────────────────────────────────────────────
    tasks = set(cfg.dataset.tasks) if cfg.dataset.tasks else None
    loader = ParkinsonLoader(
        tasks=tasks,
        include_young_controls=cfg.dataset.include_young_controls,
    )

    root = Path(cfg.dataset.root)
    if not root.is_absolute():
        # Resolve relative to repo root (one level above scripts/)
        root = Path(__file__).parent.parent / root

    logger.info("Loading dataset from: %s", root)
    samples = loader.load_samples(root)
    logger.info(
        "Loaded %d recordings  (PD=%d  HC=%d)",
        len(samples),
        sum(s.label == 1 for s in samples),
        sum(s.label == 0 for s in samples),
    )

    if not samples:
        logger.error("No samples found — check dataset root and task filter.")
        sys.exit(1)

    # ── Build training config ───────────────────────────────────────────
    train_cfg = TrainingConfig(
        sample_rate=cfg.features.sample_rate,
        n_mfcc=cfg.features.n_mfcc,
        svm_C=cfg.classical.svm_C,
        rf_n_estimators=cfg.classical.rf_n_estimators,
        seed=cfg.training.seed,
        val_ratio=cfg.training.val_ratio,
        test_ratio=cfg.training.test_ratio,
        output_dir=str(Path(cfg.output.dir) / "classical"),
        save_model=cfg.output.save_model,
    )

    # ── Train ───────────────────────────────────────────────────────────
    trainer = ClassicalTrainer(train_cfg)
    result = trainer.run(samples)

    # ── Save metrics ────────────────────────────────────────────────────
    out_dir = Path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "val_svm":   result.val_metrics_svm.to_dict()  if result.val_metrics_svm  else None,
        "val_rf":    result.val_metrics_rf.to_dict()   if result.val_metrics_rf   else None,
        "test_svm":  result.test_metrics_svm.to_dict() if result.test_metrics_svm else None,
        "test_rf":   result.test_metrics_rf.to_dict()  if result.test_metrics_rf  else None,
        "loocv_svm": result.loocv_metrics_svm.to_dict() if result.loocv_metrics_svm else None,
        "loocv_rf":  result.loocv_metrics_rf.to_dict()  if result.loocv_metrics_rf  else None,
        "top10_features_rf": sorted(
            result.feature_importances_rf.items(), key=lambda x: -x[1]
        )[:10],
        "elapsed_s": result.elapsed_s,
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info("Metrics saved to %s", metrics_path)

    # ── Pretty-print summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PARKINSON'S DETECTION — RESULTS SUMMARY")
    print("=" * 60)
    for label, m in [
        ("LOOCV SVM ", result.loocv_metrics_svm),
        ("LOOCV RF  ", result.loocv_metrics_rf),
        ("Test  SVM ", result.test_metrics_svm),
        ("Test  RF  ", result.test_metrics_rf),
    ]:
        if m:
            print(f"  {label}: {m}")
    print("\nTop-5 features (RF importance):")
    for feat, imp in sorted(
        result.feature_importances_rf.items(), key=lambda x: -x[1]
    )[:5]:
        print(f"    {feat:<35s}  {imp:.4f}")
    print(f"\nElapsed: {result.elapsed_s:.1f}s")
    print("=" * 60)
    print("\nDISCLAIMER: Research / educational tool only. "
          "NOT for clinical diagnosis.\n")


if __name__ == "__main__":
    main()
