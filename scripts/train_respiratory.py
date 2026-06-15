#!/usr/bin/env python3
"""Train and evaluate respiratory illness classifiers on the Coswara dataset.

Uses vowel-a recordings to match the single-recording inference flow.
Trains SVM + RF with class-weighted loss to handle the healthy/sick imbalance.

Usage
-----
    python scripts/train_respiratory.py

    # Custom data root:
    python scripts/train_respiratory.py --data data/coswara/Extracted_data

Output
------
    results/respiratory/classical/
        rf.pkl
        svm.pkl
        metrics.json
        feature_names.json
        feature_importances.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data.respiratory_loader import CoswaraLoader
from src.evaluation.metrics import loocv_evaluate
from src.features.audio_features import FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

POSITIVE_STATUSES = {
    "positive_mild",
    "positive_moderate",
    "positive_asymp",
    "resp_illness_not_identified",
}

CACHE_FILE = Path("results/respiratory/features.npz")
OUT_DIR    = Path("results/respiratory/classical")


def build_svm(C: float = 10.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(C=C, kernel="rbf", probability=True,
                    class_weight="balanced", random_state=42)),
    ])


def build_rf(n_estimators: int = 300) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/coswara/Extracted_data",
        help="Path to Extracted_data folder (after running data/coswara/extract_data.py)",
    )
    parser.add_argument("--tasks", nargs="+", default=["vowel-a"],
                        help="Audio tasks to include (default: vowel-a)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cached features and re-extract")
    args = parser.parse_args()

    root = Path(args.data)
    if not root.exists():
        logger.error("Data directory not found: %s", root)
        logger.error("Run:  cd data/coswara && python extract_data.py")
        sys.exit(1)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset …")
    loader = CoswaraLoader(tasks=args.tasks)
    samples = loader.load_samples(root)

    if not samples:
        logger.error("No samples found in %s", root)
        sys.exit(1)

    counts = Counter(s.label for s in samples)
    subj_ids = list({s.subject_id for s in samples})
    print(f"     {len(samples)} recordings | positive={counts[1]}  healthy={counts[0]} | {len(subj_ids)} subjects")

    # ── Feature extraction (with cache) ──────────────────────────────────────
    print("\n[2/4] Extracting features …")
    extractor = FeatureExtractor(sr=16_000, n_mfcc=13)

    if CACHE_FILE.exists() and not args.no_cache:
        data = np.load(CACHE_FILE, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        subject_ids = list(data["subject_ids"])
        feature_names = list(data["feature_names"])
        print(f"     Loaded from cache: X={X.shape}")
    else:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        X_list, y_list, subj_list = [], [], []
        t0 = time.time()
        errors = 0
        feature_names: list[str] = []

        for i, sample in enumerate(samples):
            try:
                bundle = extractor.from_file(sample.path)
                X_list.append(bundle.vector)
                y_list.append(sample.label)
                subj_list.append(sample.subject_id)
                if not feature_names:
                    feature_names = bundle.feature_names
            except Exception as exc:
                logger.warning("Skipping %s: %s", sample.path, exc)
                errors += 1
                continue

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                remaining = (len(samples) - i - 1) / rate
                print(f"     {i+1}/{len(samples)}  {elapsed:.0f}s elapsed  "
                      f"~{remaining/60:.0f}min remaining  errors={errors}")

        X = np.vstack(X_list)
        y = np.array(y_list, dtype=int)
        subject_ids = subj_list
        np.savez(CACHE_FILE, X=X, y=y, subject_ids=subject_ids, feature_names=feature_names)
        print(f"     Saved cache → {CACHE_FILE}  ({errors} errors)")
        print(f"     X={X.shape}  {time.time()-t0:.0f}s total")

    # ── Train on full dataset ─────────────────────────────────────────────────
    print("\n[3/4] Training SVM + Random Forest …")
    t0 = time.time()
    svm = build_svm()
    rf  = build_rf()
    svm.fit(X, y)
    rf.fit(X, y)
    print(f"     Done in {time.time()-t0:.1f}s")

    # ── LOOCV evaluation ──────────────────────────────────────────────────────
    print("\n[4/4] Leave-One-Subject-Out evaluation …")
    svm_m = loocv_evaluate(X, y, subject_ids, build_svm())
    rf_m  = loocv_evaluate(X, y, subject_ids, build_rf())

    counts = Counter(y)
    print(f"\n  LOOCV results ({len(set(subject_ids))}-subject leave-one-out):")
    print(f"  SVM  AUROC={svm_m.auroc:.3f}  Sens={svm_m.sensitivity:.3f}  "
          f"Spec={svm_m.specificity:.3f}  F1={svm_m.f1:.3f}  "
          f"(n+={counts[1]}, n-={counts[0]})")
    print(f"  RF   AUROC={rf_m.auroc:.3f}  Sens={rf_m.sensitivity:.3f}  "
          f"Spec={rf_m.specificity:.3f}  F1={rf_m.f1:.3f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_DIR / "rf.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(OUT_DIR / "svm.pkl", "wb") as f:
        pickle.dump(svm, f)
    with open(OUT_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    clf = rf.named_steps["clf"]
    importances = {n: round(float(v), 6)
                   for n, v in zip(feature_names, clf.feature_importances_)}
    with open(OUT_DIR / "feature_importances.json", "w") as f:
        json.dump(importances, f, indent=2)

    metrics_payload = {
        "loocv_svm": svm_m.to_dict(),
        "loocv_rf":  rf_m.to_dict(),
        "n_samples":  int(len(y)),
        "n_features": int(X.shape[1]),
        "n_subjects": len(set(subject_ids)),
        "n_positive": int(counts[1]),
        "n_negative": int(counts[0]),
        "tasks": args.tasks,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\n  Models saved → {OUT_DIR}/")
    top5 = sorted(importances.items(), key=lambda x: -x[1])[:5]
    print("  Top 5 features:")
    for feat, imp in top5:
        print(f"    {feat:<35s}  {imp:.4f}")

    print("\n" + "=" * 60)
    print("  Setup complete. Restart the server to load the new model:")
    print("  python app/server.py")
    print("=" * 60)
    print("\nDISCLAIMER: Research / educational tool only. NOT for clinical diagnosis.\n")


if __name__ == "__main__":
    main()
