#!/usr/bin/env python3
"""Train depression affect screening on RAVDESS.

Proxy labelling: sad + fearful → depressed affect (1)
                 neutral + calm + happy → neutral affect (0)

This is a research approximation. Clinical depression detection
requires the DAIC-WOZ dataset (IRB access from USC ICT).

Usage
-----
    python scripts/train_depression.py

    # Custom data path:
    python scripts/train_depression.py --data data/ravdess

Output
------
    results/depression/classical/
        rf.pkl  svm.pkl  metrics.json
        feature_names.json  feature_importances.json
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

from src.data.depression_loader import RAVDESSLoader
from src.evaluation.metrics import loocv_evaluate
from src.features.audio_features import FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CACHE_FILE = Path("results/depression/features.npz")
OUT_DIR    = Path("results/depression/classical")


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
    parser.add_argument("--data", default="data/ravdess",
                        help="Path to unzipped RAVDESS directory (contains Actor_01 … Actor_24)")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    root = Path(args.data)
    if not root.exists():
        logger.error("Data directory not found: %s", root)
        sys.exit(1)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading RAVDESS dataset …")
    loader = RAVDESSLoader()
    samples = loader.load_samples(root)

    if not samples:
        logger.error("No samples found in %s", root)
        sys.exit(1)

    counts = Counter(s.label for s in samples)
    subjects = list({s.subject_id for s in samples})
    print(f"     {len(samples)} clips | depressed-affect={counts[1]}  neutral={counts[0]} | {len(subjects)} actors")

    # ── Feature extraction ────────────────────────────────────────────────────
    print("\n[2/4] Extracting features …")
    extractor = FeatureExtractor(sr=16_000, n_mfcc=13)

    if CACHE_FILE.exists() and not args.no_cache:
        data = np.load(CACHE_FILE, allow_pickle=True)
        X, y = data["X"], data["y"]
        subject_ids  = list(data["subject_ids"])
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

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                remaining = (len(samples) - i - 1) / rate
                print(f"     {i+1}/{len(samples)}  {elapsed:.0f}s elapsed  "
                      f"~{remaining:.0f}s remaining  errors={errors}")

        X = np.vstack(X_list)
        y = np.array(y_list, dtype=int)
        subject_ids = subj_list
        np.savez(CACHE_FILE, X=X, y=y, subject_ids=subject_ids, feature_names=feature_names)
        print(f"     Saved cache → {CACHE_FILE}  ({errors} errors)  X={X.shape}  {time.time()-t0:.0f}s total")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[3/4] Training SVM + Random Forest …")
    t0 = time.time()
    svm = build_svm()
    rf  = build_rf()
    svm.fit(X, y)
    rf.fit(X, y)
    print(f"     Done in {time.time()-t0:.1f}s")

    # ── LOOCV (leave-one-actor-out) ───────────────────────────────────────────
    print("\n[4/4] Leave-One-Actor-Out evaluation …")
    svm_m = loocv_evaluate(X, y, subject_ids, build_svm())
    rf_m  = loocv_evaluate(X, y, subject_ids, build_rf())

    counts = Counter(y)
    print(f"\n  LOOCV results ({len(set(subject_ids))}-actor leave-one-out):")
    print(f"  SVM  AUROC={svm_m.auroc:.3f}  Sens={svm_m.sensitivity:.3f}  "
          f"Spec={svm_m.specificity:.3f}  F1={svm_m.f1:.3f}  "
          f"(n+={counts[1]}, n-={counts[0]})")
    print(f"  RF   AUROC={rf_m.auroc:.3f}  Sens={rf_m.sensitivity:.3f}  "
          f"Spec={rf_m.specificity:.3f}  F1={rf_m.f1:.3f}")

    # ── Save ─────────────────────────────────────────────────────────────────
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
        "dataset": "RAVDESS",
        "note": "Research proxy: sad+fearful=1, neutral+calm+happy=0. Not clinical depression.",
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\n  Models saved → {OUT_DIR}/")
    top5 = sorted(importances.items(), key=lambda x: -x[1])[:5]
    print(f"  Top 5 features:")
    for feat, imp in top5:
        print(f"    {feat:<35s}  {imp:.4f}")

    print("\n" + "=" * 60)
    print("  Restart server to load: python app/server.py")
    print("=" * 60)
    print("\nDISCLAIMER: Research proxy model — NOT clinical depression detection.\n")


if __name__ == "__main__":
    main()
