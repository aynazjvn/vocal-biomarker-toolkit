#!/usr/bin/env python
"""One-shot setup: extract features (cached), train SVM + RF, save models.

Run once before launching the demo:
    python scripts/setup_and_train.py

After this, the demo works instantly:
    python app/demo.py

Features are cached to results/parkinson/features.npz so subsequent
runs skip the 25-minute extraction step.
"""

from __future__ import annotations
import sys, time, json, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

CACHE = Path("results/parkinson/features.npz")
MODEL_DIR = Path("results/parkinson/classical")
CORPUS = Path("Italian Parkinson's Voice and speech")

# ── Step 1: load dataset ────────────────────────────────────────────────────
print("\n[1/4] Loading dataset …")
from src.data.parkinson_loader import ParkinsonLoader, subject_split
loader = ParkinsonLoader()
samples = loader.load_samples(CORPUS)
pd_n = sum(1 for s in samples if s.label == 1)
hc_n = sum(1 for s in samples if s.label == 0)
print(f"     {len(samples)} recordings | PD={pd_n}  HC={hc_n} | "
      f"{len({s.subject_id for s in samples})} subjects")

# ── Step 2: feature extraction (with disk cache) ────────────────────────────
if CACHE.exists():
    print(f"\n[2/4] Loading cached features from {CACHE} …")
    data = np.load(CACHE, allow_pickle=True)
    X            = data["X"]
    y            = data["y"]
    subj_ids     = data["subj_ids"].tolist()
    feature_names = data["feature_names"].tolist()
    print(f"     X={X.shape}  cached ✓")
else:
    print(f"\n[2/4] Extracting features for {len(samples)} files …")
    print("     (this takes ~25 min the first time, then cached to disk)\n")
    from src.features.audio_features import FeatureExtractor
    ext = FeatureExtractor()
    t0 = time.time()
    X_list, y_list, subj_list = [], [], []
    errors = 0
    for i, s in enumerate(samples):
        try:
            b = ext.from_file(s.path)
            X_list.append(b.vector)
            y_list.append(s.label)
            subj_list.append(s.subject_id)
        except Exception as e:
            errors += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            remaining = rate * (len(samples) - i - 1)
            print(f"     {i+1}/{len(samples)}  {elapsed:.0f}s elapsed  "
                  f"~{remaining/60:.0f}min remaining  errors={errors}")
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    subj_ids = subj_list
    # Get feature names from the last successful bundle
    feature_names = b.feature_names
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE,
             X=X, y=y,
             subj_ids=np.array(subj_ids),
             feature_names=np.array(feature_names))
    print(f"\n     Saved cache → {CACHE}  ({errors} errors)")
    print(f"     X={X.shape}  {time.time()-t0:.0f}s total")

# ── Step 3: train models ─────────────────────────────────────────────────────
print("\n[3/4] Training SVM + Random Forest …")
from src.models.classical import build_svm, build_rf

# Train on the full dataset (LOOCV evaluation done separately)
svm = build_svm()
rf  = build_rf(n_estimators=300)
t0 = time.time()
svm.fit(X, y)
rf.fit(X, y)
print(f"     Done in {time.time()-t0:.1f}s")

# ── Step 4: LOOCV evaluation ─────────────────────────────────────────────────
print("\n[4/4] Leave-One-Subject-Out evaluation …")
from src.evaluation.metrics import loocv_evaluate, evaluate

svm_loocv = build_svm()
rf_loocv  = build_rf(n_estimators=200)
m_svm = loocv_evaluate(X, y, subj_ids, svm_loocv)
m_rf  = loocv_evaluate(X, y, subj_ids, rf_loocv)

print(f"\n  LOOCV results (61-subject leave-one-out):")
print(f"  SVM  {m_svm}")
print(f"  RF   {m_rf}")

# ── Save everything ───────────────────────────────────────────────────────────
MODEL_DIR.mkdir(parents=True, exist_ok=True)
with open(MODEL_DIR / "svm.pkl", "wb") as f:
    pickle.dump(svm, f)
with open(MODEL_DIR / "rf.pkl", "wb") as f:
    pickle.dump(rf, f)
with open(MODEL_DIR / "feature_names.json", "w") as f:
    json.dump(feature_names, f)
with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump({
        "loocv_svm": m_svm.to_dict(),
        "loocv_rf":  m_rf.to_dict(),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }, f, indent=2)

# Feature importances
fi = dict(sorted(
    zip(feature_names, rf.named_steps["clf"].feature_importances_.tolist()),
    key=lambda x: -x[1]
))
with open(MODEL_DIR / "feature_importances.json", "w") as f:
    json.dump(fi, f, indent=2)

print(f"\n  Models saved → {MODEL_DIR}/")
print(f"  Top 5 features:")
for name, imp in list(fi.items())[:5]:
    print(f"    {name:<40s}  {imp:.4f}")

print("\n" + "="*60)
print("  Setup complete. Launch the demo with:")
print("  python app/demo.py")
print("="*60 + "\n")
print("DISCLAIMER: Research / educational tool only. NOT for clinical diagnosis.\n")
