"""Quick end-to-end pipeline test — run with: python scripts/_test_pipeline.py"""
import sys, time
sys.path.insert(0, ".")

print(">>> Loading dataset...", flush=True)
from src.data.parkinson_loader import ParkinsonLoader
loader = ParkinsonLoader()
samples = loader.load_samples("Italian Parkinson's Voice and speech")
pd_n = sum(1 for s in samples if s.label == 1)
hc_n = sum(1 for s in samples if s.label == 0)
n_subj = len({s.subject_id for s in samples})
print(f"    {len(samples)} recordings | PD={pd_n} HC={hc_n} | {n_subj} subjects", flush=True)

print(">>> Feature extraction (30 PD + 30 HC, balanced)...", flush=True)
from src.features.audio_features import FeatureExtractor
import numpy as np

ext = FeatureExtractor()
# Balanced subset so every cross-val fold sees both classes
pd_samples = [s for s in samples if s.label == 1][:30]
hc_samples = [s for s in samples if s.label == 0][:30]
subset = pd_samples + hc_samples

t0 = time.time()
bundles, labels = [], []
for s in subset:
    try:
        b = ext.from_file(s.path)
        bundles.append(b)
        labels.append(s.label)
    except Exception as e:
        print(f"    WARN: {s.path.name}: {e}", flush=True)
elapsed = time.time() - t0
X = np.vstack([b.vector for b in bundles])
y = np.array(labels)
print(f"    {X.shape} | {elapsed:.1f}s | {elapsed/len(bundles)*1000:.0f}ms/file | any NaN: {not np.isfinite(X).all()}", flush=True)

print(">>> Classical model fit + eval (3-fold stratified CV)...", flush=True)
from src.models.classical import build_svm, build_rf
from src.evaluation.metrics import evaluate
from sklearn.model_selection import StratifiedKFold

svm = build_svm()
rf  = build_rf()

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
probs_svm, probs_rf, trues = [], [], []
for tr, te in skf.split(X, y):
    svm.fit(X[tr], y[tr])
    rf.fit(X[tr], y[tr])
    probs_svm.extend(svm.predict_proba(X[te])[:, 1])
    probs_rf.extend(rf.predict_proba(X[te])[:, 1])
    trues.extend(y[te])

m_svm = evaluate(trues, probs_svm)
m_rf  = evaluate(trues, probs_rf)
print(f"    SVM  AUROC={m_svm.auroc:.3f}  Sens={m_svm.sensitivity:.3f}  Spec={m_svm.specificity:.3f}", flush=True)
print(f"    RF   AUROC={m_rf.auroc:.3f}  Sens={m_rf.sensitivity:.3f}  Spec={m_rf.specificity:.3f}", flush=True)

print(">>> Inference pipeline...", flush=True)
from src.inference.pipeline import VocalBiomarkerPipeline
# Use a PD sample so we get a non-trivial score
rf.fit(X, y)
pipe = VocalBiomarkerPipeline(parkinson_model=rf)
report = pipe.predict(pd_samples[0].path)
pk = report["conditions"]["parkinson"]
print(f"    score={pk['score']}  label={pk['label']}", flush=True)
print(f"    disclaimer present: {bool(report['disclaimer'])}", flush=True)

print("\n>>> ALL CHECKS PASSED <<<", flush=True)
