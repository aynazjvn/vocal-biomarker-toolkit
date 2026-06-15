"""FastAPI backend for the Vocal Biomarker Screening Toolkit.

Serves the HTML frontend and handles audio upload + prediction.

Run with:
    python app/server.py

Then open http://localhost:8000
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from src.inference.pipeline import VocalBiomarkerPipeline

# ---------------------------------------------------------------------------

PK_MODEL_DIR   = Path("results/parkinson/classical")
RESP_MODEL_DIR = Path("results/respiratory/classical")
DEPR_MODEL_DIR = Path("results/depression/classical")
APP_DIR        = Path(__file__).parent

app = FastAPI(title="Vocal Biomarker Screening Toolkit", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

pipeline: VocalBiomarkerPipeline | None = None
model_meta: dict = {}

# ---------------------------------------------------------------------------

def _load_model(model_dir: Path) -> tuple:
    """Load rf.pkl, feature_names.json, metrics.json from a model directory.

    Returns (model, feature_names, metrics) — all None/empty on failure.
    """
    rf_path   = model_dir / "rf.pkl"
    feat_path = model_dir / "feature_names.json"
    meta_path = model_dir / "metrics.json"

    if not rf_path.exists():
        return None, [], {}

    with open(rf_path, "rb") as f:
        rf = pickle.load(f)

    feature_names: list[str] = []
    if feat_path.exists():
        with open(feat_path) as f:
            feature_names = json.load(f)

    metrics: dict = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metrics = json.load(f)

    return rf, feature_names, metrics


@app.on_event("startup")
def load_models() -> None:
    global pipeline, model_meta

    pk_rf, pk_features, pk_meta = _load_model(PK_MODEL_DIR)
    if pk_rf is None:
        print(f"[WARN] No Parkinson model at {PK_MODEL_DIR}. Run: python scripts/setup_and_train.py")
    else:
        loocv = pk_meta.get("loocv_rf", {})
        print(f"[INFO] Parkinson model loaded — LOOCV AUROC={loocv.get('auroc','?'):.3f}")

    resp_rf, resp_features, resp_meta = _load_model(RESP_MODEL_DIR)
    if resp_rf is None:
        print(f"[INFO] No respiratory model at {RESP_MODEL_DIR} (run scripts/train_respiratory.py)")
    else:
        loocv = resp_meta.get("loocv_rf", {})
        print(f"[INFO] Respiratory model loaded — LOOCV AUROC={loocv.get('auroc','?'):.3f}")

    depr_rf, depr_features, depr_meta = _load_model(DEPR_MODEL_DIR)
    if depr_rf is None:
        print(f"[INFO] No depression model at {DEPR_MODEL_DIR} (run scripts/train_depression.py)")
    else:
        loocv = depr_meta.get("loocv_rf", {})
        print(f"[INFO] Depression model loaded — LOOCV AUROC={loocv.get('auroc','?'):.3f}")

    model_meta = {"parkinson": pk_meta, "respiratory": resp_meta, "depression": depr_meta}
    pipeline = VocalBiomarkerPipeline(
        parkinson_model=pk_rf,
        respiratory_model=resp_rf,
        depression_model=depr_rf,
        feature_names=pk_features,
        respiratory_feature_names=resp_features,
        depression_feature_names=depr_features,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(APP_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    pk_model   = pipeline.models.get("parkinson")   if pipeline else None
    resp_model = pipeline.models.get("respiratory") if pipeline else None
    depr_model = pipeline.models.get("depression")  if pipeline else None
    pk_meta    = model_meta.get("parkinson", {})
    resp_meta  = model_meta.get("respiratory", {})
    depr_meta  = model_meta.get("depression", {})
    return {
        "status": "ok",
        "models": {
            "parkinson":   {"loaded": pk_model is not None,
                            "metrics": pk_meta.get("loocv_rf", {}),
                            "n_samples": pk_meta.get("n_samples"),
                            "n_features": pk_meta.get("n_features")},
            "respiratory": {"loaded": resp_model is not None,
                            "metrics": resp_meta.get("loocv_rf", {}),
                            "n_samples": resp_meta.get("n_samples"),
                            "n_features": resp_meta.get("n_features")},
            "depression":  {"loaded": depr_model is not None,
                            "metrics": depr_meta.get("loocv_rf", {}),
                            "n_samples": depr_meta.get("n_samples"),
                            "n_features": depr_meta.get("n_features")},
        },
        # legacy fields — kept so the old frontend checkHealth still works
        "model_loaded": pk_model is not None,
        "metrics": pk_meta.get("loocv_rf", {}),
        "n_samples": pk_meta.get("n_samples"),
        "n_features": pk_meta.get("n_features"),
    }


@app.post("/api/predict")
async def predict(audio: UploadFile = File(...)) -> JSONResponse:
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialised")

    # Accept webm/ogg (from browser MediaRecorder) and wav/mp3
    suffix = Path(audio.filename).suffix if audio.filename else ".webm"
    if not suffix:
        suffix = ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        report = pipeline.predict(tmp_path)
        return JSONResponse(content=report)
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}") from exc
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    print("\n  Vocal Biomarker Screening Toolkit")
    print(f"  http://{args.host}:{args.port}\n")
    uvicorn.run("app.server:app", host=args.host, port=args.port, reload=args.reload)
