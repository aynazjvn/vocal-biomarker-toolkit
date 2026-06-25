"""FastAPI backend for the Vocal Biomarker Screening Toolkit.

Serves the HTML frontend and handles audio upload + prediction.

Run with:
    python app/server.py

Then open http://localhost:8000
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from src.inference.pipeline import VocalBiomarkerPipeline

# ---------------------------------------------------------------------------

_ROOT    = Path(__file__).parent.parent
PK_DIR   = _ROOT / "results" / "parkinson"
RESP_DIR = _ROOT / "results" / "respiratory"
DEPR_DIR = _ROOT / "results" / "depression"
APP_DIR  = Path(__file__).parent

app = FastAPI(title="Vocal Biomarker Screening Toolkit", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

pipeline: VocalBiomarkerPipeline | None = None
model_meta: dict = {}

# ---------------------------------------------------------------------------


@app.on_event("startup")
def load_models() -> None:
    global pipeline, model_meta

    pipeline = VocalBiomarkerPipeline.from_auto(
        parkinson_dir=PK_DIR,
        respiratory_dir=RESP_DIR,
        depression_dir=DEPR_DIR,
    )

    # Collect metadata for the /api/health endpoint
    model_meta = {}
    for condition, cond_dir in [
        ("parkinson",   PK_DIR),
        ("respiratory", RESP_DIR),
        ("depression",  DEPR_DIR),
    ]:
        mtype = pipeline.model_types.get(condition, "none")
        subdir = "transformer" if mtype == "transformer" else "classical"
        meta_path = cond_dir / subdir / "metrics.json"
        meta: dict = {}
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                pass
        model_meta[condition] = {**meta, "model_type": mtype}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(APP_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    def _condition_info(condition: str) -> dict:
        model = pipeline.models.get(condition) if pipeline else None
        meta  = model_meta.get(condition, {})
        mtype = meta.get("model_type", "none")

        # AUROC lives under "loocv_transformer" for transformer models,
        # under "loocv_rf" for classical — support both shapes.
        cv_block = meta.get("loocv_transformer") or meta.get("loocv_rf") or {}
        return {
            "loaded":      model is not None,
            "model_type":  mtype,
            "metrics":     cv_block,
            "n_samples":   meta.get("n_samples"),
            "n_features":  meta.get("n_features"),
        }

    pk_info = _condition_info("parkinson")
    return {
        "status": "ok",
        "models": {
            "parkinson":   pk_info,
            "respiratory": _condition_info("respiratory"),
            "depression":  _condition_info("depression"),
        },
        # legacy fields — kept so the old frontend checkHealth still works
        "model_loaded": pk_info["loaded"],
        "metrics":      pk_info["metrics"],
        "n_samples":    pk_info["n_samples"],
        "n_features":   pk_info["n_features"],
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
