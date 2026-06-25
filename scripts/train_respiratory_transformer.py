#!/usr/bin/env python3
"""Fine-tune Wav2Vec2 on the Coswara dataset for respiratory/COVID-19 screening.

Training protocol
-----------------
* Architecture: Wav2Vec2-base → mean pool → Linear(768, 1)
* Freeze:  CNN feature extractor + bottom 8 of 12 transformer layers
* Fine-tune: top 4 encoder layers + classification head
* Evaluation: 10-fold subject-grouped CV (LOOCV is too slow for ~500 subjects)
* Final model: trained on all data, saved to results/respiratory/transformer/

Dataset
-------
    Coswara-Data (cough-shallow recordings)
    GitHub: https://github.com/iiscleap/Coswara-Data
    License: CC BY 4.0
    Download: git clone https://github.com/iiscleap/Coswara-Data.git data/coswara

    Then extract the compressed files:
        cd data/coswara && python extract_data.py

    The loader expects the structure:
        data/coswara/Extracted_data/
            <date>/
                <participant_id>/
                    metadata.json
                    cough-shallow.wav
                    ...

Outputs
-------
    results/respiratory/transformer/
        model.pt
        config.json
        metrics.json
        feature_names.json

Usage
-----
    python scripts/train_respiratory_transformer.py
    python scripts/train_respiratory_transformer.py --data data/coswara/Extracted_data
    python scripts/train_respiratory_transformer.py --tasks cough-shallow cough-heavy
    python scripts/train_respiratory_transformer.py --folds 5 --no-loocv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.respiratory_loader import CoswaraLoader
from src.training.transformer_trainer import TransformerTrainer, TransformerTrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_DATA = REPO_ROOT / "data" / "coswara" / "Extracted_data"
OUT_DIR = REPO_ROOT / "results" / "respiratory" / "transformer"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 for respiratory screening")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="Coswara Extracted_data directory")
    p.add_argument("--tasks", nargs="+", default=["cough-shallow"],
                   help="Coswara audio tasks (default: cough-shallow)")
    p.add_argument("--model", default="microsoft/wavlm-base")
    p.add_argument("--freeze-encoder-layers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--loocv-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-duration", type=float, default=10.0)
    p.add_argument("--folds", type=int, default=10,
                   help="Number of CV folds (default 10; use --folds 0 for LOOCV)")
    p.add_argument("--no-cv", action="store_true",
                   help="Skip cross-validation; only train the final model")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Check data ─────────────────────────────────────────────────────
    data_root = Path(args.data)
    if not data_root.exists():
        logger.error(
            "Coswara data not found: %s\n\n"
            "  Steps to obtain the data:\n"
            "    1. git clone https://github.com/iiscleap/Coswara-Data.git data/coswara\n"
            "    2. cd data/coswara && python extract_data.py\n"
            "  Then re-run this script.\n"
            "  License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
            data_root,
        )
        sys.exit(1)

    # ── Load dataset ───────────────────────────────────────────────────
    loader = CoswaraLoader(tasks=args.tasks)
    logger.info("Loading Coswara dataset from: %s  tasks=%s", data_root, args.tasks)
    samples = loader.load_samples(data_root)

    if not samples:
        logger.error("No samples found — check --data path and --tasks filter")
        sys.exit(1)

    n_pos = sum(s.label == 1 for s in samples)
    n_neg = sum(s.label == 0 for s in samples)
    n_sub = len({s.subject_id for s in samples})
    logger.info(
        "Loaded %d recordings  positive=%d  healthy=%d  subjects=%d",
        len(samples), n_pos, n_neg, n_sub,
    )

    # ── Configure trainer ──────────────────────────────────────────────
    cfg = TransformerTrainerConfig(
        model_name=args.model,
        freeze_feature_extractor=True,
        freeze_encoder_layers=args.freeze_encoder_layers,
        max_duration_s=args.max_duration,
        sample_rate=16_000,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        loocv_epochs=args.loocv_epochs,
        output_dir=str(OUT_DIR),
        seed=args.seed,
    )
    trainer = TransformerTrainer(cfg)

    # ── Decide CV strategy ─────────────────────────────────────────────
    # With 400-500 subjects LOOCV would mean hundreds of fold trainings.
    # Default to 10-fold subject-grouped CV.
    use_loocv = args.folds == 0
    k = args.folds if not use_loocv else None

    print("\n" + "=" * 60)
    print("RESPIRATORY SCREENING — Wav2Vec2 Fine-tuning")
    print("=" * 60)
    print(f"  Dataset:        Coswara  ({len(samples)} recordings, {n_sub} subjects)")
    print(f"  Tasks:          {args.tasks}")
    print(f"  Model:          {args.model}")
    print(f"  Frozen layers:  0..{args.freeze_encoder_layers - 1}")
    cv_desc = "LOOCV" if use_loocv else f"{k}-fold subject-grouped CV"
    print(f"  CV:             {cv_desc}")
    print(f"  Output:         {OUT_DIR}")
    print()

    result = trainer.run(
        samples,
        run_loocv=use_loocv and not args.no_cv,
        run_kfold=k if not args.no_cv else None,
    )

    # ── Save metrics ───────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "loocv_transformer": {
            "auroc": result.get("auroc"),
            "cv_type": result.get("cv_type"),
        },
        "n_samples": result["n_samples"],
        "n_positive": result["n_positive"],
        "n_negative": result["n_negative"],
        "n_subjects": result["n_subjects"],
        "tasks": args.tasks,
        "model_name": result["model_name"],
        "freeze_encoder_layers": result["freeze_encoder_layers"],
        "elapsed_s": result["elapsed_s"],
    }
    if "fold_aurocs" in result:
        metrics["loocv_transformer"]["fold_aurocs"] = result["fold_aurocs"]

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(OUT_DIR / "feature_names.json", "w") as f:
        json.dump(["wav2vec2"], f)

    # ── Summary ────────────────────────────────────────────────────────
    auroc = result.get("auroc")
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    if auroc is not None:
        cv_type = result.get("cv_type", "?")
        print(f"  {cv_type.upper()} AUROC = {auroc:.4f}  (target: >0.85)")
        if auroc < 0.85:
            print("  ↳  Below target — try more epochs or --tasks cough-shallow cough-heavy")
    print(f"  Elapsed:       {result['elapsed_s']:.0f}s")
    print(f"  Checkpoint:    {OUT_DIR}/model.pt")
    model_path = OUT_DIR / "model.pt"
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1e6
        print(f"  Model size:    {size_mb:.0f} MB")
        if size_mb > 100:
            print("\n  NOTE: model.pt is large — consider git-lfs for sharing.")
    print()
    print("  To use this model, restart the server:")
    print("    python app/server.py")
    print("=" * 60)
    print("\nDISCLAIMER: Research / educational tool only. NOT for clinical diagnosis.\n")


if __name__ == "__main__":
    main()
