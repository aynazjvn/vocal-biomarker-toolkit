#!/usr/bin/env python3
"""Fine-tune Wav2Vec2 on RAVDESS for depression-affect proxy screening.

Training protocol
-----------------
* Architecture: Wav2Vec2-base → mean pool → Linear(768, 1)
* Freeze:  CNN feature extractor + bottom 8 of 12 transformer layers
* Fine-tune: top 4 encoder layers + classification head
* Evaluation: Leave-Actor-Out CV (24 actors → 24 folds)
* Final model: trained on all data, saved to results/depression/transformer/

Dataset
-------
    RAVDESS (Ryerson Audio-Visual Database of Emotional Speech)
    License: CC BY-NC-SA 4.0
    Download: https://zenodo.org/record/1188976
              → Audio_Speech_Actors_01-24.zip (198 MB)

    Unzip into: data/ravdess/
        Actor_01/, Actor_02/, …, Actor_24/

    Labelling:
        Depressed affect (label=1): sad (04), fearful (06)
        Neutral affect  (label=0): neutral (01), calm (02), happy (03)
        Skipped:                   angry, disgust, surprised

Outputs
-------
    results/depression/transformer/
        model.pt
        config.json
        metrics.json
        feature_names.json

Usage
-----
    python scripts/train_depression_transformer.py
    python scripts/train_depression_transformer.py --data data/ravdess
    python scripts/train_depression_transformer.py --epochs 15 --no-loocv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.depression_loader import RAVDESSLoader
from src.training.transformer_trainer import TransformerTrainer, TransformerTrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_DATA = REPO_ROOT / "data" / "ravdess"
OUT_DIR = REPO_ROOT / "results" / "depression" / "transformer"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 for depression affect screening")
    p.add_argument("--data", default=str(DEFAULT_DATA),
                   help="RAVDESS root directory (containing Actor_01/ … Actor_24/)")
    p.add_argument("--model", default="superb/wav2vec2-base-superb-er")
    p.add_argument("--freeze-encoder-layers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--loocv-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-duration", type=float, default=5.0,
                   help="Max clip length in seconds (RAVDESS clips are ~3-4s, default 5)")
    p.add_argument("--no-loocv", action="store_true",
                   help="Skip Leave-Actor-Out CV; only train the final model")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Check data ─────────────────────────────────────────────────────
    data_root = Path(args.data)
    if not data_root.exists():
        logger.error(
            "RAVDESS data not found: %s\n\n"
            "  Steps to obtain the data:\n"
            "    1. Download: https://zenodo.org/record/1188976\n"
            "       (Audio_Speech_Actors_01-24.zip, 198 MB)\n"
            "    2. Unzip into: %s/\n"
            "       Expected layout: Actor_01/ ... Actor_24/\n"
            "  License: CC BY-NC-SA 4.0",
            data_root, data_root,
        )
        sys.exit(1)

    # ── Load dataset ───────────────────────────────────────────────────
    loader = RAVDESSLoader()
    logger.info("Loading RAVDESS from: %s", data_root)
    samples = loader.load_samples(data_root)

    if not samples:
        logger.error("No samples found — check --data path")
        sys.exit(1)

    n_pos = sum(s.label == 1 for s in samples)
    n_neg = sum(s.label == 0 for s in samples)
    actors = {s.subject_id for s in samples}
    logger.info(
        "Loaded %d recordings  depressed=%d  neutral=%d  actors=%d",
        len(samples), n_pos, n_neg, len(actors),
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

    print("\n" + "=" * 60)
    print("DEPRESSION AFFECT SCREENING — Wav2Vec2 Fine-tuning")
    print("=" * 60)
    print(f"  Dataset:        RAVDESS  ({len(samples)} clips, {len(actors)} actors)")
    print(f"  Labels:         depressed (sad+fearful)={n_pos}  neutral={n_neg}")
    print(f"  Model:          {args.model}")
    print(f"  Frozen layers:  0..{args.freeze_encoder_layers - 1}  "
          f"(fine-tune {12 - args.freeze_encoder_layers})")
    cv_desc = "Leave-Actor-Out LOOCV (24 folds)" if not args.no_loocv else "None"
    print(f"  CV:             {cv_desc}")
    print(f"  Output:         {OUT_DIR}")
    print()

    # ── Run ────────────────────────────────────────────────────────────
    result = trainer.run(
        samples,
        run_loocv=not args.no_loocv,
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
        print(f"  {cv_type.upper()} AUROC = {auroc:.4f}  (target: >0.78)")
        if auroc < 0.78:
            print("  ↳  Below target — try more epochs, lower LR, or unfreeze more layers")
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
