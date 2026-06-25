#!/usr/bin/env python3
"""Fine-tune Wav2Vec2 on the Italian Parkinson's Voice and Speech corpus.

Training protocol
-----------------
* Architecture: Wav2Vec2-base → mean pool → Linear(768, 1)
* Freeze:  CNN feature extractor + bottom 8 of 12 transformer layers
* Fine-tune: top 4 encoder layers + classification head
* Evaluation: Leave-One-Subject-Out CV (61 subjects → 61 folds)
* Final model: trained on all data, saved to results/parkinson/transformer/

Outputs
-------
    results/parkinson/transformer/
        model.pt          — fine-tuned Wav2Vec2Classifier weights (~375 MB)
        config.json       — architecture config for loading
        metrics.json      — LOOCV AUROC + dataset statistics
        feature_names.json — ["wav2vec2"]  (for server compatibility)

Usage
-----
    python scripts/train_parkinson_transformer.py
    python scripts/train_parkinson_transformer.py --data "Italian Parkinson's Voice and speech"
    python scripts/train_parkinson_transformer.py --epochs 15 --loocv-epochs 7 --batch-size 16
    python scripts/train_parkinson_transformer.py --no-loocv   # skip CV, just train final
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.parkinson_loader import ParkinsonLoader
from src.training.transformer_trainer import TransformerTrainer, TransformerTrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_DATA = REPO_ROOT / "Italian Parkinson's Voice and speech"
OUT_DIR = REPO_ROOT / "results" / "parkinson" / "transformer"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 for Parkinson's detection")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset root directory")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Task filter, e.g. --tasks B1 B2 VA1 VE1. Default: all tasks.")
    p.add_argument("--no-young-controls", action="store_true",
                   help="Exclude the 15 young healthy controls")
    p.add_argument("--model", default="facebook/wav2vec2-base",
                   help="HuggingFace model ID")
    p.add_argument("--freeze-encoder-layers", type=int, default=8,
                   help="Number of Wav2Vec2 encoder layers to freeze (default 8/12)")
    p.add_argument("--epochs", type=int, default=10, help="Final model training epochs")
    p.add_argument("--loocv-epochs", type=int, default=5, help="Epochs per LOOCV fold")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-duration", type=float, default=10.0,
                   help="Max audio clip length in seconds (pad/truncate)")
    p.add_argument("--no-loocv", action="store_true",
                   help="Skip LOOCV; only train the final model")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load dataset ───────────────────────────────────────────────────
    data_root = Path(args.data)
    if not data_root.exists():
        logger.error(
            "Dataset not found: %s\n"
            "  Expected the Italian Parkinson's Voice and Speech corpus.\n"
            "  Download from: https://ieee-dataport.org/open-access/"
            "italian-parkinsons-voice-and-speech",
            data_root,
        )
        sys.exit(1)

    tasks = set(args.tasks) if args.tasks else None
    loader = ParkinsonLoader(
        tasks=tasks,
        include_young_controls=not args.no_young_controls,
    )

    logger.info("Loading dataset from: %s", data_root)
    samples = loader.load_samples(data_root)

    if not samples:
        logger.error("No audio samples found — check --data path and --tasks filter")
        sys.exit(1)

    n_pd  = sum(s.label == 1 for s in samples)
    n_hc  = sum(s.label == 0 for s in samples)
    n_sub = len({s.subject_id for s in samples})
    logger.info(
        "Loaded %d recordings  PD=%d  HC=%d  subjects=%d",
        len(samples), n_pd, n_hc, n_sub,
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

    # ── Run ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PARKINSON'S DETECTION — Wav2Vec2 Fine-tuning")
    print("=" * 60)
    print(f"  Dataset:        {data_root.name}  ({len(samples)} recordings, {n_sub} subjects)")
    print(f"  Model:          {args.model}")
    print(f"  Frozen layers:  0..{args.freeze_encoder_layers - 1}  (fine-tune {12 - args.freeze_encoder_layers})")
    print(f"  Epochs:         {args.epochs}  (LOOCV folds: {args.loocv_epochs})")
    print(f"  Output:         {OUT_DIR}")
    print()

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
        print(f"  {cv_type.upper()} AUROC = {auroc:.4f}  (target: >0.99)")
        if auroc < 0.99:
            print("  ↳  Below target — try more epochs or fewer frozen layers")
    print(f"  Elapsed:       {result['elapsed_s']:.0f}s")
    print(f"  Checkpoint:    {OUT_DIR}/model.pt")
    model_path = OUT_DIR / "model.pt"
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1e6
        print(f"  Model size:    {size_mb:.0f} MB")
        if size_mb > 100:
            print()
            print("  NOTE: model.pt is large. To share it:")
            print("    git lfs track 'results/parkinson/transformer/*.pt'")
            print("    git add .gitattributes results/parkinson/transformer/")
    print()
    print("  To use this model, restart the server:")
    print("    python app/server.py")
    print("=" * 60)
    print("\nDISCLAIMER: Research / educational tool only. NOT for clinical diagnosis.\n")


if __name__ == "__main__":
    main()
