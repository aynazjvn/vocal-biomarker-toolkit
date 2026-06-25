"""Fine-tuning loop for Wav2Vec2Classifier on vocal biomarker tasks.

Training strategy
-----------------
* Freeze CNN feature extractor + bottom N encoder layers (default: 8/12).
* Fine-tune top encoder layers + classification head with a small LR (1e-5).
* AdamW with cosine annealing and linear warm-up (10 % of total steps).
* BCEWithLogitsLoss with class weighting for imbalanced datasets.
* Early stopping on validation AUROC (patience configurable per fold).

Leave-One-Subject-Out CV
------------------------
``loocv_evaluate`` trains one model per subject fold on all *other* subjects,
evaluates on the held-out subject, and aggregates predictions for a single
global AUROC.  This is the standard protocol for small medical datasets.

For datasets with many subjects (e.g. Coswara, >100 subjects) use
``kfold_evaluate`` with a smaller k to keep compute tractable.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

try:
    import librosa
    _LIBROSA_OK = True
except ImportError:
    _LIBROSA_OK = False

try:
    from sklearn.metrics import roc_auc_score
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

from src.data.base_loader import AudioSample
from src.evaluation.metrics import evaluate
from src.models.wav2vec2_classifier import Wav2Vec2Classifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AudioDataset(Dataset):
    """Load raw waveforms and normalise them with the Wav2Vec2FeatureExtractor.

    All samples are padded / truncated to exactly ``max_duration_s`` seconds
    so that the default collate function produces uniform-length batches.

    When ``preload=True`` (recommended for LOOCV), all waveforms are processed
    and cached to RAM once at construction time.  The full Italian Parkinson
    corpus (831 × 10 s clips) fits in ~530 MB and caches in ~5 s; thereafter
    every ``__getitem__`` call is a pure RAM read (~0.001 s vs ~5 s from disk).

    Args:
        samples: List of :class:`~src.data.base_loader.AudioSample`.
        feature_extractor: Wav2Vec2FeatureExtractor instance for normalisation.
        max_duration_s: Clip length in seconds (default 10).
        sample_rate: Target sample rate (default 16 000 Hz).
        preload: Pre-load and cache all waveforms to RAM (default True).
    """

    def __init__(
        self,
        samples: list[AudioSample],
        feature_extractor,
        max_duration_s: float = 10.0,
        sample_rate: int = 16_000,
        preload: bool = True,
    ) -> None:
        if not _LIBROSA_OK:
            raise ImportError("librosa is required: pip install librosa")
        self.samples = samples
        self.fe = feature_extractor
        self.max_samples = int(max_duration_s * sample_rate)
        self.sr = sample_rate
        self._cache: Optional[dict] = None

        if preload:
            self._cache = {}
            logger.info("Pre-loading %d waveforms to RAM …", len(samples))
            for s in samples:
                self._cache[str(s.path)] = self._load_one(s.path)
            logger.info("Waveform cache ready (%d samples)", len(self._cache))

    def _load_one(self, path) -> "torch.Tensor":
        import torch
        waveform, _ = librosa.load(str(path), sr=self.sr, mono=True)
        if len(waveform) >= self.max_samples:
            waveform = waveform[: self.max_samples]
        else:
            waveform = np.pad(waveform, (0, self.max_samples - len(waveform)))
        processed = self.fe(
            waveform.astype(np.float32),
            sampling_rate=self.sr,
            return_tensors="pt",
            padding=False,
        )
        return processed.input_values.squeeze(0)  # (max_samples,)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        if self._cache is not None:
            input_values = self._cache[str(sample.path)]
        else:
            input_values = self._load_one(sample.path)

        return {
            "input_values": input_values,
            "label": torch.tensor(float(sample.label), dtype=torch.float32),
            "subject_id": sample.subject_id,
        }


# ---------------------------------------------------------------------------
# Trainer config
# ---------------------------------------------------------------------------

@dataclass
class TransformerTrainerConfig:
    model_name: str = "facebook/wav2vec2-base"
    freeze_feature_extractor: bool = True
    freeze_encoder_layers: int = 8

    # Audio preprocessing
    max_duration_s: float = 10.0
    sample_rate: int = 16_000

    # Optimiser
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0

    # Training schedule
    batch_size: int = 8
    num_epochs: int = 10            # for final model on all data
    loocv_epochs: int = 5           # per fold (fewer for speed)
    warmup_ratio: float = 0.10      # fraction of total steps for LR warm-up
    early_stopping_patience: int = 3

    # Checkpoint
    output_dir: str = "results/condition/transformer"

    # Class imbalance
    pos_weight: Optional[float] = None  # None → auto-compute from training set

    # Misc
    seed: int = 42
    num_workers: int = 0            # 0 = main process; avoids fork issues on some systems


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TransformerTrainer:
    """Fine-tune Wav2Vec2Classifier and optionally run Leave-One-Subject-Out CV.

    Args:
        cfg: :class:`TransformerTrainerConfig` with all hyperparameters.
    """

    def __init__(self, cfg: TransformerTrainerConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        # Suppress transformers 5.x "LOAD REPORT" chatter for expected
        # pretraining-only keys (quantizer, project_q, etc.) that are absent
        # from Wav2Vec2Model but present in the pretrained checkpoint.
        import logging as _logging
        _logging.getLogger("transformers.modeling_utils").setLevel(_logging.ERROR)

        try:
            from transformers import AutoFeatureExtractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model_name)
        except Exception as exc:
            raise RuntimeError(f"Cannot load AutoFeatureExtractor for {cfg.model_name}: {exc}") from exc

        # Global waveform cache: built once in run(), reused across all folds.
        # Key = str(sample.path) → pre-normalised input_values tensor.
        self._waveform_cache: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        samples: list[AudioSample],
        run_loocv: bool = True,
        run_kfold: Optional[int] = None,
    ) -> dict:
        """Train the final model and (optionally) evaluate with LOOCV/k-fold.

        Args:
            samples: All audio samples for this condition.
            run_loocv: Run Leave-One-Subject-Out CV to compute AUROC.
            run_kfold: If > 0, run k-fold CV instead of LOOCV (faster for
                       large datasets). Overrides ``run_loocv``.

        Returns:
            Dict with keys: ``auroc``, ``n_subjects``, ``n_samples``,
            ``n_positive``, ``n_negative``, ``elapsed_s``.
        """
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        t0 = time.time()
        result: dict = {
            "n_samples": len(samples),
            "n_positive": sum(s.label == 1 for s in samples),
            "n_negative": sum(s.label == 0 for s in samples),
            "n_subjects": len({s.subject_id for s in samples}),
            "model_name": self.cfg.model_name,
            "freeze_encoder_layers": self.cfg.freeze_encoder_layers,
        }

        # ── Pre-load all waveforms to RAM once ──────────────────────────
        # Eliminates librosa disk I/O per-batch (5s one-time vs 5s/batch).
        logger.info("Pre-caching waveforms …")
        self._build_cache(samples)

        # ── Cross-validation for AUROC estimate ─────────────────────────
        if run_kfold and run_kfold > 0:
            logger.info("Running %d-fold subject-grouped CV …", run_kfold)
            auroc, fold_aurocs = self.kfold_evaluate(samples, k=run_kfold)
            result["cv_type"] = f"{run_kfold}-fold"
            result["auroc"] = round(auroc, 4)
            result["fold_aurocs"] = [round(a, 4) for a in fold_aurocs]
        elif run_loocv:
            n_subj = result["n_subjects"]
            logger.info("Running Leave-One-Subject-Out CV (%d folds) …", n_subj)
            auroc = self.loocv_evaluate(samples)
            result["cv_type"] = "loocv"
            result["auroc"] = round(auroc, 4)
        else:
            result["cv_type"] = "none"
            result["auroc"] = None

        # ── Train final model on all data ────────────────────────────────
        logger.info("Training final model on all %d samples …", len(samples))
        model = self._make_model()
        pos_weight = self._pos_weight(samples)
        dataset = AudioDataset(
            samples, self.feature_extractor,
            max_duration_s=self.cfg.max_duration_s,
            sample_rate=self.cfg.sample_rate,
            preload=False,   # cache already in self._waveform_cache
        )
        dataset._cache = self._waveform_cache
        loader = DataLoader(
            dataset, batch_size=self.cfg.batch_size,
            shuffle=True, num_workers=0,   # num_workers=0: cache is in main-process RAM
            pin_memory=self.device.type == "cuda",
        )

        self._train_model(
            model, loader, val_loader=None,
            n_epochs=self.cfg.num_epochs, pos_weight=pos_weight,
        )

        # ── Save checkpoint ──────────────────────────────────────────────
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model.cpu()
        model.save(out)

        size_mb = (out / "model.pt").stat().st_size / 1e6
        logger.info("Saved model → %s  (%.0f MB)", out, size_mb)

        if size_mb > 100:
            logger.warning(
                "model.pt is %.0f MB — too large for standard git. "
                "Use git-lfs or download instructions instead of committing.",
                size_mb,
            )

        result["elapsed_s"] = round(time.time() - t0, 1)
        return result

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def loocv_evaluate(self, samples: list[AudioSample]) -> float:
        """Leave-One-Subject-Out CV → global AUROC across all held-out subjects."""
        subjects = list(dict.fromkeys(s.subject_id for s in samples))
        all_true: list[float] = []
        all_prob: list[float] = []

        for fold_i, held_out in enumerate(subjects):
            train_s = [s for s in samples if s.subject_id != held_out]
            test_s  = [s for s in samples if s.subject_id == held_out]

            if not train_s or not test_s:
                continue

            logger.info(
                "LOOCV fold %d/%d  hold-out=%s  train=%d  test=%d",
                fold_i + 1, len(subjects), held_out, len(train_s), len(test_s),
            )

            probs = self._fit_predict(train_s, test_s, n_epochs=self.cfg.loocv_epochs)
            all_true.extend(s.label for s in test_s)
            all_prob.extend(probs)

        if not all_true:
            return float("nan")

        auroc = float(roc_auc_score(all_true, all_prob))
        logger.info("LOOCV AUROC = %.4f", auroc)
        # Free the cached initial state — no longer needed after LOOCV
        self._loocv_init_state = None
        gc.collect()
        return auroc

    def kfold_evaluate(
        self,
        samples: list[AudioSample],
        k: int = 10,
    ) -> tuple[float, list[float]]:
        """Grouped k-fold CV (subjects stay whole in one fold)."""
        subjects = list(dict.fromkeys(s.subject_id for s in samples))
        np.random.seed(self.cfg.seed)
        np.random.shuffle(subjects)

        folds = np.array_split(subjects, k)
        all_true: list[float] = []
        all_prob: list[float] = []
        fold_aurocs: list[float] = []

        for fold_i, test_subjects in enumerate(folds):
            test_set = set(test_subjects)
            train_s = [s for s in samples if s.subject_id not in test_set]
            test_s  = [s for s in samples if s.subject_id in test_set]

            if not train_s or not test_s:
                continue

            logger.info(
                "Fold %d/%d  train=%d  test=%d",
                fold_i + 1, k, len(train_s), len(test_s),
            )

            probs = self._fit_predict(train_s, test_s, n_epochs=self.cfg.loocv_epochs)
            true = [s.label for s in test_s]
            all_true.extend(true)
            all_prob.extend(probs)

            if len(set(true)) > 1:
                fold_aurocs.append(float(roc_auc_score(true, probs)))

        auroc = float(roc_auc_score(all_true, all_prob)) if all_true else float("nan")
        logger.info("K-fold AUROC = %.4f  (mean fold = %.4f)", auroc, np.mean(fold_aurocs))
        return auroc, fold_aurocs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cache(self, samples: list[AudioSample]) -> None:
        """Pre-load and normalise all waveforms once; store in self._waveform_cache."""
        if self._waveform_cache is not None:
            return  # already built
        tmp_ds = AudioDataset(
            samples, self.feature_extractor,
            max_duration_s=self.cfg.max_duration_s,
            sample_rate=self.cfg.sample_rate,
            preload=True,
        )
        self._waveform_cache = tmp_ds._cache

    def _make_cached_dataset(self, samples: list[AudioSample]) -> AudioDataset:
        """Return a dataset that reads from the shared RAM cache."""
        ds = AudioDataset(
            samples, self.feature_extractor,
            max_duration_s=self.cfg.max_duration_s,
            sample_rate=self.cfg.sample_rate,
            preload=False,
        )
        ds._cache = self._waveform_cache
        return ds

    def _fit_predict(
        self,
        train_samples: list[AudioSample],
        test_samples: list[AudioSample],
        n_epochs: int,
    ) -> list[float]:
        """Train a fresh model on train_samples, return probs on test_samples.

        On the first call, creates the model and caches its initial weights in
        ``self._loocv_init_state`` (CPU tensors).  Subsequent folds copy those
        weights back instead of calling ``Wav2Vec2Model.from_pretrained`` again,
        eliminating ~3 HuggingFace HTTP round-trips and a model-load progress bar
        per fold.
        """
        if not hasattr(self, "_loocv_init_state") or self._loocv_init_state is None:
            model = self._make_model()
            # Clone initial state to CPU once
            self._loocv_init_state = {
                k: v.detach().clone().cpu() for k, v in model.state_dict().items()
            }
        else:
            # Re-use the existing (already-CUDA) model shell if it survived GC,
            # otherwise re-create the architecture from scratch and inject weights.
            model = self._make_model_skeleton()
            model.load_state_dict(
                {k: v.to(self.device) for k, v in self._loocv_init_state.items()}
            )

        pos_weight = self._pos_weight(train_samples)

        train_loader = DataLoader(
            self._make_cached_dataset(train_samples),
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,   # data is in main-process RAM
            pin_memory=self.device.type == "cuda",
        )
        test_loader = DataLoader(
            self._make_cached_dataset(test_samples),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self._train_model(model, train_loader, val_loader=None,
                          n_epochs=n_epochs, pos_weight=pos_weight)
        probs = self._predict(model, test_loader)

        # Free GPU memory between folds
        model.cpu()
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return probs

    def _make_model(self) -> Wav2Vec2Classifier:
        """Create a model by downloading / reading from HuggingFace cache."""
        model = Wav2Vec2Classifier(
            model_name=self.cfg.model_name,
            freeze_feature_extractor=self.cfg.freeze_feature_extractor,
            freeze_encoder_layers=self.cfg.freeze_encoder_layers,
        )
        model = model.to(self.device)
        logger.info(
            "Model: %d trainable / %d total params",
            model.n_trainable_params(), model.n_total_params(),
        )
        return model

    def _make_model_skeleton(self) -> Wav2Vec2Classifier:
        """Create an empty model shell without loading pretrained weights.

        Used for LOOCV folds 2-N: architecture is built from config only,
        then weights are immediately overwritten from self._loocv_init_state,
        avoiding repeated HuggingFace downloads.
        """
        import transformers as _tf
        old_level = _tf.logging.get_verbosity()
        _tf.logging.set_verbosity_error()
        try:
            from transformers import AutoConfig
            from src.models.wav2vec2_classifier import Wav2Vec2Classifier as _C, _make_encoder_from_config
            config = AutoConfig.from_pretrained(self.cfg.model_name)
            skeleton = _C.__new__(_C)
            import torch.nn as nn
            nn.Module.__init__(skeleton)
            skeleton.model_name = self.cfg.model_name
            skeleton.freeze_feature_extractor = self.cfg.freeze_feature_extractor
            skeleton.freeze_encoder_layers = self.cfg.freeze_encoder_layers
            skeleton.wav2vec2 = _make_encoder_from_config(config)
            hidden_size = skeleton.wav2vec2.config.hidden_size
            skeleton.dropout = nn.Dropout(0.1)
            skeleton.classifier = nn.Linear(hidden_size, 1)
        finally:
            _tf.logging.set_verbosity(old_level)

        return skeleton.to(self.device)

    def _train_model(
        self,
        model: Wav2Vec2Classifier,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        n_epochs: int,
        pos_weight: Optional[float],
    ) -> None:
        """AdamW + OneCycleLR + early stopping on val loss."""
        weight = (
            torch.tensor([pos_weight], device=self.device)
            if pos_weight is not None
            else None
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        total_steps = len(train_loader) * n_epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.cfg.learning_rate,
            total_steps=total_steps,
            pct_start=self.cfg.warmup_ratio,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, n_epochs + 1):
            train_loss = self._run_epoch(
                model, train_loader, optimizer, scheduler, criterion
            )

            if val_loader is not None:
                val_loss = self._eval_loss(model, val_loader, criterion)
                logger.info(
                    "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
                    epoch, n_epochs, train_loss, val_loss,
                )
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.cfg.early_stopping_patience:
                        logger.info("Early stopping at epoch %d", epoch)
                        break
            else:
                logger.info(
                    "Epoch %d/%d  train_loss=%.4f",
                    epoch, n_epochs, train_loss,
                )

    def _run_epoch(
        self,
        model: Wav2Vec2Classifier,
        loader: DataLoader,
        optimizer,
        scheduler,
        criterion: nn.Module,
    ) -> float:
        model.train()
        total_loss = 0.0

        for batch in loader:
            input_values = batch["input_values"].to(self.device)
            labels = batch["label"].to(self.device)

            optimizer.zero_grad()
            logits = model(input_values)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _eval_loss(
        self,
        model: Wav2Vec2Classifier,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                input_values = batch["input_values"].to(self.device)
                labels = batch["label"].to(self.device)
                logits = model(input_values)
                total_loss += criterion(logits, labels).item()
        return total_loss / len(loader)

    def _predict(
        self,
        model: Wav2Vec2Classifier,
        loader: DataLoader,
    ) -> list[float]:
        model.eval()
        probs: list[float] = []
        with torch.no_grad():
            for batch in loader:
                input_values = batch["input_values"].to(self.device)
                logits = model(input_values)
                probs.extend(torch.sigmoid(logits).cpu().tolist())
        return probs

    def _pos_weight(self, samples: list[AudioSample]) -> Optional[float]:
        """Class imbalance weight: n_negative / n_positive."""
        if self.cfg.pos_weight is not None:
            return self.cfg.pos_weight
        n_pos = sum(s.label == 1 for s in samples)
        n_neg = sum(s.label == 0 for s in samples)
        if n_pos == 0:
            return None
        return n_neg / n_pos
